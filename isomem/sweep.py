"""Sweep orchestration: idempotent, resumable, append-only (SPEC §3, §5).

Designed around the assumption that the sweep *will* die partway through — at run 41 of
142, on a laptop lid closing, with an OOM, or with a NaN. None of those may cost more than
the runs already finished.

* **Run IDs are deterministic.** `run_id(budget, bits, seed, rule)` is a pure function, so
  the same cell always maps to the same row and the same checkpoint file.
* **Results are append-only** with a `status` column. Resume reads `results.csv`, skips
  every `status == "complete"` row, and continues. Re-running the whole sweep after it
  finishes is a no-op.
* **A failed run fails alone.** NaN, collapse, or an unexpected exception marks that row
  `nan` / `collapsed` / `error` and the sweep moves to the next cell. Only a
  KeyboardInterrupt stops everything.
* **OOM backs off without changing the experiment.** The micro-batch halves and gradient
  accumulation doubles, so the effective batch is invariant (SPEC §3, OOM row).

Running across day boundaries
-----------------------------
The machine suspends for the night on its own schedule and wakes in the morning, so the
sweep is built to live inside that window rather than fight it. ``--until HH:MM`` (read
from quiet-hours' own config by default, in quiet-hours' own timezone) sets a deadline:

* a run still in flight checkpoints and stops cleanly, and resumes next morning from the
  step it reached -- not from the last eval boundary, and not from the beginning;
* a new run is not started unless there is a sensible amount of time left for it;
* SIGTERM does the same thing, so being stopped by systemd is equally safe.

Nothing is lost to a suspend, and nothing is repeated on wake.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from isomem import data as D
from isomem.accounting import ModelConfig, accounting_row, count_params
from isomem.corpus import ACTIVE as CORPUS
from isomem.eval import load_baseline
from isomem.fixed import FIXED, GRID, assert_fixed_invariants, fixed_config_hash, git_sha
from isomem.quant import TernaryCollapse
from isomem.search import InfeasibleBudget, ShapeRule, search_cell
from isomem.train import (
    GracefulStop,
    NanLoss,
    RunResult,
    make_batch_schedule,
    resolve_device,
    train_one,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# results/results.csv and runs/ for TinyStories, as they always were. A second corpus
# writes to its own file and its own checkpoint directory, and its run ids carry a suffix,
# so it can never resume from, or be mistaken for, a TinyStories run.
RESULTS_CSV = CORPUS.results_csv
CKPT_DIR = CORPUS.ckpt_dir

FIELDS = [
    "run_id", "status", "sweep", "shape_rule",
    "budget_name", "budget_bits_target", "budget_bits_actual", "bits_per_weight",
    "n_layer", "n_embd", "n_head", "mlp_ratio", "aspect", "vocab",
    "params_counted", "norm_params",
    "embedding_bits", "embedding_frac", "embedding_bound", "kv_bits_full_ctx",
    "budget_rel_error",
    "seed", "tokens_seen", "steps", "best_step", "stopped_early", "undertrained",
    "tail_drift_below_noise", "val_drift", "val_noise", "target_residual", "epochs",
    "train_loss", "val_loss_best", "val_ppl_best", "test_loss", "test_ppl",
    "bigram_baseline_ppl", "beats_baseline",
    "micro_batch", "grad_accum", "effective_batch_tokens",
    "wall_clock_s", "peak_mem_mb", "device", "lr",
    "torch_version", "cuda_version", "git_sha", "fixed_config_hash",
    "error",
]


def run_id(budget: int, bits: float, seed: int, shape_rule: str = "TIGHTEST",
           corpus_suffix: str = "") -> str:
    """Deterministic. Same inputs -> same ID, always (SPEC §4 interface)."""
    suffix = "" if shape_rule == "TIGHTEST" else f"-{shape_rule.lower()}"
    return f"B{budget}-b{bits:g}-s{seed}{suffix}{corpus_suffix}"


@dataclass(frozen=True)
class Cell:
    budget_name: str
    budget_bits: int
    bits: float
    seed: int
    shape_rule: ShapeRule
    sweep: str  # "primary" or "robustness"

    @property
    def run_id(self) -> str:
        return run_id(self.budget_bits, self.bits, self.seed, self.shape_rule,
                      CORPUS.run_suffix)


def planned_cells() -> list[Cell]:
    """Every run in the declared plan, primary first (PREREGISTRATION.md §5).

    Under a second corpus the plan is that corpus' arm (Amendment A6): the same bit-widths
    and shape rule as the primary sweep, at the budgets and seeds its config declares.
    """
    budgets = {b["name"]: b["bits"] for b in GRID["budgets_bits"]}
    cells: list[Cell] = []
    if not CORPUS.is_default:
        # Seed-major within each budget: every bit-width at seed 0, then seed 1, and so on.
        # Runs are independent and deterministic, so order changes no result -- but it means
        # a complete k-seed curve exists after k passes, instead of the last bit-width
        # having no seeds at all until the very end.
        arm = CORPUS.arm
        for name in arm["budgets"]:
            for seed in arm["seeds"]:
                for bits_val in GRID["bits_per_weight"]:
                    cells.append(Cell(name, budgets[name], float(bits_val), seed,
                                      arm["shape_rule"], arm["sweep"]))
        return cells
    for name, bits_val in [(n, b) for n in budgets for b in GRID["bits_per_weight"]]:
        for seed in GRID["seeds"]:
            cells.append(Cell(name, budgets[name], float(bits_val), seed, "TIGHTEST", "primary"))

    rob = GRID.get("robustness", {})
    if rob.get("enabled"):
        for name in rob["budgets"]:
            for bits_val in GRID["bits_per_weight"]:
                for seed in rob["seeds"]:
                    cells.append(
                        Cell(name, budgets[name], float(bits_val), seed,
                             rob["shape_rule"], "robustness")
                    )
    return cells


# --------------------------------------------------------------------------------------
# Append-only results
# --------------------------------------------------------------------------------------


def read_results() -> dict[str, dict]:
    if not RESULTS_CSV.exists():
        return {}
    with open(RESULTS_CSV, newline="") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh)}


# Terminal statuses are never retried. `infeasible` is terminal because the cell cannot
# exist at all (the embedding table alone exceeds the budget), so retrying it every tick
# only appends duplicate rows. Failures like `nan`, `error` and `oom` are NOT terminal:
# they may be transient, and a rerun is the right response.
TERMINAL_STATUSES = frozenset({"complete", "infeasible"})


def completed_run_ids() -> set[str]:
    """Run ids that must not be run again. Failures are retried on the next invocation."""
    return {
        rid for rid, row in read_results().items()
        if row.get("status") in TERMINAL_STATUSES
    }


def append_row(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def _provenance() -> dict:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "git_sha": git_sha(),
        "fixed_config_hash": fixed_config_hash() if CORPUS.is_default
        else f"{fixed_config_hash()}+{CORPUS.config_hash()}",
    }


def build_row(cell: Cell, cfg: ModelConfig, result: RunResult, schedule, status: str,
              error: str = "") -> dict:
    baseline = load_baseline()
    row = {
        **accounting_row(cfg, cell.bits, cell.budget_bits),
        **result.as_row(),
        **_provenance(),
        "status": status,
        "sweep": cell.sweep,
        "shape_rule": cell.shape_rule,
        "budget_name": cell.budget_name,
        "aspect": cfg.n_embd / cfg.n_layer,
        "target_residual": residual_at(count_params(cfg), result.tokens_seen)
        if result.tokens_seen else "",
        "epochs": result.tokens_seen / CORPUS.unique_train_tokens
        if result.tokens_seen else "",
        "micro_batch": schedule.micro_batch if schedule else "",
        "grad_accum": schedule.grad_accum if schedule else "",
        "effective_batch_tokens": schedule.effective_batch_tokens if schedule else "",
        "error": error,
    }
    if baseline:
        row["bigram_baseline_ppl"] = baseline.val_ppl
        row["beats_baseline"] = (
            result.val_loss_best < baseline.val_loss
            if result.val_loss_best not in (float("inf"), None) else ""
        )
    # The test split is NEVER touched during the sweep (SPEC §2.6). These stay empty until
    # write-up, when they are filled in by a separate, one-time script.
    row["test_loss"] = ""
    row["test_ppl"] = ""
    return row


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


class DeviceMismatch(RuntimeError):
    """This process resolved a different device type from the rows it would be compared with."""


def reference_device() -> str | None:
    """The device type the comparison's existing rows were produced on.

    This corpus' own rows if it has any, else the TinyStories primary sweep's -- a second
    corpus is compared against those, so it must match them too.
    """
    from isomem.corpus import DEFAULT, load

    for path in (RESULTS_CSV, load(DEFAULT).results_csv):
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "complete" and row.get("device"):
                    return row["device"]
    return None


def assert_device_consistent(device) -> None:
    """Refuse to append rows from a different device type than the rest (SPEC §3).

    The device is resolved at runtime (cuda -> mps -> cpu), which is right for development
    and dangerous for a sweep: if CUDA fails to initialise -- a driver update with a stale
    kernel module is enough, and the training host was in exactly that state on 2026-09-26 -- the sweep
    would silently continue on CPU, orders of magnitude slower and on different numerics
    from every row it is compared with. Better to stop and be noticed.
    """
    ref = reference_device()
    if ref and torch.device(device).type != ref:
        raise DeviceMismatch(
            f"resolved {device} but the rows this sweep is compared with were produced on "
            f"{ref}. Refusing to mix devices; check the GPU driver (nvidia-smi) and rerun."
        )


class CorpusNotReady(RuntimeError):
    """A second corpus is selected but not prepared to the standard the primary one was."""


def assert_corpus_ready() -> None:
    """Refuse to train on a second corpus that is not frozen and hashed (Amendment A6).

    Token budgets are computed from TinyStories' size for both corpora, so every cell sees
    exactly its TinyStories token count. That makes the epoch count comparable only if the
    second corpus holds at least as many unique training tokens; with fewer, its largest
    cells would repeat data more than the 4-epoch cap allows. So it is asserted, not hoped.
    """
    if CORPUS.is_default:
        return
    missing = [k for k, v in {"tokenizer.hash": CORPUS.tokenizer_hash,
                              "source.n_docs": CORPUS.n_docs,
                              "data.unique_train_tokens": CORPUS.unique_train_tokens,
                              **{f"data.hashes.{k}": v for k, v in CORPUS.split_hashes.items()},
                              }.items() if v is None]
    if missing:
        raise CorpusNotReady(f"{CORPUS.config_path.name} not prepared: {missing}. "
                             "Run scripts/size_corpus.py then scripts/prepare_data.py.")
    need = GRID["token_budget"]["unique_train_tokens"]
    if CORPUS.unique_train_tokens < need:
        raise CorpusNotReady(
            f"{CORPUS.name} has {CORPUS.unique_train_tokens:,} train tokens, fewer than "
            f"TinyStories' {need:,}: its largest cells would exceed the epoch cap."
        )


def token_budget(params: int) -> int:
    """Tokens for a cell of `params` parameters, so every cell lands at the same residual.

    See config/grid.yaml for the fitted model and why a uniform residual is the right
    target. Capped at `max_epochs` of the unique training data; at the chosen target no
    cell reaches that cap, which is the point.
    """
    tb = GRID["token_budget"]
    if tb.get("rule") != "uniform_residual":
        raise ValueError(f"unknown token budget rule {tb.get('rule')!r}")
    g = tb["gain_model"]
    a, b, decay = g["a"], g["b"], g["decay_per_doubling"]
    # Pin the constant to the measured anchor rather than carrying a fitted intercept.
    C = g["anchor_gain"] / (g["anchor_params"] ** a * g["anchor_tokens"] ** b)
    target_gain = tb["target_residual"] * (1 - decay) / decay
    tokens = (target_gain / (C * params**a)) ** (1 / b)
    return int(min(tokens, tb["max_epochs"] * tb["unique_train_tokens"]))


def residual_at(params: int, tokens: int) -> float:
    """The convergence residual a cell will be left with. Logged per run."""
    g = GRID["token_budget"]["gain_model"]
    a, b, decay = g["a"], g["b"], g["decay_per_doubling"]
    C = g["anchor_gain"] / (g["anchor_params"] ** a * g["anchor_tokens"] ** b)
    return C * params**a * tokens**b * decay / (1 - decay)


QUIET_HOURS_CONF = Path("/etc/quiet-hours/quiet-hours.conf")
QUIET_HOURS_DROPINS = Path("/etc/quiet-hours/conf.d")


def _quiet_hours_values() -> dict[str, str]:
    """Every KEY=VALUE quiet-hours reads, base config then drop-ins, last one wins."""
    if not QUIET_HOURS_CONF.exists():
        return {}
    values: dict[str, str] = {}
    files = [QUIET_HOURS_CONF] + sorted(QUIET_HOURS_DROPINS.glob("*.conf")) if \
        QUIET_HOURS_DROPINS.is_dir() else [QUIET_HOURS_CONF]
    for path in files:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip().strip('"\'')
    return values


def quiet_hours_settings() -> tuple[str, str] | None:
    """(SLEEP_AT, TIMEZONE) from quiet-hours, drop-ins last. None if not installed.

    Read rather than hardcoded: the machine's own schedule is the source of truth, so
    changing it in one place changes it for the sweep too.
    """
    values = _quiet_hours_values()
    if "SLEEP_AT" not in values:
        return None
    # quiet-hours' own tick treats these as "never sleep"; the sweep must agree, or it
    # stops for a suspend that is not coming and idles the GPU through a night it has.
    if values.get("ENABLED", "yes").lower() in ("no", "off", "0", "false"):
        return None
    return values["SLEEP_AT"], values.get("TIMEZONE", "UTC")


def _minutes(hhmm: str) -> int:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return hh * 60 + mm


def next_deadline(until: str | None, tz_name: str | None, margin_min: int = 10,
                  now=None) -> float | None:
    """Wall-clock epoch by which the sweep must have stopped, or None for no deadline.

    ``margin_min`` is subtracted so the last checkpoint is written and the process has
    exited before the machine actually starts suspending.

    Inside the quiet window itself -- between SLEEP_AT and WAKE_AT -- the deadline is
    *now*, so nothing new starts. Without that, a tick firing at 23:00:01 saw tonight's
    23:00 as already past, took tomorrow's, and started training into a suspend: it
    happened on 2026-09-24 at 20:00:01 UTC, and the run survived only because this machine
    happens to keep its CUDA context across suspend-to-RAM.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    wake = None
    if until is None:
        found = quiet_hours_settings()
        if not found:
            return None
        until, tz_name = found
        wake = _quiet_hours_values().get("WAKE_AT")
    tz = ZoneInfo(tz_name or "UTC")
    now = now or datetime.now(tz)
    if wake:
        n, s, w = now.hour * 60 + now.minute, _minutes(until), _minutes(wake)
        inside = (s <= n < w) if s < w else (n >= s or n < w)
        if inside:
            return now.timestamp() - 1
    hh, mm = (int(x) for x in until.split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - timedelta(minutes=margin_min)).timestamp()


def run_cell(cell: Cell, device, train_tokens, val_tokens, micro_batch: int | None = None,
             deadline: float | None = None):
    """Train one cell, backing off the micro-batch on OOM without changing the experiment."""
    cfg, _ = search_cell(cell.budget_bits, cell.bits, shape_rule=cell.shape_rule), None
    cfg = cfg.config
    t_cfg = FIXED["training"]
    micro = micro_batch or t_cfg["micro_batch"]

    while True:
        schedule = make_batch_schedule(t_cfg["effective_batch_tokens"], micro, cfg.block_size)
        try:
            result = train_one(
                cfg, cell.bits, seed=cell.seed, budget_bits=cell.budget_bits,
                max_tokens=token_budget(count_params(cfg)), run_id=cell.run_id,
                ckpt_dir=CKPT_DIR, device=device,
                train_tokens=train_tokens, val_tokens=val_tokens,
                micro_batch=micro, deadline=deadline,
            )
            return cfg, result, schedule, "complete", ""
        except GracefulStop as exc:
            # Checkpointed and stopped for the night. NOT recorded as a result row: the run
            # is unfinished, and the next invocation resumes it from its checkpoint.
            raise
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
            if micro <= 1:
                return cfg, RunResult(cell.run_id, seed=cell.seed), schedule, "oom", str(exc)
            micro //= 2  # effective batch is invariant: accumulation doubles to match
            print(f"    OOM -> micro_batch {micro}, grad_accum doubles "
                  f"(effective batch unchanged)", flush=True)
            torch.cuda.empty_cache()
        except NanLoss as exc:
            return cfg, RunResult(cell.run_id, seed=cell.seed), schedule, "nan", str(exc)
        except TernaryCollapse as exc:
            return cfg, RunResult(cell.run_id, seed=cell.seed), schedule, "collapsed", str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="all", choices=["all", "primary", "robustness"])
    ap.add_argument("--budget", default=None, help="restrict to one budget name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry", action="store_true", help="list the plan and exit")
    ap.add_argument("--until", default=None,
                    help="stop by HH:MM (default: read from quiet-hours config)")
    ap.add_argument("--timezone", default=None)
    ap.add_argument("--no-deadline", action="store_true",
                    help="ignore quiet hours and run until finished")
    ap.add_argument("--min-run-minutes", type=float, default=15.0,
                    help="do not start a new run with less than this much time left")
    args = ap.parse_args()

    assert_fixed_invariants()
    assert_corpus_ready()
    cells = planned_cells()
    if args.sweep != "all":
        cells = [c for c in cells if c.sweep == args.sweep]
    if args.budget:
        cells = [c for c in cells if c.budget_name == args.budget]

    done = completed_run_ids()
    todo = [c for c in cells if c.run_id not in done]

    print(f"{len(cells)} planned, {len(cells) - len(todo)} already complete, "
          f"{len(todo)} to run")
    if args.dry:
        for c in todo:
            print(f"  {c.run_id:>28}  {c.budget_name} b={c.bits} seed={c.seed} "
                  f"{c.shape_rule} [{c.sweep}]")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    device = resolve_device(args.device)
    assert_device_consistent(device)
    deadline = None if args.no_deadline else next_deadline(args.until, args.timezone)
    if deadline:
        left = (deadline - time.time()) / 3600
        print(f"deadline in {left:.2f} h (quiet hours); in-flight runs checkpoint and "
              f"resume tomorrow")
    print(f"device {device}   git {git_sha()}\n", flush=True)
    train_tokens = D.load_tokens("train")
    val_tokens = D.load_tokens("val")

    n_ok = n_fail = n_skipped = n_deferred = 0
    for i, cell in enumerate(todo, 1):
        # Re-read the schedule before every run, not once per process. One sweep process
        # can outlive a change to quiet hours -- started on a night when sleep is switched
        # off, it would otherwise carry "no deadline" into the next night and be suspended
        # mid-run, which is exactly what the checkpoint-and-exit design exists to prevent.
        if not args.no_deadline:
            deadline = next_deadline(args.until, args.timezone)

        # Only start a new run if there is a sensible amount of time left. A run already
        # part-trained (its checkpoint exists) is always resumed, since finishing it is
        # what frees the slot.
        resuming = (CKPT_DIR / f"{cell.run_id}.pt").exists()
        if deadline and not resuming:
            minutes_left = (deadline - time.time()) / 60
            if minutes_left < args.min_run_minutes:
                n_deferred = len(todo) - i + 1
                print(f"\n{minutes_left:.0f} min left before quiet hours; "
                      f"deferring {n_deferred} run(s) to the next wake", flush=True)
                break

        print(f"[{i}/{len(todo)}] {cell.run_id}  {cell.budget_name} b={cell.bits} "
              f"seed={cell.seed} [{cell.sweep}]{'  (resuming)' if resuming else ''}",
              flush=True)
        try:
            cfg, result, schedule, status, error = run_cell(
                cell, device, train_tokens, val_tokens, deadline=deadline
            )
        except GracefulStop as exc:
            print(f"    {exc}")
            why = "quiet hours" if "deadline" in str(exc) else "a stop signal"
            print(f"\nstopped by {why}; {len(todo) - i + 1} run(s) remain, "
                  f"the current one resumes from its checkpoint")
            n_deferred = len(todo) - i + 1
            break
        except InfeasibleBudget as exc:
            # The one structurally impossible cell. Recorded as a result, never skipped
            # silently (PREREGISTRATION.md §5.1).
            append_row({
                **_provenance(), "run_id": cell.run_id, "status": "infeasible",
                "sweep": cell.sweep, "shape_rule": cell.shape_rule,
                "budget_name": cell.budget_name, "budget_bits_target": cell.budget_bits,
                "bits_per_weight": cell.bits, "seed": cell.seed, "error": str(exc),
            })
            print("    INFEASIBLE (recorded as a result)\n", flush=True)
            n_skipped += 1
            continue
        except KeyboardInterrupt:
            print("\ninterrupted; completed runs are safe in results.csv")
            return 130
        except Exception as exc:  # a failed run fails alone, never the sweep
            append_row({
                **_provenance(), "run_id": cell.run_id, "status": "error",
                "sweep": cell.sweep, "budget_name": cell.budget_name,
                "bits_per_weight": cell.bits, "seed": cell.seed,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            n_fail += 1
            continue

        append_row(build_row(cell, cfg, result, schedule, status, error))
        if status == "complete":
            n_ok += 1
            flag = " UNDERTRAINED" if result.undertrained else ""
            print(f"    val {result.val_loss_best:.4f} (ppl {result.val_ppl_best:.2f})  "
                  f"{result.tokens_seen:,} tokens  {result.wall_clock_s / 60:.1f} min"
                  f"{flag}\n", flush=True)
        else:
            n_fail += 1
            print(f"    {status.upper()}: {error.splitlines()[0] if error else ''}\n",
                  flush=True)

    print(f"done: {n_ok} complete, {n_fail} failed, {n_skipped} infeasible"
          + (f", {n_deferred} deferred to next wake" if n_deferred else ""))
    print(f"results: {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
