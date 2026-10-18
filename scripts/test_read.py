"""The single read of the sealed test split (PREREGISTRATION.md §10.3, §14; §16-§17 for (c)).

What it does, exactly as §14 fixes it:

* every complete PRIMARY (TIGHTEST) TinyStories run's final-step checkpoint is evaluated
  once on the test token stream obtained through
  ``load_test_tokens(i_understand_this_is_the_final_evaluation=True)``, with the procedure
  used for every validation evaluation -- ``isomem.train.evaluate``: 50 batches of 128
  windows of 256 tokens drawn with the fixed ``EVAL_SEED`` -- and the loss is written to
  that run's ``test_loss`` (and ``test_ppl = exp(test_loss)``) in ``results/results.csv``;
* §16 extends §14: the same read also evaluates each primary run's A7 (c) weights
  (time-averaged latents), with the same procedure, for runs whose ``results/a7/m1.csv``
  status is ``complete`` (§17); those go to the log ``results/test_read.json`` as
  ``test_loss_c`` beside the final-step result. results.csv has no column for them.

Guards (each raises before the test split is opened, never warns):

* ``--i-have-the-go-ahead`` is required; without it nothing is read at all;
* read exactly once: refuses if any primary row already carries a ``test_loss``, or if the
  marker ``results/.test_read_opened`` exists. The marker is written immediately BEFORE the
  test split is opened, so a crash after opening still blocks a second read -- removing it
  is a human decision that must be recorded in PREREGISTRATION.md;
* refuses if any planned primary run is neither ``complete`` nor ``infeasible``;
* refuses a configuration that does not replay from its row (``a7_average.reconstruct``: the
  architecture search, budget, batch schedule, step count and lr rule), a checkpoint that is
  not the final step, a torch/CUDA build or device type different from the one that trained
  the run, and a checkpoint whose validation loss, re-evaluated now, differs from the one
  recorded at its final step by more than 1e-3 (the same check A7 used). The same holds for
  the (c) weights against m1.csv's ``val_c``. Every checkpoint is verified, and its sha256
  recorded, before the test split is opened; the weights evaluated on test are the very
  tensors that were verified (held in memory, not re-read from disk).

Writes: ``results/test_read.json`` (device, git_sha, torch/CUDA, per-run test_loss,
test_loss_c, checkpoint hashes, the verification numbers), then ``results/results.csv``
with ``test_loss``/``test_ppl`` filled -- both atomically (temp file, fsync, rename).

Usage, on the compute host only, once:

    ISOMEM_DATASET= .venv/bin/python scripts/test_read.py --i-have-the-go-ahead \\
        --ckpt-dir ~/bit-budget/runs --averaged-dir ~/bit-budget-a7/results/a7/averaged \\
        --m1-csv ~/bit-budget-a7/results/a7/m1.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from isomem import data as D  # noqa: E402
from isomem import train as TR  # noqa: E402
from isomem.analysis import load_rows  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS  # noqa: E402
from isomem.provenance import CheckpointProvenanceError  # noqa: E402
from isomem.fixed import FIXED, assert_fixed_invariants, git_sha  # noqa: E402
from isomem.model import IsoMemGPT  # noqa: E402
from isomem.sweep import planned_cells  # noqa: E402


def _load_a7():
    name = "a7_average"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "a7_average.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A7 = _load_a7()
reconstruct = A7.reconstruct
check_software = A7.check_software

EVAL_BATCHES = FIXED["training"]["eval_batches"]     # 50, the standard evaluation
MISMATCH_TOL = A7.MISMATCH_TOL                        # 1e-3, as in A7 M1
RESULTS_CSV = CORPUS.results_csv
DEFAULT_CKPT_DIR = REPO_ROOT / "runs"
DEFAULT_AVERAGED_DIR = REPO_ROOT / "results" / "a7" / "averaged"
DEFAULT_M1_CSV = REPO_ROOT / "results" / "a7" / "m1.csv"
DEFAULT_LOG = REPO_ROOT / "results" / "test_read.json"
MARKER_NAME = ".test_read_opened"
ALLOWED_TERMINAL = {"complete", "infeasible"}


class TestReadRefused(RuntimeError):
    """A guard of the single test read refused; the test split was NOT opened."""


class AlreadyRead(TestReadRefused):
    """The test split has already been read once (§10.3: exactly once)."""


# --------------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------------


def planned_primary_ids() -> set[str]:
    return {c.run_id for c in planned_cells() if c.sweep == "primary"}


def assert_not_read(results_csv: Path, marker: Path) -> None:
    if marker.exists():
        raise AlreadyRead(f"{marker} exists: the test split was already opened once "
                          f"({marker.read_text().strip()}). Never delete it without an "
                          f"amendment in PREREGISTRATION.md.")
    raw = load_rows(results_csv, dedupe=False)
    written = [r["run_id"] for r in raw if r.get("test_loss") not in ("", None)]
    if written:
        raise AlreadyRead(f"{len(written)} row(s) already carry a test_loss "
                          f"(e.g. {written[:3]}): the test split is read exactly once")


def primary_targets(results_csv: Path) -> list[dict]:
    """Complete primary rows (last write wins); refuses if any planned run is unfinished."""
    rows = {r["run_id"]: r for r in load_rows(results_csv) if r["sweep"] == "primary"}
    planned = planned_primary_ids()
    missing = sorted(planned - set(rows))
    unfinished = sorted(rid for rid in planned & set(rows)
                        if rows[rid]["status"] not in ALLOWED_TERMINAL)
    extra = sorted(set(rows) - planned)
    if missing or unfinished or extra:
        raise TestReadRefused(f"primary sweep not complete: missing {missing}, "
                              f"unfinished {unfinished}, unplanned {extra}")
    return sorted((r for r in rows.values() if r["status"] == "complete"),
                  key=lambda r: r["run_id"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def m1_rows(m1_csv: Path) -> dict[str, dict]:
    if not m1_csv.exists():
        raise TestReadRefused(f"{m1_csv} missing; §16 requires the (c) weights to be read too")
    with open(m1_csv, newline="") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


# --------------------------------------------------------------------------------------------
# Verification (validation split only) and evaluation
# --------------------------------------------------------------------------------------------


def build_model(setup, state: dict, device: torch.device) -> IsoMemGPT:
    model = IsoMemGPT(setup.cfg, setup.bits, rope_theta=FIXED["model"]["rope_theta"]).to(device)
    model.assert_matches_accounting()
    model.load_state_dict(state)               # strict: shapes must match the replayed config
    return model


def standard_eval(model, tokens, setup, device) -> float:
    """train.evaluate, the procedure of every validation evaluation (50 x 128 x 256, EVAL_SEED)."""
    return TR.evaluate(model, tokens, setup.schedule, EVAL_BATCHES, device)


def verify_run(row: dict, ckpt_dir: Path, averaged_dir: Path, m1: dict[str, dict],
               val_tokens, device: torch.device) -> dict:
    """Everything checkable before the test split is opened. Returns the verified state."""
    check_software(row, device)
    if row.get("device") and torch.device(device).type != row["device"]:
        raise CheckpointProvenanceError(f"{row['run_id']}: device {device}, trained on "
                                        f"{row['device']}")
    setup = reconstruct(row)
    path = Path(ckpt_dir) / f"{setup.run_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path}: final checkpoint of a complete primary run missing")
    digest = sha256_file(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["step"] != setup.max_steps:
        raise CheckpointProvenanceError(f"{setup.run_id}: checkpoint at step {ck['step']}, "
                                        f"the run has {setup.max_steps}")
    last = ck["history"][-1]
    if last["step"] != ck["step"]:
        raise CheckpointProvenanceError(f"{setup.run_id}: history ends at {last['step']}, "
                                        f"checkpoint at {ck['step']}")
    state = {k: v.detach().clone() for k, v in ck["model"].items()}
    val_now = standard_eval(build_model(setup, state, device), val_tokens, setup, device)
    if abs(val_now - float(last["val"])) > MISMATCH_TOL:
        raise CheckpointProvenanceError(f"{setup.run_id}: val now {val_now:.6f}, recorded at "
                                        f"the final step {float(last['val']):.6f}")
    out = {"setup": setup, "state": state, "ckpt_sha256": digest,
           "val_recorded_final": float(last["val"]), "val_recheck": val_now,
           "state_c": None, "c_sha256": None, "c_reason": None}

    m = m1.get(setup.run_id)
    if m is None or m.get("status") != "complete":
        out["c_reason"] = f"m1 status {m.get('status') if m else 'absent'}"
        return out
    cpath = Path(averaged_dir) / f"{setup.run_id}.pt"
    if not cpath.exists():
        raise FileNotFoundError(f"{cpath}: m1.csv says complete but the (c) weights are missing")
    cck = torch.load(cpath, map_location="cpu", weights_only=True)
    if cck.get("run_id") != setup.run_id:
        raise CheckpointProvenanceError(f"{cpath}: holds run {cck.get('run_id')!r}")
    state_c = {k: v.detach().clone() for k, v in cck["model"].items()}
    val_c_now = standard_eval(build_model(setup, state_c, device), val_tokens, setup, device)
    if abs(val_c_now - float(m["val_c"])) > MISMATCH_TOL:
        raise CheckpointProvenanceError(f"{setup.run_id}: (c) val now {val_c_now:.6f}, "
                                        f"m1.csv val_c {float(m['val_c']):.6f}")
    out.update(state_c=state_c, c_sha256=sha256_file(cpath), val_c_recorded=float(m["val_c"]),
               val_c_recheck=val_c_now)
    return out


# --------------------------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------------------------


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_marker(marker: Path, sha: str, device: torch.device) -> None:
    """Exclusive create: two concurrent invocations cannot both pass this point."""
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"opened {datetime.now(timezone.utc).isoformat()} git {sha} device {device}\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_results_csv(results_csv: Path, losses: dict[str, float]) -> None:
    """Fill test_loss/test_ppl on the LAST row of each run id (the one load_rows keeps)."""
    with open(results_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        raw = list(reader)
    if "test_loss" not in fields or "test_ppl" not in fields:
        raise AssertionError("results.csv lacks the test_loss / test_ppl columns")
    last = {r["run_id"]: i for i, r in enumerate(raw)}
    for rid, loss in losses.items():
        row = raw[last[rid]]
        if row["status"] != "complete" or row["sweep"] != "primary" or row["test_loss"]:
            raise AssertionError(f"{rid}: refusing to write test_loss into {row['status']} row")
        row["test_loss"] = repr(loss)
        row["test_ppl"] = repr(math.exp(loss))

    def w(fh):
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(raw)

    _atomic_write(results_csv, w)


# --------------------------------------------------------------------------------------------
# The read
# --------------------------------------------------------------------------------------------


def run(results_csv: Path, ckpt_dir: Path, averaged_dir: Path, m1_csv: Path, log_path: Path,
        device: torch.device) -> dict:
    marker = results_csv.parent / MARKER_NAME
    assert_not_read(results_csv, marker)
    targets = primary_targets(results_csv)
    m1 = m1_rows(m1_csv)
    sha = git_sha()
    started = datetime.now(timezone.utc).isoformat()
    print(f"device {device}   git {sha}   torch {torch.__version__}   "
          f"cuda {torch.version.cuda or ''}   {len(targets)} primary runs", flush=True)

    # Phase 1: verify everything on the validation split. The test split stays sealed.
    val_tokens = D.load_tokens("val")
    verified = {}
    for i, row in enumerate(targets, 1):
        t0 = time.time()
        verified[row["run_id"]] = verify_run(row, ckpt_dir, averaged_dir, m1, val_tokens, device)
        v = verified[row["run_id"]]
        print(f"  verify [{i}/{len(targets)}] {row['run_id']:>22}  val {v['val_recheck']:.6f}"
              f" (recorded {v['val_recorded_final']:.6f})"
              f"{'' if v['state_c'] is not None else '  no (c): ' + v['c_reason']}"
              f"  {time.time() - t0:.1f}s", flush=True)

    # Phase 2: the one read. Marker first, so that no failure after this point can ever
    # be followed by a silent second read.
    write_marker(marker, sha, device)
    test_tokens = D.load_test_tokens(i_understand_this_is_the_final_evaluation=True)
    per_run, losses = [], {}
    for i, row in enumerate(targets, 1):
        t0 = time.time()
        v = verified[row["run_id"]]
        setup = v["setup"]
        loss = standard_eval(build_model(setup, v["state"], device), test_tokens, setup, device)
        if not math.isfinite(loss):
            raise FloatingPointError(f"{setup.run_id}: non-finite test loss {loss}")
        loss_c = None
        if v["state_c"] is not None:
            loss_c = standard_eval(build_model(setup, v["state_c"], device), test_tokens,
                                   setup, device)
        losses[setup.run_id] = loss
        per_run.append({
            "run_id": setup.run_id, "budget": row["budget_name"], "b": setup.bits,
            "seed": setup.seed, "test_loss": loss, "test_ppl": math.exp(loss),
            "test_loss_c": loss_c, "c_reason": v["c_reason"],
            "val_recorded_final": v["val_recorded_final"], "val_recheck": v["val_recheck"],
            "val_c_recorded": v.get("val_c_recorded"), "val_c_recheck": v.get("val_c_recheck"),
            "ckpt_sha256": v["ckpt_sha256"], "c_sha256": v["c_sha256"],
        })
        print(f"  test [{i}/{len(targets)}] {setup.run_id:>22}  test {loss:.6f}"
              f"{'' if loss_c is None else f'  test_c {loss_c:.6f}'}  {time.time() - t0:.1f}s",
              flush=True)

    log = {
        "procedure": "PREREGISTRATION.md §14 (+§16/§17 (c) weights): train.evaluate, "
                     f"{EVAL_BATCHES} batches x micro_batch windows x block_size, "
                     f"EVAL_SEED {TR.EVAL_SEED}, final-step checkpoints",
        "test_split_sha256": FIXED["data"]["hashes"]["test"],
        "device": str(device), "git_sha": sha, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(per_run), "n_with_c": sum(r["test_loss_c"] is not None for r in per_run),
        "per_run": per_run,
    }
    _atomic_write(log_path, lambda fh: (json.dump(log, fh, indent=2), fh.write("\n")))
    write_results_csv(results_csv, losses)
    print(f"wrote {log_path} and test_loss into {results_csv} for {len(losses)} runs")
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--i-have-the-go-ahead", action="store_true",
                    help="required: this opens the sealed test split, once, forever")
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--averaged-dir", type=Path, default=DEFAULT_AVERAGED_DIR)
    ap.add_argument("--m1-csv", type=Path, default=DEFAULT_M1_CSV)
    ap.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if not args.i_have_the_go_ahead:
        print("refused: the test split is read exactly once (§10.3, §14). Pass "
              "--i-have-the-go-ahead only with the project owner's explicit go-ahead.")
        return 2
    assert_fixed_invariants()
    if not CORPUS.is_default:
        raise TestReadRefused("§14 is the TinyStories test read; unset ISOMEM_DATASET")
    device = TR.resolve_device(args.device)
    run(args.results_csv, args.ckpt_dir, args.averaged_dir, args.m1_csv, args.log, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
