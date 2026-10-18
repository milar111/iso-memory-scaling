#!/usr/bin/env python
"""A7 M2-M4: learning-rate sensitivity, half-budget residual check and seeds.

    cd ~/bit-budget-a7 && python scripts/a7_followup.py --main-dir ~/bit-budget
                                                      (normally via a7_followup_tick.sh)

PREREGISTRATION.md §16 (Amendment A7), subsections "Argmin groups", "M2", "M3" and "M4", is
the contract; this script does what they say and nothing more. Inputs: this copy's
``results/results.csv`` (the TinyStories sweep) and ``<out-dir>/m1.csv`` (M1's ``val_c`` per
run). Nothing is planned until all 142 M1 rows are recorded ``complete``. Then, once:

* **argmin groups**, per budget of the primary sweep, with ``m_c``/``s_c`` the ``val_c`` cell
  means and sample sds over seeds 0-4 and ``b*_c`` their argmin: ``b*_c``, its grid
  neighbours, and every ``b`` with ``m_c(b) - m_c(b*_c) <= max(0.03, 2 * max(s_c(b),
  s_c(b*_c)))``. Seeds for M2 and M3: 0 only where every other member is more than 0.05
  nats from ``b*_c`` on ``m_c``, else 0-2;
* **M2**: every group cell, primary shape, learning-rate multiplier 0.5 and 2 on the rule's
  value (multiplier 1 is the primary run of the same seed);
* **M3**: every group cell, primary shape, half its sweep token budget with a full cosine
  schedule over that length (``max_tokens = token_budget // 2``), paired by seed;
* **M4**: wherever on ``val_c`` the §4 criterion returns NULL at a budget, or the Welch p
  between ``b*_c`` and a group member exceeds 0.01, both cells get seeds 5, 6, ... up to the
  per-cell n at which a fixed-seed simulation gives 80% power at the observed ``val_c`` gap
  and sds, capped at 10 per cell.

That plan goes to ``<out-dir>/plan.json``, with the sha256 of the m1.csv and results.csv
bytes it was computed from, and is frozen: later invocations read it, never recompute it,
and refuse to run if either input file has changed since. After every M2 run is recorded,
a cell whose best multiplier by mean ``val_c`` (over the same seeds, multiplier 1 from
m1.csv) is 0.5 or 2 is extended to 0.25 or 4, same seeds; that decision is written once to
``<out-dir>/m2_extensions.json`` and frozen the same way.

Order: M2 and M3 runs (cheapest first, by the cell's measured sweep throughput), then the M2
extensions, then M4. Every run is trained with the repo's own ``train.train_one`` --
TinyStories, the primary run's shape, micro-batch and seed stream, everything standard
except the one varied factor -- with checkpoints in ``<out-dir>/runs/`` (resumable across
quiet hours exactly as the sweep is: ``sweep.next_deadline``, checkpoint, resume), and then
gets the M1 continuation through ``a7_average.process_run``, unchanged, so that every
comparison is on ``val_c``. Its (c) weights go to ``<out-dir>/followup/averaged/``, apart
from M1's. Rows go to ``<out-dir>/m2.csv``, ``m3.csv``, ``m4.csv``; nothing ever enters
results.csv, and nothing is written under the main checkout (an audit hook refuses it).

Guards as in a7_average.py: CUDA only (``--allow-cpu`` for the tests), the device of m1.csv,
the torch/CUDA build of the primary runs, quiet hours (nothing starts within 20 min of
SLEEP_AT), the FineWeb-Edu arm's ``.finalised`` marker in the main checkout, never a test
split, never ``ISOMEM_DATASET``.

Exit codes: 0 all planned work recorded complete; 3 work remains; 2 M1 is not complete yet,
nothing planned or run (the tick stays silent on it); 1 an error, including any recorded
row that is not complete.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The copy of isomem next to this script (see a7_average.py).
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isomem  # noqa: E402
from isomem import data as D  # noqa: E402
from isomem import train as TR  # noqa: E402
from isomem.accounting import assert_within_budget, count_params  # noqa: E402
from isomem.analysis import load_cells, null_verdict, welch  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS  # noqa: E402
from isomem.export import CheckpointProvenanceError  # noqa: E402
from isomem.fixed import FIXED, GRID, assert_fixed_invariants, git_sha, learning_rate  # noqa: E402
from isomem.quant import TernaryCollapse  # noqa: E402
from isomem.search import search_cell  # noqa: E402
from isomem.sweep import next_deadline, run_id as sweep_run_id, token_budget  # noqa: E402


def _import_m1():
    """scripts/a7_average.py, the M1 code, imported unchanged (and only once per process)."""
    name, path = "a7_average", (REPO_ROOT / "scripts" / "a7_average.py").resolve()
    mod = sys.modules.get(name)
    if mod is not None:
        if Path(mod.__file__).resolve() != path:
            raise ImportError(f"a7_average is already imported from {mod.__file__}, not {path}")
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


A7 = _import_m1()

# --- fixed by A7 (PREREGISTRATION.md §16) ---------------------------------------------------
PRIMARY = "primary"
PRIMARY_SHAPE = "TIGHTEST"
PRIMARY_SEEDS = (0, 1, 2, 3, 4)
GROUP_MARGIN_NATS = 0.03            # "max(0.03, 2 * max(s_c(b), s_c(b*_c)))"
GROUP_SD_MULTIPLE = 2.0
FAR_NATS = 0.05                     # "more than 0.05 nats from it on m_c" -> seed 0 only
SEEDS_FAR = (0,)
SEEDS_NEAR = (0, 1, 2)
M2_MULTIPLIERS = (0.5, 2.0)
M2_EXTENSION = {0.5: 0.25, 2.0: 4.0}
M3_TOKEN_FACTOR = 0.5               # "half its sweep token budget"
M4_TRIGGER_P = 0.01                 # "the Welch p between b*_c and a group member exceeds 0.01"
M4_POWER = 0.80
M4_CAP = 10                         # "capped at 10 seeds per cell"
M4_FIRST_NEW_SEED = 5

# --- left open by A7, fixed here before any M2-M4 run; every one is recorded in plan.json ---
# The level at which the power simulation counts a comparison resolved: the level M4 uses to
# call one unresolved (p > 0.01), so that "80% power" means an 80% chance of p <= 0.01.
M4_POWER_ALPHA = 0.01
# The simulation's random seed (an arbitrary value, fixed before any M1 result was read by
# this code) and its number of draws.
# The Monte-Carlo sd of a simulated power near 0.8 is sqrt(0.16 / SIM_DRAWS) = 0.003.
SIM_SEED = 20_261_002
SIM_DRAWS = 20_000

ARM_CSV = {"m2": "m2.csv", "m2-ext": "m2.csv", "m3": "m3.csv", "m4": "m4.csv"}
PLAN_NAME = "plan.json"
EXTENSIONS_NAME = "m2_extensions.json"
CKPT_SUBDIR = "runs"
AVERAGED_PARENT = "followup"        # process_run saves (c) weights to <this>/averaged/<run_id>.pt
LOCK_NAME = ".followup.lock"        # in --out-dir; a7_followup_tick.sh's own lock is elsewhere
PLAN_VERSION = 1
RESULTS_CSV = A7.RESULTS_CSV
EXIT_OK, EXIT_ERROR, EXIT_WAIT_M1, EXIT_MORE = 0, 1, 2, 3

FIELDS = [
    "run_id", "arm", "status", "paired_run_id", "reference_run_id", "budget_name",
    "budget_bits", "bits", "seed", "n_layer", "n_embd", "mlp_ratio", "params",
    "lr_multiplier", "lr", "token_factor", "max_tokens", "tokens_seen", "steps", "best_step",
    "micro_batch", "grad_accum",
    # the M1 continuation of this run, named as in m1.csv
    "val_loss_best", "val_recorded_final", "val_ckpt", "val_fresh",
    "A_mean", "A_sd", "A_evals", "val_b", "val_c", "val_d", "k", "lr_used",
    "device", "peak_mem_gib", "peak_reserved_gib", "wall_s",
    "torch_version", "cuda_version", "git_sha", "error",
]


class PlanRefused(RuntimeError):
    """A frozen record no longer matches the files it was computed from."""


# --------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> tuple[bytes, dict[str, dict]]:
    """The file's bytes and its rows by run_id, parsed from those same bytes (so a hash
    taken of them names exactly what was parsed). Last row wins, as everywhere in the repo."""
    if not path.exists():
        return b"", {}
    data = path.read_bytes()
    rows = {r["run_id"]: r for r in csv.DictReader(io.StringIO(data.decode(), newline=""))}
    return data, rows


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({f: row.get(f, "") for f in FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def write_json_once(path: Path, obj: dict) -> None:
    """Atomic, and never over an existing file: a frozen record is written exactly once."""
    if path.exists():
        raise PlanRefused(f"{path} exists; a frozen record is never rewritten")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def recorded(out_dir: Path) -> dict[str, dict]:
    """Every A7 follow-up row, by run_id, across m2.csv, m3.csv and m4.csv."""
    out: dict[str, dict] = {}
    for name in sorted(set(ARM_CSV.values())):
        _, rows = read_csv(out_dir / name)
        clash = set(out) & set(rows)
        if clash:
            raise AssertionError(f"run ids recorded in two A7 files: {sorted(clash)}")
        out.update(rows)
    return out


# --------------------------------------------------------------------------------------------
# M1: is it complete?
# --------------------------------------------------------------------------------------------


def m1_state(results_csv: Path, m1_rows: dict[str, dict]) -> tuple[str, str]:
    """("complete" | "incomplete" | "failed", why). Complete means every one of A7's 142
    runs has an m1.csv row whose status is ``complete``."""
    ids = {r["run_id"] for r in A7.target_rows(results_csv)}
    missing = ids - set(m1_rows)
    bad = sorted(r for r in ids & set(m1_rows) if m1_rows[r]["status"] != "complete")
    if bad:
        return "failed", "m1.csv rows not complete: " + ", ".join(
            f"{r} ({m1_rows[r]['status']})" for r in bad)
    if missing:
        return "incomplete", f"M1 has {len(ids) - len(missing)} of {len(ids)} runs recorded"
    return "complete", f"M1 complete: {len(ids)} runs"


# --------------------------------------------------------------------------------------------
# Cells on val_c, and the argmin groups
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellStat:
    """One primary (budget, b) cell on M1's val_c: seeds 0-4, in seed order."""

    budget_name: str
    budget_bits: int
    bits: float
    val_c: tuple[float, ...]

    @property
    def mean(self) -> float:
        return statistics.mean(self.val_c)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.val_c)


def primary_rows(results_csv: Path) -> list[dict]:
    """The primary sweep's complete runs (A7.target_rows asserts all 142 runs exist)."""
    return [r for r in A7.target_rows(results_csv) if r["sweep"] == PRIMARY]


def budget_order(names) -> list[str]:
    order = {b["name"]: i for i, b in enumerate(GRID["budgets_bits"])}
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))


def cell_stats(rows: list[dict], m1_rows: dict[str, dict]) -> dict[str, list[CellStat]]:
    """Per budget (in grid order), its cells sorted by b, each with seeds 0-4's val_c."""
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        if r["shape_rule"] != PRIMARY_SHAPE:
            raise AssertionError(f"{r['run_id']}: primary run with shape {r['shape_rule']}")
        groups.setdefault((r["budget_name"], float(r["bits_per_weight"])), []).append(r)
    out: dict[str, list[CellStat]] = {}
    for (name, bits), rs in groups.items():
        rs = sorted(rs, key=lambda r: int(r["seed"]))
        if tuple(int(r["seed"]) for r in rs) != PRIMARY_SEEDS:
            raise AssertionError(f"{name} b={bits:g}: seeds {[r['seed'] for r in rs]}")
        budgets = {int(r["budget_bits_target"]) for r in rs}
        if len(budgets) != 1:
            raise AssertionError(f"{name} b={bits:g}: budgets {budgets}")
        vals = []
        for r in rs:
            m = m1_rows[r["run_id"]]
            if m["status"] != "complete" or not math.isfinite(float(m["val_c"])):
                raise AssertionError(f"{r['run_id']}: no complete val_c in m1.csv")
            vals.append(float(m["val_c"]))
        out.setdefault(name, []).append(CellStat(name, budgets.pop(), bits, tuple(vals)))
    return {n: sorted(out[n], key=lambda c: c.bits) for n in budget_order(out)}


@dataclass(frozen=True)
class Group:
    budget_name: str
    best: CellStat
    members: tuple[CellStat, ...]            # sorted by b; b*_c included
    reasons: tuple[tuple[float, tuple[str, ...]], ...]
    seeds: tuple[int, ...]
    far: bool                                # every other member > FAR_NATS from b*_c


def argmin_group(cells: list[CellStat]) -> Group:
    """PREREGISTRATION.md §16, "Argmin groups": ``b*_c``, its grid neighbours, and every b
    with ``m_c(b) - m_c(b*_c) <= max(0.03, 2 * max(s_c(b), s_c(b*_c)))``. Seeds 0 only where
    every member other than ``b*_c`` is more than 0.05 nats from it on ``m_c``, else 0-2.

    The grid neighbours are the adjacent bit-widths among the budget's cells (256 Kbit has
    no b = 16 cell, so there b = 8 has one neighbour), as in the §4 criterion.
    """
    cells = sorted(cells, key=lambda c: c.bits)
    if len({c.budget_name for c in cells}) != 1 or len(cells) < 2:
        raise ValueError("argmin_group takes the cells of exactly one budget, at least two")
    best = min(cells, key=lambda c: c.mean)
    i = cells.index(best)
    neighbours = {cells[j].bits for j in (i - 1, i + 1) if 0 <= j < len(cells)}
    members, reasons = [], []
    for c in cells:
        why = []
        if c is best:
            why.append("argmin")
        if c.bits in neighbours:
            why.append("grid neighbour")
        if c is not best:
            gap = c.mean - best.mean
            limit = max(GROUP_MARGIN_NATS, GROUP_SD_MULTIPLE * max(c.sd, best.sd))
            if gap <= limit:
                why.append(f"m_c gap {gap:.5f} <= max({GROUP_MARGIN_NATS}, "
                           f"{GROUP_SD_MULTIPLE:g} x max sd) = {limit:.5f}")
        if why:
            members.append(c)
            reasons.append((c.bits, tuple(why)))
    far = all(c.mean - best.mean > FAR_NATS for c in members if c is not best)
    return Group(best.budget_name, best, tuple(members), tuple(reasons),
                 SEEDS_FAR if far else SEEDS_NEAR, far)


# --------------------------------------------------------------------------------------------
# M4: which comparisons, and how many seeds
# --------------------------------------------------------------------------------------------


def standard_normals(seed: int = None, draws: int = None, cap: int = None) -> np.ndarray:
    """The simulation's fixed draws: (draws, 2 cells, cap seeds) standard normals."""
    seed = SIM_SEED if seed is None else seed
    draws = SIM_DRAWS if draws is None else draws
    cap = M4_CAP if cap is None else cap
    return np.random.default_rng(seed).standard_normal((draws, 2, cap))


def simulated_power(m_a: float, s_a: float, m_b: float, s_b: float, z: np.ndarray,
                    alpha: float = None) -> dict[int, float]:
    """P(Welch p <= alpha) at n = 5 ... cap seeds per cell, if the cells' true means and sds
    are the observed ones. Each draw takes its first n values per cell, so every n sees the
    same draws. The test is ``analysis.welch``, the one the analysis applies."""
    alpha = M4_POWER_ALPHA if alpha is None else alpha
    a = (m_a + s_a * z[:, 0, :]).tolist()
    b = (m_b + s_b * z[:, 1, :]).tolist()
    out = {}
    for n in range(len(PRIMARY_SEEDS), z.shape[2] + 1):
        hits = sum(welch(x[:n], y[:n])[2] <= alpha for x, y in zip(a, b))
        out[n] = hits / len(a)
    return out


def required_n(power: dict[int, float]) -> int | None:
    """The smallest simulated n per cell reaching M4_POWER, or None if none up to the cap."""
    for n in sorted(power):
        if power[n] >= M4_POWER:
            return n
    return None


def analysis_cells(results_csv: Path, stats: dict[str, list[CellStat]]):
    """isomem.analysis cells (as the §4 analysis builds them) with val_c in place of
    val_loss_best, so the §4 criterion runs unchanged. Every cell must be usable: §4 would
    otherwise drop it silently, and the argmin groups do not."""
    by = {(c.budget_name, c.bits): c for c in load_cells(results_csv, sweep=PRIMARY)}
    out = []
    for name, cells in stats.items():
        for s in cells:
            c = by[(name, s.bits)]
            if not c.usable or c.n != len(s.val_c):
                raise AssertionError(f"{name} b={s.bits:g}: not every seed trained to its "
                                     "full budget; §4 would exclude it")
            out.append(dataclasses.replace(c, val_losses=list(s.val_c)))
    return out


def m4_comparisons(results_csv: Path, stats: dict[str, list[CellStat]],
                   groups: dict[str, Group], z: np.ndarray) -> list[dict]:
    """PREREGISTRATION.md §16, M4: the comparisons that get new seeds, with their triggers
    and simulated power. A NULL budget triggers b*_c against each adjacent cell whose
    distance from it is within the §4 ``s_max`` (the ones that make the budget NULL)."""
    acells = analysis_cells(results_csv, stats)
    out = []
    for name, cells in stats.items():
        g = groups[name]
        best = g.best
        by_bits = {c.bits: c for c in cells}
        nv = null_verdict(acells, name)
        if nv.note or nv.best_bits != best.bits:
            raise AssertionError(f"{name}: §4 argmin {nv.best_bits} vs b*_c {best.bits} "
                                 f"({nv.note})")
        triggers: dict[float, list[str]] = {}
        if nv.is_null:
            for nb in nv.adjacent:
                d = abs(by_bits[nb].mean - best.mean)
                if d <= nv.s_max:
                    triggers.setdefault(nb, []).append(
                        f"§4 NULL: gap {d:.5f} <= s_max {nv.s_max:.5f}")
        pvals = {}
        for c in g.members:
            if c is best:
                continue
            pvals[c.bits] = welch(list(best.val_c), list(c.val_c))[2]
            if pvals[c.bits] > M4_TRIGGER_P:
                triggers.setdefault(c.bits, []).append(
                    f"Welch p {pvals[c.bits]:.4g} > {M4_TRIGGER_P}")
        for bits in sorted(triggers):
            other = by_bits[bits]
            power = simulated_power(best.mean, best.sd, other.mean, other.sd, z)
            n_req = required_n(power)
            out.append({
                "budget_name": name, "best_bits": best.bits, "other_bits": bits,
                "best_budget_bits": best.budget_bits, "other_budget_bits": other.budget_bits,
                "gap": other.mean - best.mean, "sd_best": best.sd, "sd_other": other.sd,
                "welch_p": pvals[bits],     # every triggered cell is a member (neighbours are)
                "null_verdict": {"is_null": nv.is_null, "gap": nv.gap, "s_max": nv.s_max,
                                 "adjacent": nv.adjacent},
                "triggers": triggers[bits],
                "power_by_n": {str(n): p for n, p in power.items()},
                "n_required": n_req,
                "n_per_cell": n_req if n_req is not None else M4_CAP,
                "unresolved_expected_at_cap": n_req is None,
            })
    return out


def m4_seeds(comparisons: list[dict]) -> dict[tuple[str, float], int]:
    """Total seeds per cell: the largest n any of its comparisons needs, capped at M4_CAP."""
    need: dict[tuple[str, float], int] = {}
    for c in comparisons:
        n = min(M4_CAP, c["n_per_cell"])
        for bits in (c["best_bits"], c["other_bits"]):
            key = (c["budget_name"], bits)
            need[key] = max(need.get(key, len(PRIMARY_SEEDS)), n)
    return need


# --------------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    arm: str                  # m2 | m2-ext | m3 | m4
    budget_name: str
    budget_bits: int
    bits: float
    seed: int
    lr_multiplier: float
    token_factor: float
    params: int
    max_tokens: int
    paired_run_id: str        # the primary run of the same cell and seed ("" for a new seed)
    reference_run_id: str     # the primary run whose shape and batch schedule this replays
    est_train_gpu_h: float    # max_tokens / the cell's median sweep throughput

    @property
    def phase(self) -> int:
        return {"m2": 0, "m3": 0, "m2-ext": 1, "m4": 2}[self.arm]


def followup_run_id(arm: str, budget_bits: int, bits: float, seed: int,
                    lr_multiplier: float = 1.0) -> str:
    base = sweep_run_id(budget_bits, bits, seed, PRIMARY_SHAPE)
    if arm in ("m2", "m2-ext"):
        return f"{base}-lr{lr_multiplier:g}"
    if arm == "m3":
        return f"{base}-half"
    if arm == "m4":
        if seed < M4_FIRST_NEW_SEED:
            raise ValueError(f"M4 seeds start at {M4_FIRST_NEW_SEED}, not {seed}")
        return base
    raise ValueError(arm)


def cell_throughput(rows: list[dict]) -> dict[tuple[str, float], float]:
    """Median tokens/s over a cell's primary runs. results.csv's wall_clock_s covers only the
    last fragment of a resumed run (train.py), so single runs overstate throughput; the
    median over the 5 seeds is robust to a minority of fragments."""
    by: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        by.setdefault((r["budget_name"], float(r["bits_per_weight"])), []).append(
            int(r["tokens_seen"]) / float(r["wall_clock_s"]))
    return {k: statistics.median(v) for k, v in by.items()}


def planned_max_tokens(arm: str, params: int) -> int:
    """The sweep's token budget for the cell; M3 trains on exactly its integer half."""
    full = token_budget(params)
    return full // 2 if arm == "m3" else full


def _planned(arm, cell: CellStat, seed, mult, ref: dict, tps: float) -> PlannedRun:
    params = int(ref["params_counted"])
    max_tokens = planned_max_tokens(arm, params)
    return PlannedRun(
        run_id=followup_run_id(arm, cell.budget_bits, cell.bits, seed, mult), arm=arm,
        budget_name=cell.budget_name, budget_bits=cell.budget_bits, bits=cell.bits, seed=seed,
        lr_multiplier=mult, token_factor=M3_TOKEN_FACTOR if arm == "m3" else 1.0,
        params=params, max_tokens=max_tokens,
        paired_run_id=ref["run_id"] if seed in PRIMARY_SEEDS else "",
        reference_run_id=ref["run_id"], est_train_gpu_h=max_tokens / tps / 3600)


def ordered(runs) -> list[PlannedRun]:
    """M2 and M3 first, then the M2 extensions, then M4; cheapest first within each."""
    return sorted(runs, key=lambda r: (r.phase, r.est_train_gpu_h, r.run_id))


def compute_plan(results_csv: Path, m1_bytes: bytes, m1_rows: dict[str, dict],
                 results_bytes: bytes) -> dict:
    rows = primary_rows(results_csv)
    stats = cell_stats(rows, m1_rows)
    groups = {name: argmin_group(cells) for name, cells in stats.items()}
    by_seed = {(r["budget_name"], float(r["bits_per_weight"]), int(r["seed"])): r for r in rows}
    tps = cell_throughput(rows)

    runs = []
    for name, g in groups.items():
        for c in g.members:
            for seed in g.seeds:
                ref = by_seed[(name, c.bits, seed)]
                for mult in M2_MULTIPLIERS:
                    runs.append(_planned("m2", c, seed, mult, ref, tps[(name, c.bits)]))
                runs.append(_planned("m3", c, seed, 1.0, ref, tps[(name, c.bits)]))

    z = standard_normals()
    comparisons = m4_comparisons(results_csv, stats, groups, z)
    need = m4_seeds(comparisons)
    cells = {(c.budget_name, c.bits): c for cs in stats.values() for c in cs}
    for (name, bits), n in sorted(need.items()):
        for seed in range(M4_FIRST_NEW_SEED, n):
            ref = by_seed[(name, bits, 0)]
            runs.append(_planned("m4", cells[(name, bits)], seed, 1.0, ref, tps[(name, bits)]))

    budgets = []
    for name, g in groups.items():
        others = [c for c in g.members if c is not g.best]
        budgets.append({
            "budget_name": name, "argmin_bits": g.best.bits,
            "cells": [{"bits": c.bits, "budget_bits": c.budget_bits, "mean_val_c": c.mean,
                       "sd_val_c": c.sd, "n": len(c.val_c), "val_c": list(c.val_c)}
                      for c in stats[name]],
            "group": [{"bits": b, "reasons": list(why)} for b, why in g.reasons],
            "seeds_m2_m3": list(g.seeds),
            "seed_rule": {
                "every_other_member_more_than_0.05_nats": g.far,
                "min_gap_to_other_members": min(c.mean - g.best.mean for c in others),
                # §16's parenthetical "(at least 10 seed sds there)", measured, not enforced
                "min_gap_over_larger_sd": min((c.mean - g.best.mean) / max(c.sd, g.best.sd)
                                              for c in others),
            },
        })
    plan = {
        "version": PLAN_VERSION,
        "what": "PREREGISTRATION.md §16 (A7): argmin groups on val_c and the M2-M4 runs. "
                "Written once, then frozen.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "m1_csv_sha256": _sha256(m1_bytes),
        "results_csv_sha256": _sha256(results_bytes),
        "rules": {
            "group_margin_nats": GROUP_MARGIN_NATS, "group_sd_multiple": GROUP_SD_MULTIPLE,
            "far_nats": FAR_NATS, "seeds_far": list(SEEDS_FAR), "seeds_near": list(SEEDS_NEAR),
            "m2_multipliers": list(M2_MULTIPLIERS),
            "m2_extension": {str(k): v for k, v in M2_EXTENSION.items()},
            "m3_token_factor": M3_TOKEN_FACTOR, "m3_max_tokens": "token_budget(params) // 2",
            "m4_trigger_p": M4_TRIGGER_P, "m4_power": M4_POWER, "m4_cap": M4_CAP,
            "m4_power_alpha": M4_POWER_ALPHA, "sim_seed": SIM_SEED, "sim_draws": SIM_DRAWS,
            "sim_method": "numpy default_rng(sim_seed).standard_normal((draws, 2, cap)); "
                          "cell = observed mean + observed sd * z; n = first n values per "
                          "cell; power = share of draws with analysis.welch p <= "
                          "m4_power_alpha",
            "numpy_version": np.__version__,
        },
        "budgets": budgets,
        "m4": {"comparisons": comparisons,
               "seeds_per_cell": [{"budget_name": n, "bits": b, "n_total": k,
                                   "new_seeds": list(range(M4_FIRST_NEW_SEED, k))}
                                  for (n, b), k in sorted(need.items())]},
        "runs": [dataclasses.asdict(r) for r in ordered(runs)],
    }
    _check_plan(plan)
    return plan


def _check_plan(plan: dict) -> None:
    """Invariants of §16 the plan must satisfy, asserted (hard rule 1), on every load."""
    runs = plan_runs(plan)
    ids = [r.run_id for r in runs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate run ids in the plan")
    for b in plan["budgets"]:
        bits = sorted(c["bits"] for c in b["cells"])
        i = bits.index(b["argmin_bits"])
        group = {g["bits"] for g in b["group"]}
        if not {bits[j] for j in (i - 1, i, i + 1) if 0 <= j < len(bits)} <= group:
            raise AssertionError(f"{b['budget_name']}: group misses b*_c or a neighbour")
        if tuple(b["seeds_m2_m3"]) not in (SEEDS_FAR, SEEDS_NEAR):
            raise AssertionError(f"{b['budget_name']}: seeds {b['seeds_m2_m3']}")
        for bit in group:
            for arm, mults in (("m2", M2_MULTIPLIERS), ("m3", (1.0,))):
                got = sorted((r.seed, r.lr_multiplier) for r in runs if r.arm == arm
                             and r.budget_name == b["budget_name"] and r.bits == bit)
                want = sorted((s, m) for s in b["seeds_m2_m3"] for m in mults)
                if got != want:
                    raise AssertionError(f"{b['budget_name']} b={bit:g}: {arm} runs {got}")
        if any(r.arm in ("m2", "m3") and r.budget_name == b["budget_name"]
               and r.bits not in group for r in runs):
            raise AssertionError(f"{b['budget_name']}: an M2/M3 run outside the group")
    for r in runs:
        if r.arm not in ("m2", "m3", "m4"):
            raise AssertionError(f"{r.run_id}: arm {r.arm} in the frozen plan")
        if r.max_tokens != planned_max_tokens(r.arm, r.params):
            raise AssertionError(f"{r.run_id}: {r.max_tokens} tokens, not the {r.arm} budget")
        if r.arm == "m4" and not (M4_FIRST_NEW_SEED <= r.seed < M4_CAP):
            raise AssertionError(f"{r.run_id}: M4 seed outside {M4_FIRST_NEW_SEED}-{M4_CAP - 1}")
        if (r.arm == "m2") != (r.lr_multiplier != 1.0) or \
                (r.arm == "m2" and r.lr_multiplier not in M2_MULTIPLIERS):
            raise AssertionError(f"{r.run_id}: multiplier {r.lr_multiplier} in arm {r.arm}")
        if r.run_id != followup_run_id(r.arm, r.budget_bits, r.bits, r.seed, r.lr_multiplier):
            raise AssertionError(f"{r.run_id}: id does not replay from its fields")


def plan_runs(plan: dict) -> list[PlannedRun]:
    return [PlannedRun(**r) for r in plan["runs"]]


def load_or_create_plan(out_dir: Path, results_csv: Path, m1_bytes: bytes,
                        m1_rows: dict[str, dict]) -> tuple[dict, bool]:
    """The frozen plan, and whether it was created now. An existing plan is read, never
    recomputed, and refused if m1.csv or results.csv differ from the bytes it was computed
    from."""
    path = out_dir / PLAN_NAME
    results_bytes = results_csv.read_bytes()
    if path.exists():
        plan = json.loads(path.read_text())
        if plan.get("version") != PLAN_VERSION:
            raise PlanRefused(f"{path}: version {plan.get('version')}, expected {PLAN_VERSION}")
        if plan["m1_csv_sha256"] != _sha256(m1_bytes):
            raise PlanRefused(f"m1.csv has changed since {path} was frozen from it "
                              f"(sha256 {plan['m1_csv_sha256'][:12]}..., now "
                              f"{_sha256(m1_bytes)[:12]}...)")
        if plan["results_csv_sha256"] != _sha256(results_bytes):
            raise PlanRefused(f"results.csv has changed since {path} was frozen")
        _check_plan(plan)
        return plan, False
    plan = compute_plan(results_csv, m1_bytes, m1_rows, results_bytes)
    write_json_once(path, plan)
    return plan, True


# --------------------------------------------------------------------------------------------
# The M2 extensions: decided once every M2 run is recorded, then frozen
# --------------------------------------------------------------------------------------------


def rows_digest(rows: list[dict]) -> str:
    return _sha256(json.dumps(sorted(rows, key=lambda r: r["run_id"]),
                              sort_keys=True).encode())


def decide_extensions(plan: dict, rec: dict[str, dict], m1_rows: dict[str, dict]) -> dict | None:
    """PREREGISTRATION.md §16, M2: a cell whose best multiplier by mean val_c is 0.5 or 2 is
    extended to 0.25 or 4, same seeds. None until every M2 run of the plan is recorded.

    The mean of each multiplier is over the cell's M2/M3 seeds; multiplier 1 is the primary
    run of each of those seeds (its m1.csv val_c). A multiplier with a run that did not
    complete (diverged: nan, collapsed) has no mean and cannot be the best; it is listed.
    A tie goes to the rule's own value.
    """
    m2 = [r for r in plan_runs(plan) if r.arm == "m2"]
    if any(r.run_id not in rec for r in m2):
        return None
    cells: dict[tuple[str, float], list[PlannedRun]] = {}
    for r in m2:
        cells.setdefault((r.budget_name, r.bits), []).append(r)
    decisions, runs = [], []
    order = {n: i for i, n in enumerate(budget_order({k[0] for k in cells}))}
    for (name, bits), rs in sorted(cells.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        seeds = sorted({r.seed for r in rs})
        paired = {r.seed: r.paired_run_id for r in rs}
        values = {1.0: [float(m1_rows[paired[s]]["val_c"]) for s in seeds]}
        excluded = {}
        for mult in M2_MULTIPLIERS:
            got = sorted((r for r in rs if r.lr_multiplier == mult), key=lambda r: r.seed)
            if [r.seed for r in got] != seeds:
                raise AssertionError(f"{name} b={bits:g}: x{mult:g} seeds {[r.seed for r in got]}")
            bad = [f"{r.run_id} ({rec[r.run_id]['status']})" for r in got
                   if rec[r.run_id]["status"] != "complete"
                   or not math.isfinite(float(rec[r.run_id]["val_c"]))]
            if bad:
                excluded[f"{mult:g}"] = bad
            else:
                values[mult] = [float(rec[r.run_id]["val_c"]) for r in got]
        means = {m: statistics.mean(v) for m, v in values.items()}
        best = min(means, key=lambda m: (means[m], m != 1.0))
        extend = M2_EXTENSION.get(best)
        decisions.append({
            "budget_name": name, "bits": bits, "seeds": seeds,
            "mean_val_c": {f"{m:g}": v for m, v in sorted(means.items())},
            "excluded": excluded, "best_multiplier": best, "extend_to": extend,
        })
        if extend is not None:
            for r in sorted(rs, key=lambda r: r.seed):
                if r.lr_multiplier == M2_MULTIPLIERS[0]:
                    runs.append(dataclasses.replace(
                        r, arm="m2-ext", lr_multiplier=extend,
                        run_id=followup_run_id("m2-ext", r.budget_bits, r.bits, r.seed, extend)))
    return {
        "version": PLAN_VERSION,
        "what": "PREREGISTRATION.md §16 (A7) M2 extensions, decided once every M2 run was "
                "recorded. Frozen.",
        "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "m2_rows_sha256": rows_digest([rec[r.run_id] for r in m2]),
        "cells": decisions,
        "runs": [dataclasses.asdict(r) for r in ordered(runs)],
    }


def load_or_decide_extensions(out_dir: Path, plan: dict, rec: dict[str, dict],
                              m1_rows: dict[str, dict]) -> dict | None:
    path = out_dir / EXTENSIONS_NAME
    m2_ids = [r.run_id for r in plan_runs(plan) if r.arm == "m2"]
    if path.exists():
        ext = json.loads(path.read_text())
        if any(i not in rec for i in m2_ids) or \
                ext["m2_rows_sha256"] != rows_digest([rec[i] for i in m2_ids]):
            raise PlanRefused(f"the M2 rows have changed since {path} was decided from them")
        return ext
    ext = decide_extensions(plan, rec, m1_rows)
    if ext is not None:
        write_json_once(path, ext)
        for d in ext["cells"]:
            print(f"M2 {d['budget_name']} b={d['bits']:g}: mean val_c by multiplier "
                  + "  ".join(f"x{m} {v:.4f}" for m, v in d["mean_val_c"].items())
                  + f" -> best x{d['best_multiplier']:g}"
                  + (f", extended to x{d['extend_to']:g}" if d["extend_to"] else "")
                  + (f"; excluded {d['excluded']}" if d["excluded"] else ""), flush=True)
    return ext


def pending(out_dir: Path, plan: dict, m1_rows: dict[str, dict]
            ) -> tuple[list[PlannedRun], list[PlannedRun]]:
    """(unrecorded runs in execution order, every run known so far). The M2 extensions are
    known only once every M2 run is recorded; until then they are not in either list."""
    rec = recorded(out_dir)
    runs = plan_runs(plan)
    first = [r for r in ordered(r for r in runs if r.phase == 0) if r.run_id not in rec]
    if first:
        return first, runs
    ext = load_or_decide_extensions(out_dir, plan, rec, m1_rows)
    known = runs + [PlannedRun(**r) for r in ext["runs"]]
    return [r for r in ordered(known) if r.run_id not in rec], known


def outcome(out_dir: Path, plan: dict, m1_rows: dict[str, dict]) -> int:
    """0 only when every planned run (extensions included) is recorded AND complete; 1 while
    any recorded row is not complete, so the tick never retires over a failure; else 3."""
    todo, known = pending(out_dir, plan, m1_rows)
    rec = recorded(out_dir)
    bad = sorted(r.run_id for r in known if r.run_id in rec
                 and rec[r.run_id]["status"] != "complete")
    if bad:
        print("recorded follow-up runs that are not complete: " + ", ".join(
            f"{r} ({rec[r]['status']})" for r in bad), flush=True)
        return EXIT_ERROR
    extensions_known = (out_dir / EXTENSIONS_NAME).exists()
    return EXIT_OK if extensions_known and not todo else EXIT_MORE


# --------------------------------------------------------------------------------------------
# One run: train it as the sweep would, then the M1 continuation
# --------------------------------------------------------------------------------------------


def followup_setup(run: PlannedRun, ref: dict) -> "A7.Setup":
    """The run exactly as ``train_one`` will train it: the primary run's shape (the sweep's
    own search, cross-checked against ``ref``) and micro-batch, the sweep's token budget
    (halved for M3), the rule's learning rate times the multiplier. Any disagreement with
    ``ref`` raises; for an unvaried run this equals ``a7_average.reconstruct(ref)`` (tested).
    """
    if (ref["sweep"], ref["shape_rule"]) != (PRIMARY, PRIMARY_SHAPE) or \
            (int(ref["budget_bits_target"]), float(ref["bits_per_weight"])) != \
            (run.budget_bits, run.bits):
        raise CheckpointProvenanceError(f"{run.run_id}: reference {ref['run_id']} is not its "
                                        "primary cell")
    found = search_cell(run.budget_bits, run.bits, shape_rule=PRIMARY_SHAPE)
    cfg = found.config
    recorded_cfg = (int(ref["n_layer"]), int(ref["n_embd"]), float(ref["mlp_ratio"]),
                    int(ref["vocab"]), int(ref["budget_bits_actual"]))
    replayed = (cfg.n_layer, cfg.n_embd, cfg.mlp_ratio, cfg.vocab_size, found.actual_bits)
    if recorded_cfg != replayed:
        raise CheckpointProvenanceError(f"{run.run_id}: search replays to {replayed}, "
                                        f"{ref['run_id']} recorded {recorded_cfg}")
    assert_within_budget(cfg, run.bits, run.budget_bits)
    params = count_params(cfg)
    if params != run.params or params != int(ref["params_counted"]):
        raise CheckpointProvenanceError(f"{run.run_id}: {params} params, planned {run.params}")
    max_tokens = planned_max_tokens(run.arm, params)
    if max_tokens != run.max_tokens:
        raise CheckpointProvenanceError(f"{run.run_id}: budget {max_tokens}, planned "
                                        f"{run.max_tokens}")
    schedule = TR.make_batch_schedule(FIXED["training"]["effective_batch_tokens"],
                                      int(ref["micro_batch"]), cfg.block_size)
    if (schedule.grad_accum, schedule.effective_batch_tokens) != \
            (int(ref["grad_accum"]), int(ref["effective_batch_tokens"])):
        raise CheckpointProvenanceError(f"{run.run_id}: batch schedule differs from "
                                        f"{ref['run_id']}")
    max_steps = max(1, max_tokens // schedule.effective_batch_tokens)   # as train_one
    if run.arm != "m3" and max_steps != int(ref["steps"]):
        raise CheckpointProvenanceError(f"{run.run_id}: {max_steps} steps, {ref['run_id']} "
                                        f"ran {ref['steps']}")
    rule = learning_rate(cfg.n_embd)
    if not math.isclose(rule, float(ref["lr"]), rel_tol=1e-12):
        raise CheckpointProvenanceError(f"{run.run_id}: lr rule {rule}, {ref['run_id']} "
                                        f"recorded {ref['lr']}")
    base_lr = rule * run.lr_multiplier                                  # as train_one
    return A7.Setup(run.run_id, cfg, run.bits, run.budget_bits, run.seed, schedule,
                    max_steps, base_lr)


def checkpoint_matches(setup: "A7.Setup", ckpt_dir: Path) -> bool:
    """Whether a checkpoint exists to resume; raises if it is not this run's. train_one
    resumes whatever file carries the run id, without checking the learning rate."""
    path = ckpt_dir / f"{setup.run_id}.pt"
    if not path.exists():
        return False
    ck = torch.load(path, map_location="cpu", weights_only=True)
    stored = ck["result"]
    if (stored.get("run_id"), stored.get("seed"), stored.get("lr")) != \
            (setup.run_id, setup.seed, setup.base_lr) or not 0 < ck["step"] <= setup.max_steps:
        raise CheckpointProvenanceError(
            f"{path}: run {stored.get('run_id')} seed {stored.get('seed')} lr "
            f"{stored.get('lr')} step {ck['step']}; this run is seed {setup.seed} lr "
            f"{setup.base_lr} over {setup.max_steps} steps")
    return True


def train_run(run: PlannedRun, setup: "A7.Setup", ckpt_dir: Path, tokens, device,
              deadline: float | None) -> TR.RunResult:
    """``train_one``, exactly as ``sweep.run_cell`` calls it, but checkpointing to the A7
    directory and with the run's multiplier and token budget. Resumes from its checkpoint;
    raises GracefulStop at the deadline (checkpoint written)."""
    if A7._ACTIVE_GUARD is not None:          # torch.save is C-level: check the target
        A7._ACTIVE_GUARD.check_write(ckpt_dir / f"{run.run_id}.pt")
        A7._ACTIVE_GUARD.check_write(ckpt_dir / f"{run.run_id}.tmp")
    result = TR.train_one(
        setup.cfg, run.bits, seed=run.seed, budget_bits=run.budget_bits,
        max_tokens=run.max_tokens, run_id=run.run_id, ckpt_dir=ckpt_dir, device=device,
        train_tokens=tokens[0], val_tokens=tokens[1],
        micro_batch=setup.schedule.micro_batch, deadline=deadline,
        lr_multiplier=run.lr_multiplier,
    )
    got = (result.status, result.steps, result.undertrained, result.lr, result.tokens_seen)
    want = ("complete", setup.max_steps, False, setup.base_lr,
            setup.max_steps * setup.schedule.effective_batch_tokens)
    if got != want:
        raise AssertionError(f"{run.run_id}: train_one returned {got}, planned {want}")
    return result


@contextlib.contextmanager
def _replaying(setup: "A7.Setup"):
    """``a7_average.process_run`` rebuilds a run from its row with ``reconstruct``, which
    replays the SWEEP's recipe and therefore refuses every A7 variant (another learning rate,
    half the tokens, a suffixed id). Inside this block ``reconstruct`` returns ``setup`` for
    that one run id and raises for any other. a7_average.py itself is not modified: it is
    the timestamped M1 code, still running on the host."""
    original = A7.reconstruct

    def replay(row: dict):
        if row["run_id"] != setup.run_id:
            raise CheckpointProvenanceError(f"replaying {setup.run_id}, asked for "
                                            f"{row['run_id']}")
        return setup

    A7.reconstruct = replay
    try:
        yield
    finally:
        A7.reconstruct = original


def continuation(run: PlannedRun, setup: "A7.Setup", result: TR.RunResult, ckpt_dir: Path,
                 averaged_parent: Path, tokens, device, k: int) -> dict:
    """M1's continuation of the run's final checkpoint (``a7_average.process_run``):
    val_ckpt check, (a), (b), (c) = val_c, (d). Continuation 1 runs at this run's own final
    learning rate, ``lr * multiplier * min_lr_frac``."""
    row = {"run_id": run.run_id, "sweep": f"a7-{run.arm}", "budget_name": run.budget_name,
           "val_loss_best": result.val_loss_best, "best_step": result.best_step}
    with _replaying(setup):
        return A7.process_run(row, ckpt_dir, averaged_parent, *tokens, device, k)


def run_fields(run: PlannedRun, setup: "A7.Setup", device) -> dict:
    return {
        "run_id": run.run_id, "arm": run.arm, "paired_run_id": run.paired_run_id,
        "reference_run_id": run.reference_run_id, "budget_name": run.budget_name,
        "budget_bits": run.budget_bits, "bits": run.bits, "seed": run.seed,
        "n_layer": setup.cfg.n_layer, "n_embd": setup.cfg.n_embd,
        "mlp_ratio": setup.cfg.mlp_ratio, "params": run.params,
        "lr_multiplier": run.lr_multiplier, "lr": setup.base_lr,
        "token_factor": run.token_factor, "max_tokens": run.max_tokens,
        "micro_batch": setup.schedule.micro_batch, "grad_accum": setup.schedule.grad_accum,
        "device": device.type, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "", "git_sha": git_sha(),
    }


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

quiet_hours_block = A7.quiet_hours_block


def _refuse_overlap(main_dir: Path, out_dir: Path) -> str | None:
    for a, what in ((REPO_ROOT, "this script's copy"), (out_dir, "--out-dir")):
        if A7._under(a, main_dir) or A7._under(main_dir, a):
            return f"{what} {a} and --main-dir {main_dir} must not contain each other"
    return None


def _take_lock(out_dir: Path, name: str):
    """An exclusive flock on <out-dir>/<name>, or None if another process holds it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / name, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def print_plan(plan: dict) -> None:
    for b in plan["budgets"]:
        print(f"{b['budget_name']}: b*_c = {b['argmin_bits']:g}; group "
              f"{[g['bits'] for g in b['group']]}; M2/M3 seeds {b['seeds_m2_m3']}", flush=True)
    for c in plan["m4"]["comparisons"]:
        print(f"M4 {c['budget_name']} b={c['best_bits']:g} vs b={c['other_bits']:g}: gap "
              f"{c['gap']:.4f}, sds {c['sd_best']:.4f}/{c['sd_other']:.4f}, "
              f"{'; '.join(c['triggers'])} -> {c['n_per_cell']} seeds per cell"
              f"{' (cap; unresolved expected)' if c['unresolved_expected_at_cap'] else ''}",
              flush=True)
    runs = plan_runs(plan)
    for arm in ("m2", "m3", "m4"):
        rs = [r for r in runs if r.arm == arm]
        print(f"{arm}: {len(rs)} runs, ~{sum(r.est_train_gpu_h for r in rs):.1f} GPU-h of "
              "training at the sweep's throughput", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main-dir", type=Path, required=True,
                    help="the main checkout: never written; its arm marker is read")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "a7")
    ap.add_argument("--k", type=int, default=A7.K_DEFAULT)
    ap.add_argument("--allow-cpu", action="store_true", help="tests only")
    ap.add_argument("--min-free-gib", type=float, default=1.5)
    ap.add_argument("--mismatches-reviewed", action="store_true",
                    help="a human has examined every setup-mismatch row; continue")
    ap.add_argument("--check-m1", action="store_true",
                    help="only report whether M1 is complete: exit 0 yes, 2 not yet, 1 failed")
    args = ap.parse_args(argv)

    if not CORPUS.is_default:
        print(f"A7 trains TinyStories runs only; unset ISOMEM_DATASET (is {CORPUS.name})")
        return EXIT_ERROR
    here = Path(isomem.__file__).resolve().parent
    if here != REPO_ROOT / "isomem":
        print(f"imported isomem from {here}, not from {REPO_ROOT}")
        return EXIT_ERROR
    assert_fixed_invariants()
    if args.k < 2 or args.k % A7.EVAL_EVERY:
        print(f"--k must be a multiple of {A7.EVAL_EVERY} (A7: evaluate every "
              f"{A7.EVAL_EVERY} steps)")
        return EXIT_ERROR
    main_dir = args.main_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    why = _refuse_overlap(main_dir, out_dir)
    if why:
        print(why)
        return EXIT_ERROR

    if args.check_m1:                     # read-only: no lock, no file written
        _, m1_rows = read_csv(out_dir / "m1.csv")
        state, msg = m1_state(RESULTS_CSV, m1_rows)
        print(msg, flush=True)
        return {"complete": EXIT_OK, "incomplete": EXIT_WAIT_M1, "failed": EXIT_ERROR}[state]

    lock = _take_lock(out_dir, LOCK_NAME)
    if lock is None:
        print(f"another invocation holds {out_dir / LOCK_NAME}; nothing done", flush=True)
        return EXIT_MORE
    try:
        # M1's own lock, held throughout: no M1 invocation appends to m1.csv while it is read.
        m1_lock = _take_lock(out_dir, A7.LOCK_NAME)
        if m1_lock is None:
            print(f"an M1 invocation holds {out_dir / A7.LOCK_NAME}; M1 is still running",
                  flush=True)
            return EXIT_WAIT_M1
        A7._activate(A7.AccessGuard(main_dir, main_dir / "runs", RESULTS_CSV))
        try:
            return _run(args, main_dir, out_dir)
        except PlanRefused as exc:
            print(f"REFUSED: {exc}", flush=True)
            return EXIT_ERROR
        finally:
            A7._activate(None)
            m1_lock.close()
    finally:
        lock.close()


def _run(args, main_dir: Path, out_dir: Path) -> int:
    m1_bytes, m1_rows = read_csv(out_dir / "m1.csv")
    state, msg = m1_state(RESULTS_CSV, m1_rows)
    print(msg, flush=True)
    if state != "complete":
        return EXIT_WAIT_M1 if state == "incomplete" else EXIT_ERROR

    plan, new = load_or_create_plan(out_dir, RESULTS_CSV, m1_bytes, m1_rows)
    print(f"plan {'written to' if new else 'read from'} {out_dir / PLAN_NAME} "
          f"(m1.csv sha256 {plan['m1_csv_sha256'][:12]}...)", flush=True)
    if new:
        print_plan(plan)
    rec = recorded(out_dir)
    mismatched = sorted(r for r, v in rec.items() if v["status"] == "setup-mismatch")
    if mismatched and not args.mismatches_reviewed:
        print(f"a setup mismatch is recorded for {mismatched}: the setup would be wrong. "
              "Nothing more runs until a human has examined it (then --mismatches-reviewed).",
              flush=True)
        return EXIT_ERROR
    todo, known = pending(out_dir, plan, m1_rows)
    print(f"{len(known)} follow-up runs known, {len(rec)} recorded, {len(todo)} to do",
          flush=True)
    if not todo:
        return outcome(out_dir, plan, m1_rows)

    marker = main_dir / A7.ARM_MARKER
    if not marker.exists():
        print(f"the FineWeb-Edu arm is not finalised ({marker} missing); no A7 follow-up run "
              "starts before it", flush=True)
        return EXIT_MORE
    device = A7.select_device(args.allow_cpu)
    if device is None:
        print("no CUDA device: every run A7 compares with was trained on CUDA "
              "(--allow-cpu: tests)")
        return EXIT_ERROR
    devices = {r["device"] for r in m1_rows.values()} | \
        {r["device"] for r in rec.values() if r.get("device")}
    if devices - {device.type}:
        print(f"m1.csv / the follow-up files hold rows from {devices}; refusing to add "
              f"{device.type} rows")
        return EXIT_ERROR
    primary = {r["run_id"]: r for r in primary_rows(RESULTS_CSV)}
    for run in todo:
        A7.check_software(primary[run.reference_run_id], device)
    print(f"device {device}   git {git_sha()}   K {args.k}", flush=True)

    ckpt_dir = out_dir / CKPT_SUBDIR
    averaged_parent = out_dir / AVERAGED_PARENT
    tokens = None
    while True:
        todo, _ = pending(out_dir, plan, m1_rows)
        if not todo:
            break
        run = todo[0]
        why = quiet_hours_block()
        if why:
            print(f"{why}; stopping, {len(todo)} run(s) left", flush=True)
            return EXIT_MORE
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if free < args.min_free_gib * 2**30:
                print(f"{free / 2**30:.2f} GiB free on the GPU, below --min-free-gib "
                      f"{args.min_free_gib}; stopping", flush=True)
                return EXIT_MORE
        ref = primary[run.reference_run_id]
        A7.check_software(ref, device)
        setup = followup_setup(run, ref)
        resuming = checkpoint_matches(setup, ckpt_dir)
        if tokens is None:
            tokens = D.load_tokens("train"), D.load_tokens("val")
        print(f"[{run.arm}] {run.run_id}  lr x{run.lr_multiplier:g}  {run.max_tokens:,} "
              f"tokens  ~{run.est_train_gpu_h:.2f} h{'  (resuming)' if resuming else ''}",
              flush=True)
        try:
            result = train_run(run, setup, ckpt_dir, tokens, device, next_deadline(None, None))
        except TR.GracefulStop as exc:
            print(f"    {exc}; resumes next tick", flush=True)
            return EXIT_MORE
        except torch.cuda.OutOfMemoryError as exc:
            A7._release(device)
            print(f"    OOM, nothing recorded; resumes from its checkpoint next tick: "
                  f"{str(exc).splitlines()[0]}", flush=True)
            return EXIT_MORE
        except (TR.NanLoss, TernaryCollapse) as exc:
            # A failed run fails alone, as in the sweep; recorded, never retried.
            A7._release(device)
            status = "nan" if isinstance(exc, TR.NanLoss) else "collapsed"
            append_row(out_dir / ARM_CSV[run.arm], {**run_fields(run, setup, device),
                                                    "status": status, "k": args.k,
                                                    "error": str(exc)[:2000]})
            print(f"    {status.upper()} in training: {str(exc).splitlines()[0]}", flush=True)
            continue

        why = quiet_hours_block()
        if why:
            print(f"    trained; {why}, so its continuation waits for the next tick "
                  "(train_one returns at once from the final checkpoint)", flush=True)
            return EXIT_MORE
        try:
            out = continuation(run, setup, result, ckpt_dir, averaged_parent, tokens, device,
                               args.k)
        except torch.cuda.OutOfMemoryError as exc:
            A7._release(device)
            print(f"    OOM in the continuation, nothing recorded: "
                  f"{str(exc).splitlines()[0]}", flush=True)
            return EXIT_MORE
        except (TR.NanLoss, TernaryCollapse) as exc:
            status = "nan" if isinstance(exc, TR.NanLoss) else "collapsed"
            out = {"status": status, "k": args.k, "error": str(exc)[:2000]}
            print(f"    {status.upper()} in the continuation: {exc}", flush=True)
        A7._release(device)
        row = {**out, **run_fields(run, setup, device), "tokens_seen": result.tokens_seen,
               "steps": result.steps, "best_step": result.best_step,
               "val_loss_best": result.val_loss_best}
        append_row(out_dir / ARM_CSV[run.arm], row)
        if row["status"] == "setup-mismatch":
            print("    setup mismatch recorded; stopping so that a human looks before any "
                  "other run is written", flush=True)
            return EXIT_ERROR
        if row["status"] == "complete":
            print(f"    val_loss_best {row['val_loss_best']:.4f}  val_ckpt "
                  f"{row['val_ckpt']:.4f}  val_c {row['val_c']:.4f}  val_d {row['val_d']:.4f}"
                  f"  continuation {row['wall_s']:.0f}s", flush=True)
    return outcome(out_dir, plan, m1_rows)


if __name__ == "__main__":
    raise SystemExit(main())
