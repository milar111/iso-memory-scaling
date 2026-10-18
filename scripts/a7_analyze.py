#!/usr/bin/env python
"""A7 (M1-M4), A8 and the FineWeb-Edu arm (A6): the preregistered analyses, read-only.

    .venv/bin/python scripts/a7_analyze.py            # writes results/a7/ANALYSIS.txt and
                                                      # results/a7/analysis.json

PREREGISTRATION.md §4 (null criterion, Welch), §13 (A6), §16 (A7), §17, §18 and §19 (A8)
are the contract. This script computes exactly the analyses and falsifiers they state, from
the files the compute host produced, and changes none of them:

* ``results/results.csv`` (TinyStories primary + robustness) through ``isomem.analysis``;
* ``results/a7/m1.csv``, ``plan.json`` (frozen argmin groups and run plan),
  ``m2_extensions.json`` (frozen M2 extensions), ``m2.csv``, ``m3.csv`` and, once they
  exist, ``m4.csv`` and A8's ``m5_x8.json``, ``m5_x16.json`` (frozen x8 and x16 steps) and
  ``m5.csv``;
* ``results/fineweb_edu/{comparison.json, results.csv, residual_check.csv,
  residual_check.txt}``: carried, not reanalysed beyond A6's own §4 tests.

The statistics are the repo's own: ``isomem.analysis.null_verdict``, ``pairwise_tests`` and
``welch`` on cells built by ``load_cells`` with the A7 metric in place of ``val_loss_best``;
the argmin groups, the run plan, the extension rule, A8's x8 and x16 rules and the power
simulation are ``scripts/a7_followup.py``'s, imported unchanged. The frozen plan is
re-derived from m1.csv and refused if it no longer matches (as the runner refuses it); A8's
frozen steps are re-decided from their rows and refused likewise.

Every verdict carries the numbers it rests on. Anything that depends on a run not yet recorded
is marked PROVISIONAL; M4 without ``m4.csv`` and A8 without ``m5_x8.json`` are PENDING. The
last line printed is ``COMPLETE`` or ``INCOMPLETE: <what is missing>``. Exit 0 in both
cases; exit 1, with the reason, when an input is malformed or a frozen record no longer
matches the files it was computed from.

Output is deterministic (no timestamps), so ``scripts/paper_numbers.py`` can recompute it and
refuse a stale ``analysis.json``.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import re
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from isomem.analysis import (  # noqa: E402
    _betainc_half,
    load_cells,
    load_rows,
    null_verdict,
    pairwise_tests,
    welch,
)


def _load_followup():
    """scripts/a7_followup.py, the M2-M4 runner, imported unchanged (once per process)."""
    name = "a7_followup"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod       # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


F = _load_followup()

TERNARY, BF16 = 1.58, 16.0
SWEEPS = ("primary", "robustness")
# §16 "generic, not specific to low precision": gain(b=16) >= GENERIC_FRACTION * gain(b=1.58)
GENERIC_FRACTION = 0.5
# §16 M4 "unresolved" threshold and §18.7 power definition, from the runner itself
M4_P = F.M4_TRIGGER_P
M4_POWER = F.M4_POWER
FW_SWEEP = "fineweb_edu"
LN2 = math.log(2)

M1_METRICS = ("val_loss_best", "A_mean", "val_b", "val_c", "val_d", "val_fresh", "val_ckpt")
M1_REQUIRED = ("run_id", "sweep", "budget_name", "bits", "seed", "status", *M1_METRICS)
FOLLOWUP_REQUIRED = ("run_id", "arm", "status", "paired_run_id", "budget_name", "bits", "seed",
                     "lr_multiplier", "token_factor", "val_c")
FW_FILES = ("comparison.json", "results.csv", "residual_check.csv", "residual_check.txt")


class Malformed(RuntimeError):
    """An input file is unreadable, inconsistent, or no longer matches a frozen record."""


@dataclass(frozen=True)
class Paths:
    results_csv: Path = REPO / "results" / "results.csv"
    a7_dir: Path = REPO / "results" / "a7"
    fineweb_dir: Path = REPO / "results" / "fineweb_edu"

    def a7(self, name: str) -> Path:
        return self.a7_dir / name


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mean(v: list[float]) -> float:
    return st.mean(v)


def _sd(v: list[float]) -> float | None:
    return st.stdev(v) if len(v) > 1 else None


def _ms(v: list[float]) -> dict:
    return {"n": len(v), "mean": _mean(v), "sd": _sd(v), "values": list(v)}


def _num(x, file: str, what: str) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise Malformed(f"{file}: {what} is not a number: {x!r}") from None
    if not math.isfinite(f):
        raise Malformed(f"{file}: {what} is not finite: {x!r}")
    return f


def _clean(obj):
    """JSON-safe: NaN/inf -> None, tuples -> lists, float keys -> '%g' strings."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {(f"{k:g}" if isinstance(k, float) else str(k)): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def fmt(mean: float | None, sd: float | None = None, d: int = 4) -> str:
    if mean is None:
        return "-"
    s = f"{mean:.{d}f}"
    return s + (f" ± {sd:.{d}f}" if sd is not None else " (n=1)")


def fp(p: float | None) -> str:
    return "-" if p is None else (f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}")


def _read_csv(path: Path, required: tuple[str, ...]) -> tuple[bytes, dict[str, dict]]:
    """Bytes and rows by run_id (a7_followup.read_csv: last row wins), with the header
    checked. A missing file is (b"", {})."""
    if not path.exists():
        return b"", {}
    try:
        data, rows = F.read_csv(path)
        header = next(csv.reader(io.StringIO(data.decode(), newline="")), [])
    except Exception as e:  # noqa: BLE001 -- any parse failure is a malformed file
        raise Malformed(f"{path}: cannot be parsed as CSV ({type(e).__name__}: {e})") from None
    missing = [c for c in required if c not in header]
    if missing:
        raise Malformed(f"{path}: missing columns {missing}")
    for rid, r in rows.items():
        if None in r or any(v is None for v in r.values()):
            raise Malformed(f"{path}: row {rid!r} has the wrong number of fields")
    return data, rows


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise Malformed(f"{path}: cannot be read as JSON ({e})") from None


# --------------------------------------------------------------------------------------------
# Pure verdict functions (each a §16 / §4 rule, on numbers; tested on hand-made inputs)
# --------------------------------------------------------------------------------------------


def generic_vs_specific(gain: dict[float, float]) -> dict | None:
    """§16 M1 falsifier: the effect is generic, not specific to low precision, at a budget if
    ``gain(b=16) >= 0.5 * gain(b=1.58)`` (cell means). None where b=16 or b=1.58 is absent."""
    if BF16 not in gain or TERNARY not in gain:
        return None
    g16, g158 = gain[BF16], gain[TERNARY]
    threshold = GENERIC_FRACTION * g158
    return {"gain_b16": g16, "gain_b158": g158, "threshold": threshold,
            "ratio": g16 / g158 if g158 != 0 else None, "generic": g16 >= threshold}


def adjacent_reversals(before: dict[float, float], after: dict[float, float],
                       sd: dict[float, float]) -> list[dict]:
    """Every pair of bit-widths adjacent among ``after``'s keys: the difference
    ``m(b_hi) - m(b_lo)`` before and after, and whether it reversed sign by more than the
    larger of the pair's sds (``sd``, on the ``after`` metric)."""
    bits = sorted(after)
    out = []
    for lo, hi in zip(bits, bits[1:]):
        d0, d1 = before[hi] - before[lo], after[hi] - after[lo]
        s = max(sd[lo], sd[hi])
        flipped = d0 * d1 < 0
        out.append({"lo": lo, "hi": hi, "d_before": d0, "d_after": d1, "sd_max": s,
                    "reversed": flipped, "reversed_beyond_sd": flipped and abs(d1) > s})
    return out


def robust_to_averaging(best: dict[float, tuple[float, float]],
                        c: dict[float, tuple[float, float]]) -> dict:
    """§16 M1 falsifier: the primary argmin is robust to averaging at a budget if the val_c
    argmin equals the val_loss_best argmin and no adjacent gap reverses by more than the
    larger seed sd of the pair. ``best``/``c``: {bits: (mean, sd)}; sds of val_c judge."""
    a_best = min(best, key=lambda b: best[b][0])
    a_c = min(c, key=lambda b: c[b][0])
    rev = adjacent_reversals({b: v[0] for b, v in best.items()}, {b: v[0] for b, v in c.items()},
                             {b: v[1] for b, v in c.items()})
    beyond = [r for r in rev if r["reversed_beyond_sd"]]
    return {"argmin_val_loss_best": a_best, "argmin_val_c": a_c, "same_argmin": a_best == a_c,
            "pairs": rev, "reversals_beyond_sd": beyond,
            "robust": a_best == a_c and not beyond}


def best_multiplier(mults: dict[float, dict]) -> dict:
    """§16 M2 / §18.3 for one cell. ``mults``: {multiplier: {"status": "complete" | "missing" |
    "failed", "mean": ...}}. The best is the lowest mean among complete multipliers (a failed
    one cannot be best; a tie goes to 1). It is at the edge of the range when it is the lowest
    or highest complete multiplier tested: then the optimum is not bracketed. Provisional
    while any multiplier is missing."""
    done = {m: v for m, v in mults.items() if v["status"] == "complete"}
    if not done:
        raise ValueError("no complete multiplier")
    best = min(done, key=lambda m: (done[m]["mean"], m != 1.0))
    lo, hi = min(done), max(done)
    edge = "low" if best == lo else "high" if best == hi else None
    missing = sorted(m for m, v in mults.items() if v["status"] == "missing")
    failed = sorted(m for m, v in mults.items() if v["status"] == "failed")
    return {"best": best, "best_mean": done[best]["mean"], "edge": edge,
            "bracketed": edge is None, "tested": sorted(done), "missing": missing,
            "failed": failed, "provisional": bool(missing)}


def lr_falsifier(argmin_plan: float, x1: dict[float, float], tuned: dict[float, float],
                 sd: dict[float, float]) -> dict:
    """§16 M2 falsifier: with every cell at its own best multiplier, the argmin changes or an
    adjacent gap reverses by more than the larger seed sd of the pair. ``x1``: the group
    cells' mean val_c at the rule's own learning rate over the same seeds; ``tuned``: at each
    cell's best multiplier; ``sd``: the sd judged against."""
    a_x1 = min(x1, key=x1.get)
    a_tuned = min(tuned, key=tuned.get)
    rev = adjacent_reversals(x1, tuned, sd)
    beyond = [r for r in rev if r["reversed_beyond_sd"]]
    changes = a_tuned != argmin_plan
    return {"argmin_plan": argmin_plan, "argmin_x1_same_seeds": a_x1, "argmin_tuned": a_tuned,
            "argmin_changes": changes, "pairs": rev, "reversals_beyond_sd": beyond,
            "depends_on_lr": changes or bool(beyond)}


def length_sensitivity(gain_means: dict[float, float], nn_gap: float) -> dict:
    """§16 M3: at a budget, if the spread of gains across the group exceeds the
    nearest-neighbour gap of b*_c, the ranking there is sensitive to training length."""
    spread = max(gain_means.values()) - min(gain_means.values())
    return {"spread": spread, "nn_gap": nn_gap, "sensitive": spread > nn_gap}


# --------------------------------------------------------------------------------------------
# M4: Welch at the achieved n, and the minimum detectable effect by the §18.7 simulation
# --------------------------------------------------------------------------------------------


def _t_crit_table(df_lo: float, df_hi: float, alpha: float, points: int = 401):
    """|t| at which analysis.welch's p equals alpha, on a df grid (bisection on the same
    regularised incomplete beta the test uses)."""
    dfs = np.linspace(df_lo, df_hi, points)
    tc = []
    for df in dfs:
        lo, hi = 0.0, 1e3
        for _ in range(80):
            mid = (lo + hi) / 2
            p = _betainc_half(df / 2, df / (df + mid * mid))
            lo, hi = (mid, hi) if p > alpha else (lo, mid)
        tc.append(hi)
    return dfs, np.array(tc)


def minimum_detectable_effect(sd_a: float, sd_b: float, n_a: int, n_b: int,
                              z: np.ndarray | None = None, alpha: float = M4_P,
                              power: float = M4_POWER) -> dict:
    """The smallest true gap with simulated P(Welch p <= alpha) >= power at these n and sds.

    The simulation is §18.7's (a7_followup.standard_normals: seed 20261002, 20,000 draws, the
    first n values per cell). Welch's df does not depend on the gap, so each draw's critical
    |t| is read from a fine table of analysis.welch's own p; the power at the result is then
    recomputed draw by draw with analysis.welch and reported beside it."""
    z = F.standard_normals() if z is None else z
    a = sd_a * z[:, 0, :n_a]
    b = sd_b * z[:, 1, :n_b]
    va, vb = a.var(axis=1, ddof=1) / n_a, b.var(axis=1, ddof=1) / n_b
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    dfs, tcs = _t_crit_table(float(df.min()), float(df.max()), alpha)
    tc = np.interp(df, dfs, tcs)
    d0 = b.mean(axis=1) - a.mean(axis=1)

    def pw(g: float) -> float:
        return float(np.mean(np.abs(g + d0) / se >= tc))

    lo, hi = 0.0, max(sd_a, sd_b, 1e-12)
    while pw(hi) < power:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    al, bl = a.tolist(), (hi + b).tolist()
    exact = sum(welch(x, y)[2] <= alpha for x, y in zip(al, bl)) / len(al)
    return {"mde": hi, "alpha": alpha, "power": power, "n_a": n_a, "n_b": n_b,
            "sd_a": sd_a, "sd_b": sd_b, "power_at_mde_tabulated": pw(hi),
            "power_at_mde_exact_welch": exact, "draws": int(z.shape[0])}


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    results_rows: list[dict]
    m1: dict[str, dict]
    m1_bytes: bytes
    plan: dict
    ext: dict | None
    followup: dict[str, dict]       # every m2/m3/m4/m5 row by run_id
    m4_exists: bool
    sha: dict[str, str | None]
    x8: dict | None = None          # A8's frozen x8 step (m5_x8.json)
    x16: dict | None = None         # A8's frozen x16 step (m5_x16.json)


def load_inputs(p: Paths) -> Inputs:
    for need in (p.results_csv, p.a7("m1.csv"), p.a7("plan.json")):
        if not need.exists():
            raise Malformed(f"{need}: required input is missing")
    results_bytes = p.results_csv.read_bytes()
    rows = [r for r in load_rows(p.results_csv)
            if r["status"] == "complete" and r["sweep"] in SWEEPS]
    by_id = {r["run_id"]: r for r in rows}

    m1_bytes, m1 = _read_csv(p.a7("m1.csv"), M1_REQUIRED)
    f = "m1.csv"
    if set(m1) != set(by_id):
        raise Malformed(f"{f}: run ids differ from results.csv's complete TinyStories runs "
                        f"(missing {sorted(set(by_id) - set(m1))[:5]}, "
                        f"extra {sorted(set(m1) - set(by_id))[:5]})")
    for rid, r in m1.items():
        if r["status"] != "complete":
            raise Malformed(f"{f}: {rid} has status {r['status']!r}")
        bits = _num(r["bits"], f, f"{rid} bits")
        for k in M1_METRICS:
            if k == "val_b" and bits == BF16:
                if r[k] not in ("", "nan"):
                    raise Malformed(f"{f}: {rid} is b=16 but has val_b {r[k]!r}")
                continue
            _num(r[k], f, f"{rid} {k}")
        src = by_id[rid]
        if (r["sweep"], r["budget_name"], int(r["seed"])) != (
                src["sweep"], src["budget_name"], int(src["seed"])) or \
                bits != float(src["bits_per_weight"]):
            raise Malformed(f"{f}: {rid} sweep/budget/bits/seed differ from results.csv")
        if abs(float(r["val_loss_best"]) - float(src["val_loss_best"])) > 1e-9:
            raise Malformed(f"{f}: {rid} val_loss_best {r['val_loss_best']} differs from "
                            f"results.csv {src['val_loss_best']}")

    plan = _read_json(p.a7("plan.json"))
    if plan.get("version") != F.PLAN_VERSION:
        raise Malformed(f"plan.json: version {plan.get('version')}, expected {F.PLAN_VERSION}")
    if plan.get("m1_csv_sha256") != _sha256(m1_bytes):
        raise Malformed("plan.json was frozen from a different m1.csv (sha256 mismatch)")
    if plan.get("results_csv_sha256") != F.results_csv_frozen_sha256(results_bytes):
        raise Malformed("plan.json was frozen from a different results.csv (sha256 mismatch "
                        "with the §14 test-read columns blanked)")
    try:
        F._check_plan(plan)
    except (AssertionError, KeyError, TypeError) as e:
        raise Malformed(f"plan.json fails the runner's own invariants: {e}") from None
    _check_plan_reproduces(p, plan, m1)

    followup: dict[str, dict] = {}
    shas: dict[str, str | None] = {
        "results.csv": _sha256(results_bytes), "m1.csv": _sha256(m1_bytes),
        "plan.json": _sha256(p.a7("plan.json").read_bytes())}
    for name in ("m2.csv", "m3.csv", "m4.csv", "m5.csv"):
        data, rs = _read_csv(p.a7(name), FOLLOWUP_REQUIRED)
        shas[name] = _sha256(data) if p.a7(name).exists() else None
        clash = set(rs) & set(followup)
        if clash:
            raise Malformed(f"{name}: run ids recorded in two A7 files: {sorted(clash)}")
        for rid, r in rs.items():
            if r["status"] == "complete":
                _num(r["val_c"], name, f"{rid} val_c")
            want = {"m2.csv": ("m2", "m2-ext"), "m3.csv": ("m3",), "m4.csv": ("m4",),
                    "m5.csv": ("m5",)}[name]
            if r["arm"] not in want:
                raise Malformed(f"{name}: {rid} has arm {r['arm']!r}")
        followup.update(rs)

    ext = None
    if p.a7("m2_extensions.json").exists():
        ext = _read_json(p.a7("m2_extensions.json"))
        shas["m2_extensions.json"] = _sha256(p.a7("m2_extensions.json").read_bytes())
        m2_ids = [r.run_id for r in F.plan_runs(plan) if r.arm == "m2"]
        if any(i not in followup for i in m2_ids) or \
                ext.get("m2_rows_sha256") != F.rows_digest([followup[i] for i in m2_ids]):
            raise Malformed("m2_extensions.json was decided from different M2 rows "
                            "(m2_rows_sha256 mismatch)")
        # the decision itself, recomputed with the runner's rule, must equal the frozen one
        redo = F.decide_extensions(plan, followup, m1)
        if [(_cell_key(d), d["best_multiplier"], d["extend_to"]) for d in redo["cells"]] != \
                [(_cell_key(d), d["best_multiplier"], d["extend_to"]) for d in ext["cells"]] \
                or [r["run_id"] for r in redo["runs"]] != [r["run_id"] for r in ext["runs"]]:
            raise Malformed("m2_extensions.json does not reproduce from the M2 rows")
    else:
        shas["m2_extensions.json"] = None
    x8, x16 = _load_a8(p, plan, ext, followup, m1, shas)

    planned = {r.run_id: r for r in _planned_runs(plan, ext, x8, x16)}
    for rid, r in followup.items():
        pr = planned.get(rid)
        if pr is None:
            raise Malformed(f"{rid}: recorded in an A7 file but not in the frozen plan")
        got = (r["arm"], r["budget_name"], float(r["bits"]), int(r["seed"]),
               float(r["lr_multiplier"]), float(r["token_factor"]))
        want = (pr.arm, pr.budget_name, pr.bits, pr.seed, pr.lr_multiplier, pr.token_factor)
        if got != want:
            raise Malformed(f"{rid}: recorded fields {got} differ from the plan's {want}")
        if pr.paired_run_id and r["paired_run_id"] != pr.paired_run_id:
            raise Malformed(f"{rid}: paired with {r['paired_run_id']}, plan {pr.paired_run_id}")

    return Inputs(rows, m1, m1_bytes, plan, ext, followup, p.a7("m4.csv").exists(), shas,
                  x8, x16)


def _load_a8(p: Paths, plan: dict, ext: dict | None, followup: dict[str, dict],
             m1: dict[str, dict], shas: dict) -> tuple[dict | None, dict | None]:
    """A8's frozen x8 and x16 decisions (§19), each checked against the rows it was decided
    from and re-decided with the runner's own rule; None where not yet written."""
    out = []
    for name in (F.M5_X8_NAME, F.M5_X16_NAME):
        path = p.a7(name)
        shas[name] = _sha256(path.read_bytes()) if path.exists() else None
        out.append(_read_json(path) if path.exists() else None)
    x8, x16 = out
    if x8 is not None:
        if ext is None:
            raise Malformed(f"{F.M5_X8_NAME} exists but m2_extensions.json does not")
        ids = [r.run_id for r in F.m2_runs_with_extensions(plan, ext)]
        if any(i not in followup for i in ids) or \
                x8.get("m2_rows_sha256") != F.rows_digest([followup[i] for i in ids]):
            raise Malformed(f"{F.M5_X8_NAME} was decided from different M2 rows "
                            "(m2_rows_sha256 mismatch)")
        redo = F.decide_m5_x8(plan, ext, followup, m1)
        if [(_cell_key(d), d["best_multiplier"], d["step_to"]) for d in redo["cells"]] != \
                [(_cell_key(d), d["best_multiplier"], d["step_to"]) for d in x8["cells"]] \
                or [r["run_id"] for r in redo["runs"]] != [r["run_id"] for r in x8["runs"]]:
            raise Malformed(f"{F.M5_X8_NAME} does not reproduce from the M2 rows")
    if x16 is not None:
        if x8 is None:
            raise Malformed(f"{F.M5_X16_NAME} exists but {F.M5_X8_NAME} does not")
        ids = F._x16_inputs(ext, x8)
        if any(i not in followup for i in ids) or \
                x16.get("x4_x8_rows_sha256") != F.rows_digest([followup[i] for i in ids]):
            raise Malformed(f"{F.M5_X16_NAME} was decided from different x4/x8 rows "
                            "(x4_x8_rows_sha256 mismatch)")
        redo = F.decide_m5_x16(ext, x8, followup)
        if redo is None or \
                [(_cell_key(d), d["step_to"]) for d in redo["cells"]] != \
                [(_cell_key(d), d["step_to"]) for d in x16["cells"]] \
                or [r["run_id"] for r in redo["runs"]] != [r["run_id"] for r in x16["runs"]]:
            raise Malformed(f"{F.M5_X16_NAME} does not reproduce from the x4 and x8 rows")
    return x8, x16


def _cell_key(d: dict) -> tuple:
    return (d["budget_name"], float(d["bits"]))


def _planned_runs(plan: dict, ext: dict | None, x8: dict | None = None,
                  x16: dict | None = None) -> list:
    runs = F.plan_runs(plan)
    for d in (ext, x8, x16):
        if d is not None:
            runs += [F.PlannedRun(**r) for r in d["runs"]]
    return runs


def _check_plan_reproduces(p: Paths, plan: dict, m1: dict[str, dict]) -> None:
    """The frozen groups and seeds re-derived from m1.csv with the runner's own rule."""
    stats = F.cell_stats(F.primary_rows(p.results_csv), m1)
    for b in plan["budgets"]:
        g = F.argmin_group(stats[b["budget_name"]])
        if g.best.bits != b["argmin_bits"] or \
                [c.bits for c in g.members] != sorted(x["bits"] for x in b["group"]) or \
                list(g.seeds) != b["seeds_m2_m3"]:
            raise Malformed(f"plan.json: the {b['budget_name']} argmin group no longer "
                            "reproduces from m1.csv")


# --------------------------------------------------------------------------------------------
# M1
# --------------------------------------------------------------------------------------------


def _m1_values(inp: Inputs, sweep: str) -> dict[tuple[str, float], dict[str, list[float]]]:
    """Per (budget, bits): every M1 quantity, one value per seed in seed order."""
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in inp.m1.values():
        if r["sweep"] == sweep:
            groups.setdefault((r["budget_name"], float(r["bits"])), []).append(r)
    out = {}
    for key, rs in groups.items():
        rs.sort(key=lambda r: int(r["seed"]))
        v = {k: [float(r[k]) for r in rs] for k in M1_METRICS if k != "val_b"}
        v["val_b"] = [float(r["val_b"]) for r in rs] if key[1] != BF16 else []
        v["gain"] = [a - c for a, c in zip(v["A_mean"], v["val_c"])]
        v["sel_bias"] = [fr - ck for fr, ck in zip(v["val_fresh"], v["val_ckpt"])]
        v["seeds"] = [int(r["seed"]) for r in rs]
        out[key] = v
    return out


def _cells_with(p: Paths, sweep: str, vals, metric: str):
    """isomem.analysis cells of this sweep with ``metric`` in place of val_loss_best."""
    out = []
    for c in load_cells(p.results_csv, sweep=sweep):
        v = vals[(c.budget_name, c.bits)][metric]
        if not c.usable or c.n != len(v):
            raise Malformed(f"{sweep} {c.budget_name} b={c.bits:g}: not every seed is usable; "
                            "§4 would drop the cell")
        out.append(dataclasses.replace(c, val_losses=list(v)))
    return out


def _null_dict(nv) -> dict:
    return {"argmin": nv.best_bits, "best_mean": nv.best_mean, "gap": nv.gap, "s_max": nv.s_max,
            "is_null": nv.is_null, "adjacent": nv.adjacent, "note": nv.note}


def _welch_list(tests: list[dict]) -> list[dict]:
    return [{"other_bits": t["other_bits"], "gap": t["gap"], "sd_max": t["pooled_sd"],
             "p": t["p"], "significant": t["significant"], "detectable": t["detectable"]}
            for t in tests]


def analyse_m1(p: Paths, inp: Inputs) -> dict:
    out = {}
    for sweep in SWEEPS:
        vals = _m1_values(inp, sweep)
        cells_by = {m: _cells_with(p, sweep, vals, m)
                    for m in ("val_loss_best", "val_c", "A_mean")}
        budgets = []
        for c in cells_by["val_c"]:
            if c.budget_name not in budgets:
                budgets.append(c.budget_name)
        sw = {"budgets": {}}
        for name in budgets:
            keys = sorted(k for k in vals if k[0] == name)
            cells = []
            for k in keys:
                v = vals[k]
                cells.append({"bits": k[1], "seeds": v["seeds"],
                              **{m: _ms(v[m]) if v[m] else None
                                 for m in (*M1_METRICS, "gain", "sel_bias")}})
            by = {c["bits"]: c for c in cells}
            gen = generic_vs_specific({b: c["gain"]["mean"] for b, c in by.items()})
            if gen is not None:
                gen["sd_b16"], gen["sd_b158"] = by[BF16]["gain"]["sd"], by[TERNARY]["gain"]["sd"]
            rob = robust_to_averaging(
                {b: (c["val_loss_best"]["mean"], c["val_loss_best"]["sd"]) for b, c in by.items()},
                {b: (c["val_c"]["mean"], c["val_c"]["sd"]) for b, c in by.items()})
            nulls = {m: _null_dict(null_verdict(cells_by[m], name)) for m in cells_by}
            tests = {m: _welch_list(pairwise_tests(cells_by[m], name)) for m in cells_by}
            a_c = nulls["val_c"]["argmin"]
            a_b = nulls["val_loss_best"]["argmin"]
            tern = None
            if TERNARY in by:
                tern = {"val_c": by[TERNARY]["val_c"]["mean"] - by[a_c]["val_c"]["mean"],
                        "val_c_argmin": a_c,
                        "val_loss_best": by[TERNARY]["val_loss_best"]["mean"]
                        - by[a_b]["val_loss_best"]["mean"],
                        "val_loss_best_argmin": a_b}
            sw["budgets"][name] = {"cells": cells, "generic": gen, "robust": rob,
                                   "null": nulls, "welch": tests, "ternary_penalty": tern}
        out[sweep] = sw
    return out


# --------------------------------------------------------------------------------------------
# M2
# --------------------------------------------------------------------------------------------


def _rec_value(inp: Inputs, rid: str) -> tuple[str, float | None]:
    r = inp.followup.get(rid)
    if r is None:
        return "missing", None
    if r["status"] != "complete" or not math.isfinite(float(r["val_c"] or "nan")):
        return "failed", None
    return "complete", float(r["val_c"])


def _mult_entry(inp: Inputs, rs: list) -> dict:
    """One multiplier of one cell: its runs' val_c over the seeds, or why it has none
    ("failed" if any run did not complete, else "missing" if any is unrecorded)."""
    rs = sorted(rs, key=lambda r: r.seed)
    got = [_rec_value(inp, r.run_id) for r in rs]
    states = {g[0] for g in got}
    status = "failed" if "failed" in states else \
        "missing" if "missing" in states else "complete"
    v = [g[1] for g in got if g[0] == "complete"]
    return {"status": status, "run_ids": [r.run_id for r in rs],
            **(_ms(v) if v else {"n": 0, "mean": None, "sd": None, "values": []})}


def analyse_m2(inp: Inputs, m1: dict) -> dict:
    plan = inp.plan
    runs = [r for r in _planned_runs(plan, inp.ext) if r.arm in ("m2", "m2-ext")]
    missing = sorted(r.run_id for r in runs if r.run_id not in inp.followup)
    failed = sorted(r.run_id for r in runs if _rec_value(inp, r.run_id)[0] == "failed")
    n_planned = len(runs)
    ext_decided = inp.ext is not None
    m1c = {(b, c["bits"]): c for b, bd in m1["primary"]["budgets"].items() for c in bd["cells"]}
    budgets = {}
    for b in plan["budgets"]:
        name, seeds = b["budget_name"], b["seeds_m2_m3"]
        group = sorted(g["bits"] for g in b["group"])
        cells = []
        for bits in group:
            here = [r for r in runs if r.budget_name == name and r.bits == bits]
            mults: dict[float, dict] = {}
            paired = {r.seed: r.paired_run_id for r in here}
            x1 = [float(inp.m1[paired[s]]["val_c"]) for s in seeds]
            mults[1.0] = {"status": "complete", **_ms(x1), "run_ids": [paired[s] for s in seeds]}
            for m in sorted({r.lr_multiplier for r in here}):
                mults[m] = _mult_entry(inp, [r for r in here if r.lr_multiplier == m])
            bm = best_multiplier(mults)
            bm["provisional"] = bm["provisional"] or not ext_decided
            best_entry = mults[bm["best"]]
            sd_tuned, sd_src = best_entry["sd"], "seed sd at the best multiplier"
            if sd_tuned is None:
                sd_tuned = m1c[(name, bits)]["val_c"]["sd"]
                sd_src = "M1 five-seed val_c sd (one M2 seed: no sd at the best multiplier)"
            cells.append({"bits": bits, "seeds": seeds, "multipliers": mults, **bm,
                          "best_sd": best_entry["sd"], "sd_judged": sd_tuned,
                          "sd_source": sd_src})
        fal = lr_falsifier(b["argmin_bits"],
                           {c["bits"]: c["multipliers"][1.0]["mean"] for c in cells},
                           {c["bits"]: c["best_mean"] for c in cells},
                           {c["bits"]: c["sd_judged"] for c in cells})
        prov = any(c["provisional"] for c in cells)
        unbracketed = [c["bits"] for c in cells if not c["bracketed"]]
        budgets[name] = {"seeds": seeds, "cells": cells, "falsifier": fal, "provisional": prov,
                         "unbracketed": unbracketed}
    return {"n_planned": n_planned, "n_recorded": n_planned - len(missing),
            "missing_runs": missing, "failed_runs": failed, "extensions_decided": ext_decided,
            "complete": not missing and ext_decided, "budgets": budgets}


# --------------------------------------------------------------------------------------------
# M3
# --------------------------------------------------------------------------------------------


def analyse_m3(inp: Inputs, m1: dict) -> dict:
    plan = inp.plan
    runs = [r for r in F.plan_runs(plan) if r.arm == "m3"]
    missing = sorted(r.run_id for r in runs if r.run_id not in inp.followup)
    failed = sorted(r.run_id for r in runs if _rec_value(inp, r.run_id)[0] == "failed")
    budgets = {}
    for b in plan["budgets"]:
        name = b["budget_name"]
        nv = m1["primary"]["budgets"][name]["null"]["val_c"]
        cells, means = [], {}
        for bits in sorted(g["bits"] for g in b["group"]):
            rs = sorted((r for r in runs if r.budget_name == name and r.bits == bits),
                        key=lambda r: r.seed)
            pairs = []
            for r in rs:
                state, half = _rec_value(inp, r.run_id)
                full = float(inp.m1[r.paired_run_id]["val_c"])
                pairs.append({"seed": r.seed, "run_id": r.run_id, "status": state,
                              "val_c_half": half, "val_c_full": full,
                              "gain": half - full if half is not None else None})
            gains = [x["gain"] for x in pairs if x["gain"] is not None]
            ok = len(gains) == len(pairs)
            cell = {"bits": bits, "pairs": pairs, "complete": ok,
                    **(_ms(gains) if gains else {"n": 0, "mean": None, "sd": None})}
            if gains:
                means[bits] = cell["mean"]
            cells.append(cell)
        prov = any(not c["complete"] for c in cells)
        # the gap to the closest group member, reported beside the preregistered one
        mc = {c["bits"]: c["val_c"]["mean"] for c in m1["primary"]["budgets"][name]["cells"]}
        others = [x for x in mc if x != nv["argmin"] and x in {g["bits"] for g in b["group"]}]
        closest = min(others, key=lambda x: abs(mc[x] - mc[nv["argmin"]]))
        res = length_sensitivity(means, nv["gap"]) if len(means) >= 2 else None
        budgets[name] = {
            "argmin": nv["argmin"], "nn_gap": nv["gap"], "nn_adjacent": nv["adjacent"],
            "closest_member": closest, "closest_member_gap": abs(mc[closest] - mc[nv["argmin"]]),
            "cells": cells, "provisional": prov, **(res or {"spread": None, "sensitive": None}),
            "sensitive_vs_closest_member": (res["spread"] > abs(mc[closest] - mc[nv["argmin"]]))
            if res else None}
    return {"n_planned": len(runs), "n_recorded": len(runs) - len(missing),
            "missing_runs": missing, "failed_runs": failed,
            "complete": not missing and not failed, "budgets": budgets}


# --------------------------------------------------------------------------------------------
# M4
# --------------------------------------------------------------------------------------------


def analyse_m4(p: Paths, inp: Inputs, m1: dict, z: np.ndarray | None = None) -> dict:
    plan = inp.plan
    runs = [r for r in F.plan_runs(plan) if r.arm == "m4"]
    comps = plan["m4"]["comparisons"]
    base = {"n_planned": len(runs),
            "planned": [{"budget_name": c["budget_name"], "best_bits": c["best_bits"],
                         "other_bits": c["other_bits"], "triggers": c["triggers"],
                         "n_per_cell": c["n_per_cell"], "power_by_n": c["power_by_n"],
                         "unresolved_expected_at_cap": c["unresolved_expected_at_cap"],
                         "m1_gap": c["gap"], "m1_welch_p": c["welch_p"]} for c in comps]}
    if not inp.m4_exists:
        return {**base, "status": "pending", "complete": False, "n_recorded": 0,
                "missing_runs": sorted(r.run_id for r in runs), "failed_runs": [],
                "comparisons": []}
    missing = sorted(r.run_id for r in runs if r.run_id not in inp.followup)
    failed = sorted(r.run_id for r in runs if _rec_value(inp, r.run_id)[0] == "failed")

    m1v = _m1_values(inp, "primary")

    def values(name: str, bits: float) -> tuple[list[float], list[int]]:
        v = list(m1v[(name, bits)]["val_c"])
        s = list(m1v[(name, bits)]["seeds"])
        for r in sorted((r for r in runs if r.budget_name == name and r.bits == bits),
                        key=lambda r: r.seed):
            state, x = _rec_value(inp, r.run_id)
            if state == "complete":
                v.append(x)
                s.append(r.seed)
        return v, s

    z = F.standard_normals() if z is None else z
    out = []
    for c in comps:
        name, bb, ob = c["budget_name"], c["best_bits"], c["other_bits"]
        a, sa = values(name, bb)
        b, sb = values(name, ob)
        t, df, pval = welch(a, b)
        gap = st.mean(b) - st.mean(a)
        # §4 criterion at the budget, the two cells at their enlarged n, the others at 5
        cells = _cells_with(p, "primary", m1v, "val_c")
        ext = {bb: a, ob: b}
        cells = [dataclasses.replace(x, val_losses=ext[x.bits])
                 if x.budget_name == name and x.bits in ext else x for x in cells]
        nv = null_verdict(cells, name)
        mde = minimum_detectable_effect(st.stdev(a), st.stdev(b), len(a), len(b), z)
        resolved = pval <= M4_P
        out.append({"budget_name": name, "best_bits": bb, "other_bits": ob,
                    "n_best": len(a), "n_other": len(b), "seeds_best": sa, "seeds_other": sb,
                    "mean_best": st.mean(a), "sd_best": st.stdev(a),
                    "mean_other": st.mean(b), "sd_other": st.stdev(b), "gap": gap,
                    "welch_t": t, "welch_df": df, "welch_p": pval, "resolved": resolved,
                    "pair_gap_le_sd_max": abs(gap) <= max(st.stdev(a), st.stdev(b)),
                    "null_at_budget": _null_dict(nv), "mde": mde})
    status = "complete" if not missing and not failed else "provisional"
    return {**base, "status": status, "complete": status == "complete",
            "n_recorded": len(runs) - len(missing), "missing_runs": missing,
            "failed_runs": failed, "comparisons": out}


# --------------------------------------------------------------------------------------------
# A8 (§19): the learning rate bracketed upward, x8 then x16
# --------------------------------------------------------------------------------------------


def bracket_label(bm: dict) -> str:
    """§19: "a cell still best at x16 is reported as not bracketed at the cap"; otherwise
    best_multiplier's own reading (an edge of the complete multipliers is not bracketed)."""
    if bm["best"] == F.A8_CAP:
        return "not bracketed at the cap"
    return "bracketed" if bm["bracketed"] else f"not bracketed ({bm['edge']} edge)"


def analyse_a8(inp: Inputs, m1: dict, m2: dict) -> dict:
    """§19's analysis: every group cell at its best multiplier over everything it was run at
    (M2's x0.5-x4, or x0.25-x2, plus A8's x8 and x16), bracketed or not, and the §16 M2
    falsifier re-applied unchanged, beside M2's own. Pending until m5_x8.json exists; then
    provisional until every A8 run is recorded and the x16 step is decided."""
    plan, x8, x16 = inp.plan, inp.x8, inp.x16
    runs = [r for r in _planned_runs(plan, inp.ext, x8, x16) if r.arm == "m5"]
    missing = sorted(r.run_id for r in runs if r.run_id not in inp.followup)
    failed = sorted(r.run_id for r in runs if _rec_value(inp, r.run_id)[0] == "failed")

    def stepped(d: dict | None) -> list[tuple[str, float]]:
        return [] if d is None else [_cell_key(c) for c in d["cells"] if c["step_to"]]

    x8_cells, x16_cells = stepped(x8), stepped(x16)
    status = "pending" if x8 is None else \
        "complete" if x16 is not None and not missing else "provisional"
    m2_runs = [r for r in F.plan_runs(plan) if r.arm == "m2" and r.lr_multiplier == 2.0]
    m1c = {(b, c["bits"]): c for b, bd in m1["primary"]["budgets"].items() for c in bd["cells"]}
    budgets, candidates = {}, []
    for name, b2 in m2["budgets"].items():
        cells = []
        for c2 in b2["cells"]:
            bits = c2["bits"]
            mults = dict(c2["multipliers"])
            for m in F.A8_MULTIPLIERS:
                rs = [r for r in runs
                      if (r.budget_name, r.bits, r.lr_multiplier) == (name, bits, m)]
                if rs:
                    mults[m] = _mult_entry(inp, rs)
            bm = best_multiplier(mults)
            decided = x8 is not None and ((name, bits) not in x8_cells or x16 is not None)
            bm["provisional"] = bm["provisional"] or not decided
            best_entry = mults[bm["best"]]
            sd, src = best_entry["sd"], "seed sd at the best multiplier"
            if sd is None:          # as M2: one seed, so the M1 five-seed sd is judged
                sd = m1c[(name, bits)]["val_c"]["sd"]
                src = "M1 five-seed val_c sd (one seed: no sd at the best multiplier)"
            cells.append({"bits": bits, "seeds": c2["seeds"], "multipliers": mults, **bm,
                          "bracket": bracket_label(bm), "best_m2": c2["best"],
                          "x8": (name, bits) in x8_cells, "x16": (name, bits) in x16_cells,
                          "best_sd": best_entry["sd"], "sd_judged": sd, "sd_source": src})
            if x8 is None and c2["best"] == F.A8_FROM:
                here = [r for r in m2_runs if (r.budget_name, r.bits) == (name, bits)]
                candidates.append({"budget_name": name, "bits": bits, "seeds": c2["seeds"],
                                   "provisional": c2["provisional"],
                                   "est_train_gpu_h": sum(r.est_train_gpu_h for r in here)})
        f2 = b2["falsifier"]
        fal = lr_falsifier(f2["argmin_plan"],
                           {c["bits"]: c["multipliers"][1.0]["mean"] for c in cells},
                           {c["bits"]: c["best_mean"] for c in cells},
                           {c["bits"]: c["sd_judged"] for c in cells})
        budgets[name] = {
            "seeds": b2["seeds"], "cells": cells, "falsifier": fal,
            "m2_falsifier": {"argmin_tuned": f2["argmin_tuned"],
                             "depends_on_lr": f2["depends_on_lr"]},
            "provisional": any(c["provisional"] for c in cells),
            "unbracketed": [c["bits"] for c in cells if not c["bracketed"]],
            "at_cap": [c["bits"] for c in cells if c["best"] == F.A8_CAP]}
    return {"status": status, "complete": status == "complete",
            "x8_decided": x8 is not None, "x16_decided": x16 is not None,
            "n_planned": len(runs), "n_recorded": len(runs) - len(missing),
            "missing_runs": missing, "failed_runs": failed,
            "x8_cells": [{"budget_name": n, "bits": b} for n, b in x8_cells],
            "x16_cells": [{"budget_name": n, "bits": b} for n, b in x16_cells],
            "est_train_gpu_h_planned": sum(r.est_train_gpu_h for r in runs),
            "candidates": candidates, "budgets": budgets}


# --------------------------------------------------------------------------------------------
# FineWeb-Edu (A6): carried, with the §4 Welch tests A6 preregistered, and checked
# --------------------------------------------------------------------------------------------


def bpb(nats: float, tpb: float) -> float:
    """compare_corpora.bpb: loss x tokens_per_byte / ln 2."""
    return nats * tpb / LN2


def analyse_fineweb(p: Paths) -> dict:
    d = p.fineweb_dir
    absent = [n for n in FW_FILES if not (d / n).exists()]
    if absent:
        return {"complete": False, "missing_files": absent}
    comp = _read_json(d / "comparison.json")
    try:
        tpb = comp["tokens_per_byte"]
        floor = comp["bigram_floor_nats"]
        ts_name, fw_name = "tinystories", FW_SWEEP
        tpb_ts, tpb_fw = float(tpb[ts_name]), float(tpb[fw_name])
    except (KeyError, TypeError, ValueError) as e:
        raise Malformed(f"{d / 'comparison.json'}: missing key {e}") from None
    fw_cells = load_cells(d / "results.csv", sweep=FW_SWEEP)
    ts_cells = load_cells(p.results_csv, sweep="primary")
    corpora = {ts_name: ts_cells, fw_name: fw_cells}

    res_rows = list(csv.DictReader(io.StringIO((d / "residual_check.csv").read_text())))
    need = {"budget_name", "bits", "seed", "val_half", "val_full", "gain"}
    if not res_rows or not need <= set(res_rows[0]):
        raise Malformed(f"{d / 'residual_check.csv'}: missing columns {need - set(res_rows[0] if res_rows else {})}")
    txt = (d / "residual_check.txt").read_text()
    txt_verdicts = dict(re.findall(r"^(\S+): argmin.*?-> (not sensitive|sensitive) to training",
                                   txt, flags=re.M | re.S))

    budgets = {}
    for name, cb in comp["budgets"].items():
        rows = {k: sorted([c for c in v if c.budget_name == name and c.usable],
                          key=lambda c: c.bits) for k, v in corpora.items()}
        for k in (ts_name, fw_name):
            carried = {float(c["bits"]): c for c in cb["cells"][k]}
            if sorted(carried) != [c.bits for c in rows[k]]:
                raise Malformed(f"comparison.json {name} {k}: cells differ from results.csv")
            for c in rows[k]:
                x = carried[c.bits]
                if abs(x["mean"] - c.mean) > 1e-9 or abs(x["std"] - c.std) > 1e-9 or \
                        x["n"] != c.n:
                    raise Malformed(f"comparison.json {name} {k} b={c.bits:g} is stale "
                                    "against its results.csv")
        nv = {k: null_verdict(corpora[k], name) for k in corpora}
        for k in corpora:
            if nv[k].best_bits != cb["argmin"][k] or nv[k].is_null != cb["null"][k] or \
                    abs(nv[k].gap - cb["gap"][k]) > 1e-12:
                raise Malformed(f"comparison.json {name} {k}: argmin/null does not reproduce")
        tests = _welch_list(pairwise_tests(fw_cells, name))
        tests_ts = _welch_list(pairwise_tests(ts_cells, name))
        best = {k: min(rows[k], key=lambda c: c.mean) for k in rows}
        pen = {}
        for k, t in ((ts_name, tpb_ts), (fw_name, tpb_fw)):
            tern = next(c for c in rows[k] if c.bits == TERNARY)
            pen[k] = bpb(tern.mean - best[k].mean, t)
            if abs(pen[k] - cb["ternary_penalty_bpb"][k]) > 1e-12:
                raise Malformed(f"comparison.json {name}: ternary penalty does not reproduce")
        if nv[fw_name].is_null:
            reading = f"NULL on {fw_name}: no argmin is claimed at this budget"
        elif best[fw_name].bits == best[ts_name].bits:
            reading = f"same argmin (b={best[fw_name].bits:g}): the result holds on harder text"
        else:
            reading = (f"argmin MOVES from b={best[ts_name].bits:g} to b={best[fw_name].bits:g}: "
                       f"the optimum depends on the text")
        if reading != cb["reading"]:
            raise Malformed(f"comparison.json {name}: A6 reading does not reproduce")
        cells = []
        ts_by = {c.bits: c for c in rows[ts_name]}
        fw_by = {c.bits: c for c in rows[fw_name]}
        for bits in sorted(set(ts_by) | set(fw_by)):
            row = {"bits": bits}
            for k, by, t in ((ts_name, ts_by, tpb_ts), (fw_name, fw_by, tpb_fw)):
                c = by.get(bits)
                row[k] = None if c is None else {
                    "mean": c.mean, "sd": c.std, "n": c.n, "bpb": bpb(c.mean, t),
                    "sd_bpb": bpb(c.std, t), "beats_floor": c.mean < float(floor[k])}
            cells.append(row)
        # A6 residual check, recomputed from its CSV and checked against its own text
        rr = [r for r in res_rows if r["budget_name"] == name]
        gains = {float(r["bits"]): float(r["gain"]) for r in rr}
        for r in rr:
            if abs(float(r["val_half"]) - float(r["val_full"]) - float(r["gain"])) > 1e-9:
                raise Malformed(f"residual_check.csv {r['bits']}: gain != val_half - val_full")
        resid = length_sensitivity(gains, nv[fw_name].gap) if len(gains) >= 2 else None
        if resid is not None and name in txt_verdicts and \
                (txt_verdicts[name] == "sensitive") != resid["sensitive"]:
            raise Malformed(f"residual_check.txt {name}: verdict does not reproduce from the CSV")
        budgets[name] = {
            "cells": cells,
            "null": {k: _null_dict(nv[k]) for k in corpora},
            "welch_fineweb": tests, "welch_tinystories": tests_ts,
            "ternary_penalty_bpb": pen, "reading": reading,
            "same_argmin": best[fw_name].bits == best[ts_name].bits,
            "all_beat_floor": all(c.mean < float(floor[fw_name]) for c in rows[fw_name]),
            "residual_check": None if resid is None else {
                "gains": gains, "seed": sorted({int(r["seed"]) for r in rr}), **resid},
        }
    preds = comp["predictions"]
    p8 = {n: v["null"][fw_name]["argmin"] <= 6 for n, v in budgets.items()}
    p9 = {n: v["ternary_penalty_bpb"][fw_name] < v["ternary_penalty_bpb"][ts_name]
          for n, v in budgets.items()}
    p10 = {n: v["all_beat_floor"] for n, v in budgets.items()}
    for label, mine in (("P8", p8), ("P9", p9), ("P10", p10)):
        if mine != preds[label]:
            raise Malformed(f"comparison.json predictions.{label} does not reproduce")
    return {"complete": True, "tokens_per_byte": {ts_name: tpb_ts, fw_name: tpb_fw},
            "bigram_floor_nats": {k: float(v) for k, v in floor.items()},
            "bigram_floor_bpb": {ts_name: bpb(float(floor[ts_name]), tpb_ts),
                                 fw_name: bpb(float(floor[fw_name]), tpb_fw)},
            "budgets": budgets,
            "predictions": {k: {"per_budget": v, "holds": all(v.values())}
                            for k, v in (("P8", p8), ("P9", p9), ("P10", p10))}}


# --------------------------------------------------------------------------------------------
# Verdict lines (the same strings go to ANALYSIS.txt and analysis.json)
# --------------------------------------------------------------------------------------------


def _prov(flag: bool) -> str:
    return "PROVISIONAL " if flag else ""


def verdict_lines(res: dict) -> list[str]:
    L = []
    for sweep in SWEEPS:
        for name, b in res["m1"][sweep]["budgets"].items():
            tag = f"M1 {name} {sweep}"
            g = b["generic"]
            if g is None:
                L.append(f"{tag} generic-vs-specific: not applicable (no b=16 cell at this "
                         "budget)")
            else:
                L.append(f"{tag} generic-vs-specific: gain(b=16) {fmt(g['gain_b16'], g['sd_b16'])}"
                         f" vs 0.5 x gain(b=1.58) = 0.5 x {fmt(g['gain_b158'], g['sd_b158'])} = "
                         f"{g['threshold']:.4f} (ratio "
                         + ("undefined" if g["ratio"] is None else f"{g['ratio']:.3f}") + ") -> "
                         + ("GENERIC (not specific to low precision)" if g["generic"]
                            else "SPECIFIC to low precision"))
            r = b["robust"]
            by = {c["bits"]: c for c in b["cells"]}
            rv = "; ".join(
                f"b={x['lo']:g}->{x['hi']:g} {x['d_before']:+.4f} -> {x['d_after']:+.4f} "
                f"(larger val_c sd {x['sd_max']:.4f})" for x in r["reversals_beyond_sd"]) or "none"
            flips = sum(x["reversed"] for x in r["pairs"])
            L.append(f"{tag} robust-to-averaging: argmin val_loss_best b="
                     f"{r['argmin_val_loss_best']:g} "
                     f"({fmt(by[r['argmin_val_loss_best']]['val_loss_best']['mean'], by[r['argmin_val_loss_best']]['val_loss_best']['sd'])}),"
                     f" argmin val_c b={r['argmin_val_c']:g} "
                     f"({fmt(by[r['argmin_val_c']]['val_c']['mean'], by[r['argmin_val_c']]['val_c']['sd'])});"
                     f" adjacent sign flips {flips} of {len(r['pairs'])}, beyond the larger sd: {rv}"
                     " -> " + ("ROBUST to averaging" if r["robust"] else
                               "NOT ROBUST: the argmin at this budget depends on end-of-training "
                               "averaging"))
            for m in ("val_c", "A_mean"):
                n = b["null"][m]
                ws = ", ".join(f"vs b={w['other_bits']:g} gap {w['gap']:+.4f} p={fp(w['p'])}"
                               for w in b["welch"][m]
                               if w["other_bits"] in n["adjacent"])
                L.append(f"{tag} §4 on {m}: argmin b={n['argmin']:g} ({n['best_mean']:.4f}), "
                         f"gap to nearest adjacent {n['gap']:.4f}, s_max {n['s_max']:.4f} -> "
                         + ("NULL" if n["is_null"] else "not null") + f"; Welch adjacent: {ws}")
            t = b["ternary_penalty"]
            if t:
                L.append(f"{tag} ternary penalty on val_c: m_c(1.58) - m_c(b={t['val_c_argmin']:g})"
                         f" = {t['val_c']:.4f} (on val_loss_best: {t['val_loss_best']:.4f} vs "
                         f"b={t['val_loss_best_argmin']:g})")
            sb = [c["sel_bias"]["mean"] for c in b["cells"]]
            L.append(f"{tag} fixed-val-set selection bias val_fresh - val_ckpt: cell means from "
                     f"{min(sb):+.4f} to {max(sb):+.4f} (per cell in the table above)")
    m2 = res["m2"]
    for name, b in m2["budgets"].items():
        f = b["falsifier"]
        cells = "; ".join(
            f"b={c['bits']:g} best x{c['best']:g} {fmt(c['best_mean'], c['best_sd'])}"
            + ("" if c["bracketed"] else
               f" [NOT BRACKETED: best at the {c['edge']} edge of x{c['tested'][0]:g}-x{c['tested'][-1]:g}]")
            + (f" [missing x{', x'.join(f'{m:g}' for m in c['missing'])}]" if c["missing"] else "")
            for c in b["cells"])
        rv = "; ".join(f"b={x['lo']:g}->{x['hi']:g} {x['d_before']:+.4f} -> {x['d_after']:+.4f} "
                       f"(sd {x['sd_max']:.4f})" for x in f["reversals_beyond_sd"]) or "none"
        L.append(f"{_prov(b['provisional'])}M2 {name} (seeds {b['seeds']}): {cells}. Argmin "
                 f"b*_c={f['argmin_plan']:g}, at x1 on these seeds b={f['argmin_x1_same_seeds']:g}, "
                 f"each cell at its best multiplier b={f['argmin_tuned']:g}; reversals beyond the "
                 f"larger sd: {rv} -> "
                 + ("the argmin DEPENDS on learning-rate tuning" if f["depends_on_lr"] else
                    "the argmin does not depend on learning-rate tuning within the tested range")
                 + (f"; optimum not bracketed for b={', '.join(f'{x:g}' for x in b['unbracketed'])}"
                    if b["unbracketed"] else ""))
    m3 = res["m3"]
    for name, b in m3["budgets"].items():
        gs = ", ".join(f"b={c['bits']:g} {fmt(c['mean'], c['sd'])}" for c in b["cells"])
        if b["spread"] is None:
            L.append(f"{_prov(True)}M3 {name}: gains {gs}; too few cells to judge")
            continue
        L.append(f"{_prov(b['provisional'])}M3 {name}: gain val_c(T/2) - val_c(T) {gs}; spread "
                 f"{b['spread']:.4f} vs nearest-neighbour gap of b*_c={b['argmin']:g} "
                 f"{b['nn_gap']:.4f} -> "
                 + ("SENSITIVE to training length" if b["sensitive"] else
                    "not sensitive to training length")
                 + f" (gap to the closest group member b={b['closest_member']:g}: "
                   f"{b['closest_member_gap']:.4f}, against which the spread is "
                 + ("larger" if b["sensitive_vs_closest_member"] else "not larger") + ")")
    m4 = res["m4"]
    if m4["status"] == "pending":
        for c in m4["planned"]:
            L.append(f"PENDING M4 {c['budget_name']} b={c['best_bits']:g} vs b={c['other_bits']:g}"
                     f": m4.csv absent; {c['n_per_cell']} seeds per cell planned (M1 gap "
                     f"{c['m1_gap']:+.4f}, Welch p {fp(c['m1_welch_p'])}; simulated power at "
                     f"n=10 {c['power_by_n']['10']:.3f})")
    else:
        for c in m4["comparisons"]:
            nv = c["null_at_budget"]
            L.append(f"{_prov(m4['status'] != 'complete')}M4 {c['budget_name']} "
                     f"b={c['best_bits']:g} (n={c['n_best']}) vs b={c['other_bits']:g} "
                     f"(n={c['n_other']}): {fmt(c['mean_best'], c['sd_best'])} vs "
                     f"{fmt(c['mean_other'], c['sd_other'])}, gap {c['gap']:+.4f}, Welch p "
                     f"{fp(c['welch_p'])} -> "
                     + (f"RESOLVED (p <= {M4_P})" if c["resolved"] else
                        f"UNRESOLVED at this n (p > {M4_P}); minimum detectable effect "
                        f"{c['mde']['mde']:.4f} nats at 80% power")
                     + f"; §4 at the budget: argmin b={nv['argmin']:g}, gap {nv['gap']:.4f}, "
                       f"s_max {nv['s_max']:.4f} -> " + ("NULL" if nv["is_null"] else "not null"))
    L += _a8_lines(res["a8"])
    fw = res["fineweb"]
    if not fw["complete"]:
        L.append(f"PENDING FineWeb-Edu: missing {fw['missing_files']}")
    else:
        for name, b in fw["budgets"].items():
            n = b["null"][FW_SWEEP]
            near = min((w for w in b["welch_fineweb"] if w["other_bits"] in n["adjacent"]),
                       key=lambda w: abs(w["gap"]))
            L.append(f"FineWeb-Edu {name}: argmin b={n['argmin']:g} ({n['best_mean']:.4f} nats), "
                     f"gap {n['gap']:.4f} vs s_max {n['s_max']:.4f} -> "
                     + ("NULL" if n["is_null"] else "not null")
                     + f"; nearest Welch p={fp(near['p'])} (b={near['other_bits']:g}); "
                       f"TinyStories argmin b={b['null']['tinystories']['argmin']:g}; A6 reading: "
                     + b["reading"])
            L.append(f"FineWeb-Edu {name}: ternary penalty {b['ternary_penalty_bpb'][FW_SWEEP]:.4f}"
                     f" bits/byte vs TinyStories {b['ternary_penalty_bpb']['tinystories']:.4f}")
            rc = b["residual_check"]
            if rc:
                L.append(f"FineWeb-Edu {name} residual check (seed 0): gains "
                         + ", ".join(f"b={k:g} {v:+.4f}" for k, v in sorted(rc["gains"].items()))
                         + f"; spread {rc['spread']:.4f} vs gap {rc['nn_gap']:.4f} -> "
                         + ("SENSITIVE to training length" if rc["sensitive"] else
                            "not sensitive to training length"))
        for k, v in fw["predictions"].items():
            L.append(f"FineWeb-Edu {k}: " + ("HOLDS" if v["holds"] else "FAILS") + "  ["
                     + ", ".join(f"{n}: {'yes' if ok else 'NO'}" for n, ok in v["per_budget"].items())
                     + "]")
    return L


def _a8_lines(a8: dict) -> list[str]:
    L = []
    pending = a8["status"] == "pending"
    if pending:
        c = a8["candidates"]
        h = sum(x["est_train_gpu_h"] for x in c)
        L.append("PENDING A8: x8 not decided (m5_x8.json absent; it is decided once every M2 "
                 "run, extensions included, is recorded). Cells best at x4 on the M2 runs "
                 "recorded so far: "
                 + (", ".join(f"{x['budget_name']} b={x['bits']:g}" for x in c) or "none")
                 + f" ({sum(len(x['seeds']) for x in c)} x8 runs, ~{h:.1f} GPU-h of training at "
                   "the sweep's throughput; as much again at most for x16)")
    for name, b in a8["budgets"].items():
        f, f2 = b["falsifier"], b["m2_falsifier"]
        cells = "; ".join(
            f"b={c['bits']:g} best x{c['best']:g} {fmt(c['best_mean'], c['best_sd'])} "
            f"[{c['bracket']}]"
            + (f" [missing x{', x'.join(f'{m:g}' for m in c['missing'])}]" if c["missing"] else "")
            for c in b["cells"])
        rv = "; ".join(f"b={x['lo']:g}->{x['hi']:g} {x['d_before']:+.4f} -> {x['d_after']:+.4f} "
                       f"(sd {x['sd_max']:.4f})" for x in f["reversals_beyond_sd"]) or "none"
        lo = min(c["tested"][0] for c in b["cells"])
        hi = max(c["tested"][-1] for c in b["cells"])
        L.append(("PENDING " if pending else _prov(b["provisional"]))
                 + f"A8 {name} (seeds {b['seeds']}): {cells}. Argmin b*_c={f['argmin_plan']:g}, "
                   f"each cell at its best multiplier over x{lo:g}-x{hi:g} "
                   f"b={f['argmin_tuned']:g}; "
                   f"reversals beyond the larger sd: {rv} -> "
                 + ("the argmin DEPENDS on learning-rate tuning" if f["depends_on_lr"] else
                    "the argmin does not depend on learning-rate tuning within the tested range")
                 + f" (M2 alone: b={f2['argmin_tuned']:g}, "
                 + ("depends" if f2["depends_on_lr"] else "does not depend") + ")"
                 + (f"; not bracketed at the cap: b={', '.join(f'{x:g}' for x in b['at_cap'])}"
                    if b["at_cap"] else ""))
    return L


# --------------------------------------------------------------------------------------------
# The whole analysis
# --------------------------------------------------------------------------------------------


def analyse(p: Paths = Paths(), z: np.ndarray | None = None) -> dict:
    inp = load_inputs(p)
    m1 = analyse_m1(p, inp)
    m2 = analyse_m2(inp, m1)
    m3 = analyse_m3(inp, m1)
    m4 = analyse_m4(p, inp, m1, z)
    a8 = analyse_a8(inp, m1, m2)
    fw = analyse_fineweb(p)
    missing = []
    if not m2["complete"]:
        missing.append(f"M2 {len(m2['missing_runs'])} of {m2['n_planned']} runs not recorded ("
                       + ", ".join(m2["missing_runs"]) + ")"
                       if m2["extensions_decided"] else "M2 extensions not decided yet")
    if m2["failed_runs"]:
        missing.append(f"M2 failed runs (excluded as §18.3 says): {', '.join(m2['failed_runs'])}")
    if not m3["complete"]:
        missing.append(f"M3 {len(m3['missing_runs']) + len(m3['failed_runs'])} of "
                       f"{m3['n_planned']} runs missing or failed")
    if m4["status"] == "pending":
        missing.append(f"M4 pending (m4.csv absent; {m4['n_planned']} runs planned)")
    elif not m4["complete"]:
        missing.append(f"M4 {len(m4['missing_runs']) + len(m4['failed_runs'])} of "
                       f"{m4['n_planned']} runs missing or failed")
    if a8["status"] == "pending":
        missing.append("A8 pending (m5_x8.json absent: x8 is decided once every M2 run is "
                       "recorded)")
    elif not a8["complete"]:
        if a8["missing_runs"]:
            missing.append(f"A8 {len(a8['missing_runs'])} of {a8['n_planned']} runs not "
                           "recorded (" + ", ".join(a8["missing_runs"]) + ")")
        if not a8["x16_decided"]:
            missing.append("A8 x16 not decided yet")
    if a8["failed_runs"]:
        missing.append("A8 failed runs (excluded: a failed multiplier cannot be best): "
                       + ", ".join(a8["failed_runs"]))
    if not fw["complete"]:
        missing.append(f"FineWeb-Edu outputs missing: {fw['missing_files']}")
    complete = not [x for x in missing if not x.startswith(("M2 failed", "A8 failed"))]
    res = {
        "what": "PREREGISTRATION.md §16-§18 (A7, M1-M4), §19 (A8) and §13 (A6, FineWeb-Edu): "
                "the preregistered analyses, by scripts/a7_analyze.py. Deterministic.",
        "inputs_sha256": inp.sha,
        "complete": complete, "incomplete": missing,
        "groups": [{"budget_name": b["budget_name"], "argmin_bits": b["argmin_bits"],
                    "members": [g["bits"] for g in b["group"]], "seeds_m2_m3": b["seeds_m2_m3"],
                    "reasons": {f"{g['bits']:g}": g["reasons"] for g in b["group"]}}
                   for b in inp.plan["budgets"]],
        "m1": m1, "m2": m2, "m3": m3, "m4": m4, "a8": a8, "fineweb": fw,
    }
    res["verdicts"] = verdict_lines(res)
    return _clean(res)


def status_line(res: dict) -> str:
    return "COMPLETE" if res["complete"] else "INCOMPLETE: " + "; ".join(res["incomplete"])


# --------------------------------------------------------------------------------------------
# ANALYSIS.txt
# --------------------------------------------------------------------------------------------


def _rule(t: str) -> str:
    return f"\n{'=' * 100}\n{t}\n{'=' * 100}"


def render(res: dict) -> str:
    out = ["A7 (PREREGISTRATION.md §16-§18), A8 (§19) and FineWeb-Edu (§13, A6) -- "
           "scripts/a7_analyze.py",
           "Every mean is followed by its sample sd over seeds (± sd); '(n=1)' where one seed.",
           "Losses are nats per token unless marked bits/byte. Generated; do not edit.",
           "", "STATUS: " + status_line(res), "", "inputs (sha256):"]
    out += [f"  {k:<20} {v if v else '(absent)'}" for k, v in res["inputs_sha256"].items()]

    out.append(_rule("M1 -- averaging continuation (all complete TinyStories runs)"))
    out.append("gain = A_mean - val_c; sel = val_fresh - val_ckpt (fixed-val-set selection bias)")
    cols = ("val_loss_best", "A_mean", "val_b", "val_c", "val_d", "val_fresh", "gain", "sel_bias")
    for sweep in SWEEPS:
        for name, b in res["m1"][sweep]["budgets"].items():
            out.append(f"\n{name}  [{sweep}]")
            out.append(f"  {'b':>5} {'n':>2} " + " ".join(f"{c:>17}" for c in cols))
            for c in b["cells"]:
                cells = [fmt(c[k]["mean"], c[k]["sd"]) if c[k] else "-" for k in cols]
                out.append(f"  {c['bits']:>5g} {c['val_c']['n']:>2} "
                           + " ".join(f"{x:>17}" for x in cells))
            for m in ("val_c", "A_mean", "val_loss_best"):
                ws = "  ".join(f"b={w['other_bits']:g}: {w['gap']:+.4f} (sd {w['sd_max']:.4f}) "
                               f"p={fp(w['p'])}" for w in b["welch"][m])
                out.append(f"  Welch vs {m} argmin b={b['null'][m]['argmin']:g}: {ws}")

    out.append(_rule("Argmin groups (frozen in plan.json; reproduced from m1.csv)"))
    for g in res["groups"]:
        out.append(f"  {g['budget_name']}: b*_c={g['argmin_bits']:g}, members "
                   f"{[float(x) for x in g['members']]}, M2/M3 seeds {g['seeds_m2_m3']}")
        for k, v in g["reasons"].items():
            out.append(f"      b={k}: {'; '.join(v)}")

    m2 = res["m2"]
    out.append(_rule(f"M2 -- learning-rate sensitivity ({m2['n_recorded']} of {m2['n_planned']} "
                     "planned runs recorded)"))
    if m2["missing_runs"]:
        out.append("  MISSING (every verdict that depends on them is PROVISIONAL): "
                   + ", ".join(m2["missing_runs"]))
    if m2["failed_runs"]:
        out.append("  FAILED (excluded; cannot be best, §18.3): " + ", ".join(m2["failed_runs"]))
    for name, b in m2["budgets"].items():
        mults = sorted({float(m) for c in b["cells"] for m in c["multipliers"]})
        out.append(f"\n{name}  seeds {b['seeds']}  mean val_c by learning-rate multiplier "
                   "(x1 = the M1 val_c of the same seeds)")
        out.append(f"  {'b':>5} " + " ".join(f"{'x' + format(m, 'g'):>17}" for m in mults)
                   + "  best   bracketed")
        for c in b["cells"]:
            row = []
            for m in mults:
                e = c["multipliers"].get(f"{m:g}")
                row.append("-" if e is None else
                           e["status"].upper() if e["status"] != "complete" else
                           fmt(e["mean"], e["sd"]))
            out.append(f"  {c['bits']:>5g} " + " ".join(f"{x:>17}" for x in row)
                       + f"  x{c['best']:<4g} " + ("yes" if c["bracketed"] else
                                                  f"NO ({c['edge']} edge)")
                       + (" PROVISIONAL" if c["provisional"] else ""))
        f = b["falsifier"]
        out.append(f"  adjacent gaps (x1 same seeds -> each at its best; sd judged = "
                   f"{b['cells'][0]['sd_source']}):")
        for x in f["pairs"]:
            out.append(f"    b={x['lo']:g}->{x['hi']:g}: {x['d_before']:+.4f} -> {x['d_after']:+.4f}"
                       f"  sd {x['sd_max']:.4f}  "
                       + ("REVERSED beyond sd" if x["reversed_beyond_sd"] else
                          "reversed within sd" if x["reversed"] else "same sign"))

    m3 = res["m3"]
    out.append(_rule(f"M3 -- half-budget residual check ({m3['n_recorded']} of {m3['n_planned']} "
                     "runs recorded)"))
    for name, b in m3["budgets"].items():
        out.append(f"\n{name}  b*_c={b['argmin']:g}, grid neighbours {b['nn_adjacent']}")
        out.append(f"  {'b':>5} {'gain mean ± sd':>17}  per seed (val_c(T/2) - val_c(T))")
        for c in b["cells"]:
            ps = "  ".join(f"s{x['seed']}: {x['val_c_half']:.4f}-{x['val_c_full']:.4f}="
                           f"{x['gain']:+.4f}" if x["gain"] is not None else
                           f"s{x['seed']}: {x['status'].upper()}" for x in c["pairs"])
            out.append(f"  {c['bits']:>5g} {fmt(c['mean'], c['sd']):>17}  {ps}")

    m4 = res["m4"]
    out.append(_rule(f"M4 -- seeds ({m4['status'].upper()})"))
    for c in m4["planned"]:
        out.append(f"  planned: {c['budget_name']} b={c['best_bits']:g} vs b={c['other_bits']:g}"
                   f"  triggers {c['triggers']}  n per cell {c['n_per_cell']}  simulated power "
                   + " ".join(f"n={k}:{v:.3f}" for k, v in c["power_by_n"].items()))
    if m4["missing_runs"]:
        out.append("  not recorded: " + ", ".join(m4["missing_runs"]))
    for c in m4["comparisons"]:
        out.append(f"  {c['budget_name']} b={c['best_bits']:g} seeds {c['seeds_best']} "
                   f"{fmt(c['mean_best'], c['sd_best'])}; b={c['other_bits']:g} seeds "
                   f"{c['seeds_other']} {fmt(c['mean_other'], c['sd_other'])}; Welch t "
                   f"{c['welch_t']:.3f} df {c['welch_df']:.2f} p {fp(c['welch_p'])}; MDE "
                   f"{c['mde']['mde']:.4f} (power at it: tabulated "
                   f"{c['mde']['power_at_mde_tabulated']:.3f}, exact Welch "
                   f"{c['mde']['power_at_mde_exact_welch']:.3f})")

    a8 = res["a8"]
    out.append(_rule(f"A8 -- bracketing the learning rate, x8 then x16 ({a8['status'].upper()}; "
                     f"{a8['n_recorded']} of {a8['n_planned']} A8 runs recorded)"))
    if a8["status"] == "pending":
        out.append("  x8 not decided yet: every multiplier below is M2's. Cells best at x4 so "
                   "far: " + (", ".join(f"{c['budget_name']} b={c['bits']:g} ({len(c['seeds'])} "
                                       f"runs, ~{c['est_train_gpu_h']:.2f} GPU-h)"
                                       for c in a8["candidates"]) or "none"))
    else:
        out.append("  x8 to: " + (", ".join(f"{c['budget_name']} b={c['bits']:g}"
                                            for c in a8["x8_cells"]) or "none")
                   + "; x16 to: " + (", ".join(f"{c['budget_name']} b={c['bits']:g}"
                                                for c in a8["x16_cells"]) or "none"
                                     if a8["x16_decided"] else "not decided yet"))
    if a8["missing_runs"]:
        out.append("  not recorded: " + ", ".join(a8["missing_runs"]))
    if a8["failed_runs"]:
        out.append("  FAILED (excluded; cannot be best): " + ", ".join(a8["failed_runs"]))
    for name, b in a8["budgets"].items():
        mults = sorted({float(m) for c in b["cells"] for m in c["multipliers"]})
        out.append(f"\n{name}  seeds {b['seeds']}  mean val_c by learning-rate multiplier "
                   "(x1 = the M1 val_c of the same seeds)")
        out.append(f"  {'b':>5} " + " ".join(f"{'x' + format(m, 'g'):>17}" for m in mults)
                   + "  best   M2 best  bracketed")
        for c in b["cells"]:
            row = []
            for m in mults:
                e = c["multipliers"].get(f"{m:g}")
                row.append("-" if e is None else
                           e["status"].upper() if e["status"] != "complete" else
                           fmt(e["mean"], e["sd"]))
            out.append(f"  {c['bits']:>5g} " + " ".join(f"{x:>17}" for x in row)
                       + f"  x{c['best']:<4g} x{c['best_m2']:<6g} {c['bracket']}"
                       + (" PROVISIONAL" if c["provisional"] else ""))
        f = b["falsifier"]
        out.append(f"  adjacent gaps (x1 same seeds -> each at its best; sd judged = "
                   f"{b['cells'][0]['sd_source']}):")
        for x in f["pairs"]:
            out.append(f"    b={x['lo']:g}->{x['hi']:g}: {x['d_before']:+.4f} -> "
                       f"{x['d_after']:+.4f}  sd {x['sd_max']:.4f}  "
                       + ("REVERSED beyond sd" if x["reversed_beyond_sd"] else
                          "reversed within sd" if x["reversed"] else "same sign"))

    fw = res["fineweb"]
    out.append(_rule("FineWeb-Edu (A6) -- carried from results/fineweb_edu/, checked"))
    if fw["complete"]:
        tpb = fw["tokens_per_byte"]
        out.append(f"  tokens/byte: tinystories {tpb['tinystories']:.6f}, fineweb_edu "
                   f"{tpb[FW_SWEEP]:.6f}; bigram floor bits/byte: tinystories "
                   f"{fw['bigram_floor_bpb']['tinystories']:.3f}, fineweb_edu "
                   f"{fw['bigram_floor_bpb'][FW_SWEEP]:.3f}")
        for name, b in fw["budgets"].items():
            out.append(f"\n{name}")
            out.append(f"  {'b':>5} | {'TinyStories nats':>17} {'bits/byte':>9} | "
                       f"{'FineWeb-Edu nats':>17} {'bits/byte':>9} floor")
            for c in b["cells"]:
                t, f_ = c["tinystories"], c[FW_SWEEP]
                tl = f"{fmt(t['mean'], t['sd']):>17} {t['bpb']:>9.4f}" if t else " " * 27
                fl = (f"{fmt(f_['mean'], f_['sd']):>17} {f_['bpb']:>9.4f} "
                      f"{'ok' if f_['beats_floor'] else 'FAIL'}") if f_ else "-"
                out.append(f"  {c['bits']:>5g} | {tl} | {fl}")
            out.append("  FineWeb-Edu Welch vs its argmin: " + "  ".join(
                f"b={w['other_bits']:g} {w['gap']:+.4f} (sd {w['sd_max']:.4f}) p={fp(w['p'])}"
                for w in b["welch_fineweb"]))

    out.append(_rule("VERDICTS"))
    out += [f"  {v}" for v in res["verdicts"]]
    out.append("")
    out.append(status_line(res))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-csv", type=Path, default=Paths.results_csv)
    ap.add_argument("--a7-dir", type=Path, default=Paths.a7_dir)
    ap.add_argument("--fineweb-dir", type=Path, default=Paths.fineweb_dir)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where ANALYSIS.txt and analysis.json go (default: --a7-dir)")
    args = ap.parse_args(argv)
    p = Paths(args.results_csv, args.a7_dir, args.fineweb_dir)
    try:
        res = analyse(p)
    except Malformed as e:
        print(f"MALFORMED INPUT: {e}", file=sys.stderr)
        print(f"MALFORMED INPUT: {e}")
        return 1
    out = args.out_dir or p.a7_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "ANALYSIS.txt").write_text(render(res))
    (out / "analysis.json").write_text(json.dumps(res, indent=1, sort_keys=False) + "\n")
    print(f"wrote {out / 'ANALYSIS.txt'} and {out / 'analysis.json'}")
    print(status_line(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
