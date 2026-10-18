#!/usr/bin/env python
"""A6 residual check: are the cells around the argmin equally converged on this corpus?

    ISOMEM_DATASET=fineweb_edu python scripts/residual_check.py

The uniform-residual token budgets (Amendment A3) were calibrated on TinyStories. On a second
corpus the cells could be left at different distances from convergence, which would bias the
ranking. PREREGISTRATION.md A6 commits to measuring that:

For each budget, take the argmin and its neighbouring bit-widths. Retrain seed 0 of each at
HALF its token budget -- a complete run, cosine schedule over the shorter length -- and report
the gain from doubling, ``val(T/2) - val(T)``, paired against the arm's own seed-0 run at T.
If the gains differ across the triple by more than the gap between the argmin and its nearest
neighbour, the ranking at that budget is reported as sensitive to training length.

The check runs have their own ids (suffix ``-half``) and their own results file, and never
replace or enter an arm result. Idempotent and resumable: a run already recorded is skipped,
and one interrupted by quiet hours resumes from its checkpoint on the next invocation.

Exit codes: 0 all check runs recorded and the report written; 3 work remains (deferred or
interrupted); 2 the arm itself is not finished yet.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem import data as D  # noqa: E402
from isomem.accounting import count_params  # noqa: E402
from isomem.analysis import load_cells, load_rows  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS  # noqa: E402
from isomem.fixed import FIXED, git_sha  # noqa: E402
from isomem.search import search_cell  # noqa: E402
from isomem.sweep import (  # noqa: E402
    GracefulStop,
    assert_device_consistent,
    next_deadline,
    planned_cells,
    resolve_device,
    token_budget,
    train_one,
)

SUFFIX = "-half"
FACTOR = 0.5
CHECK_CSV = CORPUS.results_csv.parent / "residual_check.csv"
REPORT = CORPUS.results_csv.parent / "residual_check.txt"
FIELDS = ["run_id", "arm_run_id", "budget_name", "bits", "seed", "tokens_half",
          "tokens_full", "val_half", "val_full", "gain", "git_sha"]
MIN_RUN_MINUTES = 30


def triples() -> dict[str, list]:
    """Argmin and its neighbours per budget, from the arm's usable cells."""
    cells = load_cells(sweep=CORPUS.sweep_label)
    out = {}
    for name in CORPUS.arm["budgets"]:
        row = sorted([c for c in cells if c.budget_name == name and c.usable],
                     key=lambda c: c.bits)
        if not row:
            continue
        i = row.index(min(row, key=lambda c: c.mean))
        out[name] = row[max(0, i - 1): i + 2]
    return out


def recorded() -> dict[str, dict]:
    if not CHECK_CSV.exists():
        return {}
    with open(CHECK_CSV, newline="") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


def append(row: dict) -> None:
    new = not CHECK_CSV.exists()
    CHECK_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECK_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def report(tr: dict[str, list], rows: dict[str, dict]) -> None:
    lines = [f"A6 residual check — {CORPUS.name}", ""]
    for name, cells in tr.items():
        best = min(cells, key=lambda c: c.mean)
        others = [c for c in cells if c is not best]
        gap = min(abs(c.mean - best.mean) for c in others) if others else float("nan")
        gains = {}
        lines.append(f"{name}: argmin b={best.bits:g}, nearest-neighbour gap {gap:.4f}")
        for c in cells:
            r = next(v for v in rows.values() if v["budget_name"] == name
                     and float(v["bits"]) == c.bits)
            gains[c.bits] = float(r["gain"])
            lines.append(f"   b={c.bits:<5g} val(T/2) {float(r['val_half']):.4f}  "
                         f"val(T) {float(r['val_full']):.4f}  gain {gains[c.bits]:+.4f}")
        spread = max(gains.values()) - min(gains.values())
        verdict = ("SENSITIVE to training length" if spread > gap
                   else "not sensitive to training length")
        lines.append(f"   spread of gains {spread:.4f} vs gap {gap:.4f} -> {verdict}")
        lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    if CORPUS.is_default:
        print("the residual check belongs to a second corpus (A6); set ISOMEM_DATASET")
        return 1
    done_arm = {r["run_id"] for r in load_rows() if r["status"] in ("complete", "infeasible")}
    if not all(c.run_id in done_arm for c in planned_cells()):
        print("arm not finished; the residual check runs after it")
        return 2

    tr = triples()
    arm_rows = {r["run_id"]: r for r in load_rows() if r["status"] == "complete"}
    plan = []
    for name, cells in tr.items():
        for c in cells:
            arm_id = next(r for r in arm_rows.values() if r["budget_name"] == name
                          and float(r["bits_per_weight"]) == c.bits and r["seed"] == "0")
            plan.append((name, c, arm_id))

    have = recorded()
    todo = [p for p in plan if p[2]["run_id"] + SUFFIX not in have]
    print(f"{len(plan)} check runs planned, {len(plan) - len(todo)} recorded, {len(todo)} to run")
    if todo:
        device = resolve_device(None)
        assert_device_consistent(device)
        train_tokens, val_tokens = D.load_tokens("train"), D.load_tokens("val")
        for name, c, arm in todo:
            deadline = next_deadline(None, None)
            rid = arm["run_id"] + SUFFIX
            resuming = (CORPUS.ckpt_dir / f"{rid}.pt").exists()
            if deadline and not resuming and (deadline - time.time()) / 60 < MIN_RUN_MINUTES:
                print("too close to quiet hours; deferring")
                return 3
            budget_bits = int(arm["budget_bits_target"])
            cfg = search_cell(budget_bits, c.bits, shape_rule=arm["shape_rule"]).config
            full = token_budget(count_params(cfg))
            half = int(full * FACTOR)
            print(f"{rid}: {half:,} tokens (half of {full:,})", flush=True)
            try:
                result = train_one(
                    cfg, c.bits, seed=0, budget_bits=budget_bits, max_tokens=half,
                    run_id=rid, ckpt_dir=CORPUS.ckpt_dir, device=device,
                    train_tokens=train_tokens, val_tokens=val_tokens,
                    micro_batch=FIXED["training"]["micro_batch"], deadline=deadline,
                )
            except GracefulStop as exc:
                print(f"    {exc}; resumes next tick")
                return 3
            val_full = float(arm["val_loss_best"])
            append({"run_id": rid, "arm_run_id": arm["run_id"], "budget_name": name,
                    "bits": c.bits, "seed": 0, "tokens_half": result.tokens_seen,
                    "tokens_full": int(arm["tokens_seen"]),
                    "val_half": result.val_loss_best, "val_full": val_full,
                    "gain": result.val_loss_best - val_full, "git_sha": git_sha()})
            print(f"    val(T/2) {result.val_loss_best:.4f}  val(T) {val_full:.4f}", flush=True)
    report(tr, recorded())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
