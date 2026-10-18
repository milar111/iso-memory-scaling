"""Validate the entire grid without training anything (SPEC §4).

    python scripts/dry_run.py

Runs the architecture search for every cell, prints target vs actual bits and the
percentage error, flags embedding-bound cells, diagnoses infeasible ones with their
arithmetic, estimates wall-clock from measured throughput, and exits nonzero if any cell
fails tolerance.

The point is that a human can review and sign off on the whole experiment before a single
GPU-hour is spent. Read the table, not the summary line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem.accounting import count_params, kv_bits_at_full_context  # noqa: E402
from isomem.eval import load_baseline  # noqa: E402
from isomem.fixed import FIXED, GRID, fixed_config_hash, git_sha  # noqa: E402
from isomem.sweep import residual_at, token_budget  # noqa: E402
from isomem.search import (  # noqa: E402
    InfeasibleBudget,
    SearchResult,
    grid_cells,
    search_cell,
)

# SUSTAINED throughput on the RTX 3060, fitted from real multi-hour runs rather than short
# benchmarks (which are warmup-dominated and read ~2.5x pessimistic).
THROUGHPUT_ANCHORS = [(88_064, 1_152_000), (546_816, 619_000), (3_047_424, 244_000)]


def estimate_tokens_per_second(params: int) -> float:
    """Power-law fit through the sustained anchors. Planning only."""
    import math

    xs = [math.log(p) for p, _ in THROUGHPUT_ANCHORS]
    ys = [math.log(t) for _, t in THROUGHPUT_ANCHORS]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return math.exp(my + slope * (math.log(params) - mx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape-rule", default="TIGHTEST", choices=["TIGHTEST", "ASPECT"])
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()

    vocab = FIXED["tokenizer"]["vocab_size"]
    seeds = GRID["seeds"]
    tol = args.tol if args.tol is not None else GRID["search"]["tolerance"]

    print("=" * 100)
    print("ISO-MEMORY SCALING LAWS -- GRID DRY RUN (no training)")
    print("=" * 100)
    print(f"git {git_sha()}   fixed.yaml {fixed_config_hash()}   vocab {vocab}   "
          f"block_size {FIXED['model']['block_size']}")
    print(f"shape rule {args.shape_rule}   tolerance +/-{tol:.0%}   seeds {seeds}")

    feasible: list[tuple[str, SearchResult]] = []
    infeasible: list[tuple[str, float, str]] = []
    for name, budget, bits in grid_cells():
        try:
            feasible.append(
                (name, search_cell(budget, bits, vocab, tol=tol, shape_rule=args.shape_rule))
            )
        except InfeasibleBudget as exc:
            infeasible.append((name, bits, str(exc)))

    # ---------------------------------------------------------------- config table
    print("\n" + "-" * 100)
    print("CONFIGURATIONS")
    print("-" * 100)
    print(f"{'budget':>8} {'b':>5} {'L':>3} {'d':>5} {'h':>3} {'r':>4} {'params':>10} "
          f"{'target bits':>12} {'actual bits':>12} {'err':>7} {'emb%':>6} {'asp':>5} {'flag'}")

    last_budget = None
    for name, r in feasible:
        if last_budget and name != last_budget:
            print()
        last_budget = name
        c = r.config
        flag = "EMBEDDING-BOUND" if r.embedding_bound else ""
        print(f"{name:>8} {r.bits_per_weight:>5} {c.n_layer:>3} {c.n_embd:>5} {c.n_head:>3} "
              f"{c.mlp_ratio:>4} {count_params(c):>10,} {r.target_bits:>12,} "
              f"{r.actual_bits:>12,} {r.rel_error * 100:>+6.2f}% "
              f"{r.embedding_frac * 100:>5.1f}% {r.aspect:>5.1f} {flag}")

    # ---------------------------------------------------------------- infeasible
    if infeasible:
        print("\n" + "-" * 100)
        print(f"INFEASIBLE CELLS ({len(infeasible)}) -- reported as results, not omissions")
        print("-" * 100)
        for name, bits, report in infeasible:
            print(f"\n{name}  b={bits}")
            for line in report.splitlines():
                print(f"  {line}")

    # ---------------------------------------------------------------- excluded detail
    print("\n" + "-" * 100)
    print("EXCLUDED FROM THE BUDGET BUT LOGGED (SPEC §2.1)")
    print("-" * 100)
    print(f"{'budget':>8} {'b':>5} {'kv bits @ full ctx':>20} {'kv / weight bits':>18}")
    for name, r in feasible:
        kv = kv_bits_at_full_context(r.config)
        print(f"{name:>8} {r.bits_per_weight:>5} {kv:>20,} {kv / r.actual_bits:>17.2f}x")

    # ---------------------------------------------------------------- token budgets
    unique = GRID["token_budget"]["unique_train_tokens"]
    print("\n" + "-" * 100)
    print("TOKEN BUDGETS -- set so every cell lands at the SAME convergence residual")
    print("-" * 100)
    print(f"{'budget':>8} {'b':>5} {'params':>10} {'tokens':>15} {'epochs':>7} {'residual':>9}")
    residuals = []
    for name, r in feasible:
        n = count_params(r.config)
        t = token_budget(n)
        residuals.append(residual_at(n, t))
        print(f"{name:>8} {r.bits_per_weight:>5} {n:>10,} {t:>15,} {t / unique:>7.2f} "
              f"{residual_at(n, t):>9.5f}")
    spread = max(residuals) - min(residuals)
    print(f"\n  residual spread across the grid: {spread:.2e} "
          f"({'uniform -- cancels in the ranking' if spread < 1e-6 else 'NOT UNIFORM'})")
    print(f"  max epochs: {max(token_budget(count_params(r.config)) for _, r in feasible) / unique:.2f}"
          f"  (cap {GRID['token_budget']['max_epochs']})")

    # ---------------------------------------------------------------- cost
    print("\n" + "-" * 100)
    print("COST ESTIMATE (RTX 3060 sustained throughput; every run trains its full budget)")
    print("-" * 100)

    def budget_hours(results, n_seeds):
        per: dict[str, float] = {}
        for nm, res in results:
            n = count_params(res.config)
            hours = token_budget(n) / estimate_tokens_per_second(n) / 3600 * n_seeds
            per[nm] = per.get(nm, 0.0) + hours
        return per

    primary = budget_hours(feasible, len(seeds))
    print(f"PRIMARY sweep   shape rule {args.shape_rule}, seeds {seeds}")
    for name, hours in primary.items():
        n = sum(1 for nm, _ in feasible if nm == name)
        print(f"  {name:>8}  {n} cells x {len(seeds)} seeds = {n * len(seeds):>3} runs   "
              f"{hours:>6.1f} h")
    primary_h = sum(primary.values())
    primary_runs = len(feasible) * len(seeds)
    print(f"  {'subtotal':>8}  {primary_runs:>3} runs   {primary_h:>6.1f} h")

    # Secondary shape-robustness sweep, if declared.
    rob = GRID.get("robustness", {})
    secondary_h, secondary_runs = 0.0, 0
    if rob.get("enabled"):
        rob_seeds = rob["seeds"]
        rob_results = []
        for name, budget, bits in grid_cells():
            if name not in rob["budgets"]:
                continue
            try:
                r = search_cell(budget, bits, vocab, tol=tol, shape_rule=rob["shape_rule"])
            except InfeasibleBudget:
                continue
            rob_results.append((name, r))
        per = budget_hours(rob_results, len(rob_seeds))
        print(f"\nSECONDARY shape-robustness   rule {rob['shape_rule']}, seeds {rob_seeds}")
        for name, hours in per.items():
            n = sum(1 for nm, _ in rob_results if nm == name)
            print(f"  {name:>8}  {n} cells x {len(rob_seeds)} seeds = "
                  f"{n * len(rob_seeds):>3} runs   {hours:>6.1f} h")
        secondary_h = sum(per.values())
        secondary_runs = len(rob_results) * len(rob_seeds)
        print(f"  {'subtotal':>8}  {secondary_runs:>3} runs   {secondary_h:>6.1f} h")

    total_h = primary_h + secondary_h
    total_runs = primary_runs + secondary_runs
    print(f"\n  {'TOTAL':>8}  {total_runs:>3} runs   {total_h:>6.1f} h")
    # The machine runs between waking and quiet hours, not around the clock.
    hours_per_day = 15.5
    print(f"  {'':>8}  at {hours_per_day:.1f} h/day (07:30-23:00) that is "
          f"{total_h / hours_per_day:.1f} calendar days")
    print("  Projection only -- there is no compute ceiling and no run is refused on cost.")

    # ---------------------------------------------------------------- baseline
    print("\n" + "-" * 100)
    print("BIGRAM FLOOR (SPEC §1.4)")
    print("-" * 100)
    baseline = load_baseline()
    if baseline is None:
        print("  NOT COMPUTED -- run: python scripts/make_baseline.py")
    else:
        print(f"  bigram   val loss {baseline.val_loss:.4f}   ppl {baseline.val_ppl:.2f}")
        print(f"  unigram  val loss {baseline.unigram_val_loss:.4f}   "
              f"ppl {baseline.unigram_val_ppl:.2f}")
        print("  Any run not beating the bigram number is degenerate and gets flagged.")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 100)
    n_bound = sum(1 for _, r in feasible if r.embedding_bound)
    worst = max((abs(r.rel_error) for _, r in feasible), default=0.0)
    print(f"{len(feasible)} feasible cells x {len(seeds)} seeds = {primary_runs} runs "
          f"primary, {secondary_runs} secondary, {total_runs} total")
    print(f"worst budget error {worst * 100:.2f}%  (tolerance {tol:.0%})")
    print(f"embedding-bound cells: {n_bound}    infeasible cells: {len(infeasible)}")

    print(f"COMPUTE: projected {total_h:.1f} h over {total_runs} runs (no ceiling)")

    failures = [(n, r) for n, r in feasible if abs(r.rel_error) > tol]
    if failures:
        print("\nFAIL: cells outside tolerance:")
        for n, r in failures:
            print(f"  {n} b={r.bits_per_weight}: {r.rel_error * 100:+.2f}%")
        print("=" * 100)
        return 1
    print("\nOK: every feasible cell is within tolerance. Review the table above before"
          " running the sweep.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
