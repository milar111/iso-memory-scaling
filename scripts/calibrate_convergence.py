"""Measure how many tokens each parameter scale needs to converge (SPEC §2.5).

    python scripts/calibrate_convergence.py

Why this exists
---------------
The pilot showed patience-based early stopping never fires: a cosine schedule decaying to
`min_lr` over `max_steps` makes the loss improve until the final step by construction, so
"no improvement for 5 evals" cannot happen. Convergence is therefore *schedule-relative*,
and the question is not "when does it stop improving" but "at what token budget is the
remaining improvement smaller than our ability to measure it".

Digging into that surfaced the larger problem. At a uniform token cap, tokens-per-parameter
varies ~10x across bit-widths *within a single budget* — at 828 Kbit, b=1.58 gets 572
tok/param while b=16 gets 5,634. That is exactly SPEC §2.5's central confound: the
low-precision configurations are systematically the least-trained, so a uniform cap
measures optimization progress rather than capacity.

The fix is a token budget large enough that even the heaviest configuration converges. Then
every lighter configuration is converged too, and the imbalance stops mattering — a
converged model is converged regardless of how much further it was trained.

Convergence criterion (fixed BEFORE running this)
-------------------------------------------------
Over the evals of a completed run, let `tail` be the final quarter and `prev` the quarter
before it:

    drift = mean(prev) - mean(tail)        # positive means still improving
    noise = stdev(tail)

    CONVERGED  iff  drift <= noise

"Further training moves the number by less than the noise in measuring it." Scale-free, so
it applies unchanged at every budget, and it is the same kind of quantity as the null
criterion in PREREGISTRATION.md §4.

Configurations are deliberately OFF-GRID
----------------------------------------
The shapes below match the grid's parameter *scale* but are not the shape the search picks
for any cell, under either shape rule. So this calibration touches no cell of the
experiment and cannot shape a reported result: the only quantity read from it is a
within-run drift statistic.
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem import data as D  # noqa: E402
from isomem.accounting import ModelConfig, count_params, count_weight_bits  # noqa: E402
from isomem.train import resolve_device, train_one  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "calibration.json"

# (label, config, bits, [token budgets to test])
# Params bracket the grid: 88,064 (grid min is 32,768) to 3,047,424 (grid max 2,616,320).
CASES = [
    ("light", ModelConfig(7, 32, 3.0, 512, 256), 3.0, [100_000_000, 200_000_000, 400_000_000]),
    ("heavy", ModelConfig(8, 192, 3.0, 512, 256), 1.58,
     [300_000_000, 600_000_000, 1_200_000_000]),
]
TARGET_EVALS = 40  # eval granularity, so drift is measured on enough points


def tail_stats(history: list[dict]) -> tuple[float, float, float]:
    h = [e["val"] for e in history]
    q = max(2, len(h) // 4)
    tail, prev = h[-q:], h[-2 * q : -q]
    return st.mean(prev) - st.mean(tail), st.pstdev(tail), min(h)


def main() -> int:
    device = resolve_device()
    train_tokens = D.load_tokens("train")
    val_tokens = D.load_tokens("val")
    print(f"device {device}\n")
    print("CONVERGENCE CALIBRATION -- off-grid shapes at the grid's parameter scales")
    print("criterion fixed in advance: CONVERGED iff drift <= tail noise\n")
    print(f"{'case':>6} {'params':>10} {'tokens':>14} {'tok/par':>8} {'best val':>9} "
          f"{'drift':>8} {'noise':>8} {'ratio':>7} {'verdict':>14} {'min':>6}")

    results = []
    for label, cfg, bits, budgets in CASES:
        B = count_weight_bits(cfg, bits)
        n_params = count_params(cfg)
        for cap in budgets:
            steps = cap // 32768
            t0 = time.time()
            r = train_one(
                cfg, bits, seed=0, budget_bits=B, max_tokens=cap,
                run_id=f"calib-{label}-{cap}",
                train_tokens=train_tokens, val_tokens=val_tokens, device=device,
                micro_batch=128,                      # the adopted setting
                eval_interval=max(1, steps // TARGET_EVALS),
                eval_batches=50,
                patience=10**9,                       # never early-stop: measure the full tail
            )
            drift, noise, best = tail_stats(r.history)
            ratio = drift / noise if noise > 0 else float("inf")
            converged = drift <= noise
            mins = (time.time() - t0) / 60
            print(f"{label:>6} {n_params:>10,} {cap:>14,} {cap / n_params:>8.0f} {best:>9.4f} "
                  f"{drift:>+8.4f} {noise:>8.4f} {ratio:>7.2f} "
                  f"{'CONVERGED' if converged else 'still improving':>14} {mins:>5.0f}m",
                  flush=True)
            results.append({
                "case": label, "params": n_params, "bits": bits,
                "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "mlp_ratio": cfg.mlp_ratio,
                "tokens": cap, "tokens_per_param": cap / n_params,
                "best_val": best, "drift": drift, "noise": noise, "ratio": ratio,
                "converged": converged, "n_evals": len(r.history),
                "wall_clock_min": mins, "tokens_per_s": r.tokens_seen / (mins * 60),
                "history": r.history,
            })
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 96)
    print("RECOMMENDED TOKEN BUDGETS")
    print("=" * 96)
    for label, cfg, bits, _ in CASES:
        rows = [r for r in results if r["case"] == label]
        conv = [r for r in rows if r["converged"]]
        if conv:
            r = min(conv, key=lambda x: x["tokens"])
            print(f"  {label:>6} ({r['params']:,} params): converged at "
                  f"{r['tokens']:,} tokens = {r['tokens_per_param']:.0f} tok/param")
        else:
            r = max(rows, key=lambda x: x["tokens"])
            print(f"  {label:>6} ({r['params']:,} params): NOT converged even at "
                  f"{r['tokens']:,} tokens ({r['tokens_per_param']:.0f} tok/param, "
                  f"ratio {r['ratio']:.2f}) -- extend further")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
