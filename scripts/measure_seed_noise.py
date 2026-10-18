"""Measure the seed standard deviation (PREREGISTRATION.md §4).

    python scripts/measure_seed_noise.py

The null criterion is defined in terms of seed std, and the token budget must be chosen so
that residual under-training sits well below it. Both need a number, and it cannot be taken
from the grid without peeking at the result — so it is measured here on the same OFF-GRID
shapes used by the convergence calibration.

Reports mean, sample std, and the minimum gap the sweep could resolve at 5 seeds
(roughly 2.0 sd for 80% power on a two-sample Welch test).
"""
from __future__ import annotations
import json, statistics as st, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isomem import data as D
from isomem.accounting import ModelConfig, count_params, count_weight_bits
from isomem.train import resolve_device, train_one

OUT = Path(__file__).resolve().parent.parent / "results" / "seed_noise.json"
# Same off-grid shapes as calibrate_convergence.py, so the two are directly comparable.
CASES = [("light", ModelConfig(7, 32, 3.0, 512, 256), 3.0, 200_000_000)]
SEEDS = [0, 1, 2, 3, 4]

def main() -> int:
    dev = resolve_device()
    tr, va = D.load_tokens("train"), D.load_tokens("val")
    print(f"device {dev}   seeds {SEEDS}\n")
    results = []
    for label, cfg, bits, cap in CASES:
        B, n = count_weight_bits(cfg, bits), count_params(cfg)
        steps = cap // 32768
        vals = []
        print(f"{label}: {n:,} params, {cap:,} tokens, {len(SEEDS)} seeds")
        for s in SEEDS:
            t0 = time.time()
            r = train_one(cfg, bits, seed=s, budget_bits=B, max_tokens=cap,
                          run_id=f"noise-{label}-s{s}", train_tokens=tr, val_tokens=va,
                          device=dev, micro_batch=128, eval_interval=max(1, steps // 40),
                          eval_batches=50, patience=10**9)
            vals.append(r.val_loss_best)
            print(f"  seed {s}: val {r.val_loss_best:.6f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        mean, sd = st.mean(vals), st.stdev(vals)
        print(f"\n  mean {mean:.6f}   sample std {sd:.6f}   range {max(vals)-min(vals):.6f}")
        print(f"  minimum resolvable gap at n=5 (~2.0 sd, 80% power): {2.0*sd:.4f} nats")
        results.append({"case": label, "params": n, "tokens": cap, "seeds": SEEDS,
                        "vals": vals, "mean": mean, "std": sd,
                        "min_resolvable_gap": 2.0*sd})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
