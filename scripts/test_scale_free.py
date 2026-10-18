"""Is the convergence residual a function of tokens-per-parameter alone?

    python scripts/test_scale_free.py

This decides how the sweep's token budgets are set.

The calibration compared a light (88K) and a heavy (3.05M) configuration, but at completely
different points on their own training curves -- 4,542 vs 394 tokens/parameter -- and found
very different decay rates (0.37 vs 0.65 per doubling). That difference might be about model
SIZE, or it might just be about being earlier on the curve.

It matters because of what it implies for the experiment:

* If the residual depends only on tokens/parameter, then setting `tokens = k * params` gives
  every cell the SAME residual. It becomes a constant offset on every point, the ranking is
  unbiased, and the sweep can proceed at a budget we can actually afford.
* If it depends on model size as well, equal-k does NOT equalise residuals. Low `b` buys more
  parameters, so the low-precision configurations would carry more un-extracted loss than the
  high-precision ones -- a systematic bias pointing in the direction that flatters the
  hypothesis, and 4.3x the seed noise in size.

So: run the two smaller configurations at the SAME tokens/parameter values the heavy one was
measured at, and compare decay ratios at matched k. Cheap, because at k <= 394 the small
models need very few tokens.

All three shapes are off-grid, so nothing here touches a cell of the experiment.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isomem import data as D
from isomem.accounting import ModelConfig, count_params, count_weight_bits
from isomem.train import resolve_device, train_one

OUT = Path(__file__).resolve().parent.parent / "results" / "scale_free.json"
K_VALUES = [98, 197, 394]          # the heavy config's measured points
CASES = [("light", ModelConfig(7, 32, 3.0, 512, 256), 3.0),
         ("mid",   ModelConfig(6, 96, 2.5, 512, 256), 2.0)]
SEED = 0

def main() -> int:
    dev = resolve_device()
    tr, va = D.load_tokens("train"), D.load_tokens("val")
    print(f"device {dev}\nMatched tokens/parameter against the heavy config's k values\n")
    print(f"{'case':>6} {'params':>10} {'k':>5} {'tokens':>13} {'best val':>9} {'gain':>9} {'min':>6}")
    results = []
    for label, cfg, bits in CASES:
        B, n = count_weight_bits(cfg, bits), count_params(cfg)
        prev = None
        for k in K_VALUES:
            cap = int(n * k)
            steps = max(40, cap // 32768)
            t0 = time.time()
            r = train_one(cfg, bits, seed=SEED, budget_bits=B, max_tokens=cap,
                          run_id=f"sf-{label}-k{k}", train_tokens=tr, val_tokens=va,
                          device=dev, micro_batch=128, eval_interval=max(1, steps // 40),
                          eval_batches=50, patience=10**9)
            gain = (prev - r.val_loss_best) if prev is not None else None
            print(f"{label:>6} {n:>10,} {k:>5} {cap:>13,} {r.val_loss_best:>9.4f} "
                  f"{('%+.4f'%gain) if gain is not None else '—':>9} "
                  f"{(time.time()-t0)/60:>5.1f}m", flush=True)
            results.append({"case": label, "params": n, "k": k, "tokens": cap,
                            "best_val": r.val_loss_best, "gain": gain})
            prev = r.val_loss_best
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2) + "\n")
        print()

    print("=" * 78)
    print("DECAY RATIO AT MATCHED TOKENS/PARAMETER")
    print("=" * 78)
    print(f"{'case':>6} {'params':>10} {'decay 197->394':>16} {'residual at k=394':>19}")
    ratios = {}
    for label, _, _ in CASES:
        g = [r["gain"] for r in results if r["case"] == label and r["gain"] is not None]
        if len(g) >= 2 and g[0]:
            ratio = g[1] / g[0]
            ratios[label] = ratio
            resid = g[1] * ratio / (1 - ratio) if ratio < 1 else float("inf")
            n = next(r["params"] for r in results if r["case"] == label)
            print(f"{label:>6} {n:>10,} {ratio:>16.2f} {resid:>19.4f}")
    # the heavy numbers come from calibration.json, same k values
    calib = Path(__file__).resolve().parent.parent / "results" / "calibration.json"
    if calib.exists():
        h = [r for r in json.loads(calib.read_text()) if r["case"] == "heavy"]
        g = [h[i-1]["best_val"] - h[i]["best_val"] for i in range(1, len(h))]
        ratio = g[1] / g[0]
        ratios["heavy"] = ratio
        print(f"{'heavy':>6} {h[0]['params']:>10,} {ratio:>16.2f} "
              f"{g[1]*ratio/(1-ratio):>19.4f}   (from calibration)")

    print()
    if len(ratios) >= 2:
        spread = max(ratios.values()) - min(ratios.values())
        print(f"spread in decay ratio across a {max(r['params'] for r in results)/88064:.0f}x "
              f"parameter range: {spread:.2f}")
        print("SCALE-FREE (equal-k equalises residuals; set tokens = k * params)"
              if spread < 0.15 else
              "NOT scale-free: residual depends on model size as well as k.\n"
              "Equal-k would leave low-precision cells systematically less converged.")
    print(f"\nwritten to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
