"""Convergence-protocol pilot (SPEC §7 week 6-7, PREREGISTRATION.md §5.4 + A2).

    python scripts/pilot.py

Runs ONE held-out cell per budget at full token cap and measures whether early stopping
triggers before the cap, and where. This is the measurement that §5.4's single permitted
token-cap adjustment depends on.

Pilot cells are held out: they are EXCLUDED from the headline curve. They are chosen
mid-grid, away from the bit-widths most likely to carry the headline claim, so that
calibrating on them cannot shape the reported result.

Three outcomes:
  * stops well before the cap  -> caps are generous; real sweep cost is below the projection
  * stops just before the cap  -> caps are about right
  * never stops                -> runs would be marked undertrained and excluded; triggers
                                  the one permitted cap adjustment in §5.4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from isomem import data as D  # noqa: E402
from isomem.accounting import count_params  # noqa: E402
from isomem.eval import load_baseline  # noqa: E402
from isomem.fixed import FIXED, GRID, assert_fixed_invariants, git_sha  # noqa: E402
from isomem.search import search_cell  # noqa: E402
from isomem.train import resolve_device, train_one  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot.json"

# One held-out cell per budget. b=3 and b=6 are mid-grid and are not where P1 places the
# optimum, so calibrating the protocol on them cannot shape the headline result.
PILOT_CELLS = [("256Kbit", 3.0), ("828Kbit", 3.0), ("4Mbit", 6.0)]
PILOT_SEED = 0


def budget_bits(name: str) -> int:
    return next(b["bits"] for b in GRID["budgets_bits"] if b["name"] == name)


def token_cap(name: str) -> int:
    return GRID["token_budget"][f"max_tokens_{name}"]


def main() -> int:
    assert_fixed_invariants()
    device = resolve_device()
    baseline = load_baseline()
    print(f"device {device}   git {git_sha()}   seed {PILOT_SEED}")
    print(f"bigram floor: val loss {baseline.val_loss:.4f}\n" if baseline else "")

    train_tokens = D.load_tokens("train")
    val_tokens = D.load_tokens("val")
    results = []

    for name, bits in PILOT_CELLS:
        B = budget_bits(name)
        cap = token_cap(name)
        r = search_cell(B, bits)
        cfg = r.config
        print("=" * 90)
        print(f"PILOT {name} b={bits}   L={cfg.n_layer} d={cfg.n_embd} r={cfg.mlp_ratio}   "
              f"{count_params(cfg):,} params   cap {cap:,} tokens")
        print("=" * 90, flush=True)

        t0 = time.time()
        run = train_one(
            cfg, bits, seed=PILOT_SEED, budget_bits=B, max_tokens=cap,
            run_id=f"pilot-{name}-b{bits}",
            train_tokens=train_tokens, val_tokens=val_tokens,
            device=device, log_every=500,
        )
        elapsed = time.time() - t0

        frac = run.tokens_seen / cap
        verdict = (
            "NEVER STOPPED (undertrained)" if run.undertrained
            else "stopped early, comfortably" if frac < 0.8
            else "stopped early, only just"
        )
        print(f"\n  -> {verdict}")
        print(f"     tokens {run.tokens_seen:,} / {cap:,} = {frac:.1%} of cap")
        print(f"     best val loss {run.val_loss_best:.4f} (ppl {run.val_ppl_best:.2f}) "
              f"at step {run.best_step} of {run.steps}")
        if baseline:
            print(f"     beats bigram floor: {run.val_loss_best < baseline.val_loss}")
        print(f"     wall clock {elapsed / 60:.1f} min   "
              f"{run.tokens_seen / elapsed:,.0f} tok/s\n", flush=True)

        results.append({
            "budget": name, "bits": bits, "cap": cap,
            "n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "mlp_ratio": cfg.mlp_ratio,
            "params": count_params(cfg),
            "tokens_seen": run.tokens_seen, "frac_of_cap": frac,
            "steps": run.steps, "best_step": run.best_step,
            "stopped_early": run.stopped_early, "undertrained": run.undertrained,
            "val_loss_best": run.val_loss_best, "val_ppl_best": run.val_ppl_best,
            "train_loss": run.train_loss,
            "beats_baseline": (run.val_loss_best < baseline.val_loss) if baseline else None,
            "wall_clock_s": elapsed, "tokens_per_s": run.tokens_seen / elapsed,
            "device": str(device), "git_sha": git_sha(),
            "history": run.history,
        })
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("=" * 90)
    print("PILOT SUMMARY")
    print("=" * 90)
    print(f"{'budget':>8} {'b':>4} {'tokens/cap':>11} {'stopped':>20} {'best val':>9} "
          f"{'min':>7}")
    for x in results:
        v = "never (undertrained)" if x["undertrained"] else "early"
        print(f"{x['budget']:>8} {x['bits']:>4} {x['frac_of_cap']:>10.1%} {v:>20} "
              f"{x['val_loss_best']:>9.4f} {x['wall_clock_s'] / 60:>6.1f}")
    n_under = sum(x["undertrained"] for x in results)
    print(f"\nundertrained: {n_under} of {len(results)}")
    if n_under:
        print("ACTION: token caps need the one permitted adjustment (PREREGISTRATION.md §5.4)")
    else:
        print("Early stopping triggers at every budget. Token caps stand unchanged.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
