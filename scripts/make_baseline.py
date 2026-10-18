"""Fit the bigram floor once and cache it (SPEC §1.4).

    python scripts/make_baseline.py

Counted on train, evaluated on val, using the same tokenizer and splits as every model, so
the number is directly comparable to a run's val loss.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem.eval import BIGRAM_PATH, compute_and_cache  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    b = compute_and_cache(force="--force" in sys.argv)
    print(f"bigram   val loss {b.val_loss:.4f}  ppl {b.val_ppl:.2f}")
    print(f"unigram  val loss {b.unigram_val_loss:.4f}  ppl {b.unigram_val_ppl:.2f}")
    print(f"trained on {b.train_tokens:,} tokens, k={b.smoothing_k}  ({time.time() - t0:.0f}s)")
    print(f"cached at {BIGRAM_PATH}")
