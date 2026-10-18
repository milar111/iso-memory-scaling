"""Choose how many documents of a second corpus to use (PREREGISTRATION.md Amendment A6).

    ISOMEM_DATASET=fineweb_edu python scripts/size_corpus.py [--margin 0.03]

Token budgets are computed from TinyStories' size for both corpora, so every cell trains on
exactly its TinyStories token count. For the epoch count to match as well, the second
corpus' 98% train split must hold at least TinyStories' 805,873,616 training tokens under
its OWN tokenizer -- which does not exist until the pool is chosen. So the size is
estimated, then verified:

1. Train a 512-token BPE on the first CAL documents of the pinned shard.
2. Measure its tokens per UTF-8 byte on the next HOLD documents, which it never saw.
3. Take the smallest document prefix whose train split is predicted to hold the target
   times (1 + margin), counting one end-of-text token per document.

The estimate is conservative in one known direction: the final tokenizer is trained on far
more text and compresses slightly better, so it yields slightly FEWER tokens than predicted.
The margin absorbs that. scripts/prepare_data.py then counts the real number and refuses to
freeze the corpus if it falls short, so an optimistic estimate cannot pass silently.

Writes source.n_docs and source.sizing to the corpus config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem import data as D  # noqa: E402
from isomem import tokenizer as T  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS, write_corpus_config  # noqa: E402
from isomem.fixed import FIXED, GRID  # noqa: E402

CAL, HOLD = 100_000, 20_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.03)
    args = ap.parse_args()
    if CORPUS.is_default:
        print("TinyStories is used whole; there is nothing to size.")
        return 1

    target = GRID["token_budget"]["unique_train_tokens"]
    t0 = time.time()
    pool = D.pinned_pool()
    print(f"{CORPUS.name}: {len(pool):,} documents in the pinned shard(s)  ({time.time()-t0:.0f}s)")

    cal = pool.select(range(CAL))["text"]
    hold = pool.select(range(CAL, CAL + HOLD))["text"]
    t0 = time.time()
    tok = T.train_tokenizer(iter(cal))
    enc = tok.encode_batch(hold)
    n_tok = sum(len(e.ids) for e in enc)
    n_bytes = sum(len(t.encode("utf-8")) for t in hold)
    tpb = n_tok / n_bytes
    print(f"calibration tokenizer on {CAL:,} docs: {tpb:.4f} tokens/byte "
          f"({1/tpb:.3f} bytes/token) on {HOLD:,} held-out docs  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    lengths = np.fromiter(
        (len(t.encode("utf-8")) for batch in pool.iter(batch_size=20_000) for t in batch["text"]),
        dtype=np.int64, count=len(pool),
    )
    cum_bytes = np.cumsum(lengths)
    n = np.arange(1, len(pool) + 1)
    n_val = np.ceil(n * FIXED["data"]["val_frac"]).astype(np.int64)
    n_test = np.ceil(n * FIXED["data"]["test_frac"]).astype(np.int64)
    train_frac = (n - n_val - n_test) / n
    predicted_train = train_frac * (tpb * cum_bytes + n)       # + one EOT per document
    need = target * (1 + args.margin)
    ok = np.nonzero(predicted_train >= need)[0]
    if len(ok) == 0:
        print(f"the whole shard predicts only {predicted_train[-1]:,.0f} train tokens "
              f"< {need:,.0f}; add a shard to source.files")
        return 1
    n_docs = int(ok[0] + 1)
    print(f"byte-length scan  ({time.time()-t0:.0f}s)")
    print(f"target {target:,} train tokens x (1 + {args.margin}) = {need:,.0f}")
    print(f"-> n_docs = {n_docs:,} ({n_docs/len(pool)*100:.1f}% of the shard, "
          f"{cum_bytes[n_docs-1]/1e9:.2f} GB of text), predicted train tokens "
          f"{predicted_train[n_docs-1]:,.0f}")

    write_corpus_config(CORPUS, {
        "source.n_docs": n_docs,
        "source.sizing": {
            "target_train_tokens": int(target),
            "margin": args.margin,
            "calibration_docs": CAL,
            "holdout_docs": HOLD,
            "tokens_per_byte": round(float(tpb), 6),
            "predicted_train_tokens": int(predicted_train[n_docs - 1]),
            "shard_docs": len(pool),
        },
    })
    print(f"wrote source.n_docs to {CORPUS.config_path.name}. "
          f"Next: ISOMEM_DATASET={CORPUS.name} python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
