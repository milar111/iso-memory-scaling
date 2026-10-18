"""Build the frozen data artifacts, once (SPEC §1.2, §2.7).

    python scripts/prepare_data.py
    ISOMEM_DATASET=fineweb_edu python scripts/prepare_data.py   # a second corpus (A6)

Order matters and is enforced: splits must exist before the tokenizer can be trained on
the train split *only*, and the tokenizer must exist before anything can be encoded.

Writes:
  data/tokenizer/tokenizer.json    the one BPE, vocab 512
  data/tokens/{train,val,test}.bin flat uint16 token streams, EOT-separated
  data/splits/manifest.json        permutation seed, boundaries, document hashes
  config/fixed.yaml                hashes filled in, to be committed

Re-running is refused unless --force, because silently regenerating these files is exactly
the failure SPEC §2.7 warns about.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem import data as D  # noqa: E402
from isomem import tokenizer as T  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS, write_corpus_config  # noqa: E402
from isomem.fixed import FIXED, GRID, REPO_ROOT, write_fixed  # noqa: E402

MANIFEST_PATH = CORPUS.manifest_path
SPLITS_DIR = MANIFEST_PATH.parent


def encode_split(tok, split: str, pool, eot: int) -> tuple[Path, int]:
    """Encode one split to a flat uint16 array, documents separated by EOT."""
    out = D.tokens_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = D.split_indices(len(pool))[split]
    n_tokens = 0
    t0 = time.time()
    # Written incrementally: the train split is ~1.3e9 tokens and must never be held in RAM.
    with open(out, "wb") as fh:
        batch: list[str] = []
        done = 0

        def flush(batch: list[str]) -> int:
            encodings = tok.encode_batch(batch)
            flat: list[int] = []
            for enc in encodings:
                flat.extend(enc.ids)
                flat.append(eot)
            arr = np.asarray(flat, dtype=D.TOKEN_DTYPE)
            arr.tofile(fh)
            return len(arr)

        for doc in D.iter_split_documents(split, pool=pool):
            batch.append(doc)
            if len(batch) >= 8192:
                n_tokens += flush(batch)
                done += len(batch)
                batch = []
                print(
                    f"  {split}: {done:,}/{len(idx):,} docs  {n_tokens:,} tokens"
                    f"  {time.time() - t0:.0f}s",
                    end="\r",
                    flush=True,
                )
        if batch:
            n_tokens += flush(batch)
            done += len(batch)
    print(f"  {split}: {done:,} docs -> {n_tokens:,} tokens  ({time.time() - t0:.0f}s)" + " " * 20)
    return out, n_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if artifacts exist")
    args = ap.parse_args()

    existing = [p for p in (T.TOKENIZER_PATH, *(D.tokens_path(s) for s in D.SPLITS)) if p.exists()]
    if existing and not args.force:
        print("Refusing to regenerate frozen data artifacts. Already present:")
        for p in existing:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print("\nRegenerating changes every perplexity in the experiment (SPEC §2.7).")
        print("Pass --force if that is genuinely what you want.")
        return 1

    print(f"Corpus: {CORPUS.name}")
    print(f"Loading document pool: {D.DATASET_ID} [{D.POOL_SPLIT}]"
          + (f" rev {CORPUS.hf_revision[:12]} prefix of {CORPUS.n_docs:,} docs"
             if not CORPUS.is_default else ""))
    pool = D.load_document_pool()
    n_docs = len(pool)
    bounds = D.split_boundaries(n_docs)
    print(f"  {n_docs:,} documents")
    for name, (start, stop) in bounds.items():
        print(f"  {name:>5}: {stop - start:>9,} docs  [{start:,}:{stop:,}] of the permutation")

    # --- 1. tokenizer, on the TRAIN SPLIT ONLY -----------------------------------------
    print("\nTraining BPE tokenizer (train split only, vocab "
          f"{FIXED['tokenizer']['vocab_size']})")
    t0 = time.time()
    tok = T.train_tokenizer(D.iter_split_documents("train", pool=pool))
    T.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(T.TOKENIZER_PATH))
    tok_hash = T.tokenizer_hash()
    print(f"  vocab={tok.get_vocab_size()}  {time.time() - t0:.0f}s  sha256={tok_hash[:16]}...")

    # --- 2. document-level hashes ------------------------------------------------------
    print("\nHashing documents per split")
    doc_hashes = {}
    for split in D.SPLITS:
        t0 = time.time()
        doc_hashes[split] = D.hash_documents(split, pool=pool)
        print(f"  {split:>5}: {doc_hashes[split][:16]}...  ({time.time() - t0:.0f}s)")

    # --- 3. encode ---------------------------------------------------------------------
    print("\nEncoding splits")
    eot = tok.token_to_id(T.EOT)
    token_hashes, token_counts = {}, {}
    for split in D.SPLITS:
        path, n = encode_split(tok, split, pool, eot)
        token_counts[split] = n
        token_hashes[split] = D.hash_file(path)
        print(f"        sha256={token_hashes[split][:16]}...")

    # --- 4. freeze ---------------------------------------------------------------------
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "dataset": D.DATASET_ID,
                "corpus": CORPUS.name,
                "revision": CORPUS.hf_revision,
                "files": list(CORPUS.hf_files),
                "pool_split": D.POOL_SPLIT,
                "n_docs": n_docs,
                "split_seed": FIXED["data"]["split_seed"],
                "boundaries": {k: list(v) for k, v in bounds.items()},
                "doc_sha256": doc_hashes,
                "token_sha256": token_hashes,
                "token_counts": token_counts,
                "tokenizer_sha256": tok_hash,
                "vocab_size": tok.get_vocab_size(),
                "eot_id": eot,
            },
            indent=2,
        )
        + "\n"
    )

    if CORPUS.is_default:
        cfg = dict(FIXED)
        cfg["tokenizer"] = dict(cfg["tokenizer"], hash=tok_hash)
        cfg["data"] = dict(cfg["data"], hashes=token_hashes, doc_hashes=doc_hashes)
        write_fixed(cfg)
        written = "config/fixed.yaml"
    else:
        need = GRID["token_budget"]["unique_train_tokens"]
        if token_counts["train"] < need:
            print(f"\nREFUSED: train split has {token_counts['train']:,} tokens, fewer than "
                  f"TinyStories' {need:,}. Raise source.n_docs (scripts/size_corpus.py "
                  f"--margin) and rerun with --force. Hashes NOT written.")
            return 1
        write_corpus_config(CORPUS, {
            "tokenizer.hash": tok_hash,
            "data.hashes": token_hashes,
            "data.doc_hashes": doc_hashes,
            "data.unique_train_tokens": int(token_counts["train"]),
        })
        written = str(CORPUS.config_path.relative_to(REPO_ROOT))

    print(f"\nWrote {MANIFEST_PATH.relative_to(REPO_ROOT)} and updated {written}.")
    print("Commit both. Every later run asserts these hashes on load.")
    print(f"\nTotal tokens: train={token_counts['train']:,} "
          f"val={token_counts['val']:,} test={token_counts['test']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
