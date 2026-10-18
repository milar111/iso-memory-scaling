"""Dataset, splits, hashing and the test-split guard (SPEC §1.2, §2.6, §2.7).

Splits are *defined* by a seeded permutation of the TinyStories document pool, so they can
be reconstructed from `config/fixed.yaml` alone. Two independent hashes are recorded for
each split:

* ``doc_sha256`` — over the raw documents in split order. Proves the permutation and the
  document pool are unchanged.
* ``token_sha256`` — over the tokenized ``uint16`` array actually read during training.
  Proves the tokenizer and encoding are unchanged too.

Both are asserted on every load. SPEC §2.7: "A silently reshuffled val split makes every
number incomparable and is nearly invisible."

The test split is reachable only through :func:`load_test_tokens`, which refuses to run
unless explicitly unsealed. Everything else in the codebase uses :func:`load_tokens`,
which rejects ``"test"`` outright. That is what makes ``test_no_test_leakage`` meaningful:
there is exactly one door, and it is locked.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from isomem.corpus import ACTIVE as CORPUS
from isomem.fixed import FIXED, REPO_ROOT

SplitName = Literal["train", "val", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "val", "test")

DATA_DIR = REPO_ROOT / "data"
TOKENS_DIR = CORPUS.tokens_dir          # data/tokens for TinyStories, as it always was
CACHE_DIR = DATA_DIR / "cache"

# TinyStories' own train split is our whole document pool; we impose our own 98/1/1
# permutation on it (SPEC §1.2). Their `validation` split is deliberately unused, so that
# every document in our experiment comes from one homogeneous pool. A second corpus
# (isomem/corpus.py, Amendment A6) gets the same seed, fractions and permutation rule.
DATASET_ID = CORPUS.hf_id
POOL_SPLIT = CORPUS.pool_split

TOKEN_DTYPE = np.uint16  # vocab is 512, so uint16 is exact and halves the disk footprint


class SplitHashMismatch(AssertionError):
    """Raised when a split's content no longer matches the committed hash (SPEC §2.7)."""


class TestSplitAccessError(RuntimeError):
    """Raised when the test split is touched outside the single sanctioned moment (SPEC §2.6)."""


# --------------------------------------------------------------------------------------
# Document pool and the seeded permutation
# --------------------------------------------------------------------------------------


def load_document_pool(n_docs: int | None = None):
    """The raw document pool of the active corpus. Cached under ``data/cache``.

    TinyStories: its whole train split, as it always was. A second corpus: a prefix of
    pinned parquet shards in stored order, so the pool is reproducible from its config file
    alone. ``n_docs`` overrides the configured prefix length and exists only for sizing.
    """
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    from datasets import load_dataset  # imported lazily: not needed once tokens exist

    if CORPUS.is_default:
        return load_dataset(DATASET_ID, split=POOL_SPLIT)

    pool = pinned_pool()
    n = n_docs if n_docs is not None else CORPUS.n_docs
    if n is None:
        raise ValueError(
            f"{CORPUS.config_path} has no source.n_docs yet. Run scripts/size_corpus.py."
        )
    if n > len(pool):
        raise ValueError(f"n_docs={n:,} exceeds the {len(pool):,} documents in {files}")
    return pool.select(range(n))


def pinned_pool():
    """Every document of a second corpus' pinned shards, text column only, stored order."""
    if CORPUS.is_default:
        raise ValueError("TinyStories is not loaded from pinned shards")
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    files = [
        hf_hub_download(CORPUS.hf_id, f, repo_type="dataset", revision=CORPUS.hf_revision)
        for f in CORPUS.hf_files
    ]
    pool = load_dataset("parquet", data_files=files, split="train")
    if CORPUS.text_column != "text":
        pool = pool.rename_column(CORPUS.text_column, "text")
    return pool.select_columns(["text"])


def split_boundaries(n_docs: int) -> dict[SplitName, tuple[int, int]]:
    """Half-open [start, stop) index ranges into the permuted document order.

    Val and test are sized by ``ceil`` so neither can round down to zero, and train takes
    the remainder. Disjoint and exhaustive by construction, which is what
    ``test_splits_disjoint`` verifies rather than assumes.
    """
    d = FIXED["data"]
    n_val = int(np.ceil(n_docs * d["val_frac"]))
    n_test = int(np.ceil(n_docs * d["test_frac"]))
    n_train = n_docs - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"train split empty: n_docs={n_docs}")
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_docs),
    }


def permutation(n_docs: int) -> np.ndarray:
    """The one seeded permutation that defines every split. Deterministic forever."""
    rng = np.random.default_rng(FIXED["data"]["split_seed"])
    return rng.permutation(n_docs)


def split_indices(n_docs: int) -> dict[SplitName, np.ndarray]:
    perm = permutation(n_docs)
    return {name: perm[start:stop] for name, (start, stop) in split_boundaries(n_docs).items()}


def iter_split_documents(split: SplitName, pool=None) -> Iterator[str]:
    """Documents of ``split`` in split order.

    Used by the tokenizer trainer (train split only) and by the document hasher.
    """
    pool = load_document_pool() if pool is None else pool
    idx = split_indices(len(pool))[split]
    # Chunked gather: datasets' fancy indexing is far faster than per-row __getitem__.
    chunk = 8192
    for start in range(0, len(idx), chunk):
        rows = pool.select(idx[start : start + chunk].tolist())
        yield from rows["text"]


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------


def hash_documents(split: SplitName, pool=None) -> str:
    """SHA-256 over the split's documents in order.

    Each document is followed by a NUL byte so that concatenation is unambiguous: without
    a separator, moving a character across a document boundary would leave the hash
    unchanged.
    """
    h = hashlib.sha256()
    for doc in iter_split_documents(split, pool=pool):
        h.update(doc.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def hash_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def tokens_path(split: SplitName) -> Path:
    return TOKENS_DIR / f"{split}.bin"


# --------------------------------------------------------------------------------------
# Loading, with hash assertions and the test-split lock
# --------------------------------------------------------------------------------------


def _load_and_verify(split: SplitName) -> np.ndarray:
    path = tokens_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python scripts/prepare_data.py"
        )
    if CORPUS.is_default:
        expected = FIXED["data"]["hashes"].get(split)
    else:
        expected = CORPUS.split_hashes.get(split)
    if expected is None:
        where = "config/fixed.yaml" if CORPUS.is_default else CORPUS.config_path.name
        raise SplitHashMismatch(
            f"no committed hash for split {split!r} in {where}. "
            "Run scripts/prepare_data.py and commit the result."
        )
    actual = hash_file(path)
    if actual != expected:
        raise SplitHashMismatch(
            f"split {split!r} hash mismatch: expected {expected} got {actual}. "
            "The data changed underneath the experiment; every number is now incomparable."
        )
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def load_tokens(split: Literal["train", "val"]) -> np.ndarray:
    """Load a token array, asserting its committed hash (SPEC §2.7).

    Refuses the test split. Model selection and early stopping use val, always (SPEC §2.6).
    """
    if split == "test":
        raise TestSplitAccessError(
            "load_tokens() must never serve the test split. Val is for early stopping and "
            "model selection (SPEC §2.6). The test split is read exactly once, at write-up, "
            "via load_test_tokens(i_understand_this_is_the_final_evaluation=True)."
        )
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    return _load_and_verify(split)


def load_test_tokens(*, i_understand_this_is_the_final_evaluation: bool = False) -> np.ndarray:
    """The only door to the test split (SPEC §2.6).

    Locked behind an explicit keyword so it cannot be reached by accident, by a default
    argument, or by a loop over ``SPLITS``. Called exactly once, at write-up.
    """
    if not i_understand_this_is_the_final_evaluation:
        raise TestSplitAccessError(
            "test split is sealed until final write-up (SPEC §2.6). Use load_tokens('val')."
        )
    return _load_and_verify("test")
