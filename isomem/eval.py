"""Perplexity and the bigram floor (SPEC §1.4, §6).

The bigram baseline exists to detect a *floor effect*: a configuration too small to learn
anything will sit near the unigram/bigram entropy while still producing a plausible-looking
loss curve. Without a floor to compare against, such a point silently joins the headline
curve and flattens it. SPEC §1.4: "Any config not beating this is degenerate; flag it."

The baseline uses the same tokenizer and the same splits as every model, so the numbers are
directly comparable. It is counted on train and evaluated on val, exactly like a model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isomem import data as D
from isomem.corpus import ACTIVE as CORPUS
from isomem.fixed import FIXED

# One floor per corpus: a bigram fitted on one corpus says nothing about another.
BIGRAM_PATH = CORPUS.bigram_path
# Add-k smoothing. k=1 (Laplace) over-smooths a 512x512 table badly; 0.1 is the standard
# compromise and is fixed, not tuned against the result.
SMOOTHING_K = 0.1


def perplexity(loss: float) -> float:
    """ppl = exp(loss). Loss is mean natural-log likelihood per token."""
    return math.exp(loss)


@dataclass(frozen=True)
class BigramBaseline:
    val_loss: float
    val_ppl: float
    unigram_val_loss: float
    unigram_val_ppl: float
    vocab_size: int
    train_tokens: int
    smoothing_k: float

    def to_json(self) -> dict:
        return {
            "val_loss": self.val_loss,
            "val_ppl": self.val_ppl,
            "unigram_val_loss": self.unigram_val_loss,
            "unigram_val_ppl": self.unigram_val_ppl,
            "vocab_size": self.vocab_size,
            "train_tokens": self.train_tokens,
            "smoothing_k": self.smoothing_k,
        }


def fit_bigram(
    train: np.ndarray, val: np.ndarray, vocab: int, k: float = SMOOTHING_K, chunk: int = 1 << 24
) -> BigramBaseline:
    """Count bigrams on train, evaluate mean NLL on val. One pass, chunked."""
    counts = np.zeros(vocab * vocab, dtype=np.int64)
    unigram = np.zeros(vocab, dtype=np.int64)

    for start in range(0, len(train) - 1, chunk):
        stop = min(start + chunk, len(train) - 1)
        prev = np.asarray(train[start:stop], dtype=np.int64)
        nxt = np.asarray(train[start + 1 : stop + 1], dtype=np.int64)
        counts += np.bincount(prev * vocab + nxt, minlength=vocab * vocab)
        unigram += np.bincount(prev, minlength=vocab)

    table = counts.reshape(vocab, vocab).astype(np.float64) + k
    log_probs = np.log(table) - np.log(table.sum(axis=1, keepdims=True))

    uni = unigram.astype(np.float64) + k
    uni_log = np.log(uni) - np.log(uni.sum())

    v_prev = np.asarray(val[:-1], dtype=np.int64)
    v_next = np.asarray(val[1:], dtype=np.int64)
    bigram_nll = -log_probs[v_prev, v_next].mean()
    unigram_nll = -uni_log[v_next].mean()

    return BigramBaseline(
        val_loss=float(bigram_nll),
        val_ppl=perplexity(float(bigram_nll)),
        unigram_val_loss=float(unigram_nll),
        unigram_val_ppl=perplexity(float(unigram_nll)),
        vocab_size=vocab,
        train_tokens=int(len(train)),
        smoothing_k=k,
    )


def compute_and_cache(force: bool = False, max_train_tokens: int | None = None) -> BigramBaseline:
    """Fit the baseline once and cache it; every run compares against the same number."""
    if BIGRAM_PATH.exists() and not force:
        return BigramBaseline(**json.loads(BIGRAM_PATH.read_text()))

    vocab = FIXED["tokenizer"]["vocab_size"]
    train = D.load_tokens("train")
    if max_train_tokens:
        train = train[:max_train_tokens]
    baseline = fit_bigram(train, D.load_tokens("val"), vocab)

    BIGRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    BIGRAM_PATH.write_text(json.dumps(baseline.to_json(), indent=2) + "\n")
    return baseline


def load_baseline() -> BigramBaseline | None:
    if not BIGRAM_PATH.exists():
        return None
    return BigramBaseline(**json.loads(BIGRAM_PATH.read_text()))


def beats_baseline(val_loss: float, baseline: BigramBaseline) -> bool:
    """A model at or above the bigram floor has learned nothing worth reporting."""
    return val_loss < baseline.val_loss
