"""The one tokenizer: trained once on the train split only, hashed, then frozen (SPEC §1.2).

"Perplexity across different vocabularies is not comparable — this is a correctness issue,
not a stylistic one." Every one of the 63 runs uses this exact file, verified by hash on
load.

Byte-level BPE is used so the 512-token vocabulary has no out-of-vocabulary tokens at all:
the 256 single-byte tokens cover any input, and the remaining budget goes to merges.
Training on the train split only is not pedantry — a tokenizer fitted on val or test
leaks information about them into every model's input representation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from isomem.corpus import ACTIVE as CORPUS
from isomem.fixed import FIXED

# data/tokenizer/tokenizer.json for TinyStories, as it always was. A second corpus gets its
# own 512-token BPE, trained on ITS train split only, under the same rules.
TOKENIZER_PATH = CORPUS.tokenizer_path
TOKENIZER_DIR = TOKENIZER_PATH.parent

EOT = "<|endoftext|>"


class TokenizerHashMismatch(AssertionError):
    """Raised when the tokenizer file no longer matches the committed hash."""


def tokenizer_hash(path: Path = TOKENIZER_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_tokenizer(documents, vocab_size: int | None = None):
    """Train the byte-level BPE. Call exactly once, from ``scripts/prepare_data.py``."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    vocab_size = vocab_size or FIXED["tokenizer"]["vocab_size"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, so no UNK
        show_progress=True,
    )
    tok.train_from_iterator(documents, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(
            f"tokenizer has {tok.get_vocab_size()} tokens, expected exactly {vocab_size}"
        )
    return tok


def load_tokenizer(verify_hash: bool = True):
    """Load the frozen tokenizer, asserting its committed hash (SPEC §6)."""
    from tokenizers import Tokenizer

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"{TOKENIZER_PATH} missing. Run: python scripts/prepare_data.py"
        )
    if verify_hash:
        expected = FIXED["tokenizer"]["hash"] if CORPUS.is_default else CORPUS.tokenizer_hash
        if expected is None:
            where = "config/fixed.yaml" if CORPUS.is_default else CORPUS.config_path.name
            raise TokenizerHashMismatch(
                f"no committed tokenizer hash in {where}. "
                "Run scripts/prepare_data.py and commit the result."
            )
        actual = tokenizer_hash()
        if actual != expected:
            raise TokenizerHashMismatch(
                f"tokenizer hash mismatch: expected {expected} got {actual}. "
                "Perplexities computed under a different vocabulary are not comparable."
            )
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def eot_id(tok=None) -> int:
    tok = tok or load_tokenizer()
    return tok.token_to_id(EOT)
