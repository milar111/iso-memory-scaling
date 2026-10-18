"""Which text corpus this process trains and evaluates on.

TinyStories is the experiment's corpus (SPEC §1.2) and the default. A second corpus exists
only as a separately preregistered arm (PREREGISTRATION.md Amendment A6): the same grid
cells, the same architectures, the same token budgets and the same protocol, on different
text. It answers one question -- does the optimum precision depend on how hard the text
is -- and it can never override the TinyStories result.

Selection is by the ``ISOMEM_DATASET`` environment variable, read once at import. Every
path a corpus touches is derived from its one profile, so the tokens a run reads, the
tokenizer that encoded them, the baseline it is judged against, the checkpoint it resumes
from and the results file it writes cannot disagree about which corpus they belong to.

The default profile keeps TinyStories on its original paths and its hashes in
``config/fixed.yaml``, and the modules that consume a profile read those through their
original code path when the default is active. Nothing about the finished sweep changes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from isomem.fixed import CONFIG_DIR, FIXED, GRID, REPO_ROOT

DEFAULT = "tinystories"
CORPORA_DIR = CONFIG_DIR / "corpora"
ENV_VAR = "ISOMEM_DATASET"


class UnknownCorpus(ValueError):
    """Raised when ISOMEM_DATASET names a corpus with no profile in config/corpora/."""


@dataclass(frozen=True)
class Corpus:
    name: str
    is_default: bool
    hf_id: str
    hf_revision: str | None
    hf_files: tuple[str, ...]
    pool_split: str
    text_column: str
    n_docs: int | None
    tokens_dir: Path
    tokenizer_path: Path
    bigram_path: Path
    results_csv: Path
    ckpt_dir: Path
    manifest_path: Path
    config_path: Path
    run_suffix: str
    sweep_label: str
    tokenizer_hash: str | None
    split_hashes: dict[str, str | None] = field(default_factory=dict)
    doc_hashes: dict[str, str | None] = field(default_factory=dict)
    unique_train_tokens: int | None = None
    arm: dict[str, Any] | None = None

    def config_hash(self) -> str:
        """SHA-256 of this corpus' config file, for the provenance of every row it writes."""
        return hashlib.sha256(self.config_path.read_bytes()).hexdigest()[:16]


def _tinystories() -> Corpus:
    data = REPO_ROOT / "data"
    return Corpus(
        name=DEFAULT,
        is_default=True,
        hf_id="roneneldan/TinyStories",
        hf_revision=None,
        hf_files=(),
        pool_split="train",
        text_column="text",
        n_docs=None,
        tokens_dir=data / "tokens",
        tokenizer_path=data / "tokenizer" / "tokenizer.json",
        bigram_path=REPO_ROOT / "results" / "bigram_baseline.json",
        results_csv=REPO_ROOT / "results" / "results.csv",
        ckpt_dir=REPO_ROOT / "runs",
        manifest_path=data / "splits" / "manifest.json",
        config_path=CONFIG_DIR / "fixed.yaml",
        run_suffix="",
        sweep_label="primary",
        tokenizer_hash=FIXED["tokenizer"]["hash"],
        split_hashes=dict(FIXED["data"]["hashes"]),
        doc_hashes=dict(FIXED["data"]["doc_hashes"]),
        unique_train_tokens=GRID["token_budget"]["unique_train_tokens"],
        arm=None,
    )


def _from_yaml(name: str) -> Corpus:
    path = CORPORA_DIR / f"{name}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in CORPORA_DIR.glob("*.yaml")) if CORPORA_DIR.is_dir() else []
        raise UnknownCorpus(f"no corpus profile {path}; known: {[DEFAULT, *known]}")
    with open(path) as fh:
        c = yaml.safe_load(fh)
    if c.get("name") != name:
        raise UnknownCorpus(f"{path} declares name={c.get('name')!r}, expected {name!r}")
    src, arm = c["source"], c["arm"]
    root = REPO_ROOT / "data" / name
    results = REPO_ROOT / "results" / name
    return Corpus(
        name=name,
        is_default=False,
        hf_id=src["hf_id"],
        hf_revision=src["revision"],
        hf_files=tuple(src["files"]),
        pool_split=src.get("split", "train"),
        text_column=src.get("text_column", "text"),
        n_docs=src.get("n_docs"),
        tokens_dir=root / "tokens",
        tokenizer_path=root / "tokenizer" / "tokenizer.json",
        bigram_path=results / "bigram_baseline.json",
        results_csv=results / "results.csv",
        ckpt_dir=REPO_ROOT / "runs" / name,
        manifest_path=root / "splits" / "manifest.json",
        config_path=path,
        run_suffix=arm["run_suffix"],
        sweep_label=arm["sweep"],
        tokenizer_hash=c["tokenizer"]["hash"],
        split_hashes=dict(c["data"]["hashes"]),
        doc_hashes=dict(c["data"]["doc_hashes"]),
        unique_train_tokens=c["data"].get("unique_train_tokens"),
        arm=arm,
    )


def load(name: str | None = None) -> Corpus:
    name = name or DEFAULT
    return _tinystories() if name == DEFAULT else _from_yaml(name)


def write_corpus_config(corpus: Corpus, updates: dict[str, Any]) -> None:
    """Fill in values only known once the data exists (hashes, sizes). Non-default only.

    TinyStories' values live in ``config/fixed.yaml`` and are written by ``write_fixed``.
    Only the leading comment block survives, exactly as with ``write_fixed``.
    """
    if corpus.is_default:
        raise ValueError("the default corpus is configured in config/fixed.yaml")
    lines = corpus.config_path.read_text().splitlines(keepends=True)
    header = ""
    for line in lines:
        if not line.startswith("#"):
            break
        header += line
    with open(corpus.config_path) as fh:
        cfg = yaml.safe_load(fh)
    for dotted, value in updates.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    with open(corpus.config_path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


ACTIVE: Corpus = load(os.environ.get(ENV_VAR) or DEFAULT)
