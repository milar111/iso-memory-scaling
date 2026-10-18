"""Loading and enforcement of the invariants held fixed across all runs (SPEC §1.2).

Changing any value in ``config/fixed.yaml`` invalidates cross-run comparison, so the file
is hashed and that hash goes into every results row. If it changes mid-sweep, the rows
from before and after are distinguishable — which is the whole point of SPEC §5's insistence
on ``git_sha`` per row, applied to configuration as well as code.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
FIXED_PATH = CONFIG_DIR / "fixed.yaml"
GRID_PATH = CONFIG_DIR / "grid.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


FIXED: dict[str, Any] = _load_yaml(FIXED_PATH)
GRID: dict[str, Any] = _load_yaml(GRID_PATH)


def write_fixed(cfg: dict[str, Any]) -> None:
    """Rewrite ``config/fixed.yaml``, preserving the leading comment block.

    Used only by the data-preparation scripts, to fill in hashes that cannot be known
    until the data exists.

    Only the LEADING comment block survives -- yaml.safe_dump cannot round-trip comments.
    Anything explanatory that must persist belongs in that header or in the code, not in an
    inline comment, which this function would silently drop.
    """
    lines = FIXED_PATH.read_text().splitlines(keepends=True)
    header = ""
    for line in lines:
        if not line.startswith("#"):
            break
        header += line
    with open(FIXED_PATH, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    FIXED.clear()
    FIXED.update(_load_yaml(FIXED_PATH))


@lru_cache(maxsize=1)
def fixed_config_hash() -> str:
    """SHA-256 of `config/fixed.yaml`, logged per run alongside `git_sha`."""
    return hashlib.sha256(FIXED_PATH.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Short git SHA of the working tree, suffixed ``-dirty`` when uncommitted (SPEC §5).

    Falls back to a committed ``.git_sha`` stamp file when there is no git checkout. The
    compute host receives the repository by rsync without ``.git``, and "unknown" in that
    column would defeat the whole point of SPEC §5: knowing which code produced which row.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        stamp = REPO_ROOT / ".git_sha"
        if stamp.exists():
            return stamp.read_text().strip() or "unknown"
        return "unknown"


# --------------------------------------------------------------------------------------
# The single learning-rate rule (SPEC §1.2, §2.8)
# --------------------------------------------------------------------------------------


def learning_rate(n_embd: int) -> float:
    """``lr = base_lr * (ref_embd / n_embd) ** exponent``.

    One deterministic function of width, applied identically to every run. SPEC §2.8:
    per-config tuning is how you manufacture whichever conclusion you want, usually
    without noticing. There is deliberately no per-run override anywhere in this codebase.
    """
    lr_cfg = FIXED["lr"]
    return lr_cfg["base_lr"] * (lr_cfg["ref_embd"] / n_embd) ** lr_cfg["exponent"]


def assert_fixed_invariants() -> None:
    """Assert at startup that nothing comparison-critical has drifted (SPEC §1.2)."""
    m, d, o, t = FIXED["model"], FIXED["data"], FIXED["optimizer"], FIXED["training"]
    q = FIXED["quantization"]

    # Perplexity is not comparable across vocabularies or context lengths.
    assert FIXED["tokenizer"]["vocab_size"] == 512, "vocab must be 512 for all runs"
    assert m["block_size"] == 256, "block_size must be 256 for all runs"

    # Budget confounds: learned positions and biases would consume budget and vary with width.
    assert m["pos_encoding"] == "rope", "RoPE only (learned positions confound the budget)"
    assert m["bias"] is False, "no biases anywhere"
    assert m["tie_embeddings"] is True, "embedding and head are tied, counted once"
    assert m["norm"] == "rmsnorm" and m["norm_dtype"] == "bfloat16"
    assert m["head_dim"] == 32, "n_head is derived as n_embd // 32"

    assert abs(d["train_frac"] + d["val_frac"] + d["test_frac"] - 1.0) < 1e-12

    assert o["name"] == "adamw" and (o["beta1"], o["beta2"]) == (0.9, 0.95)
    assert o["weight_decay"] == 0.1 and o["grad_clip"] == 1.0

    # bf16, never fp16 (SPEC §3, NaN row).
    assert t["dtype"] == "bfloat16", "fp16 is a documented NaN failure mode; use bf16"
    # Effective batch must stay constant across every run (SPEC §3, OOM row).
    assert t["effective_batch_tokens"] > 0
    assert t["effective_batch_tokens"] % (t["micro_batch"] * m["block_size"]) == 0, (
        "micro_batch must divide the effective batch, or grad_accum would be fractional"
    )
    # Convergence is assessed, never triggered mid-run (Amendment A3).
    assert t["convergence_rule"] == "tail_drift_below_noise"
    assert "early_stop_patience_evals" not in t, "early stopping was reintroduced"

    assert q["scheme"] == "absmean", "absmax collapses low-bit tensors with outliers"
    assert q["step_rule"] == "maxlloyd_optimal_loading"
    assert q["ternary_rule"] == "bitnet_b158"
    assert q["weights_only"] is True, "activations stay bf16 (SPEC §2.4)"
    assert q["qat_from_step_0"] is True, "PTQ below 4 bits measures PTQ, not precision"

    # Token budgets equalise the convergence residual across cells (Amendment A3).
    tb = GRID["token_budget"]
    assert tb["rule"] == "uniform_residual"
    assert 0 < tb["target_residual"] < 1 and tb["max_epochs"] <= 4.0, (
        "repetition beyond ~4 epochs stops being free"
    )
