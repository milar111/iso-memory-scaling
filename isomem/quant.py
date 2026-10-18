"""Quantization-aware training primitives (SPEC §2.2, §2.3, §2.4).

The failure this module exists to prevent: the forward pass silently uses unquantized
weights, gradients look healthy, loss curves look beautiful, and the experiment has
measured nothing. Every property below has a blocking test (the tests are not part of this release).

Design
------
Weights only; activations stay bf16 (SPEC §2.4). Latent weights are fp32 and are quantized
in the forward pass at every step, from step 0 — never post-training (SPEC §2.3).

Two-level API, deliberately:

* :func:`quantize_to_grid` — pure, stateless projection onto a symmetric integer grid of a
  given step size. Exactly idempotent, because a grid is a grid.
* :func:`compute_step` — derives that step from the tensor's *absmean*.
* :func:`fake_quant` — composes the two, and is what training calls.

Why absmean everywhere, not absmax
----------------------------------
SPEC §2.2 mandates absmean for ternary, per BitNet: "Absmax on a tensor with one outlier
drives every other weight to zero." That argument does not stop at ternary. With absmax
and a single 100x outlier the step becomes ~max/Q_pos, so at b=2 (step ~ 50*mu) and even
b=4 (step ~ 12.5*mu) every ordinary weight rounds to zero. Only around b=8 does absmax
survive an outlier.

That matters here more than in an ordinary QAT paper, because absmax would degrade
*differently at different b* — precisely the axis under study — and it would degrade in the
direction that flatters the hypothesis. A confound that manufactures your expected answer
is the worst kind. So: absmean at every bit-width, and the scheme is held fixed across the
sweep like every other hyperparameter (SPEC §2.8).

Choosing the step from the absmean
----------------------------------
``mean|w|`` estimates scale robustly, but says nothing about how far the grid should
reach. For a Gaussian, ``sigma = mean|w| * sqrt(pi/2)``. Given sigma, the step that
minimises quantization MSE for an N-level symmetric uniform quantizer is a classical
result (Max 1960; Lloyd 1982) and depends only on N, not on the data. Those optimal
loading factors are tabulated in :data:`OPTIMAL_STEP_OVER_SIGMA`.

This is a fixed, published rule applied identically to every run. It is not tuned on our
data, and there is deliberately no per-configuration override.

Ternary is the one special case, because SPEC §2.2 mandates BitNet exactly:
``clip(round(w / mean|w|), -1, 1) * mean|w|``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TERNARY_BITS = 1.58
BF16_CONTROL_BITS = 16
EPS = 1e-8

# Optimal step size / sigma for an N-level symmetric uniform quantizer on a Gaussian
# source (Max-Lloyd). Depends only on the level count, never on our data.
OPTIMAL_STEP_OVER_SIGMA: dict[int, float] = {
    4: 0.9957,     # b = 2
    8: 0.5860,     # b = 3
    16: 0.3352,    # b = 4
    64: 0.1041,    # b = 6
    256: 0.0308,   # b = 8
}

# mean|w| -> sigma for a zero-mean Gaussian: sigma = mean|w| * sqrt(pi/2)
_ABSMEAN_TO_SIGMA = math.sqrt(math.pi / 2)


class TernaryCollapse(AssertionError):
    """Raised when a quantized tensor has collapsed to (almost) all zeros (SPEC §3)."""


def n_levels(bits: float) -> int:
    """Number of distinct values the grid may take at ``bits`` bits per weight.

    Ternary is 3 values, {-1, 0, +1}, stored at log2(3) ~ 1.58 bits. Every other width is
    a full 2**b integer grid, so b=2 gives 4 levels — genuinely one more than ternary,
    which is the comparison the experiment is built to make.
    """
    if bits == TERNARY_BITS:
        return 3
    if bits != int(bits) or bits < 1:
        raise ValueError(f"unsupported bit-width {bits}")
    return 2 ** int(bits)


def grid_range(bits: float) -> tuple[int, int]:
    """Inclusive integer code range. Asymmetric for b>=2 so all 2**b codes are usable."""
    if bits == TERNARY_BITS:
        return -1, 1
    n = n_levels(bits)
    return -(n // 2), n // 2 - 1


def absmean(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor mean absolute value. The robust scale statistic (SPEC §2.2)."""
    return w.detach().abs().mean()


def compute_step(w: torch.Tensor, bits: float) -> torch.Tensor:
    """Quantization step size for ``w`` at ``bits``, derived from its absmean.

    Detached: the scale is a constant of the forward pass, never a path for gradients.
    If it were differentiable the STE would be subtly wrong in a way that every listed
    test in SPEC §6 would still pass.
    """
    mu = absmean(w).clamp_min(EPS)
    if bits == TERNARY_BITS:
        return mu  # BitNet b1.58, exactly
    loading = OPTIMAL_STEP_OVER_SIGMA.get(n_levels(bits))
    if loading is None:
        raise ValueError(f"no optimal loading factor tabulated for bits={bits}")
    return mu * _ABSMEAN_TO_SIGMA * loading


def quantize_to_grid(w: torch.Tensor, bits: float, step: torch.Tensor | float) -> torch.Tensor:
    """Project ``w`` onto the symmetric integer grid of spacing ``step``.

    Pure and stateless, so it is exactly idempotent: a value already on the grid maps to
    itself. This is the property `test_idempotent` checks.
    """
    if bits == BF16_CONTROL_BITS:
        return w
    step_t = torch.as_tensor(step, dtype=w.dtype, device=w.device).detach()
    qmin, qmax = grid_range(bits)
    codes = torch.clamp(torch.round(w / step_t), qmin, qmax)
    return codes * step_t


def quantize_codes(w: torch.Tensor, bits: float, step: torch.Tensor | float) -> torch.Tensor:
    """The integer codes themselves."""
    step_t = torch.as_tensor(step, dtype=w.dtype, device=w.device).detach()
    qmin, qmax = grid_range(bits)
    return torch.clamp(torch.round(w / step_t), qmin, qmax)


def quantize(w: torch.Tensor, bits: float) -> torch.Tensor:
    """Quantize-dequantize with the absmean-derived step. No gradient handling."""
    if bits == BF16_CONTROL_BITS:
        return w
    return quantize_to_grid(w, bits, compute_step(w, bits))


class FakeQuant(torch.autograd.Function):
    """STE. Forward quantizes to <= 2**b levels; backward passes straight through.

    The straight-through estimator is what makes QAT trainable at all: the forward pass is
    a step function whose true derivative is zero almost everywhere, so the backward pass
    substitutes the identity. SPEC §2.2 requires the latent fp32 weight to receive a
    nonzero gradient — `test_ste_gradient_flows` checks the gradient's *value*, not merely
    that it is non-None, because a scale that leaked into the backward pass would still
    produce a plausible-looking nonzero gradient.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, bits: float) -> torch.Tensor:  # type: ignore[override]
        return quantize(w, bits)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


def fake_quant(w: torch.Tensor, bits: float) -> torch.Tensor:
    """Quantize in the forward pass, pass gradients straight through to the latent fp32."""
    if bits == BF16_CONTROL_BITS:
        return w
    return FakeQuant.apply(w, bits)


def nonzero_fraction(w: torch.Tensor) -> float:
    """Fraction of entries that survived quantization as nonzero."""
    return (w != 0).float().mean().item()


def assert_not_collapsed(w: torch.Tensor, name: str, min_nonzero_frac: float = 0.05) -> None:
    """Crash if a quantized tensor has collapsed to zeros (SPEC §3, first row).

    Checked every ``collapse_check_interval_steps``. Crashes with the layer name, because
    a warning in a 60-run sweep is a warning nobody reads.
    """
    frac = nonzero_fraction(w)
    if frac <= min_nonzero_frac:
        raise TernaryCollapse(
            f"quantized tensor {name!r} collapsed: only {frac:.4%} of weights are nonzero "
            f"(minimum {min_nonzero_frac:.2%}). Check the scaling scheme and the LR."
        )


class QuantLinear(nn.Module):
    """Linear layer with fp32 latent weights, quantized in the forward pass every step.

    No bias, anywhere (SPEC §1.2). Activations are untouched: weights-only quantization
    (SPEC §2.4). At ``bits == 16`` this is an ordinary linear layer — the bf16 control.
    """

    def __init__(self, in_features: int, out_features: int, bits: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        # Latent weights are fp32 and stay fp32 (SPEC §2.2).
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def quantized_weight(self) -> torch.Tensor:
        return fake_quant(self.weight, self.bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.quantized_weight()
        # Cast to the activation dtype, never the other way round: the latent parameter
        # must not be silently downcast in place.
        return nn.functional.linear(x, w.to(x.dtype), None)

    def check_not_collapsed(self, name: str, min_nonzero_frac: float = 0.05) -> None:
        if self.bits == BF16_CONTROL_BITS:
            return
        with torch.no_grad():
            assert_not_collapsed(self.quantized_weight(), name, min_nonzero_frac)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, bias=False"
