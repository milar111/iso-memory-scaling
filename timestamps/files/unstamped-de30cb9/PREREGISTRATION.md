# Preregistration — Iso-Memory Scaling Laws for Quantized Transformers (Phase 1)

**Committed 2026-09-13, before any sweep run.** Nothing in this document was written with
knowledge of a result. At the time of commit the repository has trained nothing beyond
20–60-step smoke tests on tiny configurations, and no model at any grid configuration has
been trained past 60 steps.

Provenance: `git log` for this file establishes the timestamp. Every results row carries
`git_sha` and the hash of `config/fixed.yaml`, so rows produced before and after any change
are distinguishable.

---

## 1. Question and hypothesis

**Question.** Given a fixed budget of *B bits* of weight storage, which bits-per-weight `b`
produces the best language model?

**Hypothesis under test.** The quantization literature reports results at *matched parameter
count*. Deployment on SRAM-based hardware constrains **bits**, not parameters. If the
memory-matched ranking differs from the parameter-matched ranking, the field is reporting on
the wrong axis.

**We are not trying to produce a slope.** A curve flat within seed noise is a publishable
result ("precision allocation does not matter below X") and will be reported as such. §4
defines, numerically and in advance, what counts as flat.

---

## 2. Predictions

Recorded before seeing any result, so they can be scored honestly afterwards.

**P1 — The curve is U-shaped in val loss, with the optimum at intermediate precision,
2–4 bits.** Reasoning in both directions:

- At `b = 1.58`, the budget buys the most parameters, but at 32K–2.6M parameters there is
  little redundancy for ternary to exploit, and the iso-bit constraint forces extreme
  depth-narrow shapes (at 256 Kbit, 17 layers × 32 wide) that optimise poorly.
- At `b = 8` and `b = 16`, the model is small and increasingly embedding-dominated: the
  embedding's share of the budget rises from 9.8% at `b=1.58` to 49.5% at `b=8`
  (256 Kbit), so most of the budget stops being transformer at all.

**P2 — The optimum moves toward lower `b` as the budget grows.** More absolute capacity
should make per-weight precision matter less. Concretely: `b* = 4` at 256 Kbit, `b* = 3` at
828 Kbit, `b* = 2` at 4 Mbit. This is a genuine prediction and may well be wrong.

**P3 — The matched-parameter and matched-bits argmins differ** (the §8 "money plot"). This
is the paper if it holds. Under matched parameters the literature's ranking puts high
precision at or near the top; under matched bits, P1 puts the optimum at 2–4 bits.

**P4 — Every feasible configuration beats the bigram floor** (val loss 3.1648, ppl 23.68).
Configurations that do not are degenerate and are flagged, not silently averaged in. We
consider it plausible that the smallest cells (256 Kbit at `b = 8`, 32,768 parameters) fail
this and have to be reported as floor-limited.

**P5 — Seed standard deviation is small relative to adjacent-`b` gaps at 4 Mbit, and may
not be at 256 Kbit.** Smaller models are noisier. We expect the null criterion in §4 to be
most at risk of firing at the smallest budget.

---

## 3. What is fixed, and what was decided when

Everything in `config/fixed.yaml` is held constant across all runs and asserted at startup:
TinyStories, one BPE tokenizer at vocab 512 trained on the train split only, `block_size`
256, RoPE, RMSNorm, no biases, tied embeddings, AdamW (β = 0.9/0.95, wd 0.1, clip 1.0).

**One learning-rate rule for every run**, `lr = 3e-3 * (64 / n_embd) ** 0.5`, with no
per-configuration override anywhere in the codebase (enforced by
`test_learning_rate_has_no_per_config_override`). It was **not** tuned, on a held-out cell
or otherwise. If it later proves badly wrong for some width, the response is to tune it on
**one** cell, apply the resulting rule unchanged everywhere, rerun everything, and say so
here — not to adjust it per configuration.

Quantization: absmean scaling at every bit-width, QAT from step 0, weights only.

---

## 4. Null criterion — stated numerically

Let `m(b)` be the mean of `val_loss_best` over seeds at bit-width `b` for a given budget,
and `s(b)` the sample standard deviation over those seeds. Let `b* = argmin m(b)`.

Let `gap = min |m(b) - m(b*)|` over the bit-widths adjacent to `b*` in the grid, and
`s_max = max s(b)` over `b*` and its adjacent bit-widths.

> **The result is NULL at that budget if `gap <= s_max`.**

A null budget is reported as null — in the abstract, not only in an appendix — and its
argmin is not described as "the best" anywhere in the paper.

**Secondary, reported alongside.** Welch's t-test between `b*` and each adjacent bit-width,
α = 0.05, two-sided, with exact p-values reported whatever they are. We will not describe a
non-significant argmin as a finding.

**Power.** With 5 seeds per cell, a two-sample Welch test has roughly 80% power against a
true gap of about 2.0 standard deviations. Gaps smaller than that we are not powered to
detect, and we will say so rather than reporting them as trends.

**The seed standard deviation is printed next to every claimed gap**, in every table and on
every plot. If intervals overlap, the abstract says so.

---

## 5. Run plan

### 5.1 Primary sweep — 100 runs

20 feasible cells × **5 seeds** (`[0,1,2,3,4]`), search rule `TIGHTEST`.

Five seeds, not the spec's minimum of three, because §4's criterion is defined in terms of
the seed standard deviation and a std estimated from 3 points carries roughly ±40%
uncertainty — which would make the claim and the null equally weak.

The 21st cell, **B = 256 Kbit at b = 16, does not exist.** With vocab 512 and `n_embd ≥ 32`,
the tied embedding table alone costs `512 × 32 × 16 = 262,144` bits = **exactly 100.0% of
the budget**, leaving nothing for a transformer block; the smallest legal configuration
needs 201% of budget. This is reported as a result — the embedding-bound regime at its
limit — not as an omission.

### 5.2 Secondary sweep — shape robustness, 42 runs

The same cells at 828 Kbit and 4 Mbit under the `ASPECT` search rule, 3 seeds.

**Declared secondary in advance. It can never override the primary sweep**, and the primary
argmin is the reported argmin regardless of what the secondary shows. Its one job is to
answer: *does the argmin move when the depth/width choice changes?* If it does not, the
ranking is not an artifact of shape. If it does, that is reported as a limitation on the
strength of the claim — not as a better result.

256 Kbit is deliberately excluded: only 1 of its 6 cells picks a different shape under
`ASPECT`, so 5 of 6 reruns would be bit-identical duplicates.

### 5.3 Compute

Upper bound **115.5 hours** for 142 runs, assuming no run ever early-stops. Hard ceiling
**140 hours** in `config/grid.yaml`; `dry_run.py` exits nonzero above it and the sweep
refuses to start a run that would cross it.

### 5.3a Run plan as configured

Exact identifiers, so this document and `config/grid.yaml` cannot silently drift apart.
`test_preregistration_declares_the_run_plan` asserts they match.

```yaml
seeds:            [0, 1, 2, 3, 4]          # primary, 5 seeds
bits_per_weight:  [1.58, 2, 3, 4, 6, 8, 16]
budgets:          256Kbit (262144) | 828Kbit (847872) | 4Mbit (4194304)
infeasible:       256Kbit x b=16
primary:          20 cells x 5 seeds = 100 runs, shape_rule TIGHTEST
robustness:       828Kbit, 4Mbit x 7 cells x 3 seeds = 42 runs, shape_rule ASPECT
total:            142 runs, 115.5 h upper bound, ceiling_hours 140
```

### 5.4 The one permitted adjustment, declared in advance

The token caps (200M / 300M / 600M) were set a priori. The Step 6 convergence pilot, on
**one** held-out cell, measures whether early stopping triggers before the cap.

> If the pilot shows early stopping does not trigger reliably, the token caps may be
> adjusted **once**, **before the main sweep begins**, and the change recorded in this file
> with its date and rationale. **No adjustment after the main sweep starts.** The pilot cell
> is excluded from the headline curve.

This is the only parameter of the experiment that may move after this document is committed.

---

## 6. Planned analyses

Anything not on this list is exploratory and will be labelled exploratory in the write-up.

1. **Headline.** `val_loss_best` vs **actual** bits-per-weight, one line per budget, shaded
   ±1 seed std, best point per budget annotated. Actual bits, never target bits.
2. **Money plot.** The same data as matched-parameter-count (left) and matched-bits (right).
   Whether the argmin moves between panels is the primary test of P3.
3. **Null criterion (§4) applied per budget**, reported whatever it says.
4. **Bigram-floor comparison.** Every run flagged `beats_baseline` true/false; degenerate
   runs flagged in every table, never silently included.
5. **Embedding-bound points** (`embedding_frac > 0.5`) marked distinctly on every plot and
   reported as a separate regime.
6. **Undertrained runs** (hit the token cap without early-stopping) marked and **excluded
   from the headline curve**, with the count reported.
7. **Shape robustness (§5.2):** does the argmin move between `TIGHTEST` and `ASPECT`?

## 7. Exploratory — labelled as such in the write-up

- `val_loss` against aspect ratio, and whether curve structure tracks shape rather than `b`.
- Whether the optimum shifts with budget scale beyond what P2 predicts.
- Story-quality / generation evaluation.
- KV-inclusive accounting as an alternative x-axis (see L2 below).
- Training-dynamics differences between bit-widths at equal tokens.

---

## 8. Stated limitations

**L1 — Weights-only quantization.** Activations stay bf16. The question is about SRAM for
weights. An activation ablation is future work, not scope.

**L2 — The KV cache is excluded from the budget, and it is large.** This is a scope
decision, and at these sizes it is a consequential one: at full context the KV cache is
**16.9× the entire weight budget** at 256 Kbit / b=1.58, 6.2× at 828 Kbit / b=1.58, and
still 0.87–3.7× at the largest budgets. Because wider models carry larger KV caches, a
full hardware accounting would penalise the high-precision configurations — that is, it
would push in the same direction as our hypothesis, so excluding it is the *conservative*
choice for the claim we are making. `kv_bits_full_ctx` is logged for every run and
tabulated in `docs/CONFIG_TABLE.txt`. Any claim of the form "this fits in the Tang Nano's
828 Kbit BSRAM" refers to **weights only** and the paper will say so explicitly wherever it
appears, including in the abstract if the claim appears there.

**L3 — At the smaller budgets, precision and shape are the same axis.** Within ±5% at
256 Kbit, the reachable aspect ratios are *disjoint* across bit-widths (b=3 can only reach
4.0–5.3, b=4 only 5.3–8.0, b=6 only 10.7, b=8 only 16.0). Holding weight bits fixed while
varying precision necessarily walks the model from deep-narrow to shallow-wide. No search
rule can disentangle them. §5.2 bounds the effect where shape *is* free; where it is not,
the confound is stated and not explained away.

**L4 — One dataset, one vocabulary, one context length.** TinyStories, vocab 512,
`block_size` 256. Perplexity is not comparable across any of these, so no cross-setting
claim is made.

**L5 — No PTQ comparison.** Everything is QAT from step 0. We make no claim about
post-training quantization.

**L6 — Scale.** Maximum 4 Mbit of weights (≈2.6M parameters). No claim extrapolates beyond
that, and in particular no claim is made about whether these rankings hold at billion-
parameter scale.

**L7 — Fractional bit-widths are idealised.** Ternary is counted at the grid's declared
1.58 bits/weight; real trit packing (5 trits per byte) costs 1.6. The difference is 1.3%,
inside tolerance. The Phase 2 export uses the true packed cost.

**L8 — Residual nondeterminism.** MPS was verified bitwise deterministic at a fixed seed.
CUDA runs may carry residual cuDNN nondeterminism; the device is logged per run and any
cross-device comparison will be flagged.

---

## 9. Deviations from the specification, declared in advance

Full text in `docs/PROGRESS.md`. Summarised here so that none is discovered post hoc:

| | Deviation |
|---|---|
| D1 | 828 Kbit read as binary 847,872 bits; the spec is internally inconsistent (its worked example implies 828,000). 2.4% apart, inside tolerance. |
| D2 | The grid is 20 cells, not 21: 256 Kbit × b=16 is structurally impossible (§5.1). |
| D3 | Ternary counted at 1.58 bits/weight, not log2(3) = 1.585 (see L7). |
| D4 | Absmean scaling at **every** bit-width, not only ternary. With one 100× outlier, absmax leaves 0.001% of weights nonzero at b = 1.58, 2 **and** 4. Absmax would degrade differently at different `b` — the axis under study — in the direction that flatters the hypothesis. Step size for b ≥ 2 is the Max-Lloyd optimal loading factor for the level count, a fixed published rule not tuned on our data. |
| D5 | Idempotence is tested at the grid level and as integer-code stability. Value-idempotence is impossible under absmean scaling at any width; absmax is value-idempotent precisely because the maximum saturates, which is what makes it collapse. |
| D6 | RMSNorm gains have fp32 latents, counted at 16 bits. |
| D7 | Weight decay excludes 1-D norm gains (GPT-2/nanoGPT convention), identically in every run. |
| D8 | The search tie-break is a named rule, because "minimise \|actual − B\|" does not identify a unique configuration (see L3, §5.2). |

---

## 10. Commitments

1. Report what happened, **including if it is boring**. A flat curve is the result.
2. Do not tune the experiment until it produces a slope.
3. The test split is read **exactly once**, at write-up, through
   `load_test_tokens(i_understand_this_is_the_final_evaluation=True)`. Model selection and
   early stopping use val only.
4. Print the seed standard deviation next to every claimed gap.
5. Report the predictions in §2 as scored, whether or not they held.
6. Publish `results.csv` complete — including failed, degenerate and undertrained runs, with
   their status — not only the rows that made the headline curve.
