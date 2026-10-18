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

Upper bound **115.5 hours** for 142 runs, assuming no run ever early-stops.

**There is no compute ceiling** (Amendment A1). Compute is not the binding constraint on
this project; a defensible result is. `dry_run.py` projects wall-clock so the cost of a plan
is visible before it starts, but nothing is enforced and no run is refused on cost grounds.
If the pilot or the sweep shows more compute is needed for a sound result, more compute is
spent.

This removes a *budget* limit only. The *protocol* limit in §5.4 is unaffected and stands.

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
total:            142 runs, 115.5 h upper bound, no compute ceiling
```

### 5.4 The one permitted adjustment, declared in advance

The token caps (200M / 300M / 600M) were set a priori. The Step 6 convergence pilot measures
whether early stopping triggers before the cap, on **one held-out cell per budget**
(three cells; widened by Amendment A2 from the original one cell).

This rule is about scientific integrity, not cost, and **it is not relaxed by the removal of
the compute ceiling in A1** — if anything it binds harder, since no external limit now stops
an experiment from being re-run until it produces a slope.

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

---

## 11. Amendments

Recorded rather than silently applied. Each states what changed, when, why, and whether any
result existed at the time. **No model at any grid configuration had been trained past 60
steps when A1 and A2 were made**, so neither could have been influenced by a result.

### A1 — Compute ceiling removed (2026-09-13)

**Was:** hard ceiling of 140 GPU-hours in `config/grid.yaml`; `dry_run.py` exited nonzero
above it. **Now:** no ceiling; wall-clock is projected but never enforced.

**Why:** the project's objective is a result that survives review, and compute is not the
binding constraint. A cost cap could only ever have forced a *reduction* in scope.

**Effect on the claims: none, or conservative.** Removing a cost cap can only permit more
runs or more tokens, both of which tighten error bars. It cannot make any reported effect
larger. §5.4's protocol limit is explicitly unaffected.

### A2 — Convergence pilot widened from one cell to three (2026-09-13)

**Was:** the §5.4 pilot ran on one held-out cell. **Now:** one held-out cell per budget —
256 Kbit, 828 Kbit and 4 Mbit — since the three budgets have different token caps
(200M / 300M / 600M) and a calibration at one budget says little about the others.

**Why:** A1 removed the cost reason for testing only one. Convergence behaviour is the thing
§5.4's one permitted adjustment depends on, so it is calibrated per budget rather than
extrapolated.

**Effect on the claims: none.** Pilot cells are excluded from the headline curve, exactly as
before. The number of permitted adjustments is unchanged: still **once, before the main
sweep, and never after**.

### A3 — Convergence protocol replaced (2026-09-14)

Made **before any sweep run**, on the evidence of held-out pilot cells and off-grid
calibration only. No cell of the experiment has been trained under the final protocol, so
nothing here could have been influenced by a result.

**Was.** SPEC §2.5: train to convergence, early-stop on val loss with patience 5 evals,
mark anything hitting the token cap as undertrained. Token caps of 200M / 300M / 600M per
budget tier.

**Now.** No early stopping at all. Every run completes its full cosine schedule, and the
token budget is set **per cell** so that every cell finishes with the *same* convergence
residual.

**Why the mechanism changed.** Three held-out pilot cells ran to 100% of their cap without
early stopping ever firing, with the best val loss at 98%, 96% and 100% of training. That
is structural, not incidental: a cosine schedule decaying to `min_lr` over `max_steps`
improves until the final step by construction, so "no improvement for 5 evals" cannot
occur. Worse, stopping early would truncate the schedule and leave the learning rate high,
producing a **worse** model than simply finishing — early stopping here would degrade
precisely the runs it fired on.

**Why the budgets changed, which matters more.** Under the original uniform caps,
tokens-per-parameter varied ~10x across bit-widths *within one budget* (at 828 Kbit, b=1.58
got 572 tok/param and b=16 got 5,634). Since low `b` buys more parameters, the
low-precision configurations were systematically the least-trained — SPEC §2.5's central
confound, pointing in the direction that flatters the hypothesis.

**What was measured.** Eight runs over a 35x parameter range (88K–3.05M) and a 70x token
range (8.6M–1.2B), all on **off-grid shapes** so no cell of the experiment was touched:

* the decay per doubling is scale-free at **0.65** (0.71 / 0.62 / 0.65 across the range);
* remaining improvement fits `gain ~ N^0.179 * T^-0.793`, mean |log residual| 0.106.

The token exponent dominates the parameter exponent, so **convergence is governed almost
entirely by total tokens and barely by model size**. An intermediate proposal — budgets
proportional to parameter count — was tested and **rejected**: at equal tokens-per-parameter
the residual spans 0.547 / 0.105 / 0.046 across the three scales, which would have biased
*against high precision*, again in the direction that flatters the hypothesis.

**The rule adopted.** `target_residual = 0.020`, per-cell tokens solved from the fit, capped
at 4 epochs. This is the **tightest residual reachable by every cell**: the largest cell
needs 3.22e9 tokens = 3.99 epochs. Anything tighter caps the large cells and not the small
ones and the spread reappears — at 0.011, fourteen of nineteen cells cap and the spread
becomes 1.2x the seed standard deviation. The number is set by the data and the epoch
limit, not chosen.

A uniform residual is a constant offset on every val loss and therefore cancels in the
ranking. It is **reported, not hidden**: `target_residual` and `epochs` are columns in
`results.csv`, and the write-up states that all reported losses sit ~0.020 nats above their
fully-converged values.

**Convergence is now assessed, not triggered.** `val_drift`, `val_noise` and `converged` are
logged per run from the eval history. An earlier criterion (`drift <= tail noise`) was tried
and **rejected before use**: under a cosine schedule the final quarter always sees the same
relative LR drop, so tail drift measures schedule shape rather than convergence —
quadrupling the tokens left the ratio unchanged at 1.6–3.2.

**Seed noise, measured for §4.** 5 seeds on an off-grid config: **σ = 0.0075 nats**, giving a
minimum resolvable gap of **0.015 nats** at n=5. The uniform residual spread across the grid
is < 1e-6 nats, i.e. no systematic bias at all on the axis under study.

**Effect on the claims.** The protocol is strictly more conservative than the preregistered
one: every configuration is trained further, and the residual that remains is equalised
rather than left correlated with bit-width. No prediction in §2, no null criterion in §4 and
no planned analysis in §6 is altered.

---

## 11b. Amendment A5 — two limitations added 2026-09-16

Added after the sweep began, from a first pass at the literature and from calibrating the
generated text against the dataset's own paper. L1–L8 were written before any result;
these were not, and must never be presented as though they were.

**L9 — Uniform bit-width.** Every weight in every model carries the same `b`. Variable
Bit-width Quantization (arXiv 2607.02893) reports that spending a fixed bit budget unevenly
beats spending it uniformly. If that holds here, `b*` is the best **uniform** allocation at
a budget, not the best allocation. Nothing in this study tests non-uniform allocation, and
no claim here may be read as ruling it out.

**L10 — The whole grid sits below the dataset's own coherence threshold.** Eldan & Li
(arXiv 2305.07759), who built TinyStories, place fluent and consistent story generation
below 10M parameters. The largest model anywhere in this study is 2,616,320 parameters
(4 Mbit at ternary) — 3.8× under that — and the headline 828 Kbit winner is 140,064, which
is 71× under. Sampled text from these models has correct local grammar and drifting
meaning, which is what that size predicts.

This does not invalidate the comparison: validation loss is well defined and sensitive
regardless of whether samples read well, and the ranking is stable across the 3.23× budget
range scored so far (`b* = 6` at both 256 Kbit and 828 Kbit) even as loss improves from
2.3315 to 1.8580. But it bounds the claim. We measure which precision spends a small
memory budget best; we do not show that the ranking survives into the regime where models
produce coherent text, and 4 Mbit is a third point, not a demonstration.

---

## 12. Amendment A4 — a prediction recorded before the 4 Mbit data exists

**Recorded 2026-09-16. At this moment 0 of 35 primary 4 Mbit runs have started**, which is
checkable: no row in `results.csv` carries `budget_name == "4Mbit"`, and the sweep's own
ordering puts that budget after the two already scored.

This is **not** part of the original preregistration and must never be presented as though
it were. P1–P5 were written before any result. This one was written after seeing two
budgets, and is recorded here only so that the third budget is a genuine out-of-sample test
rather than a curve fitted to data already in hand.

**The observation.** The cost of ternary relative to the best bit-width at the same budget
fell sharply between the two budgets scored:

| budget | best `b` | best val | ternary val | penalty |
|---|---|---|---|---|
| 256 Kbit | 6 | 2.3315 | 2.8416 | 0.5102 |
| 828 Kbit | 6 | 1.8580 | 2.0688 | 0.2108 |

Two points fix a power law exactly and test it not at all: `penalty ∝ B^-0.753`.

**P6 — the ternary penalty at 4 Mbit will be 0.063 nats, and in any case below 0.12.**
The point estimate follows the fitted exponent. The looser bound is the claim being tested:
that the penalty continues to fall substantially rather than flattening. If the penalty at
4 Mbit lands above 0.12, the trend is not a power law over this range and the extrapolation
below is void.

**P7 — the argmin stays at `b = 6` at 4 Mbit.** This contradicts the original P2, which
predicted the optimum would move to `b = 2` by 4 Mbit. P2 is already unsupported between the
first two budgets. Recording the opposite prediction makes 4 Mbit decisive either way.

**What this does not license.** Extrapolating the exponent to where ternary breaks even
gives roughly 50 Mbit, which is 57× beyond the measured range and ~8M ternary weights. That
number does not belong in the paper as a claim. It is written here only to be honest that
the extrapolation was computed and deliberately not relied upon: three points spanning 16×
cannot support a projection 57× out, and the published claim stays inside the measured
range.

---

## 13. Amendment A6 — a second corpus (2026-09-26)

**Recorded before any run of this arm exists**, which is checkable: `results/fineweb_edu/`
holds only the bigram floor, and no row anywhere carries a `-fwedu` run id.

**Authority.** SPEC §10 and AGENTS.md exclude "datasets other than TinyStories". The
project owner lifted that exclusion on 2026-09-26 for this one arm, after the primary
sweep had finished. Nothing else in the exclusion list changes.

**Why.** The strongest objection to the TinyStories result is that TinyStories is simple
text, and simple text may need less capacity -- which would favour spending bits on
precision rather than on parameter count, and could by itself produce an optimum at b=6.
This arm tests that mechanism directly.

### Design — only the text changes

| | TinyStories (primary) | FineWeb-Edu (this arm) |
|---|---|---|
| source | `roneneldan/TinyStories`, train split | `HuggingFaceFW/fineweb-edu` @ `87f09149ef47…`, `sample/10BT/000_00000.parquet`, first 359,595 documents in stored order (ODC-By) |
| tokenizer | 512-token byte-level BPE on its own train split | the same, on **its** own train split |
| splits | seed 1337, 98/1/1 | seed 1337, 98/1/1 |
| unique train tokens | 805,873,616 | 830,708,553 (sized to be ≥ TinyStories', +3.1%) |
| token budget per cell | `token_budget(params)` | **identical** — computed with TinyStories' size, by construction |
| architectures | search at vocab 512 | identical (same budgets, same vocab, same search) |
| optimiser, LR rule, schedule, batch | fixed.yaml | identical |

Because the token count per cell is identical and the corpus is at least as large, every
cell sees the same number of tokens and at most the same number of epochs as its
TinyStories counterpart. Model shape, compute and repetition are held fixed; the text is
the only variable.

**Measured difficulty.** In bits per byte, which is comparable across tokenizers, the
bigram floor is 1.973 on TinyStories and **2.739 on FineWeb-Edu** (39% higher). Bytes per
token under each corpus' own tokenizer: TinyStories 2.314, FineWeb-Edu 2.021.

**Why not WikiText-103.** About 150M tokens under a 512-token BPE, so the largest cells
would need 10-20 epochs against TinyStories' 1.5-4. That changes the repetition regime,
which the primary sweep showed matters (the post-minimum rise). FineWeb-Edu can be sized
to match.

**Budgets: 256 Kbit and 4 Mbit, 5 seeds, `TIGHTEST`, all seven widths.** 828 Kbit is
excluded deliberately: on TinyStories its argmin is not established (NULL between b=6 and
b=8, and it moves to b=4 under `ASPECT`), so a change there could not be interpreted. At the
other two budgets the TinyStories answer is clean -- decisive at 256 Kbit, and stable under
both shape rules at 4 Mbit. 256 Kbit × b=16 is infeasible here for the same reason as before.
Runs are ordered seed-major, so a complete k-seed curve exists after k passes.

**Cost, measured not modelled:** the same shapes at the same token counts took 81.2 GPU-h
on TinyStories (256 Kbit 9.9 h, 4 Mbit 71.3 h).

### Status and analysis

**Secondary, like §5.2. It can never override the TinyStories result.** Its one job is to
answer whether the finding holds on harder text.

- Primary metric `val_loss_best`; the §4 null criterion and Welch tests, unchanged.
- **Losses are never compared across corpora in nats per token**: the tokenizers differ.
  Any cross-corpus magnitude is converted to bits per byte (`loss × tokens_per_byte / ln 2`)
  using each corpus' measured val `tokens_per_byte` (0.432179 and 0.494814).
- Within a corpus, everything is compared exactly as in the primary sweep.

### Predictions, recorded before any result

**P8 — the optimum does not move to higher precision.** At each budget the FineWeb-Edu
argmin is at b ≤ 6. The objection's mechanism predicts b < 6 (harder text favours
parameters); b = 6 would mean the TinyStories optimum is not an artefact of easy text.

**P9 — the ternary penalty shrinks on harder text.** At each budget, the gap between
ternary and the argmin, in bits per byte, is smaller on FineWeb-Edu than on TinyStories
(TinyStories: 256 Kbit 0.5102 nats → 0.3181 bits/byte; 4 Mbit 0.0842 nats → 0.0525 bits/byte).

**P10 — every feasible configuration beats the FineWeb-Edu bigram floor** (3.8371 nats per
token, 2.739 bits/byte). The 256 Kbit cells are the ones at risk of a floor effect; any cell
that fails is flagged and excluded, not averaged in.

### Interpretation rules, fixed now

- Same argmin as TinyStories at a budget, null not fired → the result holds on harder text
  at that budget.
- Different argmin, null not fired → reported as *the optimum depends on the text*, with its
  direction and size in bits per byte. The TinyStories result stands as measured; the
  generality of the claim is narrowed accordingly.
- Null fired → no argmin is claimed for FineWeb-Edu at that budget.

### What this arm does not establish, and the check that will

The uniform-residual token budgets (A3) were calibrated on TinyStories. On FineWeb-Edu the
cells might be left at different distances from convergence, which could bias the ranking.
That is measured, not assumed: **after the arm completes, seed 0 of the three cells around
each budget's argmin is retrained at HALF its token budget** -- a complete run, its cosine
schedule spanning the shorter length -- as a separate run that never replaces an arm
result. The gain from doubling, `val(T/2) − val(T)` paired against the arm's own seed-0 run,
is reported per cell. If those gains differ across the triple by more than the gap between
the argmin and its nearest neighbour, the FineWeb-Edu ranking at that budget is reported as
sensitive to training length.

**Correction, same day, before any row of this arm existed.** The first version of this
paragraph said *twice* the token budget. Measured before running it, doubling would take the
4 Mbit cells to 5.4–6.7 epochs of the 830.7M-token corpus, far past the 4-epoch cap, so the
check would be confounded by repetition exactly where it matters most. Measuring the same
per-doubling gain from below keeps every check run at ≤ 1.7 epochs, costs a quarter as much,
and is the quantity A3's gain model is written in. Under that model the gain from `T/2` to
`T` and from `T` to `2T` differ by the same factor `2^b` for every cell, so the comparison
across the triple -- the only thing the check is used for -- is unaffected.

**Correction, 2026-09-27, to the last sentence above. The check itself is unchanged.**
"Unaffected" was wrong. The check compares the *spread* of the gains with the argmin's
nearest-neighbour gap, not their ordering. Under the gain model every gain from `T/2` to `T`
is `2^-b = 2^0.793 ≈ 1.73` times the gain from `T` to `2T` for every cell (derived from the
fitted model, not measured), so the spread grows by that factor while the gap does not:
measuring from below is stricter than the doubling check it replaced, by that factor. Under
the same model the residual left after `T` is `ρ/(1−ρ) = 0.65/0.35 ≈ 1.86` times the gain
from `T/2` to `T` (`isomem.sweep.residual_at`), so a "not sensitive" verdict bounds the
spread of *residuals* at about 1.86 times the gap, not at the gap. Each gain is also a
single-seed difference, and the noise of the check itself has not been measured. The
write-up states all three. None of them changes the criterion, the cells retrained, or how
the verdict is reported.

---

## 14. Note, 2026-09-27 — the test-set analysis, fixed before the test split is read

Recorded before the test split has been opened (no results row carries a `test_loss`). It
changes no prediction, criterion or TinyStories verdict; it fixes in advance what the one
permitted read of the test split (§10.3) will measure and how it will be analysed, because
that read cannot be repeated.

**What is measured.** Checkpoints hold the *final-step* weights, not the weights at
`val_loss_best`, so the test read measures the final step. Each complete primary run's
final-step checkpoint is evaluated once on the test token stream through
`load_test_tokens(i_understand_this_is_the_final_evaluation=True)`, with the procedure used
for every validation evaluation (50 batches of 128 windows of 256 tokens drawn with the fixed
evaluation seed), and the loss is written to that run's `test_loss` column.

**How it is analysed**, per budget, on the primary (`TIGHTEST`) sweep only:

1. per-cell mean and sample standard deviation of the final-step test loss over the 5 seeds;
2. the §4 null criterion applied to those means and standard deviations, unchanged;
3. the two-sided Welch test of the test-loss argmin against each adjacent bit-width, with
   the §4 power rule (gap against 2.0 × the larger seed sd of the pair).

The validation verdicts on `val_loss_best` remain the preregistered primary result. The test
analysis is reported beside them as the unselected estimate, whatever it shows; where it
disagrees with the validation verdict at a budget, the paper says so at that budget and in
the abstract.

---

## 15. Correction to Amendment A4 (2026-09-27)

A4 says: "At this moment 0 of 35 primary 4 Mbit runs have started". **That was false.** The
compute host's `results/sweep.log` shows the first 4 Mbit primary tick at 2026-09-15 20:00:01
UTC. Reconstructed from that tick and the rows' `wall_clock_s`, `B4194304-b1.58-s0`
completed at about 03:28 UTC on 2026-09-16 (`val_loss_best` 1.4159), 4 h 19 min before A4 was
committed (b008193, 07:46:41 UTC). `B4194304-b1.58-s1` completed within about one minute of
the commit, so the order of those two events cannot be resolved. The check A4 cites (no
`4Mbit` row in `results.csv`) was evidently run against the Mac's unsynced copy of the file,
or not run at all.

Consequences, fixed now:

* **P6** concerns exactly the 4 Mbit ternary cell. A result for it existed when P6 was
  written. Whether its author had seen that result is not recorded, so P6 is **not** a clean
  out-of-sample prediction. It is scored as recorded and reported as post hoc.
* **P7** (argmin stays at `b = 6`) concerns cells with `b != 1.58`, none of which had started.
  The only 4 Mbit number other than ternary in existence was the `b = 6` pilot run (A2). P7
  stands as recorded, with that stated.

---

## 16. Amendment A7 — artifact checks on the training recipe (2026-09-27)

**Recorded after the TinyStories sweep finished and before any run below exists.** This
section is committed and externally timestamped (OpenTimestamps, hash of this commit) before
the first A7 result is read. Nothing here changes a TinyStories prediction, criterion or
verdict: the §4 verdicts on `val_loss_best` stay the preregistered primary result. A7 adds
sensitivity analyses, reported beside them.

### Why

A literature review and an independent re-measurement (`docs/RELATED_WORK.md` §2, 2026-09-27)
found three recipe choices whose effect on the comparison between bit-widths is measured to be
large enough, or unmeasured and plausibly large enough, to move the argmin at 828 Kbit and
4 Mbit:

1. **End-of-training weight noise.** Latent weights pile up at quantization decision
   boundaries at low `b`. On the seven local seed-0 checkpoints (828 Kbit `b <= 8`, 4 Mbit
   ternary), continuing training 400 steps at the final learning rate and rounding the
   time-averaged code improved val loss relative to `val_loss_best` by 0.134 / 0.116 / 0.106 /
   0.092 / 0.042 / 0.018 nats at `b` = 1.58 / 2 / 3 / 4 / 6 / 8 (828 Kbit), and by 0.082 at
   4 Mbit ternary. After it, 828 Kbit `b = 3` sits 0.007 behind `b = 6`, inside one seed sd.
   Seed 0 only, one trajectory, no `b = 16` control, run on the Mac. **These numbers have been
   seen.** They motivated this amendment and are not part of its results: A7 recomputes them
   on the compute host with the procedure below.
2. **One untuned learning-rate rule.** Factor-2 learning-rate errors are reported to cost
   0.004–0.021 nats at small scale, the size of the 4 Mbit and 828 Kbit gaps. Not measured here.
3. **Extrapolated token budgets.** The A3 gain model is applied beyond its calibrated token
   range in 19 of 20 cells (by `T`), and was calibrated with size and `b` confounded. Not
   checked on TinyStories.

Two comparisons are also underpowered at `n = 5`: 4 Mbit `b = 6` vs `b = 4`, and 828 Kbit
`b = 6` vs `b = 8` (power 0.21).

### M1 — averaging continuation (all 142 complete TinyStories runs)

For every complete run, primary and robustness, starting from its final checkpoint with the
model **and** AdamW state restored:

* continue `K = 400` optimizer steps at the schedule's final learning rate
  (`lr * min_lr_frac`), on the run's own batch stream (micro-steps continuing from its final
  step), with the run's weight decay, gradient clip and micro-batch. Nothing is written to
  `results.csv` or to the run's checkpoint.
* **(a)** evaluate the last iterate every 25 steps (16 evaluations). `A` = their mean, the
  no-selection baseline.
* **(b)** for `b < 16`: round the time-average of each weight's integer code over the 400
  steps; scale by the per-tensor step of the final latent weights. `val_b`.
* **(c)** for every `b`, including 16: average the fp32 latent weights over the 400 steps and
  quantize the average with the run's own quantizer. `val_c` — **the corrected metric of A7.**
* **(d)** a separate continuation from the same checkpoint, 400 steps with the learning rate
  annealed linearly from `lr * min_lr_frac` to 0; last iterate. `val_d`.
* Before continuing, the checkpoint is evaluated as saved; if that differs from the run's
  recorded final validation loss by more than 1e-3, the run is aborted and reported (the setup
  would be wrong). It is also evaluated on 50 fresh validation batches drawn with a different
  seed, which measures the selection bias of the fixed evaluation set.
* The (c) weights are saved (`runs/averaged/<run_id>.pt`) so that the single test read (§14)
  can evaluate them. **§14 is extended accordingly:** that read also evaluates each primary
  run's (c) weights, with the same procedure, reported beside the final-step result.

All evaluations use the standard procedure (50 batches of 128 × 256 tokens, fixed evaluation
seed) on the validation split. M1 runs on the compute host beside the FineWeb-Edu arm, from a
separate directory, with a GPU-memory guard; it never touches the arm's code, files or results.

**Analysis.** `gain(run) = A - val_c`. Per budget and sweep: cell means and seed sds of `val_c`;
the §4 null criterion and Welch tests applied to them unchanged. The same on `A`.

**Falsifiers, fixed now.**
* The effect is **generic, not specific to low precision**, if at a budget
  `gain(b = 16) >= 0.5 * gain(b = 1.58)` (cell means).
* The primary argmin is **robust to averaging** at a budget if the `val_c` argmin equals the
  `val_loss_best` argmin and no adjacent gap reverses by more than the larger seed sd of the
  pair. Otherwise the paper says, in the abstract, that the argmin at that budget depends on
  end-of-training averaging.

### Argmin groups — which cells M2–M4 use, fixed now

After M1, at each budget of the primary sweep, with `m_c` and `s_c` the `val_c` cell means and
sds and `b*_c` their argmin, the **argmin group** is `b*_c`, its grid neighbours, and every `b`
with `m_c(b) - m_c(b*_c) <= max(0.03, 2 * max(s_c(b), s_c(b*_c)))`.

The 0.03-nat margin is an **assumption**, not a measurement: it rounds up the largest per-cell
shifts the remaining checks are expected to produce (factor-2 learning-rate error up to 0.021
nats from the literature; the M1 single-trajectory spread of up to 0.022 nats). A cell further
than that from the argmin is not expected to overtake it through M2–M4.

At a budget where every group member other than `b*_c` is more than 0.05 nats from it on
`m_c` (at least 10 seed sds there), M2 and M3 use seed 0 only; elsewhere seeds 0–2.

### M2 — learning-rate sensitivity

Every argmin-group cell, primary shape, learning-rate multiplier in {0.5, 2} applied to the
rule's value (multiplier 1 is the existing run of the same seed). If a cell's best multiplier
by `val_c` mean is 0.5 or 2, that cell is extended to 0.25 or 4 respectively, same seeds.
These runs are a **sensitivity arm under hard rule 5**: they carry their own run-id suffix and
results file and never enter the primary analysis. Every M2 run gets the M1 continuation.

**Falsifier:** with every cell at its own best multiplier (by `val_c`), the argmin changes or
an adjacent gap reverses by more than the larger seed sd of the pair. Then the paper says the
argmin at that budget depends on learning-rate tuning.

### M3 — half-budget residual check on TinyStories

Every argmin-group cell, primary shape, retrained at half its sweep token budget with a full
cosine schedule over that length, as in A6. `gain = val_c(T/2) - val_c(T)`, paired by seed.
At a budget, if the spread of gains across the group exceeds the nearest-neighbour gap of
`b*_c`, the ranking there is reported as **sensitive to training length**.

### M4 — seeds

Wherever, on `val_c`, the §4 criterion returns NULL at a budget or the Welch p between `b*_c`
and a group member exceeds 0.01, both cells get new seeds (5, 6, …) until the number simulated
for 80% power at the observed `val_c` gap and sds is reached, **capped at 10 seeds per cell**.
A comparison still unresolved at the cap is reported as unresolved at that `n`, with its
minimum detectable effect. New seeds are recorded in A7's own results file; the preregistered
5-seed analysis is not altered.

### Order and budget

M1 first (about 4 GPU-hours), then the argmin groups, then M2, M3 and M4 once the FineWeb-Edu
arm has finished (2026-10-02), with quiet hours kept. Estimated 76–112 GPU-hours for M2–M4
depending on the groups, finishing about 2026-10-07 to 10-09. All A7 outputs live in
`results/a7/`. The code for each step is committed, with tests, before that step runs.

### What A7 will not do

It does not change the primary metric, the primary verdicts or any prediction's score. It does
not tune the quantizer, the embedding precision, or the architecture search; those remain
limitations with the measured magnitudes in `docs/RELATED_WORK.md`, unless separately amended.

---

## 17. Note to Amendment A7, M1 (2026-09-27) — two details fixed before the first M1 run

Recorded after the M1 code was reviewed and **before any M1 run exists** on the compute host.
Neither changes what A7 measures; both make the text say exactly what the code does.

* **Where the (c) weights live, and which are read.** §16 names `runs/averaged/<run_id>.pt`.
  On the compute host `runs/` belongs to the main checkout, which trains the FineWeb-Edu arm
  and which M1 never writes. The (c) weights are therefore saved in M1's own output
  directory: `results/a7/averaged/<run_id>.pt` in the A7 directory (`~/bit-budget-a7`). A
  run's file is written only after its continuation (d) has also succeeded, and the §14 test
  read loads the (c) weights **only for runs whose `results/a7/m1.csv` status is
  `complete`**; any other file there is ignored.
* **Continuation (d)'s schedule.** §16 says 400 steps with the learning rate annealed
  linearly from `lr * min_lr_frac` to 0. Step `t = 0 … 399` uses
  `lr * min_lr_frac * (1 - t/400)`: 400 non-zero updates, the line reaching 0 at the end of
  the last step. The other reading (`1 - t/399`, the last step at exactly 0) would make only
  399 updates, because AdamW at learning rate 0 changes no weight — measured (the decoupled
  decay and the Adam update both scale by the learning rate) and pinned by
  `tests/test_a7_average.py`.
