# Results Summary

**Scale:** 5 datasets × 28 error types × 3 severities = **420 combinations**,
each trained on 8 models, giving **5,760 measurements** with zero failures.

The count is not 420 × 8 × 2, because only the 20 automated-cleaning error types
have a second (cleaned) condition; the 8 external-intervention types have no
cleaner to apply:

```
dirty      420 combinations × 8 models                = 3,360
cleaned    (20 types × 3 severities × 5 datasets) × 8 = 2,400
                                                        -----
                                                        5,760
```

**Status:** measurements final. Not yet reviewed by a supervisor.

| RQ | Question | Status |
|---|---|---|
| RQ1 | Which cleaning action gives the biggest improvement? | Answered — §2 |
| RQ2 | Which gives the best return versus its cost? | Answered — §3 |
| RQ3 | Does priority depend on the AI model? | Answered — §5 |
| RQ4 | Does priority depend on the dataset? | Answered, weakest — §6 |

---

## 1. Which errors cause the most damage (input to RQ1)

Relative accuracy drop, mean over 5 datasets × 8 models.

| Error type | low | medium | high | overall | Route |
|---|---|---|---|---|---|
| `annotator_bias` | 1.56 | 8.57 | 11.41 | **7.18** | external |
| `missing_values_mnar` | 3.60 | 7.00 | 9.02 | **6.54** | automated |
| `ambiguous_labels` | 1.97 | 5.11 | 7.74 | **4.94** | external |
| `label_noise_asymmetric` | 0.85 | 3.76 | 6.16 | **3.59** | external |
| `contextual_errors` | 3.35 | 3.06 | 4.27 | **3.56** | automated |
| `data_leakage` | 0.66 | 2.21 | 6.43 | **3.10** | automated |
| `invalid_values` | 2.63 | 2.42 | 2.82 | **2.62** | automated |
| `concept_drift` | 0.75 | 2.13 | 4.74 | 2.54 | external |
| `data_heterogeneity` | 1.06 | 2.15 | 4.03 | 2.41 | automated |
| `data_inconsistency` | 1.35 | 1.18 | 1.18 | 1.24 | automated |

Damage rises monotonically with severity for **7 of the 10** error types causing
at least 1pp. Three do not: `contextual_errors` (3.35 → 3.06 → 4.27),
`invalid_values` (2.63 → 2.42 → 2.82) and `data_inconsistency` (1.35 → 1.18 →
1.18) dip at medium severity before recovering, or plateau. All three are
value-corruption types whose damage depends on how far the corrupted values fall
outside the column's range, not only on how many cells are hit — so a higher
corruption rate does not translate directly into more damage. The label and
missingness types, where severity maps directly onto how much signal is destroyed,
are monotonic without exception.

Of the 28 types, **10 cause at least 1pp of damage and only 8 exceed 2.5pp**; the
remaining 18 sit near zero, and 3 are marginally negative (measurement noise around
zero, not a benefit from corruption). That concentration is itself a central input
to prioritisation — it identifies where remediation effort is *not* warranted.

### 1.1 Mechanism — the strongest finding

Seven pairs corrupt at an **identical rate**, differing only in placement.

| Pair | Random | Targeted | Difference | p |
|---|---|---|---|---|
| missing values | 0.36 (MCAR) | **6.54** (MNAR) | +6.18 | **<0.0001** |
| label noise | 0.96 (symmetric) | **3.59** (asymmetric) | +2.63 | **<0.0001** |
| duplicates | 0.05 (random) | **0.52** (minority class) | +0.46 | **0.0014** |
| duplicates | 0.05 (random) | 0.01 (majority class) | −0.04 | 0.74 |
| outliers | 0.38 (all features) | 0.62 (top predictors) | +0.24 | 0.22 |
| feature noise | 0.08 (all features) | 0.33 (top predictors) | +0.25 | 0.12 |
| text errors | −0.29 (typo) | −0.07 (category swap) | +0.23 | 0.19 |

**MNAR missingness causes 18× the damage of MCAR at an identical missing rate.**

The three significant results share one property: the mechanism ties the
corruption **to the target variable** — missingness that depends on the label,
one-directional flips that shift the class prior, duplication of the minority
class. The four null results merely relocate corruption among features or rows.

This is a narrower and more defensible claim than "mechanism always matters".

---

## 2. RQ1 — which cleaning action gives the biggest improvement

| Cleaning action | Damage | Net improvement | Eligible damage | Eligible improvement | Conditional recovery | Coverage |
|---|---|---|---|---|---|---|
| **KNN imputation (MNAR)** | 6.54 | **+4.32** | 8.55 | 5.85 | 68.4% | 79% |
| **IQR clipping (contextual)** | 3.56 | **+2.64** | 7.01 | 5.00 | 71.3% | 53% |
| **Sentinel detection** | 2.62 | **+2.58** | 6.68 | 6.12 | 91.6% | 43% |
| **Consistency rules** | 1.24 | **+1.21** | 4.33 | 3.62 | 83.5% | 38% |
| **Leakage removal** | 3.10 | **+0.92** | 4.81 | 2.93 | 60.8% | 68% |

**Five substantial interventions**, spanning +0.92 to +4.32pp improvement and
61–92% conditional recovery.

### Reading the table

The three quantities use **different row sets, by design**:

```
Damage_pp, Net_Improvement_pp    mean over ALL rows for the error type
Eligible_*                       mean over rows with damage ≥ 0.5pp only
Conditional_Recovery_%           Σ(improvement) / Σ(damage) over eligible rows
                                 = Eligible_Improvement_pp / Eligible_Damage_pp
Coverage_%                       share of rows that were eligible
```

So `5.85 / 8.55 = 68.4%` reproduces exactly, while `4.32 / 6.54 = 66%` does not
equal the reported figure. **This is not an inconsistency** — the first averages
over every row including undamaged ones, the second is computed only where damage
was material. Both denominators are published so any figure can be checked.

### The useful null: outlier cleaning

| Action | Damage | Improvement |
|---|---|---|
| `outliers` | 0.38pp | +0.15pp |
| `outliers_targeted` | 0.62pp | +0.06pp |

Outlier cleaning works correctly — it simply has almost nothing to recover. This
is a deliberate low-return comparison point, and it supports a conclusion the
effectiveness ranking alone cannot reach: **cleanability is not sufficient
justification for cleaning; expected recoverable damage matters too.**

---

## 3. RQ2 — cost-effectiveness and break-even

> **The intervention that maximised predictive recovery was not the intervention
> that maximised cost-effectiveness: KNN imputation recovered 4.32pp but required
> 14.5s, whereas IQR clipping and sentinel detection recovered 2.64pp and 2.58pp
> respectively at approximately 0.01s — an impact–efficiency trade-off that an
> effectiveness-only analysis would miss.**

| Action | Improvement | Cleaning cost | Cost band |
|---|---|---|---|
| IQR clipping (contextual) | +2.64pp | 0.012s | negligible |
| Sentinel detection | +2.58pp | 0.011s | negligible |
| Consistency rules | +1.21pp | 0.022s | negligible |
| Leakage removal | +0.92pp | 0.002s | negligible |
| **KNN imputation (MNAR)** | **+4.32pp** | **14.5s** | **substantial** |

KNN buys **the largest absolute improvement** — it is not "worse". It buys that
improvement at roughly three orders of magnitude more compute.

### Cost bands, not fine ordering

Measured timing noise makes precise ordering below 0.1s uninterpretable:

```
within-cleaner std          0.010s
spread across the fast group 0.002s – 0.027s
coefficient of variation     42% – 98%
```

The noise is the same size as the differences, so a pp-per-second ranking inside
that band would be decided by scheduler jitter. **Only across-band comparisons
are interpreted.** Within the negligible band, actions are ranked by improvement.

**Run-to-run reproducibility.** The cost table was measured twice on the same
machine. Cell-level medians moved by 0.0005s (5.9%) at the median, but the
substantial band was markedly less stable than the fast group:

```
negligible  n=273   median |change| 0.0005s   median |relative| 5.6%
substantial n= 27   median |change| 0.4509s   median |relative| 21.5%
largest mover: Bank Marketing missing_values_mnar high, 87.8s -> 67.9s (-22.6%)
```

No cell crossed the 0.1s boundary in either direction, and substantial-band
membership was identical across runs. So the **band assignment reproduces exactly
while the absolute seconds do not** — expensive operations vary by roughly 20%
run to run, which is a further reason the analysis is denominated in bands. Point
estimates such as "14.5s" should be read as order-of-magnitude, not precise.

### Robustness to the cost definition

```
primary       cleaning execution time only
sensitivity   cleaning time + intervention-induced change in training time

PRIMARY    cost-band agreement under the two definitions  = 0.800 (16/20 actions)
THRESHOLD  substantial-band membership at 0.05 / 0.10 / 0.20s = identical
SECONDARY  raw rank correlation between efficiency orderings, rho = 0.923
```

The three actions in the substantial band — `missing_values`,
`missing_values_mnar`, `typographical_errors` — are substantial at **every** tested
cutoff, so the identity of the expensive interventions does not depend on where the
line is drawn. The *agreement statistic itself* is cutoff-sensitive (0.50 at 0.05s,
0.80 at 0.10s, 0.90 at 0.20s), because adding the training delta pushes several
fast actions above a 0.05s line but not above a 0.20s one. The stable claim is
therefore about **which interventions are expensive**, not about the precise
agreement figure.

The economic ranking is **robust to the two tested definitions of marginal compute
cost**. That is the claim the experiment supports; it is not a claim of invariance
to every conceivable costing framework — a definition that priced developer time,
memory, or energy could order these actions differently.

The primary robustness statistic is **band-level agreement**, not the raw rank
correlation. Raw Spearman is computed over 20 actions, most of which sit inside the
negligible band where this document has already declared ordering uninterpretable;
letting those ranks dominate the robustness number would contradict the cost-band
argument. It is retained as a secondary diagnostic, with the caveat that
fast-action ranks are noise-sensitive.

### Break-even for external intervention

Human effort is not measurable here and no wage rate is assumed. Instead:

```
C*_human(ρ) = (ρ × ceiling) / E_benchmark
```

`E_benchmark` is **not** `max_j(improvement_j / cost_j)`. Taking the literal
maximum would select whichever action happened to record the smallest runtime,
and this document has already established that timings below 0.1s are
noise-dominated — leakage removal shows ~455 pp/s purely because its 0.002s
measurement sits at the bottom of the jitter band.

`E_benchmark` is instead the efficiency of the **highest-benefit action within
the negligible-cost band**: IQR clipping, at 2.64pp for 0.012s = **222.4 pp/s**.
That is a stable reference point rather than an artefact of timing noise, and the
choice is conservative **with respect to this study's own conclusion**. At
222.4 pp/s the benchmark is less than half the 454.7 pp/s literal maximum, and a
smaller denominator yields *larger* break-even thresholds — so this choice makes
human intervention easier to justify than `max_j` would. The finding that C*
remains small therefore survives the assumption that works against it.

**`E_benchmark` is an order-of-magnitude reference, not a precise efficiency
estimate.** Its runtime (0.012s) lies inside the noise-dominated negligible-cost
band, so the same argument that forbids ranking a 0.012s cleaner against a 0.024s
one also forbids treating 222.4 pp/s as exact. C\* values are therefore
**approximate compute-equivalent thresholds, not point estimates**, and the
trailing digits below should not be read as significant.

The analysis was repeated using **three independent timing runs** on the same
machine:

| | IQR clipping | `E_benchmark` | `annotator_bias` C\*(ρ=1.0) |
|---|---|---|---|
| run 1 | 0.0127s | 207.5 pp/s | 0.035 |
| run 2 | 0.0123s | 215.0 pp/s | 0.033 |
| run 3 | 0.0118s | 222.4 pp/s | 0.032 |

The denominator spans 207.5–222.4 pp/s (7% range) and every C\* stays below 0.04
compute-seconds. **The conclusion is unchanged under any of the three runs**,
which is the level of precision the claim actually needs: not that C\* equals
0.032, but that it is orders of magnitude below what human correction could
plausibly cost.

A fourth measurement, taken while a second cost job was running concurrently,
returned 0.0215s and 50.8s — inflated 1.8–3.5×. It is excluded as a contention
artefact rather than measurement variance, and is retained in
`frozen/` for inspection. Even under those figures every C\* stayed below 0.06
and no action changed cost band.

| Error type | Ceiling (pp) | C* at ρ=0.25 | ρ=0.50 | ρ=0.75 | ρ=1.00 |
|---|---|---|---|---|---|
| `annotator_bias` | 7.18 | 0.008 | 0.016 | 0.024 | 0.032 |
| `ambiguous_labels` | 4.94 | 0.006 | 0.011 | 0.017 | 0.022 |
| `label_noise_asymmetric` | 3.59 | 0.004 | 0.008 | 0.012 | 0.016 |
| `concept_drift` | 2.54 | 0.003 | 0.006 | 0.009 | 0.011 |

**C\* is a compute-equivalent break-even threshold**, not a claim about wages or
seconds of human labour. It states how cheap human correction would need to be,
in units of the best automated alternative, to match it.

`ceiling` is the **experimental maximum recoverable damage** under this simulation
design: restoring the injected ground-truth labels returns the data to the clean
condition, verified to machine precision. It is a ceiling *within the constructed
experiment*, where ground truth is known. It does not imply the same ceiling is
identifiable in a real dataset — which is precisely why ρ is a free parameter
rather than an assumed constant.

Because automated cleaning is close to free, the thresholds are extremely low.
The practical reading: **for these error types, human correction competes on
outcome quality, not on efficiency.**

### Economic regimes

| Regime | Actions |
|---|---|
| High return, negligible cost | IQR clipping, sentinel detection, consistency rules |
| High return, substantial cost | KNN imputation — largest absolute gain, ~1000× the cost |
| Moderate return | Leakage removal |
| Low return despite being cleanable | Outlier cleaning — little damage to recover |
| No automated route | 8 types requiring external intervention |

---

## 4. Class balancing — reported separately

Excluded from the accuracy-denominated ranking because its principal benefits
occur in recall and F1 despite a fall in accuracy, with AUC essentially unchanged.
It moves the decision threshold rather than degrading the model.

| Dataset | Positive class share | Accuracy | Recall | F1 |
|---|---|---|---|---|
| Bank Marketing | 11.7% | −6.83 | **+47.62** | **+12.35** |
| Adult Income | 24.9% | −3.15 | **+26.69** | +5.98 |
| Telco Churn | 26.6% | −3.95 | **+27.11** | +8.17 |
| Breast Cancer | 37.3% | +0.18 | +1.09 | +0.32 |
| German Credit | 70.0% | −2.90 | **−11.61** | −4.13 |

Recall gains from oversampling were **strongly associated with the initial
positive-class prevalence (r ≈ −0.93, n = 5)**. With only five datasets this is a
compelling exploratory relationship, not an established general law.

The observed directional pattern: oversampling raised recall substantially where
the positive class was the minority (11.7% → +47.6pp), had almost no effect near
balance (37.3% → +1.1pp), and **reduced** recall where the positive class was
already the majority (70.0% → −11.6pp).

The mechanism is straightforward and makes the direction unsurprising: when the
positive class is already the majority, oversampling the *minority* class means
adding negative examples, so a fall in positive-class recall is exactly what
should be expected. The useful conclusion is therefore not "more imbalance means
more benefit" but that **class-balancing effects depend on which class is
initially under-represented relative to the class being scored.**

---

## 5. RQ3 — priority agrees more within model families than across them

```
mean top-5 overlap    3.32 / 5
mean Spearman rho     0.605
```

| Comparison | rho |
|---|---|
| within tree-based models | **0.770** |
| within non-tree models | 0.674 |
| **across families** | **0.517** |

Cleaning priorities show **greater agreement within model families than across
model families, consistent with a family-level effect.** This is a descriptive
comparison of rank correlations over 8 models, not a causal test, so it is
reported as an association rather than a demonstrated dependency.

**Agreement is far from complete.** Only the two most damaging error types hold
their position across all eight models (`annotator_bias` ranks 1–4,
`missing_values_mnar` 1–5). Below those the ranking moves substantially:
`contextual_errors` ranges from 1st to 24th and `invalid_values` from 1st to 25th
depending on the model, and **19 of 28 error types shift by 10 places or more**.

The result therefore supports a **family-level pattern in how priorities cluster**.
It does not establish that knowing the model family is sufficient for
prioritisation — that would require comparing prediction from family against
prediction from the specific algorithm, which this design does not do.

---

## 6. RQ4 — partly dataset-dependent (weakest result)

```
mean top-5 overlap    2.70 / 5
mean Spearman rho     0.633
```

Verdict: partly dependent, report with caveats. This is the least secure of the
four answers.

Dataset size does **not** explain the disagreement. Mean agreement with the other
datasets against training-set size:

| Dataset | Train rows | Mean rho vs others |
|---|---|---|
| Breast Cancer | 455 | 0.549 |
| German Credit | 800 | 0.643 |
| Telco Churn | 5,625 | 0.732 |
| Adult Income | 24,129 | 0.691 |
| Bank Marketing | 36,168 | 0.551 |

`Spearman(rows, agreement) = 0.300` — weak and non-monotonic, with the largest
dataset showing among the lowest agreement. Small-sample noise is therefore not a
sufficient explanation, and the residual dependence is unexplained by any variable
measured here. With five datasets the question cannot be resolved further; it
would need substantially more datasets spanning more domains.

---

## 7. Limitations

1. **Runtime is hardware and environment dependent, and absolute seconds are not
   reproducible.** Re-measurement moved negligible-band cells 5.6% and
   substantial-band cells 21.5%, yet band membership reproduced exactly with zero
   boundary crossings. Cross-band comparisons are defensible; within-band
   differences and precise runtimes are not.

   The stable RQ2 finding is **not** the 0.800 agreement coefficient or the exact
   14.5s: it is that the genuinely expensive interventions remain consistently
   distinguishable from the negligible-cost group, and that this separation
   survives both tested marginal-cost definitions and every threshold between
   0.05s and 0.20s.
2. **ρ is a sensitivity parameter, not a measurement.** Real erroneous labels are
   not identifiable from ground truth outside a simulation.
3. **Class-imbalance effects vary with the initial class balance** (r ≈ −0.93,
   n = 5) and are reported per dataset; the pooled figure would mislead. With five
   datasets this is an exploratory association, not an established law.
4. **One cleaning method per error type.** Results describe the method tested, not
   the error type's ceiling.
5. **Text-level errors produced null results** (`categorical_errors` −0.07pp,
   `typographical_errors` −0.29pp). They genuinely inject — the audit confirms
   2.7–10.9% of categorical cells changed depending on severity (median 5.7%;
   Breast Cancer is excluded as it carries no text). The null effect is **consistent with** robustness
   introduced by the ordinal-encoding pipeline, in which swapping or misspelling a
   category maps it to a different arbitrary integer, **although an encoding
   ablation would be required to establish that mechanism.** A different encoding
   (one-hot, target encoding) is where this answer would most plausibly change.
6. **`data_heterogeneity`: the corruption does real damage, but the tested
   remediation is redundant.** Damage is 2.41pp and rises with severity
   (1.06 → 2.15 → 4.03), so the corruption is *not* neutralised. What is
   neutralised is the *within-training-fold* scale change: standardisation is
   fitted on the corrupted training data, so that fold returns to mean 0 / std 1.
   Because errors are injected into the training fold only, the test set stays on
   its original scale and the train-derived scaler mis-maps it — measured on
   Breast Cancer, the standardised test column moves to mean −2.01 (×2), −3.18
   (×5), −3.57 (×10) with its standard deviation collapsing from 0.92 to 0.09.
   That train/test mismatch is the damage.

   The 0.00 recovery is therefore a statement about the **remediation**, not the
   error: re-standardising features repeats what preprocessing already does, and
   under the no-oracle rule the cleaner cannot see that the test set is on a
   different scale. A remediation that aligned the two scales could plausibly
   recover this damage; the one tested here cannot.
7. **Accuracy is the primary metric**, and is the wrong one for interventions that
   shift the decision threshold — hence class balancing is separated.
8. **`external_intervention` is a statement about this protocol, not about what
   software can do.** Eight error types were routed to external intervention
   because no automated repair was evaluated for them here. That is a scope
   decision, and it should not be read as "software cannot fix these". Automated
   approaches exist in the literature for several — confident-learning methods for
   label noise, drift detectors for concept and data drift, reweighting for
   representativeness — and evaluating them under this cost framework is a natural
   extension. What this study establishes is the damage those errors cause and the
   compute-equivalent budget a human correction would have to beat, not that
   automation is impossible.
9. **Two error types were excluded as design-inapplicable** (temporal ordering,
   missing metadata); see METHODOLOGY.md.

---

## 8. Headline findings

1. **Target-linked corruption mechanisms can dominate corruption rate alone.**
   How corruption is distributed can matter more than how much is present, and in
   this study it did so specifically where the mechanism was target-dependent:
   MNAR missingness caused 18× the damage of MCAR at an identical rate. Three of
   seven matched pairs were significant, and all three tie corruption to the
   target; the four that merely relocate corruption among features or rows were
   null. The broader claim "mechanism always matters" is *not* supported.
2. **The most effective intervention is not the most efficient one.** KNN recovers
   the most (+4.32pp) at ~1000× the compute of IQR clipping (+2.64pp).
3. **Cleanability does not justify cleaning.** Outlier cleaning works and is
   nearly worthless, because there is little damage to recover.
4. **The most damaging error is not automatically repairable.** `annotator_bias`
   causes the largest loss (7.18pp) and no automated remedy was evaluated for
   it under this protocol.
5. **Cleaning priority agrees more within model families than across them**
   (ρ 0.770 / 0.674 within, 0.517 across) — consistent with a family-level effect.
6. **Some interventions optimise a different objective entirely** — class
   balancing trades accuracy for recall, and in this sample the size and direction
   of that trade tracked the initial class balance.
