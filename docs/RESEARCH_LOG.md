# Research Log

**Project:** Prioritizing Data Quality Improvements for Machine Learning —
A Cost–Benefit Framework
**Rebuild started:** 9 August 2026

---

## Authoritative figures

Timing-derived numbers were remeasured three times; **run 4 of 10 August is
authoritative**. Earlier values appear in this log as history and are marked as
such where they occur.

| Quantity | Value |
|---|---|
| KNN imputation (MNAR) cost | **14.5s** |
| `E_benchmark` | **222.4 pp/s** |
| `annotator_bias` C*(ρ=1.0) | **0.032** |
| Rank correlation between cost definitions | **0.923** |

Everything not derived from timing — damage, recovery, mechanism, RQ3, RQ4, class
balancing — comes from `results/experiment_results.csv`, which has not changed
since the experiment run.

---

## Steps performed, and the result of each

The eleven planned workflow steps, with what each produced. Every figure here is
reproduced in `RESULTS_SUMMARY.md` and traceable to a file in `results/`.

| # | Step | What was done | Result |
|---|---|---|---|
| 1 | Literature review | Error taxonomy assembled | 28 error types in scope: 22 base + 6 mechanism variants |
| 2 | Select datasets | 7 candidates screened | **5 kept**, 2 rejected: Blogger (AUC 0.559, no signal to lose) and Credit Card Fraud (0.17% minority makes accuracy uninformative) |
| 3 | Train baseline models | 8 models, 4 tree / 4 non-tree | All baselines pass the AUC ≥ 0.65 preflight; Telco 0.806 |
| 4 | Inject one error at a time | 28 types × 3 severities × 5 datasets, injected into **raw** data before encoding | 420 combinations, seeded per experiment (SHA-256); audit: 0 inert, 0 broken |
| 5 | Measure degradation | 8 models per combination | **3,360 dirty measurements.** Worst: `annotator_bias` 7.18pp, `missing_values_mnar` 6.54pp. Only 10 of 28 exceed 1pp |
| 6 | Clean data | 20 cleaners, no oracle information | 8 types have no automated route and are measured for damage only |
| 7 | Retrain | Same 8 models on cleaned data | **2,400 cleaned measurements** → 5,760 total, 0 failures |
| 8 | Measure recovery (RQ1) | Improvement and conditional recovery | **5 substantial interventions**, +0.92 to +4.32pp, 61–92% conditional recovery. Best: KNN imputation +4.32pp |
| 9 | Estimate cost | Every cleaner timed, median of 3 repeats, then **re-measured entirely** | 300 cost observations. Band assignment reproduced exactly; absolute seconds moved 5.6% (fast) / 21.5% (slow) |
| 10 | Compute ROI (RQ2) | Efficiency, cost bands, break-even, sensitivity | **Effectiveness ≠ efficiency:** KNN +4.32pp at ~14.5s vs IQR clipping +2.64pp at ~0.01s. Break-even C\*(ρ=1.0) = 0.032 compute-seconds for `annotator_bias` |
| 11 | Prepare recommendations | Two orderings produced | Damage order vs recommended action order — they disagree, which is the study's practical contribution |

**Additional analyses beyond the planned steps**

| Analysis | Result |
|---|---|
| Mechanism comparison (7 matched pairs) | 3 significant, all target-linked. MNAR does **18× the damage of MCAR at an identical missing rate** (p < 0.0001) |
| RQ3 — model dependency | Agreement within families (ρ 0.770 tree, 0.674 non-tree) exceeds across families (0.517) |
| RQ4 — dataset dependency | Mean ρ 0.633. Dataset-size explanation **tested and rejected** (Spearman 0.300) |
| Cost-definition robustness | Band-level agreement 0.800 (16/20); expensive set identical at 0.05/0.10/0.20s cutoffs |
| Class balancing | Separated from the accuracy ranking: Bank Marketing accuracy −6.83pp but recall +47.62pp |

---

## Why this is a rebuild

A previous version of this study ran to completion and produced 1,800 results.
Those results were not trustworthy, and the reasons are worth recording because
they shaped every design decision here.

The earlier pipeline encoded categorical columns to integers **before** injecting
errors. Two error types — categorical errors and typographical errors — act on
text, so they found nothing to corrupt. One injected nothing at all and reported
0.00% damage; the other perturbed integers and reported that as "typos". Neither
was a finding. Both were the absence of an experiment.

Alongside that, an audit of the original injectors found three that ignored the
severity parameter entirely (low, medium and high produced byte-identical data),
one producing values of 1e152 through a compounding loop, and a shared random
seed that made every error type's results depend on how many random numbers the
others had consumed.

The decisive lesson was procedural rather than technical: **the earlier study ran
three full experiments before validating its instruments.** This rebuild runs the
audits first.

---

## Pipeline change

```
BEFORE   load → encode → scale → split → inject → train
AFTER    load → split → inject → encode → scale → train
```

Two consequences, both intended:

Text-level error types can now act on genuine text. German Credit carries 13
categorical columns, Telco 15, Bank Marketing 9, Adult Income 8 — all previously
destroyed by the loader before injection.

The encoder and scaler are now fitted on the **corrupted** training fold, as they
would be in production. Previously they saw only clean data, so injected outliers
never influenced the scaling the way real ones do.

---

## Dataset corrections

**The file labelled "NASA CM1" is not NASA CM1.** Its ARFF metadata identifies it
as OpenML `blogger` — 100 instances, five nominal attributes about blogging
behaviour (high/low/medium, left/middle/right, impression/news/political/
scientific/tourism). The previous methodology described it as software defect
prediction and claimed cross-domain software-quality validation. That claim was
unsupported.

**Blogger was then dropped entirely.** Baseline AUC 0.559 against 0.50 for random
guessing, F1 0.328, 20 test rows. Degradation cannot be measured on a model with
no signal to lose — any damage figure it produced would be noise presented as
measurement. A minimum baseline AUC of 0.65 is now a pre-flight check.

**Telco Customer Churn replaced it.** 7,032 rows, 15 of 19 features categorical,
26.6% churn, baseline AUC 0.806. It gives the text-level error types somewhere
substantial to act, with genuine predictive signal.

**Credit Card Fraud was considered and rejected** — a 0.17% minority class makes
accuracy uninformative.

---

## Error types removed

Two were removed as **design-inapplicable**, distinct from "tested and harmless":

**Temporal ordering errors.** All eight model families treat the training set as
an unordered bag of rows, so shuffling row order is invisible to them. This is not
a dataset limitation — adding a timestamped dataset would not fix it. Measuring it
requires sequence models, or order-derived features (lags, rolling windows)
combined with temporal train/test splitting: three coupled changes amounting to a
separate study.

**Missing metadata.** Column names never reach the model, so no accuracy is
recoverable. Its real cost is developer time spent decoding undocumented fields,
which is a human-subjects measurement.

Both had previously reported ≈0.00pp. Reporting those zeros next to genuine null
results would imply they were tested and found harmless.

Final scope: **28 error types**, 20 with automated cleaning implemented, 8
requiring external intervention.

---

## Bugs found and fixed

Every one of these was in code written for this rebuild. They are listed because
the pattern matters more than the individual fixes: **each produced a plausible
number rather than an error**, and each was found by checking results against
what theory predicted.

### Found by the pre-run audits

**Integer dtype writes.** Numeric injectors wrote floats into integer columns,
which pandas refuses. Never surfaced before because the old pipeline standardised
everything to float first. Fixed by casting numeric columns to float at load.

**`std` on text columns.** Injectors computing statistics across all columns
raised on string data once text survived to the injection stage.

**Division by zero.** Injectors computing rates over numeric cells failed on a
fully-categorical dataset, where that count is zero.

**Audit false positives.** The audit's own metric measured the *fraction* of cells
changed, which saturates at 100% for Gaussian noise at every severity, and could
not see column renames at all. Three injectors were flagged as broken that were
working correctly. Fixed by measuring magnitude and renames as separate signals.

### Found by inspecting results

**Affine cleaning.** `clean_feature_noise` computed `0.7·x + 0.3·median` — an
affine transform, which the pipeline's standardisation undoes exactly. It was
mathematically incapable of improving anything on any dataset, and would have
produced a confident, meaningless zero. Replaced with rank-order median smoothing.

**Winsorisation defeated by its own corruption.** Clipping to the 1st/99th
percentile cannot remove corruption that constitutes 5–15% of the data, because
the injected values *become* those percentiles. On German Credit the 1%/99%
bounds were exactly the injected −30 and 106 while the true range was 4 to 72.
Fixed for `domain_violations` first, then — after the same failure appeared in the
results for `outliers`, `outliers_targeted` and `contextual_errors` — for all four.
`contextual_errors` recovery went from 0.10pp to 2.64pp.

**Train/test transform mismatch.** Cleaners received only the training set. Any
cleaner fitting a transform applied it to training data while the test set kept
its original values; preprocessing then derived its scaling from the transformed
training data and applied it to untransformed test data. Measured directly: train
column at mean 0.000/std 1.000, test at **mean 21.205/std 11.694**. Two cleaners
"recovered" −34pp and −31pp. Fixed by passing the test frame through the same
fitted transform.

**Blind sentinel replacement.** Bank Marketing encodes "never previously
contacted" as `pdays = −1`. The cleaner replaced every −1 as a sentinel, costing
**12.8 percentage points** — it destroyed more real information than the
corruption did. Now a value is only treated as a sentinel when it falls outside
`[Q1 − 3·IQR, Q3 + 3·IQR]` for its own column.

### The audit gap that let three of these through

`audit_pipeline_overlap.py` was written specifically to catch preprocessing
interactions, and compared only **training** matrices. The train/test mismatch is
by definition invisible to that comparison. The audit now checks that standardised
test columns sit near mean 0 / std 1 after cleaning, with thresholds calibrated
against observed behaviour: the real bug produced a test mean of 21.2, while
benign small-dataset variation reaches about 3.5 with no measurable accuracy
effect, so the threshold sits at 10.

---

## Findings that are not bugs

Three results look like defects and are not. Each is recorded so they are not
"fixed" later by someone reading the numbers cold.

**`data_heterogeneity` shows real damage (2.41pp) but 0.00 recovery.** An earlier
draft of this log recorded it as "≈0 damage", which contradicted the measured
result and has been corrected.

The injector multiplies whole columns by a constant. Standardisation is fitted on
the corrupted training fold, so within that fold the constant factor is removed
exactly. But errors are injected into the **training fold only**, so the test set
retains its original scale and the train-derived scaler mis-maps it. Measured
directly on Breast Cancer, the standardised test column moves to mean −2.01 at ×2,
−3.18 at ×5 and −3.57 at ×10, with its standard deviation collapsing from 0.92 to
0.09. That mismatch is the damage, and it explains why damage rises monotonically
with severity (1.06 → 2.15 → 4.03).

The 0.00 recovery is a property of the **remediation**: re-standardising features
repeats what preprocessing already does, and the no-oracle rule prevents the
cleaner from seeing that the test set sits on a different scale. A repair that
aligned the two scales might recover this damage; the one tested cannot.

The previous study reported 3.00pp for this error type from a pipeline that fitted
the scaler on clean data before injection — a different quantity again.

**`class_imbalance` cleaning reduces accuracy.** Oversampling moves the decision
threshold rather than degrading the model: on Bank Marketing, accuracy −6.43pp
against recall **+51.64pp** and F1 **+17.21pp**, with AUC essentially unchanged
(0.870 → 0.865). Accuracy is simply the wrong metric for this intervention, so it
is reported separately rather than ranked at the bottom of an accuracy-denominated
table.

> *Provenance:* the figures above are from the intermediate run that prompted this
> interpretation. The final regenerated values are accuracy **−6.83pp**, recall
> **+47.62pp**, F1 **+12.35pp**, AUC 0.873 → 0.870, and are reported in
> `RESULTS_SUMMARY.md`. The magnitudes shifted; the interpretation did not. Report
> the summary figures, not these.

**The oracle ceiling experiment was circular and was abandoned.** Restoring the
true labels *is* the clean data, so a "perfect correction" run reproduces the clean
baseline exactly — verified at 0.725000 = 0.725000, ceiling 4.44pp = damage
4.44pp. The ceiling equals the damage by mathematical identity, and running it
would have reprinted the damage column. The break-even analysis therefore treats
human effectiveness ρ as a free parameter instead.

---

## Validation performed

| Check | Result |
|---|---|
| Injector audit — 28 types × 5 datasets × 3 severities | 27/28 auto-passed, 0 inert, 0 broken; the 28th (`data_leakage`) was manually verified working — see next row, so effectively 28/28 |
| `data_leakage` flagged as non-scaling | verified working: severity scales correlation 0.28 → 0.60 → 0.90; the audit metric counts cells, not fidelity |
| Pipeline overlap — corruption and cleaning survive preprocessing | pass |
| Train/test scale alignment after cleaning | pass, 0 mismatches |
| Pre-flight — 24 checks | all pass |
| Experiment run | 420/420 combinations, 5,760 rows, 0 failures |
| Targeted rerun isolation | 24 untouched error types bit-for-bit identical |
| Timing reliability | fine ordering below 0.1s is noise (std 0.010s vs 0.002–0.027s spread); cost reported in bands |
| Cost re-measurement (2 runs, same machine) | band assignment identical, 0 cells crossed 0.1s; absolute seconds moved 5.6% (negligible band) and 21.5% (substantial band) |
| Cost-definition robustness | band-level agreement 0.800 (16/20); substantial-band membership identical at 0.05/0.10/0.20s cutoffs |

---

## Items raised and how they were settled

None of these remain open; they are kept because the resolution is part of the
record.

- ~~Cost CSV regeneration~~ **done.** Regenerated 300 rows; the `feature_noise`
  label now reads `rank-order median smoothing` on all rows. Re-measurement moved
  the headline cost figures slightly (KNN 17.7s → 15.1s, E_benchmark 207.5 →
  215.0 pp/s, rho 0.926 → 0.923) without changing any band assignment or the
  economic conclusion. The pre-rerun CSV was preserved to separate runtime
  reproducibility from cost-definition robustness.
  *(Historical: those figures were superseded on 10 August — see the final entry.
  Authoritative values are KNN 14.5s, `E_benchmark` 222.4 pp/s.)*
- **RQ4 remains the weakest question.** An earlier draft attributed the
  disagreement to dataset *size*. That was carried over from the previous study,
  where the 100-row Blogger dataset anchored the relationship. With Blogger
  dropped it does not hold: `Spearman(train rows, mean agreement) = 0.300` across
  the five datasets, non-monotonic, with the largest dataset (Bank Marketing,
  36,168 rows) showing among the lowest agreement (0.551). Size is therefore not
  a sufficient explanation and the claim has been removed. The residual dependence
  is unexplained by any variable measured here, and resolving it needs more
  datasets spanning more domains.
- **Text-level errors produced null results.** They now genuinely inject — the
  audit confirms 2.7–10.9% of categorical cells changed depending on severity
  (median 5.7%, measured on the four datasets that carry text). The null result is **consistent
  with** robustness induced by the ordinal-encoding pipeline; **an encoding
  ablation would be required to establish that mechanism.** What is established is
  the observation (text corruption → near-zero performance effect), not the
  explanation. This is the case where a different encoding (one-hot, target
  encoding) would most plausibly change the answer.

---

**Status at the time of the entries above:** RQ1, RQ3 and RQ4 measured. RQ2 cost
model built and populated. Next: figures and write-up.

---

## Final entry — 9 August 2026

RQ2 was completed: cost bands, the break-even model with `E_benchmark` replacing
the jitter-driven `max_j` denominator, and the sensitivity analysis over the two
marginal-cost definitions.

The cost table was then regenerated (the `feature_noise` label had been corrected
after the first measurement) and a stability audit run against the preserved
earlier run. Band assignment reproduced exactly with zero boundary crossings, and
the expensive-intervention set was identical at 0.05s, 0.10s and 0.20s cutoffs,
while absolute seconds moved 21.5% in the substantial band. Headline cost figures
shifted slightly (KNN 17.7s → 15.1s, `E_benchmark` 207.5 → 215.0 pp/s, ρ 0.926 →
0.923) without changing any band or conclusion.

The RQ4 dataset-size explanation was recomputed and **rejected** — see Open items.

**Status at the close of 9 August 2026:** RQ1–RQ4 measured and answered, RQ2 cost
model and break-even complete, frozen. *Superseded by the entry below — the cost
figures quoted above (KNN 15.1s, `E_benchmark` 215.0 pp/s) were remeasured on
10 August.*

---

## Final entry — 10 August 2026

**A contaminated cost measurement was detected and corrected.**
`results/cleaning_costs.csv` was overwritten after the freeze by a cost job left
running from an earlier session; its timings were inflated 1.8–3.5× by CPU
contention. The checksum manifest detected the change — the documents did not
silently follow it.

A fourth measurement was taken with nothing else running. Three clean runs agree
and the contaminated one is the clear outlier:

```
                 run1     run2     run3 (contaminated)   run4
IQR clipping    0.0127   0.0123        0.0215          0.0118
KNN (MNAR)      17.73    15.08         50.80           14.51
```

Substantial-band membership was identical in all four, including the contaminated
one. **Authoritative figures are now run 4:** KNN 14.5s, `E_benchmark`
222.4 pp/s, `annotator_bias` C*(ρ=1.0) 0.032. No conclusion changed, and the
study now reports three independent timing runs rather than two.

**Four further corrections**, all found by rechecking claims against the tables
beside them:

1. *"Damage rises monotonically with severity for every substantial error type"* —
   contradicted by its own table. Only 7 of 10; `contextual_errors`,
   `invalid_values` and `data_inconsistency` dip at medium severity.
2. *"The audit confirms 5%+ of categorical cells changed"* — true only at medium
   and high severity; at low it is 2.7–3.9%. Corrected to the measured range.
3. *"preflight.py — 20 checks"* — it runs 24.
4. **`data_heterogeneity` was described as showing "≈0 damage"** while the results
   table reported 2.41pp rising with severity. The measured damage is correct: the
   injector scales whole columns, standardisation removes that factor *within the
   training fold*, but injection is train-only, so the test set keeps its original
   scale and the train-derived scaler mis-maps it (test column mean −2.01 at ×2,
   −3.18 at ×5, −3.57 at ×10). That mismatch is the damage. The 0.00 recovery is a
   property of the **remediation**, which repeats work preprocessing already does.

One code fix: `data_cleaner.py` silently defaulted unrouted error types to
`prevention_only` despite `METHODOLOGY.md` promising no default fallback. It never
fired, but a silent default of exactly that kind previously turned "no cleaner was
written" into a published claim. It now raises `KeyError`.

**Status: RQ1–RQ4 measured and answered. Analysis closed. Every figure in these
documents is checked by `verify_documents.py` (193 checks) against `results/*.csv`.**
