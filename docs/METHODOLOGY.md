# Methodology

**Status:** protocol fixed 2026-08-09, before the recovery results were observed.
Not yet reviewed by a supervisor.

## Research Questions

**RQ1** Which cleaning action gives the biggest improvement?
**RQ2** Which cleaning action provides the best return compared with its cost?
**RQ3** Does the best priority depend on the AI model?
**RQ4** Does the best priority depend on the dataset?

Degradation measurement is an **input** to RQ1, not RQ1 itself: the unit of
analysis is the cleaning action and its improvement, not the error and its damage.

---

## Datasets

| Dataset | Domain | Rows | Features | Categorical | Numeric | Class 1 |
|---|---|---|---|---|---|---|
| Breast Cancer Wisconsin | Healthcare | 569 | 30 | 0 | 30 | 37.3% |
| German Credit | Finance | 1,000 | 20 | 13 | 7 | 70.0% |
| Adult Income | Census | 30,162 | 14 | 8 | 6 | 24.9% |
| Bank Marketing | Marketing | 45,211 | 16 | 9 | 7 | 11.7% |
| Telco Customer Churn | Telecom | 7,032 | 19 | 15 | 4 | 26.6% |

Selected for binary targets, public availability, domain diversity, and a mix of
categorical and numeric features. Breast Cancer is fully numeric, which isolates
numeric error types from any text confound; Telco is majority-categorical, which
gives the text-level error types somewhere substantial to act.

**Two datasets were excluded during the study.**

*Blogger (OpenML, 100 rows).* Dropped because the models could not learn from it:
baseline AUC 0.559 against 0.50 for random guessing, F1 0.328, and 20 test rows
where a single flipped prediction moves accuracy by 5 percentage points.
Degradation cannot be measured on a model with no signal to lose. A minimum
baseline AUC of 0.65 is now enforced as a pre-flight check.

*Credit Card Fraud.* Considered and rejected: a 0.17% minority class makes
accuracy uninformative, since predicting the majority class scores 99.83%.

---

## Models

Eight, balanced four tree-based against four non-tree, so RQ3 can compare
agreement **within** a model family against agreement **across** families, rather
than only across individual algorithms. This is a descriptive comparison of rank
correlations, not a causal test of family membership.

| Non-tree | Tree-based |
|---|---|
| Logistic Regression (`max_iter=1000`) | Decision Tree (`max_depth=10`) |
| Gaussian Naive Bayes | Random Forest (`n_estimators=100, max_depth=10`) |
| KNN (`n_neighbors=5`) | XGBoost (`n_estimators=100, max_depth=6, lr=0.1`) |
| MLP (`hidden_layer_sizes=(64,), max_iter=200`) | LightGBM (`n_estimators=100, max_depth=6, lr=0.1`) |

All expose `predict_proba`, which the AUC calculation requires. Models without it
were deliberately excluded rather than changing the metric code.

Every model is built from a single shared factory (`src/models.py`) imported by
both the baseline and experiment scripts, so the two definitions cannot diverge.

---

## Pipeline

Two paths are run for every combination, identical except for the cleaning step.
This is the experimental comparison:

```
DIRTY       load → split → INJECT →         encode → impute → scale → train
REMEDIATED  load → split → INJECT → CLEAN → encode → impute → scale → train
```

Cleaning sits **before** encoding for the same reason injection does: fuzzy-matching
a typo back to a known category is impossible once the column is an integer. The
8 external-intervention types have no CLEAN step, which is why they have no
remediated condition.

Errors are injected into **raw** data, before any preprocessing. This has two
consequences that matter:

**Text-level error types can act.** Encoding categoricals to integers before
injection leaves typos and category errors with nothing to corrupt.

**Preprocessing observes the corruption.** The encoder and scaler are fitted on
the corrupted training fold, exactly as they would be in production where nobody
knows the data is dirty. Fitting them on clean data beforehand means an injected
outlier never widens the standard deviation the way a real one does.

Preprocessing is never fitted on the test set. Unseen categories — which typos
produce by construction — map to a reserved code.

### Reproducibility

Each injection is seeded per experiment:

```python
seed = SHA256(random_state | error_type | severity | X.shape)
```

Seeding once per session makes every injection draw from one shared stream, so
each result depends on how many random numbers earlier injections consumed —
changing any one injector silently changes the data produced for all the others.
SHA-256 rather than `hash()`, whose string hashing is randomised per process.

**Verified:** a targeted rerun of four error types left the other 24 bit-for-bit
identical.

---

## Experimental Design

```
5 datasets × 28 error types × 3 severities = 420 dirty-data combinations
```

Each combination is evaluated across 8 models, giving **3,360** dirty
measurements. For the 20 error types with an implemented automated cleaner, a
second cleaned condition is evaluated:

```
dirty     420 combinations × 8 models                = 3,360
cleaned   (20 types × 3 severities × 5 datasets) × 8 = 2,400
                                                       -----
total                                                  5,760
```

The total is **not** 420 × 8 × 2 = 6,720: the 8 external-intervention types have
no cleaner, so they have no cleaned condition.

28 error types: 22 base types plus 6 targeted mechanism variants.

### Mechanism variants

Seven pairs corrupt the same data at the **same rate**, differing only in where
the corruption lands, which isolates mechanism from error type:

| Error type | Random | Targeted |
|---|---|---|
| `missing_values` | MCAR — uniform | MNAR — depends on the target |
| `label_noise` | symmetric | one-directional |
| `duplicates` | uniform | majority class / minority class |
| `outliers` | all features | top predictors |
| `feature_noise` | all features | top predictors, noise energy matched |
| text errors | typo (invalid value) | category swap (valid but wrong) |

### Excluded error types

Two were removed as **design-inapplicable** rather than merely harmless:

*Temporal ordering errors* — every model family here treats the training set as
an unordered bag of rows, so shuffling row order is invisible to them. Measuring
this would require sequence models, or order-derived features with temporal
splitting: a separate study, not a timestamped dataset.

*Missing metadata* — column names never reach the model, so there is no accuracy
to lose or recover. Its cost is developer time, which is a human-subjects
measurement rather than an ML experiment.

Both previously reported ≈0.00pp. Reporting those zeros alongside genuine null
results would imply they were tested and found harmless.

---

## Remediation

### Taxonomy

Two independent fields, deliberately separated:

**`Remediation_Route`** — what the correct remedy *is*, assigned by explicit
judgement for all 28 error types with no default fallback:

- `automated_cleaning` — repairable by an automated pipeline (20 types)
- `external_intervention` — repairable only via human effort or new data (8 types)

`external_intervention` does **not** mean impossible, and it is a statement about
**this protocol** rather than about software in general. These errors are repaired
routinely in practice, and automated approaches exist in the literature for several
of them — confident-learning methods for label noise, drift detectors for concept
and data drift, reweighting for representativeness. None was evaluated here, so the
route records that the remedy in scope is human effort or new data, carrying a
different kind of cost. Claims in this study are therefore about what was measured,
not about what is achievable.

**`Recovery_Evaluation_Status`** — whether this study *measured* recovery. An
error type with an automated route but no implemented cleaner yields **NaN, never
0**, so it is not penalised in the ROI ranking for a gap in implementation.

### Protocol rules, fixed in advance

1. **No oracle information.** Cleaners never see the uncorrupted data or which
   rows were corrupted. They operate on the corrupted training fold only.
2. **Cleaning happens on raw data**, before encoding — fuzzy-matching a typo back
   to a known category is impossible once the column is an integer.
3. **Fitted transforms apply to the test set** with the same training-derived
   parameters. Transforming training data alone leaves the two frames on
   different scales.
4. **Leakage detection threshold `|corr| > 0.5`**, fixed before results were
   observed and never tuned against them.

### Metric definitions

```
Damage_pp       = (clean accuracy − dirty accuracy) × 100      mean over ALL rows
Improvement_pp  = (cleaned accuracy − dirty accuracy) × 100    mean over ALL rows

Eligible rows   = rows with Damage_pp ≥ 0.5

Conditional_Recovery_% = Σ(improvement) / Σ(damage) over ELIGIBLE rows only
                       = Eligible_Improvement_pp / Eligible_Damage_pp
```

**`Net_Improvement_pp ÷ Damage_pp` does not equal `Conditional_Recovery_%`,** and
this is by design: the first two average over every row including undamaged ones,
while the ratio is computed only where damage was material. Both denominators are
reported so any figure can be reproduced.

The 0.5pp threshold is an **operational** rule for metric stability, not a
significance test. Below it the denominator is too small for a relative estimate
to mean anything. Absolute values are always retained.

Per-row recovery ratios are never averaged. Where corrupted data outperformed the
clean baseline the denominator is negative and the ratio flips sign, so failure
would read as success. Aggregates sum before dividing.

---

## Cost model (RQ2)

Two regimes, priced differently because their costs differ in kind.

**Automated cleaning** — cost is measurable:

```
efficiency_j = improvement_j / cleaning_seconds_j        [pp per second]
```

Primary cost is cleaning execution time. A sensitivity variant adds the
intervention-induced change in training time, since remediations that resize the
training set genuinely raise it:

```
C_j = T_clean,j + max(0, T_train_after,j − T_train_without,j)
```

**Cost bands.** Measured timing noise (within-cleaner std 0.010s against a
0.002–0.027s spread across the fast group) makes fine-grained ordering below
0.1s uninterpretable. Cost is therefore reported in order-of-magnitude bands, and
only across-band comparisons are treated as meaningful.

The 0.1s cutoff is an **operational threshold**, not a natural divide in the data.
Band membership is therefore re-derived at 0.05s, 0.10s and 0.20s
(`audit_cost_stability.py`); the reported conclusion is the one that holds across
all three. Robustness to the cost definition is assessed primarily as **band-level
agreement** between the two definitions, with the raw rank correlation retained as
a secondary diagnostic, because the raw correlation is dominated by the fast group
whose internal ordering is noise.

**External intervention** — cost is human effort, which this study cannot measure
and does not invent a rate for. The question is inverted:

```
C*_human(ρ) = (ρ × ceiling) / E_benchmark
```

`E_benchmark` is the efficiency of the **highest-benefit automated action within
the negligible-cost band** (IQR clipping, 2.64pp / 0.012s = 222.4 pp/s). It is
deliberately not `max_j(improvement_j / cost_j)`: that maximum would select
whichever action recorded the smallest runtime, and timings below 0.1s are
noise-dominated, so it would be set by scheduler jitter rather than by efficiency.
The choice is conservative **with respect to this study's own conclusion**: at
222.4 pp/s the benchmark is less than half the 454.7 pp/s literal maximum, and a
smaller denominator yields *larger* break-even thresholds, so it makes human
intervention easier to justify than `max_j` would.

`E_benchmark` is an **order-of-magnitude reference, not a precise efficiency
estimate** — its runtime lies inside the band this study declares noise-dominated,
so the same argument that forbids ranking a 0.012s cleaner against a 0.024s one
also forbids treating 222.4 pp/s as exact. C* values are approximate
compute-equivalent thresholds, not point estimates. The break-even was recomputed
under **three** independent timing runs (`E_benchmark` 207.5 / 215.0 / 222.4 pp/s,
a 7% spread; `annotator_bias` C*(ρ=1.0) 0.035 / 0.033 / 0.032, every C* below
0.04 under all three), and the conclusion is unchanged by which run is used. A
fourth measurement taken while another job ran concurrently is excluded as a
CPU-contention artefact, not measurement variance; even under those inflated
figures no action changed cost band.

`ceiling` is the **experimental maximum recoverable damage** — the oracle ceiling
under this simulation design. For label corruptions it equals the measured
damage, because restoring the injected ground-truth labels returns the data to
the clean condition, verified to machine precision. It is a ceiling *within the
constructed experiment*, where ground truth is known by construction; it does not
imply the same ceiling is identifiable in a real dataset.

`ρ ∈ (0,1]` is re-annotation effectiveness, left as a free parameter and reported
as a curve rather than assumed. C* is a **compute-equivalent break-even
threshold**, not a claim about wages or seconds of human labour.

### Metric-mismatched interventions

Class-balancing is excluded from the accuracy-denominated ranking. Its benefits
appear in recall and F1 while accuracy falls and AUC is essentially unchanged —
it moves the decision threshold rather than degrading the model. Ranking it by
accuracy would be mathematically correct and substantively misleading.

---

## Validation

Three audits run **before** the study, not after:

**`audit_injectors.py`** — every injector on every dataset at every severity:
does it change the data, does the corruption scale with severity, is the output
still trainable. Signals include cell changes, label changes, magnitude shift and
structural changes, because no single measure covers every injector.

**`audit_pipeline_overlap.py`** — does the corruption survive preprocessing, does
the cleaning survive it, and are train and test left on the same scale. The last
check exists because a cleaner fitting a transform on training data alone is
invisible to a training-only comparison.

**`preflight.py`** — 24 checks covering the model set, dataset integrity,
injector/route/cleaner consistency, baseline quality and stale artefacts. Exits
non-zero if any fails.

---

## Limitations

1. **Runtime is hardware and environment dependent.** Cross-band cost comparisons
   are defensible; small within-band differences are not. Re-measuring the full
   cost table on the same machine moved negligible-band cells by 5.6% and
   substantial-band cells by 21.5% (largest single move 87.8s → 67.9s), while
   **band membership reproduced exactly and no cell crossed the 0.1s boundary**.
   Negligible operations are intrinsically hard to rank because scheduler noise is
   comparable to execution time; expensive operations vary materially in absolute
   seconds. Band membership — which is what the economic argument requires — is
   the reproducible quantity. Absolute runtimes are order-of-magnitude evidence.
2. **Human effectiveness ρ is a sensitivity parameter**, because real erroneous
   labels are not identifiable from ground truth outside a simulation.
3. **One cleaning method per error type.** A stronger method could change
   recovery; results describe the method tested, not the error type's ceiling.
4. **Class-imbalance effects vary with the initial class balance** (r ≈ −0.93,
   n = 5) and are reported per dataset rather than pooled. With five datasets this
   is an exploratory association, not an established law.
5. **Text-level errors are tested on four of five datasets** — Breast Cancer is
   fully numeric and carries no text to corrupt.
6. **Accuracy is the primary metric.** For interventions that shift the decision
   threshold it is the wrong one, which is why class-balancing is separated.
7. **`data_heterogeneity` recovery is 0.00 because the remediation is
   redundant, not because the corruption is harmless.** The corruption causes
   2.41pp of damage, rising monotonically with severity. Standardisation removes
   the constant factor *within the training fold*, but injection is train-only, so
   the test set keeps its original scale and the train-derived scaler mis-maps it.
   The tested repair, re-standardising features, only repeats what preprocessing
   already performs and — under the no-oracle rule — cannot detect the train/test
   scale mismatch that constitutes the damage. Results describe this remediation,
   not the error type's ceiling.
