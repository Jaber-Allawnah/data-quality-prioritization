# Frozen Analysis Version

**Frozen:** 9 August 2026
**Scope:** RQ1–RQ4 measured, RQ2 cost model and break-even complete.
**Status:** internally frozen, not yet reviewed by a supervisor.

Everything below reproduces the submitted results. Later changes should be
distinguishable from this state — see *Verifying* at the end.

---

## What is frozen

| Artefact | Content |
|---|---|
| `METHODOLOGY.md` | protocol, definitions, limitations |
| `RESEARCH_LOG.md` | rebuild rationale, bugs found, rejected explanations |
| `RESULTS_SUMMARY.md` | the results package |
| `results/experiment_results.csv` | 5,760 measurements, 420 combinations |
| `results/cleaning_costs.csv` | 300 cost observations (regenerated 18:00) |
| `frozen/cleaning_costs_PREVIOUS_RUN.csv` | the earlier cost run, retained deliberately |
| `frozen/*_output.txt` | captured stdout of the three analysis scripts |
| `frozen/CHECKSUMS.txt` | SHA-256 over 36 files |
| `src/`, audit and analysis scripts | the code that produced all of the above |

The previous cost run is kept because it is what separates **runtime
reproducibility** (same definition, measured twice) from **cost-definition
robustness** (same measurement, two definitions). Those are different questions
and the study answers both.

---

## Why this state, and not a further round

The final audit did not leave every number unchanged. Re-measurement moved
KNN imputation 17.7s → 15.1s, `E_benchmark` 207.5 → 215.0 pp/s, and ρ
0.926 → 0.923. What it did establish is that the **substantive conclusions
survived better measurement and stricter definitions**:

- expensive-intervention set identical at 0.05s, 0.10s and 0.20s cutoffs
- zero cells crossed the 0.1s boundary between runs
- band-level agreement 0.800 (16/20) across the two marginal-cost definitions
- band membership reproduced exactly despite 21.5% movement in absolute seconds

The stable RQ2 claim is therefore **not** the 0.800 coefficient or the 15.1s
figure. It is that genuinely expensive interventions stay consistently
distinguishable from the negligible-cost group, across both cost definitions and
every reasonable threshold.

## Corrections that shaped the final state

Three cases where a plausible-looking statistic conflicted with its conceptual
definition, and the quantity was traced rather than the number explained:

**The `max_j` denominator.** The break-even formula selected the literal maximum
pp/s, which is whichever action recorded the smallest runtime — while the same
document argued sub-0.1s timings are noise. Replaced with an explicit
`E_benchmark`, the highest-*benefit* action inside the negligible band.

**`Total_Compute_Seconds` in the stability audit.** Produced a band agreement of
0.150 that looked like a robustness failure. That column carries a ~3.8s training
floor, so a 0.1s threshold marks nearly everything substantial by construction.
The sensitivity definition prices only the training time the intervention *adds*;
corrected, agreement is 0.800.

**The RQ4 dataset-size explanation.** Carried over from the previous study, where
a 100-row dataset anchored it. Recomputed after that dataset was dropped:
Spearman(rows, agreement) = 0.300, non-monotonic, with the largest dataset showing
among the lowest agreement. Deleted rather than softened, and recorded as a
tested-and-rejected candidate explanation.

No further exploratory hypotheses were tested against RQ4. With five datasets and
no pre-registered prediction, testing additional descriptors until one cleared
would manufacture exactly the kind of convenient mechanism that was just retracted.

---

## Amendments after the freeze

Recorded so a changed hash is never unexplained. Evidence files
(`results/*.csv`) are unchanged by all of these — verified by diffing every
manifest entry before re-hashing.

**2026-08-09, documentation only — 2 files (`METHODOLOGY.md`, `RESEARCH_LOG.md`).**
Supervisor review found three internal inconsistencies:

1. `METHODOLOGY.md` stated "each trains 8 models twice = 5,760 rows", which
   multiplies out to 6,720. The total 5,760 was right, the factorisation wrong:
   only the 20 automated types have a cleaned condition. Rewritten to show
   3,360 dirty + 2,400 cleaned explicitly.
2. `RESEARCH_LOG.md` still carried "RQ1, RQ3 and RQ4 measured … next: figures"
   as its *current* status. Relabelled as status-at-the-time and a final entry
   added covering RQ2 completion, the cost regeneration and the freeze.
3. `RESEARCH_LOG.md` quoted class-balancing figures (−6.43 / +51.64 / +17.21pp)
   from the intermediate run that prompted the interpretation, which differ from
   the final regenerated values (−6.83 / +47.62 / +12.35pp, AUC 0.873 → 0.870).
   The historical figures are retained as research record with a provenance note
   pointing at `RESULTS_SUMMARY.md` as authoritative. The magnitudes shifted; the
   interpretation did not.

None of these changed a result, a conclusion, or a number in the evidence base.

**2026-08-09, second review — claim strength, documentation only.** Six further
points, all reducing the strength of claims to match the evidence:

4. **`E_benchmark` precision.** Its runtime (0.012s) sits inside the band this
   study declares noise-dominated, so 215.0 pp/s cannot be treated as exact while
   0.012s vs 0.024s is declared uninterpretable. C\* is now described as an
   approximate compute-equivalent threshold, and the break-even was **recomputed
   under both independent timing runs**: `E_benchmark` 207.5 vs 215.0 pp/s (3.4%
   apart), `annotator_bias` C\*(ρ=1.0) 0.0346 vs 0.0334, every C\* below 0.05
   under both. The conclusion is unchanged by which run is used.
5. **"Mechanism dominates error type" was broader than the evidence.** Only 3 of 7
   matched pairs were significant, and all three tie corruption to the target.
   Restated as "target-linked corruption mechanisms can dominate corruption rate
   alone", with the note that "mechanism always matters" is not supported.
6. **RQ3 was stated too categorically.** "Priority depends on model family" became
   "priorities agree more within model families than across them, consistent with
   a family-level effect" — a descriptive rank-correlation comparison over 8
   models, not a causal test.
7. **Text-error explanation aligned across documents.** `RESEARCH_LOG.md` claimed
   ordinal encoding "absorbs" the corruption; it now matches `RESULTS_SUMMARY.md`
   — consistent with, pending an encoding ablation. The observation is
   established, the explanation is not.
8. **`data_heterogeneity` wording made implementation-specific** in both documents,
   so the ≈0 result is not read as "standardisation makes heterogeneity harmless".
   **[Superseded 2026-08-10 — see the final amendment below.** This entry still
   described a "≈0 result", which was itself wrong: measured damage is 2.41pp. The
   ≈0 figure is the *recovery*, not the damage.**]**
9. **Pipeline diagram now shows both paths** (dirty and remediated), making the
   experimental comparison and the position of the CLEAN step explicit.

Again: no result, conclusion or evidence file changed — only how strongly the
findings are stated.

**2026-08-09, third review — full recheck of all four documents.** Every figure in
`RESULTS_SUMMARY.md` was recomputed from `results/*.csv` and reproduced exactly:
the severity table, the RQ1 recovery table, all 7 mechanism pairs with p-values,
the class-balancing table, RQ3 within/across correlations, RQ4 per-dataset
agreement, the cost table, and the full break-even grid. Four wording problems and
one gap were fixed:

10. **"Roughly eight cause material damage" had no stated threshold** and did not
    match the data: 10 types exceed 1pp, 8 exceed 2.5pp. Now states both counts
    with their thresholds, and notes that 3 types are marginally negative
    (noise around zero, not a benefit from corruption).
11. **`r = −0.933` was false precision on n = 5** and moved to −0.934 under a
    different aggregation order. Reported as `r ≈ −0.93` in both documents.
12. **`METHODOLOGY.md` still framed RQ3 causally** ("test whether priority depends
    on model family"); aligned with the softened claim, and the `E_benchmark`
    conservatism direction plus the two-run sensitivity were carried across from
    `RESULTS_SUMMARY.md` so the two documents state the same caveats.
13. **`RESEARCH_LOG.md` cleanups:** a mangled line-wrap in the
    `data_heterogeneity` entry; "27/28 valid" in the validation table read as a
    failure when the 28th was manually verified working; and "Open items" listed
    only resolved items, so it is now "Items raised and how they were settled".
14. **Added a step-by-step record** to `RESEARCH_LOG.md` mapping all 11 workflow
    steps to what was done and what each produced, plus the five analyses beyond
    the planned steps. This was a genuine gap: the results were all present but
    never presented as a walk through the process.

Evidence files unchanged.

**2026-08-10, data_heterogeneity contradiction corrected.** Supervisor review found
that the documents claimed "corruption does not survive this preprocessing" and
"shows ~0 damage" while the results table reported **2.41pp damage rising 1.06 ->
2.15 -> 4.03 with severity**. Those cannot both be true, and the measured damage is
correct.

Mechanism established by direct measurement rather than assumed: the injector
scales whole columns, and standardisation - fitted on the corrupted training fold -
does remove that factor WITHIN the training fold. But errors are injected into the
training fold only, so the test set keeps its original scale and the train-derived
scaler mis-maps it. On Breast Cancer the standardised test column moves to mean
-2.01 at x2, -3.18 at x5 and -3.57 at x10, with std collapsing 0.92 -> 0.09. That
train/test mismatch IS the damage, which is why it rises with severity.

The 0.00 recovery is therefore a property of the **remediation**, not the error:
re-standardising features repeats what preprocessing already does, and the
no-oracle rule prevents the cleaner from seeing the test set. Every
"corruption is neutralised / harmless" statement in RESULTS_SUMMARY.md,
METHODOLOGY.md, RESEARCH_LOG.md and the workbook has been replaced with language
about the remediation being redundant. No measured number changed.

**2026-08-10, contamination found and corrected — the manifest earned its keep.**
`results/cleaning_costs.csv` was overwritten at 18:48 on 9 August, after the
freeze, by a cost job left running from before a context compaction. Its timings
were inflated 1.8-3.5x by CPU contention. The manifest detected the change; the
documents did not silently follow it.

A fourth clean measurement was taken with no other job running. Three clean runs
agree closely and the contaminated one is the clear outlier:

```
                  run1     run2     run3(CONTAM)   run4
IQR clipping     0.0127   0.0123     0.0215       0.0118
KNN (MNAR)       17.73    15.08      50.80        14.51
```

Substantial-band membership was identical in all four, including the contaminated
one. Figures updated to run 4: `E_benchmark` 215.0 -> 222.4 pp/s, KNN 15.1 ->
14.5s, `annotator_bias` C*(rho=1.0) 0.033 -> 0.032. No conclusion changed.

Three further errors were found by a full recheck and fixed:

15. **"Damage rises monotonically with severity for every substantial error
    type"** was contradicted by the table three lines above it. Only 7 of 10 are
    monotonic; `contextual_errors`, `invalid_values` and `data_inconsistency` dip
    at medium severity. Corrected, with the reason stated.
16. **"The audit confirms 5%+ of categorical cells changed"** held only at medium
    and high severity; at low it is 2.7-3.9%. Corrected to the measured range
    2.7-10.9% (median 5.7%).
17. **"preflight.py - 20 checks"** actually runs 24. Corrected in both documents.

And one code-level fix: `METHODOLOGY.md` states remediation routes are assigned
with **no default fallback**, but `data_cleaner.py` silently defaulted unrouted
types to `prevention_only`. It never fired, but a silent default of exactly this
kind previously turned "no cleaner was written" into a published claim. It now
raises `KeyError`, so the documented guarantee is enforced rather than asserted.

---

## Verifying

```
python frozen/make_manifest.py     # rewrites frozen/CHECKSUMS.txt
```

Re-run and diff against the committed `CHECKSUMS.txt` (ignoring the `generated`
timestamp line). Any differing hash is a file changed since the freeze. The
manifest deliberately excludes itself — writing it changes it, so a self-entry
would record the previous version's digest and could never reproduce. Verified
byte-identical across consecutive runs. Re-running an analysis script overwrites its
`results/*.csv`, so expect timing-derived files to differ on re-measurement even
when conclusions do not — that is the 21.5% substantial-band variance documented
above, not a corruption.

Changes from here should be driven by supervisor feedback, a reproducibility
failure, or a clearly justified additional analysis — not by further opportunistic
improvement.
