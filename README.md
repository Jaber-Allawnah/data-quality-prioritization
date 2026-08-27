# data-quality-prioritization

A framework for prioritising data quality remediation using damage, recovery, and compute cost across 28 injected error types, 20 automated-remediation routes, 5 datasets, and 8 models.

Damage is measured for all 28 error types. Recovery and measured compute
cost are only available for the 20 that have an implemented automated
cleaner; the remaining 8 are routed to external (human) intervention and
are measured for damage only, with a compute-equivalent break-even
threshold reported for the four of those with the greatest damage. See
`docs/METHODOLOGY.md` for the full framework and `docs/RESULTS_SUMMARY.md`
for headline numbers.

This is a **curated reproducibility package**, not a mirror of the working
research directory. Build artefacts, superseded/contaminated runs, and
generated convenience files are deliberately left out — see `frozen/README.md`
and `docs/RESEARCH_LOG.md` for what was excluded and why.

## Repository layout

```
src/           Core pipeline: data loading, error injection, cleaning, models, degradation testing
scripts/       Orchestration scripts - run the pipeline, analyse results, audit correctness
data/          Not committed - fetch_data.py downloads and hash-verifies the 5 source datasets
results/       Authoritative output CSVs the paper's tables are drawn from
frozen/        Verification evidence (checksums, audits, timing-stability runs) - see frozen/README.md
docs/          METHODOLOGY.md, RESULTS_SUMMARY.md, FROZEN.md, RESEARCH_LOG.md
```

The public repository reorganises experiment entry-point scripts under
`scripts/`. In the original working directory they sat next to `src/` and
resolved it as a sibling (`sys.path.insert(0, .../src)`); moving them
required updating that one line in each of the 9 affected scripts to
resolve `src/` one level up instead. Only relative import-path resolution
was changed; experimental implementations and analysis logic are
unchanged. `REPO_CHECKSUMS.txt` records SHA-256 hashes of every file in
`src/` and `scripts/` as they exist in this repository, separately from
`frozen/CHECKSUMS.txt`, which documents the original, unmodified
experiment-freeze artefacts and is untouched. The reorganisation was
verified by re-running the full `preflight.py` suite (24/24 checks pass)
against the code in its new location.

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
cd data && python fetch_data.py && cd ..
```

`requirements.txt` is **not** a captured environment lock — no venv/conda
export exists from the original run (9-10 August 2026). It records the
versions currently used to run this codebase; see the comment at the top of
that file.

## Reproducing experimental conditions vs re-measuring runtime

These are different guarantees:

- **Experimental conditions** (which data was corrupted, how, and what any
  model saw) are fully deterministic and reconstructable. Every injection
  is seeded as
  `SHA256(f'{random_state}|{error_type}|{severity_level}|{n_rows}x{n_cols}')[:8]`
  (`src/error_injector.py`), so a given (dataset, error type, severity)
  combination produces bit-identical corrupted data on any machine, any
  run. This is what `frozen/CHECKSUMS.txt` and `results/injector_audit.csv`
  verify.
- **Runtime** (the cost figures in RQ2 - seconds of cleaning compute) is
  measured wall-clock (`time.perf_counter()`) on a single machine, with no
  CPU/RAM specification recorded at run time. It is environment-dependent
  and **will not reproduce the same absolute seconds** on different
  hardware. The paper itself treats these figures as order-of-magnitude
  cost bands rather than precise values for exactly this reason (see
  `docs/RESULTS_SUMMARY.md` and `frozen/timing_runs/`, which document three
  independent runs of the same measurement moving by up to 7%, plus a
  fourth run inflated 1.8-3.5x by unrelated CPU contention on the same
  machine).

## Pipeline order

Full step-by-step provenance, including every bug found and fixed along the
way, is in `docs/RESEARCH_LOG.md`. In brief, from the repository root once
`data/` is populated:

1. `python src/data_loader.py` - load and split the five datasets, cache to `results/datasets_raw.pkl`
2. `python scripts/preflight.py` - 24 pre-run checks (baseline AUC thresholds, dtype handling, etc.)
3. `python scripts/audit_injectors.py` - verify all 28 error types actually corrupt data at every severity
4. `python scripts/run_baselines.py` - train baseline (uncorrupted) models
5. `python scripts/run_experiments.py` - inject, measure damage, clean, measure recovery -> `results/experiment_results.csv`
6. `python scripts/measure_cleaning_cost.py` - time every cleaner -> `results/cleaning_costs.csv`
7. `python scripts/audit_pipeline_overlap.py` - verify corruption/cleaning survive preprocessing correctly
8. `python scripts/analyse_results.py` - damage table, RQ1, mechanism comparisons, RQ3, RQ4
9. `python scripts/build_roi_model.py` - RQ2: cost bands, `E_benchmark`, break-even thresholds
10. `python scripts/audit_cost_stability.py frozen/cleaning_costs_PREVIOUS_RUN.csv` - two-run timing stability check (takes the comparison file as an argument; every other script above takes none)
11. `python scripts/verify_documents.py` - cross-checks every figure quoted in `docs/` against `results/*.csv`

All commands above are run from the repository root - the scripts resolve
`data/`, `results/`, and `src/` relative to that.

`scripts/apply_cost_update.py` and `scripts/rerun_extreme_value_cleaners.py`
are one-off historical patch scripts used during the winsorisation and
cost-label fixes documented in `docs/RESEARCH_LOG.md` ("Bugs found and
fixed") - not part of a from-scratch run. `scripts/make_excel_summary.py`
generates a supplementary Excel workbook from `results/*.csv` and is
optional.

## Claims-to-evidence map

| Claim in the paper | Evidence |
|---|---|
| Damage by error type (Results, "Damage") | `results/experiment_results.csv`, `frozen/analysis_output.txt` |
| RQ1 - effectiveness ranking | `results/rq1_cleaning_actions.csv` |
| RQ2 - cost bands, `E_benchmark`, break-even | `results/rq2_automated_efficiency.csv`, `results/rq2_breakeven.csv`, `frozen/rq2_model_output.txt` |
| RQ2 - three-run `E_benchmark` reliability | `frozen/timing_runs/run1_roi_output.txt`, `frozen/timing_runs/run2_cost_output.txt`, `results/cleaning_costs.csv` (run 4, authoritative) |
| RQ2 - two-run cost-band stability | `frozen/cleaning_costs_PREVIOUS_RUN.csv` (previous) + `results/cleaning_costs.csv` (current) + `frozen/audit_cost_stability_output.txt` |
| RQ3 - model dependency | `results/rq3_model_dependency.csv` |
| RQ4 - dataset dependency | `results/rq4_dataset_dependency.csv` |
| Mechanism comparisons | `results/mechanism_comparison.csv` |
| Injector correctness (0 inert, 0 broken) | `results/injector_audit.csv` |
| Pipeline/preprocessing correctness | `results/pipeline_overlap_audit.csv` |
| Every quantitative claim in `docs/` | `verify_documents.py` (checked programmatically against `results/*.csv`) |

## License

MIT - see `LICENSE`.
