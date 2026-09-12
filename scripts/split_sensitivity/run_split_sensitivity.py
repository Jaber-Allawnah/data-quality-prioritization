"""
Split-sensitivity analysis: repeats the scoped experiment grid on two new
stratified train/test partitions (split seeds 100, 200), to test whether the
main conclusions depend on the original train/test split.

Design (confirmed by read-only investigation before writing this script):
  - Split seed varies (100, 200) via DataLoader(random_state=...); this is
    the ONLY thing that changes relative to the original protocol.
  - Injector, cleaner, and tester random states are all held FIXED at 42,
    matching the original run, so injection/training randomness is not
    conflated with split variation.
  - Injection seeds are derived from (injector.random_state, error_type,
    severity, X.shape) and do NOT depend on which specific rows are in the
    split (shape is identical across any 80/20 split of the same dataset),
    so this isolation is exact, not approximate.

Scope: 21 of the 28 error types (not the full taxonomy), chosen to cover
exactly what the paper's headline conclusions rest on:
  - the 11 error types in tab:damage (>=1pp overall damage)
  - the 5 headline RQ1 remediation actions' error types (all already inside
    the 11 above)
  - the 10 additional error types needed for the 7 matched mechanism
    comparisons in tab:mechanism-results

This is a sensitivity analysis, not a full rerun: it is scoped deliberately
to answer "do the headline conclusions depend on the split", not to
reproduce every table in the paper. See background.tex / results.tex for
the full-taxonomy figures, which remain seed-42-split only.

Frozen artifacts untouched by this script:
  results/datasets_raw.pkl, results/experiment_results*.csv,
  results/baseline_results*.csv, and all src/ files.

Usage:
  python run_split_sensitivity.py --dry-run   # smoke test only
  python run_split_sensitivity.py             # full run (hours)

Output:
  results/split_sensitivity/split100/baseline_results.csv
  results/split_sensitivity/split100/experiment_results.csv
  results/split_sensitivity/split100/metadata.json
  results/split_sensitivity/split200/... (same three files)
"""

import argparse
import datetime
import json
import os
import pickle
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner
from degradation_tester import DegradationTester

SPLIT_SEEDS = [100, 200]
FIXED_SEED = 42  # injector / cleaner / tester -- unchanged from the original protocol
SEVERITIES = ['low', 'medium', 'high']

# 21 of 28 error types: the 11 tab:damage types plus the 10 additional types
# needed for the 7 matched mechanism comparisons. Hard-coded deliberately so
# the scope is transparent and cannot silently drift to the full 28.
SCOPED_ERROR_TYPES = sorted(set([
    # tab:damage (>=1pp overall damage), includes all 5 RQ1 headline types
    'annotator_bias', 'missing_values_mnar', 'ambiguous_labels',
    'contextual_errors', 'label_noise_asymmetric', 'data_leakage',
    'concept_drift', 'data_heterogeneity', 'invalid_values',
    'data_inconsistency', 'label_noise',
    # additional types needed for the 7 matched mechanism comparisons
    'missing_values', 'outliers', 'outliers_targeted', 'duplicates',
    'duplicates_minority', 'duplicates_targeted', 'typographical_errors',
    'categorical_errors', 'feature_noise', 'feature_noise_targeted',
]))
assert len(SCOPED_ERROR_TYPES) == 21, f'expected 21, got {len(SCOPED_ERROR_TYPES)}'


def load_split_datasets(split_seed):
    path = f'results/split{split_seed}/datasets_raw.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_baseline(split_seed, datasets, out_dir):
    tester = DegradationTester(random_state=FIXED_SEED)
    for key, d in datasets.items():
        print(f"  baseline: {d['name']}...", end=' ', flush=True)
        n_ok, n_total = tester.evaluate(
            d['X_train'], d['y_train'], d['X_test'], d['y_test'],
            d['categorical_cols'], d['numeric_cols'],
            d['name'], 'none', 'none', {'baseline': True, 'split_seed': split_seed}, 'clean'
        )
        print(f'{n_ok}/{n_total} models')

    df = pd.DataFrame(tester.results)
    out_path = os.path.join(out_dir, 'baseline_results.csv')
    df.to_csv(out_path, index=False)
    print(f'  saved: {out_path} ({len(df)} rows)\n')
    return df


def run_experiment(split_seed, datasets, error_types, out_dir):
    tester = DegradationTester(random_state=FIXED_SEED)
    total = len(datasets) * len(error_types) * len(SEVERITIES)
    done = 0
    skipped_clean = 0

    for key, d in datasets.items():
        name = d['name']
        X_train, X_test = d['X_train'], d['X_test']
        y_train, y_test = d['y_train'], d['y_test']
        cat, num = d['categorical_cols'], d['numeric_cols']

        print(f'  {name}')

        for error_type in error_types:
            for severity in SEVERITIES:
                done += 1
                injector = ErrorInjector(random_state=FIXED_SEED)
                cleaner = DataCleaner(random_state=FIXED_SEED)

                try:
                    X_dirty, y_dirty, meta = injector.inject_error(
                        X_train, y_train, error_type, severity
                    )
                except Exception as e:
                    print(f'    INJECT FAILED {error_type} {severity}: {type(e).__name__}: {e}')
                    continue

                tester.evaluate(X_dirty, y_dirty, X_test, y_test, cat, num,
                                name, error_type, severity,
                                {**meta, 'split_seed': split_seed}, 'dirty')

                X_clean, y_clean, X_test_clean, clean_meta = cleaner.clean(
                    X_dirty, y_dirty, error_type, X_test
                )

                if clean_meta['recovery_evaluation_status'] == 'evaluated' \
                        and clean_meta['method'] != 'none':
                    tester.evaluate(X_clean, y_clean, X_test_clean, y_test, cat, num,
                                    name, error_type, severity,
                                    {**meta, **clean_meta, 'split_seed': split_seed}, 'cleaned')
                else:
                    skipped_clean += 1

        print(f'  [{done}/{total}]')

    df = tester.save(os.path.join(out_dir, 'experiment_results.csv'))
    print(f'  {len(tester.failures)} failures, {skipped_clean} cleaning skipped\n')
    return df


def write_metadata(split_seed, error_types, out_dir, dry_run):
    meta = {
        'split_seed': split_seed,
        'injector_seed': FIXED_SEED,
        'cleaner_seed': FIXED_SEED,
        'tester_seed': FIXED_SEED,
        'severities': SEVERITIES,
        'included_error_types': error_types,
        'n_error_types': len(error_types),
        'dry_run': dry_run,
        'timestamp': datetime.datetime.now().isoformat(),
        'script': 'run_split_sensitivity.py',
    }
    path = os.path.join(out_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='smoke test: 1 dataset, 1 error type, 1 severity, both splits')
    args = parser.parse_args()

    error_types = ['missing_values_mnar'] if args.dry_run else SCOPED_ERROR_TYPES
    scope_note = ' (dry run: 1 dataset, 1 error type, 1 severity, all 8 models)' \
        if args.dry_run else ''
    print(f'Error types in scope: {len(error_types)}{scope_note}')

    for split_seed in SPLIT_SEEDS:
        print('=' * 78)
        print(f'SPLIT SEED {split_seed}{"  [DRY RUN]" if args.dry_run else ""}')
        print('=' * 78)

        out_dir = f'results/split_sensitivity/split{split_seed}'
        os.makedirs(out_dir, exist_ok=True)

        datasets = load_split_datasets(split_seed)
        if args.dry_run:
            # restrict to a single dataset for the smoke test; all 8 models
            # still run per the existing DegradationTester behaviour, which
            # is fast enough (seconds) that trimming to "one model" is not
            # worth special-casing the frozen src/degradation_tester.py.
            first_key = next(iter(datasets))
            datasets = {first_key: datasets[first_key]}
            global SEVERITIES
            SEVERITIES = ['low']  # dry run: 1 severity, matching the smoke-test spec

        started = time.time()
        run_baseline(split_seed, datasets, out_dir)
        run_experiment(split_seed, datasets, error_types, out_dir)
        write_metadata(split_seed, error_types, out_dir, args.dry_run)

        elapsed = (time.time() - started) / 60
        print(f'split {split_seed} done in {elapsed:.1f} min\n')


if __name__ == '__main__':
    main()
