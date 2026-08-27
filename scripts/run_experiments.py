"""
Main experiment — Steps 4 to 7 in a single pass
===============================================

For every (dataset, error type, severity):

    1. inject the error into the RAW training data
    2. preprocess and train every model     -> DIRTY performance    (Steps 4-5)
    3. apply the cleaning protocol
    4. preprocess and train every model     -> CLEANED performance  (Steps 6-7)

Dirty and cleaned are measured in the same pass so the corruption is injected
once rather than twice, which removes any chance of the two conditions seeing
different corrupted data.

Preprocessing is fitted separately for each condition, on that condition's own
training fold - the encoder and scaler must see exactly what the model sees.

Output: results/experiment_results.csv  (one row per dataset x error x severity
        x model x condition)
"""

import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner
from degradation_tester import DegradationTester

SEVERITIES = ['low', 'medium', 'high']
OUT = 'results/experiment_results.csv'


def registered_error_types():
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, 'src', 'error_injector.py'), encoding='utf-8').read()
    return re.findall(r"'([a-z_]+)': self\.inject_", src)


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    error_types = registered_error_types()
    total = len(datasets) * len(error_types) * len(SEVERITIES)

    print('=' * 78)
    print('EXPERIMENT: inject -> measure damage -> clean -> measure recovery')
    print('=' * 78)
    print(f'{len(datasets)} datasets x {len(error_types)} error types x '
          f'{len(SEVERITIES)} severities = {total} combinations')
    print('each trains 8 models twice (dirty + cleaned)\n')

    tester = DegradationTester(random_state=42)
    started = time.time()
    done = 0
    skipped_clean = 0

    for key, d in datasets.items():
        name = d['name']
        X_train, X_test = d['X_train'], d['X_test']
        y_train, y_test = d['y_train'], d['y_test']
        cat, num = d['categorical_cols'], d['numeric_cols']

        print(f"{'=' * 78}\n{name}\n{'=' * 78}")

        for error_type in error_types:
            print(f'  {error_type}', end='', flush=True)

            for severity in SEVERITIES:
                done += 1
                injector = ErrorInjector(random_state=42)
                cleaner = DataCleaner(random_state=42)

                try:
                    X_dirty, y_dirty, meta = injector.inject_error(
                        X_train, y_train, error_type, severity
                    )
                except Exception as e:
                    print(f'\n    INJECT FAILED {severity}: {type(e).__name__}: {e}')
                    continue

                tester.evaluate(X_dirty, y_dirty, X_test, y_test, cat, num,
                                name, error_type, severity, meta, 'dirty')

                # The test frame is handed to the cleaner so that any fitted
                # transform (rescaling, quantile mapping, learned clip bounds)
                # is applied to it with the SAME training-derived parameters.
                # Transforming training data alone leaves the two frames on
                # different scales, which destroys accuracy far more thoroughly
                # than the corruption being repaired.
                X_clean, y_clean, X_test_clean, clean_meta = cleaner.clean(
                    X_dirty, y_dirty, error_type, X_test
                )

                if clean_meta['recovery_evaluation_status'] == 'evaluated' \
                        and clean_meta['method'] != 'none':
                    tester.evaluate(X_clean, y_clean, X_test_clean, y_test, cat, num,
                                    name, error_type, severity,
                                    {**meta, **clean_meta}, 'cleaned')
                else:
                    # No automated repair: cleaned is identical to dirty by
                    # construction, so retraining would burn compute to
                    # reproduce numbers already recorded.
                    skipped_clean += 1

            print(f'  [{done}/{total}]')

    df = tester.save(OUT)

    elapsed = (time.time() - started) / 60
    print(f'\ncombinations   : {done}/{total}')
    print(f'rows recorded  : {len(df)}')
    print(f'cleaning skipped (no automated repair): {skipped_clean}')
    print(f'failures       : {len(tester.failures)}')
    print(f'elapsed        : {elapsed:.1f} min')


if __name__ == '__main__':
    main()
