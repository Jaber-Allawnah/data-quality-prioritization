"""
Targeted rerun: the four extreme-value error types.

`outliers`, `outliers_targeted`, `contextual_errors` and `domain_violations` all
corrupt data by writing values far outside a column's normal range. Three of them
were still cleaned by percentile winsorisation, which fails once the corruption
rate exceeds the trim rate: the injected values BECOME the 1st and 99th
percentiles, so clipping to them changes nothing. On German Credit the 1%/99%
bounds were exactly the injected -30 and 106 while the true range was 4 to 72.

`domain_violations` had already been switched to IQR-based clipping for this
reason. Leaving the other three on the broken method was an inconsistency rather
than a decision, so all four now share it. The interquartile range is estimated
from the middle half of the data and survives contamination in the tails.

Only these four are recomputed. Per-experiment seeding makes every other error
type independent, so their existing rows stay valid and are carried over
untouched.

The corrected rows REPLACE the originals in results/experiment_results.csv.
"""

import os
import pickle
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner
from degradation_tester import DegradationTester

TARGETS = ['outliers', 'outliers_targeted', 'contextual_errors', 'domain_violations']
SEVERITIES = ['low', 'medium', 'high']
RESULTS = 'results/experiment_results.csv'


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    existing = pd.read_csv(RESULTS)
    print(f'existing rows: {len(existing)}')
    print(f'recomputing  : {TARGETS}\n')

    tester = DegradationTester(random_state=42)
    started = time.time()

    for key, d in datasets.items():
        name = d['name']
        cat, num = d['categorical_cols'], d['numeric_cols']
        print(f'{name}', end='', flush=True)

        for error_type in TARGETS:
            for severity in SEVERITIES:
                injector = ErrorInjector(random_state=42)
                cleaner = DataCleaner(random_state=42)

                X_dirty, y_dirty, meta = injector.inject_error(
                    d['X_train'], d['y_train'], error_type, severity
                )
                tester.evaluate(X_dirty, y_dirty, d['X_test'], d['y_test'], cat, num,
                                name, error_type, severity, meta, 'dirty')

                X_clean, y_clean, X_test_clean, clean_meta = cleaner.clean(
                    X_dirty, y_dirty, error_type, d['X_test']
                )
                tester.evaluate(X_clean, y_clean, X_test_clean, d['y_test'], cat, num,
                                name, error_type, severity,
                                {**meta, **clean_meta}, 'cleaned')
        print(' done')

    corrected = pd.DataFrame(tester.results)

    # Replace, never append: stale rows for these error types must not survive.
    kept = existing[~existing.Error_Type.isin(TARGETS)]
    merged = pd.concat([kept, corrected], ignore_index=True)

    print(f'\ndropped (stale) : {len(existing) - len(kept)}')
    print(f'added (corrected): {len(corrected)}')
    print(f'total           : {len(merged)}')

    if len(merged) != len(existing):
        print('WARNING: row count changed - investigate before trusting the merge')

    merged.to_csv(RESULTS, index=False)
    print(f'failures        : {len(tester.failures)}')
    print(f'elapsed         : {(time.time() - started) / 60:.1f} min')


if __name__ == '__main__':
    main()
