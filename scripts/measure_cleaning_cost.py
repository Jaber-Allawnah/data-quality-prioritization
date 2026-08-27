"""
Measure the computational cost of each cleaning action.

The ROI denominator needs this and the main experiment never captured it. No
model training is involved, so it is cheap: each cleaner is timed on each
dataset at each severity, on the same corrupted data the study used.

Two costs are recorded, because they behave differently:

  clean_seconds  - wall time to run the cleaning action itself
  train_seconds  - wall time to train the models afterwards, taken from the
                   experiment results. Some remediations change the cost of
                   training too: oversampling for class imbalance enlarges the
                   training set, so it is paid for twice.

Only `automated_cleaning` routes have a compute cost. The `external_intervention`
routes cost human time instead, which is not measurable here and is handled by
the break-even analysis rather than by inventing a rate.

Output: results/cleaning_costs.csv
"""

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner, REMEDIATION_ROUTE, AUTOMATED

SEVERITIES = ['low', 'medium', 'high']
REPEATS = 3          # median of three, so one scheduling hiccup cannot dominate


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    cleanable = sorted(DataCleaner(42)._methods())
    rows = []

    for key, d in datasets.items():
        name = d['name']
        print(f'{name}', end='', flush=True)

        for error_type in cleanable:
            for severity in SEVERITIES:
                injector = ErrorInjector(random_state=42)
                X_dirty, y_dirty, _ = injector.inject_error(
                    d['X_train'], d['y_train'], error_type, severity
                )

                timings = []
                rows_after = len(X_dirty)
                for _ in range(REPEATS):
                    cleaner = DataCleaner(random_state=42)
                    start = time.perf_counter()
                    X_clean, y_clean, _, meta = cleaner.clean(
                        X_dirty, y_dirty, error_type, d['X_test']
                    )
                    timings.append(time.perf_counter() - start)
                    rows_after = len(X_clean)

                rows.append({
                    'Dataset': name,
                    'Error_Type': error_type,
                    'Severity': severity,
                    'Route': REMEDIATION_ROUTE[error_type][0],
                    'Method': meta['method'],
                    'Clean_Seconds': float(np.median(timings)),
                    'Rows_Before': len(X_dirty),
                    'Rows_After': rows_after,
                    'Row_Change_Pct': (rows_after - len(X_dirty)) / len(X_dirty) * 100,
                })
        print(' done')

    costs = pd.DataFrame(rows)

    # Training cost after cleaning, from the experiment results. Remediations
    # that resize the training set change this, and it is part of their price.
    exp = pd.read_csv('results/experiment_results.csv')
    train = (exp[exp.Condition == 'cleaned']
             .groupby(['Dataset', 'Error_Type', 'Severity'])
             .Train_Seconds.sum().rename('Train_Seconds_After_Clean').reset_index())
    costs = costs.merge(train, on=['Dataset', 'Error_Type', 'Severity'], how='left')
    costs['Total_Compute_Seconds'] = (costs.Clean_Seconds
                                      + costs.Train_Seconds_After_Clean.fillna(0))

    costs.to_csv('results/cleaning_costs.csv', index=False)

    pd.set_option('display.width', 200)
    print('\n' + '=' * 92)
    print('COMPUTATIONAL COST PER CLEANING ACTION (median seconds)')
    print('=' * 92)
    summary = (costs.groupby(['Error_Type', 'Method'])
               .agg(Clean_Seconds=('Clean_Seconds', 'mean'),
                    Train_Seconds=('Train_Seconds_After_Clean', 'mean'),
                    Total_Seconds=('Total_Compute_Seconds', 'mean'),
                    Row_Change_Pct=('Row_Change_Pct', 'mean'))
               .reset_index().sort_values('Total_Seconds', ascending=False))
    print(summary.round(3).to_string(index=False))

    print('\nSaved: results/cleaning_costs.csv')


if __name__ == '__main__':
    main()
