"""
Step 3 — baseline performance on clean data.

Trains every model on each uncorrupted dataset, through the SAME preprocessing
the experiments use. That matters: if the baseline used different preprocessing,
part of every measured "degradation" would just be the pipeline difference.

Output: results/baseline_results.csv
"""

import os
import pickle
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from degradation_tester import DegradationTester


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    tester = DegradationTester(random_state=42)

    for key, d in datasets.items():
        print(f"training on {d['name']}...", end=' ', flush=True)
        n_ok, n_total = tester.evaluate(
            d['X_train'], d['y_train'], d['X_test'], d['y_test'],
            d['categorical_cols'], d['numeric_cols'],
            d['name'], 'none', 'none', {'baseline': True}, 'clean'
        )
        print(f'{n_ok}/{n_total} models')

    df = pd.DataFrame(tester.results)
    df = df.rename(columns={'Accuracy': 'Accuracy'})
    df.to_csv('results/baseline_results.csv', index=False)

    print(f'\n{len(df)} baseline rows saved\n')
    print(df.groupby('Model')[['Accuracy', 'F1', 'AUC']].mean().round(4).to_string())
    print()
    print(df.groupby('Dataset')[['Accuracy', 'F1', 'AUC']].mean().round(4).to_string())

    if tester.failures:
        print(f'\nfailures: {len(tester.failures)}')
        for f in tester.failures[:5]:
            print(' ', f)


if __name__ == '__main__':
    main()
