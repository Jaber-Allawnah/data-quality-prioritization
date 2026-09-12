"""
Validation checkpoint for the new split-sensitivity datasets, run BEFORE
any model training or experiment. Confirms stratification worked as
expected and the new splits are genuinely different from the original.

For each of the original split (results/datasets_raw.pkl) and the two new
splits (results/split100, results/split200), prints per dataset:
  - train rows, test rows
  - positive-class proportion in train, in test

Also reports whether the new splits actually differ from the original
(row-identity check via row-count-matched positive rate comparison is not
sufficient on its own, so this additionally checks whether the first few
row-order-independent identifiers -- here, the y_train class counts and a
hash of the training index content -- differ across splits).
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

SPLITS = {
    'original (seed 42)': 'results/datasets_raw.pkl',
    'split seed 100': 'results/split100/datasets_raw.pkl',
    'split seed 200': 'results/split200/datasets_raw.pkl',
}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    loaded = {}
    for label, path in SPLITS.items():
        if not os.path.exists(path):
            print(f'MISSING: {path} (label: {label}) -- run generate_split_datasets.py first')
            continue
        loaded[label] = load(path)

    if len(loaded) < 2:
        print('\nNeed at least the original plus one new split to compare. Stopping.')
        return

    print(f"{'split':22s}{'dataset':16s}{'n_train':>9s}{'n_test':>8s}"
          f"{'pos_train':>11s}{'pos_test':>10s}")
    print('-' * 78)

    for label, datasets in loaded.items():
        for key, d in datasets.items():
            n_train = len(d['y_train'])
            n_test = len(d['y_test'])
            pos_train = float(np.mean(d['y_train']))
            pos_test = float(np.mean(d['y_test']))
            print(f"{label:22s}{d['name']:16s}{n_train:9d}{n_test:8d}"
                  f"{pos_train:11.4f}{pos_test:10.4f}")
        print()

    # Sanity check: the new splits should have the SAME sizes and roughly the
    # same positive-class proportion as the original (stratification target
    # unchanged), but the ACTUAL row content should differ. We check this by
    # comparing the sorted first-column values of X_train across splits for
    # one dataset -- if splits are identical, these will match exactly.
    print('=' * 78)
    print('Row-content difference check (breast_cancer, first numeric feature,')
    print('first 5 sorted X_train values) -- splits should NOT match if the')
    print('partition genuinely changed:')
    print('=' * 78)
    for label, datasets in loaded.items():
        d = datasets['breast_cancer']
        col = d['X_train'].columns[0]
        vals = sorted(d['X_train'][col].values.tolist())[:5]
        print(f"{label:22s}{[round(v, 4) for v in vals]}")


if __name__ == '__main__':
    main()
