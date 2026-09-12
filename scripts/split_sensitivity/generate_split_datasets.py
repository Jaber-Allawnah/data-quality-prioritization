"""
Generates new stratified train/test splits for the split-sensitivity
analysis, using the SAME DataLoader class as the frozen pipeline, just with
a different random_state for the split step.

Injector/cleaner/tester random states are NOT touched by this script —
they stay at their existing values (42) in the runner script that consumes
these datasets, so the ONLY thing varying relative to the original run is
the train/test partition itself.

Does not touch results/datasets_raw.pkl (the frozen original split).

Output: results/split100/datasets_raw.pkl, results/split200/datasets_raw.pkl
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from data_loader import DataLoader

SPLIT_SEEDS = [100, 200]


def main():
    for split_seed in SPLIT_SEEDS:
        out_dir = f'results/split{split_seed}'
        os.makedirs(out_dir, exist_ok=True)

        print('=' * 78)
        print(f'SPLIT SEED {split_seed}')
        print('=' * 78)

        loader = DataLoader(random_state=split_seed)
        datasets = loader.load_all()

        out_path = os.path.join(out_dir, 'datasets_raw.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(datasets, f)
        print(f'Saved: {out_path}\n')


if __name__ == '__main__':
    main()
