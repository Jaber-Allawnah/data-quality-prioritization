"""
Severity-sensitivity check for the DAMAGE ranking (not RQ1 recovery), for
the reviewer comment that "low/medium/high" mean different, non-comparable
things across error types, and that the main damage ranking should be shown
not to depend too much on the specific severity values chosen.

`severity_sensitivity_check.py` already does something similar but only for
the 5 RQ1 headline (error_type -> automated cleaner) actions, and measures
recovery, not the damage ranking itself. This script instead:

  - covers the same 21-error-type scope already used for
    run_split_sensitivity.py (the 11 tab:damage types plus the 10 types
    needed for the 7 mechanism comparisons) -- reused here rather than
    re-justified, and NOT extended to the full 28-type taxonomy;
  - perturbs each type's own native severity_map rate (error_injector.py) by
    the same 0.7x / 1.5x convention as severity_sensitivity_check.py, so the
    perturbation is still "ordered within each error type", never a shared
    magnitude across error types;
  - measures DAMAGE ONLY (dirty vs. clean baseline accuracy) -- no cleaning
    step -- since the question is whether the damage ranking is stable, not
    whether recovery is;
  - runs at seed 42 only. A 3-seed run of this scope was estimated at ~17
    hours from observed per-cell timing (each cell still trains all 8
    models from scratch; there is no cleaning to skip that would make this
    faster than the main experiment), so it was scoped down to one seed as
    a deliberate, disclosed trade-off: this check answers whether the
    DAMAGE RANKING depends on the severity values chosen, a question
    orthogonal to the separate 3-seed check of injection/training-randomness
    sensitivity that the main repeated-experiment protocol already covers.

Bypasses ErrorInjector.inject_error()'s severity_map lookup and calls the
underlying per-mechanism method directly with a custom rate, replicating
_seed_for's deterministic seeding with a distinct severity label per
schedule so seeds don't collide with the main experiment's.

Output: results/severity_sensitivity_damage.csv
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
from degradation_tester import DegradationTester

SEEDS = [42]  # scoped to one seed for runtime reasons (~17h at 3 seeds observed
              # empirically; see run log discussion) -- this check answers
              # whether the damage ranking depends on severity choice, a
              # question orthogonal to the separate 3-seed repeated-experiment
              # check of injection/training randomness (Section on repeated
              # experiments), so single-seed scope here is a deliberate,
              # disclosed trade-off, not a silent shortcut.
SEVERITIES = ['low', 'medium', 'high']

# Same 21-type scope as run_split_sensitivity.py, reused deliberately.
SCOPED_ERROR_TYPES = sorted(set([
    'annotator_bias', 'missing_values_mnar', 'ambiguous_labels',
    'contextual_errors', 'label_noise_asymmetric', 'data_leakage',
    'concept_drift', 'data_heterogeneity', 'invalid_values',
    'data_inconsistency', 'label_noise',
    'missing_values', 'outliers', 'outliers_targeted', 'duplicates',
    'duplicates_minority', 'duplicates_targeted', 'typographical_errors',
    'categorical_errors', 'feature_noise', 'feature_noise_targeted',
]))
assert len(SCOPED_ERROR_TYPES) == 21, f'expected 21, got {len(SCOPED_ERROR_TYPES)}'

# Native severity_map rates, copied from error_injector.py's inject_error(),
# restricted to the 21 scoped types.
NATIVE_RATES = {
    'missing_values': {'low': 10, 'medium': 20, 'high': 30},
    'label_noise': {'low': 5, 'medium': 10, 'high': 15},
    'duplicates': {'low': 5, 'medium': 10, 'high': 15},
    'outliers': {'low': 1, 'medium': 5, 'high': 10},
    'invalid_values': {'low': 2, 'medium': 5, 'high': 10},
    'feature_noise': {'low': 5, 'medium': 15, 'high': 25},
    'categorical_errors': {'low': 5, 'medium': 10, 'high': 15},
    'data_inconsistency': {'low': 5, 'medium': 10, 'high': 15},
    'typographical_errors': {'low': 5, 'medium': 10, 'high': 15},
    'data_leakage': {'low': 10, 'medium': 20, 'high': 30},
    'annotator_bias': {'low': 10, 'medium': 20, 'high': 30},
    'ambiguous_labels': {'low': 5, 'medium': 10, 'high': 15},
    'data_heterogeneity': {'low': 10, 'medium': 20, 'high': 30},
    'concept_drift': {'low': 10, 'medium': 20, 'high': 30},
    'contextual_errors': {'low': 5, 'medium': 10, 'high': 15},
    'missing_values_mnar': {'low': 10, 'medium': 20, 'high': 30},
    'duplicates_targeted': {'low': 5, 'medium': 10, 'high': 15},
    'outliers_targeted': {'low': 1, 'medium': 5, 'high': 10},
    'duplicates_minority': {'low': 5, 'medium': 10, 'high': 15},
    'label_noise_asymmetric': {'low': 5, 'medium': 10, 'high': 15},
    'feature_noise_targeted': {'low': 5, 'medium': 15, 'high': 25},
}
assert set(NATIVE_RATES) == set(SCOPED_ERROR_TYPES)

# data_leakage's agreement formula saturates at leakage_rate=32; the "higher"
# schedule is capped below that, same as severity_sensitivity_check.py.
LEAKAGE_CAP = 31


def build_schedules():
    schedules = {}
    for et, rates in NATIVE_RATES.items():
        lower = {s: round(r * 0.7, 2) for s, r in rates.items()}
        higher = {s: round(r * 1.5, 2) for s, r in rates.items()}
        if et == 'data_leakage':
            higher = {s: min(v, LEAKAGE_CAP) for s, v in higher.items()}
        schedules[et] = {'lower': lower, 'higher': higher}
    return schedules


SCHEDULES = build_schedules()


def run_for_seed(seed, datasets):
    tester = DegradationTester(random_state=seed)
    injector = ErrorInjector(random_state=seed)
    total = len(SCOPED_ERROR_TYPES) * 2 * len(SEVERITIES) * len(datasets)
    done = 0
    started = time.time()

    for error_type in SCOPED_ERROR_TYPES:
        method = getattr(injector, f'inject_{error_type}')
        for schedule_name in ['lower', 'higher']:
            rates = SCHEDULES[error_type][schedule_name]
            for severity in SEVERITIES:
                rate = rates[severity]
                label = f'{severity}_{schedule_name}'  # distinct seed label

                for key, d in datasets.items():
                    done += 1
                    name = d['name']
                    X_train, X_test = d['X_train'], d['X_test']
                    y_train, y_test = d['y_train'], d['y_test']
                    cat, num = d['categorical_cols'], d['numeric_cols']

                    np.random.seed(injector._seed_for(error_type, label, X_train))
                    X_dirty, y_dirty, meta = method(X_train, y_train, rate)
                    meta['schedule'] = schedule_name
                    meta['severity_label'] = severity
                    meta['rate_used'] = rate
                    meta['seed'] = seed

                    tester.evaluate(X_dirty, y_dirty, X_test, y_test, cat, num,
                                     name, error_type, severity, meta, 'dirty')

                    print(f'[{done}/{total}] seed={seed} {error_type} {schedule_name} '
                          f'{severity}(rate={rate}) {name}', flush=True)

    elapsed = (time.time() - started) / 60
    print(f'seed {seed} DONE in {elapsed:.1f} min', flush=True)
    df = tester.save(f'results/severity_sensitivity_damage_seed{seed}_tmp.csv')
    df['Random_State'] = seed
    return df


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    frames = [run_for_seed(seed, datasets) for seed in SEEDS]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv('results/severity_sensitivity_damage.csv', index=False)
    for seed in SEEDS:
        tmp = f'results/severity_sensitivity_damage_seed{seed}_tmp.csv'
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f'\nSaved {len(out)} rows -> results/severity_sensitivity_damage.csv')


if __name__ == '__main__':
    main()
