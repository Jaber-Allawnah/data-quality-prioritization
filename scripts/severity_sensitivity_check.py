"""
Scoped severity-magnitude sensitivity check for the 5 RQ1 headline actions,
requested by the supervisor (point #5, severity justification half). NOT a
full 28-error-type rerun - see docs/RESEARCH_LOG.md entry for why this is
scoped and what it does/doesn't establish.

For each of the 5 headline (error_type -> automated cleaner) pairs, perturbs
that error type's OWN native severity parameter (not an external uniform
schedule) to a "lower" and "higher" alternative, using the same mechanism-
specific unit as the original injector. Leakage's "higher" schedule is
capped below its agreement-formula's saturation point (0.98, hit at
leakage_rate=32) rather than mirroring the x1.5 multiplier used elsewhere.

Run across all 3 seeds (42, 7, 123) used in the main repeated-experiments
check, so this isn't conditioned on a single split.

Bypasses ErrorInjector.inject_error()'s severity_map lookup and calls the
underlying per-mechanism methods directly with custom rates, replicating
the same deterministic seeding scheme (_seed_for) with a distinct severity
label per schedule so seeds don't collide with the main experiment's.

Output: results/severity_sensitivity.csv
"""
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner
from degradation_tester import DegradationTester

SEEDS = [42, 7, 123]
SEVERITIES = ['low', 'medium', 'high']

# error_type -> (injector method name, rate kwarg name, {schedule: {severity: rate}})
SCHEDULES = {
    'missing_values_mnar': {
        'method': 'inject_missing_values_mnar', 'kwarg': 'missing_rate',
        'lower':  {'low': 7,   'medium': 14, 'high': 21},
        'higher': {'low': 15,  'medium': 30, 'high': 45},
    },
    'contextual_errors': {
        'method': 'inject_contextual_errors', 'kwarg': 'contextual_error_rate',
        'lower':  {'low': 3.5, 'medium': 7,   'high': 10.5},
        'higher': {'low': 7.5, 'medium': 15,  'high': 22.5},
    },
    'invalid_values': {
        'method': 'inject_invalid_values', 'kwarg': 'invalid_rate',
        'lower':  {'low': 1.4, 'medium': 3.5, 'high': 7},
        'higher': {'low': 3,   'medium': 7.5, 'high': 15},
    },
    'data_inconsistency': {
        'method': 'inject_data_inconsistency', 'kwarg': 'inconsistency_rate',
        'lower':  {'low': 3.5, 'medium': 7,   'high': 10.5},
        'higher': {'low': 7.5, 'medium': 15,  'high': 22.5},
    },
    'data_leakage': {
        'method': 'inject_data_leakage', 'kwarg': 'leakage_rate',
        'lower':  {'low': 7,   'medium': 14, 'high': 21},
        'higher': {'low': 12,  'medium': 22, 'high': 31},   # capped below saturation (32)
    },
}


def run_for_seed(seed, datasets):
    out = f'results/severity_sensitivity_seed{seed}.csv'
    tester = DegradationTester(random_state=seed)
    started = time.time()
    total = len(SCHEDULES) * 2 * len(SEVERITIES) * len(datasets)
    done = 0

    for error_type, spec in SCHEDULES.items():
        for schedule_name in ['lower', 'higher']:
            rates = spec[schedule_name]
            for severity in SEVERITIES:
                rate = rates[severity]
                label = f'{severity}_{schedule_name}'  # distinct seed label

                for key, d in datasets.items():
                    done += 1
                    name = d['name']
                    X_train, X_test = d['X_train'], d['X_test']
                    y_train, y_test = d['y_train'], d['y_test']
                    cat, num = d['categorical_cols'], d['numeric_cols']

                    injector = ErrorInjector(random_state=seed)
                    cleaner = DataCleaner(random_state=seed)

                    import numpy as np
                    np.random.seed(injector._seed_for(error_type, label, X_train))
                    method = getattr(injector, spec['method'])
                    X_dirty, y_dirty, meta = method(X_train, y_train, **{spec['kwarg']: rate})
                    meta['schedule'] = schedule_name
                    meta['severity_label'] = severity
                    meta['rate_used'] = rate
                    meta['seed'] = seed

                    tester.evaluate(X_dirty, y_dirty, X_test, y_test, cat, num,
                                    name, error_type, severity, meta, 'dirty')

                    X_clean, y_clean, X_test_clean, clean_meta = cleaner.clean(
                        X_dirty, y_dirty, error_type, X_test
                    )
                    if clean_meta['recovery_evaluation_status'] == 'evaluated' \
                            and clean_meta['method'] != 'none':
                        tester.evaluate(X_clean, y_clean, X_test_clean, y_test, cat, num,
                                        name, error_type, severity,
                                        {**meta, **clean_meta}, 'cleaned')

                    print(f'[{done}/{total}] seed={seed} {error_type} {schedule_name} '
                          f'{severity}(rate={rate}) {name}', flush=True)

    df = tester.save(out)
    elapsed = (time.time() - started) / 60
    print(f'seed {seed} DONE: {len(df)} rows, {len(tester.failures)} failures, '
          f'{elapsed:.1f} min', flush=True)


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)
    for seed in SEEDS:
        run_for_seed(seed, datasets)


if __name__ == '__main__':
    main()
