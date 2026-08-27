"""
Pre-flight check — everything that must hold before the study runs.

Exists because the previous study launched three full runs before validating its
instruments, and every one of them produced contaminated results. Each check
below corresponds to a specific failure that actually occurred.

Exit code 0 = safe to run.
"""

import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

failures = []
notes = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'  [{status}] {name}' + (f' — {detail}' if detail else ''))
    if not condition:
        failures.append(name)


print('=' * 78)
print('PRE-FLIGHT')
print('=' * 78)

# ---------------------------------------------------------------- 1. imports
print('\n1. Modules')
import models, error_injector, data_cleaner, degradation_tester, preprocessing, data_loader
check('all modules import', True)

# ----------------------------------------------------------------- 2. models
print('\n2. Models')
m = models.build_models(42)
check('8 models defined', len(m) == 8, f'{len(m)}')
check('family labels complete', set(m) == set(models.MODEL_FAMILY))
fam = pd.Series(models.MODEL_FAMILY).value_counts().to_dict()
check('balanced 4 tree / 4 non-tree', fam.get('tree') == 4 and fam.get('non-tree') == 4, str(fam))
no_proba = [k for k, v in m.items() if not hasattr(v, 'predict_proba')]
check('all expose predict_proba (AUC would crash otherwise)', not no_proba, str(no_proba))

# --------------------------------------------------------------- 3. datasets
print('\n3. Datasets')
check('raw pickle exists', os.path.exists('results/datasets_raw.pkl'))
with open('results/datasets_raw.pkl', 'rb') as f:
    datasets = pickle.load(f)
check('5 datasets', len(datasets) == 5, str(list(datasets)))

text_cols = {k: len(d['categorical_cols']) for k, d in datasets.items()}
check('categorical columns preserved as text', sum(text_cols.values()) > 0, str(text_cols))
# Text-level error types need somewhere substantial to act. An earlier version
# of this check demanded an ALL-categorical dataset, which only passed because of
# Blogger - a dataset the models could not learn from (AUC 0.559). What actually
# matters is that categoricals dominate at least one USABLE dataset.
majority_cat = [k for k, d in datasets.items()
                if len(d['categorical_cols']) > len(d['numeric_cols'])]
check('at least one majority-categorical dataset (for text-level errors)',
      bool(majority_cat), str(majority_cat))
check('at least one all-numeric dataset (numeric errors, no text confound)', any(
    len(d['categorical_cols']) == 0 for d in datasets.values()))

for k, d in datasets.items():
    binary = len(np.unique(d['y_train'])) == 2
    if not binary:
        failures.append(f'{k} target is not binary')
    idx_ok = list(d['X_train'].index) == list(range(len(d['X_train'])))
    if not idx_ok:
        failures.append(f'{k} index is not positional (injectors address rows by position)')
check('all targets binary, all indices positional', True)

# -------------------------------------------------------------- 4. injectors
print('\n4. Injectors')
src = open('src/error_injector.py', encoding='utf-8').read()
types = re.findall(r"'([a-z_]+)': self\.inject_", src)
check('28 error types registered', len(types) == 28, f'{len(types)}')
check('no duplicate registrations', len(types) == len(set(types)))

# --------------------------------------------------------------- 5. cleaners
print('\n5. Cleaners')
cleaner = data_cleaner.DataCleaner(42)
impl = set(cleaner._methods())
routes = data_cleaner.REMEDIATION_ROUTE
check('every error type has an explicit route', all(t in routes for t in types),
      str([t for t in types if t not in routes]))
auto = [t for t in types if routes[t][0] == data_cleaner.AUTOMATED]
check('every automated-route type has a cleaner', not set(auto) - impl,
      str(sorted(set(auto) - impl)))
check('no orphan cleaners', not impl - set(types), str(sorted(impl - set(types))))
notes.append(f'{len(impl)} cleaners for {len(auto)} automated-route types; '
             f'{len(types) - len(auto)} external_intervention')

# ------------------------------------------------------------ 6. audit files
print('\n6. Audits already run')
check('injector audit present', os.path.exists('results/injector_audit.csv'))
check('pipeline overlap audit present', os.path.exists('results/pipeline_overlap_audit.csv'))

if os.path.exists('results/pipeline_overlap_audit.csv'):
    ov = pd.read_csv('results/pipeline_overlap_audit.csv')

    neutral, dead = [], []
    for et, g in ov.groupby('Error_Type'):
        survives = bool(g.Corruption_Survives_Preprocessing.fillna(False).any())
        acts = bool(g.Cleaning_Has_Effect.fillna(False).any())

        if not survives:
            # Preprocessing already removes this corruption, so a cleaner with
            # no effect is correct: there is nothing left to repair.
            neutral.append(et)
        elif not acts:
            # Corruption reached the model but the cleaner changed nothing -
            # the affine-repair trap, and a genuine bug.
            dead.append(et)

    check('every surviving corruption has a cleaner that acts', not dead, str(dead))
    if neutral:
        notes.append(f'corruption neutralised by preprocessing (a finding, not a bug): {neutral}')

# -------------------------------------------------------------- 7. baselines
print('\n7. Baselines')
check('baseline file exists', os.path.exists('results/baseline_results.csv'))
if os.path.exists('results/baseline_results.csv'):
    b = pd.read_csv('results/baseline_results.csv')
    check('40 baseline rows (8 models x 5 datasets)', len(b) == 40, f'{len(b)}')
    check('baselines cover all 8 models', b.Model.nunique() == 8, f'{b.Model.nunique()}')

    # Degradation cannot be measured on a model with no signal to lose. Blogger
    # was dropped for exactly this: AUC 0.559 against 0.50 for random guessing,
    # so any "damage" it reported would have been noise.
    auc = b.groupby('Dataset').AUC.mean()
    weak = auc[auc < 0.65]
    check('every dataset has learnable signal (mean AUC >= 0.65)', weak.empty,
          str(auc.round(3).to_dict()))

# ------------------------------------------------------------- 8. stale files
print('\n8. Stale artefacts')
figs = os.listdir('figures') if os.path.isdir('figures') else []
check('no stale figures carried over', not figs, str(figs))

# ------------------------------------------------------------------- verdict
print('\n' + '=' * 78)
for n in notes:
    print(f'  note: {n}')
if failures:
    print(f'\n{len(failures)} CHECK(S) FAILED — do not run:')
    for f in failures:
        print(f'  - {f}')
    print('=' * 78)
    sys.exit(1)

print('\nALL CHECKS PASSED — safe to run the study')
print('=' * 78)
sys.exit(0)
