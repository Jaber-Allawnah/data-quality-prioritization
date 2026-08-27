"""
Injector Audit — RUN THIS BEFORE THE STUDY
==========================================

Checks every injector on every dataset at every severity, and answers three
questions before a single experiment is run:

  1. Does it actually change the data?          (an inert injector yields a fake
                                                 "no impact" finding)
  2. Does the corruption grow with severity?    (a flat injector means low,
                                                 medium and high are the same
                                                 experiment run three times)
  3. Is the output still trainable?             (single-class targets, infinities
                                                 and runaway magnitudes break the
                                                 models rather than test them)

In the previous study this audit was written AFTER three full runs, and it found
two inert injectors, three that ignored severity entirely, and one producing
values of 1e152. All of that had already contaminated results. Running it first
costs minutes; running it last cost days.

Output: results/injector_audit.csv
"""

import pickle
import re
import sys
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector

SEVERITIES = ['low', 'medium', 'high']


def registered_error_types():
    """Read the dispatcher's own registry so the audit cannot drift from it."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, 'src', 'error_injector.py'), encoding='utf-8').read()
    return re.findall(r"'([a-z_]+)': self\.inject_", src)


def measure(X, y, Xc, yc):
    """
    Quantify how much an injection actually changed.

    Four separate signals, because no single one covers every injector and using
    only the first produces false alarms:

      cells_changed  - fraction of cells altered
      magnitude      - total absolute numeric shift. Needed because Gaussian
                       noise touches 100% of cells at EVERY severity, so the
                       fraction saturates while the amount still scales.
      renames        - columns whose NAME changed, which a value-based
                       comparison would miss entirely.
      row/col delta  - structural changes
    """
    row_delta = Xc.shape[0] - X.shape[0]
    col_delta = Xc.shape[1] - X.shape[1]

    cells_changed = np.nan
    labels_changed = np.nan
    magnitude = 0.0

    renames = sum(1 for a, b in zip(list(X.columns), list(Xc.columns)) if a != b)

    if row_delta == 0:
        shared = [c for c in X.columns if c in Xc.columns]
        if shared:
            a = X[shared].reset_index(drop=True).astype(str).values
            b = Xc[shared].reset_index(drop=True).astype(str).values
            # Compared as strings so text and numeric columns are handled
            # identically; NaN prints the same on both sides, so pre-existing
            # NaN is not counted as a change.
            cells_changed = (a != b).mean() * 100

            num = [c for c in shared if c in X.select_dtypes(include=[np.number]).columns]
            if num:
                av = X[num].reset_index(drop=True).to_numpy(dtype=float, na_value=np.nan)
                # to_numpy rather than stack()/unstack(): the latter reshapes the
                # whole frame twice per injection and dominated the audit's
                # runtime on the larger datasets.
                bv = Xc[num].reset_index(drop=True).apply(
                    pd.to_numeric, errors='coerce'
                ).to_numpy(dtype=float, na_value=np.nan)
                magnitude = float(np.abs(np.nan_to_num(bv) - np.nan_to_num(av)).sum())

        ya, yb = np.asarray(y), np.asarray(yc)
        if ya.shape == yb.shape:
            labels_changed = (ya != yb).mean() * 100

    return cells_changed, labels_changed, row_delta, col_delta, magnitude, renames


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    error_types = registered_error_types()
    total = len(error_types) * len(datasets) * len(SEVERITIES)
    print(f'Auditing {len(error_types)} error types x {len(datasets)} datasets '
          f'x {len(SEVERITIES)} severities = {total} injections\n')

    rows = []
    for key, d in datasets.items():
        X, y = d['X_train'], d['y_train']
        numeric = X.select_dtypes(include=[np.number])
        original_max = np.abs(numeric.values.astype(float)).max() if numeric.shape[1] else 1.0

        for error_type in error_types:
            for severity in SEVERITIES:
                injector = ErrorInjector(random_state=42)
                try:
                    Xc, yc, meta = injector.inject_error(X, y, error_type, severity)
                except Exception as e:
                    rows.append({'Dataset': d['name'], 'Error_Type': error_type,
                                 'Severity': severity, 'cells_changed_pct': np.nan,
                                 'labels_changed_pct': np.nan, 'row_delta': np.nan,
                                 'col_delta': np.nan, 'magnitude': np.nan, 'renames': np.nan,
                                 'problem': f'EXCEPTION {type(e).__name__}: {e}'})
                    continue

                cells, labels, rd, cd, mag, ren = measure(X, y, Xc, yc)

                problems = []
                if len(np.unique(np.asarray(yc))) < 2:
                    problems.append('single-class target')
                num = Xc.select_dtypes(include=[np.number])
                if num.shape[1]:
                    vals = num.values.astype(float)
                    if np.isinf(vals).any():
                        problems.append('infinities')
                    peak = np.nanmax(np.abs(vals)) if vals.size else 0
                    if peak > original_max * 1000:
                        problems.append(f'runaway magnitude {peak:.1e}')

                rows.append({
                    'Dataset': d['name'], 'Error_Type': error_type, 'Severity': severity,
                    'cells_changed_pct': cells, 'labels_changed_pct': labels,
                    'row_delta': rd, 'col_delta': cd,
                    'magnitude': mag, 'renames': ren,
                    'problem': '; '.join(problems),
                })

    audit = pd.DataFrame(rows)
    # Magnitude is log-compressed so a large numeric shift cannot swamp the
    # other signals, and renames are weighted so a metadata-only injector
    # registers as active.
    audit['total_change'] = (
        audit[['cells_changed_pct', 'labels_changed_pct']].fillna(0).sum(axis=1)
        + audit[['row_delta', 'col_delta']].abs().fillna(0).sum(axis=1)
        + np.log1p(audit['magnitude'].fillna(0))
        + audit['renames'].fillna(0) * 10
    )
    audit.to_csv('results/injector_audit.csv', index=False)

    print('=' * 104)
    print('INJECTOR AUDIT')
    print('=' * 104)
    print(f"{'error_type':34s}{'cells%':>8s}{'labels%':>9s}{'rowD':>7s}{'colD':>6s}"
          f"{'scales':>8s}   verdict")
    print('-' * 104)

    inert, flat, broken = [], [], []
    for error_type in registered_error_types():
        sub = audit[audit.Error_Type == error_type]

        by_sev = sub.groupby('Severity').total_change.mean().reindex(SEVERITIES)
        scales = bool(by_sev.is_monotonic_increasing and by_sev.iloc[-1] > by_sev.iloc[0])

        # "Inert" is judged per dataset: an injector that cannot act on a fully
        # numeric dataset is correct behaviour, not a defect. Only an injector
        # that does nothing ANYWHERE is broken.
        max_change = np.nan_to_num(sub.total_change).max()
        problems = sub[sub.problem.astype(bool)]

        if max_change == 0:
            verdict, _ = 'INERT everywhere', inert.append(error_type)
        elif problems.shape[0]:
            verdict, _ = f'BROKEN: {problems.problem.iloc[0][:34]}', broken.append(error_type)
        elif not scales:
            verdict, _ = 'does NOT scale with severity', flat.append(error_type)
        else:
            verdict = 'valid'

        print(f'{error_type:34s}{np.nan_to_num(sub.cells_changed_pct.mean()):8.2f}'
              f'{np.nan_to_num(sub.labels_changed_pct.mean()):9.2f}'
              f'{np.nan_to_num(sub.row_delta.abs().mean()):7.0f}'
              f'{np.nan_to_num(sub.col_delta.abs().mean()):6.0f}'
              f'{("yes" if scales else "NO"):>8s}   {verdict}')

    print('-' * 104)
    n = len(registered_error_types())
    print(f'\nVALID              : {n - len(inert) - len(flat) - len(broken)}/{n}')
    print(f'INERT everywhere   : {len(inert)}  {inert}')
    print(f'Does not scale     : {len(flat)}  {flat}')
    print(f'Broken output      : {len(broken)}  {broken}')

    # Which error types can act on which datasets - text injectors legitimately
    # do nothing on fully numeric data, and that must not read as a defect.
    print('\nPer-dataset coverage (blank = injector cannot act on that dataset):')
    cover = audit.pivot_table(index='Error_Type', columns='Dataset',
                              values='total_change', aggfunc='max').fillna(0)
    inactive = cover[(cover == 0).any(axis=1)]
    if len(inactive):
        for et, row in inactive.iterrows():
            dead = [c for c in cover.columns if row[c] == 0]
            print(f'  {et:34s} inactive on: {dead}')
    else:
        print('  every injector acts on every dataset')

    print('\nSaved: results/injector_audit.csv')
    return 1 if (inert or flat or broken) else 0


if __name__ == '__main__':
    sys.exit(main())
