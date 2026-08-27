"""
Pipeline Overlap Audit — RUN THIS BEFORE THE STUDY, alongside audit_injectors.py
================================================================================

Answers two questions that injector-level checks cannot:

  1. Does the CORRUPTION survive preprocessing?
     An injector can visibly change the raw data and still have no effect on the
     model, because encoding/imputation/scaling can undo it. Standardisation
     removes any constant scale factor, so multiplying a column by 10 is
     invisible downstream. That is a real finding about the error type, but it
     must be identified deliberately rather than mistaken for "no impact".

  2. Does the CLEANING survive preprocessing?
     Same trap, and this one is a bug rather than a finding. Any repair of the
     form a*x + b is exactly cancelled by standardisation. An earlier version of
     clean_feature_noise shrank values toward the column median - affine, and so
     incapable of improving anything on any dataset. It looked reasonable and
     would have produced a confident, meaningless zero.

Both checks compare what the MODEL actually sees, not what the raw frame looks
like.

Output: results/pipeline_overlap_audit.csv
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from error_injector import ErrorInjector
from data_cleaner import DataCleaner
from preprocessing import prepare

SEVERITY = 'high'          # the level most likely to show an effect


def frames_differ(a, b, tol=1e-9):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return True
    return not np.allclose(a.values, b.values, atol=tol)


def main():
    with open('results/datasets_raw.pkl', 'rb') as f:
        datasets = pickle.load(f)

    cleaners = sorted(DataCleaner(42)._methods())
    rows = []

    for key, d in datasets.items():
        X, y = d['X_train'], d['y_train']
        cat, num = d['categorical_cols'], d['numeric_cols']
        clean_prepped, _ = prepare(X, d['X_test'], cat, num)

        for error_type in cleaners:
            injector = ErrorInjector(random_state=42)
            cleaner = DataCleaner(random_state=42)

            try:
                X_dirty, y_dirty, _ = injector.inject_error(X, y, error_type, SEVERITY)
                X_cleaned, y_cleaned, X_test_cleaned, meta = cleaner.clean(
                    X_dirty, y_dirty, error_type, d['X_test'])

                dirty_prepped, dirty_test = prepare(X_dirty, d['X_test'], cat, num)
                cleaned_prepped, cleaned_test = prepare(X_cleaned, X_test_cleaned, cat, num)

                corruption_survives = frames_differ(clean_prepped, dirty_prepped)
                cleaning_acts = frames_differ(dirty_prepped, cleaned_prepped)

                # TRAIN/TEST SCALE ALIGNMENT. This is the check that was missing
                # and let a -34pp bug through: a cleaner that fits a transform on
                # training data alone leaves the test set on a different scale,
                # and comparing only training matrices cannot see it. After a
                # correct clean, standardised test columns should sit near mean
                # 0 and std 1, like the training columns.
                note = ''
                tm = float(np.abs(cleaned_test.mean()).max())
                ts = float(cleaned_test.std().max())
                # Thresholds calibrated against observed behaviour: the real
                # bug produced a test mean of 21.2 and std of 11.7, while benign
                # train/test variation on small datasets reaches about 3.5 with
                # no measurable accuracy effect. 10 sits clearly between them.
                aligned = tm < 10.0 and ts < 10.0
                if not aligned:
                    note = f'TRAIN/TEST SCALE MISMATCH after cleaning (test |mean| max {tm:.2f}, std max {ts:.2f})'
            except Exception as e:
                corruption_survives = cleaning_acts = aligned = None
                note = f'{type(e).__name__}: {e}'

            rows.append({
                'Dataset': d['name'], 'Error_Type': error_type,
                'Corruption_Survives_Preprocessing': corruption_survives,
                'Cleaning_Has_Effect': cleaning_acts,
                'Train_Test_Aligned': None if corruption_survives is None else aligned,
                'Note': note,
            })

    audit = pd.DataFrame(rows)
    audit.to_csv('results/pipeline_overlap_audit.csv', index=False)

    print('=' * 96)
    print('PIPELINE OVERLAP AUDIT (severity = %s)' % SEVERITY)
    print('=' * 96)
    print(f"{'error_type':34s}{'corruption survives':>22s}{'cleaning acts':>16s}   verdict")
    print('-' * 96)

    neutralised, dead_cleaner = [], []
    for error_type in cleaners:
        sub = audit[audit.Error_Type == error_type]
        # An injector that cannot act on a given dataset (no numeric columns, or
        # no text columns) is correct behaviour, so judge on the best case.
        surv = bool(sub.Corruption_Survives_Preprocessing.fillna(False).any())
        acts = bool(sub.Cleaning_Has_Effect.fillna(False).any())

        if not surv:
            verdict = 'corruption neutralised by preprocessing'
            neutralised.append(error_type)
        elif not acts:
            verdict = 'CLEANER HAS NO EFFECT - check it is not affine'
            dead_cleaner.append(error_type)
        else:
            verdict = 'ok'

        print(f'{error_type:34s}{str(surv):>22s}{str(acts):>16s}   {verdict}')

    print('-' * 96)
    print(f'\nCorruption neutralised by preprocessing : {neutralised or "none"}')
    print('  -> a real finding: the error costs nothing once the standard')
    print('     pipeline runs. Report it, do not treat it as a broken injector.')
    print(f'\nCleaners with no effect                 : {dead_cleaner or "none"}')
    print('  -> investigate. An affine repair (a*x + b) is cancelled by')
    print('     standardisation and can never improve anything.')

    # The check that was missing, and that let a -34pp bug reach the results:
    # a cleaner fitting a transform on training data alone leaves the test set
    # on a different scale. Comparing training matrices cannot detect it.
    misaligned = sorted(audit[audit.Train_Test_Aligned == False].Error_Type.unique())
    print(f'\nTRAIN/TEST SCALE MISMATCH after cleaning : {misaligned or "none"}')
    print('  -> the cleaner fitted a transform on training data without applying')
    print('     it to test, so the model trains on one scale and is judged on another.')

    print('\nSaved: results/pipeline_overlap_audit.csv')
    return 1 if (dead_cleaner or misaligned) else 0


if __name__ == '__main__':
    sys.exit(main())
