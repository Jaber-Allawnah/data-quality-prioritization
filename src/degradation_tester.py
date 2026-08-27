"""
Degradation Tester
==================

Trains every model on corrupted training data and evaluates on the CLEAN test
set, so the number produced is the generalisation cost of the corruption rather
than the difficulty of a corrupted test set.

The pipeline order is the substantive change from the previous study:

    corrupted raw data -> encode -> impute -> scale -> train

Preprocessing is fitted on the CORRUPTED training fold, exactly as it would be
in production where nobody knows the data is dirty. Previously the encoder and
scaler were fitted on clean data before injection, so injected outliers never
widened the standard deviation the way real ones do.

Models come from the shared factory in models.py, so the definitions here can
never drift from the baseline script's - a divergence that silently corrupted
360 rows of the previous study.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from models import build_models
from preprocessing import prepare

warnings.filterwarnings('ignore')


class DegradationTester:
    """Measure model performance on corrupted training data."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.failures = []

    def evaluate(self, X_train_dirty, y_train_dirty, X_test, y_test,
                 categorical_cols, numeric_cols, dataset_name, error_type,
                 severity, metadata, condition):
        """
        Preprocess, train every model, and record metrics.

        `condition` distinguishes the same combination measured before and after
        cleaning ('dirty' / 'cleaned'), so both live in one table without
        needing to be joined later.
        """
        try:
            X_tr, X_te = prepare(X_train_dirty, X_test, categorical_cols, numeric_cols)
        except Exception as e:
            self.failures.append({
                'Dataset': dataset_name, 'Error_Type': error_type,
                'Severity': severity, 'Model': 'ALL', 'Condition': condition,
                'Error': f'preprocessing failed: {type(e).__name__}: {e}',
            })
            return 0, 0

        y_tr = np.asarray(y_train_dirty)
        y_te = np.asarray(y_test)

        models = build_models(self.random_state)
        n_ok = 0

        for model_name, model in models.items():
            try:
                start = time.time()
                model.fit(X_tr, y_tr)
                train_seconds = time.time() - start

                y_pred = model.predict(X_te)
                y_proba = model.predict_proba(X_te)[:, 1]

                row = {
                    'Dataset': dataset_name,
                    'Error_Type': error_type,
                    'Severity': severity,
                    'Model': model_name,
                    'Condition': condition,
                    'Accuracy': accuracy_score(y_te, y_pred),
                    'Precision': precision_score(y_te, y_pred, zero_division=0),
                    'Recall': recall_score(y_te, y_pred, zero_division=0),
                    'F1': f1_score(y_te, y_pred, zero_division=0),
                    'AUC': roc_auc_score(y_te, y_proba),
                    'Train_Seconds': train_seconds,
                    'N_Train_Rows': len(X_tr),
                    'N_Features': X_tr.shape[1],
                    'Metadata': str(metadata),
                }
            except Exception as e:
                # One model failing must not cost the other seven.
                self.failures.append({
                    'Dataset': dataset_name, 'Error_Type': error_type,
                    'Severity': severity, 'Model': model_name,
                    'Condition': condition,
                    'Error': f'{type(e).__name__}: {e}',
                })
                continue

            self.results.append(row)
            n_ok += 1

        return n_ok, len(models)

    def save(self, filepath):
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f'\nSaved {len(df)} rows -> {filepath}')

        if self.failures:
            fail_path = filepath.replace('.csv', '_failures.csv')
            pd.DataFrame(self.failures).to_csv(fail_path, index=False)
            print(f'{len(self.failures)} failures -> {fail_path}')
        return df
