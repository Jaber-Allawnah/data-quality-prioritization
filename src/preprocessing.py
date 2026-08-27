"""
Preprocessing — encode, impute, scale
=====================================

Runs AFTER error injection, and is fitted on the (possibly corrupted) training
data only. That ordering is the point of the rebuild:

  * Injectors see raw text, so text-level error types can actually corrupt
    something instead of silently doing nothing.
  * The encoder and scaler observe the corrupted data, exactly as they would in
    production where nobody knows the data is dirty. Previously the scaler was
    fitted on clean data, so an injected outlier never widened the standard
    deviation the way a real one does.

Never fitted on the test set. The test set is only ever transformed, using
statistics learned from training - anything else leaks test information into
training and inflates every result.
"""

import numpy as np
import pandas as pd


MISSING_TOKEN = '__missing__'
UNSEEN_CODE = -1


class Preprocessor:
    """Ordinal-encode categoricals, mean-impute, then standardise."""

    def __init__(self, categorical_cols, numeric_cols):
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.category_maps_ = {}
        self.means_ = {}
        self.stds_ = {}
        self.columns_ = None

    # ------------------------------------------------------------------ fit

    def fit(self, X):
        """Learn category codes, imputation means and scaling from X only."""
        X = X.copy()
        self.columns_ = list(X.columns)

        # Categorical columns present in this frame. Recomputed rather than
        # assumed: an injector may add columns (redundant_features), so the
        # declared list can be out of date.
        cat_cols = [c for c in self.columns_ if c in self.categorical_cols]
        num_cols = [c for c in self.columns_ if c not in cat_cols]

        for col in cat_cols:
            values = X[col].astype(str).fillna(MISSING_TOKEN)
            # Sorted for determinism: dict iteration order would otherwise make
            # the codes depend on row order.
            categories = sorted(values.unique())
            self.category_maps_[col] = {v: i for i, v in enumerate(categories)}

        for col in num_cols:
            series = pd.to_numeric(X[col], errors='coerce')
            series = series.replace([np.inf, -np.inf], np.nan)
            mean = series.mean()
            # An all-NaN column has no mean; 0.0 keeps it present but inert
            # rather than letting it drop and change the matrix width.
            self.means_[col] = 0.0 if pd.isna(mean) else float(mean)

        return self

    # ------------------------------------------------------------ transform

    def transform(self, X):
        """Apply the learned encoding, imputation and scaling."""
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        for col in self.columns_:
            if col not in X.columns:
                # Column absent from this frame (e.g. test set lacks a column an
                # injector added to training). Fill with the training mean so the
                # matrix width still matches.
                out[col] = self.means_.get(col, 0.0)
                continue

            if col in self.category_maps_:
                values = X[col].astype(str).fillna(MISSING_TOKEN)
                # Categories never seen in training map to a single reserved
                # code. Typos and invented categories land here, which is
                # precisely the production behaviour being simulated.
                out[col] = values.map(self.category_maps_[col]).fillna(UNSEEN_CODE)
            else:
                series = pd.to_numeric(X[col], errors='coerce')
                series = series.replace([np.inf, -np.inf], np.nan)
                out[col] = series.fillna(self.means_.get(col, 0.0))

        out = out.astype(float)

        # Scaling is computed here rather than in fit() so that the statistics
        # come from the fully encoded matrix, categorical codes included.
        if not self.stds_:
            for col in out.columns:
                std = out[col].std()
                self.stds_[col] = 1.0 if (std == 0 or pd.isna(std)) else float(std)
                self.means_[f'__scale_mean__{col}'] = float(out[col].mean())

        for col in out.columns:
            centre = self.means_.get(f'__scale_mean__{col}', 0.0)
            out[col] = (out[col] - centre) / self.stds_.get(col, 1.0)

        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def prepare(X_train, X_test, categorical_cols, numeric_cols):
    """
    Fit on the training data, transform both.

    Returns aligned frames the models can consume directly.
    """
    pre = Preprocessor(categorical_cols, numeric_cols)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    # Guarantee identical column sets and order; a mismatch here surfaces later
    # as an unreadable shape error inside a model.
    X_test_p = X_test_p[X_train_p.columns]
    return X_train_p, X_test_p
