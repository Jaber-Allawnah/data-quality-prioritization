"""
Data Loader — raw features, no encoding
=======================================

Loads each dataset and splits it, and stops there. Categorical columns are left
as TEXT.

This is the single structural difference from the previous pipeline, and the
reason for the rebuild. Encoding categoricals to integers inside the loader
meant errors were injected into an already-numeric matrix, so error types that
act on text - typographical errors, categorical errors - had nothing to operate
on and silently did nothing.

Real data quality errors occur at collection time, before any preprocessing. So
the order here is:

    load -> split -> INJECT -> encode -> scale -> train

rather than the previous:

    load -> encode -> scale -> split -> inject -> train

The second consequence matters too: encoding and scaling are now fitted on the
CORRUPTED training data, exactly as they would be in production where nobody
knows the data is dirty. Previously the scaler saw only clean data, so injected
outliers never influenced the scaling the way real ones would.

Column names are anonymised to feature_1..feature_n to match the original study,
with the categorical/numeric split recorded in metadata so downstream code knows
which is which without inspecting dtypes.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2


class DataLoader:
    """Load raw datasets, split, and leave all preprocessing to the pipeline."""

    def __init__(self, data_dir='data', random_state=RANDOM_STATE):
        self.data_dir = data_dir
        self.random_state = random_state

    # ------------------------------------------------------------------ utils

    def _path(self, filename):
        return os.path.join(self.data_dir, filename)

    def _finalise(self, X, y, name):
        """
        Anonymise column names, record which columns are categorical, split.

        The categorical/numeric split is captured HERE, before any corruption,
        so an injector that changes values can never change a column's declared
        type midway through the study.
        """
        X = X.copy()
        X.columns = [f'feature_{i}' for i in range(1, X.shape[1] + 1)]

        # Defined as "not numeric" rather than "== object": this pandas version
        # gives text columns a dedicated `str` dtype, so an object check finds
        # nothing and every dataset would look fully numeric - which is exactly
        # the silent failure this rebuild exists to remove.
        numeric = list(X.select_dtypes(include=[np.number]).columns)
        categorical = [c for c in X.columns if c not in numeric]

        # Categoricals are stored as plain strings so injectors can edit
        # characters without fighting pandas' categorical dtype.
        for c in categorical:
            X[c] = X[c].astype(str).str.strip()

        # Numeric columns are cast to float even when the source is integral.
        # pandas refuses to write a float into an int64 column, so an injector
        # writing 69.44 into an integer "age" raises TypeError. Feature matrices
        # are float in practice, so this matches real pipelines as well as
        # removing an entire class of dtype failure.
        for c in numeric:
            X[c] = X[c].astype(float)

        y = pd.Series(np.asarray(y).astype(int), name='target')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )

        # Reset to a positional index: several injectors address rows by
        # position, and a shuffled index would silently corrupt the wrong rows.
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        return {
            'name': name,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'categorical_cols': categorical,
            'numeric_cols': numeric,
            'n_rows': len(X),
            'class_balance': float(y.mean()),
        }

    # -------------------------------------------------------------- datasets

    def load_breast_cancer(self):
        """UCI Breast Cancer Wisconsin — all numeric, target is B/M."""
        df = pd.read_csv(self._path('wdbc.data'), header=None)
        y = (df[1] == 'M').astype(int)
        X = df.drop(columns=[0, 1])          # drop ID and diagnosis
        return self._finalise(X, y, 'Breast Cancer')

    def load_german_credit(self):
        """UCI German Credit — 13 categorical attributes kept as text."""
        df = pd.read_csv(self._path('german.data'), header=None, sep=r'\s+')
        y = (df.iloc[:, -1] == 1).astype(int)   # 1 = good credit
        X = df.iloc[:, :-1]
        return self._finalise(X, y, 'German Credit')

    def load_adult_income(self):
        """UCI Adult — 8 categorical attributes kept as text. '?' is missing."""
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country', 'income']
        df = pd.read_csv(self._path('adult.data'), names=columns,
                         sep=r',\s*', engine='python', na_values='?')
        # Rows with missing values are dropped so the study injects missingness
        # deliberately rather than inheriting it, matching the original design.
        df = df.dropna()
        y = (df['income'] == '>50K').astype(int)
        X = df.drop(columns=['income'])
        return self._finalise(X, y, 'Adult Income')

    def load_bank_marketing(self):
        """UCI Bank Marketing — 9 categorical attributes kept as text."""
        df = pd.read_csv(self._path('bank-full.csv'), sep=';')
        y = (df['y'] == 'yes').astype(int)
        X = df.drop(columns=['y'])
        return self._finalise(X, y, 'Bank Marketing')

    def load_telco_churn(self):
        """
        Telco Customer Churn (IBM sample, via OpenML) — 7,043 customers.

        Added to replace the Blogger dataset, which the models could not learn
        from (baseline AUC 0.559 against 0.50 for random). This one carries the
        most categorical columns in the study - 15 of 19 features - which makes
        it the primary test bed for the text-level error types.

        Parsed with pandas rather than scipy.io.arff: the attributes are declared
        STRING rather than NOMINAL, which scipy's reader handles poorly. Values
        containing spaces are single-quoted in the file, hence quotechar.

        Two columns need care:
          customerID    - an identifier, dropped. Left in, it is a unique key per
                          row and models can memorise it.
          TotalCharges  - declared STRING and blank for customers with tenure 0.
                          Coerced to numeric; the resulting rows are dropped so
                          the study injects missingness deliberately rather than
                          inheriting it, matching how Adult Income is handled.
        """
        path = self._path('dataset.arff')

        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()

        header = [ln for ln in lines if ln.strip().upper().startswith('@ATTRIBUTE')]
        names = [ln.split()[1] for ln in header]
        start = next(i for i, ln in enumerate(lines)
                     if ln.strip().upper().startswith('@DATA')) + 1

        from io import StringIO
        df = pd.read_csv(StringIO(''.join(lines[start:])), names=names,
                         quotechar="'", skipinitialspace=True)

        df = df.drop(columns=['customerID'])
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])

        y = (df['Churn'].astype(str).str.strip() == 'Yes').astype(int)
        X = df.drop(columns=['Churn'])
        return self._finalise(X, y, 'Telco Churn')

    def load_blogger(self):
        """
        OpenML "Blogger" — 100 instances, 5 nominal attributes.

        NAMING CORRECTION: the previous study labelled this file "NASA CM1" and
        described it as software defect prediction. The ARFF metadata identifies
        it as `blogger`, with attributes such as (high/low/medium),
        (left/middle/right) and (impression/news/political/scientific/tourism).
        It is not a software engineering dataset, and the earlier claim of
        cross-domain software-quality validation was unsupported.

        The data is unchanged - 100 rows, ~32% positive - only the name and
        domain claim are corrected. It is fully categorical, which makes it the
        strongest test of the text-level error types this rebuild enables.
        """
        from scipy.io import arff
        data, _ = arff.loadarff(self._path('php7gmqTJ.arff'))
        df = pd.DataFrame(data)

        # ARFF strings arrive as bytes.
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)

        target = df.columns[-1]
        y = df[target].isin(['Y', 'yes', 'true', '1', 1, True]).astype(int)
        X = df.drop(columns=[target])
        return self._finalise(X, y, 'Blogger')

    # ------------------------------------------------------------------- all

    def load_all(self):
        datasets = {
            'breast_cancer': self.load_breast_cancer(),
            'german_credit': self.load_german_credit(),
            'adult_income': self.load_adult_income(),
            'bank_marketing': self.load_bank_marketing(),
            'telco_churn': self.load_telco_churn(),
        }

        print(f"{'dataset':18s}{'rows':>8s}{'features':>10s}{'categorical':>13s}"
              f"{'numeric':>9s}{'class 1':>9s}")
        print('-' * 68)
        for key, d in datasets.items():
            print(f"{key:18s}{d['n_rows']:8d}{d['X_train'].shape[1]:10d}"
                  f"{len(d['categorical_cols']):13d}{len(d['numeric_cols']):9d}"
                  f"{d['class_balance']:9.3f}")

        return datasets


if __name__ == '__main__':
    import pickle
    loader = DataLoader()
    datasets = loader.load_all()
    with open('results/datasets_raw.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    print('\nSaved: results/datasets_raw.pkl (raw, unencoded)')
