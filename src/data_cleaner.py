"""
Data Cleaner — remediation for RQ2
==================================

Supplies the number RQ2 needs and degradation alone cannot give:

    ROI = (Cleaned - Dirty) / Cost

Assuming instead that cleaning restores the clean baseline makes the improvement
identical to the degradation already measured, so the ROI ranking would just be
the impact ranking rescaled by cost - a framework with no measured benefit in it.

Two rules govern every cleaner here:

1. NO ORACLE INFORMATION. A cleaner never sees the uncorrupted data, and is
   never told which rows or columns were corrupted, because a practitioner would
   not have that. Cleaners operate on the corrupted TRAINING fold only.

2. CLEANING HAPPENS ON RAW DATA, before encoding and scaling. That is what
   allows text-level repairs - fuzzy-matching a typo back to a known category is
   impossible once the column is an integer.

Error types with no principled automated repair are declared as such rather than
faked. "This error can only be prevented, or fixed by a human" is a finding the
cost model needs, not a gap.
"""

import difflib

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer


# ---------------------------------------------------------------- taxonomy

AUTOMATED = 'automated_cleaning'
EXTERNAL = 'external_intervention'
PREVENTION = 'prevention_only'

EVALUATED = 'evaluated'
NOT_EVALUATED = 'not_evaluated_under_protocol'

# Correlation above which a feature is treated as leaking the target. Fixed in
# advance of running the recovery experiment, never tuned against its output.
LEAKAGE_CORRELATION_THRESHOLD = 0.5

# Similarity above which a typo is matched back to a known category.
FUZZY_MATCH_THRESHOLD = 0.75

# Conceptual remediation route for EVERY error type, assigned by explicit
# judgement rather than by default. In the previous study any error type without
# an implemented cleaner silently fell through to "prevention only", which turned
# "no cleaner was written" into the claim "prevention is the only option" and
# handed those types an unjustified ROI penalty.
#
# EXTERNAL does not mean impossible - these errors are repaired routinely in
# practice, just not by an automated pipeline, and so carry a different kind of
# cost that the ROI model must price separately.
REMEDIATION_ROUTE = {
    # -- repairable automatically ------------------------------------------
    'missing_values': (AUTOMATED, 'imputation'),
    'missing_values_mnar': (AUTOMATED, 'imputation, though MNAR limits what is recoverable'),
    'duplicates': (AUTOMATED, 'exact-match deduplication'),
    'duplicates_targeted': (AUTOMATED, 'exact-match deduplication'),
    'duplicates_minority': (AUTOMATED, 'exact-match deduplication'),
    'outliers': (AUTOMATED, 'winsorisation'),
    'outliers_targeted': (AUTOMATED, 'winsorisation'),
    'invalid_values': (AUTOMATED, 'sentinel detection then imputation'),
    'domain_violations': (AUTOMATED, 'range rules'),
    'contextual_errors': (AUTOMATED, 'plausibility rules'),
    'redundant_features': (AUTOMATED, 'correlation-based feature selection'),
    'feature_noise': (AUTOMATED, 'robust smoothing'),
    'feature_noise_targeted': (AUTOMATED, 'robust smoothing'),
    'data_leakage': (AUTOMATED, 'correlation screening then feature removal'),
    # Expect zero damage AND zero recovery, and both are correct. The injector
    # multiplies columns by a constant, and the pipeline standardises before
    # training, which mathematically undoes any constant factor. Scale
    # heterogeneity is therefore neutralised for free by standard preprocessing.
    # (The previous study reported 3.00pp damage for this error type, but its
    # scaler was fitted on clean data BEFORE injection, so the distortion
    # survived artificially.)
    'data_heterogeneity': (AUTOMATED, 'unit harmonisation - already performed by the '
                                      'standard preprocessing, so no separate '
                                      'remediation cost applies'),
    'class_imbalance': (AUTOMATED, 'resampling - alters the training distribution '
                                   'rather than repairing wrong values'),
    'imbalanced_feature_distribution': (AUTOMATED, 'distribution transformation'),
    'typographical_errors': (AUTOMATED, 'fuzzy matching against known categories'),
    'categorical_errors': (AUTOMATED, 'category validation against a controlled vocabulary'),
    'data_inconsistency': (AUTOMATED, 'rule-based consistency checks'),

    # -- repairable only outside an automated pipeline ----------------------
    'label_noise': (EXTERNAL, 'requires re-labelling; a wrong label is a legal '
                              'value and cannot be identified from the data alone'),
    'label_noise_asymmetric': (EXTERNAL, 'requires re-labelling'),
    'annotator_bias': (EXTERNAL, 'requires re-annotation, ideally by a second annotator'),
    'ambiguous_labels': (EXTERNAL, 'boundary cases need expert adjudication - they are '
                                   'genuinely ambiguous rather than mistaken'),
    'concept_drift': (EXTERNAL, 'the labels were correct when collected; only newer '
                                'data from the current distribution can fix this'),
    'data_drift': (EXTERNAL, 'requires newer data from the current distribution'),
    'data_representativeness': (EXTERNAL, 'requires collecting data from under-represented '
                                          'groups; it cannot be synthesised'),
    'provenance_issues': (EXTERNAL, 'requires verifying sources outside the dataset'),
}



class DataCleaner:
    """Realistic practitioner responses, applied to raw (pre-encoding) data."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    # ------------------------------------------------------- numeric repairs

    def _numeric(self, X):
        return list(X.select_dtypes(include=[np.number]).columns)

    def _categorical(self, X):
        return [c for c in X.columns if c not in self._numeric(X)]

    def clean_missing_values(self, X, y):
        """
        KNN imputation for numeric columns, modal fill for categorical.

        Fitted on the corrupted training fold only - never the test set, which
        would leak test information into the very experiment measuring
        leakage-free repair. The target is never used as an imputation input,
        which would leak label information into the features.
        """
        X_clean = X.copy()
        num = self._numeric(X_clean)
        cat = self._categorical(X_clean)

        for col in cat:
            if X_clean[col].isna().any():
                mode = X_clean[col].mode()
                X_clean[col] = X_clean[col].fillna(mode.iloc[0] if len(mode) else 'unknown')

        if num and X_clean[num].isna().any().any():
            imputer = KNNImputer(n_neighbors=5)
            X_clean[num] = imputer.fit_transform(X_clean[num])

        return X_clean, y

    def clean_duplicates(self, X, y):
        """Drop exact duplicate rows - the one near-complete repair available."""
        combined = X.copy()
        combined['__label__'] = np.asarray(y)
        deduped = combined.drop_duplicates()
        y_clean = deduped['__label__'].reset_index(drop=True)
        X_clean = deduped.drop(columns='__label__').reset_index(drop=True)
        return X_clean, y_clean

    def clean_outliers(self, X, y, X_test=None):
        """
        Winsorise numeric columns to the 1st/99th percentile.

        Caps extremes rather than deleting rows, so sample size and label
        distribution are preserved. Note this fails when the corruption rate
        exceeds the trimming percentage: the corrupted values themselves move
        the quantiles, so the rule can no longer distinguish them.
        """
        X_clean = X.copy()
        test_clean = None if X_test is None else X_test.copy()

        for col in self._numeric(X_clean):
            low, high = X_clean[col].quantile([0.01, 0.99])
            X_clean[col] = X_clean[col].clip(low, high)
            # Bounds learned from training are applied to test as well: a
            # production pipeline clips incoming data with the same thresholds.
            if test_clean is not None and col in test_clean.columns:
                test_clean[col] = test_clean[col].clip(low, high)

        return (X_clean, y) if X_test is None else (X_clean, y, test_clean)

    def clean_extreme_values_iqr(self, X, y, X_test=None, k=1.5):
        """
        Clip numeric columns to IQR-based bounds: [Q1 - k*IQR, Q3 + k*IQR].

        Percentile winsorisation cannot be used here. Domain violations are
        injected at up to 15% of cells, so the corrupted values BECOME the 1st
        and 99th percentiles - clipping to them is a no-op. On German Credit the
        1%/99% bounds were exactly the injected -30 and 106, while the true range
        was 4 to 72.

        The interquartile range is estimated from the middle half of the data and
        so survives contamination in the tails, which is precisely the situation
        here.

        The trade-off is real and worth reporting: IQR bounds are tighter than
        the observed range, so legitimate extreme values are clipped too. That
        false-positive cost belongs in the ROI model rather than being hidden.
        """
        X_clean = X.copy()
        test_clean = None if X_test is None else X_test.copy()

        for col in self._numeric(X_clean):
            q1, q3 = X_clean[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            low, high = q1 - k * iqr, q3 + k * iqr
            X_clean[col] = X_clean[col].clip(low, high)
            if test_clean is not None and col in test_clean.columns:
                test_clean[col] = test_clean[col].clip(low, high)

        return (X_clean, y) if X_test is None else (X_clean, y, test_clean)

    def clean_invalid_values(self, X, y):
        """
        Replace sentinel codes with the column median, but only where the
        sentinel is genuinely out of range for that column.

        Blind replacement of -999 / -1 / 9999 is unsafe: these are legitimate
        values in some datasets. Bank Marketing encodes "never previously
        contacted" as pdays = -1, and replacing it cost 12.8 percentage points
        of accuracy - the cleaner destroyed far more real information than the
        corruption ever did.

        A candidate is therefore treated as a sentinel only when it falls outside
        [Q1 - 3*IQR, Q3 + 3*IQR] for its own column. That is how a practitioner
        distinguishes "9999 in a column ranging 0-100" from "-1 among values
        that legitimately include -1".
        """
        X_clean = X.copy()
        candidates = [-999, -1, 9999]

        for col in self._numeric(X_clean):
            series = X_clean[col]
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            low, high = q1 - 3 * iqr, q3 + 3 * iqr

            sentinels = [v for v in candidates
                         if (series == v).any() and (v < low or v > high)]
            if not sentinels:
                continue

            mask = series.isin(sentinels)
            replacement = series[~mask].median()
            X_clean.loc[mask, col] = replacement

        return X_clean, y

    def clean_redundant_features(self, X, y, threshold=0.95):
        """Drop one of each pair of near-perfectly correlated numeric features."""
        X_clean = X.copy()
        num = self._numeric(X_clean)
        if len(num) < 2:
            return X_clean, y
        corr = X_clean[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > threshold)]
        if drop and len(drop) < X_clean.shape[1]:
            X_clean = X_clean.drop(columns=drop)
        return X_clean, y

    def clean_feature_noise(self, X, y, X_test=None):
        """
        Rank-order median smoothing: denoise within each numeric column.

        Values are sorted, a rolling median is taken over the sorted order, and
        the result is mapped back. Nearby values in a column are treated as
        neighbours, which is a defensible assumption; nearby ROWS are not, which
        is why an ordinary rolling window was rejected - row order is arbitrary
        in tabular data.

        The transform MUST be non-affine. An earlier version shrank each value
        toward the column median (0.7*x + 0.3*median), which is affine - and the
        pipeline standardises afterwards, so the scaling undid the repair
        exactly. It could never have improved anything, on any dataset. Anything
        of the form a*x + b is invisible downstream of standardisation.

        Recovery is partial by construction: noise cannot be separated from
        signal without knowing which is which.
        """
        X_clean = X.copy()
        window = 5

        for col in self._numeric(X_clean):
            series = X_clean[col]
            if series.nunique() < window:
                continue
            order = series.sort_values().index
            smoothed = (series.loc[order]
                        .rolling(window, min_periods=1, center=True)
                        .median())
            X_clean[col] = smoothed.reindex(X_clean.index)

        # The test set is deliberately NOT smoothed. Smoothing repairs noise
        # already present in the training data; it is not a transform the
        # feature space carries into inference, and applying it to clean test
        # data would be denoising something that was never corrupted.
        return (X_clean, y) if X_test is None else (X_clean, y, X_test)

    def clean_data_leakage(self, X, y, threshold=LEAKAGE_CORRELATION_THRESHOLD):
        """
        Drop features suspiciously correlated with the target.

        The standard practitioner screen, using only information available
        without an oracle. A genuinely strong predictor can be discarded by it -
        that false-positive risk is a real cost of the method, and part of what
        the experiment measures rather than a flaw in it.
        """
        X_clean = X.copy()
        y_arr = np.asarray(y, dtype=float)
        suspicious = []

        for col in self._numeric(X_clean):
            values = X_clean[col].values.astype(float)
            if np.std(values) == 0:
                continue
            corr = np.corrcoef(values, y_arr)[0, 1]
            if not np.isnan(corr) and abs(corr) > threshold:
                suspicious.append(col)

        if suspicious and len(suspicious) < X_clean.shape[1]:
            X_clean = X_clean.drop(columns=suspicious)
        return X_clean, y

    def clean_data_heterogeneity(self, X, y, X_test=None):
        """Re-standardise numeric columns, removing arbitrary per-source scales."""
        X_clean = X.copy()
        test_clean = None if X_test is None else X_test.copy()

        for col in self._numeric(X_clean):
            mean, std = X_clean[col].mean(), X_clean[col].std()
            if not std or np.isnan(std):
                continue
            X_clean[col] = (X_clean[col] - mean) / std
            # The SAME training statistics are applied to test. Rescaling train
            # alone leaves the two frames on different scales, which destroys
            # accuracy far more thoroughly than the corruption ever did.
            if test_clean is not None and col in test_clean.columns:
                test_clean[col] = (test_clean[col] - mean) / std

        return (X_clean, y) if X_test is None else (X_clean, y, test_clean)

    def clean_imbalanced_feature_distribution(self, X, y, X_test=None):
        """
        Re-spread concentrated numeric distributions by rank.

        Expect partial recovery at best: the injector REPLACES values with random
        draws, so the information is destroyed rather than distorted, and no
        transform can restore values that no longer exist.
        """
        X_clean = X.copy()
        test_clean = None if X_test is None else X_test.copy()

        num = self._numeric(X_clean)
        if not num:
            return (X_clean, y) if X_test is None else (X_clean, y, test_clean)

        transformer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, max(10, len(X_clean))),
            random_state=self.random_state,
        )
        X_clean[num] = transformer.fit_transform(X_clean[num])
        # Fitted on training, applied to test - the transform is part of the
        # feature space and must follow the data to inference.
        if test_clean is not None:
            shared = [c for c in num if c in test_clean.columns]
            if len(shared) == len(num):
                test_clean[num] = transformer.transform(test_clean[num])

        return (X_clean, y) if X_test is None else (X_clean, y, test_clean)

    def clean_class_imbalance(self, X, y):
        """
        Oversample the minority class back toward parity.

        Remediation rather than cleaning: it alters the training distribution
        instead of repairing incorrect values.
        """
        y_arr = np.asarray(y)
        counts = pd.Series(y_arr).value_counts()
        if len(counts) < 2:
            return X, y

        majority, minority = counts.index[0], counts.index[-1]
        deficit = int(counts[majority] - counts[minority])
        if deficit <= 0:
            return X, y

        pool = np.where(y_arr == minority)[0]
        rng = np.random.RandomState(self.random_state)
        picks = rng.choice(pool, deficit, replace=True)

        X_clean = pd.concat([X, X.iloc[picks]], ignore_index=True)
        y_clean = pd.concat([pd.Series(y_arr), pd.Series(y_arr[picks])], ignore_index=True)
        return X_clean, y_clean

    # --------------------------------------------------------- text repairs

    def clean_typographical_errors(self, X, y):
        """
        Fuzzy-match rare category values back to frequent ones.

        A typo produces a value seen once or twice while the correct spelling
        appears thousands of times. Values below a frequency floor are matched
        against the frequent vocabulary and snapped to the closest match above
        the similarity threshold.

        Only possible because cleaning now runs on RAW text. Once a column has
        been encoded to integers, "marired" and "married" are just two unrelated
        codes and no string similarity exists to exploit.
        """
        X_clean = X.copy()

        for col in self._categorical(X_clean):
            counts = X_clean[col].value_counts()
            if len(counts) < 2:
                continue

            # A value appearing in under 1% of rows is treated as suspect; the
            # rest form the trusted vocabulary.
            floor = max(2, int(len(X_clean) * 0.01))
            frequent = counts[counts >= floor].index.tolist()
            rare = counts[counts < floor].index.tolist()

            if not frequent or not rare:
                continue

            mapping = {}
            for value in rare:
                match = difflib.get_close_matches(
                    str(value), [str(f) for f in frequent],
                    n=1, cutoff=FUZZY_MATCH_THRESHOLD
                )
                if match:
                    mapping[value] = match[0]

            if mapping:
                X_clean[col] = X_clean[col].replace(mapping)

        return X_clean, y

    def clean_categorical_errors(self, X, y):
        """
        Validate categories against the observed vocabulary.

        This one is expected to recover LITTLE, and that is the finding rather
        than a weak method: the injector swaps a valid category for another
        valid category, so the result passes every validation rule. Only values
        outside the vocabulary can be caught, and a swap never produces one.

        The contrast with typographical errors - same columns, same rate, but
        one produces invalid values and the other does not - isolates
        detectability from damage.
        """
        X_clean = X.copy()
        for col in self._categorical(X_clean):
            counts = X_clean[col].value_counts()
            if len(counts) < 2:
                continue
            floor = max(2, int(len(X_clean) * 0.005))
            valid = counts[counts >= floor].index.tolist()
            if not valid:
                continue
            # Anything outside the vocabulary becomes the column mode.
            X_clean[col] = X_clean[col].where(X_clean[col].isin(valid), counts.index[0])
        return X_clean, y

    def clean_data_inconsistency(self, X, y, X_test=None):
        """Consistency pass: winsorise numerics and normalise category spelling."""
        if X_test is None:
            X_clean, _ = self.clean_outliers(X, y)
            X_clean, _ = self.clean_typographical_errors(X_clean, y)
            return X_clean, y

        X_clean, _, test_clean = self.clean_outliers(X, y, X_test)
        X_clean, _ = self.clean_typographical_errors(X_clean, y)
        return X_clean, y, test_clean

    # -------------------------------------------------------------- dispatch

    def _methods(self):
        return {
            'missing_values': ('KNN imputation', self.clean_missing_values),
            'missing_values_mnar': ('KNN imputation', self.clean_missing_values),
            'duplicates': ('drop exact duplicates', self.clean_duplicates),
            'duplicates_targeted': ('drop exact duplicates', self.clean_duplicates),
            'duplicates_minority': ('drop exact duplicates', self.clean_duplicates),
            'outliers': ('IQR clipping', self.clean_extreme_values_iqr),
            'outliers_targeted': ('IQR clipping', self.clean_extreme_values_iqr),
            'invalid_values': ('sentinel to NaN then impute', self.clean_invalid_values),
            'domain_violations': ('IQR clipping', self.clean_extreme_values_iqr),
            'contextual_errors': ('IQR clipping', self.clean_extreme_values_iqr),
            'redundant_features': ('drop correlated features', self.clean_redundant_features),
            'feature_noise': ('rank-order median smoothing', self.clean_feature_noise),
            'feature_noise_targeted': ('rank-order median smoothing', self.clean_feature_noise),
            'data_leakage': (f'drop features with |corr| > {LEAKAGE_CORRELATION_THRESHOLD}',
                             self.clean_data_leakage),
            'data_heterogeneity': ('re-standardise features', self.clean_data_heterogeneity),
            'imbalanced_feature_distribution': ('quantile transform',
                                                self.clean_imbalanced_feature_distribution),
            'class_imbalance': ('oversample minority', self.clean_class_imbalance),
            'typographical_errors': ('fuzzy match to known categories',
                                     self.clean_typographical_errors),
            'categorical_errors': ('validate against vocabulary',
                                   self.clean_categorical_errors),
            'data_inconsistency': ('consistency rules', self.clean_data_inconsistency),
        }

    def clean(self, X, y, error_type, X_test=None):
        """
        Apply the cleaning method matched to an error type.

        Returns (X_clean, y_clean, metadata) where metadata carries the
        conceptual route and whether recovery was actually measured. Those are
        deliberately separate: the route says what the right remedy IS, the
        status says whether this study measured it.
        """
        # No silent default. METHODOLOGY.md states routes are assigned by
        # explicit judgement for all 28 types with no fallback, so an unrouted
        # type must fail loudly rather than acquire a route by accident. A silent
        # default is what previously turned "no cleaner was written" into the
        # published claim "prevention is the only option".
        if error_type not in REMEDIATION_ROUTE:
            raise KeyError(
                f'{error_type!r} has no explicit remediation route. Add one to '
                f'REMEDIATION_ROUTE - do not let it default.'
            )
        route, reason = REMEDIATION_ROUTE[error_type]
        methods = self._methods()

        if error_type not in methods:
            if route == AUTOMATED:
                # Repairable in principle but not implemented here. Recovery
                # must read as NOT MEASURED, never as zero, or the error type is
                # penalised in the ROI ranking for a gap in implementation
                # rather than an absence of benefit.
                status, note = NOT_EVALUATED, 'automated repair possible but not implemented'
            else:
                # No automated repair exists to write, so zero recovery under
                # the automated protocol is a genuine, informative measurement.
                status, note = EVALUATED, 'no automated repair exists; remedy lies outside the pipeline'
            meta = {
                'remediation_route': route,
                'recovery_evaluation_status': status,
                'method': 'none',
                'reason': f'{reason} - {note}',
            }
            return (X, y, meta) if X_test is None else (X, y, X_test, meta)

        name, fn = methods[error_type]

        # Cleaners that fit a transform accept the test frame and return it
        # transformed with the SAME fitted parameters. Those that only repair
        # values in place ignore it.
        import inspect
        takes_test = 'X_test' in inspect.signature(fn).parameters

        if takes_test and X_test is not None:
            X_clean, y_clean, test_clean = fn(X, y, X_test)
        else:
            X_clean, y_clean = fn(X, y)
            test_clean = X_test

        meta = {
            'remediation_route': route,
            'recovery_evaluation_status': EVALUATED,
            'method': name,
            'reason': reason,
            'rows_before': len(X),
            'rows_after': len(X_clean),
            'test_transformed': bool(takes_test and X_test is not None),
        }
        return (X_clean, y_clean, meta) if X_test is None else (X_clean, y_clean, test_clean, meta)
