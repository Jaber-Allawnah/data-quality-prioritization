"""
Error Injector - Complete Version with All 24 Data Quality Error Types
Systematically injects 24 data quality errors into clean datasets
"""

import hashlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Implementation version per error type. Bump whenever an injector's SEMANTICS
# change, so results produced by different implementations can never be pooled
# by accident. Absent from this map means version 1, the original implementation.
#
#   data_leakage v2 (2026-08-08): rewritten from cross-row feature contamination
#       (which injected no target information at all) to overwriting a feature
#       with target-derived values. v1 and v2 measure different phenomena and
#       their results are NOT comparable.
ERROR_TYPE_VERSIONS = {
    'data_leakage': 2,
}


class ErrorInjector:
    """Inject various data quality errors into datasets"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def _seed_for(self, error_type, severity_level, X):
        """
        Derive a deterministic seed unique to this specific experiment.

        Seeding only once in __init__ makes every injection draw from one shared
        stream, so each result depends on how many random numbers the injections
        before it happened to consume. Changing any one injector then silently
        changes the data produced for all the others, and running an error type
        alone gives different data than running it inside the full loop.

        Seeding per experiment instead makes each (error_type, severity, dataset)
        independent of call order and stable across code changes elsewhere.

        SHA-256 is used rather than hash(), whose string hashing is randomised
        per process unless PYTHONHASHSEED is fixed, which would make runs differ.
        """
        key = f'{self.random_state}|{error_type}|{severity_level}|{X.shape[0]}x{X.shape[1]}'
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16)
    
    # =========================================================================
    # ERROR TYPE 1: MISSING VALUES (MCAR - Missing Completely At Random)
    # =========================================================================
    
    def inject_missing_values(self, X, y, missing_rate=0.10):
        """
        Inject missing values (NaN) into dataset
        
        Parameters:
        - missing_rate: Percentage of data to set as NaN (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Sensors fail, data not collected, transmission error
        """
        X_corrupted = X.copy()
        
        total_cells = X_corrupted.shape[0] * X_corrupted.shape[1]
        n_missing = int(total_cells * missing_rate / 100)
        
        rows = np.random.choice(X_corrupted.shape[0], n_missing, replace=True)
        cols = np.random.choice(X_corrupted.shape[1], n_missing, replace=True)
        
        for r, c in zip(rows, cols):
            X_corrupted.iloc[r, c] = np.nan
        
        actual_missing_pct = ((X_corrupted.isna().sum().sum() / total_cells * 100)
                              if total_cells else 0.0)
        
        return X_corrupted, y, {
            'error_type': 'missing_values',
            'intended_rate': missing_rate,
            'actual_rate': actual_missing_pct,
            'cells_corrupted': n_missing
        }
    
    
    # =========================================================================
    # ERROR TYPE 2: LABEL NOISE (Random Label Flipping)
    # =========================================================================
    
    def inject_label_noise(self, X, y, noise_rate=0.05):
        """
        Inject label noise by randomly flipping labels
        
        Parameters:
        - noise_rate: Percentage of labels to flip (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Mislabeled data, annotation error, typo in target
        """
        y_corrupted = y.copy()
        
        unique_classes = np.unique(y_corrupted)
        
        if len(unique_classes) != 2:
            raise ValueError("Label noise only supports binary classification")
        
        n_flip = int(len(y_corrupted) * noise_rate / 100)
        indices_to_flip = np.random.choice(len(y_corrupted), n_flip, replace=False)
        
        for idx in indices_to_flip:
            y_corrupted.iloc[idx] = 1 - y_corrupted.iloc[idx]
        
        return X, y_corrupted, {
            'error_type': 'label_noise',
            'intended_rate': noise_rate,
            'actual_rate': (n_flip / len(y)) * 100,
            'labels_flipped': n_flip
        }
    
    
    # =========================================================================
    # ERROR TYPE 3: DUPLICATE RECORDS
    # =========================================================================
    
    def inject_duplicates(self, X, y, duplicate_rate=0.05):
        """
        Inject duplicate records by appending copies of random rows
        
        Parameters:
        - duplicate_rate: Percentage of current dataset size to duplicate (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Data loaded twice, copy-paste error, ETL bug
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        
        n_duplicates = int(len(X_corrupted) * duplicate_rate / 100)
        indices_to_duplicate = np.random.choice(len(X_corrupted), n_duplicates, replace=True)
        
        X_duplicated = X_corrupted.iloc[indices_to_duplicate]
        y_duplicated = y_corrupted.iloc[indices_to_duplicate]
        
        X_corrupted = pd.concat([X_corrupted, X_duplicated], ignore_index=True)
        y_corrupted = pd.concat([y_corrupted, y_duplicated], ignore_index=True)
        
        return X_corrupted, y_corrupted, {
            'error_type': 'duplicates',
            'intended_rate': duplicate_rate,
            'actual_rate': (n_duplicates / len(X)) * 100,
            'duplicates_added': n_duplicates,
            'original_rows': len(X),
            'corrupted_rows': len(X_corrupted)
        }
    
    
    # =========================================================================
    # ERROR TYPE 4: OUTLIERS (Statistical Anomalies)
    # =========================================================================
    
    def inject_outliers(self, X, y, outlier_rate=0.01):
        """
        Inject outliers using extreme values (3-5 standard deviations away)
        
        Parameters:
        - outlier_rate: Percentage of numerical values to make extreme (0.01 = 1%)
        
        Severity levels: 1%, 5%, 10%
        
        Simulation: Sensor malfunction, data entry error, measurement error
        """
        X_corrupted = X.copy()
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        total_numerical_cells = len(X_corrupted) * len(numerical_cols)
        n_outliers = int(total_numerical_cells * outlier_rate / 100)
        
        outliers_added = 0
        
        for col in numerical_cols:
            col_mean = X_corrupted[col].mean()
            col_std = X_corrupted[col].std()
            
            n_col_outliers = int(len(X_corrupted) * outlier_rate / 100)
            
            if n_col_outliers > 0:
                outlier_indices = np.random.choice(len(X_corrupted), n_col_outliers, replace=False)
                
                for idx in outlier_indices:
                    extreme_value = col_mean + np.random.choice([-1, 1]) * 4 * col_std
                    X_corrupted.iloc[idx, X_corrupted.columns.get_loc(col)] = extreme_value
                    outliers_added += 1
        
        return X_corrupted, y, {
            'error_type': 'outliers',
            'intended_rate': outlier_rate,
            'actual_rate': ((outliers_added / total_numerical_cells * 100)
                            if total_numerical_cells else 0.0),
            'outliers_added': outliers_added
        }
    
    
    # =========================================================================
    # ERROR TYPE 5: INVALID VALUES (Domain Constraint Violations)
    # =========================================================================
    
    def inject_invalid_values(self, X, y, invalid_rate=0.02):
        """
        Inject invalid/out-of-range values that violate domain constraints
        
        Parameters:
        - invalid_rate: Percentage of values to make invalid (0.02 = 2%)
        
        Severity levels: 2%, 5%, 10%
        
        Simulation: Wrong unit, out of range, negative where impossible,
                   data type error, encoding error
        """
        X_corrupted = X.copy()
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        total_cells = len(X_corrupted) * len(numerical_cols)
        n_invalid = int(total_cells * invalid_rate / 100)
        
        rows = np.random.choice(len(X_corrupted), n_invalid, replace=True)
        col_indices = np.random.choice(len(numerical_cols), n_invalid, replace=True)
        
        invalid_values_added = 0
        col_list = list(numerical_cols)
        
        for r, c_idx in zip(rows, col_indices):
            if c_idx < len(col_list):
                col = col_list[c_idx]
                invalid_value = np.random.choice([-999, -1, 9999])
                X_corrupted.at[r, col] = float(invalid_value)
                invalid_values_added += 1
        
        return X_corrupted, y, {
            'error_type': 'invalid_values',
            'intended_rate': invalid_rate,
            'actual_rate': ((invalid_values_added / total_cells * 100)
                            if total_cells else 0.0),
            'invalid_values_added': invalid_values_added
        }
    
    
    # =========================================================================
    # ERROR TYPE 6: CLASS IMBALANCE (Artificially skew class distribution)
    # =========================================================================
    
    def inject_class_imbalance(self, X, y, imbalance_rate=0.10):
        """
        Create class imbalance by removing samples from minority class
        
        Parameters:
        - imbalance_rate: How much to reduce minority class (0.10 = 10% removed)
        
        Severity levels: 10%, 25%, 40%
        
        Simulation: Undersampling of rare events, collection bias
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        
        class_counts = y_corrupted.value_counts()
        minority_class = class_counts.idxmin()
        
        minority_indices = np.where(y_corrupted == minority_class)[0]
        
        if len(minority_indices) == 0:
            return X_corrupted, y_corrupted, {
                'error_type': 'class_imbalance',
                'intended_rate': imbalance_rate,
                'minority_removed': 0
            }
        
        n_to_remove = int(len(minority_indices) * imbalance_rate / 100)
        
        if n_to_remove > 0:
            indices_to_remove = np.random.choice(minority_indices, min(n_to_remove, len(minority_indices)), replace=False)
            
            mask = np.ones(len(X_corrupted), dtype=bool)
            mask[indices_to_remove] = False
            
            X_corrupted = X_corrupted.iloc[mask].reset_index(drop=True)
            y_corrupted = y_corrupted.iloc[mask].reset_index(drop=True)
        
        return X_corrupted, y_corrupted, {
            'error_type': 'class_imbalance',
            'intended_rate': imbalance_rate,
            'minority_removed': n_to_remove,
            'original_rows': len(X),
            'corrupted_rows': len(X_corrupted)
        }
    
    
    # =========================================================================
    # ERROR TYPE 7: FEATURE NOISE (Add random noise to features)
    # =========================================================================
    
    def inject_feature_noise(self, X, y, noise_rate=0.05):
        """
        Add random noise to feature values (Gaussian noise)
        
        Parameters:
        - noise_rate: Standard deviation of noise as % of feature std (0.05 = 5%)
        
        Severity levels: 5%, 15%, 25%
        
        Simulation: Measurement error, sensor drift, rounding error
        """
        X_corrupted = X.copy()
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        noise_added = 0
        
        for col in numerical_cols:
            col_std = X_corrupted[col].std()
            noise_std = col_std * (noise_rate / 100)
            
            noise = np.random.normal(0, noise_std, len(X_corrupted))
            X_corrupted[col] = X_corrupted[col] + noise
            noise_added += len(X_corrupted)
        
        return X_corrupted, y, {
            'error_type': 'feature_noise',
            'intended_rate': noise_rate,
            'noise_added_to_cells': noise_added,
            'numerical_columns': len(numerical_cols)
        }
    
    
    # =========================================================================
    # ERROR TYPE 8: CATEGORICAL ENCODING ERRORS
    # =========================================================================
    
    def inject_categorical_errors(self, X, y, encoding_error_rate=0.05):
        """
        Replace a category with a DIFFERENT VALID category from the same column.

        e.g. marital status "married" -> "single". The value stays legal, so no
        validation rule catches it, but the record now says something untrue and
        the model learns the wrong association. This is a category ASSIGNMENT
        error, distinct from a typo, which produces an invalid value (see
        inject_typographical_errors).

        Requires raw text columns. Under the previous pipeline, which encoded
        categoricals to integers before injection, this injector found no text
        and silently did nothing - its 0.00% result was the absence of an
        experiment rather than a finding.
        """
        X_corrupted = X.copy()
        categorical_cols = [c for c in X_corrupted.columns
                            if c not in X_corrupted.select_dtypes(include=[np.number]).columns]

        if len(categorical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'categorical_errors',
                'categorical_columns_found': 0,
                'cells_corrupted': 0,
                'note': 'dataset has no categorical columns'
            }

        # Rate applies across all categorical cells, matching how every other
        # injector defines its rate.
        total_cells = len(X_corrupted) * len(categorical_cols)
        n_errors = int(total_cells * encoding_error_rate / 100)

        rows = np.random.choice(len(X_corrupted), n_errors, replace=True)
        cols = np.random.choice(len(categorical_cols), n_errors, replace=True)

        # Pools come from the ORIGINAL column, so a replacement can never be a
        # value some earlier injection introduced.
        pools = {c: sorted(X[c].astype(str).unique()) for c in categorical_cols}

        cells_corrupted = 0
        for r, ci in zip(rows, cols):
            col = categorical_cols[ci]
            pool = pools[col]
            if len(pool) < 2:
                continue
            pos = X_corrupted.columns.get_loc(col)
            current = str(X_corrupted.iat[r, pos])
            alternatives = [v for v in pool if v != current]
            if not alternatives:
                continue
            X_corrupted.iat[r, pos] = alternatives[np.random.randint(len(alternatives))]
            cells_corrupted += 1

        return X_corrupted, y, {
            'error_type': 'categorical_errors',
            'mechanism': 'valid category swapped for another valid category',
            'intended_rate': encoding_error_rate,
            'categorical_columns': len(categorical_cols),
            'cells_corrupted': cells_corrupted
        }

    def inject_data_inconsistency(self, X, y, inconsistency_rate=0.05):
        """
        Create data inconsistencies (conflicting or contradictory values)
        
        Parameters:
        - inconsistency_rate: Percentage of rows to make inconsistent (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Conflicting information in different fields, 
                   logic errors (e.g., age=5 but work-years=10)
        """
        X_corrupted = X.copy()
        
        n_inconsistent = int(len(X_corrupted) * inconsistency_rate / 100)
        inconsistent_indices = np.random.choice(len(X_corrupted), min(n_inconsistent, len(X_corrupted)), replace=False)
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns.tolist()
        
        inconsistencies_added = 0
        
        for idx in inconsistent_indices:
            if len(numerical_cols) >= 2:
                col1, col2 = np.random.choice(numerical_cols, 2, replace=False)
                
                val1 = X_corrupted.at[idx, col1]
                val2 = X_corrupted.at[idx, col2]
                
                X_corrupted.at[idx, col1] = val2
                X_corrupted.at[idx, col2] = val1
                
                inconsistencies_added += 1
        
        return X_corrupted, y, {
            'error_type': 'data_inconsistency',
            'intended_rate': inconsistency_rate,
            'inconsistencies_added': inconsistencies_added,
            'rows_affected': n_inconsistent
        }
    
    
    # =========================================================================
    # ERROR TYPE 10: TYPOGRAPHICAL ERRORS
    # =========================================================================
    
    def inject_typographical_errors(self, X, y, typo_rate=0.05):
        """
        Character-level typos in categorical text, producing INVALID values.

        e.g. "married" -> "marired". Unlike a category swap the result is not a
        legal category at all, so it becomes an unseen level the encoder must
        handle - which is what a data-entry mistake does to a real pipeline.

        Four mistake types are simulated: transposition, deletion, duplication
        and adjacent-key substitution.

        Numeric columns are left alone. The previous implementation added
        Gaussian noise to numbers and called it a typo, which measured feature
        noise under a different name.
        """
        X_corrupted = X.copy()
        categorical_cols = [c for c in X_corrupted.columns
                            if c not in X_corrupted.select_dtypes(include=[np.number]).columns]

        if len(categorical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'typographical_errors',
                'categorical_columns_found': 0,
                'typos_added': 0,
                'note': 'dataset has no categorical columns'
            }

        total_cells = len(X_corrupted) * len(categorical_cols)
        n_typos = int(total_cells * typo_rate / 100)

        rows = np.random.choice(len(X_corrupted), n_typos, replace=True)
        cols = np.random.choice(len(categorical_cols), n_typos, replace=True)

        # Rough QWERTY adjacency so substitutions resemble real slips.
        neighbours = {
            'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg',
            'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
            'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'q': 'wa', 'r': 'et',
            's': 'ad', 't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
            'y': 'tu', 'z': 'x',
        }

        typos_added = 0
        for r, ci in zip(rows, cols):
            col = categorical_cols[ci]
            pos = X_corrupted.columns.get_loc(col)
            text = str(X_corrupted.iat[r, pos])

            if len(text) < 2:
                X_corrupted.iat[r, pos] = text + 'x'
                typos_added += 1
                continue

            kind = np.random.randint(4)
            chars = list(text)
            i = np.random.randint(len(chars) - 1)

            if kind == 0:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]      # transpose
            elif kind == 1:
                chars.pop(i)                                          # drop
            elif kind == 2:
                chars.insert(i, chars[i])                             # duplicate
            else:
                lower = chars[i].lower()                              # wrong key
                if lower in neighbours:
                    options = neighbours[lower]
                    chars[i] = options[np.random.randint(len(options))]
                else:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]

            X_corrupted.iat[r, pos] = ''.join(chars)
            typos_added += 1

        return X_corrupted, y, {
            'error_type': 'typographical_errors',
            'mechanism': 'character-level typos creating invalid categories',
            'intended_rate': typo_rate,
            'categorical_columns': len(categorical_cols),
            'typos_added': typos_added
        }

    def inject_data_drift(self, X, y, drift_rate=0.10):
        """
        Simulate data drift by gradually shifting feature distributions
        
        Parameters:
        - drift_rate: Percentage of rows to apply drift to (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Data distribution changes over time, feature scaling changes
        """
        X_corrupted = X.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        n_drift_rows = int(len(X_corrupted) * drift_rate / 100)
        drift_start_idx = len(X_corrupted) - n_drift_rows
        
        for col in numerical_cols:
            col_mean = X_corrupted[col].mean()
            col_std = X_corrupted[col].std()
            
            drift_amount = 0.5 * col_std
            X_corrupted.loc[drift_start_idx:, col] = X_corrupted.loc[drift_start_idx:, col] + drift_amount
        
        return X_corrupted, y, {
            'error_type': 'data_drift',
            'intended_rate': drift_rate,
            'actual_rate': drift_rate,
            'rows_affected': n_drift_rows,
            'drift_amount_std': 0.5
        }
    
    
    # =========================================================================
    # ERROR TYPE 12: DATA LEAKAGE (Test set information in training)
    # =========================================================================
    
    def inject_data_leakage(self, X, y, leakage_rate=0.10):
        """
        Overwrite one feature with target-derived values.

        REIMPLEMENTED 2026-08-08. The previous version copied feature values
        between random rows, which is cross-row contamination, not leakage: no
        variable carried target information, so there was nothing a practitioner
        could detect and nothing with leakage's characteristic signature.
        Results produced by that version are NOT comparable to these and must
        not be pooled with them.

        Leakage proper means a feature encodes the answer during training but
        does not behave that way at prediction time. Here the least predictive
        feature is overwritten with a corrupted copy of the target, so the model
        learns to lean on a shortcut that the untouched test set will not honour.

        An EXISTING column is overwritten rather than a new one appended, so the
        training and test schemas stay identical and the test set keeps its
        genuine values - which is precisely the production/training mismatch
        being simulated.

        Severity raises the fidelity of the leak, and with it the correlation
        that makes it detectable:
            low  (10%): agrees with the target 65% of the time (r ~ 0.30)
            med  (20%): 80% (r ~ 0.60)
            high (30%): 95% (r ~ 0.90)
        """
        X_corrupted = X.copy()
        y_arr = np.asarray(y)

        numerical_cols = list(X_corrupted.select_dtypes(include=[np.number]).columns)
        if not numerical_cols:
            return X_corrupted, y, {'error_type': 'data_leakage', 'leaked_feature': None}

        # Sacrifice the WEAKEST feature, so the gain comes from the injected
        # shortcut rather than from destroying a genuinely useful predictor.
        ranked = [c for c in self._feature_importance(X, y) if c in numerical_cols]
        leaked_col = ranked[-1]

        agreement = min(0.5 + (leakage_rate / 100) * 1.5, 0.98)

        reveals = np.random.rand(len(y_arr)) < agreement
        leaked_values = np.where(reveals, y_arr, 1 - y_arr).astype(float)

        # Scale onto the column's own range so the leak is not detectable by
        # magnitude alone - only by its correlation with the target.
        col_min, col_max = X[leaked_col].min(), X[leaked_col].max()
        X_corrupted[leaked_col] = col_min + leaked_values * (col_max - col_min)

        actual_corr = abs(np.corrcoef(X_corrupted[leaked_col].values.astype(float), y_arr)[0, 1])

        return X_corrupted, y, {
            'error_type': 'data_leakage',
            'mechanism': 'weakest feature overwritten with target-derived values',
            'intended_rate': leakage_rate,
            'leaked_feature': leaked_col,
            'agreement_with_target': agreement,
            'correlation_with_target': float(actual_corr),
        }
    
    
    # =========================================================================
    # ERROR TYPE 13: REDUNDANT FEATURES (Correlated/duplicate features)
    # =========================================================================
    
    def inject_redundant_features(self, X, y, redundancy_rate=0.10):
        """
        Inject redundant features by duplicating or creating highly correlated features
        
        Parameters:
        - redundancy_rate: Percentage of features to duplicate (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Duplicate columns, highly correlated features
        """
        X_corrupted = X.copy()
        
        n_features_to_duplicate = int(X_corrupted.shape[1] * redundancy_rate / 100)
        
        if n_features_to_duplicate == 0:
            n_features_to_duplicate = 1
        
        original_feature_names = list(X_corrupted.columns)
        features_to_duplicate = np.random.choice(
            original_feature_names, 
            min(n_features_to_duplicate, len(original_feature_names)), 
            replace=False
        )
        
        numeric_cols = set(X_corrupted.select_dtypes(include=[np.number]).columns)
        redundant_features_added = 0

        for feature in features_to_duplicate:
            duplicate_name = f'{feature}_dup'

            if feature in numeric_cols:
                # Near-duplicate: the original plus slight noise, which is how
                # redundancy usually appears (a re-derived or re-measured field).
                noise = np.random.normal(0, X_corrupted[feature].std() * 0.05,
                                         len(X_corrupted))
                X_corrupted[duplicate_name] = X_corrupted[feature] + noise
            else:
                # Text columns are copied verbatim: std is undefined for strings,
                # and an exactly duplicated categorical column is the realistic
                # form of categorical redundancy.
                X_corrupted[duplicate_name] = X_corrupted[feature]

            redundant_features_added += 1
        
        X_corrupted.columns = [str(c) for c in X_corrupted.columns]
        
        return X_corrupted, y, {
            'error_type': 'redundant_features',
            'intended_rate': redundancy_rate,
            'features_added': redundant_features_added,
            'original_features': X_corrupted.shape[1] - redundant_features_added,
            'new_shape': X_corrupted.shape
        }
    
    
    # =========================================================================
    # ERROR TYPE 14: ANNOTATOR BIAS (Systematic label bias)
    # =========================================================================
    
    def inject_annotator_bias(self, X, y, bias_rate=0.10):
        """
        Inject systematic annotator bias (labels biased for certain patterns)
        
        Parameters:
        - bias_rate: Percentage of dataset to bias labeling (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Annotators label more strictly for certain groups
        """
        X_corrupted = X.copy()
        y_biased = y.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            n_bias = int(len(y_biased) * bias_rate / 100)
            bias_indices = np.random.choice(len(y_biased), min(n_bias, len(y_biased)), replace=False)
            for idx in bias_indices:
                y_biased.iloc[idx] = 1 - y_biased.iloc[idx]
            
            return X_corrupted, y_biased, {
                'error_type': 'annotator_bias',
                'intended_rate': bias_rate,
                'labels_biased': n_bias,
                'bias_pattern': 'random (no numerical columns)'
            }
        
        bias_col = numerical_cols[0]
        
        threshold = X_corrupted[bias_col].quantile(1 - bias_rate / 100)
        high_value_indices = X_corrupted[X_corrupted[bias_col] >= threshold].index
        
        n_to_bias = int(len(high_value_indices) * 0.5)
        indices_to_bias = np.random.choice(high_value_indices, min(n_to_bias, len(high_value_indices)), replace=False)
        
        for idx in indices_to_bias:
            y_biased.iloc[idx] = 1 - y_biased.iloc[idx]
        
        return X_corrupted, y_biased, {
            'error_type': 'annotator_bias',
            'intended_rate': bias_rate,
            'labels_biased': len(indices_to_bias),
            'bias_pattern': f'high-value samples in {bias_col}'
        }
    
    
    # =========================================================================
    # ERROR TYPE 15: AMBIGUOUS LABELS (Hard-to-classify boundary cases)
    # =========================================================================
    
    def inject_ambiguous_labels(self, X, y, ambiguity_rate=0.05):
        """
        Inject label ambiguity by flipping labels of boundary-case samples
        
        Parameters:
        - ambiguity_rate: Percentage of labels to make ambiguous (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Ambiguous samples at decision boundary
        """
        X_corrupted = X.copy()
        y_ambiguous = y.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            n_ambiguous = int(len(y_ambiguous) * ambiguity_rate / 100)
            ambiguous_indices = np.random.choice(len(y_ambiguous), min(n_ambiguous, len(y_ambiguous)), replace=False)
            for idx in ambiguous_indices:
                y_ambiguous.iloc[idx] = 1 - y_ambiguous.iloc[idx]
            
            return X_corrupted, y_ambiguous, {
                'error_type': 'ambiguous_labels',
                'intended_rate': ambiguity_rate,
                'ambiguous_labels': n_ambiguous,
                'boundary_method': 'random (no numerical columns)'
            }
        
        feature_distances = []
        
        for idx in range(len(X_corrupted)):
            distances = []
            for col in numerical_cols:
                median_val = X_corrupted[col].median()
                distance = abs(X_corrupted.iloc[idx][col] - median_val)
                distances.append(distance)
            
            avg_distance = np.mean(distances)
            feature_distances.append(avg_distance)
        
        feature_distances = np.array(feature_distances)
        
        n_boundary = int(len(X_corrupted) * ambiguity_rate / 100)
        boundary_indices = np.argsort(feature_distances)[:min(n_boundary, len(X_corrupted))]
        
        for idx in boundary_indices:
            y_ambiguous.iloc[idx] = 1 - y_ambiguous.iloc[idx]
        
        return X_corrupted, y_ambiguous, {
            'error_type': 'ambiguous_labels',
            'intended_rate': ambiguity_rate,
            'ambiguous_labels': len(boundary_indices),
            'boundary_method': 'median distance in feature space'
        }
    
    
    # =========================================================================
    # ERROR TYPE 16: DOMAIN VALUE VIOLATIONS
    # =========================================================================
    
    def inject_domain_violations(self, X, y, violation_rate=0.05):
        """
        Inject values that violate domain constraints/business rules
        
        Parameters:
        - violation_rate: Percentage of values to violate domain (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Impossible values (age > 150, temperature < -273C)
        """
        X_corrupted = X.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'domain_violations',
                'violations_added': 0
            }
        
        total_cells = len(X_corrupted) * len(numerical_cols)
        n_violations = int(total_cells * violation_rate / 100)
        
        # Size both arrays by n_violations. zip() pairs them elementwise and stops
        # at the shorter one, so capping col_indices at the column count would pin
        # every severity to exactly len(numerical_cols) violations.
        rows = np.random.choice(len(X_corrupted), n_violations, replace=True)
        col_indices = np.random.choice(len(numerical_cols), n_violations, replace=True)
        
        violations_added = 0
        col_list = list(numerical_cols)

        # Column bounds are taken from the ORIGINAL data, once, before any
        # injection. Reading them from X_corrupted inside the loop would feed
        # each violation back into the range used by the next one, compounding
        # the values exponentially instead of holding them just outside the
        # domain.
        col_bounds = {}
        for col in col_list:
            c_min, c_max = X[col].min(), X[col].max()
            col_bounds[col] = (c_min, c_max, (c_max - c_min) if c_max != c_min else 1)

        for r, c_idx in zip(rows, col_indices):
            if c_idx < len(col_list):
                col = col_list[c_idx]
                col_min, col_max, col_range = col_bounds[col]

                violation_value = np.random.choice([
                    col_min - 0.5 * col_range,
                    col_max + 0.5 * col_range
                ])

                X_corrupted.at[r, col] = float(violation_value)
                violations_added += 1
        
        return X_corrupted, y, {
            'error_type': 'domain_violations',
            'intended_rate': violation_rate,
            'actual_rate': (violations_added / total_cells) * 100,
            'violations_added': violations_added
        }
    
    
    # =========================================================================
    # ERROR TYPE 17: DATA HETEROGENEITY (Mixed data types and formats)
    # =========================================================================
    
    def inject_data_heterogeneity(self, X, y, heterogeneity_rate=0.10):
        """
        Introduce data heterogeneity by using different scales/units
        
        Parameters:
        - heterogeneity_rate: Percentage of features to change scale (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Data from different sources with different scales/units,
                   inconsistent measurement standards, data heterogeneity
        """
        X_corrupted = X.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'data_heterogeneity',
                'features_scaled': 0
            }
        
        n_features_to_scale = int(len(numerical_cols) * heterogeneity_rate / 100)
        
        if n_features_to_scale == 0:
            n_features_to_scale = 1
        
        features_to_scale = np.random.choice(list(numerical_cols), min(n_features_to_scale, len(numerical_cols)), replace=False)
        
        features_scaled = 0
        
        for col in features_to_scale:
            # Apply different scale to simulate data from different source
            scale_factor = np.random.uniform(0.1, 10.0)
            X_corrupted[col] = X_corrupted[col] * scale_factor
            features_scaled += 1
        
        return X_corrupted, y, {
            'error_type': 'data_heterogeneity',
            'intended_rate': heterogeneity_rate,
            'features_scaled': features_scaled,
            'heterogeneity_type': 'different scales/units'
        }
    
    
    # =========================================================================
    # ERROR TYPE 18: CONCEPT DRIFT (Labels/relationships change over time)
    # =========================================================================
    
    def inject_concept_drift(self, X, y, drift_rate=0.10):
        """
        Simulate concept drift by changing label patterns over time
        
        Parameters:
        - drift_rate: Percentage of rows (latter rows) to apply drift to (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Labels change meaning over time, relationships shift,
                   different labeling criteria applied later
        """
        X_corrupted = X.copy()
        y_drifted = y.copy()
        
        n_drift_rows = int(len(y_drifted) * drift_rate / 100)
        drift_start_idx = len(y_drifted) - n_drift_rows
        
        # Flip labels for latter rows to simulate concept drift
        labels_changed = 0
        
        for idx in range(drift_start_idx, len(y_drifted)):
            y_drifted.iloc[idx] = 1 - y_drifted.iloc[idx]
            labels_changed += 1
        
        return X_corrupted, y_drifted, {
            'error_type': 'concept_drift',
            'intended_rate': drift_rate,
            'rows_drifted': n_drift_rows,
            'labels_changed': labels_changed,
            'drift_start_index': drift_start_idx
        }
    
    
    # =========================================================================
    # ERROR TYPE 19: DATA REPRESENTATIVENESS
    # =========================================================================
    
    def inject_data_representativeness(self, X, y, representativeness_rate=0.10):
        """
        Reduce dataset representativeness by removing underrepresented samples
        
        Parameters:
        - representativeness_rate: Percentage of minority samples to remove (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Dataset doesn't match production distribution
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        
        class_counts = y_corrupted.value_counts()
        minority_class = class_counts.idxmin()
        
        minority_indices = np.where(y_corrupted == minority_class)[0]
        
        if len(minority_indices) == 0:
            return X_corrupted, y_corrupted, {
                'error_type': 'data_representativeness',
                'intended_rate': representativeness_rate,
                'samples_removed': 0
            }
        
        n_to_remove = int(len(minority_indices) * representativeness_rate / 100)
        
        if n_to_remove > 0:
            indices_to_remove = np.random.choice(minority_indices, min(n_to_remove, len(minority_indices)), replace=False)
            
            mask = np.ones(len(X_corrupted), dtype=bool)
            mask[indices_to_remove] = False
            
            X_corrupted = X_corrupted.iloc[mask].reset_index(drop=True)
            y_corrupted = y_corrupted.iloc[mask].reset_index(drop=True)
        
        return X_corrupted, y_corrupted, {
            'error_type': 'data_representativeness',
            'intended_rate': representativeness_rate,
            'samples_removed': n_to_remove,
            'original_rows': len(X),
            'corrupted_rows': len(X_corrupted)
        }
    
    
    # =========================================================================
    # ERROR TYPE 20: IMBALANCED FEATURE DISTRIBUTION
    # =========================================================================
    
    def inject_imbalanced_feature_distribution(self, X, y, imbalance_rate=0.10):
        """
        Create imbalanced feature distributions by concentrating values
        
        Parameters:
        - imbalance_rate: Percentage of features to imbalance (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Features with extreme value concentrations
        """
        X_corrupted = X.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'imbalanced_feature_distribution',
                'features_imbalanced': 0
            }
        
        n_features_to_imbalance = int(len(numerical_cols) * imbalance_rate / 100)
        
        if n_features_to_imbalance == 0:
            n_features_to_imbalance = 1
        
        features_to_imbalance = np.random.choice(list(numerical_cols), min(n_features_to_imbalance, len(numerical_cols)), replace=False)
        
        imbalanced_count = 0
        
        for col in features_to_imbalance:
            col_min = X_corrupted[col].min()
            col_max = X_corrupted[col].max()
            col_range = col_max - col_min if col_max != col_min else 1
            
            narrow_min = col_min
            narrow_max = col_min + 0.3 * col_range
            
            n_to_concentrate = int(len(X_corrupted) * 0.7)
            concentrate_indices = np.random.choice(len(X_corrupted), n_to_concentrate, replace=False)
            
            for idx in concentrate_indices:
                concentrated_value = np.random.uniform(narrow_min, narrow_max)
                X_corrupted.at[idx, col] = concentrated_value
            
            imbalanced_count += 1
        
        return X_corrupted, y, {
            'error_type': 'imbalanced_feature_distribution',
            'intended_rate': imbalance_rate,
            'features_imbalanced': imbalanced_count
        }
    
    
    # =========================================================================
    # ERROR TYPE 21: MISSING METADATA
    def inject_provenance_issues(self, X, y, provenance_contamination_rate=0.10):
        """
        Introduce provenance issues by mixing data from different sources
        
        Parameters:
        - provenance_contamination_rate: Percentage of rows to contaminate (0.10 = 10%)
        
        Severity levels: 10%, 20%, 30%
        
        Simulation: Data from unknown sources mixed in, different schemas
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        
        n_to_contaminate = int(len(X_corrupted) * provenance_contamination_rate / 100)
        
        if n_to_contaminate == 0:
            return X_corrupted, y_corrupted, {
                'error_type': 'provenance_issues',
                'rows_contaminated': 0
            }
        
        contaminate_indices = np.random.choice(len(X_corrupted), min(n_to_contaminate, len(X_corrupted)), replace=False)
        
        contaminated_count = 0
        
        for idx in contaminate_indices:
            scale_factor = np.random.uniform(0.5, 2.0)
            
            for col in X_corrupted.select_dtypes(include=[np.number]).columns:
                X_corrupted.at[idx, col] = X_corrupted.at[idx, col] * scale_factor
            
            contaminated_count += 1
        
        return X_corrupted, y_corrupted, {
            'error_type': 'provenance_issues',
            'intended_rate': provenance_contamination_rate,
            'rows_contaminated': contaminated_count
        }
    
    
    # =========================================================================
    # ERROR TYPE 23: TEMPORAL ORDERING ERRORS
    def inject_contextual_errors(self, X, y, contextual_error_rate=0.05):
        """
        Inject contextually impossible values (technically valid but make no sense)
        
        Parameters:
        - contextual_error_rate: Percentage of values to make contextually wrong (0.05 = 5%)
        
        Severity levels: 5%, 10%, 15%
        
        Simulation: Age=350, Salary=-50000, impossible values
        """
        X_corrupted = X.copy()
        
        numerical_cols = X_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_corrupted, y, {
                'error_type': 'contextual_errors',
                'contextual_errors_added': 0
            }
        
        total_cells = len(X_corrupted) * len(numerical_cols)
        n_errors = int(total_cells * contextual_error_rate / 100)
        
        # Size both arrays by n_errors. zip() pairs them elementwise and stops at
        # the shorter one, so capping col_indices at the column count would pin
        # every severity to exactly len(numerical_cols) errors.
        rows = np.random.choice(len(X_corrupted), n_errors, replace=True)
        col_indices = np.random.choice(len(numerical_cols), n_errors, replace=True)
        
        contextual_errors_added = 0
        col_list = list(numerical_cols)

        # Maxima come from the ORIGINAL data, computed once. Reading the max
        # back out of X_corrupted inside the loop would multiply each injected
        # value by the previous one, growing them exponentially (observed:
        # 5.5 -> 2.9e152) rather than keeping them implausible-but-finite.
        col_maxima = {col: X[col].max() for col in col_list}

        for r, c_idx in zip(rows, col_indices):
            if c_idx < len(col_list):
                col = col_list[c_idx]
                col_max = col_maxima[col]

                contextual_error = col_max * np.random.uniform(5, 20)

                X_corrupted.at[r, col] = float(contextual_error)
                contextual_errors_added += 1
        
        return X_corrupted, y, {
            'error_type': 'contextual_errors',
            'intended_rate': contextual_error_rate,
            'contextual_errors_added': contextual_errors_added
        }
    
    
    # =========================================================================
    # UNIFIED INTERFACE
    # =========================================================================
    
    # =========================================================================
    # TARGETED MECHANISM VARIANTS
    #
    # Each of the three injectors below corrupts exactly what its random
    # counterpart corrupts, at the same rate, changing only WHERE the
    # corruption lands. Pairing them isolates the effect of the mechanism from
    # the effect of the error type, which the random-only design confounds.
    # =========================================================================

    def _feature_importance(self, X, y):
        """
        Rank features by absolute correlation with the target.

        Used as a cheap, deterministic stand-in for model-based importance so
        the injector stays fast and adds no randomness of its own.
        """
        y_arr = np.asarray(y, dtype=float)
        scores = {}
        # Numeric columns only: correlation is undefined for text, and passing
        # strings to corrcoef raises rather than returning nan.
        for col in X.select_dtypes(include=[np.number]).columns:
            values = X[col].astype(float).values
            if np.std(values) == 0:
                scores[col] = 0.0
            else:
                corr = np.corrcoef(values, y_arr)[0, 1]
                scores[col] = 0.0 if np.isnan(corr) else abs(corr)
        return sorted(scores, key=scores.get, reverse=True)

    def inject_missing_values_mnar(self, X, y, missing_rate=0.10):
        """
        Missing Not At Random: missingness concentrated in one class.

        The random counterpart (inject_missing_values) is MCAR - every cell is
        equally likely to go missing, which imputation handles well and models
        largely absorb. Here the probability of going missing depends on the
        target, so imputation fills from the wrong distribution and biases the
        model. Same number of missing cells, very different information loss.
        """
        X_corrupted = X.copy()
        y_arr = np.asarray(y)

        total_cells = X_corrupted.shape[0] * X_corrupted.shape[1]
        n_missing = int(total_cells * missing_rate / 100)

        # Rows of the positive class are 9x more likely to lose a value.
        weights = np.where(y_arr == 1, 0.9, 0.1)
        weights = weights / weights.sum()

        rows = np.random.choice(X_corrupted.shape[0], n_missing, replace=True, p=weights)
        cols = np.random.choice(X_corrupted.shape[1], n_missing, replace=True)

        for r, c in zip(rows, cols):
            X_corrupted.iloc[r, c] = np.nan

        actual_missing_pct = ((X_corrupted.isna().sum().sum() / total_cells * 100)
                              if total_cells else 0.0)

        return X_corrupted, y, {
            'error_type': 'missing_values_mnar',
            'mechanism': 'MNAR - missingness depends on target',
            'intended_rate': missing_rate,
            'actual_rate': actual_missing_pct,
        }

    def inject_duplicates_targeted(self, X, y, duplicate_rate=0.05):
        """
        Duplicate rows from a single class only.

        The random counterpart samples rows uniformly, which reproduces the
        existing class balance and leaves the learning problem essentially
        unchanged. Duplicating one class instead reweights the training
        distribution, which is what makes duplication harmful in practice.
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        y_arr = np.asarray(y)

        n_duplicates = int(len(X_corrupted) * duplicate_rate / 100)

        # Duplicate the majority class, pushing the balance further from parity.
        counts = pd.Series(y_arr).value_counts()
        target_class = counts.index[0]
        candidates = np.where(y_arr == target_class)[0]

        if len(candidates) == 0:
            candidates = np.arange(len(X_corrupted))

        indices = np.random.choice(candidates, n_duplicates, replace=True)

        X_corrupted = pd.concat([X_corrupted, X_corrupted.iloc[indices]], ignore_index=True)
        y_corrupted = pd.concat([y_corrupted, y_corrupted.iloc[indices]], ignore_index=True)

        return X_corrupted, y_corrupted, {
            'error_type': 'duplicates_targeted',
            'mechanism': f'all duplicates drawn from class {target_class}',
            'intended_rate': duplicate_rate,
            'duplicates_added': n_duplicates,
            'corrupted_rows': len(X_corrupted),
        }

    def inject_outliers_targeted(self, X, y, outlier_rate=0.01):
        """
        Place outliers only in the most predictive features.

        The random counterpart spreads outliers evenly across every column, so
        most of them land in features the model barely uses. Concentrating the
        same number of outliers in the top predictors attacks the signal the
        model actually depends on.
        """
        X_corrupted = X.copy()
        numerical_cols = list(X_corrupted.select_dtypes(include=[np.number]).columns)

        if not numerical_cols:
            return X_corrupted, y, {'error_type': 'outliers_targeted', 'outliers_added': 0}

        # Same budget as the random version: rate% of ALL numerical cells,
        # but concentrated into the top third of features by predictiveness.
        total_numerical_cells = len(X_corrupted) * len(numerical_cols)
        n_outliers = int(total_numerical_cells * outlier_rate / 100)

        ranked = [c for c in self._feature_importance(X, y) if c in numerical_cols]
        top_k = max(1, len(ranked) // 3)
        target_cols = ranked[:top_k]

        outliers_added = 0
        for _ in range(n_outliers):
            col = target_cols[np.random.randint(len(target_cols))]
            idx = np.random.randint(len(X_corrupted))

            col_mean = X[col].mean()
            col_std = X[col].std()
            if col_std == 0:
                continue

            extreme = col_mean + np.random.choice([-1, 1]) * 4 * col_std
            X_corrupted.iloc[idx, X_corrupted.columns.get_loc(col)] = extreme
            outliers_added += 1

        return X_corrupted, y, {
            'error_type': 'outliers_targeted',
            'mechanism': f'confined to top {top_k} predictive features: {target_cols[:5]}',
            'intended_rate': outlier_rate,
            'outliers_added': outliers_added,
        }

    def inject_duplicates_minority(self, X, y, duplicate_rate=0.05):
        """
        Duplicate rows from the MINORITY class only.

        Paired with inject_duplicates_targeted, which duplicates the majority
        class. Testing both directions separates "duplication is harmful" from
        "shifting the class balance in a particular direction is harmful":
        majority duplication deepens existing imbalance, while minority
        duplication acts like naive oversampling and moves the balance toward
        parity. The two are expected to behave differently, and lumping them
        together as "targeted duplication" would hide that.
        """
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        y_arr = np.asarray(y)

        n_duplicates = int(len(X_corrupted) * duplicate_rate / 100)

        counts = pd.Series(y_arr).value_counts()
        target_class = counts.index[-1]
        candidates = np.where(y_arr == target_class)[0]

        if len(candidates) == 0:
            candidates = np.arange(len(X_corrupted))

        indices = np.random.choice(candidates, n_duplicates, replace=True)

        X_corrupted = pd.concat([X_corrupted, X_corrupted.iloc[indices]], ignore_index=True)
        y_corrupted = pd.concat([y_corrupted, y_corrupted.iloc[indices]], ignore_index=True)

        return X_corrupted, y_corrupted, {
            'error_type': 'duplicates_minority',
            'mechanism': f'all duplicates drawn from minority class {target_class}',
            'intended_rate': duplicate_rate,
            'duplicates_added': n_duplicates,
        }

    def inject_label_noise_asymmetric(self, X, y, noise_rate=0.05):
        """
        Flip labels in ONE direction only (class 1 -> class 0).

        The random counterpart flips labels symmetrically, so errors in both
        directions partly cancel and the class balance is preserved. Real
        annotation error is rarely symmetric: annotators systematically miss
        the positive class, or default to the safer label under uncertainty.
        Asymmetric noise both corrupts labels AND shifts the prior, so it is
        expected to be far more damaging at an identical flip rate.
        """
        X_corrupted = X.copy()
        y_corrupted = np.asarray(y).copy()

        n_flips = int(len(y_corrupted) * noise_rate / 100)

        # Only positive-class rows are eligible, so every flip moves the same way.
        eligible = np.where(y_corrupted == 1)[0]
        if len(eligible) == 0:
            eligible = np.arange(len(y_corrupted))

        # Never erase the positive class entirely. On an imbalanced dataset the
        # requested flip count can exceed the number of positives (Bank
        # Marketing is 11.7% positive, so a 15% one-way flip would consume all
        # of them), leaving a single-class training set: models cannot fit,
        # AUC is undefined, and the run yields nothing to compare. Capping at
        # 80% keeps the corruption severe but the experiment measurable.
        max_flips = int(len(eligible) * 0.8)
        capped = n_flips > max_flips
        n_flips = min(n_flips, max_flips)

        flip_idx = np.random.choice(eligible, n_flips, replace=False)
        y_corrupted[flip_idx] = 0

        return X_corrupted, pd.Series(y_corrupted, index=X_corrupted.index), {
            'error_type': 'label_noise_asymmetric',
            'mechanism': 'one-directional flips (1 -> 0)',
            'intended_rate': noise_rate,
            'labels_flipped': n_flips,
            # True when the dataset lacked enough positives to honour the
            # requested rate. Such cells are not comparable to the symmetric
            # counterpart at the same nominal rate and must be flagged, not
            # silently averaged in.
            'rate_capped': capped,
        }

    def inject_feature_noise_targeted(self, X, y, noise_rate=0.05):
        """
        Add noise ONLY to the most predictive features.

        The random counterpart perturbs every column equally, so most of the
        noise budget lands on features the model barely relies on. Concentrating
        the same noise magnitude on the top predictors degrades the signal the
        model actually uses, which is the realistic case when a key sensor or
        a heavily-used field is miscalibrated.
        """
        X_corrupted = X.copy()
        numerical_cols = list(X_corrupted.select_dtypes(include=[np.number]).columns)

        if not numerical_cols:
            return X_corrupted, y, {'error_type': 'feature_noise_targeted', 'features_affected': 0}

        ranked = [c for c in self._feature_importance(X, y) if c in numerical_cols]
        top_k = max(1, len(ranked) // 3)
        target_cols = ranked[:top_k]

        # Match the TOTAL noise energy of the random counterpart, which spreads
        # noise of relative size (noise_rate/100) across all columns. Applying
        # that same per-column magnitude to a third of the columns would inject
        # only a third as much noise overall, so any difference in degradation
        # would measure the smaller dose rather than the targeting. Variance
        # adds across columns, so scaling the per-column sigma by
        # sqrt(n_all / n_targeted) keeps the totals equal.
        scale = np.sqrt(len(numerical_cols) / len(target_cols))

        for col in target_cols:
            col_std = X_corrupted[col].std()
            if col_std == 0:
                continue
            sigma = col_std * (noise_rate / 100) * scale
            X_corrupted[col] = X_corrupted[col] + np.random.normal(0, sigma, len(X_corrupted))

        return X_corrupted, y, {
            'error_type': 'feature_noise_targeted',
            'mechanism': f'noise confined to top {top_k} predictive features',
            'intended_rate': noise_rate,
            'features_affected': len(target_cols),
        }

    def inject_error(self, X, y, error_type, severity_level):
        """
        Unified interface for error injection
        
        Parameters:
        - error_type: One of 24 error types
        - severity_level: 'low', 'medium', 'high'
        """
        
        severity_map = {
            'missing_values': {'low': 10, 'medium': 20, 'high': 30},
            'label_noise': {'low': 5, 'medium': 10, 'high': 15},
            'duplicates': {'low': 5, 'medium': 10, 'high': 15},
            'outliers': {'low': 1, 'medium': 5, 'high': 10},
            'invalid_values': {'low': 2, 'medium': 5, 'high': 10},
            'class_imbalance': {'low': 10, 'medium': 25, 'high': 40},
            'feature_noise': {'low': 5, 'medium': 15, 'high': 25},
            'categorical_errors': {'low': 5, 'medium': 10, 'high': 15},
            'data_inconsistency': {'low': 5, 'medium': 10, 'high': 15},
            'typographical_errors': {'low': 5, 'medium': 10, 'high': 15},
            'data_drift': {'low': 10, 'medium': 20, 'high': 30},
            'data_leakage': {'low': 10, 'medium': 20, 'high': 30},
            'redundant_features': {'low': 10, 'medium': 20, 'high': 30},
            'annotator_bias': {'low': 10, 'medium': 20, 'high': 30},
            'ambiguous_labels': {'low': 5, 'medium': 10, 'high': 15},
            'domain_violations': {'low': 5, 'medium': 10, 'high': 15},
            'data_heterogeneity': {'low': 10, 'medium': 20, 'high': 30},
            'concept_drift': {'low': 10, 'medium': 20, 'high': 30},
            'data_representativeness': {'low': 10, 'medium': 20, 'high': 30},
            'imbalanced_feature_distribution': {'low': 10, 'medium': 20, 'high': 30},
            'provenance_issues': {'low': 10, 'medium': 20, 'high': 30},
            'contextual_errors': {'low': 5, 'medium': 10, 'high': 15},
            # Targeted variants use rates identical to their random counterparts
            # so any difference in degradation is attributable to the mechanism
            # alone, not to a different amount of corruption.
            'missing_values_mnar': {'low': 10, 'medium': 20, 'high': 30},
            'duplicates_targeted': {'low': 5, 'medium': 10, 'high': 15},
            'outliers_targeted': {'low': 1, 'medium': 5, 'high': 10},
            'duplicates_minority': {'low': 5, 'medium': 10, 'high': 15},
            'label_noise_asymmetric': {'low': 5, 'medium': 10, 'high': 15},
            'feature_noise_targeted': {'low': 5, 'medium': 15, 'high': 25}
        }
        
        if error_type not in severity_map:
            raise ValueError(f"Unknown error type: {error_type}")

        # Reseed per experiment so this injection is independent of how many
        # random draws any earlier injection consumed. See _seed_for.
        np.random.seed(self._seed_for(error_type, severity_level, X))

        error_rate = severity_map[error_type][severity_level]
        
        error_methods = {
            'missing_values': self.inject_missing_values,
            'label_noise': self.inject_label_noise,
            'duplicates': self.inject_duplicates,
            'outliers': self.inject_outliers,
            'invalid_values': self.inject_invalid_values,
            'class_imbalance': self.inject_class_imbalance,
            'feature_noise': self.inject_feature_noise,
            'categorical_errors': self.inject_categorical_errors,
            'data_inconsistency': self.inject_data_inconsistency,
            'typographical_errors': self.inject_typographical_errors,
            'data_drift': self.inject_data_drift,
            'data_leakage': self.inject_data_leakage,
            'redundant_features': self.inject_redundant_features,
            'annotator_bias': self.inject_annotator_bias,
            'ambiguous_labels': self.inject_ambiguous_labels,
            'domain_violations': self.inject_domain_violations,
            'data_heterogeneity': self.inject_data_heterogeneity,
            'concept_drift': self.inject_concept_drift,
            'data_representativeness': self.inject_data_representativeness,
            'imbalanced_feature_distribution': self.inject_imbalanced_feature_distribution,
            'provenance_issues': self.inject_provenance_issues,
            'contextual_errors': self.inject_contextual_errors,
            'missing_values_mnar': self.inject_missing_values_mnar,
            'duplicates_targeted': self.inject_duplicates_targeted,
            'outliers_targeted': self.inject_outliers_targeted,
            'duplicates_minority': self.inject_duplicates_minority,
            'label_noise_asymmetric': self.inject_label_noise_asymmetric,
            'feature_noise_targeted': self.inject_feature_noise_targeted
        }
        
        X_corrupted, y_corrupted, metadata = error_methods[error_type](X, y, error_rate)

        # Stamp every injection with its implementation version so results from
        # different implementations of the same error type cannot be merged.
        metadata['injection_version'] = ERROR_TYPE_VERSIONS.get(error_type, 1)

        return X_corrupted, y_corrupted, metadata