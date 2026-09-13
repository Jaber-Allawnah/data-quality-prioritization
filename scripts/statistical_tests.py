"""
Statistical tests for the RQ1 headline result and the 7 mechanism
comparisons, computed at the pairing granularity the paper actually
describes: Dataset x Model x Severity x Random_State (N = 5x8x3x3 = 360
matched cells per comparison).

This replaces two things that were previously broken or missing:

1. `analyse_results.py`'s `mechanism()` function used `scipy.stats.ttest_ind`
   (an UNPAIRED, independent-samples test) on `Damage_pp` values that were
   never matched by dataset/model/severity/seed, and it only ever read the
   single seed-42 file (`results/experiment_results.csv`), giving N=120 at
   best, not the N=360 the paper's methodology section claims.
2. No script anywhere computed the RQ1 headline paired-statistics (paired
   t-test / Wilcoxon / 95% CI / Cohen's d_z) that the paper cites for the
   KNN-imputation-for-MNAR-missingness result.

Also adds two SEPARATE cluster-level bootstrap sensitivity analyses
(resampling by Dataset, and separately by Model) alongside the paired test,
as a robustness check for the dependence-among-observations concern -- NOT
a single joint cluster-robust estimator, since dataset and model are
crossed (not nested) clusters here.

Inputs: results/{baseline,experiment}_results{,_seed7,_seed123}.csv
Outputs:
  results/rq1_headline_statistics.csv
  results/mechanism_comparison.csv   (overwrites the old unpaired version)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

RNG_SEED = 12345          # bootstrap RNG seed, fixed for reproducibility
N_BOOTSTRAP = 10000
SEEDS = [42, 7, 123]

MECHANISM_PAIRS = [
    ('missing_values', 'missing_values_mnar', 'MCAR', 'MNAR (target-related)'),
    ('label_noise', 'label_noise_asymmetric', 'symmetric', 'one-directional'),
    ('duplicates', 'duplicates_targeted', 'random rows', 'majority class'),
    ('duplicates', 'duplicates_minority', 'random rows', 'minority class'),
    ('outliers', 'outliers_targeted', 'all features', 'top predictors'),
    ('feature_noise', 'feature_noise_targeted', 'all features', 'top predictors'),
    ('typographical_errors', 'categorical_errors', 'invalid value (typo)',
     'valid but wrong category'),
]

KEY = ['Dataset', 'Model', 'Severity', 'Random_State']


# --------------------------------------------------------------- loading

def load_all_seeds():
    """Merge the three seed files into one long df with a Random_State column."""
    base_frames, exp_frames = [], []
    for seed in SEEDS:
        suffix = '' if seed == 42 else f'_seed{seed}'
        b = pd.read_csv(f'results/baseline_results{suffix}.csv')
        e = pd.read_csv(f'results/experiment_results{suffix}.csv')
        b['Random_State'] = seed
        e['Random_State'] = seed
        base_frames.append(b)
        exp_frames.append(e)
    base = pd.concat(base_frames, ignore_index=True)
    exp = pd.concat(exp_frames, ignore_index=True)

    clean = (base.groupby(['Dataset', 'Model', 'Random_State']).Accuracy.mean()
             .rename('Clean_Accuracy').reset_index())

    dirty = (exp[exp.Condition == 'dirty']
             .groupby(['Dataset', 'Error_Type', 'Severity', 'Model', 'Random_State'])
             .Accuracy.mean().rename('Dirty_Accuracy').reset_index())

    cleaned = (exp[exp.Condition == 'cleaned']
               .groupby(['Dataset', 'Error_Type', 'Severity', 'Model', 'Random_State'])
               .Accuracy.mean().rename('Cleaned_Accuracy').reset_index())

    df = dirty.merge(clean, on=['Dataset', 'Model', 'Random_State'], how='left')
    df = df.merge(cleaned, on=['Dataset', 'Error_Type', 'Severity', 'Model', 'Random_State'],
                  how='left')
    df['Damage_pp'] = (df.Clean_Accuracy - df.Dirty_Accuracy) * 100
    df['Improvement_pp'] = (df.Cleaned_Accuracy - df.Dirty_Accuracy) * 100
    return df


# --------------------------------------------------------- shared stats

def paired_stats(diff):
    """diff: array of paired differences (B - A). Returns dict of standard
    paired-test statistics."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    t_p = stats.ttest_rel(diff, np.zeros(n)).pvalue if n > 1 else np.nan
    w_p = stats.wilcoxon(diff).pvalue if n > 1 and np.any(diff != 0) else np.nan
    mean, sd = diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else np.nan
    ci_lo, ci_hi = (mean - 1.96 * se, mean + 1.96 * se) if n > 1 else (np.nan, np.nan)
    d_z = mean / sd if sd > 0 else np.nan
    return {'n': n, 'mean_diff_pp': mean, 't_p': t_p, 'wilcoxon_p': w_p,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'd_z': d_z}


def cluster_bootstrap(cells, cluster_col, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Block-bootstrap the paired mean difference by resampling whole
    clusters (all rows sharing a cluster_col value) with replacement.
    `cells` must have a 'diff' column. Returns (ci_lo, ci_hi, boot_means)."""
    rng = np.random.default_rng(seed)
    clusters = cells[cluster_col].unique()
    k = len(clusters)
    boot_means = np.empty(n_boot)
    grouped = {c: cells.loc[cells[cluster_col] == c, 'diff'].values for c in clusters}
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=k, replace=True)
        vals = np.concatenate([grouped[c] for c in sampled])
        boot_means[i] = vals.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return ci_lo, ci_hi, boot_means


def bootstrap_row(cells):
    ds_lo, ds_hi, _ = cluster_bootstrap(cells, 'Dataset')
    md_lo, md_hi, _ = cluster_bootstrap(cells, 'Model')
    return {'dataset_boot_ci_lo': ds_lo, 'dataset_boot_ci_hi': ds_hi,
            'model_boot_ci_lo': md_lo, 'model_boot_ci_hi': md_hi,
            'n_dataset_clusters': cells.Dataset.nunique(),
            'n_model_clusters': cells.Model.nunique()}


# ------------------------------------------------------------------ RQ1

def rq1_headline(df):
    print('=' * 100)
    print('RQ1 HEADLINE — KNN imputation for missing_values_mnar: cleaned vs dirty accuracy')
    print('=' * 100)

    cells = df[(df.Error_Type == 'missing_values_mnar') & df.Cleaned_Accuracy.notna()].copy()
    cells['diff'] = cells.Improvement_pp
    assert len(cells) == 360, f'expected 360 matched cells, got {len(cells)}'

    stats_row = paired_stats(cells['diff'])
    stats_row.update(bootstrap_row(cells))
    stats_row['comparison'] = 'missing_values_mnar: cleaned vs dirty (KNN imputation)'

    out = pd.DataFrame([stats_row])
    print(out.round(4).to_string(index=False))
    out.to_csv('results/rq1_headline_statistics.csv', index=False)
    return out


# ------------------------------------------------------------- mechanism

def mechanism(df):
    print()
    print('=' * 100)
    print('MECHANISM — same error, same rate, different placement (paired, N=360 per pair)')
    print('=' * 100)

    rows = []
    raw_p = []
    for a, b, la, lb in MECHANISM_PAIRS:
        ga = df[df.Error_Type == a][KEY + ['Damage_pp']].rename(columns={'Damage_pp': 'Damage_A'})
        gb = df[df.Error_Type == b][KEY + ['Damage_pp']].rename(columns={'Damage_pp': 'Damage_B'})
        m = ga.merge(gb, on=KEY, how='inner')
        if m.empty:
            continue
        m['diff'] = m.Damage_B - m.Damage_A
        assert len(m) == 360, f'{a} vs {b}: expected 360 matched cells, got {len(m)}'

        st = paired_stats(m['diff'])
        boot = bootstrap_row(m)
        row = {
            'Comparison': f'{a} vs {b}', 'Mechanism_A': la, 'Mechanism_B': lb,
            'Damage_A_pp': m.Damage_A.mean(), 'Damage_B_pp': m.Damage_B.mean(),
            **st, **boot,
        }
        rows.append(row)
        raw_p.append(st['t_p'])

    # Holm-Bonferroni step-down over the paired t-test p-values
    order = np.argsort(raw_p)
    holm_p = np.empty(len(raw_p))
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = raw_p[idx] * (len(raw_p) - rank)
        running_max = max(running_max, adj)
        holm_p[idx] = min(running_max, 1.0)

    for row, hp in zip(rows, holm_p):
        row['p_holm'] = hp
        row['significant_holm'] = hp < 0.05

    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv('results/mechanism_comparison.csv', index=False)
    return out


if __name__ == '__main__':
    df = load_all_seeds()
    rq1_out = rq1_headline(df)
    mech_out = mechanism(df)
    print()
    print('Saved: results/rq1_headline_statistics.csv, results/mechanism_comparison.csv')
