"""
Split-sensitivity comparison analysis: compares split100 and split200
against the frozen original (seed-42-split) results, for the same 21
scoped error types, to check whether headline conclusions hold.

Does not touch the paper or any frozen file. Read-only analysis, printed
to stdout for manual review before deciding what (if anything) to write.
"""
import numpy as np
import pandas as pd
from scipy import stats

SCOPED_ERROR_TYPES = sorted(set([
    'annotator_bias', 'missing_values_mnar', 'ambiguous_labels',
    'contextual_errors', 'label_noise_asymmetric', 'data_leakage',
    'concept_drift', 'data_heterogeneity', 'invalid_values',
    'data_inconsistency', 'label_noise',
    'missing_values', 'outliers', 'outliers_targeted', 'duplicates',
    'duplicates_minority', 'duplicates_targeted', 'typographical_errors',
    'categorical_errors', 'feature_noise', 'feature_noise_targeted',
]))

RQ1_ACTIONS = {
    'missing_values_mnar': 'KNN imputation (MNAR)',
    'contextual_errors': 'IQR clipping (contextual)',
    'invalid_values': 'Sentinel detection',
    'data_inconsistency': 'Consistency rules',
    'data_leakage': 'Leakage removal',
}

MECH_PAIRS = [
    ('missing_values', 'missing_values_mnar', 'missing values vs. MNAR'),
    ('label_noise', 'label_noise_asymmetric', 'label noise vs. asymmetric'),
    ('outliers', 'outliers_targeted', 'outliers vs. targeted'),
    ('duplicates', 'duplicates_minority', 'duplicates vs. minority'),
    ('typographical_errors', 'categorical_errors', 'typographical vs. categorical'),
    ('duplicates', 'duplicates_targeted', 'duplicates vs. majority'),
    ('feature_noise', 'feature_noise_targeted', 'feature noise vs. targeted'),
]


def build_damage_improve(exp_path, base_path):
    exp = pd.read_csv(exp_path)
    base = pd.read_csv(base_path)
    clean = base.groupby(['Dataset', 'Model']).Accuracy.mean().rename('Clean').reset_index()
    dirty = (exp[exp.Condition == 'dirty']
             .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
             .Accuracy.mean().rename('Dirty').reset_index())
    cleaned = (exp[exp.Condition == 'cleaned']
               .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
               .Accuracy.mean().rename('Cleaned').reset_index())
    d = (dirty.merge(clean, on=['Dataset', 'Model'])
               .merge(cleaned, on=['Dataset', 'Error_Type', 'Severity', 'Model'], how='left'))
    d['Damage_pp'] = (d.Clean - d.Dirty) * 100
    d['Improvement_pp'] = (d.Cleaned - d.Dirty) * 100
    return d


def rq1_table(df, label):
    print(f"\n--- RQ1 headline actions [{label}] ---")
    rows = []
    for et, name in RQ1_ACTIONS.items():
        g = df[df.Error_Type == et]
        imp = g.Improvement_pp.dropna()
        n = len(imp)
        mean = imp.mean()
        se = imp.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        tval = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        lo, hi = mean - tval * se, mean + tval * se
        rows.append({'Action': name, 'Damage': g.Damage_pp.mean(),
                      'Improvement': mean, 'CI_low': lo, 'CI_high': hi, 'n': n})
    out = pd.DataFrame(rows).sort_values('Improvement', ascending=False)
    print(out.round(3).to_string(index=False))
    return out


def damage_table(df, label):
    print(f"\n--- Damage ranking (21 scoped types) [{label}] ---")
    d = df.groupby('Error_Type').Damage_pp.mean().sort_values(ascending=False)
    print(d.round(3).to_string())
    return d


def mechanism_table(df, label):
    print(f"\n--- Mechanism contrasts [{label}] ---")
    KEY = ['Dataset', 'Severity', 'Model']
    rows = []
    for a, b, name in MECH_PAIRS:
        ga = df[df.Error_Type == a][KEY + ['Damage_pp']].rename(columns={'Damage_pp': 'A'})
        gb = df[df.Error_Type == b][KEY + ['Damage_pp']].rename(columns={'Damage_pp': 'B'})
        m = ga.merge(gb, on=KEY, how='inner')
        diff = m.B - m.A
        if len(diff) < 2:
            rows.append({'Pair': name, 'n': len(diff), 'Diff_pp': diff.mean() if len(diff) else np.nan,
                         'p': np.nan, 'd_z': np.nan})
            continue
        t, p = stats.ttest_rel(m.B, m.A)
        dz = diff.mean() / diff.std(ddof=1)
        rows.append({'Pair': name, 'n': len(diff), 'Diff_pp': diff.mean(), 'p': p, 'd_z': dz})
    out = pd.DataFrame(rows)
    # Holm-Bonferroni
    pvals = out.p.fillna(1.0).values
    order = np.argsort(pvals)
    m_tests = len(pvals)
    holm = np.empty(m_tests)
    running_max = 0
    for rank, idx in enumerate(order):
        adj = (m_tests - rank) * pvals[idx]
        running_max = max(running_max, adj)
        holm[idx] = min(running_max, 1.0)
    out['p_holm'] = holm
    out['sig_holm'] = out['p_holm'] < 0.05
    print(out.round(4).to_string(index=False))
    return out


for split in (100, 200):
    exp_path = f'results/split_sensitivity/split{split}/experiment_results.csv'
    base_path = f'results/split_sensitivity/split{split}/baseline_results.csv'
    df = build_damage_improve(exp_path, base_path)
    label = f'split {split}'
    print('=' * 78)
    print(f'SPLIT {split}')
    print('=' * 78)
    rq1_table(df, label)
    damage_table(df, label)
    mechanism_table(df, label)
    print()
