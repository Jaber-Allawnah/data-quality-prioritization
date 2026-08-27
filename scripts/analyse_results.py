"""
Analysis — RQ1 to RQ4
=====================

Consumes results/experiment_results.csv and results/baseline_results.csv and
produces every table the write-up needs.

RQ1  Which cleaning action gives the biggest improvement?
RQ2  (inputs only - the cost model is the next stage)
RQ3  Does the best priority depend on the AI model?
RQ4  Does the best priority depend on the dataset?

Three metrics are always reported together, because the middle one alone is
misleading:

  Net improvement      - across ALL cases; the honest bottom line
  Conditional recovery - within cases that suffered material damage
  Coverage             - the share of cases the conditional figure rests on

Per-row recovery ratios are never averaged. Where the corrupted model beat the
clean baseline the denominator is negative and the ratio flips sign, so failure
reads as success. Aggregates sum before dividing, over eligible rows only.
"""

import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_cleaner import REMEDIATION_ROUTE, AUTOMATED
from models import MODEL_FAMILY

# Operational threshold for metric stability, fixed in advance. NOT a
# significance test: below this the recovery denominator is too small for a
# relative estimate to mean anything. Absolute values are always retained.
MATERIAL_PP = 0.5
TOP_N = 5


def load():
    base = pd.read_csv('results/baseline_results.csv')
    exp = pd.read_csv('results/experiment_results.csv')

    clean = (base.groupby(['Dataset', 'Model']).Accuracy.mean()
             .rename('Clean_Accuracy').reset_index())

    dirty = (exp[exp.Condition == 'dirty']
             .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
             .Accuracy.mean().rename('Dirty_Accuracy').reset_index())

    cleaned = (exp[exp.Condition == 'cleaned']
               .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
               .Accuracy.mean().rename('Cleaned_Accuracy').reset_index())

    df = dirty.merge(clean, on=['Dataset', 'Model'], how='left')
    df = df.merge(cleaned, on=['Dataset', 'Error_Type', 'Severity', 'Model'], how='left')

    df['Damage_pp'] = (df.Clean_Accuracy - df.Dirty_Accuracy) * 100
    df['Improvement_pp'] = (df.Cleaned_Accuracy - df.Dirty_Accuracy) * 100
    df['Route'] = df.Error_Type.map(lambda e: REMEDIATION_ROUTE.get(e, ('unknown', ''))[0])
    df['Family'] = df.Model.map(MODEL_FAMILY)
    return df


# ------------------------------------------------------------------- RQ1

def rq1(df):
    print('=' * 108)
    print('DEGRADATION — which errors cause the most damage (input to RQ1)')
    print('=' * 108)

    piv = df.pivot_table(index='Error_Type', columns='Severity',
                         values='Damage_pp', aggfunc='mean')[['low', 'medium', 'high']]
    piv['overall'] = df.groupby('Error_Type').Damage_pp.mean()
    piv['route'] = [REMEDIATION_ROUTE.get(e, ('', ''))[0] for e in piv.index]
    piv = piv.sort_values('overall', ascending=False)
    print(piv.round(2).to_string())

    print()
    print('=' * 108)
    print('RQ1 — which CLEANING ACTION gives the biggest improvement')
    print('=' * 108)

    rows = []
    for error_type, g in df.groupby('Error_Type'):
        if g.Cleaned_Accuracy.isna().all():
            continue
        eligible = g[g.Damage_pp >= MATERIAL_PP]
        denom = eligible.Damage_pp.sum()
        rows.append({
            'Error_Type': error_type,
            'Route': REMEDIATION_ROUTE.get(error_type, ('', ''))[0],
            # Means over ALL rows for this error type.
            'Damage_pp': g.Damage_pp.mean(),
            'Net_Improvement_pp': g.Improvement_pp.mean(),
            # Damage restricted to the ELIGIBLE subset, so the conditional
            # recovery figure below has a denominator a reader can reproduce.
            # Without it, dividing Net_Improvement_pp by Damage_pp gives a
            # different number and the table looks internally inconsistent.
            'Eligible_Damage_pp': eligible.Damage_pp.mean() if len(eligible) else np.nan,
            'Eligible_Improvement_pp': eligible.Improvement_pp.mean() if len(eligible) else np.nan,
            'Conditional_Recovery_%': (eligible.Improvement_pp.sum() / denom * 100)
                                       if denom > 0 else np.nan,
            'Coverage_%': len(eligible) / len(g) * 100,
        })

    out = pd.DataFrame(rows).sort_values('Net_Improvement_pp', ascending=False)
    print(out.round(2).to_string(index=False))
    out.to_csv('results/rq1_cleaning_actions.csv', index=False)

    print()
    print('DEFINITIONS — the three columns use DIFFERENT row sets, by design:')
    print(f'  Damage_pp / Net_Improvement_pp   mean over ALL rows for the error type')
    print(f'  Eligible_*                       mean over rows with damage >= {MATERIAL_PP}pp only')
    print(f'  Conditional_Recovery_%           SUM(improvement) / SUM(damage) over those')
    print(f'                                   eligible rows == '
          f'Eligible_Improvement_pp / Eligible_Damage_pp')
    print(f'  Coverage_%                       share of rows that were eligible')
    print()
    print('  Net_Improvement_pp / Damage_pp does NOT equal Conditional_Recovery_%,')
    print('  because the first two average over every row including undamaged ones,')
    print('  while the ratio is computed only where damage was material.')

    print('\nError types with NO automated cleaning (recovery is 0 by construction):')
    for e in sorted(df[df.Cleaned_Accuracy.isna()].Error_Type.unique()):
        route, reason = REMEDIATION_ROUTE.get(e, ('', ''))
        print(f'  {e:32s} {route:22s} {reason[:52]}')
    return out


# --------------------------------------------------------------- mechanism

def mechanism(df):
    print()
    print('=' * 108)
    print('MECHANISM — same error, same rate, different placement')
    print('=' * 108)

    pairs = [('missing_values', 'missing_values_mnar', 'MCAR', 'MNAR (target-related)'),
             ('label_noise', 'label_noise_asymmetric', 'symmetric', 'one-directional'),
             ('duplicates', 'duplicates_targeted', 'random rows', 'majority class'),
             ('duplicates', 'duplicates_minority', 'random rows', 'minority class'),
             ('outliers', 'outliers_targeted', 'all features', 'top predictors'),
             ('feature_noise', 'feature_noise_targeted', 'all features', 'top predictors'),
             ('typographical_errors', 'categorical_errors', 'invalid value (typo)',
              'valid but wrong category')]

    rows = []
    for a, b, la, lb in pairs:
        ga, gb = df[df.Error_Type == a], df[df.Error_Type == b]
        if not len(ga) or not len(gb):
            continue
        _, p = stats.ttest_ind(gb.Damage_pp, ga.Damage_pp)
        rows.append({
            'Comparison': f'{a} vs {b}',
            'Mechanism_A': la, 'Damage_A_pp': ga.Damage_pp.mean(),
            'Mechanism_B': lb, 'Damage_B_pp': gb.Damage_pp.mean(),
            'Difference_pp': gb.Damage_pp.mean() - ga.Damage_pp.mean(),
            'p': p, 'Significant': p < 0.05,
        })

    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv('results/mechanism_comparison.csv', index=False)
    return out


# -------------------------------------------------------------- RQ3 / RQ4

def dependency(df, group_col, label, out_csv):
    print()
    print('=' * 108)
    print(label)
    print('=' * 108)

    table = df.pivot_table(index='Error_Type', columns=group_col,
                           values='Damage_pp', aggfunc='mean')
    groups = list(table.columns)

    print(f'\nTop {TOP_N} most damaging, per {group_col.lower()}:')
    tops = {}
    for g in groups:
        top = table[g].nlargest(TOP_N)
        tops[g] = list(top.index)
        print(f'  {g:22s} ' + ', '.join(f'{e}({v:.1f})' for e, v in top.items()))

    overlaps, corrs = [], []
    for a, b in combinations(groups, 2):
        overlaps.append(len(set(tops[a]) & set(tops[b])))
        corrs.append(stats.spearmanr(table[a], table[b])[0])

    print(f'\nmean top-{TOP_N} overlap : {np.mean(overlaps):.2f}/{TOP_N}')
    print(f'mean Spearman rho    : {np.mean(corrs):.3f}')

    ranks = table.rank(ascending=False)
    summary = pd.DataFrame({
        'Mean_Damage_pp': table.mean(axis=1),
        'Best_Rank': ranks.min(axis=1).astype(int),
        'Worst_Rank': ranks.max(axis=1).astype(int),
    })
    summary['Rank_Range'] = summary.Worst_Rank - summary.Best_Rank
    summary = summary.sort_values('Mean_Damage_pp', ascending=False)
    summary.to_csv(out_csv)

    print(f'\nRank stability, 10 most damaging (0 = never moves):')
    print(summary.head(10).round(2).to_string())

    if np.mean(overlaps) >= 4 and np.mean(corrs) >= 0.8:
        verdict = f'LARGELY INDEPENDENT of {group_col.lower()} — one ranking is defensible'
    elif np.mean(overlaps) >= 3 or np.mean(corrs) >= 0.6:
        verdict = f'PARTLY dependent on {group_col.lower()} — report with caveats'
    else:
        verdict = f'DEPENDS on {group_col.lower()} — a single universal ranking is NOT defensible'
    print(f'\nVERDICT: {verdict}')
    return np.mean(overlaps), np.mean(corrs), summary


def family_split(df):
    """RQ3 refinement: does priority split by model FAMILY rather than algorithm?"""
    print()
    print('=' * 108)
    print('RQ3 refinement — tree vs non-tree')
    print('=' * 108)

    table = df.pivot_table(index='Error_Type', columns='Model',
                           values='Damage_pp', aggfunc='mean')
    trees = [m for m, f in MODEL_FAMILY.items() if f == 'tree' and m in table.columns]
    others = [m for m, f in MODEL_FAMILY.items() if f == 'non-tree' and m in table.columns]

    within_tree = [stats.spearmanr(table[a], table[b])[0] for a, b in combinations(trees, 2)]
    within_other = [stats.spearmanr(table[a], table[b])[0] for a, b in combinations(others, 2)]
    across = [stats.spearmanr(table[a], table[b])[0] for a in trees for b in others]

    print(f'  agreement WITHIN tree models      : rho = {np.mean(within_tree):.3f}')
    print(f'  agreement WITHIN non-tree models  : rho = {np.mean(within_other):.3f}')
    print(f'  agreement ACROSS families         : rho = {np.mean(across):.3f}')

    if np.mean(across) < min(np.mean(within_tree), np.mean(within_other)) - 0.1:
        print('\n  -> Models agree more within their family than across it: cleaning')
        print('     priority depends on model FAMILY, not on the specific algorithm.')
    else:
        print('\n  -> No clear family split; disagreement is not explained by model type.')


def main():
    df = load()
    df.to_csv('results/analysis_base.csv', index=False)
    print(f'{len(df)} dataset x error x severity x model combinations\n')

    rq1(df)
    mechanism(df)
    dependency(df, 'Model', 'RQ3 — DOES PRIORITY DEPEND ON THE MODEL?',
               'results/rq3_model_dependency.csv')
    family_split(df)
    dependency(df, 'Dataset', 'RQ4 — DOES PRIORITY DEPEND ON THE DATASET?',
               'results/rq4_dataset_dependency.csv')

    print('\nSaved: rq1_cleaning_actions.csv, mechanism_comparison.csv,')
    print('       rq3_model_dependency.csv, rq4_dataset_dependency.csv')


if __name__ == '__main__':
    main()
