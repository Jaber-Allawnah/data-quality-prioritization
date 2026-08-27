"""
Cost-measurement stability audit.

Run after regenerating results/cleaning_costs.csv. The regeneration changed only a
cosmetic Method label, but it re-measured every timing, so any movement in the
primary-vs-sensitivity rank correlation reflects timing stability, NOT the label.

Four diagnostics:
  1. cell-by-cell median timing, previous vs current
  2. is the movement concentrated in the sub-0.1s jitter band?
  3. did any substantial-cost action (notably KNN imputation) change rank?
  4. band-collapsed ranking - the robustness test that matches the actual claim

The substantive RQ2 claim is BETWEEN cost bands. If band-level ordering holds while
raw Spearman moves because the negligible group reshuffles internally, the economic
conclusion is stable and should be reported as such rather than as instability.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NEGLIGIBLE_COST_SECONDS = 0.1
KEY = ['Dataset', 'Error_Type', 'Severity']

PREV = sys.argv[1] if len(sys.argv) > 1 else 'cleaning_costs_PREV.csv'
CURR = 'results/cleaning_costs.csv'


def band(s):
    return np.where(s < NEGLIGIBLE_COST_SECONDS, 'negligible', 'substantial')


def add_marginal(costs):
    """Attach the SENSITIVITY cost definition, matching build_roi_model.py exactly.

        C_j = T_clean,j + max(0, T_train_after,j - T_train_without,j)

    NOT Total_Compute_Seconds. That column is Clean_Seconds + the FULL training
    time, so it carries a ~3.8s floor; thresholding it at 0.1s marks nearly every
    action 'substantial' by construction and produces a meaningless band
    agreement. The sensitivity definition prices only the training time the
    intervention ADDS, which is what makes it comparable to cleaning time.
    """
    exp = pd.read_csv('results/experiment_results.csv')
    no_clean = (exp[exp.Condition == 'dirty']
                .groupby(KEY).Train_Seconds.sum()
                .rename('Train_Seconds_No_Clean').reset_index())
    c = costs.merge(no_clean, on=KEY, how='left')
    delta = (c.Train_Seconds_After_Clean.fillna(0)
             - c.Train_Seconds_No_Clean.fillna(0))
    c['Marginal_Compute_Seconds'] = c.Clean_Seconds + delta.clip(lower=0)
    return c


def main():
    prev = add_marginal(pd.read_csv(PREV))
    curr = add_marginal(pd.read_csv(CURR))

    print('=' * 78)
    print('0. SHAPE AND LABEL')
    print('=' * 78)
    print(f'previous rows: {len(prev)}   current rows: {len(curr)}')
    labels = sorted(curr.loc[curr.Error_Type.str.startswith('feature_noise'),
                             'Method'].unique())
    print(f'feature_noise label(s): {labels}')
    stale = curr.Method.str.contains('shrink toward median', na=False).sum()
    print(f'rows still carrying the stale label: {stale}')

    m = prev.merge(curr, on=KEY, suffixes=('_prev', '_curr'))
    print(f'matched cells: {len(m)} of {len(curr)}')

    print()
    print('=' * 78)
    print('1. CELL-BY-CELL MEDIAN TIMING')
    print('=' * 78)
    m['delta_s'] = m.Clean_Seconds_curr - m.Clean_Seconds_prev
    m['rel_pct'] = 100 * m.delta_s / m.Clean_Seconds_prev.replace(0, np.nan)
    print(f'median absolute change : {m.delta_s.abs().median():.6f}s')
    print(f'max absolute change    : {m.delta_s.abs().max():.6f}s')
    print(f'median |relative| change: {m.rel_pct.abs().median():.1f}%')
    print('\nlargest 8 absolute movers:')
    cols = KEY + ['Clean_Seconds_prev', 'Clean_Seconds_curr', 'delta_s', 'rel_pct']
    print(m.reindex(m.delta_s.abs().sort_values(ascending=False).index)
           .head(8)[cols].round(5).to_string(index=False))

    print()
    print('=' * 78)
    print('2. IS MOVEMENT CONCENTRATED IN THE SUB-0.1s JITTER BAND?')
    print('=' * 78)
    m['band_prev'] = band(m.Clean_Seconds_prev)
    for b, g in m.groupby('band_prev'):
        print(f'{b:12s} n={len(g):4d}  median |delta| {g.delta_s.abs().median():.6f}s'
              f'  median |rel| {g.rel_pct.abs().median():6.1f}%')
    moved = m[m.band_prev != band(m.Clean_Seconds_curr)]
    print(f'\ncells that CROSSED the 0.1s band boundary: {len(moved)}')
    if len(moved):
        print(moved[cols].round(5).to_string(index=False))

    print()
    print('=' * 78)
    print('3. DID ANY SUBSTANTIAL-COST ACTION CHANGE RANK?')
    print('=' * 78)
    for name, d in (('previous', prev), ('current', curr)):
        agg = (d.groupby('Error_Type')
                .agg(clean_s=('Clean_Seconds', 'mean'),
                     total_s=('Marginal_Compute_Seconds', 'mean'))
                .sort_values('clean_s', ascending=False))
        agg['band'] = band(agg.clean_s)
        sub = agg[agg.band == 'substantial']
        print(f'{name}: substantial-band actions = '
              f'{list(sub.index) if len(sub) else "none"}')
        if len(sub):
            print(sub.round(4).to_string())

    print()
    print('=' * 78)
    print('4. PRIMARY - BAND-COLLAPSED ROBUSTNESS TO THE COST DEFINITION')
    print('=' * 78)
    print('The RQ2 claim is between bands, not within the negligible group.')
    for name, d in (('previous', prev), ('current', curr)):
        agg = d.groupby('Error_Type').agg(clean_s=('Clean_Seconds', 'mean'),
                                          total_s=('Marginal_Compute_Seconds', 'mean'))
        raw = spearmanr(agg.clean_s, agg.total_s).correlation
        # collapse: every negligible action ties, substantial ranked after
        same = (band(agg.clean_s) == band(agg.total_s))
        print(f'{name:9s} band-level agreement = {same.mean():.3f} '
              f'({same.sum()}/{len(agg)} actions in the same band under both cost '
              f'definitions)   [secondary: raw Spearman = {raw:.3f}]')

    print()
    print('=' * 78)
    print('5. IS THE 0.1s CUTOFF DRIVING THE CONCLUSION?')
    print('=' * 78)
    print('Band membership re-derived at three thresholds. The conclusion should')
    print('be the one that holds across all of them.')
    agg = curr.groupby('Error_Type').agg(clean_s=('Clean_Seconds', 'mean'),
                                         total_s=('Marginal_Compute_Seconds', 'mean'))
    memberships = {}
    for thr in (0.05, 0.10, 0.20):
        sub = sorted(agg.index[agg.clean_s >= thr])
        memberships[thr] = set(sub)
        same = ((agg.clean_s >= thr) == (agg.total_s >= thr))
        print(f'  threshold {thr:.2f}s -> substantial: {sub if sub else "none"}')
        print(f'                     band agreement across cost definitions: '
              f'{same.mean():.3f} ({same.sum()}/{len(agg)})')
    stable = set.intersection(*memberships.values())
    unstable = set.union(*memberships.values()) - stable
    print(f'\nsubstantial at EVERY threshold : {sorted(stable) or "none"}')
    print(f'bounces between bands          : {sorted(unstable) or "none"}')
    if not unstable:
        print('\n=> Band membership is insensitive to the exact cutoff over '
              '0.05-0.20s.')
    else:
        print('\n=> Some actions bounce; 0.1s must be described as an OPERATIONAL '
              'threshold, not a natural divide.')

    print()
    print('INTERPRETATION')
    print('If band-level agreement holds while raw Spearman moves, the economic')
    print('conclusion is stable and the movement is within-band jitter. If a')
    print('substantial-cost action changed band or rank, that is a real finding.')
    print('A moved Spearman is a statement about timing stability, NOT evidence')
    print('that the cosmetic Method label mattered.')


if __name__ == '__main__':
    main()
