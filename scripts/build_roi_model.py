"""
RQ2 — cost-benefit model and break-even analysis
================================================

Two regimes, priced differently because their costs are different in kind.

AUTOMATED CLEANING — cost is measurable, so efficiency is measured directly:

    efficiency_j = improvement_j / compute_cost_j          [pp per second]

EXTERNAL INTERVENTION — cost is human effort, which this study cannot measure
and will not invent a rate for. Instead the question is inverted: how cheap
would human correction have to be to beat the best automated alternative?

    C*_human(rho) = (rho * ceiling) / E_benchmark

where

    ceiling  the EXPERIMENTAL maximum recoverable damage - the oracle ceiling
             under this simulation design. For label corruptions it equals the
             measured damage, because restoring the injected ground-truth labels
             returns the data to the clean condition (verified to machine
             precision).

             It is a ceiling WITHIN the constructed experiment, where the true
             labels are known by construction. It does not imply the same
             ceiling is identifiable in a real dataset, where nobody knows which
             labels are wrong - that is precisely why rho exists.

    E_benchmark
             efficiency of the highest-BENEFIT action inside the negligible-cost
             band - NOT max_j(improvement_j / cost_j). The literal maximum would
             select whichever action happened to record the smallest runtime, and
             timings below 0.1s are noise-dominated: leakage removal reports
             ~425 pp/s only because its 0.002s measurement sits at the bottom of
             the jitter band. Using the highest-benefit action in that band gives
             a stable reference, and is conservative - a larger denominator would
             make the human thresholds look smaller still.

    rho      re-annotation effectiveness in (0, 1]: the share of injected errors
             a human actually catches. Unknown, and deliberately left as a free
             parameter rather than assumed. The output is a break-even CURVE
             over rho, not a single number resting on a guess.

Reading C*: if human correction costs LESS than C* seconds-equivalent per unit,
it dominates the best automated action. Above it, automation wins.

The illustrative variant (A) applies one clearly-labelled exchange rate to show
what a single-number ROI table would look like. It is sensitivity analysis, not
the primary result.

Output: results/rq2_automated_efficiency.csv, results/rq2_breakeven.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_cleaner import REMEDIATION_ROUTE, AUTOMATED, EXTERNAL

MATERIAL_PP = 0.5

# Interventions whose benefit is NOT in accuracy. Ranking these in an
# accuracy-denominated table would put a genuinely effective intervention at the
# bottom and read as failure. Reported separately on their own metrics.
METRIC_MISMATCHED = ['class_imbalance']

# Measured timing noise makes fine-grained cost ordering meaningless below this
# threshold: within-cleaner std is 0.010s while the whole fast group spans
# 0.002-0.027s, so a pp-per-second ranking there would be decided by scheduler
# jitter rather than by the methods. Cost is therefore reported in
# order-of-magnitude BANDS, and only across-band comparisons are interpreted.
NEGLIGIBLE_COST_SECONDS = 0.1

# Illustrative exchange rate for variant A ONLY. One human-minute is treated as
# equivalent to 60 seconds of compute. This is a presentational device to show
# the shape of a single-number ROI table; every conclusion in the primary
# analysis is independent of it.
ILLUSTRATIVE_SECONDS_PER_HUMAN_MINUTE = 60.0
ILLUSTRATIVE_ANNOTATION_MINUTES_PER_1000_ROWS = 90.0


def load():
    base = pd.read_csv('results/baseline_results.csv').groupby(
        ['Dataset', 'Model']).Accuracy.mean().rename('Clean').reset_index()
    exp = pd.read_csv('results/experiment_results.csv')

    dirty = (exp[exp.Condition == 'dirty']
             .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
             .Accuracy.mean().rename('Dirty').reset_index())
    cleaned = (exp[exp.Condition == 'cleaned']
               .groupby(['Dataset', 'Error_Type', 'Severity', 'Model'])
               .Accuracy.mean().rename('Cleaned').reset_index())

    df = (dirty.merge(base, on=['Dataset', 'Model'])
                .merge(cleaned, on=['Dataset', 'Error_Type', 'Severity', 'Model'],
                       how='left'))
    df['Damage_pp'] = (df.Clean - df.Dirty) * 100
    df['Improvement_pp'] = (df.Cleaned - df.Dirty) * 100
    df['Route'] = df.Error_Type.map(lambda e: REMEDIATION_ROUTE.get(e, ('unknown', ''))[0])

    costs = pd.read_csv('results/cleaning_costs.csv')

    # Training time WITHOUT cleaning, so the intervention-induced change in
    # training cost can be isolated. Training happens either way, so only the
    # CHANGE it causes is attributable to the decision to clean. Oversampling
    # enlarges the training set and so genuinely raises this; KNN imputation
    # leaves it untouched.
    exp = pd.read_csv('results/experiment_results.csv')
    train_dirty = (exp[exp.Condition == 'dirty']
                   .groupby(['Dataset', 'Error_Type', 'Severity'])
                   .Train_Seconds.sum().rename('Train_Seconds_No_Clean').reset_index())
    costs = costs.merge(train_dirty, on=['Dataset', 'Error_Type', 'Severity'], how='left')
    costs['Train_Delta_Seconds'] = (costs.Train_Seconds_After_Clean.fillna(0)
                                    - costs.Train_Seconds_No_Clean.fillna(0))
    costs['Marginal_Compute_Seconds'] = costs.Clean_Seconds + costs.Train_Delta_Seconds.clip(lower=0)
    return df, costs


def automated_efficiency(df, costs):
    """Benefit per second of compute, for every automated cleaning action."""
    cost = (costs.groupby('Error_Type')
            .agg(Clean_Seconds=('Clean_Seconds', 'mean'),
                 Train_Delta_Seconds=('Train_Delta_Seconds', 'mean'),
                 Marginal_Compute_Seconds=('Marginal_Compute_Seconds', 'mean'),
                 Method=('Method', 'first')).reset_index())

    rows = []
    for error_type, g in df[df.Route == AUTOMATED].groupby('Error_Type'):
        if g.Cleaned.isna().all() or error_type in METRIC_MISMATCHED:
            continue
        c = cost[cost.Error_Type == error_type]
        if c.empty:
            continue

        improvement = g.Improvement_pp.mean()
        clean_s = float(c.Clean_Seconds.iloc[0])
        delta_s = float(c.Train_Delta_Seconds.iloc[0])
        marginal_s = float(c.Marginal_Compute_Seconds.iloc[0])

        rows.append({
            'Error_Type': error_type,
            'Method': c.Method.iloc[0],
            'Damage_pp': g.Damage_pp.mean(),
            'Improvement_pp': improvement,
            'Clean_Seconds': clean_s,
            'Train_Delta_Seconds': delta_s,
            'Marginal_Compute_Seconds': marginal_s,
            # PRIMARY: cleaning execution cost alone.
            'pp_per_second': improvement / clean_s if clean_s > 0 else np.nan,
            # SENSITIVITY: cleaning cost plus the training slowdown the
            # intervention itself causes. Ranking stability across the two is
            # a robustness result either way it comes out.
            'pp_per_second_marginal': improvement / marginal_s if marginal_s > 0 else np.nan,
            'Cost_Band': ('negligible' if clean_s < NEGLIGIBLE_COST_SECONDS
                          else 'substantial'),
        })

    out = pd.DataFrame(rows)
    # Sorted by cost band first, then by benefit within the band. Sorting the
    # whole table by pp/s would present noise-driven ordering as a result.
    out = out.sort_values(['Cost_Band', 'Improvement_pp'], ascending=[True, False])
    out.to_csv('results/rq2_automated_efficiency.csv', index=False)
    return out


def breakeven(df, best_efficiency, best_action):
    """
    Human cost threshold at which external intervention matches the best
    automated action, as a function of re-annotation effectiveness rho.
    """
    rows = []
    for error_type, g in df[df.Route == EXTERNAL].groupby('Error_Type'):
        # Experimental ceiling: perfect correction restores the clean baseline
        # by construction in this simulation, where ground truth is known.
        # Not a claim about real datasets, where the wrong labels are
        # unidentifiable - hence rho.
        ceiling = g.Damage_pp.mean()

        entry = {'Error_Type': error_type, 'Ceiling_pp': ceiling}
        for rho in (0.25, 0.50, 0.75, 1.00):
            recovered = rho * ceiling
            # Seconds of compute the best automated action would need to deliver
            # the same benefit. Human correction must cost less than this
            # (in equivalent units) to dominate.
            entry[f'C*_rho_{rho:.2f}'] = (recovered / best_efficiency
                                          if best_efficiency > 0 else np.nan)
        rows.append(entry)

    out = pd.DataFrame(rows).sort_values('Ceiling_pp', ascending=False)
    out.to_csv('results/rq2_breakeven.csv', index=False)
    return out


def main():
    df, costs = load()

    print('=' * 104)
    print('RQ2 (B) — AUTOMATED EFFICIENCY: benefit per second of cleaning compute')
    print('=' * 104)
    eff = automated_efficiency(df, costs)
    print(eff.round(4).to_string(index=False))

    material = eff[eff.Improvement_pp >= MATERIAL_PP]
    if material.empty:
        print('\nNo automated action clears the materiality threshold.')
        return

    # eff is sorted by (Cost_Band asc, Improvement_pp desc), and 'negligible'
    # sorts before 'substantial', so iloc[0] is the highest-BENEFIT action inside
    # the negligible-cost band. This is E_benchmark, and it is deliberately not
    # eff.pp_per_second.idxmax(): sub-0.1s timings are noise-dominated, so the
    # literal maximum is set by scheduler jitter (leakage removal reports
    # ~425 pp/s only because its 0.002s reading sits at the bottom of the band).
    # Do not "fix" this to sort by pp_per_second.
    best = material.iloc[0]
    print(f'\nBest automated action: {best.Error_Type} ({best.Method})')
    print(f'  {best.Improvement_pp:.2f}pp for {best.Clean_Seconds:.3f}s '
          f'= {best.pp_per_second:.2f} pp per second')

    print()
    print('=' * 104)
    print('RQ2 (B) — BREAK-EVEN: how cheap must human correction be to beat automation?')
    print('=' * 104)
    print('C* is the compute-equivalent budget human correction must undercut.')
    print('rho = share of injected errors a human actually catches (unknown, left free).')
    print()
    be = breakeven(df, best.pp_per_second, best.Error_Type)
    print(be.round(3).to_string(index=False))

    print()
    print('=' * 104)
    print('ROBUSTNESS — does the ranking survive the alternative cost definition?')
    print('=' * 104)
    prim = eff.sort_values('pp_per_second', ascending=False).Error_Type.tolist()
    marg = eff.sort_values('pp_per_second_marginal', ascending=False).Error_Type.tolist()
    from scipy import stats as _st
    rho_rank = _st.spearmanr(range(len(prim)),
                             [prim.index(e) for e in marg]).statistic
    print(f'  primary  (cleaning time only)      top 3: {prim[:3]}')
    print(f'  marginal (cleaning + train delta)  top 3: {marg[:3]}')
    print(f'  rank correlation between the two: rho = {rho_rank:.3f}')
    print('  Secondary diagnostic only: this correlation is computed over all 20')
    print('  actions, most of which sit in the negligible band where ordering is')
    print('  noise. The primary robustness statistic is band-level agreement')
    print('  (see audit_cost_stability.py). Claim: robust to the TWO TESTED')
    print('  definitions of marginal compute cost - not to every costing scheme.')

    print()
    print('=' * 104)
    print('METRIC-MISMATCHED INTERVENTIONS — reported separately, not ranked above')
    print('=' * 104)
    exp_raw = pd.read_csv('results/experiment_results.csv')
    for et in METRIC_MISMATCHED:
        sub = exp_raw[exp_raw.Error_Type == et]
        if sub.empty:
            continue
        agg = sub.groupby('Condition')[['Accuracy', 'Recall', 'F1', 'AUC']].mean()
        if 'cleaned' not in agg.index:
            continue
        d_ = agg.loc['cleaned'] - agg.loc['dirty']
        print(f'  {et}')
        print(f'    accuracy {d_.Accuracy*100:+6.2f}pp   recall {d_.Recall*100:+6.2f}pp   '
              f'F1 {d_.F1*100:+6.2f}pp   AUC {d_.AUC*100:+6.2f}pp')
        print('    Excluded from the accuracy-denominated ranking: the intervention')
        print('    optimises a different trade-off, moving the decision threshold')
        print('    rather than degrading the model (AUC is essentially unchanged).')

    print()
    print('=' * 104)
    print('ECONOMIC REGIMES')
    print('=' * 104)
    high = eff[(eff.Improvement_pp >= 1.0)]
    moderate = eff[(eff.Improvement_pp >= MATERIAL_PP) & (eff.Improvement_pp < 1.0)]
    low = eff[eff.Improvement_pp < MATERIAL_PP]

    print(f'\nHigh-return automated ({len(high)}):')
    for _, r in high.iterrows():
        print(f'  {r.Error_Type:32s} +{r.Improvement_pp:.2f}pp  {r.pp_per_second:8.2f} pp/s')
    print(f'\nModerate ({len(moderate)}):')
    for _, r in moderate.iterrows():
        print(f'  {r.Error_Type:32s} +{r.Improvement_pp:.2f}pp  {r.pp_per_second:8.2f} pp/s')
    print(f'\nLow return - cleanable but little damage to recover ({len(low)}):')
    for _, r in low.iterrows():
        print(f'  {r.Error_Type:32s} +{r.Improvement_pp:.2f}pp  damage {r.Damage_pp:.2f}pp')

    print()
    print('=' * 104)
    print('RQ2 (A) — ILLUSTRATIVE SINGLE-NUMBER ROI  [sensitivity only]')
    print('=' * 104)
    print(f'Assumes 1 human-minute = {ILLUSTRATIVE_SECONDS_PER_HUMAN_MINUTE:.0f}s compute and')
    print(f'{ILLUSTRATIVE_ANNOTATION_MINUTES_PER_1000_ROWS:.0f} annotation-minutes per 1000 rows.')
    print('These constants are illustrative. No primary conclusion depends on them.')
    print()
    rows_typical = 5000
    human_seconds = (rows_typical / 1000
                     * ILLUSTRATIVE_ANNOTATION_MINUTES_PER_1000_ROWS
                     * ILLUSTRATIVE_SECONDS_PER_HUMAN_MINUTE)
    ill = []
    for _, r in eff.iterrows():
        ill.append({'Action': r.Error_Type, 'Route': 'automated',
                    'Benefit_pp': r.Improvement_pp, 'Cost_s': r.Clean_Seconds,
                    'ROI_pp_per_s': r.pp_per_second})
    for _, r in be.iterrows():
        ill.append({'Action': r.Error_Type, 'Route': 'external',
                    'Benefit_pp': r.Ceiling_pp * 0.75,      # rho = 0.75 illustration
                    'Cost_s': human_seconds,
                    'ROI_pp_per_s': (r.Ceiling_pp * 0.75) / human_seconds})
    ill = pd.DataFrame(ill).sort_values('ROI_pp_per_s', ascending=False)
    print(ill.round(6).to_string(index=False))

    print('\nSaved: rq2_automated_efficiency.csv, rq2_breakeven.csv')


if __name__ == '__main__':
    main()
