"""
Severity-sensitivity comparison for the DAMAGE ranking: compares the
21-error-type damage ranking under the original severity schedule against
the lower (x0.7) and higher (x1.5) native-unit schedules produced by
severity_sensitivity_damage.py, to check whether the headline damage
ranking depends on the specific severity values chosen.

Read-only analysis, printed to stdout for manual review before writing
anything into the paper.
"""
import numpy as np
import pandas as pd

SCOPED_ERROR_TYPES = sorted(set([
    'annotator_bias', 'missing_values_mnar', 'ambiguous_labels',
    'contextual_errors', 'label_noise_asymmetric', 'data_leakage',
    'concept_drift', 'data_heterogeneity', 'invalid_values',
    'data_inconsistency', 'label_noise',
    'missing_values', 'outliers', 'outliers_targeted', 'duplicates',
    'duplicates_minority', 'duplicates_targeted', 'typographical_errors',
    'categorical_errors', 'feature_noise', 'feature_noise_targeted',
]))

SEEDS = [42]  # severity_sensitivity_damage.py is scoped to seed 42 only
              # (runtime); compare against the ORIGINAL schedule's seed-42
              # data only too, so this is an apples-to-apples single-seed
              # comparison rather than 3-seed-pooled vs 1-seed.


def load_baseline():
    frames = []
    for seed in SEEDS:
        suffix = '' if seed == 42 else f'_seed{seed}'
        b = pd.read_csv(f'results/baseline_results{suffix}.csv')
        b['Random_State'] = seed
        frames.append(b)
    base = pd.concat(frames, ignore_index=True)
    return base.groupby(['Dataset', 'Model', 'Random_State']).Accuracy.mean() \
               .rename('Clean_Accuracy').reset_index()


def original_damage(clean):
    """Damage_pp under the ORIGINAL severity schedule, for the 21 scoped
    types, from the main 3-seed experiment files (dirty condition only)."""
    frames = []
    for seed in SEEDS:
        suffix = '' if seed == 42 else f'_seed{seed}'
        e = pd.read_csv(f'results/experiment_results{suffix}.csv')
        e['Random_State'] = seed
        frames.append(e)
    exp = pd.concat(frames, ignore_index=True)
    exp = exp[(exp.Condition == 'dirty') & exp.Error_Type.isin(SCOPED_ERROR_TYPES)]
    dirty = (exp.groupby(['Dataset', 'Error_Type', 'Severity', 'Model', 'Random_State'])
             .Accuracy.mean().rename('Dirty_Accuracy').reset_index())
    d = dirty.merge(clean, on=['Dataset', 'Model', 'Random_State'], how='left')
    d['Damage_pp'] = (d.Clean_Accuracy - d.Dirty_Accuracy) * 100
    return d


def schedule_damage(clean, schedule_name):
    df = pd.read_csv('results/severity_sensitivity_damage.csv')
    df = df[(df.Condition == 'dirty') & (df.Metadata.str.contains(f"'schedule': '{schedule_name}'"))]
    # severity_label in metadata is the ORIGINAL severity this rate stands in
    # for; the Severity column itself is already set to that same label by
    # DegradationTester.evaluate's positional argument in the runner script.
    d = df.groupby(['Dataset', 'Error_Type', 'Severity', 'Model', 'Random_State']) \
          .Accuracy.mean().rename('Dirty_Accuracy').reset_index()
    d = d.merge(clean, on=['Dataset', 'Model', 'Random_State'], how='left')
    d['Damage_pp'] = (d.Clean_Accuracy - d.Dirty_Accuracy) * 100
    return d


def ranking(df, label):
    r = df.groupby('Error_Type').Damage_pp.mean().sort_values(ascending=False)
    print(f'\n--- Damage ranking (21 scoped types) [{label}] ---')
    print(r.round(3).to_string())
    return r


def compare_top_n(rankings, n):
    tops = {label: set(r.index[:n]) for label, r in rankings.items()}
    orig = tops['original']
    print(f'\ntop-{n} overlap with original:')
    for label, s in tops.items():
        if label == 'original':
            continue
        print(f'  {label}: {len(orig & s)}/{n} shared, missing={sorted(orig - s)}, '
              f'new={sorted(s - orig)}')


def main():
    clean = load_baseline()
    rankings = {
        'original': ranking(original_damage(clean), 'original'),
        'lower': ranking(schedule_damage(clean, 'lower'), 'lower (x0.7)'),
        'higher': ranking(schedule_damage(clean, 'higher'), 'higher (x1.5)'),
    }
    for n in (3, 5, 10):
        compare_top_n(rankings, n)

    print('\n--- side-by-side rank position (1 = most damaging) ---')
    table = pd.DataFrame({
        label: r.rank(ascending=False).astype(int) for label, r in rankings.items()
    }).sort_values('original')
    print(table.to_string())


if __name__ == '__main__':
    main()
