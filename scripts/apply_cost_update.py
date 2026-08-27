"""
Rewrite every timing-derived figure in the documents from the current CSVs.

Run AFTER build_roi_model.py has regenerated results/rq2_*.csv, then run
verify_documents.py. Prints old -> new for each figure so the change is visible.

Only touches timing-derived numbers. RQ1 / RQ3 / RQ4 / mechanism / class-balance
figures are not timing-derived and are never rewritten by this script.
"""
import io
import re

import pandas as pd

DOCS = ['RESULTS_SUMMARY.md', 'METHODOLOGY.md', 'RESEARCH_LOG.md', 'FROZEN.md']
CEILINGS = {'annotator_bias': 7.181, 'ambiguous_labels': 4.938,
            'label_noise_asymmetric': 3.590, 'concept_drift': 2.540}
LABELS = {'contextual_errors': 'IQR clipping (contextual)',
          'invalid_values': 'Sentinel detection',
          'data_inconsistency': 'Consistency rules',
          'data_leakage': 'Leakage removal',
          'missing_values_mnar': 'KNN imputation (MNAR)'}

changes = []


def sub(text, old, new, where):
    """Replace and record. Missing `old` is reported, not silently ignored."""
    if old not in text:
        changes.append(f'  !! NOT FOUND in {where}: {old!r}')
        return text
    changes.append(f'  {where}: {old!r} -> {new!r}')
    return text.replace(old, new)


def fmt_sec(s):
    return f'{s:.3f}' if s < 1 else f'{s:.1f}'


def main():
    rq2 = pd.read_csv('results/rq2_automated_efficiency.csv').set_index('Error_Type')
    costs = pd.read_csv('results/cleaning_costs.csv')

    ctx = rq2.loc['contextual_errors']
    E = ctx.Improvement_pp / ctx.Clean_Seconds
    lead = rq2.pp_per_second.max()

    print(f'E_benchmark      = {E:.1f} pp/s  (IQR clipping, '
          f'{ctx.Improvement_pp:.2f}pp / {ctx.Clean_Seconds:.4f}s)')
    print(f'literal max_j    = {lead:.1f} pp/s')
    print(f'KNN (MNAR) cost  = {rq2.loc["missing_values_mnar"].Clean_Seconds:.4f}s')
    sub_band = sorted(costs.groupby('Error_Type').Clean_Seconds.mean()
                      .pipe(lambda s: s[s >= 0.1]).index)
    print(f'substantial band = {sub_band}')
    print()

    rs = io.open('RESULTS_SUMMARY.md', encoding='utf-8').read()

    # ---- cost table rows -------------------------------------------------
    for et, label in LABELS.items():
        new_s = rq2.loc[et].Clean_Seconds
        pat = re.compile(r'(\| \*{0,2}' + re.escape(label) +
                         r'\*{0,2} \| \*{0,2}\+[\d.]+pp\*{0,2} \| \*{0,2})'
                         r'([\d.]+)(s\*{0,2} \|)')
        m = pat.search(rs)
        if m:
            changes.append(f'  cost table {et}: {m.group(2)}s -> {fmt_sec(new_s)}s')
            rs = pat.sub(lambda mm: mm.group(1) + fmt_sec(new_s) + mm.group(3),
                         rs, count=1)
        else:
            changes.append(f'  !! cost table row not matched: {label}')

    # ---- break-even grid -------------------------------------------------
    for et, ceil in CEILINGS.items():
        vals = ' | '.join(f'{r * ceil / E:.3f}' for r in (0.25, 0.5, 0.75, 1.0))
        pat = re.compile(r'^\| `' + et + r'` \| [\d.]+ \| [\d.]+ \| [\d.]+ \| '
                         r'[\d.]+ \| [\d.]+ \|$', re.M)
        new_row = f'| `{et}` | {ceil:.2f} | {vals} |'
        if pat.search(rs):
            changes.append(f'  break-even {et} -> {vals}')
            rs = pat.sub(new_row, rs, count=1)
        else:
            changes.append(f'  !! break-even row not matched: {et}')

    # ---- E_benchmark and the literal maximum ------------------------------
    rs = sub(rs, '2.64pp for 0.012s = **215.0 pp/s**',
             f'{ctx.Improvement_pp:.2f}pp for {ctx.Clean_Seconds:.3f}s = '
             f'**{E:.1f} pp/s**', 'RESULTS_SUMMARY E_benchmark')
    rs = sub(rs, '215.0 pp/s the benchmark is less than half the 444.4 pp/s',
             f'{E:.1f} pp/s the benchmark is less than half the {lead:.1f} pp/s',
             'RESULTS_SUMMARY conservatism')
    rs = sub(rs, 'treating 215.0 pp/s as exact',
             f'treating {E:.1f} pp/s as exact', 'RESULTS_SUMMARY precision note')
    rs = sub(rs, 'leakage removal shows ~444 pp/s',
             f'leakage removal shows ~{lead:.0f} pp/s', 'RESULTS_SUMMARY jitter')

    io.open('RESULTS_SUMMARY.md', 'w', encoding='utf-8').write(rs)

    # ---- METHODOLOGY ------------------------------------------------------
    md = io.open('METHODOLOGY.md', encoding='utf-8').read()
    md = sub(md, '(IQR clipping, 2.64pp / 0.012s = 215.0 pp/s)',
             f'(IQR clipping, {ctx.Improvement_pp:.2f}pp / '
             f'{ctx.Clean_Seconds:.3f}s = {E:.1f} pp/s)', 'METHODOLOGY E_benchmark')
    md = sub(md, 'at\n215.0 pp/s the benchmark is less than half the 444.4 pp/s',
             f'at\n{E:.1f} pp/s the benchmark is less than half the {lead:.1f} pp/s',
             'METHODOLOGY conservatism')
    md = sub(md, 'treating 215.0 pp/s as exact', f'treating {E:.1f} pp/s as exact',
             'METHODOLOGY precision note')
    io.open('METHODOLOGY.md', 'w', encoding='utf-8').write(md)

    print('CHANGES')
    for c in changes:
        print(c)
    missed = [c for c in changes if '!!' in c]
    print()
    if missed:
        print(f'{len(missed)} TARGETS NOT FOUND - fix these before trusting the docs')
    else:
        print('all targets matched')


if __name__ == '__main__':
    main()
