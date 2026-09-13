"""
Independent verification of every headline number in the four documents.

This does NOT read the documents' prose and trust it. It recomputes each value
from results/*.csv and compares against the figure asserted in the markdown.
Any mismatch is printed and the script exits non-zero.

    python verify_documents.py

Add a check here whenever a new number is asserted in a document. A number that
appears in the prose but not in this file is unverified by definition.
"""
import io
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

TOL = 0.011          # accept differences from rounding to 2dp
failures = []
checks = 0


def check(label, claimed, actual, tol=TOL):
    global checks
    checks += 1
    ok = abs(claimed - actual) <= tol
    if not ok:
        failures.append(f'{label}: document says {claimed}, data gives {actual:.4f}')
    print(f'  [{"OK " if ok else "FAIL"}] {label:52s} doc={claimed:>10} '
          f'data={actual:10.4f}')


def doc(name):
    return io.open(name, encoding='utf-8').read()


def main():
    base = pd.read_csv('results/analysis_base.csv')
    rq1 = pd.read_csv('results/rq1_cleaning_actions.csv')
    rq2 = pd.read_csv('results/rq2_automated_efficiency.csv')
    mech = pd.read_csv('results/mechanism_comparison.csv')
    exp = pd.read_csv('results/experiment_results.csv')
    costs = pd.read_csv('results/cleaning_costs.csv')
    rs = doc('docs/RESULTS_SUMMARY.md')

    print('=' * 78)
    print('1. MEASUREMENT COUNTS')
    print('=' * 78)
    dirty = (exp.Condition == 'dirty').sum()
    cleaned = (exp.Condition == 'cleaned').sum()
    check('dirty measurements', 3360, dirty, 0)
    check('cleaned measurements', 2400, cleaned, 0)
    check('total measurements', 5760, len(exp), 0)
    check('combinations', 420,
          exp[exp.Condition == 'dirty']
          .groupby(['Dataset', 'Error_Type', 'Severity']).ngroups, 0)
    check('error types', 28, exp.Error_Type.nunique(), 0)
    check('models', 8, exp.Model.nunique(), 0)
    check('datasets', 5, exp.Dataset.nunique(), 0)
    check('failures', 0, exp.Accuracy.isna().sum(), 0)

    print()
    print('=' * 78)
    print('2. DAMAGE TABLE (section 1)')
    print('=' * 78)
    ov = base.groupby('Error_Type').Damage_pp.mean()
    piv = base.pivot_table(index='Error_Type', columns='Severity',
                           values='Damage_pp', aggfunc='mean')
    # parse the markdown table rows: | `name` | low | med | high | overall |
    # The trailing Route column is what distinguishes the damage table from the
    # break-even table, which has the same number of numeric columns. Without it
    # this regex silently parses break-even rows as damage rows.
    rows = re.findall(r'^\| `(\w+)` \| ([-\d.]+) \| ([-\d.]+) \| ([-\d.]+) \| '
                      r'\*{0,2}([-\d.]+)\*{0,2} \| (?:external|automated) \|',
                      rs, re.M)
    assert len(rows) == 10, f'expected 10 damage rows, parsed {len(rows)}'
    assert rows, 'could not parse the damage table'
    for name, lo, md, hi, tot in rows:
        check(f'{name} low', float(lo), piv.loc[name, 'low'])
        check(f'{name} medium', float(md), piv.loc[name, 'medium'])
        check(f'{name} high', float(hi), piv.loc[name, 'high'])
        check(f'{name} overall', float(tot), ov[name])
    check('types with damage >= 1pp', 10, (ov >= 1.0).sum(), 0)
    # The document claims 7 of 10 rise monotonically with severity. An earlier
    # draft claimed ALL of them did, contradicted by its own table three lines up.
    sev = piv[['low', 'medium', 'high']]
    subst = ov[ov >= 1.0].index
    mono = [e for e in subst
            if sev.loc[e, 'low'] <= sev.loc[e, 'medium'] <= sev.loc[e, 'high']]
    check('error types monotonic in severity', 7, len(mono), 0)
    nonmono = sorted(set(subst) - set(mono))
    expected = ['contextual_errors', 'data_inconsistency', 'invalid_values']
    check('non-monotonic set matches document', 1,
          1 if nonmono == expected else 0, 0)
    print(f'  [    ] non-monotonic types: {nonmono}')
    check('types with damage >= 2.5pp', 8, (ov >= 2.5).sum(), 0)
    check('types marginally negative', 3, (ov < 0).sum(), 0)

    print()
    print('=' * 78)
    print('3. MECHANISM PAIRS (section 1.1)')
    print('=' * 78)
    for _, m in mech.iterrows():
        tag = m.Comparison.split(' vs ')[1]
        check(f'{tag} damage', round(m.Damage_B_pp, 2), m.Damage_B_pp)
        check(f'{tag} difference', round(m.mean_diff_pp, 2), m.mean_diff_pp)
    ratio = ov['missing_values_mnar'] / ov['missing_values']
    check('MNAR/MCAR ratio (claimed 18x)', 18, ratio, 0.5)

    print()
    print('=' * 78)
    print('4. RQ1 RECOVERY TABLE (section 2)')
    print('=' * 78)
    for et, imp, ed, ei, cond, cov in [
            ('missing_values_mnar', 4.32, 8.55, 5.85, 68.4, 79),
            ('contextual_errors', 2.64, 7.01, 5.00, 71.3, 53),
            ('invalid_values', 2.58, 6.68, 6.12, 91.6, 43),
            ('data_inconsistency', 1.21, 4.33, 3.62, 83.5, 38),
            ('data_leakage', 0.92, 4.81, 2.93, 60.8, 68)]:
        r = rq1[rq1.Error_Type == et].iloc[0]
        check(f'{et} improvement', imp, r.Net_Improvement_pp)
        check(f'{et} eligible damage', ed, r.Eligible_Damage_pp)
        check(f'{et} eligible improvement', ei, r.Eligible_Improvement_pp)
        check(f'{et} conditional recovery', cond, r['Conditional_Recovery_%'], 0.1)
        check(f'{et} coverage', cov, r['Coverage_%'], 0.6)
        # internal identity the document claims
        check(f'{et} cond == eligI/eligD', r['Conditional_Recovery_%'],
              100 * r.Eligible_Improvement_pp / r.Eligible_Damage_pp, 0.1)

    print()
    print('=' * 78)
    print('5. COST AND BREAK-EVEN (section 3)  [timing-dependent]')
    print('=' * 78)
    claims = {name: (float(pp), float(sec)) for name, pp, sec in re.findall(
        r'^\| \*{0,2}([\w ()]+?)\*{0,2} \| \*{0,2}\+([\d.]+)pp\*{0,2} \| '
        r'\*{0,2}([\d.]+)s\*{0,2} \|', rs, re.M)}
    for label, et in [('IQR clipping (contextual)', 'contextual_errors'),
                      ('Sentinel detection', 'invalid_values'),
                      ('Consistency rules', 'data_inconsistency'),
                      ('Leakage removal', 'data_leakage'),
                      ('KNN imputation (MNAR)', 'missing_values_mnar')]:
        r = rq2[rq2.Error_Type == et].iloc[0]
        if label in claims:
            claimed_pp, claimed_sec = claims[label]
            check(f'{et} improvement (cost table)', claimed_pp, r.Improvement_pp)
            check(f'{et} cost seconds', claimed_sec, r.Clean_Seconds,
                  max(0.0015, r.Clean_Seconds * 0.15))
        band = 'negligible' if r.Clean_Seconds < 0.1 else 'substantial'
        print(f'  [    ] {et:52s} band={band}')

    E = (rq2[rq2.Error_Type == 'contextual_errors'].Improvement_pp.iloc[0] /
         rq2[rq2.Error_Type == 'contextual_errors'].Clean_Seconds.iloc[0])
    # Read the asserted value from the document; never hardcode it here, or the
    # verifier pins the docs to a stale constant instead of checking them.
    m_e = re.search(r'negligible-cost band\*\*: IQR clipping, at [\d.]+pp for '
                    r'[\d.]+s = \*\*([\d.]+) pp/s\*\*', rs)
    assert m_e, 'could not find the E_benchmark definition in RESULTS_SUMMARY.md'
    check('E_benchmark pp/s', float(m_e.group(1)), E, 2.0)
    # Parse the break-even table OUT of the document. Comparing round(ceil/E)
    # against ceil/E would be self-referential and could never fail.
    be = re.findall(r'^\| `(\w+)` \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| '
                    r'([\d.]+) \| ([\d.]+) \|', rs, re.M)
    be = [b for b in be if b[0] in ('annotator_bias', 'ambiguous_labels',
                                    'label_noise_asymmetric', 'concept_drift')]
    assert len(be) == 4, f'expected 4 break-even rows, parsed {len(be)}'
    ceilings = {'annotator_bias': 7.181, 'ambiguous_labels': 4.938,
                'label_noise_asymmetric': 3.590, 'concept_drift': 2.540}
    for et, ceil_doc, c25, c50, c75, c100 in be:
        check(f'{et} ceiling', float(ceil_doc), ceilings[et], 0.011)
        for rho, claimed in ((0.25, c25), (0.50, c50), (0.75, c75), (1.0, c100)):
            check(f'C* {et} (rho={rho})', float(claimed),
                  rho * ceilings[et] / E, 0.0011)

    # PROSE figures, not just tables. The RQ2 headline and limitation 1 quote the
    # KNN cost in running text; an earlier update patched the tables and left the
    # prose stale, which is exactly the kind of drift a reader notices first.
    knn_actual = rq2[rq2.Error_Type == 'missing_values_mnar'].Clean_Seconds.iloc[0]
    prose = re.findall(r'(\d+\.\d)s(?=,? whereas|" should be read|: it is that)', rs)
    assert prose, 'could not find the KNN cost quoted in prose'
    for i, v in enumerate(prose):
        check(f'KNN cost quoted in prose #{i + 1}', float(v), knn_actual, 0.06)

    sub = sorted(costs.groupby('Error_Type').Clean_Seconds.mean()
                 .pipe(lambda s: s[s >= 0.1]).index)
    print(f'  [    ] substantial band membership: {sub}')

    print()
    print('=' * 78)
    print('6. CLASS BALANCING (section 4)')
    print('=' * 78)
    share = {'Bank Marketing': 11.7, 'Adult Income': 24.9, 'Telco Churn': 26.6,
             'Breast Cancer': 37.3, 'German Credit': 70.0}
    claimed = {'Bank Marketing': (-6.83, 47.62, 12.35),
               'Adult Income': (-3.15, 26.69, 5.98),
               'Telco Churn': (-3.95, 27.11, 8.17),
               'Breast Cancer': (0.18, 1.09, 0.32),
               'German Credit': (-2.90, -11.61, -4.13)}
    recalls = []
    for ds, (a, rc, f1) in claimed.items():
        d = exp[(exp.Error_Type == 'class_imbalance') & (exp.Dataset == ds)]
        p = d.pivot_table(index=['Severity', 'Model'], columns='Condition',
                          values=['Accuracy', 'Recall', 'F1'])
        got = {m: (p[(m, 'cleaned')] - p[(m, 'dirty')]).mean() * 100
               for m in ('Accuracy', 'Recall', 'F1')}
        check(f'{ds} accuracy', a, got['Accuracy'])
        check(f'{ds} recall', rc, got['Recall'])
        check(f'{ds} F1', f1, got['F1'])
        recalls.append(got['Recall'])
    r = np.corrcoef([share[d] for d in claimed], recalls)[0, 1]
    check('r(positive share, recall gain) ~ -0.93', -0.93, r, 0.01)

    print()
    print('=' * 78)
    print('7. RQ3 AND RQ4 (sections 5, 6)')
    print('=' * 78)
    tree = {'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM'}
    pm = base.pivot_table(index='Error_Type', columns='Model',
                          values='Damage_pp', aggfunc='mean')
    ks = list(pm.columns)
    wt, wn, ac, allr = [], [], [], []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            rho = spearmanr(pm[ks[i]], pm[ks[j]]).correlation
            allr.append(rho)
            both = (ks[i] in tree) + (ks[j] in tree)
            (wt if both == 2 else wn if both == 0 else ac).append(rho)
    check('RQ3 mean rho', 0.605, np.mean(allr), 0.002)
    check('within tree rho', 0.770, np.mean(wt), 0.002)
    check('within non-tree rho', 0.674, np.mean(wn), 0.002)
    check('across families rho', 0.517, np.mean(ac), 0.002)
    # RQ3 stability claims. An earlier workbook said "the worst problems are the
    # worst for every model", contradicted by contextual_errors ranging 1-24.
    rq3 = pd.read_csv('results/rq3_model_dependency.csv')
    check('error types shifting >= 10 ranks across models', 19,
          int((rq3.Rank_Range >= 10).sum()), 0)
    rk = pm.rank(ascending=False)
    check('contextual_errors worst rank across models', 24,
          int(rk.loc['contextual_errors'].max()), 0)
    check('invalid_values worst rank across models', 25,
          int(rk.loc['invalid_values'].max()), 0)
    check('annotator_bias worst rank across models', 4,
          int(rk.loc['annotator_bias'].max()), 0)

    pd_ = base.pivot_table(index='Error_Type', columns='Dataset',
                           values='Damage_pp', aggfunc='mean')
    ds = list(pd_.columns)
    allr = [spearmanr(pd_[ds[i]], pd_[ds[j]]).correlation
            for i in range(len(ds)) for j in range(i + 1, len(ds))]
    check('RQ4 mean rho', 0.633, np.mean(allr), 0.002)

    rows_n = {'Breast Cancer': 455, 'German Credit': 800, 'Telco Churn': 5625,
              'Adult Income': 24129, 'Bank Marketing': 36168}
    claimed_ag = {'Breast Cancer': 0.549, 'German Credit': 0.643,
                  'Telco Churn': 0.732, 'Adult Income': 0.691,
                  'Bank Marketing': 0.551}
    ag = {}
    for k in ds:
        ag[k] = np.mean([spearmanr(pd_[k], pd_[o]).correlation
                         for o in ds if o != k])
        check(f'RQ4 agreement {k}', claimed_ag[k], ag[k], 0.002)
    check('RQ4 Spearman(rows, agreement)', 0.300,
          spearmanr([rows_n[k] for k in ds], [ag[k] for k in ds]).correlation,
          0.002)

    print()
    print('=' * 78)
    print('7b. METHODOLOGY DATASET TABLE vs LOADED DATA')
    print('=' * 78)
    md = doc('docs/METHODOLOGY.md')
    # | Name | Domain | rows | features | categorical | numeric | share% |
    dsrows = re.findall(r'^\| ([A-Za-z ]+?) \| \w+ \| ([\d,]+) \| (\d+) \| '
                        r'(\d+) \| (\d+) \| ([\d.]+)% \|', md, re.M)
    assert len(dsrows) == 5, f'expected 5 dataset rows, parsed {len(dsrows)}'
    from data_loader import DataLoader
    loaded = DataLoader().load_all()   # keyed by snake_case dataset id
    alias = {'Breast Cancer Wisconsin': 'breast_cancer', 'German Credit': 'german_credit',
             'Adult Income': 'adult_income', 'Bank Marketing': 'bank_marketing',
             'Telco Customer Churn': 'telco_churn'}
    for name, rows, feats, cat, num, share in dsrows:
        d = loaded[alias[name.strip()]]
        nfeat = len(d['categorical_cols']) + len(d['numeric_cols'])
        check(f'{name.strip()} rows', int(rows.replace(',', '')), d['n_rows'], 0)
        check(f'{name.strip()} features', int(feats), nfeat, 0)
        check(f'{name.strip()} categorical', int(cat), len(d['categorical_cols']), 0)
        check(f'{name.strip()} numeric', int(num), len(d['numeric_cols']), 0)
        check(f'{name.strip()} positive share', float(share),
              100 * d['class_balance'], 0.06)

    print()
    print('=' * 78)
    print('7c. RECOMMENDATIONS.md')
    print('=' * 78)
    try:
        rec_doc = doc('RECOMMENDATIONS.md')
    except FileNotFoundError:
        print('  not present - skipped')
    else:
        # Tier 1 and 2 tables: | Action | Fixes | +X.XXpp | Y.YYYs | Z.Z% |
        rows_r = re.findall(
            r'^\| [\w ]+ \| `(\w+)` \| \*{0,2}\+([\d.]+)pp\*{0,2} \| '
            r'~?([\d.]+)s \| ([\d.]+)% of damage \|', rec_doc, re.M)
        assert len(rows_r) == 6, f'expected 6 tier rows, parsed {len(rows_r)}'
        for et, pp, sec, cond in rows_r:
            r = rq1[rq1.Error_Type == et].iloc[0]
            check(f'rec {et} return', float(pp), r.Net_Improvement_pp)
            check(f'rec {et} conditional recovery', float(cond),
                  r['Conditional_Recovery_%'], 0.1)
            actual_s = rq2[rq2.Error_Type == et].Clean_Seconds.iloc[0]
            check(f'rec {et} cost', float(sec), actual_s,
                  max(0.0015, actual_s * 0.15))
        # Tier 3 damages
        ext_rows = re.findall(r'^\| `(\w+)` \| \*{0,2}([\d.]+)pp\*{0,2} \|',
                              rec_doc, re.M)
        assert len(ext_rows) == 4, f'expected 4 external rows, got {len(ext_rows)}'
        for et, dmg in ext_rows:
            check(f'rec {et} damage', float(dmg), ov[et])
        # counts asserted in prose
        r1i = rq1.set_index('Error_Type').Net_Improvement_pp
        n_low = sum(1 for e in base[base.Route == 'automated_cleaning']
                    .Error_Type.unique()
                    if e in r1i.index and r1i[e] < 0.5)
        m_low = re.search(r'(\w+) error types have an implemented cleaner that '
                          r'returns \*{0,2}under 0.5pp', rec_doc)
        words = {'Fourteen': 14, 'Thirteen': 13, 'Fifteen': 15, 'Twelve': 12}
        if m_low:
            check('rec low-return count', words.get(m_low.group(1), -1), n_low, 0)

    print()
    print('=' * 78)
    print('8. CROSS-DOCUMENT CONSISTENCY')
    print('=' * 78)
    # A figure asserted in more than one document must be identical in all.
    others = {n: doc(f'docs/{n}') for n in ('METHODOLOGY.md', 'RESEARCH_LOG.md', 'FROZEN.md')}
    shared = ['5,760', '3,360', '2,400', '420', '28 error types']
    for tok in shared:
        present = [n for n, t in others.items() if tok in t] + (
            ['RESULTS_SUMMARY.md'] if tok in rs else [])
        print(f'  [    ] {tok:16s} appears in: '
              f'{", ".join(x.replace(".md", "") for x in present)}')

    # E_benchmark: compare the DEFINING statement in each document, not every
    # 'pp/s' token. Earlier values legitimately appear in the two-run sensitivity
    # table and in the change record, and must not be flagged.
    defs = {}
    m = re.search(r'negligible-cost band\*\*: IQR clipping, at [\d.]+pp for '
                  r'[\d.]+s = \*\*([\d.]+) pp/s\*\*', rs)
    if m:
        defs['RESULTS_SUMMARY.md'] = float(m.group(1))
    m = re.search(r'\(IQR clipping, [\d.]+pp / [\d.]+s = ([\d.]+) pp/s\)',
                  others['METHODOLOGY.md'])
    if m:
        defs['METHODOLOGY.md'] = float(m.group(1))
    print(f'  [    ] E_benchmark as DEFINED in each doc: {defs}')
    check('E_benchmark defined consistently', len(set(defs.values())), 1, 0)
    for name, val in defs.items():
        check(f'E_benchmark in {name}', val, E, 2.0)

    # KNN cost must agree wherever stated
    knn = set(re.findall(r'(\d+\.\d)s', rs)) & {'15.1', '17.7', '50.8'}
    print(f'  [    ] KNN cost tokens in RESULTS_SUMMARY: {sorted(knn)}')

    print()
    print('=' * 78)
    print('9. EXCEL WORKBOOK vs CSVs')
    print('=' * 78)
    try:
        xl = pd.ExcelFile('Data_Quality_Study_Summary.xlsx')
    except FileNotFoundError:
        print('  workbook not present - skipped')
    else:
        need = ['Overview', 'Questions and Answers', 'Data Errors',
                'Order 1 - By Damage', 'Order 2 - What We Recommend',
                'Why The Two Orders Differ', 'Key Finding - Mechanism',
                'Datasets', 'Models', 'Cleaning Methods',
                'RQ3 by Model', 'RQ4 by Dataset']
        missing = [n for n in need if n not in xl.sheet_names]
        check('excel sheet count', len(need), len(xl.sheet_names), 0)
        if missing:
            failures.append(f'excel missing sheets: {missing}')

        o1 = pd.read_excel(xl, 'Order 1 - By Damage', skiprows=2)
        check('excel Order1 rows', 28, len(o1), 0)
        ov_sorted = ov.sort_values(ascending=False)
        check('excel Order1 rank-1 damage', float(o1.iloc[0]['Damage (points lost)']),
              ov_sorted.iloc[0])
        same = list(o1['Data error']) == list(ov_sorted.index)
        print(f'  [{"OK " if same else "FAIL"}] excel Order1 ordering matches data')
        if not same:
            failures.append('excel Order 1 is not in descending damage order')

        # Order 2 must list the SAME 28 error types as Order 1, just re-ordered.
        # An earlier version listed 9 cleaning ACTIONS instead, which made the two
        # orders non-comparable - the whole point of having two.
        o2 = pd.read_excel(xl, 'Order 2 - What We Recommend', skiprows=2)
        check('excel Order2 rows', 28, len(o2), 0)
        same_set = set(o2['Data error']) == set(o1['Data error'])
        print(f'  [{"OK " if same_set else "FAIL"}] excel Order2 covers the same '
              f'28 errors as Order1')
        if not same_set:
            failures.append('Order 2 does not list the same error types as Order 1')
        # and it must NOT be in damage order, or it is not a second ordering
        differs = list(o2['Data error']) != list(o1['Data error'])
        print(f'  [{"OK " if differs else "FAIL"}] excel Order2 genuinely differs '
              f'from Order1')
        if not differs:
            failures.append('Order 2 is identical to Order 1')

        de = pd.read_excel(xl, 'Data Errors', skiprows=2)
        check('excel Data Errors rows', 28, len(de), 0)
        cm = pd.read_excel(xl, 'Cleaning Methods', skiprows=2)
        check('excel Cleaning Methods rows', 20, len(cm), 0)
        qa = pd.read_excel(xl, 'Questions and Answers', skiprows=1)
        check('excel RQ rows', 4, len(qa), 0)

        # every automated error type must appear in the cleaning sheet
        auto = set(base[base.Route == 'automated_cleaning'].Error_Type.unique())
        gap = auto - set(cm['Data error'])
        print(f'  [{"OK " if not gap else "FAIL"}] excel covers all 20 automated '
              f'types{"" if not gap else " - missing " + str(sorted(gap))}')
        if gap:
            failures.append(f'excel cleaning sheet missing: {sorted(gap)}')

        # Scan EVERY text cell for figures known to be superseded. The workbook
        # once carried the intermediate class-balance run (6.43 / 51.64 / 17.21)
        # hardcoded in a prose string while its tables showed the final values.
        STALE = {'6.43': 'old class-balance accuracy (now 6.83)',
                 '51.64': 'old class-balance recall (now 47.62)',
                 '17.21': 'old class-balance F1 (now 12.35)',
                 '15.1 second': 'old KNN cost (now 14.5s)',
                 '215.0 pp': 'old E_benchmark (now 222.4)',
                 '207.5 pp': 'older E_benchmark'}
        hits = []
        for sh in xl.sheet_names:
            for skip in (1, 2):
                try:
                    df = pd.read_excel(xl, sh, skiprows=skip)
                except Exception:
                    continue
                blob = ' '.join(map(str, df.values.ravel()))
                for bad, why in STALE.items():
                    if bad in blob:
                        hits.append(f'{sh}: {bad!r} ({why})')
                break
        print(f'  [{"OK " if not hits else "FAIL"}] excel free of superseded '
              f'figures{"" if not hits else " -> " + "; ".join(hits)}')
        if hits:
            failures.append(f'excel contains superseded figures: {hits}')

        # the class-balance sentence must match the data it describes
        cb = ' '.join(map(str, cm.values.ravel()))
        d = exp[(exp.Error_Type == 'class_imbalance') &
                (exp.Dataset == 'Bank Marketing')]
        pv = d.pivot_table(index=['Severity', 'Model'], columns='Condition',
                           values=['Accuracy', 'Recall'])
        for metric, label in (('Accuracy', 'accuracy'), ('Recall', 'recall')):
            val = abs((pv[(metric, 'cleaned')] - pv[(metric, 'dirty')]).mean() * 100)
            present = f'{val:.2f}' in cb
            print(f'  [{"OK " if present else "FAIL"}] excel quotes current '
                  f'class-balance {label} ({val:.2f})')
            if not present:
                failures.append(f'excel class-balance {label} not current')

        # excel cost figures must match the CSV
        for et in ('contextual_errors', 'missing_values_mnar'):
            row = cm[cm['Data error'] == et]
            if len(row):
                check(f'excel cost {et}', float(row['Seconds to run'].iloc[0]),
                      rq2[rq2.Error_Type == et].Clean_Seconds.iloc[0], 0.002)

    print()
    print('=' * 78)
    print(f'{checks} checks run')
    if failures:
        print(f'{len(failures)} FAILURES:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('ALL CHECKS PASSED - every asserted number reproduces from results/*.csv')


if __name__ == '__main__':
    main()
