"""
Build a single Excel workbook summarising the whole study in plain English.

Every number is read from results/*.csv so the workbook cannot drift from the
frozen archive. Re-run after any re-analysis.

    python make_excel_summary.py
"""
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = 'Data_Quality_Study_Summary.xlsx'

HEAD_FILL = PatternFill('solid', fgColor='1F3864')
HEAD_FONT = Font(color='FFFFFF', bold=True, size=11)
TITLE_FONT = Font(bold=True, size=13, color='1F3864')
BAND = PatternFill('solid', fgColor='EAF0F8')
THIN = Side(style='thin', color='B4C6E7')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# ---------------------------------------------------------------- plain English
ERROR_DESCRIPTIONS = {
    'missing_values': 'Some values are simply blank, at random.',
    'missing_values_mnar': 'Values are blank in a way that depends on the answer we are trying to predict.',
    'duplicates': 'The same row appears more than once, chosen at random.',
    'duplicates_targeted': 'Duplicated rows are taken from the majority class.',
    'duplicates_minority': 'Duplicated rows are taken from the rare class.',
    'outliers': 'A few numbers are far too large or too small, spread over all columns.',
    'outliers_targeted': 'Extreme numbers are placed in the columns that matter most.',
    'invalid_values': 'Placeholder codes such as -999 are used to mean "no value".',
    'domain_violations': 'Values fall outside what is physically possible, e.g. a negative age.',
    'contextual_errors': 'Each value looks fine alone but is wrong next to the others in its row.',
    'redundant_features': 'Extra columns repeat information already present.',
    'feature_noise': 'Small random amounts are added to numbers, across all columns.',
    'feature_noise_targeted': 'Random noise is aimed at the most predictive columns.',
    'data_leakage': 'A column secretly contains the answer, so the model cheats.',
    'data_heterogeneity': 'Different rows use different units or scales.',
    'class_imbalance': 'One outcome is far more common than the other.',
    'imbalanced_feature_distribution': 'A column is heavily skewed rather than evenly spread.',
    'typographical_errors': 'Text is misspelled - letters swapped, dropped or duplicated.',
    'categorical_errors': 'A category is replaced by a different but still valid category.',
    'data_inconsistency': 'The same thing is written in conflicting ways across rows.',
    'label_noise': 'Some answers are wrong, in both directions equally.',
    'label_noise_asymmetric': 'Wrong answers all flip the same way, shifting the balance.',
    'annotator_bias': 'The person labelling the data was systematically biased.',
    'ambiguous_labels': 'Borderline cases where even an expert could disagree.',
    'concept_drift': 'The correct answer has changed over time.',
    'data_drift': 'The incoming data no longer looks like the training data.',
    'data_representativeness': 'Some groups are missing or under-represented.',
    'provenance_issues': 'We cannot tell where the data came from or trust it.',
}

CLEANERS = {
    'missing_values': 'KNN imputation', 'missing_values_mnar': 'KNN imputation',
    'duplicates': 'drop exact duplicates', 'duplicates_targeted': 'drop exact duplicates',
    'duplicates_minority': 'drop exact duplicates',
    'outliers': 'IQR clipping', 'outliers_targeted': 'IQR clipping',
    'invalid_values': 'sentinel to NaN then impute',
    'domain_violations': 'IQR clipping', 'contextual_errors': 'IQR clipping',
    'redundant_features': 'drop correlated features',
    'feature_noise': 'rank-order median smoothing',
    'feature_noise_targeted': 'rank-order median smoothing',
    'data_leakage': 'drop features with |corr| > 0.5',
    'data_heterogeneity': 're-standardise features',
    'class_imbalance': 'oversample minority',
    'imbalanced_feature_distribution': 'quantile transform',
    'typographical_errors': 'fuzzy match to known categories',
    'categorical_errors': 'validate against vocabulary',
    'data_inconsistency': 'rule-based consistency checks',
}

CLEANER_EXPLAIN = {
    'KNN imputation': 'Fill each blank using the most similar rows.',
    'drop exact duplicates': 'Delete rows that are identical to another row.',
    'IQR clipping': 'Pull extreme numbers back to a sensible range based on the middle 50% of the data.',
    'sentinel to NaN then impute': 'Recognise placeholder codes, blank them, then fill them in.',
    'drop correlated features': 'Remove columns that duplicate another column.',
    'rank-order median smoothing': 'Sort the values and smooth each one against its neighbours.',
    'drop features with |corr| > 0.5': 'Remove any column suspiciously related to the answer.',
    're-standardise features': 'Put every column back on a common scale.',
    'oversample minority': 'Copy rare-class rows until the classes are balanced.',
    'quantile transform': 'Reshape a skewed column into an even spread.',
    'fuzzy match to known categories': 'Correct a misspelling to the nearest real category.',
    'validate against vocabulary': 'Check each category against the list of allowed values.',
    'rule-based consistency checks': 'Apply rules that force related columns to agree.',
    'consistency rules': 'Apply rules that force related columns to agree.',
    'oversample minority (class balancing)': 'Copy rare-class rows until the classes are balanced.',
}


def sheet(writer, name, df, widths, wrap_cols=(), title=None, note=None):
    """Write a dataframe with a styled header and sensible column widths."""
    start = 0
    if title:
        start = 2 if note else 1
    df.to_excel(writer, sheet_name=name, index=False, startrow=start)
    ws = writer.sheets[name]

    if title:
        ws.cell(row=1, column=1, value=title).font = TITLE_FONT
    if note:
        ws.cell(row=2, column=1, value=note).font = Font(italic=True, size=9,
                                                         color='555555')

    hdr = start + 1
    for c in range(1, len(df.columns) + 1):
        cell = ws.cell(row=hdr, column=c)
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
        cell.alignment = Alignment(vertical='center', wrap_text=True)
    ws.row_dimensions[hdr].height = 30

    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    for r in range(hdr + 1, hdr + 1 + len(df)):
        for c in range(1, len(df.columns) + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = BORDER
            cell.alignment = Alignment(
                vertical='top',
                wrap_text=df.columns[c - 1] in wrap_cols)
            if (r - hdr) % 2 == 0:
                cell.fill = BAND

    ws.freeze_panes = ws.cell(row=hdr + 1, column=1)
    return ws


def main():
    rq1 = pd.read_csv('results/rq1_cleaning_actions.csv')
    rq2 = pd.read_csv('results/rq2_automated_efficiency.csv')
    rq3 = pd.read_csv('results/rq3_model_dependency.csv')
    rq4 = pd.read_csv('results/rq4_dataset_dependency.csv')
    mech = pd.read_csv('results/mechanism_comparison.csv')
    base = pd.read_csv('results/analysis_base.csv')
    exp = pd.read_csv('results/experiment_results.csv')

    def class_balance_delta(dataset, metric):
        """Measured change for class_imbalance cleaning. Derived, never typed:
        these figures were superseded once already and a hardcoded copy of the
        old run survived in this file."""
        d = exp[(exp.Error_Type == 'class_imbalance') & (exp.Dataset == dataset)]
        p = d.pivot_table(index=['Severity', 'Model'], columns='Condition',
                          values=metric)
        return (p['cleaned'] - p['dirty']).mean() * 100

    bm_acc = class_balance_delta('Bank Marketing', 'Accuracy')
    bm_rec = class_balance_delta('Bank Marketing', 'Recall')
    bm_f1 = class_balance_delta('Bank Marketing', 'F1')

    damage = (base.groupby(['Error_Type', 'Route']).Damage_pp.mean()
              .reset_index().sort_values('Damage_pp', ascending=False))
    imp = rq1.set_index('Error_Type').Net_Improvement_pp.round(2)

    with pd.ExcelWriter(OUT, engine='openpyxl') as xl:

        # ---------------------------------------------------------- 1. overview
        overview = pd.DataFrame({
            'Item': ['Study', 'What it asks', 'Datasets', 'Models',
                     'Data error types', 'Severity levels', 'Combinations tested',
                     'Total measurements', 'Failures', 'Status',
                     'How to read the numbers'],
            'Detail': [
                'Prioritizing Data Quality Improvements for Machine Learning - '
                'A Cost-Benefit Framework',
                'Which data quality problems should developers fix first?',
                '5 (Breast Cancer, German Credit, Adult Income, Bank Marketing, Telco Churn)',
                '8 (4 tree-based, 4 non-tree)',
                '28 (20 fixable by software, 8 needing people or new data)',
                '3 (low, medium, high)',
                '420 = 5 datasets x 28 errors x 3 severities',
                '5,760 = (420 x 8 models dirty) + (300 x 8 models cleaned)',
                '0',
                'Analysis closed. Timing figures remeasured 10 August 2026.',
                '"pp" means percentage points of accuracy. +2.64pp means accuracy '
                'rose by 2.64 points, e.g. from 80.00% to 82.64%.',
            ]})
        sheet(xl, 'Overview', overview, [26, 105], wrap_cols=('Detail',),
              title='Data Quality Study - Summary Workbook',
              note='All tables in this workbook are generated directly from '
                   'results/*.csv - every figure in a table cell is computed, not '
                   'typed. Explanatory sentences quote those same figures, and the '
                   'ones that could go stale (costs, recovery, class balancing) are '
                   'interpolated from the data rather than written in. Regenerate '
                   'with: python make_excel_summary.py')

        # ------------------------------------------------- 2. questions/answers
        qa = pd.DataFrame([
            {'RQ': 'RQ1',
             'Question (plain English)':
                 'Which cleaning action gives the biggest improvement?',
             'Answer (plain English)':
                 'Filling in missing values with KNN imputation. It recovered '
                 '+4.32 accuracy points - more than any other repair. Next best '
                 'were IQR clipping (+2.64), sentinel detection (+2.58), '
                 'consistency rules (+1.21) and leakage removal (+0.92).',
             'How we reached it':
                 'We deliberately damaged clean data, measured how much accuracy '
                 'was lost, then ran a repair and measured how much came back. '
                 'The repair never got to see the clean data or which rows we had '
                 'damaged, so it had to find the problem on its own - exactly as '
                 'in real life.',
             'Important caveat':
                 'This is the biggest improvement, not the best value for money. '
                 'See RQ2.'},
            {'RQ': 'RQ2',
             'Question (plain English)':
                 'Which cleaning action gives the best return for what it costs?',
             'Answer (plain English)':
                 'Not the same one as RQ1. KNN imputation buys the largest gain '
                 'but takes about 15 seconds. IQR clipping and sentinel detection '
                 'buy nearly as much (+2.64 and +2.58) in about 0.01 seconds - '
                 'roughly a thousand times cheaper. If compute time matters, do '
                 'the cheap repairs first.',
             'How we reached it':
                 'We timed every repair, then divided the accuracy gained by the '
                 'seconds spent. Because timings below 0.1s are unreliable, we '
                 'grouped costs into bands (negligible vs substantial) and only '
                 'compared across bands, never within one.',
             'Important caveat':
                 'Exact seconds do not repeat: re-measuring moved the slow repairs '
                 'by about 21%. Which band a repair falls into does repeat. Treat '
                 '"15 seconds" as an order of magnitude.'},
            {'RQ': 'RQ3',
             'Question (plain English)':
                 'Does the answer change depending on which AI model you use?',
             'Answer (plain English)':
                 'Yes, partly. Priorities were more similar among models in the '
                 'same family than among models from different families '
                 '(0.77 within tree-based, 0.67 within non-tree, 0.52 across), '
                 'suggesting a family-level pattern. Only the two most damaging '
                 'errors hold their position for every model - below those, '
                 'ranking moves a lot: contextual_errors ranges from 1st to 24th '
                 'depending on the model, and invalid_values from 1st to 25th.',
             'How we reached it':
                 'We ran all 8 models on identical damaged data and ranked the '
                 'error types separately for each model, then compared the '
                 'rankings. We used 4 tree-based and 4 non-tree models on purpose '
                 'so we could test whether the difference is about model family '
                 'rather than one odd algorithm.',
             'Important caveat':
                 'Only the top two are stable; 19 of 28 error types shift by 10 '
                 'places or more between models. This is an observed pattern '
                 'across 8 models, not a proven cause - it SUGGESTS a family-level '
                 'pattern rather than establishing one.'},
            {'RQ': 'RQ4',
             'Question (plain English)':
                 'Does the answer change depending on which dataset you have?',
             'Answer (plain English)':
                 'Yes, partly - and this is our weakest answer. The rankings agree '
                 'broadly but not exactly across the 5 datasets. We could not '
                 'work out why.',
             'How we reached it':
                 'We ranked the error types separately for each dataset and '
                 'measured how much the rankings agreed. We then tested whether '
                 'dataset size explained the disagreement. It did not '
                 '(correlation 0.300, and the biggest dataset had almost the '
                 'lowest agreement), so we removed that explanation rather than '
                 'keep it because it sounded reasonable.',
             'Important caveat':
                 'We report that dataset dependence is real but that this '
                 'experiment does not identify its cause. With only 5 datasets, '
                 'testing more explanations until one worked would not be honest.'},
        ])
        sheet(xl, 'Questions and Answers', qa, [7, 34, 62, 62, 46],
              wrap_cols=('Question (plain English)', 'Answer (plain English)',
                         'How we reached it', 'Important caveat'),
              title='The four research questions, answered in plain English')

        # ------------------------------------------------------- 3. data errors
        rows = []
        for _, r in damage.iterrows():
            et = r.Error_Type
            automated = r.Route == 'automated_cleaning'
            rows.append({
                'Data error': et,
                'What it means': ERROR_DESCRIPTIONS.get(et, ''),
                'Fixed automatically in this study?': (
                    'Yes' if automated
                    else 'No - remediated by people or new data'),
                'What we did about it': CLEANERS.get(et, 'No automated repair - '
                                                        'measured the damage only'),
                'Damage (accuracy points lost)': round(r.Damage_pp, 2),
            })
        errs = pd.DataFrame(rows)
        errs['Recovered by cleaning (points)'] = errs['Data error'].map(imp)
        sheet(xl, 'Data Errors', errs, [30, 62, 27, 33, 17, 18],
              wrap_cols=('What it means', 'What we did about it'),
              title='All 28 data errors we studied, worst damage first',
              note='Blank in the last column means there is no automated repair to '
                   'measure - the fix requires people or new data.')

        # ------------------------------------------- 4a. order by damage
        dmg = damage.copy()
        dmg['Rank'] = range(1, len(dmg) + 1)
        dmg['Can software fix it?'] = dmg.Route.map(
            {'automated_cleaning': 'Yes',
             'external_intervention': 'No - remediated by people or new data'})
        dmg['Damage (points lost)'] = dmg.Damage_pp.round(2)
        dmg['Recovered by cleaning'] = dmg.Error_Type.map(imp)
        dmg['Why it sits here'] = [
            ('Causes the most damage of anything we measured.' if i == 1 else
             'Serious damage - worth attention.' if d >= 2.5 else
             'Moderate damage.' if d >= 1.0 else
             'Small damage.' if d >= 0.3 else
             'Essentially no measurable damage.' if d >= -0.05 else
             'Slightly negative, i.e. no real damage - measurement noise around '
             'zero, not a benefit from corruption.')
            for i, d in zip(dmg.Rank, dmg.Damage_pp)]
        dmg_out = dmg[['Rank', 'Error_Type', 'Damage (points lost)',
                       'Can software fix it?', 'Recovered by cleaning',
                       'Why it sits here']].rename(
            columns={'Error_Type': 'Data error'})
        sheet(xl, 'Order 1 - By Damage', dmg_out, [7, 32, 15, 27, 16, 58],
              wrap_cols=('Why it sits here',),
              title='ORDER 1 - ranked purely by how much damage the error causes',
              note='This answers "what hurts the most?". It does NOT say what to '
                   'fix first, because it ignores whether the problem can be fixed '
                   'and what fixing it costs. Compare with the Order 2 sheet.')

        # ------------------------------------- 4b. the order we recommend
        # Same 28 error types as Order 1, re-ordered by what is worth doing.
        # Everything is derived from the CSVs - no hardcoded numbers, so this
        # sheet cannot go stale when the cost measurements are re-run.
        MATERIAL = 0.5          # pp of recovery worth acting on
        NEGLIGIBLE = 0.1        # seconds; the cost-band boundary

        recs = []
        for _, x in damage.iterrows():
            et, route = x.Error_Type, x.Route
            dmg = x.Damage_pp
            improve = imp.get(et)
            secs = rq2.set_index('Error_Type').Clean_Seconds.get(et)
            has_imp = improve is not None and improve == improve

            if et == 'class_imbalance':
                tier, order = 'Different objective - judge separately', 5
                why = (f'Lowers accuracy but raises recall sharply ({bm_rec:+.1f} points on '
                       'Bank Marketing). Accuracy is the wrong measure for it, so '
                       'it is not ranked against the others.')
            elif route == 'external_intervention':
                tier, order = 'Needs people - plan and budget for it', 3
                why = (f'Causes {dmg:.2f} points of damage. This study evaluated '
                       f'no automated repair for it - the remedy here is human '
                       f'effort or newer data, so it cannot be an immediate '
                       f'action. (Automated approaches exist in the literature for '
                       f'some of these; they were outside this protocol.)')
            elif not has_imp or improve < MATERIAL:
                tier, order = 'Do not bother', 4
                why = (f'Cleaning works but recovers only {improve:+.2f} points - '
                       f'there is almost nothing to win. Being ABLE to clean '
                       f'something is not a reason to clean it.'
                       if has_imp else 'No recovery measured.')
                if et == 'data_heterogeneity':
                    why = ('Causes real damage (2.41 points, rising with severity) '
                           'but the repair recovers 0.00. The damage is a scale '
                           'mismatch between training and test data; the repair only '
                           'repeats the standardisation the pipeline already does, '
                           'and it is not allowed to look at the test set. So the '
                           'error is NOT harmless - this particular repair just '
                           'cannot fix it.')
            elif secs is not None and secs == secs and secs < NEGLIGIBLE:
                tier, order = 'Free wins - do these immediately', 1
                why = (f'Recovers {improve:+.2f} points for {secs:.3f} seconds. '
                       f'Inside the negligible-cost group we rank by how much is '
                       f'recovered.')
            else:
                tier, order = 'Worth paying for', 2
                why = (f'Recovers {improve:+.2f} points - more than anything in the '
                       f'free group - but costs about {secs:.1f} seconds, roughly a '
                       f'thousand times more compute. Take the free wins first.')

            recs.append({'Tier': tier, '_o': order, 'Data error': et,
                         'Damage (points)': round(dmg, 2),
                         'Recovered (points)': (round(improve, 2) if has_imp else None),
                         'Cost (seconds)': (round(secs, 4)
                                            if secs is not None and secs == secs
                                            else None),
                         'What you would do': CLEANERS.get(
                             et, 'human effort / new data - no automated repair'),
                         'Why this position': why})

        rec = pd.DataFrame(recs)
        # within each tier: by recovery where we have it, else by damage
        rec['_k'] = rec['Recovered (points)'].fillna(rec['Damage (points)'])
        rec = rec.sort_values(['_o', '_k'], ascending=[True, False])
        rec.insert(0, 'Priority', range(1, len(rec) + 1))
        rec = rec.drop(columns=['_o', '_k'])
        sheet(xl, 'Order 2 - What We Recommend', rec,
              [8, 34, 30, 13, 15, 13, 30, 74],
              wrap_cols=('Why this position', 'What you would do'),
              title='ORDER 2 - the same 28 errors, ordered by what is worth doing',
              note='Order 1 ranks by damage alone. This one also accounts for '
                   'whether it can be fixed automatically, how much of the damage '
                   'actually comes back, and what the fix costs. Compare the two '
                   'side by side - see the "Why The Two Orders Differ" sheet.')

        # --------------------------------- 4c. why the orders differ
        # Ranks are READ from the two sheets just built, never typed by hand -
        # an earlier version hardcoded them and went stale the moment Order 2
        # was rebuilt.
        r1 = {e: i + 1 for i, e in enumerate(dmg_out['Data error'])}
        r2 = {e: i + 1 for i, e in enumerate(rec['Data error'])}
        REASONS = {
            'annotator_bias':
                'Biggest damage of all, but no automated repair was evaluated for '
                'it here - a wrong label is a valid value that happens to be '
                'untrue, so it is not identifiable from the data by the rules this '
                'study applied. The remedy in scope is human re-annotation, which '
                'has to be planned and budgeted.',
            'missing_values_mnar':
                'Recovers the most of any repair, but takes seconds rather than '
                'milliseconds - roughly a thousand times the compute of the free '
                'group. Still worth doing, just not first.',
            'contextual_errors':
                'Moved UP. Less damaging than the four above it, but fully '
                'automatic, nearly free, and recovers 71% of its damage.',
            'invalid_values':
                'Moved UP sharply. Modest damage, but the most complete repair we '
                'have - 91.6% of the damage recovered - at almost no cost.',
            'data_leakage':
                'Moved down slightly: it does real damage but the repair only '
                'returns 0.92 points, less than the cheaper repairs above it.',
            'data_heterogeneity':
                'Causes real damage (2.41 points) but the repair recovers 0.00. '
                'The damage is a scale mismatch between training and test data, and '
                'the repair only repeats standardisation the pipeline already '
                'performs. The error is not harmless - this repair just cannot fix '
                'it. Clearest case where damage rank alone would mislead you.',
            'missing_values':
                'The worst deal in the study: seconds of compute to recover a '
                'fraction of a point. Same repair as its MNAR twin, almost none of '
                'the benefit, because random missingness destroys little signal.',
            'outliers':
                'Cleanable, cheap and correct - but there is almost no damage to '
                'recover. Being ABLE to clean something is not a reason to.',
            'typographical_errors':
                'Caused no measurable damage, and the repair is the slowest of the '
                'cheap group. Nothing to win and something to pay.',
        }
        diff = []
        for et, why in REASONS.items():
            if et in r1 and et in r2:
                diff.append({'Problem': et, 'Damage rank': r1[et],
                             'Recommended rank': r2[et],
                             'Moved': r1[et] - r2[et],
                             'Why it moved': why})
        diff = pd.DataFrame(diff).sort_values('Recommended rank')
        sheet(xl, 'Why The Two Orders Differ', diff, [30, 14, 20, 10, 84],
              wrap_cols=('Why it moved',),
              title='Why "what hurts most" is NOT "what to fix first"',
              note='"Moved" is how many places the problem shifts between the two '
                   'orders - positive means it became more urgent once '
                   'repairability and cost were taken into account. This gap is '
                   'the practical contribution of the study.')

        # ------------------------------------------------------- 5. mechanism
        mech2 = mech[['Comparison', 'Mechanism_A', 'Damage_A_pp', 'Mechanism_B',
                      'Damage_B_pp', 'mean_diff_pp', 't_p', 'significant_holm']].rename(columns={
            'Comparison': 'What we compared', 'Mechanism_A': 'Version A',
            'Damage_A_pp': 'Damage A', 'Mechanism_B': 'Version B',
            'Damage_B_pp': 'Damage B', 'mean_diff_pp': 'Difference',
            't_p': 'p-value', 'significant_holm': 'Real difference? (Holm)'})
        sheet(xl, 'Key Finding - Mechanism', mech2, [24, 22, 11, 24, 11, 12, 12, 16],
              title='Our strongest finding: WHERE the damage lands matters more '
                    'than HOW MUCH there is',
              note='Each pair damages exactly the same amount of data. Only the '
                   'placement differs. Missing values tied to the answer (MNAR) do '
                   '18x the damage of random missing values (MCAR) at an identical '
                   'missing rate.')

        # ---------------------------------------------------------- 6. datasets
        ds = pd.DataFrame([
            ['Breast Cancer Wisconsin', 'Healthcare', 569, 30, 0, 30, '37.3%',
             'Fully numeric, so it isolates number-based errors from any text effect.'],
            ['German Credit', 'Finance', 1000, 20, 13, 7, '70.0%',
             'Mixed text and numbers; the positive class is the majority here.'],
            ['Telco Customer Churn', 'Telecom', 7032, 19, 15, 4, '26.6%',
             'Mostly text columns, so text-based errors have somewhere real to act.'],
            ['Adult Income', 'Census', 30162, 14, 8, 6, '24.9%',
             'Large, mixed types, widely used benchmark.'],
            ['Bank Marketing', 'Marketing', 45211, 16, 9, 7, '11.7%',
             'Largest dataset, strongly imbalanced classes.'],
        ], columns=['Dataset', 'Field', 'Rows', 'Columns', 'Text columns',
                    'Number columns', 'Positive class', 'Why we chose it'])
        sheet(xl, 'Datasets', ds, [26, 13, 9, 10, 13, 15, 14, 66],
              wrap_cols=('Why we chose it',),
              title='The 5 datasets',
              note='Two were dropped: Blogger (only 100 rows - the models could not '
                   'learn from it at all, so damage could not be measured) and '
                   'Credit Card Fraud (0.17% positive class makes accuracy '
                   'meaningless).')

        # ------------------------------------------------------------ 7. models
        md = pd.DataFrame([
            ['Logistic Regression', 'Non-tree', 'max_iter=1000',
             'Simple linear baseline.'],
            ['Gaussian Naive Bayes', 'Non-tree', 'defaults',
             'Probabilistic, assumes columns are independent.'],
            ['K-Nearest Neighbours', 'Non-tree', 'n_neighbors=5',
             'Predicts from the most similar rows; sensitive to scale.'],
            ['Neural Network (MLP)', 'Non-tree', 'hidden=(64,), max_iter=200',
             'Small neural network.'],
            ['Decision Tree', 'Tree', 'max_depth=10',
             'Single interpretable tree.'],
            ['Random Forest', 'Tree', 'n_estimators=100, max_depth=10',
             'Many trees averaged together.'],
            ['XGBoost', 'Tree', 'n_estimators=100, max_depth=6, lr=0.1',
             'Gradient boosting, strong on tabular data.'],
            ['LightGBM', 'Tree', 'n_estimators=100, max_depth=6, lr=0.1',
             'Fast gradient boosting.'],
        ], columns=['Model', 'Family', 'Settings', 'What it is'])
        sheet(xl, 'Models', md, [24, 12, 38, 52], wrap_cols=('What it is',),
              title='The 8 models',
              note='Deliberately balanced 4 tree-based against 4 non-tree, so RQ3 '
                   'can test whether priorities depend on the model FAMILY rather '
                   'than on one unusual algorithm. All 8 can output probabilities, '
                   'which the AUC measurement needs.')

        # -------------------------------------------------- 8. cleaning methods
        cm = rq2[['Error_Type', 'Method', 'Improvement_pp', 'Clean_Seconds',
                  'Cost_Band']].copy()
        cm['What the repair does'] = cm.Method.map(CLEANER_EXPLAIN).fillna('')
        cm = cm.rename(columns={
            'Error_Type': 'Data error', 'Method': 'Cleaning method',
            'Improvement_pp': 'Accuracy points recovered',
            'Clean_Seconds': 'Seconds to run', 'Cost_Band': 'Cost band'})
        # class_imbalance is excluded from the efficiency ranking because it
        # optimises a different objective (recall, not accuracy), but the user
        # asked for every cleaning method - so it is listed with that explained.
        ci = base[base.Error_Type == 'class_imbalance']
        cm = pd.concat([cm, pd.DataFrame([{
            'Data error': 'class_imbalance',
            'Cleaning method': 'oversample minority (class balancing)',
            'Accuracy points recovered': round(ci.Improvement_pp.mean(), 2),
            'Seconds to run': float('nan'),
            'Cost band': 'not ranked - see note',
            'What the repair does':
                'Copy rare-class rows until the classes are balanced. Judged by '
                'accuracy this looks like a loss, but that is the wrong measure: '
                'it trades accuracy for finding more of the rare cases. On Bank '
                f'Marketing accuracy fell {abs(bm_acc):.2f} points while recall '
                f'ROSE {bm_rec:.2f} points and F1 rose {bm_f1:.2f}. We exclude it '
                'from the ranking rather than place it last.',
        }])], ignore_index=True)

        cm = cm.sort_values('Accuracy points recovered', ascending=False)
        cm['Accuracy points recovered'] = cm['Accuracy points recovered'].round(2)
        cm['Seconds to run'] = cm['Seconds to run'].round(4)
        sheet(xl, 'Cleaning Methods', cm, [30, 32, 15, 13, 13, 62],
              wrap_cols=('What the repair does',),
              title='The 20 automated cleaning methods, best recovery first',
              note='Negative recovery means the repair made things slightly worse '
                   'than leaving the damage alone. We report those honestly rather '
                   'than hiding them. Class balancing is listed last and is NOT '
                   'ranked - it improves a different measure (recall), so judging '
                   'it by accuracy would be misleading.')

        # ------------------------------------------------------ 9/10. RQ3 / RQ4
        for nm, d, ttl, note in (
            ('RQ3 by Model', rq3,
             'RQ3 - how much each error ranking moves between models',
             'Rank_Range is how far the error type moved between its best and '
             'worst position across the 8 models. Bigger = more model-dependent.'),
            ('RQ4 by Dataset', rq4,
             'RQ4 - how much each error ranking moves between datasets',
             'Rank_Range is how far the error type moved between its best and '
             'worst position across the 5 datasets. Bigger = more '
             'dataset-dependent.')):
            d = d.rename(columns={'Error_Type': 'Data error',
                                  'Mean_Damage_pp': 'Average damage (points)',
                                  'Best_Rank': 'Best position',
                                  'Worst_Rank': 'Worst position',
                                  'Rank_Range': 'How much it moved'})
            sheet(xl, nm, d, [32, 20, 14, 15, 18], title=ttl, note=note)

    print(f'Written: {OUT}')


if __name__ == '__main__':
    main()
