"""Generate the SHA-256 manifest for the frozen analysis version.

Run from the project root:  python frozen/make_manifest.py

Any later supervisor-requested change can then be distinguished from the
submitted results by re-running this and diffing frozen/CHECKSUMS.txt.
"""
import datetime
import glob
import hashlib
import io
import os

TARGETS = (['FROZEN.md', 'METHODOLOGY.md', 'RESEARCH_LOG.md', 'RESULTS_SUMMARY.md',
            'RECOMMENDATIONS.md', 'verify_documents.py', 'make_excel_summary.py',
            'run_experiments.py', 'build_roi_model.py', 'analyse_results.py',
            'measure_cleaning_cost.py', 'audit_injectors.py',
            'audit_pipeline_overlap.py', 'audit_cost_stability.py',
            'preflight.py']
           + sorted(glob.glob('src/*.py'))
           + sorted(glob.glob('results/*.csv'))
           + sorted(glob.glob('frozen/*.py'))
           + sorted(glob.glob('frozen/*.txt'))
           + sorted(glob.glob('frozen/*.csv')))

# The manifest must not hash itself: writing it changes it, so a self-entry
# records the PREVIOUS version's digest and the manifest can never reproduce.
TARGETS = [f for f in TARGETS if os.path.basename(f) != 'CHECKSUMS.txt']

rows = []
for f in TARGETS:
    if os.path.isfile(f):
        digest = hashlib.sha256(open(f, 'rb').read()).hexdigest()
        rows.append((f.replace(os.sep, '/'), digest, os.path.getsize(f)))

width = max(len(r[0]) for r in rows)
body = '\n'.join(f'{f:<{width}}  {h}  {s:>9,}' for f, h, s in rows)

io.open('frozen/CHECKSUMS.txt', 'w', encoding='utf-8').write(
    'SHA-256 manifest - frozen analysis version\n'
    f'generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n'
    f'{len(rows)} files\n\n{body}\n')

print(f'manifest written: {len(rows)} files')
