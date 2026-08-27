# Data

Raw dataset files are **not committed** to this repository — they are
third-party data, not ours to redistribute as a primary copy. Run
`python fetch_data.py` from this directory to download them; each download
is verified against `checksums.txt` and the script exits with an error
rather than proceeding on a mismatch.

`src/data_loader.py` expects exactly these six files, unmodified, in this
directory:

| File | Dataset | Source | Rows used |
|---|---|---|---|
| `wdbc.data`, `wdbc.names` | Breast Cancer Wisconsin (Diagnostic) | [UCI ML Repository, dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 |
| `german.data` | Statlog (German Credit Data) | [UCI ML Repository, dataset 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1,000 |
| `adult.data` | Adult (Census Income) | [UCI ML Repository, dataset 2](https://archive.ics.uci.edu/dataset/2/adult) | 32,561 (rows with `?` dropped before injection) |
| `bank-full.csv` | Bank Marketing | [UCI ML Repository, dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 45,211 |
| `dataset.arff` | Telco Customer Churn (IBM sample) | [OpenML, dataset 42178](https://www.openml.org/d/42178) | 7,043 (rows with blank `TotalCharges` dropped) |

All five UCI datasets are CC BY 4.0 licensed. Give appropriate credit if you
redistribute them.

## Load-time handling

`DataLoader` (`src/data_loader.py`) reads these files as-is and performs no
corruption or cleaning — only the minimal parsing needed to get a usable
feature matrix, done once, identically, before any error is injected:

- **Adult Income**: rows containing `?` are dropped, so the study injects
  missingness deliberately rather than inheriting it from the source file.
- **Telco Churn**: `customerID` is dropped (a unique key a model could
  memorise); `TotalCharges` is coerced to numeric and its blank rows
  (customers with zero tenure) are dropped, for the same reason as Adult
  Income.
- **Breast Cancer Wisconsin**: the ID and diagnosis columns are separated
  from the feature matrix; diagnosis `M`/`B` becomes the binary target.
- **German Credit**: whitespace-separated, no header; the final column
  becomes the binary target (1 = good credit).
- **Bank Marketing**: semicolon-separated; the `y` column becomes the binary
  target.

Categorical columns are kept as text (not integer-encoded) at this stage —
encoding happens later in the pipeline, after error injection, which is the
central methodological point of this study (see `docs/RESEARCH_LOG.md`,
"Pipeline change").

## Reconstructing `results/datasets_raw.pkl`

This repository does not ship `datasets_raw.pkl` — it is a pickle cache of
the loaded, split datasets, fully regenerable by running:

```bash
python src/data_loader.py
```

from the repository root, once the six files above are present in `data/`.
