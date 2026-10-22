# Data

| Path | Contents | In git? |
|---|---|---|
| `raw/dataset_full.csv` | Full dataset, 88,647 × 112 (25 MB). Get it with `make data` | No |
| `sample/dataset_sample.csv` | 5,000-row balanced random sample (tests, CI, `make train-sample`) | Yes |
| `allowlist.txt` | Registered domains trusted by the scanner's reputation layer (illustrative) | Yes |

**Source:** Vrbančič G., Fister I. Jr., Podgorelec V. (2020). *Datasets for phishing websites detection.* Data in Brief 33:106438.
https://github.com/GregaVrbancic/Phishing-Dataset (CC BY 4.0). Feature definitions are in that repository's README.

**Encoding notes**
- `-1` = component absent (no path → all `*_directory` and `*_file` features are −1; no query → all `*_params` features are −1)
  or a failed network look-up. It is **not** NaN.
- URLs were featurised without the scheme (`http://`), and the `?` before the query is not counted.
- Label: `phishing` = 1 (30,647), 0 (58,000).
