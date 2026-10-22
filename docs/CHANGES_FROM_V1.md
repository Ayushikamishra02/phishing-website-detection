# What changed from v1 (capstone notebook) and why

v1: one notebook that loaded `dataset_full.csv`, chose features, trained a default random forest and printed
accuracy 0.709 / precision 0.642 / recall 0.355, then pickled the model.

## Bugs fixed
| v1 code | Problem | Effect | v2 |
|---|---|---|---|
| `correlation_matrix[correlation_matrix['url_shortened'].abs() > 0.1]` | Selected features correlated with **`url_shortened`** instead of the target `phishing` | Only 3 weak features kept: `qty_vowels_domain`, `domain_length`, `url_shortened` | All features; selection by cross-validated model comparison |
| `data.drop_duplicates()` (result not saved back to `data`) | Did nothing | 1,438 duplicates stayed in and could land in both train and test | Duplicates and 12 contradictory rows removed, logged |
| `data = data.dropna()` | Missing values are encoded as `-1`, not NaN | Nothing removed; `-1` semantics ignored | `-1` documented as "absent / lookup failed"; trees use it natively, LR gets NaN + indicator |
| Accuracy / precision / recall at 0.5 only | Accuracy is misleading at 35% prevalence; no false-positive control | Recall 35% looked acceptable next to 71% accuracy | ROC-AUC, PR-AUC, **TPR at 1% FPR**, thresholds chosen for an FPR budget |
| `GridSearchCV`, `StandardScaler`, `f1_score` imported, unused | Dead code | — | Removed |
| "Model deployment" = `pickle.dump` | No way to use the model on a new website | — | URL feature extractor + FastAPI + Streamlit + Docker |

## New analysis
- **Dataset shortcut discovered** (homepages vs deep links) and quantified: a one-line rule scores 86.4%.
- Stratified evaluation; per-URL-type thresholds; deep-link-only control model.
- URL-only vs domain-intelligence vs full feature sets × 4 model families, 5-fold CV.
- SHAP explanations, which exposed `qty_dollar_directory` as a hidden path flag.
- Dead features found (`url_google_index`, `domain_google_index`).

## Engineering
- Package `src/phishguard`, `python -m phishguard.train` reproduces every number and figure.
- 37 tests including **property tests** that keep the URL extractor consistent with the training data.
- CI on Python 3.11/3.12, Docker image, model card.

## Result
Same test set, v1 method vs v2 full model: accuracy **70.0% → 97.5%**, recall **35% → 96%**, ROC-AUC **0.71 → 0.996**.
