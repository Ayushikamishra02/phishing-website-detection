# Publishing guide (for you, not part of the analysis)

## Push
Either replace your old `CAPSTONE-PROJECT--Ayushika-mishra` repo (keeps its history; then rename it in Settings) or create a new one:
```bash
cd phishing-website-detection
git init && git add . && git commit -m "PhishGuard v2: phishing detection with shortcut analysis, API and dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/phishing-website-detection.git
git push -u origin main
```
Then replace `YOUR-USERNAME` / `YOUR-PROFILE` in README.md and un-comment the CI badge.

**About → description:** *Phishing website detection with LightGBM/XGBoost, evaluated at a 1% false-positive budget,
plus a dataset-shortcut analysis, SHAP explanations, FastAPI and Streamlit.*
**Topics:** `machine-learning` `cybersecurity` `phishing-detection` `lightgbm` `xgboost` `shap` `explainable-ai` `fastapi` `streamlit` `data-science`

Deploy the dashboard free on https://share.streamlit.io (main file `app/streamlit_app.py`; the trained models are committed).

## Resume bullets
- Rebuilt a phishing-website classifier (87k sites, 109 features) from 70% to **97.5% accuracy / 0.996 ROC-AUC**, catching
  **93% of phishing at a 1% false-positive rate**; fixed a feature-selection bug that used the wrong target column.
- Discovered a **dataset shortcut** (80% of legitimate URLs are homepages vs 2% of phishing), showed a one-line rule reaches
  86%, and redesigned evaluation and **per-URL-type thresholds** so deep links stay within the 1% FPR budget.
- Reverse-engineered the dataset's feature definitions into a **raw-URL feature extractor** validated with property tests,
  and shipped it behind a **FastAPI** endpoint and **Streamlit** scanner with **SHAP** explanations and a reputation allowlist.

## Interview talking points
- **Why not just report 97.5%?** Because 86% comes from a collection artifact. Show `shortcut.png` and the stratified table.
- **Why TPR at 1% FPR?** Security tools live or die on false alarms; accuracy hides that trade-off.
- **How did SHAP help?** "$ count in path" ranked #2. It's a −1/0 path flag, which confirmed the shortcut.
- **How would you productionise?** Fresh labelled feeds (PhishTank/OpenPhish + Tranco deep links), time-based validation,
  live WHOIS/DNS collection, monitoring for drift, human review for blocks.
