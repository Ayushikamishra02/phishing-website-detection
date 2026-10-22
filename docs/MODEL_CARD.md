# Model Card: PhishGuard

## Models
| Name | Algorithm | Inputs | When to use |
|---|---|---|---|
| `lexical` | XGBoost (600 trees) | 98 features computed from the URL string only | Pre-click screening (e-mail gateway, browser extension); first-stage triage |
| `full` | LightGBM (800 trees) | 98 lexical + 11 domain-intelligence features (domain age, DNS, TLS, redirects) | Second-stage decision when WHOIS/DNS/TLS look-ups are available |

Both were selected by 5-fold cross-validated PR-AUC among logistic regression, random forest, XGBoost and LightGBM.
Each ships with **two thresholds**, one for homepages (URL without a path) and one for deep links, each chosen so that
at most 1% of legitimate validation sites of that type are flagged.

## Intended use
Demonstrating and evaluating phishing-detection techniques; decision *support* for analysts. **Not** a production
security control: do not use it as the only reason to block a site or quarantine an e-mail.

## Training data
Vrbančič et al. (2020), full variant, CC BY 4.0. 88,647 → 87,197 rows after removing 1,438 exact duplicates and
12 rows with contradictory labels; 2 dead features (Google-index look-ups, ~100% zero) dropped. Split 60/20/20,
stratified by label × URL type.

## Performance (test set, n = 17,440)
| | ROC-AUC | PR-AUC | Caught @ 1% FPR (all) | Deep links: caught / FPR (per-type threshold) | Homepages: caught / FPR |
|---|---|---|---|---|---|
| lexical | 0.982 | 0.972 | 78% | 60% / 1.5% | 17% / 1.0% |
| full | 0.996 | 0.993 | 93% | 79% / 0.8% | 72% / 0.8% |

## Known issues and risks
1. **Shortcut / collection bias.** 80% of legitimate URLs are homepages vs 2% of phishing URLs. Path presence alone gives
   86% accuracy. Mitigations: stratified evaluation, per-URL-type thresholds, a deep-link-only control model
   (same deep-link AUC), and SHAP checks. Residual risk: real-world legitimate deep links (documentation, product
   pages) can still score high with the URL-only model, so the scanner adds a reputation allowlist and warns on deep links.
2. **Reverse-engineered feature extractor.** The dataset has no raw URLs and no extraction code. Conventions were inferred
   from identities that hold for >95% of rows and are enforced by property tests. `qty_tld_url` and
   `tld_present_params` depend on an unknown TLD list and are approximate.
3. **Temporal drift.** Data was collected around 2020; phishing techniques and hosting patterns change. Retrain and
   re-evaluate on fresh feeds before any use.
4. **No domain-level grouping.** Several URLs from one campaign might fall in both train and test, which would make
   results optimistic.
5. **Allowlist risk.** Compromised trusted domains (e.g. phishing hosted on a legitimate file-sharing site) would be passed
   by the allowlist. Real systems combine reputation with content analysis.

## Ethical considerations
False positives block legitimate businesses; false negatives expose users. Thresholds should be set with the
deploying organisation's cost of each error, and decisions should be reviewable. SHAP explanations are returned
with every verdict for that reason.
