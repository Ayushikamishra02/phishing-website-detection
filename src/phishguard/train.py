"""Run every experiment and save the deployable models.

    python -m phishguard.train            # full dataset (~10 min on a laptop)
    python -m phishguard.train --sample   # 5k-row sample, ~1 min (CI / smoke test)
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import __version__, plots
from .data import FEATURE_SETS, ROOT, TARGET, clean, has_path, load_raw
from .evaluate import metrics, threshold_for_fpr
from .explain import shap_values
from .modeling import MODELS, make_pipeline, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")

MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MAX_FPR = 0.01          # operating point: flag at most 1% of legitimate sites


def v1_method(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Faithful re-run of the original notebook's approach on our split:
    features chosen by |corr| > 0.1 with `url_shortened` (not the target!), default random forest."""
    X = train.drop(columns=TARGET)
    corr = X.corr()["url_shortened"].abs()
    feats = corr[corr > 0.1].index.tolist()
    rf = RandomForestClassifier(random_state=42, n_jobs=-1).fit(train[feats], train[TARGET])
    m = metrics(test[TARGET], rf.predict_proba(test[feats])[:, 1], 0.5)
    m["features_used"] = feats
    return m


def cross_validate_all(train: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    rows = []
    for fs, feats in FEATURE_SETS.items():
        for name in MODELS:
            s = cross_validate(make_pipeline(name, feats, seed), train[feats], train[TARGET], cv=cv,
                               scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"}, n_jobs=1)
            rows.append({"feature_set": fs, "model": name,
                         "roc_auc": s["test_roc_auc"].mean(), "roc_auc_std": s["test_roc_auc"].std(),
                         "pr_auc": s["test_pr_auc"].mean(), "pr_auc_std": s["test_pr_auc"].std(),
                         "fit_seconds": s["fit_time"].mean()})
            log.info("CV %-8s %-20s ROC-AUC %.4f  PR-AUC %.4f", fs, name, rows[-1]["roc_auc"], rows[-1]["pr_auc"])
    return pd.DataFrame(rows)


def stratified_eval(y, s, mask_path, threshold) -> list[dict]:
    out = []
    for subset, m in [("all URLs", np.ones_like(mask_path, bool)), ("homepages", ~mask_path),
                      ("deep links", mask_path)]:
        r = metrics(y[m], s[m], threshold)
        r["subset"] = subset
        r["phishing_share"] = float(y[m].mean())
        out.append(r)
    return out


def _ranking(y, s) -> dict:
    m = metrics(y, s)
    return {k: m[k] for k in ("roc_auc", "pr_auc", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr")}


def run(sample: bool = False, seed: int = 42) -> dict:
    df, cleaning = clean(load_raw(sample=sample))
    train, valid, test = split(df, seed)
    log.info("train %d | valid %d | test %d", len(train), len(valid), len(test))
    y_test, path_test = test[TARGET].to_numpy(), has_path(test).to_numpy()

    # 1. Baselines and the shortcut ------------------------------------------------
    path_share = df.groupby(TARGET)["directory_length"].apply(lambda s: float((s != -1).mean())).to_dict()
    majority = float((y_test == 0).mean())
    rule_acc = float(((path_test).astype(int) == y_test).mean())
    v1 = v1_method(train, test)
    log.info("Majority %.4f | has-path rule %.4f | v1 method %.4f", majority, rule_acc, v1["accuracy"])

    # 2. Model x feature-set comparison --------------------------------------------
    cv = cross_validate_all(train, 3 if sample else 5, seed)
    best = cv.loc[cv.groupby("feature_set")["pr_auc"].idxmax()].set_index("feature_set")["model"].to_dict()
    log.info("Best per feature set: %s", best)

    # 3. Final models: fit on train, threshold on valid, evaluate on test ----------
    results, scores, artifacts = {}, {}, {}
    path_valid = has_path(valid).to_numpy()
    for fs in ["lexical", "full"]:
        feats, name = FEATURE_SETS[fs], best[fs]
        pipe = make_pipeline(name, feats, seed).fit(train[feats], train[TARGET])
        sv = pipe.predict_proba(valid[feats])[:, 1]
        yv = valid[TARGET].to_numpy()
        # (a) one global threshold, (b) one threshold per URL type, each at <=1% FPR on validation
        thr = threshold_for_fpr(yv, sv, MAX_FPR)
        seg_thr = {"homepage": threshold_for_fpr(yv[~path_valid], sv[~path_valid], MAX_FPR),
                   "deep_link": threshold_for_fpr(yv[path_valid], sv[path_valid], MAX_FPR)}
        s = pipe.predict_proba(test[feats])[:, 1]
        thr_vec = np.where(path_test, seg_thr["deep_link"], seg_thr["homepage"])
        scores[fs] = s
        results[fs] = {
            "model": name, "threshold": thr, "segment_thresholds": seg_thr,
            "test": metrics(y_test, s, thr),
            # shifting scores by the per-row threshold lets us reuse metrics(); ranking metrics are
            # taken from the unshifted scores because they are threshold-free
            "test_segment_thresholds": metrics(y_test, s - thr_vec + 0.5, 0.5) | _ranking(y_test, s),
            "test_at_0.5": metrics(y_test, s, 0.5),
            "stratified": stratified_eval(y_test, s, path_test, thr),
            "stratified_segment_thresholds": [
                r | _ranking(y_test[m], s[m]) for r, m in zip(
                    stratified_eval(y_test, s - thr_vec + 0.5, path_test, 0.5),
                    [np.ones_like(path_test, bool), ~path_test, path_test])],
        }
        artifacts[fs] = pipe
        t, ts = results[fs]["test"], results[fs]["test_segment_thresholds"]
        log.info("%s/%s global thr=%.3f recall %.4f FPR %.4f | segment thr=%s recall %.4f FPR %.4f", fs, name,
                 thr, t["recall"], t["fpr"], {k: round(v, 3) for k, v in seg_thr.items()}, ts["recall"], ts["fpr"])

    # deep-link-only training: does the full model still work without the shortcut?
    tp, tep = train[has_path(train)], test[has_path(test)]
    feats = FEATURE_SETS["full"]
    p_dl = make_pipeline(best["full"], feats, seed).fit(tp[feats], tp[TARGET])
    deep_only = metrics(tep[TARGET], p_dl.predict_proba(tep[feats])[:, 1])

    # 4. Explanations ----------------------------------------------------------------
    shap_sample = test.sample(min(2000, len(test)), random_state=seed)
    importance = {}
    for fs in ["lexical", "full"]:
        feats = FEATURE_SETS[fs]
        vals = shap_values(artifacts[fs], shap_sample[feats])
        importance[fs] = (pd.Series(np.abs(vals).mean(0), index=feats).sort_values(ascending=False)
                          .head(15).round(4).to_dict())
        plots.shap_beeswarm(vals, shap_sample[feats],
                            f"What drives the {fs} model (SHAP, test sample)", f"shap_{fs}", FIG)

    # 5. Figures ---------------------------------------------------------------------
    full_v2 = results["full"]["test_at_0.5"]
    plots.shortcut(path_share, {"Always 'legitimate'": majority, "Rule: has path": rule_acc,
                                "v1 notebook method": v1["accuracy"], "v2 full model": full_v2["accuracy"]}, FIG)
    plots.model_grid(cv, "pr_auc", FIG)
    plots.roc_low_fpr({
        "full model, all URLs": (y_test, scores["full"], plots.GREEN, "-"),
        "full model, deep links": (y_test[path_test], scores["full"][path_test], plots.GREEN, "--"),
        "lexical model, all URLs": (y_test, scores["lexical"], plots.AMBER, "-"),
        "lexical model, deep links": (y_test[path_test], scores["lexical"][path_test], plots.AMBER, "--"),
    }, FIG)
    strat = pd.DataFrame(
        [{**r, "model": fs, "thresholds": "global"} for fs in results for r in results[fs]["stratified"]]
        + [{**r, "model": fs, "thresholds": "per URL type"} for fs in results
           for r in results[fs]["stratified_segment_thresholds"]])
    plots.stratified(strat, FIG)
    plots.v1_vs_v2(v1, full_v2, FIG)

    # 6. Persist ----------------------------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    for fs, pipe in artifacts.items():
        joblib.dump(pipe, MODELS_DIR / f"{fs}_model.joblib", compress=3)
    meta = {
        fs: {"model": results[fs]["model"], "threshold": results[fs]["threshold"],
             "segment_thresholds": results[fs]["segment_thresholds"],
             "features": FEATURE_SETS[fs], "max_fpr_target": MAX_FPR}
        for fs in results
    }
    meta["_info"] = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "trained_on_sample": sample, "n_train": len(train), "package": __version__,
                     "python": platform.python_version(), "sklearn": sklearn.__version__}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    report = {
        "cleaning": cleaning.steps,
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "path_share_by_class": {"legitimate": path_share[0], "phishing": path_share[1]},
        "baselines": {"majority_accuracy": majority, "has_path_rule_accuracy": rule_acc, "v1_method": v1},
        "cv": cv.round(5).to_dict(orient="records"),
        "best_model_per_feature_set": best,
        "final": results,
        "deep_link_trained_full_model": deep_only,
        "shap_importance": importance,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    strat.to_csv(REPORTS / "stratified_results.csv", index=False)
    cv.to_csv(REPORTS / "cv_results.csv", index=False)
    log.info("Done. Models -> %s, report -> %s", MODELS_DIR, REPORTS / "metrics.json")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="use the bundled 5k-row sample")
    run(sample=ap.parse_args().sample)


if __name__ == "__main__":
    main()
