import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from phishguard.data import (
    DEAD_FEATURES,
    FEATURE_SETS,
    LEXICAL_FEATURES,
    NETWORK_FEATURES,
    TARGET,
    clean,
    has_path,
    load_raw,
)
from phishguard.evaluate import metrics, threshold_for_fpr, tpr_at_fpr
from phishguard.modeling import make_pipeline, split


@pytest.fixture(scope="module")
def data():
    df, report = clean(load_raw(sample=True))
    return df, report


def test_feature_groups_partition_columns(data):
    df, _ = data
    assert not set(LEXICAL_FEATURES) & set(NETWORK_FEATURES)
    assert set(LEXICAL_FEATURES) | set(NETWORK_FEATURES) | {TARGET} == set(df.columns)
    assert not set(DEAD_FEATURES) & set(df.columns)


def test_clean_removes_duplicates(data):
    df, report = data
    assert not df.duplicated().any()
    assert any("duplicate" in s for s in report.steps)


def test_split_is_disjoint_and_stratified(data):
    df, _ = data
    tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    for part in (va, te):
        assert abs(part[TARGET].mean() - tr[TARGET].mean()) < 0.02
        assert abs(has_path(part).mean() - has_path(tr).mean()) < 0.02


@pytest.mark.parametrize("model", ["logistic_regression", "lightgbm"])
@pytest.mark.parametrize("fs", ["lexical", "full"])
def test_models_learn(data, model, fs):
    df, _ = data
    tr, _, te = split(df)
    feats = FEATURE_SETS[fs]
    p = make_pipeline(model, feats).fit(tr[feats], tr[TARGET]).predict_proba(te[feats])[:, 1]
    assert roc_auc_score(te[TARGET], p) > 0.9


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    s = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)
    for budget in (0.001, 0.01, 0.05):
        t = threshold_for_fpr(y, s, budget)
        assert (s[y == 0] >= t).mean() <= budget
    assert 0 <= tpr_at_fpr(y, s, 0.01) <= 1


def test_metrics_keys():
    m = metrics(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.7]), 0.5)
    assert m["accuracy"] == 1.0 and m["fpr"] == 0.0 and m["n"] == 4
