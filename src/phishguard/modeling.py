"""Model zoo, preprocessing and data splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from .data import TARGET, has_path

MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def _minus_one_to_nan(X):
    X = np.asarray(X, dtype=float).copy()
    X[X == -1] = np.nan
    return X


def make_pipeline(name: str, features: list[str], seed: int = 42, **params) -> Pipeline:
    """Trees consume -1 natively (it is a meaningful 'absent' value).
    Logistic regression gets -1 -> NaN -> median + missing-indicator, then log1p + scaling."""
    if name == "logistic_regression":
        prep = ColumnTransformer([("num", Pipeline([
            ("nan", FunctionTransformer(_minus_one_to_nan)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
        ]), features)])
        model = LogisticRegression(C=1.0, max_iter=3000, **params)
    else:
        prep = ColumnTransformer([("num", "passthrough", features)])
        if name == "random_forest":
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                           random_state=seed, **params)
        elif name == "xgboost":
            model = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.7, eval_metric="logloss", n_jobs=-1,
                                  random_state=seed, **params)
        elif name == "lightgbm":
            model = LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, random_state=seed, verbose=-1, **params)
        else:
            raise ValueError(name)
    prep.set_output(transform="default")
    return Pipeline([("prep", prep), ("model", model)])


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 split stratified on (label x has-path) so every split has the same mix
    of the easy (homepage) and hard (deep-link) cases."""
    strata = df[TARGET].astype(str) + "_" + has_path(df).astype(str)
    trainval, test = train_test_split(df, test_size=0.2, stratify=strata, random_state=seed)
    strata_tv = strata.loc[trainval.index]
    train, valid = train_test_split(trainval, test_size=0.25, stratify=strata_tv, random_state=seed)
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
