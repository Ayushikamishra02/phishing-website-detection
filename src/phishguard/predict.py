"""Scoring helpers used by the CLI, API and dashboard.

    python -m phishguard.predict "http://paypal.com.account-verify.xyz/login.php?id=1"
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import joblib
import pandas as pd

from .data import ROOT, has_path
from .explain import explain_one
from .url_features import _is_ip, extract, split_url

MODELS_DIR = ROOT / "models"
ALLOWLIST_PATH = ROOT / "data" / "allowlist.txt"
_MULTI_PART_SUFFIXES = {"co.uk", "org.uk", "gov.uk", "ac.uk", "co.in", "org.in", "gov.in", "com.au", "net.au",
                        "co.jp", "com.br", "co.za", "com.mx", "co.nz", "com.sg"}


@lru_cache(maxsize=1)
def allowlist() -> frozenset[str]:
    if not ALLOWLIST_PATH.exists():
        return frozenset()
    lines = ALLOWLIST_PATH.read_text().splitlines()
    return frozenset(x.strip().lower() for x in lines if x.strip() and not x.startswith("#"))


def registered_domain(url: str) -> str:
    """'https://login.paypal.com.evil.xyz/a' -> 'evil.xyz'; 'www.bbc.co.uk' -> 'bbc.co.uk'."""
    host = split_url(url)[0].split("@")[-1].split(":")[0].lower().rstrip(".")
    if _is_ip(host):
        return host
    labels = host.split(".")
    n = 3 if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES else 2
    return ".".join(labels[-n:])


@lru_cache(maxsize=2)
def load(kind: str = "lexical"):
    path = MODELS_DIR / f"{kind}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m phishguard.train` first")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return joblib.load(path), meta[kind]


def verdict(p: float, threshold: float) -> str:
    if p >= threshold:
        return "phishing"
    if p >= 0.5:
        return "suspicious"
    return "likely legitimate"


def segment_threshold(meta: dict, deep_link: bool) -> float:
    return meta["segment_thresholds"]["deep_link" if deep_link else "homepage"]


def score_frame(X: pd.DataFrame, kind: str = "full") -> pd.DataFrame:
    model, meta = load(kind)
    p = model.predict_proba(X[meta["features"]])[:, 1]
    thr = [segment_threshold(meta, d) for d in has_path(X)]
    return pd.DataFrame({"phishing_probability": p.round(4),
                         "verdict": [verdict(x, t) for x, t in zip(p, thr)]}, index=X.index)


def scan_url(url: str, explain: bool = True) -> dict:
    """URL-only screening with the lexical model."""
    model, meta = load("lexical")
    x = pd.DataFrame([extract(url)])
    p = float(model.predict_proba(x[meta["features"]])[0, 1])
    deep = bool(has_path(x).iloc[0])
    thr = meta["segment_thresholds"]["deep_link" if deep else "homepage"]
    reg = registered_domain(url)
    trusted = reg in allowlist()
    out = {
        "url": url,
        "registered_domain": reg,
        "phishing_probability": round(p, 4),
        "verdict": "trusted domain" if trusted else verdict(p, thr),
        "threshold": round(thr, 4),
        "model": f"lexical/{meta['model']}",
        "has_path": deep,
        "note": (f"'{reg}' is on the reputation allowlist; the model score is shown for transparency only."
                 if trusted else
                 "Deep link: URL-only screening has a high false-positive rate on legitimate deep links. "
                 "Confirm with domain intelligence (full model) before blocking." if deep else None),
    }
    if explain:
        out["top_reasons"] = explain_one(model, x, meta["features"])
    return out


def main():
    for url in sys.argv[1:] or [input("URL: ")]:
        print(json.dumps(scan_url(url), indent=2))


if __name__ == "__main__":
    main()
