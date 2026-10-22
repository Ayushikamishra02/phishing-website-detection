"""PhishGuard API.

    uvicorn api.main:app --reload        ->  http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from phishguard import __version__
from phishguard.data import FEATURE_SETS
from phishguard.explain import explain_one
from phishguard.predict import load, scan_url, segment_threshold, verdict

app = FastAPI(
    title="PhishGuard: phishing website detection",
    version=__version__,
    description="`/scan` screens a raw URL using URL-only (lexical) features. "
                "`/score` scores a site with the full model when domain-intelligence features are available. "
                "Research / portfolio project.",
)


class ScanRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=4096, examples=["http://secure-paypa1.com-login.xyz/update.php?id=7"])
    explain: bool = True

    @field_validator("url")
    @classmethod
    def no_whitespace_only(cls, v: str) -> str:
        if not v.strip() or " " in v.strip():
            raise ValueError("URL must be non-empty and contain no spaces")
        return v.strip()


class Reason(BaseModel):
    feature: str
    label: str
    value: float
    shap: float
    direction: str


class ScanResponse(BaseModel):
    url: str
    phishing_probability: float
    registered_domain: str
    verdict: Literal["phishing", "suspicious", "likely legitimate", "trusted domain"]
    threshold: float
    model: str
    has_path: bool
    note: str | None = None
    top_reasons: list[Reason] | None = None


class ScoreRequest(BaseModel):
    features: dict[str, float] = Field(..., description="All 109 lexical + network features (use -1 for unknown)")


class ScoreResponse(BaseModel):
    phishing_probability: float
    verdict: str
    threshold: float
    top_reasons: list[Reason]


def _require(kind: str):
    try:
        return load(kind)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health():
    try:
        load("lexical")
        load("full")
        return {"status": "ok"}
    except FileNotFoundError:
        return {"status": "models missing"}


@app.get("/model-info")
def model_info():
    return {k: {kk: v for kk, v in _require(k)[1].items() if kk != "features"} for k in ("lexical", "full")}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    _require("lexical")
    return scan_url(req.url, explain=req.explain)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    model, meta = _require("full")
    missing = set(FEATURE_SETS["full"]) - set(req.features)
    if missing:
        raise HTTPException(422, f"Missing {len(missing)} features, e.g. {sorted(missing)[:5]}")
    x = pd.DataFrame([{f: req.features[f] for f in meta["features"]}])
    p = float(model.predict_proba(x)[0, 1])
    thr = segment_threshold(meta, x["directory_length"].iloc[0] != -1)
    return ScoreResponse(phishing_probability=round(p, 4), verdict=verdict(p, thr), threshold=round(thr, 4),
                         top_reasons=explain_one(model, x, meta["features"]))
