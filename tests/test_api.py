import pytest
from fastapi.testclient import TestClient

from phishguard.data import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "full_model.joblib").exists(),
                                reason="models not trained - run `make train` or `make train-sample`")


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_scan_returns_verdict_and_reasons(client):
    r = client.post("/scan", json={"url": "http://paypal.com.verify-account.xyz/signin/update.php?id=1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["phishing_probability"] <= 1
    assert body["verdict"] in {"phishing", "suspicious", "likely legitimate"}
    assert body["has_path"] is True and body["note"]
    assert len(body["top_reasons"]) == 6


def test_homepage_scores_lower_than_lookalike(client):
    home = client.post("/scan", json={"url": "unknown-local-bakery.com"}).json()
    bad = client.post("/scan", json={"url": "http://paypal.com.verify-account.xyz/signin/update.php?id=1"}).json()
    assert home["phishing_probability"] < bad["phishing_probability"]
    assert home["has_path"] is False and home["note"] is None


def test_scan_rejects_bad_input(client):
    assert client.post("/scan", json={"url": "  "}).status_code == 422
    assert client.post("/scan", json={"url": "has space.com"}).status_code == 422


def test_score_requires_all_features(client):
    assert client.post("/score", json={"features": {"length_url": 10}}).status_code == 422


def test_score_full_model(client):
    from phishguard.data import FEATURE_SETS, clean, load_raw
    df, _ = clean(load_raw(sample=True))
    row = df.iloc[0][FEATURE_SETS["full"]].astype(float).to_dict()
    r = client.post("/score", json={"features": row})
    assert r.status_code == 200, r.text
    assert len(r.json()["top_reasons"]) == 6


def test_allowlisted_domain_vs_lookalike(client):
    trusted = client.post("/scan", json={"url": "https://github.com/pandas-dev/pandas"}).json()
    fake = client.post("/scan", json={"url": "http://github.com.login-session.app/auth"}).json()
    assert trusted["verdict"] == "trusted domain"
    assert fake["verdict"] != "trusted domain" and fake["registered_domain"] == "login-session.app"
