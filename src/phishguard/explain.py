"""SHAP explanations (tree models)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

PRETTY = {
    "time_domain_activation": "Domain age (days)",
    "time_domain_expiration": "Days until domain expires",
    "directory_length": "Path length",
    "qty_slash_url": "'/' count in URL",
    "qty_slash_directory": "'/' count in path",
    "length_url": "URL length",
    "domain_length": "Domain length",
    "qty_dot_domain": "'.' count in domain",
    "qty_dot_url": "'.' count in URL",
    "qty_hyphen_domain": "'-' in domain",
    "qty_vowels_domain": "Vowels in domain",
    "file_length": "File-name length",
    "params_length": "Query length",
    "qty_params": "Number of query params",
    "ttl_hostname": "DNS TTL",
    "asn_ip": "Hosting ASN",
    "time_response": "DNS response time",
    "qty_nameservers": "Name servers",
    "qty_mx_servers": "Mail (MX) servers",
    "qty_ip_resolved": "Resolved IPs",
    "domain_spf": "SPF record",
    "tls_ssl_certificate": "Valid TLS certificate",
    "qty_redirects": "Redirects",
    "qty_tld_url": "TLD occurrences",
    "email_in_url": "E-mail in URL",
    "url_shortened": "URL shortener",
    "domain_in_ip": "IP address as domain",
    "qty_at_url": "'@' in URL",
    "qty_equal_url": "'=' in URL",
}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("qty_"):
        ch, _, part = name[4:].rpartition("_")
        return f"{ch} count in {part}"
    return name.replace("_", " ")


def shap_values(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = pipe.named_steps["prep"].transform(X)
    vals = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    return vals[:, :, 1] if vals.ndim == 3 else vals


def explain_one(pipe: Pipeline, x: pd.DataFrame, features: list[str], top_k: int = 6) -> list[dict]:
    v = shap_values(pipe, x[features])[0]
    order = np.argsort(-np.abs(v))[:top_k]
    return [{
        "feature": features[i], "label": pretty(features[i]),
        "value": float(x.iloc[0][features[i]]), "shap": round(float(v[i]), 4),
        "direction": "towards phishing" if v[i] > 0 else "towards legitimate",
    } for i in order]
