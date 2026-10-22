"""Data acquisition, validation, cleaning and feature groups.

Dataset: Vrbančič, Fister & Podgorelec (2020), "Datasets for phishing websites detection",
Data in Brief 33:106438. CC BY 4.0. Full variant: 88,647 websites x 111 features.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "dataset_full.csv"
SAMPLE_PATH = ROOT / "data" / "sample" / "dataset_sample.csv"
SOURCE_URL = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
TARGET = "phishing"

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------
_CHARS = ["dot", "hyphen", "underline", "slash", "questionmark", "equal", "at", "and", "exclamation",
          "space", "tilde", "comma", "plus", "asterisk", "hashtag", "dollar", "percent"]

URL_FEATURES = [f"qty_{c}_url" for c in _CHARS] + ["qty_tld_url", "length_url"]
DOMAIN_FEATURES = [f"qty_{c}_domain" for c in _CHARS] + [
    "qty_vowels_domain", "domain_length", "domain_in_ip", "server_client_domain"]
DIRECTORY_FEATURES = [f"qty_{c}_directory" for c in _CHARS] + ["directory_length"]
FILE_FEATURES = [f"qty_{c}_file" for c in _CHARS] + ["file_length"]
PARAMS_FEATURES = [f"qty_{c}_params" for c in _CHARS] + [
    "params_length", "tld_present_params", "qty_params"]
OTHER_LEXICAL = ["email_in_url", "url_shortened"]

# Everything computable from the URL string alone -> usable *before* a page is visited
LEXICAL_FEATURES = (URL_FEATURES + DOMAIN_FEATURES + DIRECTORY_FEATURES + FILE_FEATURES
                    + PARAMS_FEATURES + OTHER_LEXICAL)

# Require DNS / WHOIS / TLS / HTTP lookups at scan time
NETWORK_FEATURES = [
    "time_response", "domain_spf", "asn_ip", "time_domain_activation", "time_domain_expiration",
    "qty_ip_resolved", "qty_nameservers", "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate",
    "qty_redirects",
]

# ~99.8% zeros in both classes: the Google look-ups evidently failed during collection
DEAD_FEATURES = ["url_google_index", "domain_google_index"]

FEATURE_SETS = {
    "lexical": LEXICAL_FEATURES,
    "network": NETWORK_FEATURES,
    "full": LEXICAL_FEATURES + NETWORK_FEATURES,
}


@dataclass
class CleaningReport:
    steps: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        log.info(msg)
        self.steps.append(msg)

    def __str__(self) -> str:
        return "\n".join(f"- {s}" for s in self.steps)


def download(dest: Path = RAW_PATH, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", SOURCE_URL)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as r:
        dest.write_bytes(r.read())
    return dest


def load_raw(path: Path | str | None = None, sample: bool = False) -> pd.DataFrame:
    path = Path(path) if path else (SAMPLE_PATH if sample else download())
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    expected = set(LEXICAL_FEATURES + NETWORK_FEATURES + DEAD_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be binary 0/1")
    if df.isna().any().any():
        raise ValueError("Unexpected NaN values (the source encodes missing as -1)")
    if (df[LEXICAL_FEATURES] < -1).any().any():
        raise ValueError("Lexical features must be >= -1")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    rep = CleaningReport()
    n0 = len(df)
    out = df.drop_duplicates()
    rep.add(f"Removed {n0 - len(out):,} exact duplicate rows (v1 called drop_duplicates() without assigning the result)")

    feats = [c for c in out.columns if c != TARGET]
    n_labels = out.groupby(feats)[TARGET].transform("nunique")
    conflicting = int((n_labels > 1).sum())
    out = out[n_labels == 1]
    rep.add(f"Removed {conflicting} rows whose identical feature vectors carry both labels")

    out = out.drop(columns=DEAD_FEATURES)
    rep.add(f"Dropped {len(DEAD_FEATURES)} dead features {DEAD_FEATURES} (≈99.8% zeros in both classes)")

    rep.add("Kept -1 as an explicit 'absent / lookup failed' value; NOT dropped by dropna() because it is not NaN")
    out = out.reset_index(drop=True)
    rep.add(f"Final: {len(out):,} rows, {out[TARGET].mean():.1%} phishing")
    return out, rep


def has_path(df: pd.DataFrame) -> pd.Series:
    """True when the URL has a path component (directory_length == -1 means no path)."""
    return df["directory_length"] != -1
