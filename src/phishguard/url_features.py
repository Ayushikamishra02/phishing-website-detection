"""Turn a raw URL string into the dataset's lexical features.

The dataset ships features only (no raw URLs) and no extraction code, so these rules
were *reverse-engineered* from identities that hold for >95% of rows, e.g.

* URLs are stored without the scheme:     no path  =>  length_url == domain_length, qty_slash_url == 0
* "directory" is the whole path (it includes the file name):  qty_dot_url == qty_dot_domain + qty_dot_directory
* the "?" separating the query is not counted:  length_url == domain + directory + params lengths
* a component that is absent gets -1 for all of its features

`tests/test_url_features.py` asserts these identities on extracted features, so the extractor and the
training data follow the same conventions. Two features (`qty_tld_url`, `tld_present_params`) are
approximations because the authors' TLD list is unknown. See docs/MODEL_CARD.md.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlsplit

from .data import LEXICAL_FEATURES

CHARS = {
    "dot": ".", "hyphen": "-", "underline": "_", "slash": "/", "questionmark": "?", "equal": "=",
    "at": "@", "and": "&", "exclamation": "!", "space": " ", "tilde": "~", "comma": ",", "plus": "+",
    "asterisk": "*", "hashtag": "#", "dollar": "$", "percent": "%",
}

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly",
    "shorturl.at", "rebrand.ly", "tiny.cc", "lnkd.in", "db.tt", "qr.ae", "rb.gy", "v.gd", "x.co", "tr.im",
    "cli.gs", "mcaf.ee", "su.pr", "po.st", "bc.vc", "u.to", "j.mp", "s.id", "shorte.st", "soo.gd",
}

_GTLDS = (
    "com net org info biz gov edu mil int co io ai app dev xyz top online site club shop store tech live "
    "me tv cc ws mobi name pro asia tk ml ga cf gq icu vip work click link win loan bid review"
).split()
_CCTLDS = (
    "ac ad ae af ag al am ao ar at au az ba bd be bg bh bo br by bz ca ch cl cn cr cu cy cz de dk do dz ec "
    "ee eg es et eu fi fj fr ge gh gr gt hk hn hr hu id ie il in iq ir is it jm jo jp ke kg kh kr kw kz la lb "
    "lk lt lu lv ly ma md mk mn mo mt mu mx my na ng ni nl no np nz om pa pe ph pk pl pr ps pt py qa ro rs ru "
    "rw sa se sg si sk sn su sv sy th tn tr tw tz ua ug uk us uy uz ve vn za zm zw"
).split()
TLDS = set(_GTLDS) | set(_CCTLDS)
_TLD_RE = re.compile(r"\.(" + "|".join(sorted(TLDS, key=len, reverse=True)) + r")(?=$|[^a-z0-9])", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _counts(text: str, suffix: str) -> dict[str, int]:
    return {f"qty_{name}_{suffix}": text.count(ch) for name, ch in CHARS.items()}


def _absent(suffix: str) -> dict[str, int]:
    return {f"qty_{name}_{suffix}": -1 for name in CHARS}


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def split_url(url: str) -> tuple[str, str | None, str | None]:
    """Return (domain, directory_or_None, params_or_None) using the dataset's conventions."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    parts = urlsplit(url)
    domain = parts.netloc
    directory = parts.path or None
    params = parts.query or None
    return domain, directory, params


def extract(url: str) -> dict[str, int]:
    """URL string -> {feature: value} for every feature in LEXICAL_FEATURES."""
    domain, directory, params = split_url(url)
    f: dict[str, int] = {}

    f.update(_counts(domain, "domain"))
    host = domain.split("@")[-1].split(":")[0].lower()
    f["qty_vowels_domain"] = sum(host.count(v) for v in "aeiou")
    f["domain_length"] = len(domain)
    f["domain_in_ip"] = int(_is_ip(host))
    f["server_client_domain"] = int("server" in host or "client" in host)

    if directory is not None:
        f.update(_counts(directory, "directory"))
        f["directory_length"] = len(directory)
        file = directory.rsplit("/", 1)[-1]
        f.update(_counts(file, "file"))
        f["file_length"] = len(file)
    else:
        f.update(_absent("directory"))
        f["directory_length"] = -1
        f.update(_absent("file"))
        f["file_length"] = -1

    if params is not None:
        f.update(_counts(params, "params"))
        f["params_length"] = len(params)
        f["tld_present_params"] = int(bool(_TLD_RE.search(params)))
        f["qty_params"] = len(parse_qsl(params))
    else:
        f.update(_absent("params"))
        f["params_length"] = -1
        f["tld_present_params"] = -1
        f["qty_params"] = -1

    full = domain + (directory or "") + (params or "")
    f.update(_counts(full, "url"))
    f["length_url"] = len(full)
    f["qty_tld_url"] = len(_TLD_RE.findall(full.lower()))
    f["email_in_url"] = int(bool(_EMAIL_RE.search(full)))
    f["url_shortened"] = int(host.removeprefix("www.") in SHORTENERS)

    return {k: f[k] for k in LEXICAL_FEATURES}


def extract_many(urls: list[str]):
    import pandas as pd

    return pd.DataFrame([extract(u) for u in urls], columns=LEXICAL_FEATURES)
