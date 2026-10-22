"""The URL extractor must follow the same conventions as the training data.

Identities below were measured on the dataset (they hold for >95% of its rows) and are
asserted here on URLs we generate, so the deployed scanner speaks the training data's language.
"""
import pytest

from phishguard.data import LEXICAL_FEATURES
from phishguard.url_features import extract, split_url

URLS = [
    "google.com",
    "https://www.example.co.uk/",
    "http://login.example.com/account/verify.php?id=12&token=abc",
    "secure-paypal.com.verify-account.xyz/signin/update.html",
    "http://192.168.0.1/admin/",
    "bit.ly/3xYzAb",
    "http://example.com/path/to/file.tar.gz?a=1&b=&c",
    "example.org/~user/a+b,c*d$e%20f!g",
]


@pytest.mark.parametrize("url", URLS)
def test_returns_every_lexical_feature(url):
    f = extract(url)
    assert list(f) == LEXICAL_FEATURES
    assert all(isinstance(v, int) for v in f.values())
    assert all(v >= -1 for v in f.values())


@pytest.mark.parametrize("url", URLS)
def test_dataset_identities(url):
    f = extract(url)
    dir_len = max(f["directory_length"], 0)
    params_len = max(f["params_length"], 0)
    # length_url == domain + directory (incl. file) + params; scheme and '?' are not counted
    assert f["length_url"] == f["domain_length"] + dir_len + params_len
    # character counts add up across components
    for ch in ("dot", "hyphen", "slash", "equal", "and"):
        parts = [f[f"qty_{ch}_domain"], max(f[f"qty_{ch}_directory"], 0), max(f[f"qty_{ch}_params"], 0)]
        assert f[f"qty_{ch}_url"] == sum(parts), ch
    # the file is part of the directory
    if f["directory_length"] != -1:
        assert f["directory_length"] >= f["file_length"] >= 0
        assert f["qty_slash_file"] == 0


def test_absent_components_are_minus_one():
    f = extract("example.com")
    assert f["directory_length"] == f["file_length"] == f["params_length"] == -1
    assert f["qty_params"] == f["tld_present_params"] == -1
    assert f["qty_slash_url"] == 0                     # no scheme, no path


def test_scheme_is_ignored():
    assert extract("https://example.com/a") == extract("example.com/a")


def test_specific_signals():
    assert extract("http://10.0.0.1/x")["domain_in_ip"] == 1
    assert extract("bit.ly/abc")["url_shortened"] == 1
    assert extract("a.com/?u=john@mail.com")["email_in_url"] == 1
    assert extract("myserver.example.com")["server_client_domain"] == 1
    assert extract("a.com/x?a=1&b=2&c=3")["qty_params"] == 3
    assert extract("a.com/x?next=evil.com")["tld_present_params"] == 1


def test_split_url():
    assert split_url("https://a.com/b/c.html?x=1") == ("a.com", "/b/c.html", "x=1")
    assert split_url("a.com") == ("a.com", None, None)


def test_registered_domain_and_allowlist():
    from phishguard.predict import allowlist, registered_domain
    assert registered_domain("https://login.paypal.com.evil-site.xyz/a") == "evil-site.xyz"
    assert registered_domain("www.bbc.co.uk/news") == "bbc.co.uk"
    assert registered_domain("http://user@github.com:443/x") == "github.com"
    assert registered_domain("http://192.168.10.24/bank") == "192.168.10.24"
    assert "github.com" in allowlist()
    assert registered_domain("http://github.com.login-session.app/auth") not in allowlist()
