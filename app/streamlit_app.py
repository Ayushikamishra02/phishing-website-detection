"""PhishGuard dashboard: URL scanner + evaluation report.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phishguard.predict import scan_url  # noqa: E402
from phishguard.url_features import extract  # noqa: E402

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")
FIG = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"
COLORS = {"phishing": "#dc2626", "suspicious": "#d97706", "likely legitimate": "#16a34a",
          "trusted domain": "#2563eb"}

EXAMPLES = {
    "Homepage": "wikipedia.org",
    "Look-alike login": "http://paypal.com.account-verify.secure-login.xyz/signin/update.php?id=5531&session=a8f2",
    "IP-address host": "http://192.168.10.24/bank/login/index.html",
    "E-mail in URL": "http://mail-service-update.com/confirm?user=jane.doe@example.com",
    "Legit deep link": "docs.python.org/3/library/urllib.parse.html",
    "Look-alike of allowlisted": "http://github.com.login-session.app/auth",
}

st.title("🛡️ PhishGuard: phishing website detection")
st.caption("Gradient-boosted models on 87k labelled websites · URL-only screening + full model with domain "
           "intelligence · portfolio project, not a security product")

tab_scan, tab_report, tab_about = st.tabs(["URL scanner", "Model report", "How it works"])

with tab_scan:
    if "url" not in st.session_state:
        st.session_state.url = EXAMPLES["Look-alike login"]
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for c, (label, u) in zip(cols, EXAMPLES.items()):
        if c.button(label, use_container_width=True):
            st.session_state.url = u
    url = st.text_input("URL to scan", key="url")

    if url.strip():
        try:
            res = scan_url(url.strip())
        except FileNotFoundError:
            st.error("Models not found. Run `make train` first.")
            st.stop()
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Phishing probability", f"{res['phishing_probability']:.1%}")
        c2.markdown(f"#### Verdict<br><span style='color:{COLORS[res['verdict']]};font-size:1.8rem;"
                    f"font-weight:700'>{res['verdict'].upper()}</span>", unsafe_allow_html=True)
        c3.markdown(f"**Model:** `{res['model']}` · **threshold for this URL type:** {res['threshold']:.3f} "
                    f"(≤ 1% false positives on validation) · **registered domain:** `{res['registered_domain']}`")
        if res["note"]:
            (c3.info if res["verdict"] == "trusted domain" else c3.warning)(res["note"])

        st.subheader("Why? (SHAP contributions)")
        r = pd.DataFrame(res["top_reasons"])
        r["label"] = r["label"] + " = " + r["value"].map(lambda v: f"{v:g}")
        st.altair_chart(alt.Chart(r).mark_bar().encode(
            x=alt.X("shap:Q", title="Contribution (log-odds); positive = towards phishing"),
            y=alt.Y("label:N", sort=alt.EncodingSortField("shap", op="max", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=320)),
            color=alt.condition("datum.shap > 0", alt.value("#dc2626"), alt.value("#16a34a")),
            tooltip=["label", alt.Tooltip("shap:Q", format=".3f")]).properties(height=260),
            use_container_width=True)

        with st.expander("All extracted URL features"):
            feats = pd.Series(extract(url.strip()), name="value")
            st.dataframe(feats[feats != 0].to_frame(), use_container_width=True)

with tab_report:
    if not METRICS.exists():
        st.info("Run `make train` to generate the report.")
    else:
        m = json.loads(METRICS.read_text())
        full, lex = m["final"]["full"], m["final"]["lexical"]
        deep = {fs: next(r for r in m["final"][fs]["stratified_segment_thresholds"] if r["subset"] == "deep links")
                for fs in ("full", "lexical")}
        k = st.columns(5)
        k[0].metric("Full model ROC-AUC", f"{full['test']['roc_auc']:.4f}")
        k[1].metric("Full: deep-link phishing caught", f"{deep['full']['recall']:.0%}",
                    f"at {deep['full']['fpr']:.1%} FPR", delta_color="off")
        k[2].metric("URL-only ROC-AUC", f"{lex['test']['roc_auc']:.4f}")
        k[3].metric("URL-only: deep-link phishing caught", f"{deep['lexical']['recall']:.0%}",
                    f"at {deep['lexical']['fpr']:.1%} FPR", delta_color="off")
        k[4].metric("'Has a path' rule accuracy", f"{m['baselines']['has_path_rule_accuracy']:.1%}")
        cols = st.columns(2)
        for i, f in enumerate(["shortcut", "stratified_performance", "roc_low_fpr", "model_comparison",
                               "shap_full", "shap_lexical", "v1_vs_v2"]):
            p = FIG / f"{f}.png"
            if p.exists():
                cols[i % 2].image(str(p), use_container_width=True)

with tab_about:
    card = ROOT / "docs" / "MODEL_CARD.md"
    st.markdown(card.read_text() if card.exists() else "See README.md")
