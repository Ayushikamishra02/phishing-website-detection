"""Figures for the report / README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .explain import pretty  # noqa: E402

RED, BLUE, GREEN, GREY, AMBER = "#dc2626", "#2563eb", "#16a34a", "#94a3b8", "#d97706"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def shortcut(path_share: dict, baselines: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    labels = ["Legitimate", "Phishing"]
    vals = [path_share[0] * 100, path_share[1] * 100]
    axes[0].bar(labels, vals, color=[BLUE, RED], width=0.55)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    axes[0].set(title="Share of URLs that have a path", ylabel="% of URLs", ylim=(0, 110))

    names, accs = list(baselines), [baselines[k] * 100 for k in baselines]
    colors = [GREY, AMBER, RED, GREEN][: len(names)]
    axes[1].barh(names[::-1], accs[::-1], color=colors[::-1])
    for i, v in enumerate(accs[::-1]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[1].set(title="Test accuracy", xlim=(50, 103), xlabel="Accuracy (%)")
    fig.suptitle("A one-line rule ('URL has a path → phishing') beats the v1 model", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, out, "shortcut")


def model_grid(cv: pd.DataFrame, metric: str, out: Path):
    grid = cv.pivot(index="feature_set", columns="model", values=metric).loc[["lexical", "network", "full"]]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    lo = grid.values.min() - 0.02
    im = ax.imshow(grid.values, cmap="Greens", vmin=lo, vmax=1)
    ax.set_xticks(range(grid.shape[1]), [c.replace("_", "\n") for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{r} features" for r in grid.index])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9,
                    color="white" if v > lo + 0.6 * (1 - lo) else "black")
    ax.grid(False)
    ax.set_title(f"5-fold CV {metric.replace('_', '-').upper()} (training set)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out, "model_comparison")


def roc_low_fpr(curves: dict, out: Path):
    """curves: {label: (y_true, y_score, color, linestyle)}"""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, (y, s, c, ls) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=c, ls=ls, lw=2, label=label)
    ax.axvline(0.01, color="black", lw=0.8, ls=":")
    ax.text(0.0105, 0.04, "1% FPR budget", fontsize=8)
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="False-positive rate (log scale)",
           ylabel="Phishing caught (TPR)", title="ROC at low false-positive rates (test set)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out, "roc_low_fpr")


def stratified(strat: pd.DataFrame, out: Path):
    """Recall and FPR on homepages vs deep links, global vs per-URL-type thresholds."""
    subsets = ["homepages", "deep links"]
    combos = [("lexical", "global", "#fcd34d"), ("lexical", "per URL type", AMBER),
              ("full", "global", "#86efac"), ("full", "per URL type", GREEN)]
    x = np.arange(len(subsets))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (model, thr, c) in enumerate(combos):
        sub = strat[(strat["model"] == model) & (strat["thresholds"] == thr)].set_index("subset").loc[subsets]
        off = (i - 1.5) * w
        axes[0].bar(x + off, sub["recall"] * 100, w, color=c, label=f"{model} model, {thr} threshold")
        axes[1].bar(x + off, sub["fpr"] * 100, w, color=c)
        for xi, r, f in zip(x, sub["recall"], sub["fpr"]):
            axes[0].text(xi + off, r * 100 + 1.5, f"{r:.0%}", ha="center", fontsize=7)
            axes[1].text(xi + off, f * 100 + 0.08, f"{f:.1%}", ha="center", fontsize=7)
    axes[1].axhline(1, color="black", ls=":", lw=1)
    axes[1].text(1.45, 1.08, "1% budget", fontsize=8, ha="right")
    axes[0].set(title="Phishing caught (recall)", ylabel="%", ylim=(0, 105), xticks=x)
    axes[1].set(title="Legitimate sites wrongly flagged (FPR)", ylabel="%", xticks=x)
    axes[0].set_xticklabels(subsets)
    axes[1].set_xticklabels(subsets)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("One global threshold breaks the 1% FPR budget on deep links; per-URL-type thresholds bring it back to ~1%",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, out, "stratified_performance")


def shap_beeswarm(values: np.ndarray, X: pd.DataFrame, title: str, name: str, out: Path, n: int = 15):
    plt.figure()
    shap.summary_plot(values, X.values, feature_names=[pretty(c) for c in X.columns], max_display=n,
                      show=False, plot_size=(8.5, 6))
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close()


def v1_vs_v2(v1: dict, v2: dict, out: Path):
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, [v1[k] * 100 for k in keys], 0.4, color=GREY, label="v1 (original notebook)")
    ax.bar(x + 0.2, [v2[k] * 100 for k in keys], 0.4, color=GREEN, label="v2 (this project, full model)")
    for i, k in enumerate(keys):
        ax.text(i - 0.2, v1[k] * 100 + 1, f"{v1[k]:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, v2[k] * 100 + 1, f"{v2[k]:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x, [k.title() for k in keys])
    ax.set(ylim=(0, 110), ylabel="%", title="v1 vs v2 on the same test set")
    ax.legend(loc="lower right")
    _save(fig, out, "v1_vs_v2")
