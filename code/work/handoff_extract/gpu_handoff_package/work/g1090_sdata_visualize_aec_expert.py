from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_aec_expert_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds  # noqa: E402
from g1090_sdata_aec_assault import load_dataset, row_norm  # noqa: E402
from g1090_sdata_aec_counterattack import make_pipeline, score_estimator  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import expert_oof_test, feature_sets_no_scanner  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 20260626
BLUE = "#395D9C"
TEAL = "#008C7A"
RED = "#B33A3A"
DARK = "#26313D"
GRAY = "#7C8794"
LIGHT = "#EEF2F6"


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#64707D") -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        fc=fc,
        ec=ec,
        lw=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=DARK, fontweight="bold")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            color="#53606E",
            lw=1.4,
            shrinkA=3,
            shrinkB=3,
        )
    )


def build_direct_feature_frame(train: dict, test: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return feature_sets_no_scanner(train, test)["direct_curve"]


def fit_final_aec_expert(train: dict, test: dict) -> dict:
    tr_df, te_df = build_direct_feature_frame(train, test)
    xtr = tr_df.to_numpy(dtype=float)
    xte = te_df.to_numpy(dtype=float)
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)

    oof, test_score = expert_oof_test(xtr, xte, ytr, folds, "linsvm_C0.2", 128)
    final = make_pipeline("linsvm_C0.2", 128, xtr.shape[1], SEED + 99)
    final.fit(xtr, ytr)
    final_score = score_estimator(final, xte)

    selector = final.named_steps["select"]
    clf = final.named_steps["clf"]
    support = selector.get_support()
    coef_full = np.zeros(xtr.shape[1], dtype=float)
    coef_full[support] = clf.coef_.ravel()
    coefs = pd.DataFrame(
        {
            "feature": tr_df.columns,
            "selected": support.astype(int),
            "svm_standardized_coef": coef_full,
            "abs_coef": np.abs(coef_full),
            "stream": ["a128" if i < 128 else "crop" for i in range(256)],
            "position": [i if i < 128 else i - 128 for i in range(256)],
        }
    )
    coefs.sort_values("abs_coef", ascending=False).to_csv(OUT_DIR / "aec_expert_linear_svm_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["g1090_oof", "sdata_external", "sdata_external_final_refit"],
            "auc": [auc_or_nan(ytr, oof), auc_or_nan(yte, test_score), auc_or_nan(yte, final_score)],
        }
    ).to_csv(OUT_DIR / "aec_expert_auc_summary.csv", index=False)
    return {
        "train": train,
        "test": test,
        "tr_df": tr_df,
        "te_df": te_df,
        "oof": oof,
        "test_score": test_score,
        "final_score": final_score,
        "coefs": coefs,
        "selected_count": int(support.sum()),
        "auc_oof": auc_or_nan(ytr, oof),
        "auc_test": auc_or_nan(yte, test_score),
    }


def plot_mean_curves(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str) -> None:
    pos = np.arange(128)
    centered = row_norm(x) - 1.0
    for cls, color, label in [(0, GRAY, "Not low SMI"), (1, RED, "Low SMI")]:
        m = y == cls
        mean = np.nanmean(centered[m], axis=0)
        lo = np.nanpercentile(centered[m], 25, axis=0)
        hi = np.nanpercentile(centered[m], 75, axis=0)
        ax.plot(pos, mean, color=color, lw=2.2, label=label)
        ax.fill_between(pos, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0, color="#AAB3BD", lw=1)
    ax.set_xlim(0, 127)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Anatomic AEC position")
    ax.set_ylabel("Centered AEC")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)


def plot_coef_map(ax: plt.Axes, coefs: pd.DataFrame) -> None:
    pos = np.arange(128)
    a = coefs[coefs["stream"].eq("a128")].sort_values("position")
    c = coefs[coefs["stream"].eq("crop")].sort_values("position")
    ax.axhline(0, color="#30343A", lw=1)
    ax.plot(pos, a["svm_standardized_coef"], color=BLUE, lw=2.0, label="a128 stream")
    ax.plot(pos, c["svm_standardized_coef"], color=TEAL, lw=2.0, label="cropped stream")
    for df, color in [(a, BLUE), (c, TEAL)]:
        top = df.sort_values("abs_coef", ascending=False).head(12)
        ax.scatter(top["position"], top["svm_standardized_coef"], color=color, edgecolor="white", s=52, zorder=4)
    ax.set_xlim(0, 127)
    lim = max(0.02, float(np.nanmax(np.abs(coefs["svm_standardized_coef"]))) * 1.15)
    ax.set_ylim(-lim, lim)
    ax.set_title("B. Linear SVM Coefficient Map", loc="left", fontweight="bold")
    ax.set_xlabel("Anatomic AEC position")
    ax.set_ylabel("Standardized coefficient")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=9)


def plot_score_distribution(ax: plt.Axes, y: np.ndarray, score: np.ndarray, auc: float) -> None:
    bins = np.linspace(np.percentile(score, 1), np.percentile(score, 99), 28)
    ax.hist(score[y == 0], bins=bins, color=GRAY, alpha=0.45, density=True, label="Not low SMI")
    ax.hist(score[y == 1], bins=bins, color=RED, alpha=0.45, density=True, label="Low SMI")
    ax.axvline(np.median(score[y == 0]), color=GRAY, lw=2)
    ax.axvline(np.median(score[y == 1]), color=RED, lw=2)
    ax.set_title(f"C. AEC Expert Score in sdata (AUC {auc:.3f})", loc="left", fontweight="bold")
    ax.set_xlabel("Linear SVM decision score")
    ax.set_ylabel("Density")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=9)


def make_method_figure(d: dict) -> None:
    fig = plt.figure(figsize=(14.0, 10.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.15, 1.0], hspace=0.52, wspace=0.35)

    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.set_axis_off()
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.text(0.0, 0.98, "A. AEC Expert Construction", ha="left", va="top", fontsize=14, fontweight="bold", color=DARK)
    boxes = [
        ((0.03, 0.35), (0.16, 0.30), "Aligned AEC curves\n128 a128 + 128 crop", "#EAF0F8"),
        ((0.25, 0.35), (0.15, 0.30), "Patient-wise\nmean normalization", "#EAF7F4"),
        ((0.46, 0.35), (0.15, 0.30), "256 direct\ncurve features", "#FFF5E6"),
        ((0.67, 0.35), (0.13, 0.30), "SelectKBest\n128 features", "#F2EEF8"),
        ((0.86, 0.35), (0.11, 0.30), "Linear SVM\nscore", "#F9ECEC"),
    ]
    for xy, wh, text, fc in boxes:
        add_box(ax_flow, xy, wh, text, fc)
    for start, end in [((0.19, 0.50), (0.25, 0.50)), ((0.40, 0.50), (0.46, 0.50)), ((0.61, 0.50), (0.67, 0.50)), ((0.80, 0.50), (0.86, 0.50))]:
        add_arrow(ax_flow, start, end)
    ax_flow.text(
        0.50,
        0.16,
        "AEC expert score = w1*x1 + w2*x2 + ... + w128*x128 + b, where x values are standardized selected AEC positions.",
        ha="center",
        va="center",
        fontsize=10,
        color="#4C5865",
    )

    ax1 = fig.add_subplot(gs[1, 0])
    plot_mean_curves(ax1, d["test"]["a128"], d["test"]["y"].astype(int), "Mean a128 Shape")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 1])
    plot_mean_curves(ax2, d["test"]["crop"], d["test"]["y"].astype(int), "Mean Cropped Shape")

    ax3 = fig.add_subplot(gs[1, 2])
    plot_score_distribution(ax3, d["test"]["y"].astype(int), d["test_score"], d["auc_test"])

    ax4 = fig.add_subplot(gs[2, :2])
    plot_coef_map(ax4, d["coefs"])

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_axis_off()
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.text(0.0, 0.98, "D. How It Enters the Final Gate", ha="left", va="top", fontsize=12, fontweight="bold", color=DARK)
    formula = "Final score =\nclinical score +\n0.25 x boundary gate x\nAEC expert score"
    add_box(ax5, (0.05, 0.55), (0.90, 0.28), formula, "#EAF7F4", ec=TEAL)
    ax5.text(
        0.06,
        0.40,
        f"Selected {d['selected_count']}/256 features\nOOF AUC {d['auc_oof']:.3f} | External AUC {d['auc_test']:.3f}",
        ha="left",
        va="top",
        fontsize=9.2,
        color=DARK,
        fontweight="bold",
        linespacing=1.25,
    )
    ax5.text(
        0.06,
        0.24,
        "Key message:\nAEC alone is weak as a detector.\nIts value is a small shape-based\nmodifier near the clinical boundary.",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#4C5865",
        linespacing=1.28,
    )

    fig.suptitle("AEC Expert: Scanner-Free 256-Position Linear SVM from Direct AEC Curves", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.012,
        "Direct AEC features were patient-mean normalized and centered; coefficient map shows the refit linear SVM in standardized feature space.",
        fontsize=9,
        color="#58606A",
    )
    fig.savefig(OUT_DIR / "figure_6_aec_expert_method.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_6_aec_expert_method.svg", bbox_inches="tight")
    plt.close(fig)


def make_feature_matrix_thumbnail(d: dict) -> None:
    x = d["te_df"].to_numpy(dtype=float)
    y = d["test"]["y"].astype(int)
    order = np.argsort(d["test_score"])
    x_ord = x[order]
    y_ord = y[order]
    # Clip for display, preserving shape rather than outlier magnitude.
    vmax = np.nanpercentile(np.abs(x_ord), 98)
    vmax = max(vmax, 0.2)

    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    im = ax.imshow(np.clip(x_ord, -vmax, vmax), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axvline(127.5, color="#202832", lw=1.5)
    ax.set_title("256 Direct AEC Features Sorted by AEC Expert Score", loc="left", fontweight="bold")
    ax.set_xlabel("Feature position: a128 0-127 | crop 0-127")
    ax.set_ylabel("sdata patients sorted by SVM score")
    ax.set_xticks([0, 64, 120, 136, 192, 255], ["a128 0", "64", "127", "crop 0", "64", "127"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Centered normalized AEC")
    low_rows = np.flatnonzero(y_ord == 1)
    if len(low_rows):
        ax.scatter(np.full(len(low_rows), 260), low_rows, s=5, color=RED, clip_on=False, label="Low SMI")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.text(0.02, 0.015, "Red side ticks mark observed low-SMI patients; this panel is for intuition, not model performance.", fontsize=9, color="#58606A")
    fig.savefig(OUT_DIR / "figure_7_aec_feature_matrix_thumbnail.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_7_aec_feature_matrix_thumbnail.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    d = fit_final_aec_expert(train, test)
    make_method_figure(d)
    make_feature_matrix_thumbnail(d)
    print("AEC expert OOF AUC:", f"{d['auc_oof']:.6f}")
    print("AEC expert external AUC:", f"{d['auc_test']:.6f}")
    print("Selected features:", d["selected_count"])
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
