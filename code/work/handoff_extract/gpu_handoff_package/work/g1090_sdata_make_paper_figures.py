from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import load_dataset, threshold_youden  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import (  # noqa: E402
    clinical_gate_base,
    expert_oof_test,
    feature_sets_no_scanner,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 20260626
CLIN = "#395D9C"
AEC = "#008C7A"
RED = "#B33A3A"
GRAY = "#7C8794"
LIGHT = "#EEF2F6"


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def calibrate_prob(train_score: np.ndarray, y_train: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000)
    model.fit(train_score.reshape(-1, 1), y_train)
    return (
        model.predict_proba(train_score.reshape(-1, 1))[:, 1],
        model.predict_proba(test_score.reshape(-1, 1))[:, 1],
    )


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000)
    model.fit(logit_p, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def quantile_calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y.astype(int), "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    out = (
        df.groupby("bin", observed=True)
        .agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"), events=("y", "sum"))
        .reset_index(drop=True)
    )
    return out


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    pred = p >= pt
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return float(tp / n - fp / n * pt / (1 - pt))


def net_benefit_binary(y: np.ndarray, pred: np.ndarray, pt: float) -> float:
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return float(tp / n - fp / n * pt / (1 - pt))


def find_deescalation_threshold(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray, clinical_th: float, cap: float) -> float:
    mask = clinical_score >= clinical_th
    scores = np.sort(np.unique(aec_score[mask]))
    best = None
    for th in scores:
        low = mask & (aec_score <= th)
        n = int(low.sum())
        if n == 0:
            continue
        rate = float(y[low].mean())
        if rate <= cap:
            best = th
    if best is None:
        return float(scores.min())
    return float(best)


def find_high_priority_threshold(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray, clinical_th: float, target_ppv: float) -> float:
    mask = clinical_score >= clinical_th
    scores = np.sort(np.unique(aec_score[mask]))
    best = None
    for th in scores:
        high = mask & (aec_score >= th)
        n = int(high.sum())
        if n == 0:
            continue
        ppv = float(y[high].mean())
        if ppv >= target_ppv:
            # Keep the broadest group that reaches the target.
            best = th
            break
    if best is None:
        return float(scores.max())
    return float(best)


def reconstruct() -> dict:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)

    clinical_tr, clinical_te, _, _ = clinical_oof_test(train, test, folds)

    feats = feature_sets_no_scanner(train, test)
    xtr = feats["direct_curve"][0].to_numpy(dtype=float)
    xte = feats["direct_curve"][1].to_numpy(dtype=float)
    aec_tr, aec_te = expert_oof_test(xtr, xte, ytr, folds, "linsvm_C0.2", 128)

    sex_tr = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_te = test["meta"]["PatientSex"].astype(str).to_numpy()
    gates = clinical_gate_base(ytr, clinical_tr, clinical_te, sex_tr, sex_te)
    gtr, gte = gates["female_boundary"]
    clinical_tr_z, clinical_te_z, _, _ = zfit_apply(clinical_tr, clinical_te)
    aec_tr_z, aec_te_z, _, _ = zfit_apply(aec_tr, aec_te)
    gate_tr = clinical_tr_z + 0.25 * gtr * aec_tr_z
    gate_te = clinical_te_z + 0.25 * gte * aec_te_z

    clinical_th = threshold_youden(ytr, clinical_tr)
    gate_th = threshold_youden(ytr, gate_tr)

    clinical_prob_tr, clinical_prob_te = calibrate_prob(clinical_tr, ytr, clinical_te)
    gate_prob_tr, gate_prob_te = calibrate_prob(gate_tr, ytr, gate_te)

    return {
        "train": train,
        "test": test,
        "ytr": ytr,
        "yte": yte,
        "clinical_tr": clinical_tr,
        "clinical_te": clinical_te,
        "gate_tr": gate_tr,
        "gate_te": gate_te,
        "clinical_th": clinical_th,
        "gate_th": gate_th,
        "clinical_prob_tr": clinical_prob_tr,
        "clinical_prob_te": clinical_prob_te,
        "gate_prob_tr": gate_prob_tr,
        "gate_prob_te": gate_prob_te,
    }


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def make_calibration_figure(d: dict) -> pd.DataFrame:
    y = d["yte"]
    rows = []
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.plot([0, 1], [0, 1], color="#AAB3BD", lw=1.5, ls="--", label="Ideal")
    for label, prob, color in [
        ("Clinical", d["clinical_prob_te"], CLIN),
        ("Clinical + AEC gate", d["gate_prob_te"], AEC),
    ]:
        bins = quantile_calibration(y, prob)
        slope, intercept = calibration_slope_intercept(y, prob)
        brier = brier_score_loss(y, prob)
        auc = auc_or_nan(y, prob)
        bins.to_csv(OUT_DIR / f"calibration_bins_{label.lower().replace(' ', '_').replace('+', 'plus')}.csv", index=False)
        rows.append({"model": label, "auc": auc, "brier": brier, "calibration_slope": slope, "calibration_intercept": intercept})
        ax.plot(bins["mean_pred"], bins["obs_rate"], marker="o", lw=2.2, color=color, label=f"{label} (Brier {brier:.3f})")
        ax.scatter(bins["mean_pred"], bins["obs_rate"], s=np.clip(bins["n"], 35, 140), color=color, alpha=0.20, edgecolor="none")
    ax.set_xlim(0, 0.70)
    ax.set_ylim(0, 0.70)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed low-SMI rate")
    ax.set_title("External Calibration in sdata", loc="left", fontweight="bold")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.02, 0.01, "Platt calibration fitted on g1090 out-of-fold scores; deciles shown in sdata.", fontsize=8.5, color="#58606A")
    savefig(fig, "figure_1_calibration")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    return summary


def make_dca_figure(d: dict) -> pd.DataFrame:
    y = d["yte"]
    pts = np.round(np.arange(0.02, 0.501, 0.01), 4)
    prev = float(y.mean())
    clin_pred = d["clinical_te"] >= d["clinical_th"]
    gate_pred = d["gate_te"] >= d["gate_th"]
    rows = []
    for pt in pts:
        nb_clin_risk = net_benefit(y, d["clinical_prob_te"], pt)
        nb_gate_risk = net_benefit(y, d["gate_prob_te"], pt)
        nb_clin_rule = net_benefit_binary(y, clin_pred, pt)
        nb_gate_rule = net_benefit_binary(y, gate_pred, pt)
        nb_all = prev - (1 - prev) * pt / (1 - pt)
        rows.append(
            {
                "threshold_probability": pt,
                "NB_clinical_risk": nb_clin_risk,
                "NB_aec_gate_risk": nb_gate_risk,
                "NB_clinical_fixed_rule": nb_clin_rule,
                "NB_aec_gate_fixed_rule": nb_gate_rule,
                "NB_all": nb_all,
                "NB_none": 0.0,
                "delta_NB_fixed_rule": nb_gate_rule - nb_clin_rule,
                "net_interventions_avoided_per_100_fixed_rule": (nb_gate_rule - nb_clin_rule) / (pt / (1 - pt)) * 100,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "decision_curve_values.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), gridspec_kw={"width_ratios": [1.05, 0.95]})
    ax = axes[0]
    ax.plot(out["threshold_probability"], out["NB_clinical_risk"], color=CLIN, lw=2.4, label="Clinical risk")
    ax.plot(out["threshold_probability"], out["NB_aec_gate_risk"], color=AEC, lw=2.4, label="Clinical + AEC gate risk")
    ax.plot(out["threshold_probability"], out["NB_all"], color="#7F8790", lw=1.5, ls=":", label="Treat all")
    ax.axhline(0, color="#30343A", lw=1.0, label="Treat none")
    ax.set_xlim(0.02, 0.50)
    ax.set_ylim(-0.08, 0.16)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("A. Risk-based Decision Curve", loc="left", fontweight="bold")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sel = out[(out["threshold_probability"] >= 0.05) & (out["threshold_probability"] <= 0.40)]
    ax.plot(sel["threshold_probability"], sel["net_interventions_avoided_per_100_fixed_rule"], color=AEC, lw=2.8)
    ax.fill_between(
        sel["threshold_probability"].to_numpy(),
        0,
        sel["net_interventions_avoided_per_100_fixed_rule"].to_numpy(),
        color=AEC,
        alpha=0.15,
    )
    ax.axhline(0, color="#30343A", lw=1.0)
    ax.set_xlim(0.05, 0.40)
    ax.set_ylim(0, max(12, sel["net_interventions_avoided_per_100_fixed_rule"].max() + 1))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net interventions avoided per 100")
    ax.set_title("B. Fixed Rule Gain vs Clinical", loc="left", fontweight="bold")
    ax.grid(True, color="#D8DEE6", lw=0.7, alpha=0.8)
    fig.text(0.02, 0.01, "Panel B uses train-derived binary decision thresholds for clinical and AEC-gated rules.", fontsize=8.5, color="#58606A")
    savefig(fig, "figure_2_decision_curve")
    return out


def make_reclassification_figure(d: dict) -> pd.DataFrame:
    y = d["yte"]
    clinical_pos = d["clinical_te"] >= d["clinical_th"]
    gate_pos = d["gate_te"] >= d["gate_th"]
    groups = [
        ("Clinical+ / AEC+", clinical_pos & gate_pos),
        ("Clinical+ / AEC-", clinical_pos & ~gate_pos),
        ("Clinical- / AEC+", ~clinical_pos & gate_pos),
        ("Clinical- / AEC-", ~clinical_pos & ~gate_pos),
    ]
    rows = []
    for name, mask in groups:
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "events": int(y[mask].sum()),
                "non_events": int(mask.sum() - y[mask].sum()),
                "prevalence": float(y[mask].mean()) if mask.sum() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reclassification_2x2.csv", index=False)

    mat_n = np.array([[rows[0]["n"], rows[1]["n"]], [rows[2]["n"], rows[3]["n"]]])
    mat_e = np.array([[rows[0]["events"], rows[1]["events"]], [rows[2]["events"], rows[3]["events"]]])
    mat_p = mat_e / mat_n

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    im = ax.imshow(mat_p, cmap="YlGnBu", vmin=0, vmax=max(0.35, float(np.nanmax(mat_p))))
    ax.set_xticks([0, 1], ["AEC positive", "AEC negative"])
    ax.set_yticks([0, 1], ["Clinical positive", "Clinical negative"])
    ax.set_title("External Reclassification at Train-Derived Thresholds", loc="left", fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if mat_p[i, j] > 0.20 else "#17202A"
            ax.text(
                j,
                i,
                f"n={mat_n[i, j]}\nlow SMI={mat_e[i, j]}\n{mat_p[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold" if (i, j) in [(0, 0), (0, 1)] else "normal",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Observed low-SMI prevalence")
    ax.tick_params(length=0)
    fig.text(0.02, 0.01, "The key low-risk cell is clinical-positive/AEC-negative: 103 patients, 4 low-SMI events.", fontsize=8.5, color="#58606A")
    savefig(fig, "figure_3_reclassification_heatmap")
    return out


def bootstrap_delta_ci(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, th_a: float, th_b: float, metric: str, n_boot: int = 2500) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        ma = metric_at_threshold(yi, score_a[idx], th_a)
        mb = metric_at_threshold(yi, score_b[idx], th_b)
        if metric == "auc":
            if len(np.unique(yi)) < 2:
                continue
            delta = auc_rank(yi, score_b[idx]) - auc_rank(yi, score_a[idx])
        else:
            delta = mb[metric] - ma[metric]
        if np.isfinite(delta):
            vals.append(float(delta))
    arr = np.asarray(vals)
    point = arr.mean() if metric == "auc" else metric_at_threshold(y, score_b, th_b)[metric] - metric_at_threshold(y, score_a, th_a)[metric]
    return float(point), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def make_subgroup_forest(d: dict) -> pd.DataFrame:
    y = d["yte"]
    meta = d["test"]["meta"]
    scanner = meta["Manufacturer"].astype(str).to_numpy()
    sex = meta["PatientSex"].astype(str).to_numpy()
    groups: list[tuple[str, np.ndarray]] = [
        ("Overall", np.ones(len(y), dtype=bool)),
        ("Male", sex == "M"),
        ("Female", sex == "F"),
    ]
    for name, n in pd.Series(scanner).value_counts().items():
        if n >= 40:
            groups.append((str(name), scanner == str(name)))

    rows = []
    for name, mask in groups:
        ya = y[mask]
        ca = d["clinical_te"][mask]
        ga = d["gate_te"][mask]
        clin_m = metric_at_threshold(ya, ca, d["clinical_th"])
        gate_m = metric_at_threshold(ya, ga, d["gate_th"])
        fp_removed = clin_m["fp"] - gate_m["fp"]
        delta_spec, lo, hi = bootstrap_delta_ci(ya, ca, ga, d["clinical_th"], d["gate_th"], "specificity")
        rows.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "events": int(ya.sum()),
                "auc_clinical": auc_or_nan(ya, ca),
                "auc_aec_gate": auc_or_nan(ya, ga),
                "delta_auc": auc_or_nan(ya, ga) - auc_or_nan(ya, ca),
                "delta_specificity": delta_spec,
                "delta_specificity_ci_low": lo,
                "delta_specificity_ci_high": hi,
                "delta_sensitivity": gate_m["sensitivity"] - clin_m["sensitivity"],
                "delta_ppv": gate_m["ppv"] - clin_m["ppv"],
                "fp_removed": fp_removed,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "subgroup_forest_delta_specificity.csv", index=False)

    plot_df = out.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(9.8, 6.5))
    ax.axvline(0, color="#30343A", lw=1)
    ax.errorbar(
        plot_df["delta_specificity"],
        y_pos,
        xerr=[
            plot_df["delta_specificity"] - plot_df["delta_specificity_ci_low"],
            plot_df["delta_specificity_ci_high"] - plot_df["delta_specificity"],
        ],
        fmt="o",
        color=AEC,
        ecolor=AEC,
        elinewidth=1.6,
        capsize=3,
        markersize=6,
    )
    ax.set_yticks(y_pos, plot_df["group"])
    ax.set_xlabel("Specificity gain of AEC gate vs clinical")
    ax.set_title("Subgroup Consistency of False-Positive Reduction", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#D8DEE6", lw=0.7, alpha=0.8)
    xmax = max(0.50, float(plot_df["delta_specificity_ci_high"].max() + 0.16))
    ax.set_xlim(-0.06, xmax)
    for i, row in plot_df.iterrows():
        ax.text(
            xmax - 0.01,
            i,
            f"n={int(row['n'])}, events={int(row['events'])}, FP -{int(row['fp_removed'])}",
            ha="right",
            va="center",
            fontsize=8.5,
            color="#58606A",
        )
    fig.text(0.02, 0.01, "Points are specificity differences; bars are bootstrap 95% intervals within each subgroup.", fontsize=8.5, color="#58606A")
    savefig(fig, "figure_4_subgroup_forest_specificity")
    return out


def make_waterfall(d: dict) -> pd.DataFrame:
    y = d["yte"]
    clinical_pos = d["clinical_te"] >= d["clinical_th"]
    score = d["gate_te"]
    triage_path = ROOT / "work" / "analysis_g1090_sdata_more_more" / "triage_deescalation_highpriority.csv"
    if triage_path.exists():
        triage = pd.read_csv(triage_path)
        deesc_th = float(
            triage.loc[triage["band"].eq("deescalate_train_event_rate_cap_0.100"), "threshold"].iloc[0]
        )
        high_th = float(
            triage.loc[triage["band"].eq("high_priority_train_ppv_target_0.45"), "threshold"].iloc[0]
        )
    else:
        deesc_th = find_deescalation_threshold(d["ytr"], d["clinical_tr"], d["gate_tr"], d["clinical_th"], cap=0.10)
        high_th = find_high_priority_threshold(d["ytr"], d["clinical_tr"], d["gate_tr"], d["clinical_th"], target_ppv=0.45)

    df = pd.DataFrame(
        {
            "y": y,
            "clinical_score": d["clinical_te"],
            "aec_gate_score": score,
            "clinical_positive": clinical_pos,
            "aec_gate_positive": score >= d["gate_th"],
            "deescalation_cap10": clinical_pos & (score <= deesc_th),
            "high_priority_ppv45": clinical_pos & (score >= high_th),
        }
    )
    cpos = df[df["clinical_positive"]].copy().sort_values("aec_gate_score").reset_index(drop=True)
    cpos["rank"] = np.arange(1, len(cpos) + 1)
    cpos.to_csv(OUT_DIR / "clinical_positive_waterfall_patient_scores.csv", index=False)

    colors = np.where(cpos["y"].to_numpy() == 1, RED, "#B9C1CB")
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(cpos["rank"], cpos["aec_gate_score"], color=colors, width=1.0, edgecolor="none")
    ax.axhline(d["gate_th"], color=AEC, lw=2.0, label="AEC gate Youden threshold")
    ax.axhline(deesc_th, color="#38414A", lw=1.8, ls="--", label="Train low-risk cap 10%")
    ax.axhline(high_th, color=RED, lw=1.8, ls="--", label="Train high-priority PPV >=45%")
    low_n = int((cpos["aec_gate_score"] < d["gate_th"]).sum())
    low_e = int(cpos.loc[cpos["aec_gate_score"] < d["gate_th"], "y"].sum())
    de_n = int(cpos["deescalation_cap10"].sum())
    de_e = int(cpos.loc[cpos["deescalation_cap10"], "y"].sum())
    hi_n = int(cpos["high_priority_ppv45"].sum())
    hi_e = int(cpos.loc[cpos["high_priority_ppv45"], "y"].sum())
    ax.add_patch(Rectangle((0.5, ax.get_ylim()[0]), max(low_n, 1), ax.get_ylim()[1] - ax.get_ylim()[0], color=AEC, alpha=0.07, zorder=-1))
    ax.add_patch(Rectangle((0.5, ax.get_ylim()[0]), max(de_n, 1), ax.get_ylim()[1] - ax.get_ylim()[0], color="#38414A", alpha=0.06, zorder=-1))
    ax.add_patch(Rectangle((len(cpos) - hi_n + 0.5, ax.get_ylim()[0]), max(hi_n, 1), ax.get_ylim()[1] - ax.get_ylim()[0], color=RED, alpha=0.06, zorder=-1))
    ax.text(5, ax.get_ylim()[1] * 0.92, f"Low-risk by AEC gate\nn={low_n}, events={low_e}", ha="left", va="top", fontsize=10, color=AEC, fontweight="bold")
    ax.text(5, ax.get_ylim()[1] * 0.76, f"Strict de-escalation\nn={de_n}, events={de_e}", ha="left", va="top", fontsize=9, color="#38414A")
    ax.text(len(cpos) - hi_n + 5, ax.get_ylim()[1] * 0.92, f"High-priority\nn={hi_n}, events={hi_e}\nPPV={hi_e / hi_n:.1%}", ha="left", va="top", fontsize=10, color=RED, fontweight="bold")
    ax.set_xlim(0, len(cpos) + 1)
    ax.set_xlabel("Clinical-positive patients sorted by AEC-gated score")
    ax.set_ylabel("AEC-gated score")
    ax.set_title("Clinical-Positive Waterfall: AEC Reorders Risk", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#D8DEE6", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=9)
    handles = [
        Rectangle((0, 0), 1, 1, color=RED),
        Rectangle((0, 0), 1, 1, color="#B9C1CB"),
    ]
    ax.legend(handles + ax.get_legend_handles_labels()[0], ["Low SMI", "Not low SMI"] + ax.get_legend_handles_labels()[1], frameon=False, ncol=5, loc="lower right", fontsize=8.5)
    fig.text(0.02, 0.01, "Bars are sdata patients already flagged by the clinical model; red bars are observed low-SMI cases.", fontsize=8.5, color="#58606A")
    savefig(fig, "figure_5_clinical_positive_waterfall")
    return cpos


def make_summary_table(d: dict) -> pd.DataFrame:
    y = d["yte"]
    rows = []
    for name, score, prob, th in [
        ("Clinical", d["clinical_te"], d["clinical_prob_te"], d["clinical_th"]),
        ("Clinical + AEC gate", d["gate_te"], d["gate_prob_te"], d["gate_th"]),
    ]:
        m = metric_at_threshold(y, score, th)
        rows.append(
            {
                "model": name,
                "auc": auc_or_nan(y, score),
                "brier": brier_score_loss(y, prob),
                "threshold": th,
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "ppv": m["ppv"],
                "npv": m["npv"],
                "tp": m["tp"],
                "fn": m["fn"],
                "tn": m["tn"],
                "fp": m["fp"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "paper_figure_model_summary.csv", index=False)
    return out


def main() -> None:
    d = reconstruct()
    summary = make_summary_table(d)
    calibration = make_calibration_figure(d)
    dca = make_dca_figure(d)
    reclass = make_reclassification_figure(d)
    forest = make_subgroup_forest(d)
    waterfall = make_waterfall(d)

    key = {
        "summary": summary.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "reclassification": reclass.to_dict(orient="records"),
        "dca_010_040_fixed_net_interventions_per_100_min": float(
            dca[(dca["threshold_probability"] >= 0.10) & (dca["threshold_probability"] <= 0.40)][
                "net_interventions_avoided_per_100_fixed_rule"
            ].min()
        ),
        "dca_010_040_fixed_net_interventions_per_100_max": float(
            dca[(dca["threshold_probability"] >= 0.10) & (dca["threshold_probability"] <= 0.40)][
                "net_interventions_avoided_per_100_fixed_rule"
            ].max()
        ),
        "forest": forest.to_dict(orient="records"),
        "waterfall_n": int(len(waterfall)),
    }
    pd.Series({"output_dir": str(OUT_DIR)}).to_csv(OUT_DIR / "output_location.txt", header=False)
    print(pd.DataFrame(key["summary"]).to_string(index=False))
    print("\nCalibration")
    print(pd.DataFrame(key["calibration"]).to_string(index=False))
    print("\nReclassification")
    print(pd.DataFrame(key["reclassification"]).to_string(index=False))
    print("\nDCA fixed-rule NIA per 100 from pt 0.10-0.40:", key["dca_010_040_fixed_net_interventions_per_100_min"], key["dca_010_040_fixed_net_interventions_per_100_max"])
    print("\nSaved figures to", OUT_DIR)


if __name__ == "__main__":
    main()
