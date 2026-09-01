from __future__ import annotations

# docs/논문 materials&methods 초안.docx에 캡션만 있고 이미지가 없던 3개 항목(Table 2: AUC/DeLong 마스터
# 표, Table 3: Youden 임계값 기준 민감도/특이도/정확도 표, Figure 5: 연구설계 개략도)을 만드는 스크립트.
# table1_baseline_characteristics.py(표 렌더링 스타일: 진한 헤더 #161616 + 줄무늬 행)와
# figure_manuscript.py(박스/화살표 다이어그램 스타일: FancyBboxPatch/FancyArrowPatch, 흰 박스 + 최종
# 상태 초록 박스)의 관례를 그대로 따른다. 원자료는 이미 실행되어 있는
# outputs/step_disease_logistic/{auc_delta_summary.csv, logistic_regression_summary.csv}를 그대로 읽으며,
# 이 스크립트 자체는 모델을 재학습하지 않는다(step_disease_logistic.py를 먼저 실행해 두어야 함).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 ±/→ 등을 인코딩 못 해 print에서 죽는 것 방지

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGISTIC_DIR = PROJECT_ROOT / "outputs" / "step_disease_logistic"
TABLE_OUT_DIR = PROJECT_ROOT / "outputs" / "table"
FIGURE_OUT_DIR = PROJECT_ROOT / "outputs" / "figure"

FEATURE_ORDER = ["HTN", "DM", "CKD"]
FEATURE_LABELS = {"HTN": "Hypertension", "DM": "Diabetes mellitus", "CKD": "Chronic kidney disease"}
MODEL_ORDER = ["clinic4", "clinic4_meanmAs", "clinic4_meanmAs_aec", "clinic4_vatsat", "clinic4_vatsat_aec"]
MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_meanmAs": "clinic4 + mean mAs",
    "clinic4_meanmAs_aec": "clinic4 + mean mAs + AEC",
    "clinic4_vatsat": "clinic4 + VAT + SAT",
    "clinic4_vatsat_aec": "clinic4 + VAT + SAT + AEC",
}
COHORT_ORDER = ["internal", "external"]
COHORT_LABELS = {"internal": "Internal", "external": "External"}


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ======================================================================================
# Table 2: AUC / paired DeLong test master table (all nested comparisons, both families)
# ======================================================================================

def build_table2() -> pd.DataFrame:
    delta = pd.read_csv(LOGISTIC_DIR / "auc_delta_summary.csv")
    rows = []
    for feat in FEATURE_ORDER:
        for cohort in COHORT_ORDER:
            r = delta[(delta["feature"] == feat) & (delta["cohort"] == cohort)].iloc[0]
            rows.append({
                "Outcome": FEATURE_LABELS[feat],
                "Cohort": COHORT_LABELS[cohort],
                "Family": "A: +mean mAs",
                "AUC clinic4": f"{r['auc_clinic4']:.3f}",
                "AUC +covariate": f"{r['auc_clinic4_meanmAs']:.3f}",
                "AUC +covariate+AEC": f"{r['auc_clinic4_meanmAs_aec']:.3f}",
                "DeLong P (+covariate)": format_p(r["delong_p_clinic4_meanmAs_minus_clinic4"]),
                "DeLong P (+AEC)": format_p(r["delong_p_clinic4_meanmAs_aec_minus_clinic4_meanmAs"]),
            })
            rows.append({
                "Outcome": FEATURE_LABELS[feat],
                "Cohort": COHORT_LABELS[cohort],
                "Family": "B: +VAT+SAT",
                "AUC clinic4": f"{r['auc_clinic4']:.3f}",
                "AUC +covariate": f"{r['auc_clinic4_vatsat']:.3f}",
                "AUC +covariate+AEC": f"{r['auc_clinic4_vatsat_aec']:.3f}",
                "DeLong P (+covariate)": format_p(r["delong_p_clinic4_vatsat_minus_clinic4"]),
                "DeLong P (+AEC)": format_p(r["delong_p_clinic4_vatsat_aec_minus_clinic4_vatsat"]),
            })
    return pd.DataFrame(rows)


def run_table2() -> None:
    table = build_table2()
    stem = "table2_auc_delong_summary"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 3: sensitivity / specificity / accuracy at the internal-OOF Youden threshold
# ======================================================================================

def build_table3() -> pd.DataFrame:
    summary = pd.read_csv(LOGISTIC_DIR / "logistic_regression_summary.csv")
    rows = []
    for feat in FEATURE_ORDER:
        for model in MODEL_ORDER:
            for cohort in COHORT_ORDER:
                r = summary[(summary["feature"] == feat) & (summary["model"] == model)
                            & (summary["cohort"] == cohort)].iloc[0]
                rows.append({
                    "Outcome": FEATURE_LABELS[feat],
                    "Model": MODEL_LABELS[model],
                    "Cohort": COHORT_LABELS[cohort],
                    "n (n_pos)": f"{int(r['n'])} ({int(r['n_pos'])})",
                    "AUC": f"{r['auc']:.3f}",
                    "Sensitivity": f"{r['sensitivity']:.1%}",
                    "Specificity": f"{r['specificity']:.1%}",
                    "Accuracy": f"{r['accuracy']:.1%}",
                })
    return pd.DataFrame(rows)


def run_table3() -> None:
    table = build_table3()
    stem = "table3_sensitivity_specificity"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 4: continuous NRI / IDI for adding the AEC score on top of each covariate step
# (clinic4+mean mAs -> +AEC, clinic4+VAT+SAT -> +AEC only; the covariate-only increments
# are not the focus of the Results narrative and are left out of the manuscript table)
# ======================================================================================

AEC_STEP_PAIRS = {
    "A: +mean mAs -> +AEC": ("clinic4_meanmAs", "clinic4_meanmAs_aec"),
    "B: +VAT+SAT -> +AEC": ("clinic4_vatsat", "clinic4_vatsat_aec"),
}


def build_table4() -> pd.DataFrame:
    nri_idi = pd.read_csv(LOGISTIC_DIR / "nri_idi_summary.csv")
    rows = []
    for feat in FEATURE_ORDER:
        for cohort in COHORT_ORDER:
            for family, (baseline, extended) in AEC_STEP_PAIRS.items():
                r = nri_idi[(nri_idi["feature"] == feat) & (nri_idi["cohort"] == cohort)
                            & (nri_idi["baseline_model"] == baseline) & (nri_idi["extended_model"] == extended)].iloc[0]
                rows.append({
                    "Outcome": FEATURE_LABELS[feat],
                    "Cohort": COHORT_LABELS[cohort],
                    "Family": family,
                    "Continuous NRI": f"{r['nri']:.3f}",
                    "IDI": f"{r['idi']:.4f}",
                })
    return pd.DataFrame(rows)


def run_table4() -> None:
    table = build_table4()
    stem = "table4_nri_idi_aec_increment"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 5: scanner-vendor subgroup AUC for the fullest model (clinic4+VAT+SAT+AEC), the
# robustness check referenced in Results 3-3 (vendor subgroups reused from the frozen
# external/internal model, no re-training per vendor)
# ======================================================================================

VENDOR_ORDER = ["Siemens", "GE", "Philips", "Other"]


MIN_N_POS_VENDOR = 5  # peer review M6: n_pos<5짜리 AUC 셀은 해석 불가능에 가까워 억제


def build_table5() -> pd.DataFrame:
    scanner = pd.read_csv(PROJECT_ROOT / "outputs" / "step_disease_scanner" / "scanner_subgroup_auc.csv")
    scanner = scanner[scanner["model"] == "clinic4_vatsat_aec"]
    rows = []
    for feat in FEATURE_ORDER:
        for cohort in COHORT_ORDER:
            sub = scanner[(scanner["feature"] == feat) & (scanner["cohort"] == cohort)]
            for vendor in VENDOR_ORDER:
                v = sub[sub["scanner"] == vendor]
                if v.empty:
                    continue
                r = v.iloc[0]
                suppressed = int(r["n_pos"]) < MIN_N_POS_VENDOR
                rows.append({
                    "Outcome": FEATURE_LABELS[feat],
                    "Cohort": COHORT_LABELS[cohort],
                    "Vendor": vendor,
                    "n (n_pos)": f"{int(r['n'])} ({int(r['n_pos'])})",
                    "AUC": "NR†" if suppressed else f"{r['auc']:.3f}",
                })
    return pd.DataFrame(rows)


def run_table5() -> None:
    table = build_table5()
    stem = "table5_scanner_vendor_auc"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 6: calibration (slope, intercept, Brier score) for all 30 model/outcome/cohort
# combinations — peer review M5 (no calibration was reported in the original draft)
# ======================================================================================

SUPPLEMENTAL_DIR = PROJECT_ROOT / "outputs" / "manuscript_supplemental"


def build_table6() -> pd.DataFrame:
    calib = pd.read_csv(SUPPLEMENTAL_DIR / "calibration_summary.csv")
    rows = []
    for feat in FEATURE_ORDER:
        for model in MODEL_ORDER:
            for cohort in COHORT_ORDER:
                r = calib[(calib["feature"] == feat) & (calib["model"] == model)
                          & (calib["cohort"] == cohort.lower())].iloc[0]
                rows.append({
                    "Outcome": FEATURE_LABELS[feat],
                    "Model": MODEL_LABELS[model],
                    "Cohort": COHORT_LABELS[cohort],
                    "Calibration slope": f"{r['calibration_slope']:.3f}",
                    "Calibration intercept": f"{r['calibration_intercept']:.3f}",
                    "Brier score": f"{r['brier_score']:.3f}",
                })
    return pd.DataFrame(rows)


def run_table6() -> None:
    table = build_table6()
    stem = "table6_calibration"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 7: Benjamini-Hochberg FDR sensitivity analysis for the 24 paired DeLong tests
# (peer review M2 — Bonferroni across correlated tests may be overly conservative;
# BH-FDR is reported here as the prespecified sensitivity check)
# ======================================================================================

def build_table7() -> pd.DataFrame:
    fdr = pd.read_csv(SUPPLEMENTAL_DIR / "delong_bh_fdr.csv")
    fdr["feature"] = pd.Categorical(fdr["feature"], categories=FEATURE_ORDER, ordered=True)
    fdr = fdr.sort_values(["feature", "cohort", "baseline_model", "extended_model"])
    rows = []
    for _, r in fdr.iterrows():
        rows.append({
            "Outcome": FEATURE_LABELS[r["feature"]],
            "Cohort": COHORT_LABELS[r["cohort"]],
            "Comparison": f"{MODEL_LABELS[r['baseline_model']]} -> {MODEL_LABELS[r['extended_model']]}",
            "AUC diff": f"{r['auc_diff']:.4f}",
            "DeLong P": format_p(r["p_value"]),
            "BH-FDR q": format_p(r["bh_q"]),
        })
    return pd.DataFrame(rows)


def run_table7() -> None:
    table = build_table7()
    stem = "table7_delong_bh_fdr"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 8: AEC component-outcome association before and after residualizing each FPCA
# component on age/sex/height/weight/VAT/SAT (peer review M7/M3 — tests directly whether
# the raw AEC-outcome correlation survives adjustment for known adiposity/demographic
# predictors, rather than relying on the AEC-vs-VAT/SAT correlation alone)
# ======================================================================================

def build_table8() -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    resid = pd.read_csv(SUPPLEMENTAL_DIR / "aec_residualized_partial_association.csv")
    _, bh_q, _, _ = multipletests(resid["p_residualized"], alpha=0.05, method="fdr_bh")
    resid["bh_q"] = bh_q
    resid["outcome"] = pd.Categorical(resid["outcome"], categories=FEATURE_ORDER, ordered=True)
    resid = resid.sort_values(["outcome", "cohort", "component"])
    rows = []
    for _, r in resid.iterrows():
        rows.append({
            "Outcome": FEATURE_LABELS[r["outcome"]],
            "Cohort": r["cohort"].capitalize(),
            "AEC component": r["component"],
            "Raw r (vs outcome)": f"{r['r_raw']:.3f}",
            "Raw P": format_p(r["p_raw"]),
            "Residualized r*": f"{r['r_residualized_on_demog_vat_sat']:.3f}",
            "Residualized P": format_p(r["p_residualized"]),
            "Residualized BH-FDR q": format_p(r["bh_q"]),
        })
    return pd.DataFrame(rows)


def run_table8() -> None:
    table = build_table8()
    stem = "table8_aec_residualized_association"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Table 9: fitted logistic-regression coefficients (standardized-predictor scale) for the
# fullest model in each family (clinic4+mean_mAs+AEC, clinic4+VAT+SAT+AEC), all 3 outcomes
# (peer review R3, TRIPOD item 15a — the full model was never reported, only performance
# metrics; these are the exact models fit on the entire internal cohort and frozen for
# external scoring in step_disease_logistic.py, coef_sheets -> *_logistic_coefficients.xlsx)
# ======================================================================================

TERM_LABELS = {
    "sex_M": "Sex (male=1)", "age": "Age (z-score)", "height": "Height (z-score)",
    "weight": "Weight (z-score)", "mean_mAs": "mean_mAs (z-score)",
    "VAT(내장지방)_SUM": "VAT (z-score)", "SAT(피하지방)_SUM": "SAT (z-score)",
    "fpca_pc1": "AEC FPCA PC1 (z-score)", "fpca_pc2": "AEC FPCA PC2 (z-score)",
    "fpca_pc3": "AEC FPCA PC3 (z-score)", "intercept": "Intercept",
}
TERM_ORDER = ["intercept", "sex_M", "age", "height", "weight", "mean_mAs",
              "VAT(내장지방)_SUM", "SAT(피하지방)_SUM", "fpca_pc1", "fpca_pc2", "fpca_pc3"]


def build_table9() -> pd.DataFrame:
    rows = []
    for feat in FEATURE_ORDER:
        slug = FEATURES_SLUG[feat]
        xl = pd.ExcelFile(LOGISTIC_DIR / slug / f"{slug}_logistic_coefficients.xlsx")
        for model_name in ["clinic4_meanmAs_aec", "clinic4_vatsat_aec"]:
            coef = xl.parse(model_name).set_index("term")
            for term in TERM_ORDER:
                if term not in coef.index:
                    continue
                r = coef.loc[term]
                rows.append({
                    "Outcome": FEATURE_LABELS[feat],
                    "Model": MODEL_LABELS[model_name],
                    "Term": TERM_LABELS[term],
                    "Coefficient (log-odds)": f"{r['coefficient']:.3f}",
                    "Odds ratio": f"{r['odds_ratio']:.3f}" if term != "intercept" else "—",
                })
    return pd.DataFrame(rows)


FEATURES_SLUG = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}


def run_table9() -> None:
    table = build_table9()
    stem = "table9_fullest_model_coefficients"
    TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))


# ======================================================================================
# Figure 5: calibration (reliability) plot for the fullest model (clinic4+VAT+SAT+AEC),
# internal vs external, one line per outcome — visualizes the Table 6 calibration slopes
# ======================================================================================

def plot_calibration(out_path: Path, n_bins: int = 5) -> None:
    preds = pd.read_csv(PROJECT_ROOT / "outputs" / "step_disease_logistic" / "predictions.csv")
    preds = preds[preds["model"] == "clinic4_vatsat_aec"]
    colors = {"HTN": "#2a78d6", "DM": "#1baf7a", "CKD": "#e2622e"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Ideal")
        for feat in FEATURE_ORDER:
            g = preds[(preds["feature"] == feat) & (preds["cohort"] == cohort)]
            bins = pd.qcut(g["score"], q=n_bins, duplicates="drop")
            grp = g.groupby(bins, observed=True).agg(mean_pred=("score", "mean"), obs_freq=("y", "mean"))
            ax.plot(grp["mean_pred"], grp["obs_freq"], marker="o", color=colors[feat],
                    label=FEATURE_LABELS[feat])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (per quintile)", fontsize=14)
        ax.set_ylabel("Observed frequency", fontsize=14)
        ax.set_title(COHORT_LABELS[cohort], fontweight="bold", color="#161616", fontsize=16.8)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=12.6, frameon=False)
    fig.tight_layout()
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot to {out_path}")


# ======================================================================================
# Figure 6: forest plot of the AEC FPCA component-outcome correlations (raw vs
# residualized on age/sex/height/weight/VAT/SAT), replacing Table 4/8 in the manuscript
# body — 논문 docx 정리(2026-08-28): 본문에 table+figure가 같이 있으면 중복이므로 figure만 본문에
# 남기고 table은 appendix로 옮기는데, Table 4(FPCA 성분-outcome 상관)는 원래 figure가 없던 table-only
# 항목이라 사용자 지시("table만 있을 때 figure 대체가 개선이면 생성")에 따라 새로 만든다. Table
# 4/8의 핵심 메시지(raw r이 residualize 후 대부분 유의성을 잃는 감쇠)를 forest plot으로 시각화하면
# 19행짜리 숫자표보다 한눈에 들어온다. 95% CI는 Table 4 캡션과 동일하게 Fisher z 변환으로 계산
# (df = n - 3 - k, k=6 covariates); 이 df로 계산한 CI가 기존 docx Table 4의 CI 문자열과 정확히
# 일치함을 별도로 검증했다(예: internal PC1 HTN r=0.042 -> CI -0.013 to 0.097).
# ======================================================================================

N_COHORT = {"internal": 1259, "external": 1123}  # Table 1과 동일
N_COVARIATES_RESIDUALIZED = 6  # age, sex, height, weight, VAT, SAT
COMPONENT_ORDER = ["PC1", "PC2", "PC3"]


def fisher_ci(r: float, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
    df = n - 3 - k
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(df)
    z_crit = 1.959963984540054
    lo, hi = z - z_crit * se, z + z_crit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def plot_fpca_forest(out_path: Path) -> None:
    from statsmodels.stats.multitest import multipletests

    resid = pd.read_csv(SUPPLEMENTAL_DIR / "aec_residualized_partial_association.csv")
    _, bh_q, _, _ = multipletests(resid["p_residualized"], alpha=0.05, method="fdr_bh")
    resid["bh_q"] = bh_q
    resid["ci_lo"], resid["ci_hi"] = zip(*resid.apply(
        lambda r: fisher_ci(r["r_residualized_on_demog_vat_sat"], N_COHORT[r["cohort"]],
                             N_COVARIATES_RESIDUALIZED), axis=1))

    colors = {"HTN": "#2a78d6", "DM": "#1baf7a", "CKD": "#e2622e"}

    rows = [(feat, comp) for feat in FEATURE_ORDER for comp in COMPONENT_ORDER]
    y_pos = {rc: i for i, rc in enumerate(reversed(rows))}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5), sharey=True)
    for ax, cohort in zip(axes, ["internal", "external"]):
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
        sub = resid[resid["cohort"] == cohort]
        for _, r in sub.iterrows():
            y = y_pos[(r["outcome"], r["component"])]
            color = colors[r["outcome"]]
            ax.plot(r["r_raw"], y, marker="o", markersize=5, markerfacecolor="white",
                     markeredgecolor="#9a9a9a", zorder=2)
            ax.errorbar(r["r_residualized_on_demog_vat_sat"], y,
                         xerr=[[r["r_residualized_on_demog_vat_sat"] - r["ci_lo"]],
                               [r["ci_hi"] - r["r_residualized_on_demog_vat_sat"]]],
                         fmt="o", markersize=7, color=color, ecolor=color, elinewidth=1.6,
                         capsize=3, zorder=3)
            if r["bh_q"] < 0.05:
                marker = "**"
            elif r["p_residualized"] < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                ax.text(r["ci_hi"] + 0.012, y, marker, va="center", ha="left", fontsize=15.4,
                         color=color, fontweight="bold")
        ax.set_yticks([y_pos[rc] for rc in rows])
        ax.set_yticklabels([f"{FEATURE_LABELS[feat]} – {comp}" for feat, comp in rows], fontsize=13.3)
        ax.set_xlim(-0.4, 0.4)
        ax.set_xlabel("Pearson r (vs outcome)", fontsize=14)
        ax.set_title(COHORT_LABELS[cohort], fontweight="bold", color="#161616", fontsize=16.8)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="x", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#9a9a9a",
                    markersize=6, label="Raw r"),
        plt.Line2D([0], [0], marker="o", color="#161616", markerfacecolor="#161616", markersize=7,
                    label="Residualized r (95% CI)†"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=11.9, frameon=False,
                    bbox_to_anchor=(0.0, -0.14))
    fig.text(0.5, -0.02,
              "* residualized P<.05   ** residualized BH-FDR q<.05   †residualized on age, sex, height, "
              "weight, VAT, SAT",
              ha="center", fontsize=11.2, color="#3a3a3a")
    fig.tight_layout()
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA forest plot to {out_path}")


# ======================================================================================
# Figure 5: study-design schematic (internal 5-fold CV + external frozen validation,
# the two nested model families, and the prespecified AEC score feeding both)
# ======================================================================================

BORDER = "#161616"
WHITE = "white"
ACCENT = "#eef3fb"
FINAL_GREEN = "#d9ead3"


def box(ax, cx, cy, w, h, text, facecolor, fontsize=15, fontweight="normal", edgecolor=BORDER, lw=1.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight, zorder=3,
             linespacing=1.35)


def arrow(ax, xy_from, xy_to, lw=1.6, style="-|>", color=BORDER):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=14, linewidth=lw,
                                  color=color, zorder=1))

def main() -> None:
    run_table2()
    run_table3()
    run_table4()
    run_table5()
    run_table6()
    run_table7()
    run_table8()
    run_table9()
    plot_calibration(FIGURE_OUT_DIR / "fig5_calibration.png")
    plot_fpca_forest(FIGURE_OUT_DIR / "fig6_fpca_forest.png")


if __name__ == "__main__":
    main()
