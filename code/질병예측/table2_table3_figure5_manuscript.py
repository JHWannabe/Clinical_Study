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


def render_table_image(table: pd.DataFrame, title: str, footnote: str, out_path: Path,
                        col_widths: list[float], figsize: tuple[float, float], fontsize: int = 13) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(cellText=table.to_numpy(), colLabels=list(table.columns), colWidths=col_widths,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 2.3)
    for (row_i, _col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=fontsize)
            cell.set_facecolor("#161616")
        else:
            cell.set_facecolor("#f2f1ee" if row_i % 2 == 0 else "white")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.995)
    fig.text(0.02, 0.01, footnote, fontsize=10, color="#3a3a3a", ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table image to {out_path}")


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
    table.to_csv(TABLE_OUT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))

    render_table_image(
        table,
        title="Table 2. AUC and paired DeLong test results by outcome, model family, and cohort",
        footnote="\"+covariate\" = mean mAs (Family A) or VAT+SAT (Family B); \"+covariate+AEC\" additionally "
                  "adds the prespecified AEC score. \"DeLong P (+covariate)\" tests clinic4 vs. clinic4+covariate; "
                  "\"DeLong P (+AEC)\" tests clinic4+covariate vs. clinic4+covariate+AEC (the AEC increment). "
                  "Source: outputs/step_disease_logistic/auc_delta_summary.csv (2026-08-24 run).",
        out_path=TABLE_OUT_DIR / f"{stem}.png",
        col_widths=[0.14, 0.09, 0.12, 0.10, 0.12, 0.14, 0.14, 0.14],
        figsize=(16, 1.8 + 0.55 * len(table)),
        fontsize=12,
    )


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
    table.to_csv(TABLE_OUT_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    table.to_excel(TABLE_OUT_DIR / f"{stem}.xlsx", index=False)
    print(table.to_string(index=False))

    render_table_image(
        table,
        title="Table 3. Sensitivity, specificity, and accuracy at the internal-OOF Youden-optimal threshold",
        footnote="Threshold fixed per model from the internal out-of-fold (OOF) ROC curve (Youden's J index), "
                  "then applied unchanged to the external cohort. Source: outputs/step_disease_logistic/"
                  "logistic_regression_summary.csv (2026-08-24 run).",
        out_path=TABLE_OUT_DIR / f"{stem}.png",
        col_widths=[0.16, 0.22, 0.09, 0.12, 0.10, 0.12, 0.12, 0.10],
        figsize=(15, 1.8 + 0.42 * len(table)),
        fontsize=11,
    )


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


def plot_figure5(out_path: Path) -> None:
    # Single top-to-bottom chain, one box per stage, no branching arrows -> no risk of
    # arrows crossing through other boxes. The two model families are listed as text
    # *inside* one box (stage 3) rather than as separate parallel tracks.
    cx = 6.0
    w_main, h_main = 10.5, 1.5
    gap = 0.85
    fig, ax = plt.subplots(figsize=(13, 15.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 17.3)  # provisional; tightened to actual content extent below
    ax.axis("off")

    stages = [
        ("Internal cohort (Gangnam, n=1088)\nraw AEC-128 tube-current curves recorded by the scanner",
         WHITE, 1.4),
        ("FPCA fit once on internal-cohort curves only\nnumber of components k chosen by a Kneedle elbow "
         "criterion (Figure 2c)\n-> prespecified AEC score (FPCA PC1-k)",
         FINAL_GREEN, 1.9),
        ("Two nested logistic-regression model families (both start from clinic4 = age, sex, height, weight)\n\n"
         "Family A (scan-technique baseline):   clinic4  ->  + mean_mAs  ->  + mean_mAs + AEC score\n"
         "Family B (CT-adiposity baseline):     clinic4  ->  + VAT + SAT  ->  + VAT + SAT + AEC score",
         ACCENT, 2.3),
        ("Internal training: stratified 5-fold cross-validation on the internal cohort\n"
         "-> out-of-fold (OOF) predicted probabilities -> internal AUC + Youden-optimal threshold\n"
         "(for AEC-including models, FPCA is refit within each fold's training partition only, to avoid leakage)",
         WHITE, 2.0),
        ("Final model of each family refit once on the full internal cohort, then frozen\n"
         "(no further fitting on external data)",
         WHITE, 1.4),
        ("External cohort (Sinchon, n=925): single frozen application per family\n"
         "-> external AUC, and sensitivity/specificity/accuracy at the internal-fixed threshold",
         ACCENT, 1.6),
        ("Paired DeLong test for each nested comparison (clinic4 vs. +covariate; "
         "+covariate vs. +covariate+AEC),\nperformed independently in the internal and external cohorts "
         "(Table 2, Figure 3, Figure 4)",
         FINAL_GREEN, 1.7),
    ]

    y_center = 16.1 - stages[0][2] / 2
    prev_bottom = None
    for text, facecolor, h in stages:
        if prev_bottom is not None:
            y_center = prev_bottom - gap - h / 2
            arrow(ax, (cx, prev_bottom), (cx, y_center + h / 2 + 0.05))
        box(ax, cx, y_center, w_main, h, text, facecolor, fontsize=12.5, fontweight="normal")
        prev_bottom = y_center - h / 2

    ax.set_ylim(prev_bottom - 0.6, 17.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 5 to {out_path}")


def run_figure5() -> None:
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_figure5(FIGURE_OUT_DIR / "fig5_study_design.png")


def main() -> None:
    run_table2()
    run_table3()
    run_figure5()


if __name__ == "__main__":
    main()
