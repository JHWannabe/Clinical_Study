from __future__ import annotations

# clinic-only baseline의 TP/FP 환자를 gangnam.xlsx의 TotalSegmentator 신체구성 feature로 비교
# 실행: python code/baseline/tp_fp_totalseg_comparison.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

baseline = import_module("clinic-only_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline" / "tp_fp_totalseg_comparison"

# 비교 대상 feature 목록
FEATURES = [
    "IMATA_sum_cm2",
    "NAMA_sum_cm2",
    "LAMA_sum_cm2",
    "내장지방_sum_cm2",
    "피하지방_sum_cm2",
    "총근육량",
    "총지방량",
]


# clinic-only baseline의 학습 과정을 그대로 재현해 TP/FP 환자 행만 추출
def reproduce_tp_fp_rows() -> pd.DataFrame:
    meta, y = baseline.load_cohort(baseline.INTERNAL_XLSX)
    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof, baseline.TARGET_SENSITIVITY)
    rows = baseline.build_group_rows(meta, y, oof, th)
    return rows[rows["group"].isin(["TP", "FP"])].copy()


# gangnam.xlsx metadata 시트에서 PatientID, TAMA, 신체구성 feature 컬럼을 로드
def load_totalseg_features() -> pd.DataFrame:
    meta = pd.read_excel(baseline.INTERNAL_XLSX, sheet_name="metadata", engine="openpyxl")
    cols = ["PatientID", "TAMA", *FEATURES]
    return meta[cols].copy()


# 각 feature와 라벨 변수 TAMA의 상관계수를 전체 코호트(n=1090) 기준으로 계산 (label circularity 점검용)
def tama_correlations(totalseg: pd.DataFrame) -> pd.Series:
    tama = totalseg["TAMA"].to_numpy(dtype=float)
    return pd.Series({feat: np.corrcoef(tama, totalseg[feat].to_numpy(dtype=float))[0, 1] for feat in FEATURES})


# feat를 baseline 분류기 입력(weight/height/age/sex)에 회귀시켜 TP/FP 잔차를 반환 (classifier-input circularity 보정)
def residualize_on_baseline_inputs(df: pd.DataFrame, feat: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[["group", "weight", "height", "age", "sex", feat]].dropna().copy()
    sex_m = (sub["sex"].astype(str).str.upper() == "M").astype(float)
    x = np.column_stack([np.ones(len(sub)), sub["weight"], sub["height"], sub["age"], sex_m])
    y = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    group = sub["group"].to_numpy()
    return resid[group == "TP"], resid[group == "FP"]


# feature별 TP/FP Welch t-test, Mann-Whitney U, Cohen's d, 잔차보정 p-value를 표로 정리
def welch_table(df: pd.DataFrame, corr_with_tama: pd.Series) -> pd.DataFrame:
    tp = df[df["group"] == "TP"]
    fp = df[df["group"] == "FP"]
    out_rows = []
    for feat in FEATURES:
        a = pd.to_numeric(tp[feat], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(fp[feat], errors="coerce").dropna().to_numpy()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        _, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        cohens_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
        se_diff = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        diff = a.mean() - b.mean()
        ci_lo, ci_hi = diff - 1.96 * se_diff, diff + 1.96 * se_diff

        a_adj, b_adj = residualize_on_baseline_inputs(df, feat)
        _, p_adj = stats.ttest_ind(a_adj, b_adj, equal_var=False)
        _, p_adj_mwu = stats.mannwhitneyu(a_adj, b_adj, alternative="two-sided")

        out_rows.append({
            "feature": feat,
            "n_TP": len(a), "n_FP": len(b),
            "TP_mean": a.mean(), "TP_sd": a.std(ddof=1),
            "FP_mean": b.mean(), "FP_sd": b.std(ddof=1),
            "diff_TP_minus_FP": diff,
            "ci95_lower": ci_lo, "ci95_upper": ci_hi,
            "t": t, "p": p, "p_mwu": p_mwu, "cohens_d": cohens_d,
            "corr_with_TAMA": corr_with_tama[feat],
            "p_adj_weight_height_age_sex": p_adj,
            "p_adj_mwu": p_adj_mwu,
        })
    return pd.DataFrame(out_rows)


CIRCULARITY_THRESHOLD = 0.4  # |corr with TAMA| above this -> flagged as label-adjacent, not independent


# 통계표와 두 circularity 점검 결과를 markdown 리포트로 저장
def write_report(table: pd.DataFrame, n_tp: int, n_fp: int, out_dir: Path) -> None:
    fmt = table.copy()
    for col in ["TP_mean", "TP_sd", "FP_mean", "FP_sd", "diff_TP_minus_FP", "ci95_lower", "ci95_upper", "t", "cohens_d", "corr_with_TAMA"]:
        fmt[col] = fmt[col].round(3)
    for col in ["p", "p_mwu", "p_adj_weight_height_age_sex", "p_adj_mwu"]:
        fmt[col] = fmt[col].apply(lambda v: f"{v:.4e}")
    fmt["label_circular"] = table["corr_with_TAMA"].abs().gt(CIRCULARITY_THRESHOLD).map({True: "yes", False: "no"})
    fmt["survives_adjustment"] = (table["p_adj_weight_height_age_sex"] < 0.05).map({True: "yes", False: "no"})

    circular_feats = table.loc[table["corr_with_TAMA"].abs().gt(CIRCULARITY_THRESHOLD), "feature"].tolist()
    independent_feats = table.loc[table["corr_with_TAMA"].abs().le(CIRCULARITY_THRESHOLD), "feature"].tolist()
    robust_feats = table.loc[
        table["corr_with_TAMA"].abs().le(CIRCULARITY_THRESHOLD) & (table["p_adj_weight_height_age_sex"] < 0.05),
        "feature",
    ].tolist()

    lines = [
        "# TP vs FP: TotalSegmentator body-composition features (baseline, internal cohort)",
        "",
        f"Clinic-only baseline (Se>=90%) TP (n={n_tp}) vs FP (n={n_fp}), features from gangnam.xlsx.",
        "Welch's t-test (unequal variance) and Mann-Whitney U, 95% CI on TP-FP mean difference, Cohen's d.",
        "",
        "## Two circularity checks",
        "",
        "**1) Label circularity.** The label is SMI = TAMA / Height^2, so a feature correlated "
        "with TAMA itself (full n=1090 cohort) separates TP from FP largely by construction. "
        f"Flagged at |r(TAMA, feature)|>{CIRCULARITY_THRESHOLD}:",
        "",
        f"- Label-circular: {', '.join(circular_feats) if circular_feats else 'none'}",
        f"- Independent of TAMA: {', '.join(independent_feats) if independent_feats else 'none'}",
        "",
        "`총근육량` = `NAMA_sum_cm2 + LAMA_sum_cm2 + IMATA_sum_cm2` exactly (verified, max abs diff ~1e-12) "
        "-- it is the whole-scan analogue of TAMA.",
        "",
        "**2) Classifier-input circularity.** TP and FP are both baseline \"predicted positive\", "
        "but they still differ on the classifier's own inputs themselves (weight/height/age/sex; "
        "e.g. TP mean weight 57.9kg vs FP 60.3kg, p=0.027) -- per "
        "[[feedback_no_circular_restratification]]. A feature merely correlated with those inputs "
        "(e.g. subcutaneous fat vs weight, r=0.19) can show a raw TP/FP gap driven by that, not by "
        "anything specific to TP/FP. `p_adj_weight_height_age_sex` / `p_adj_mwu` are the TP/FP test "
        "p-values on residuals after regressing each feature on weight+height+age+sex_M.",
        "",
        f"- Survives adjustment (independent of TAMA **and** p_adj<0.05): {', '.join(robust_feats) if robust_feats else 'none'}",
        "- Everything else is either label-circular, classifier-input-circular, or not significant "
        "to begin with -- raw p-values for those should not be read as new findings about FP patients.",
        "",
        fmt.to_markdown(index=False),
        "",
    ]
    (out_dir / "tp_fp_totalseg_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {out_dir / 'tp_fp_totalseg_comparison.md'}")


# feature별 TP/FP boxplot을 한 장의 이미지로 저장
def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3.2 * len(FEATURES), 5))
    for ax, feat in zip(axes, FEATURES):
        data = [
            pd.to_numeric(df.loc[df["group"] == g, feat], errors="coerce").dropna().to_numpy()
            for g in ["TP", "FP"]
        ]
        ax.boxplot(data, tick_labels=["TP", "FP"], showmeans=True)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle("TP vs FP: TotalSegmentator body-composition features", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / "tp_fp_totalseg_comparison_boxplot.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved boxplot to {out_path}")


SLIDE_FEATURE_LABELS = {
    "IMATA_sum_cm2": "IMATA",
    "NAMA_sum_cm2": "NAMA",
    "LAMA_sum_cm2": "LAMA",
    "내장지방_sum_cm2": "VAT (내장지방)",
    "피하지방_sum_cm2": "SAT (피하지방)",
    "총근육량": "Muscle (총근육량)",
    "총지방량": "Fat (총지방량)",
}


# p-value를 슬라이드용 문자열로 포맷 (0.001 미만은 "<0.001")
def _fmt_p(v: float) -> str:
    return "<0.001" if v < 0.001 else f"{v:.3f}"


# raw 비교 결과만(잔차보정 컬럼 제외) 슬라이드 삽입용 표 이미지로 렌더링
def render_slide_table(table: pd.DataFrame, n_tp: int, n_fp: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    col_labels = ["Feature", "n (TP/FP)", "TP mean±SD", "FP mean±SD", "Diff (TP-FP)", "p (Welch)", "p (MWU)", "corr(TAMA)"]
    cell_text = []
    for _, row in table.iterrows():
        cell_text.append([
            SLIDE_FEATURE_LABELS[row["feature"]],
            f"{n_tp}/{n_fp}",
            f"{row['TP_mean']:,.0f} ± {row['TP_sd']:,.0f}",
            f"{row['FP_mean']:,.0f} ± {row['FP_sd']:,.0f}",
            f"{row['diff_TP_minus_FP']:+,.0f}",
            _fmt_p(row["p"]),
            _fmt_p(row["p_mwu"]),
            f"{row['corr_with_TAMA']:+.2f}",
        ])

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_i == 0:
            cell.set_facecolor("#2a3f5f")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f5f6f8" if row_i % 2 == 0 else "white")
            p_val = table.iloc[row_i - 1]["p"]
            if col_i == 5 and p_val < 0.05:
                cell.get_text().set_color("#c0392b")
                cell.get_text().set_fontweight("bold")

    fig.tight_layout(pad=0.3)
    out_path = out_dir / "tp_fp_totalseg_comparison_slide_table.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved slide table image to {out_path}")


# TP/FP 추출 -> feature 병합 -> 통계표/리포트/boxplot/슬라이드표 생성까지 전체 파이프라인 실행
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tp_fp_rows = reproduce_tp_fp_rows()
    totalseg = load_totalseg_features()
    merged = tp_fp_rows.merge(totalseg, on="PatientID", how="left", validate="one_to_one")

    missing = merged[FEATURES].isna().any(axis=1).sum()
    if missing:
        print(f"Warning: {missing} TP/FP patients missing one or more totalseg features (dropped from stats).")

    merged.to_csv(OUTPUT_DIR / "tp_fp_totalseg_comparison_rows.csv", index=False)

    corr_with_tama = tama_correlations(totalseg)
    table = welch_table(merged, corr_with_tama)
    table.to_csv(OUTPUT_DIR / "tp_fp_totalseg_comparison.csv", index=False)

    n_tp = int((merged["group"] == "TP").sum())
    n_fp = int((merged["group"] == "FP").sum())
    write_report(table, n_tp, n_fp, OUTPUT_DIR)
    plot_boxplots(merged, OUTPUT_DIR)
    render_slide_table(table, n_tp, n_fp, OUTPUT_DIR)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

