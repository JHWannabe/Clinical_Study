from __future__ import annotations

# clinic-only baseline의 TP/FP 환자를 gangnam.xlsx의 TotalSegmentator 신체구성 feature로 비교
# 실행: python code/baseline/tp_fp_totalseg_comparison.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


# 결측 없는 z-score 표준화 (평균0, 표준편차1)
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


# group(TP=1/FP=0) ~ feat + weight+height+age+sex_M 다변량 로지스틱회귀 -> feat의 조정 OR·95%CI·p
# 잔차보정 t-test와 달리 공변량을 통제한 채로 TP/FP 자체를 직접 모델링하는 표준적 방법
def logistic_adjusted_or(df: pd.DataFrame, feat: str) -> tuple[float, float, float, float]:
    sub = df[["group", "weight", "height", "age", "sex", feat]].dropna().copy()
    sex_m = (sub["sex"].astype(str).str.upper() == "M").astype(float)
    y = (sub["group"] == "TP").astype(float).to_numpy()
    x = pd.DataFrame({
        "feat": _zscore(pd.to_numeric(sub[feat], errors="coerce")),
        "weight": _zscore(sub["weight"]),
        "height": _zscore(sub["height"]),
        "age": _zscore(sub["age"]),
        "sex_M": sex_m,
    })
    x = sm.add_constant(x)
    try:
        model = sm.Logit(y, x).fit(disp=0)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")
    coef, se, p = model.params["feat"], model.bse["feat"], model.pvalues["feat"]
    return float(np.exp(coef)), float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se)), float(p)


# Cohen's d의 percentile bootstrap 95% CI (완전 벡터화, n_boot회 재표집)
def bootstrap_cohens_d_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    ra = a[rng.integers(0, na, size=(n_boot, na))]
    rb = b[rng.integers(0, nb, size=(n_boot, nb))]
    var_a, var_b = ra.var(axis=1, ddof=1), rb.var(axis=1, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    d = np.where(pooled_sd > 0, (ra.mean(axis=1) - rb.mean(axis=1)) / pooled_sd, np.nan)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


# feature별 TP/FP Welch t-test, Mann-Whitney U, Cohen's d(+bootstrap CI), 잔차보정 p-value, 다변량 adjusted OR, BH-FDR을 표로 정리
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
        d_ci_lo, d_ci_hi = bootstrap_cohens_d_ci(a, b)
        se_diff = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        diff = a.mean() - b.mean()
        ci_lo, ci_hi = diff - 1.96 * se_diff, diff + 1.96 * se_diff

        a_adj, b_adj = residualize_on_baseline_inputs(df, feat)
        _, p_adj = stats.ttest_ind(a_adj, b_adj, equal_var=False)
        _, p_adj_mwu = stats.mannwhitneyu(a_adj, b_adj, alternative="two-sided")

        or_adj, or_ci_lo, or_ci_hi, or_p = logistic_adjusted_or(df, feat)

        out_rows.append({
            "feature": feat,
            "n_TP": len(a), "n_FP": len(b),
            "TP_mean": a.mean(), "TP_sd": a.std(ddof=1),
            "FP_mean": b.mean(), "FP_sd": b.std(ddof=1),
            "diff_TP_minus_FP": diff,
            "ci95_lower": ci_lo, "ci95_upper": ci_hi,
            "t": t, "p": p, "p_mwu": p_mwu,
            "cohens_d": cohens_d, "d_ci95_lower": d_ci_lo, "d_ci95_upper": d_ci_hi,
            "corr_with_TAMA": corr_with_tama[feat],
            "p_adj_weight_height_age_sex": p_adj,
            "p_adj_mwu": p_adj_mwu,
            "OR_adj_per_1SD": or_adj, "OR_adj_ci95_lower": or_ci_lo, "OR_adj_ci95_upper": or_ci_hi, "OR_adj_p": or_p,
        })
    out = pd.DataFrame(out_rows)
    out["p_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
    out["p_adj_fdr"] = multipletests(out["p_adj_weight_height_age_sex"], method="fdr_bh")[1]
    out["OR_adj_p_fdr"] = multipletests(out["OR_adj_p"], method="fdr_bh")[1]
    return out


# TP/FP 병합 데이터에서 7개 feature 간 Pearson 상관행렬 계산 (다중공선성/중복 feature 점검용)
def feature_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    return x.corr()


# feature별 VIF = 1/(1-R^2), 나머지 6개 feature로 자신을 회귀했을 때의 분산팽창 (3개 이상 얽힌 다중공선성까지 탐지, pairwise r로는 못 잡음)
# 7개 feature 행렬(절편 포함 8열)의 실제 rank는 6 -- 항등식이 2개 있어서(총근육량=NAMA+LAMA+IMATA,
# 총지방량=내장지방+피하지방, 둘 다 최대 절대오차 ~1e-12) 전 feature의 VIF가 정확히 inf로 나오는 것이
# 정상 동작. pairwise |r|은 이 항등식을 못 잡는다 (예: IMATA-NAMA r=-0.02인데도 완전공선).
def feature_vif(df: pd.DataFrame) -> pd.Series:
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    x = sm.add_constant(x)
    return pd.Series({feat: variance_inflation_factor(x.values, x.columns.get_loc(feat)) for feat in FEATURES})


REDUNDANCY_THRESHOLD = 0.8  # |r| above this between two features -> flagged as redundant (near-duplicate signal)


# 상관행렬에서 |r|>threshold인 feature 쌍을 (feat_a, feat_b, r) 리스트로 추출
def redundant_pairs(corr: pd.DataFrame, threshold: float = REDUNDANCY_THRESHOLD) -> list[tuple[str, str, float]]:
    pairs = []
    for i, fa in enumerate(FEATURES):
        for fb in FEATURES[i + 1:]:
            r = corr.loc[fa, fb]
            if abs(r) > threshold:
                pairs.append((fa, fb, float(r)))
    return pairs


# 대칭행렬인 상관행렬을 하삼각(대각선 포함)만 남긴 markdown 표 문자열로 변환
def lower_triangle_markdown(corr: pd.DataFrame) -> str:
    tri = corr.round(2).astype(object)
    for i, fa in enumerate(FEATURES):
        for j, fb in enumerate(FEATURES):
            if j > i:
                tri.loc[fa, fb] = ""
    return tri.to_markdown()


# feature 상관행렬을 heatmap 이미지로 저장 (대각선 포함 하삼각만 표기, 상삼각은 대칭이라 생략)
def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    values = corr.to_numpy().copy()
    masked = np.ma.masked_array(values, mask=mask)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(masked, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels([SLIDE_FEATURE_LABELS[f] for f in FEATURES], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([SLIDE_FEATURE_LABELS[f] for f in FEATURES], fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            if mask[i, j]:
                continue
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation matrix (TP+FP)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "tp_fp_totalseg_feature_correlation.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved correlation heatmap to {out_path}")


CIRCULARITY_THRESHOLD = 0.4  # |corr with TAMA| above this -> flagged as label-adjacent, not independent


# 통계표, circularity 점검, BH-FDR, adjusted OR, feature 상관행렬/VIF를 markdown 리포트로 저장
def write_report(table: pd.DataFrame, corr: pd.DataFrame, vif: pd.Series, n_tp: int, n_fp: int, out_dir: Path) -> None:
    def fmt_p(v: float) -> str:
        return f"{v:.4e}"

    desc = pd.DataFrame({
        "feature": table["feature"],
        "n_TP": table["n_TP"], "n_FP": table["n_FP"],
        "TP_mean±SD": table.apply(lambda r: f"{r['TP_mean']:.2f}±{r['TP_sd']:.2f}", axis=1),
        "FP_mean±SD": table.apply(lambda r: f"{r['FP_mean']:.2f}±{r['FP_sd']:.2f}", axis=1),
        "diff(TP-FP)": table["diff_TP_minus_FP"].round(3),
        "diff_ci95": table.apply(lambda r: f"[{r['ci95_lower']:.2f}, {r['ci95_upper']:.2f}]", axis=1),
        "cohens_d": table["cohens_d"].round(3),
        "d_ci95(boot)": table.apply(lambda r: f"[{r['d_ci95_lower']:.2f}, {r['d_ci95_upper']:.2f}]", axis=1),
        "p_welch": table["p"].apply(fmt_p),
        "p_welch_fdr": table["p_fdr"].apply(fmt_p),
        "p_mwu": table["p_mwu"].apply(fmt_p),
    })

    adj = pd.DataFrame({
        "feature": table["feature"],
        "corr_TAMA": table["corr_with_TAMA"].round(3),
        "label_circular": table["corr_with_TAMA"].abs().gt(CIRCULARITY_THRESHOLD).map({True: "yes", False: "no"}),
        "p_resid_adj": table["p_adj_weight_height_age_sex"].apply(fmt_p),
        "p_resid_adj_fdr": table["p_adj_fdr"].apply(fmt_p),
        "OR_per_1SD": table["OR_adj_per_1SD"].round(3),
        "OR_ci95": table.apply(lambda r: f"[{r['OR_adj_ci95_lower']:.2f}, {r['OR_adj_ci95_upper']:.2f}]", axis=1),
        "OR_p": table["OR_adj_p"].apply(fmt_p),
        "OR_p_fdr": table["OR_adj_p_fdr"].apply(fmt_p),
    })

    circular_feats = table.loc[table["corr_with_TAMA"].abs().gt(CIRCULARITY_THRESHOLD), "feature"].tolist()
    independent_feats = table.loc[table["corr_with_TAMA"].abs().le(CIRCULARITY_THRESHOLD), "feature"].tolist()
    fdr_survives = table["corr_with_TAMA"].abs().le(CIRCULARITY_THRESHOLD) & (table["p_adj_fdr"] < 0.05)
    or_survives = table["corr_with_TAMA"].abs().le(CIRCULARITY_THRESHOLD) & (table["OR_adj_p_fdr"] < 0.05)
    robust_feats = table.loc[fdr_survives, "feature"].tolist()
    robust_both_feats = table.loc[fdr_survives & or_survives, "feature"].tolist()

    redund = redundant_pairs(corr)
    redund_lines = [f"- `{a}` vs `{b}`: r={r:+.2f}" for a, b, r in redund] if redund else ["- |r|>%.1f 초과 쌍 없음" % REDUNDANCY_THRESHOLD]

    vif_df = pd.DataFrame({"feature": FEATURES, "VIF": vif[FEATURES].round(2).to_numpy()})

    lines = [
        "# TP vs FP: TotalSegmentator 체성분 feature 비교 (baseline, 내부 코호트)",
        "",
        f"Clinic-only baseline (Se>=90%) TP (n={n_tp}) vs FP (n={n_fp}), feature는 gangnam.xlsx 기준.",
        "Welch's t-test(이분산 가정), Mann-Whitney U, Cohen's d(bootstrap 95% CI, 10000회 resampling), "
        "7개 feature에 대해 Benjamini-Hochberg FDR 적용.",
        "",
        "## 기술통계 + 원시 검정",
        "",
        desc.to_markdown(index=False),
        "",
        "## 순환성(circularity) 점검 2가지",
        "",
        "**1) 라벨 순환성(Label circularity).** 라벨은 SMI = TAMA / Height^2로 정의되므로, "
        "TAMA 자체와 상관된 feature(전체 n=1090 코호트 기준)는 정의상 TP/FP를 구분하게 된다. "
        f"|r(TAMA, feature)|>{CIRCULARITY_THRESHOLD} 기준으로 플래그:",
        "",
        f"- 라벨 순환(label-circular): {', '.join(circular_feats) if circular_feats else '없음'}",
        f"- TAMA와 독립: {', '.join(independent_feats) if independent_feats else '없음'}",
        "",
        "`총근육량` = `NAMA_sum_cm2 + LAMA_sum_cm2 + IMATA_sum_cm2` (검증 완료, 최대 절대오차 ~1e-12) "
        "-- TAMA의 전체 스캔(whole-scan) 버전에 해당.",
        "",
        "**2) 분류기 입력 순환성(Classifier-input circularity).** TP와 FP는 둘 다 baseline에서 "
        "\"predicted positive\"이지만, 분류기 자체의 입력값(체중/신장/나이/성별)에서도 서로 차이가 난다"
        "(예: TP 평균 체중 57.9kg vs FP 60.3kg, p=0.027) -- "
        "[[feedback_no_circular_restratification]] 참고. 서로 독립적인 두 가지 보정 방법을 함께 제시: "
        "(a) `p_resid_adj` -- feature를 weight+height+age+sex_M에 회귀시킨 잔차(residual)에 대한 "
        "TP/FP t-test; (b) `OR_per_1SD` -- group(TP=1/FP=0) ~ feature + weight + height + age + sex_M "
        "다변량 로지스틱 회귀에서, 나머지 4개 공변량을 고정했을 때 feature 자체의 1-SD 증가당 "
        "odds ratio. FDR 열은 각 방법 내에서 7개 feature에 대해 BH 보정을 적용한 값.",
        "",
        adj.to_markdown(index=False),
        "",
        f"- 잔차 보정을 통과(TAMA와 독립 **그리고** p_resid_adj_fdr<0.05): "
        f"{', '.join(robust_feats) if robust_feats else '없음'}",
        f"- **두 보정 방법 모두** 통과(TAMA와 독립, p_resid_adj_fdr<0.05 **그리고** "
        f"OR_p_fdr<0.05): {', '.join(robust_both_feats) if robust_both_feats else '없음'} -- "
        "본 리포트에서 가장 엄격한 근거 기준.",
        "- 그 외 feature는 라벨 순환이거나 분류기-입력 순환이거나 애초에 유의하지 않음 -- "
        "이들의 원시 p-value를 FP 환자에 대한 새로운 소견으로 해석해서는 안 됨.",
        "",
        "## Feature 상관관계 / 중복성 점검",
        "",
        "7개 feature 간 pairwise Pearson r (TP+FP 전체 행 기준), 대칭행렬이므로 하삼각만 표시. "
        f"|r|>{REDUNDANCY_THRESHOLD} 초과 쌍은 사실상 중복 신호이므로, 두 feature가 같은 방향을 보여도 "
        "독립적인 확증으로 볼 수 없음:",
        "",
        lower_triangle_markdown(corr),
        "",
        *redund_lines,
        "",
        f"**VIF (Variance Inflation Factor).** feature를 나머지 6개 feature에 회귀시킨 R^2로부터 "
        f"VIF=1/(1-R^2) 계산 -- pairwise r과 달리 3개 이상 feature가 함께 얽힌 다중공선성까지 탐지.",
        "",
        vif_df.to_markdown(index=False),
        "",
        "7개 feature 전부 VIF=inf. 원인은 두 항등식(accounting identity) -- "
        "`총근육량`=`NAMA_sum_cm2`+`LAMA_sum_cm2`+`IMATA_sum_cm2`, "
        "`총지방량`=`내장지방_sum_cm2`+`피하지방_sum_cm2` (둘 다 실측, 최대 절대오차 ~1e-12) -- "
        "이 7-feature 행렬(절편 포함 8열)의 rank를 6으로 만들기 때문. "
        "pairwise |r|은 이를 못 잡는다: 예를 들어 `IMATA_sum_cm2`와 `NAMA_sum_cm2`의 r=-0.02, "
        "`NAMA_sum_cm2`와 `총근육량`의 r=+0.90으로 threshold(0.8)를 넘는 쌍조차 일부일 뿐인데도 "
        "실제로는 완전공선(perfect collinearity)이다 -- 위 상관행렬만 보고 \"|r|>0.8 쌍만 조심하면 된다\"고 "
        "판단하면 안 됨.",
        "",
        f"- 실무 함의: `총근육량`·`총지방량`(합계)과 그 구성요소(`NAMA_sum_cm2`/`LAMA_sum_cm2`/"
        "`IMATA_sum_cm2`, `내장지방_sum_cm2`/`피하지방_sum_cm2`)를 같은 회귀/분류 모델에 동시 투입 금지 "
        "-- 합계 1개 또는 구성요소 전체 중 하나만 선택.",
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
            p_val_mwu = table.iloc[row_i - 1]["p_mwu"]
            if col_i == 5 and p_val < 0.05:
                cell.get_text().set_color("#c0392b")
                cell.get_text().set_fontweight("bold")
            if col_i == 6 and p_val_mwu < 0.05:
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

    feat_corr = feature_correlation_matrix(merged)
    feat_corr.to_csv(OUTPUT_DIR / "tp_fp_totalseg_feature_correlation.csv")

    feat_vif = feature_vif(merged)
    feat_vif.rename("VIF").to_csv(OUTPUT_DIR / "tp_fp_totalseg_feature_vif.csv", index_label="feature")

    n_tp = int((merged["group"] == "TP").sum())
    n_fp = int((merged["group"] == "FP").sum())
    write_report(table, feat_corr, feat_vif, n_tp, n_fp, OUTPUT_DIR)
    plot_boxplots(merged, OUTPUT_DIR)
    plot_correlation_heatmap(feat_corr, OUTPUT_DIR)
    render_slide_table(table, n_tp, n_fp, OUTPUT_DIR)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

