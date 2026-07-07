from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, load_dataset  # noqa: E402
from aec_offset_score import apply_clinical, clinical_raw, fit_clinical  # noqa: E402
from aec128_common_shape_feature import load_aec128  # noqa: E402
from aec128_highdim_aec_only import build_feature_banks, crossfit_fixed_hyper, make_model  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_within_clinical_score"
FEATURE_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_deep_feature_mining"
SEED = 20260629


def clinical_oof_external(train: dict, test: dict) -> tuple[np.ndarray, np.ndarray]:
    """임상 모델을 5-fold로 학습해 train의 out-of-fold 점수를, 전체 train으로 재학습한 모델로 외부 점수를 구함."""
    y = train["y"].astype(int)
    xtr = clinical_raw(train["meta"])
    xte = clinical_raw(test["meta"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(xtr, y):
        model = fit_clinical(xtr[tr_idx], y[tr_idx])
        oof[va_idx] = apply_clinical(model, xtr[va_idx])
    final = fit_clinical(xtr, y)
    external = apply_clinical(final, xte)
    return oof, external


def highdim_raw_log_scores() -> tuple[np.ndarray, np.ndarray]:
    """고차원(128차원) 로그 프로파일 특징에 L2 릿지 로지스틱(C=0.01)을 고정 적용해 train OOF/외부 점수를 계산."""
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    y = g["y"].astype(int)
    xtr = build_feature_banks(g["x"])["raw_log128"].to_numpy(dtype=float)
    xte = build_feature_banks(s["x"])["raw_log128"].to_numpy(dtype=float)
    oof, _ = crossfit_fixed_hyper(xtr, y, penalty="l2", c=0.01, class_weight_label="none")
    final = make_model("l2", 0.01, None)
    final.fit(xtr, y)
    external = final.decision_function(xte)
    return oof, external


def zscore(x: np.ndarray) -> np.ndarray:
    """배열을 평균 0, 표준편차 1로 표준화 (표준편차가 0/비정상이면 1로 대체)."""
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - np.nanmean(x)) / sd


def adjusted_logit(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray) -> dict:
    """임상점수로 보정한 후 AEC점수의 계수·오즈비·Wald p값과 LRT 카이제곱을 계산 (p값 자체는 lrt_p_value에서 별도 계산)."""
    c = zscore(clinical_score)
    a = zscore(aec_score)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    chi2 = 2 * (m1.llf - m0.llf)
    return {
        "clinical_adjusted_auc_score_auc": float(roc_auc_score(y, aec_score)),
        "clinical_adjusted_auc_score_ap": float(average_precision_score(y, aec_score)),
        "aec_beta_per_sd_adjusted_for_clinical": float(m1.params[2]),
        "aec_or_per_sd_adjusted_for_clinical": float(np.exp(m1.params[2])),
        "aec_wald_p_adjusted_for_clinical": float(m1.pvalues[2]),
        "lrt_chi2_1df": float(chi2),
        "lrt_p": float(sm.stats.stattools.stats.chi2.sf(chi2, 1)) if False else np.nan,
    }


def lrt_p_value(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray) -> float:
    """임상점수만 넣은 모델과 AEC점수까지 넣은 모델의 우도비검정(LRT) p-value를 계산."""
    from scipy import stats

    c = zscore(clinical_score)
    a = zscore(aec_score)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    return float(stats.chi2.sf(2 * (m1.llf - m0.llf), 1))


def fixed_effect_high_low(y: np.ndarray, bins: np.ndarray, aec_high: np.ndarray) -> dict:
    """임상점수 구간(bin)을 고정효과로 통제한 로지스틱 회귀로 "구간 내 AEC 높음 vs 낮음"의 오즈비를 추정하고, Mantel-Haenszel 스타일의 구간별 오즈비 중앙값도 함께 계산."""
    from scipy import stats

    dummies = pd.get_dummies(bins, prefix="bin", drop_first=True, dtype=float)
    x = pd.concat([pd.Series(aec_high.astype(float), name="aec_high"), dummies], axis=1)
    x = sm.add_constant(x, has_constant="add")
    m = sm.Logit(y, x).fit(disp=False, maxiter=1000)
    beta = float(m.params["aec_high"])
    p = float(m.pvalues["aec_high"])

    # Mantel-Haenszel-style pooled OR across bins as a simple sensitivity check.
    ors = []
    for b in np.unique(bins):
        hi = aec_high & (bins == b)
        lo = (~aec_high) & (bins == b)
        a = np.sum(y[hi] == 1) + 0.5
        b0 = np.sum(y[hi] == 0) + 0.5
        c = np.sum(y[lo] == 1) + 0.5
        d = np.sum(y[lo] == 0) + 0.5
        ors.append((a / b0) / (c / d))
    return {
        "bin_fixed_effect_or_high_vs_low": float(np.exp(beta)),
        "bin_fixed_effect_beta": beta,
        "bin_fixed_effect_p": p,
        "median_bin_or_high_vs_low": float(np.median(ors)),
    }


def assign_deciles(score: np.ndarray) -> np.ndarray:
    """점수를 순위로 변환해(동점 문제 회피) 10개 등분 구간(10분위) 번호를 매김."""
    # qcut can fail with duplicated cutpoints; rank makes bins stable and equal-sized.
    ranks = pd.Series(score).rank(method="first").to_numpy()
    return pd.qcut(ranks, q=10, labels=False).astype(int)


def within_bin_summary(
    cohort: str,
    score_name: str,
    y: np.ndarray,
    clinical_score: np.ndarray,
    aec_score: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """임상점수 10분위 구간마다 그 구간 내 AEC 중앙값 기준으로 고/저를 나눠, 구간별·전체 사건율표와
    구간 고정효과 오즈비, 임상점수 보정 후 AEC의 조건부 연관성(LRT)까지 모두 계산."""
    bins = assign_deciles(clinical_score)
    aec_high = np.zeros(len(y), dtype=bool)
    rows = []
    for b in sorted(np.unique(bins)):
        mask = bins == b
        med = np.median(aec_score[mask])
        high = mask & (aec_score >= med)
        low = mask & (aec_score < med)
        aec_high[high] = True
        for label, m in [("AEC-low within same clinical bin", low), ("AEC-high within same clinical bin", high)]:
            n = int(np.sum(m))
            e = int(np.sum(y[m]))
            rows.append(
                {
                    "cohort": cohort,
                    "aec_score": score_name,
                    "clinical_decile": int(b + 1),
                    "clinical_score_min": float(np.min(clinical_score[mask])),
                    "clinical_score_max": float(np.max(clinical_score[mask])),
                    "group": label,
                    "n": n,
                    "events": e,
                    "event_rate": float(e / n) if n else np.nan,
                    "aec_median_in_bin": float(med),
                }
            )
    pooled_rows = []
    for label, m in [("AEC-low within clinical deciles", ~aec_high), ("AEC-high within clinical deciles", aec_high)]:
        n = int(np.sum(m))
        e = int(np.sum(y[m]))
        pooled_rows.append(
            {
                "cohort": cohort,
                "aec_score": score_name,
                "group": label,
                "n": n,
                "events": e,
                "event_rate": float(e / n),
            }
        )
    fixed = fixed_effect_high_low(y, bins, aec_high)
    adj = adjusted_logit(y, clinical_score, aec_score)
    adj["lrt_p"] = lrt_p_value(y, clinical_score, aec_score)
    pooled = {
        "cohort": cohort,
        "aec_score": score_name,
        **{f"{r['group']}_{k}": v for r in pooled_rows for k, v in r.items() if k not in ["cohort", "aec_score", "group"]},
        **fixed,
        **adj,
    }
    return pd.DataFrame(rows), pooled


def plot_results(bin_df: pd.DataFrame, pooled_df: pd.DataFrame) -> None:
    """대표 점수의 임상 10분위별 AEC 고/저 사건율 곡선과, 모든 점수x코호트 조합의 구간고정효과 오즈비 막대그래프를 PNG로 저장."""
    primary = "highdim_raw_log128_ridge"
    df = bin_df[bin_df["aec_score"].eq(primary)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, cohort in zip(axes, ["g1090", "sdata"]):
        sub = df[df["cohort"].eq(cohort)]
        pivot = sub.pivot(index="clinical_decile", columns="group", values="event_rate")
        ax.plot(
            pivot.index,
            pivot["AEC-low within same clinical bin"],
            marker="o",
            lw=2.0,
            color="#2F6F73",
            label="AEC-low within decile",
        )
        ax.plot(
            pivot.index,
            pivot["AEC-high within same clinical bin"],
            marker="o",
            lw=2.0,
            color="#C84630",
            label="AEC-high within decile",
        )
        ax.set_title(cohort, loc="left", fontweight="bold")
        ax.set_xlabel("Clinical score decile")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed low-SMI event rate")
    axes[0].legend(frameon=False)
    fig.suptitle("Within similar clinical score: AEC-high vs AEC-low", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "within_clinical_decile_highdim_aec_high_vs_low.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    order = pooled_df.sort_values(["cohort", "bin_fixed_effect_or_high_vs_low"])
    labels = [f"{r.cohort}\n{r.aec_score}" for _, r in order.iterrows()]
    x = np.arange(len(order))
    ax.bar(x, order["bin_fixed_effect_or_high_vs_low"], color="#4C78A8")
    ax.axhline(1.0, color="#555555", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("OR for low SMI: AEC-high vs AEC-low, clinical-bin fixed effect")
    ax.set_title("Does AEC separate risk among similar clinical scores?", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "within_clinical_decile_or_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 점수가 "비슷한" 환자들끼리 묶어 놓고 봐도, AEC 점수가
    높은 쪽이 실제로 더 위험한가? — 임상점수 자체의 영향을 층화로 통제한 검증):

    1. train(g1090)/test(sdata)를 로드하고, 임상 모델의 OOF/외부 점수와 "고차원 로그128 릿지"
       AEC 점수를 구한다. 딥 특징 채굴 CSV에서 두 개의 추가 AEC 후보(late_edge, cylindrical
       rebound mass)도 가져와 총 3개 AEC 점수 후보를 준비한다.
    2. 3개 AEC 후보 x 2개 코호트(g1090/sdata) 조합마다 within_bin_summary를 실행:
       - 임상점수를 10분위로 나누고, 각 분위 안에서 AEC 점수가 그 분위 중앙값보다 높은/낮은
         환자로 다시 나눈다 (임상점수가 "같은 수준"인 사람들끼리 비교).
       - 분위별 AEC 고/저 그룹의 사건율을 표로 만들고, 분위를 고정효과로 통제한 로지스틱 회귀로
         "AEC 높음 vs 낮음"의 오즈비를 추정하며, 임상점수 보정 후 AEC의 조건부 연관성(LRT)도 계산.
    3. 모든 결과를 분위별 표와 요약표로 각각 CSV로 저장.
    4. 대표 AEC 점수의 분위별 사건율 곡선과, 전체 조합의 오즈비 막대그래프를 그려 PNG로 저장.
    5. 방법론 설명과 함께 전체 요약을 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext = clinical_oof_external(train, test)
    highdim_oof, highdim_ext = highdim_raw_log_scores()

    ftr = pd.read_csv(FEATURE_DIR / "g1090_aec128_deep_features_patient_level.csv")
    fte = pd.read_csv(FEATURE_DIR / "sdata_aec128_deep_features_patient_level.csv")
    scores = {
        "late_edge": (
            ftr["haar_haar_l5_b12_right_minus_left"].to_numpy(dtype=float),
            fte["haar_haar_l5_b12_right_minus_left"].to_numpy(dtype=float),
        ),
        "cylindrical_rebound_mass": (
            ftr["cyl_cyl_late_positive_plus_mid_negative"].to_numpy(dtype=float),
            fte["cyl_cyl_late_positive_plus_mid_negative"].to_numpy(dtype=float),
        ),
        "highdim_raw_log128_ridge": (highdim_oof, highdim_ext),
    }

    bin_frames = []
    pooled_rows = []
    for score_name, (score_tr, score_te) in scores.items():
        bdf, prow = within_bin_summary("g1090", score_name, train["y"].astype(int), clinical_oof, score_tr)
        bin_frames.append(bdf)
        pooled_rows.append(prow)
        bdf, prow = within_bin_summary("sdata", score_name, test["y"].astype(int), clinical_ext, score_te)
        bin_frames.append(bdf)
        pooled_rows.append(prow)

    bin_df = pd.concat(bin_frames, ignore_index=True)
    pooled_df = pd.DataFrame(pooled_rows)
    bin_df.to_csv(OUT_DIR / "aec_high_low_within_clinical_deciles.csv", index=False)
    pooled_df.to_csv(OUT_DIR / "aec_conditional_within_clinical_score_summary.csv", index=False)
    plot_results(bin_df, pooled_df)
    with open(OUT_DIR / "aec_within_clinical_score_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Patients were split into 10 clinical-score bins. Within each bin, AEC-high and AEC-low were defined by the within-bin median AEC score. Bin fixed-effect logistic regression tested AEC-high vs AEC-low while conditioning on clinical-score bin.",
                "scores": list(scores.keys()),
                "summary": pooled_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(pooled_df.to_string(index=False))
    print(OUT_DIR / "aec_high_low_within_clinical_deciles.csv")
    print(OUT_DIR / "aec_conditional_within_clinical_score_summary.csv")


if __name__ == "__main__":
    main()
