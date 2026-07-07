from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_sex_subgroup_gate_performance"
RANKED_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_midrange_feature_refit" / "midrange_feature_search_train_ranked.csv"
FEATURE_SHORTS = [
    "visual_trough_depth__early_041_056__mid_053_076__tail_101_128",
    "norm_slope_085_096_mean",
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
]
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def choose_feature_settings() -> pd.DataFrame:
    """aec_midrange_feature_refit의 순위표에서 미리 지정한 4개 "중후반 구간" 특징(FEATURE_SHORTS)의 최적 폭·람다 설정을 가져온다."""
    ranked = pd.read_csv(RANKED_PATH)
    ranked["feature_short"] = (
        ranked["feature"].astype(str).str.replace("bank_norm__", "", regex=False).str.replace("midrange__", "", regex=False)
    )
    rows = []
    for short in FEATURE_SHORTS:
        rows.append(ranked[ranked["feature_short"].eq(short)].iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def binary_counts(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """mask로 지정한 부분집합(예: 남성 또는 여성)에서만 tp/fn/tn/fp와 정확도, 민감도, 특이도, PPV, NPV를 계산."""
    yy = y[mask].astype(bool)
    pp = pred[mask].astype(bool)
    tp = int(np.sum(yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    fp = int(np.sum(~yy & pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "n": int(mask.sum()),
        "events": int(yy.sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "accuracy": float((tp + tn) / max(1, len(yy))),
        "sensitivity": float(sens) if np.isfinite(sens) else np.nan,
        "specificity": float(spec) if np.isfinite(spec) else np.nan,
        "ppv": float(ppv) if np.isfinite(ppv) else np.nan,
        "npv": float(npv) if np.isfinite(npv) else np.nan,
    }


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray, mask: np.ndarray) -> dict:
    """지정한 성별 부분집합 내에서, 임상단독 판정과 게이트적용 후 최종판정을 비교해 민감도손실/특이도이득/정확도변화의 대응 정확검정 p값과 놓친/제거된 표본수를 계산."""
    yy = y.astype(bool)
    m = mask.astype(bool)
    sens_loss = int(np.sum(m & yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(m & yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(m & ~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(m & ~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(m & ~cc & fc))
    acc_loss = int(np.sum(m & cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
        "tp_lost_n": sens_loss,
        "fp_removed_n": spec_gain,
    }


def fisher_event_p(y: np.ndarray, kept: np.ndarray, deesc: np.ndarray, mask: np.ndarray) -> float:
    """지정한 성별 부분집합 내에서 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    m = mask.astype(bool)
    a = int(np.sum(m & kept & (y == 1)))
    b = int(np.sum(m & kept & (y == 0)))
    c = int(np.sum(m & deesc & (y == 1)))
    d = int(np.sum(m & deesc & (y == 0)))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def subgroup_row(
    cohort: str,
    sex: str,
    op: str,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    deesc: np.ndarray,
    sex_mask: np.ndarray,
) -> dict:
    """한 코호트·성별·운영점 조합에 대해 임상단독 대비 게이트적용 후 성능 변화(민감도손실/특이도이득/정확도/균형정확도)와 하향조정군 사건비율·각종 p값을 계산."""
    final_pos = clinical_pos & ~deesc
    base = binary_counts(y, clinical_pos, sex_mask)
    post = binary_counts(y, final_pos, sex_mask)
    clinical_positive = sex_mask & clinical_pos
    kept = sex_mask & final_pos
    deesc_sub = sex_mask & deesc
    return {
        "cohort": cohort,
        "sex": sex,
        "operating_point": op,
        "n": int(sex_mask.sum()),
        "events": int(y[sex_mask].sum()),
        "clinical_positive_n": int(clinical_positive.sum()),
        "clinical_positive_events": int(y[clinical_positive].sum()),
        "clinical_positive_event_rate": float(y[clinical_positive].mean()) if clinical_positive.any() else np.nan,
        "clinical_sensitivity": base["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_sensitivity": post["sensitivity"],
        "post_specificity": post["specificity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["sensitivity"] * 0.5 + base["specificity"] * 0.5,
        "post_balanced_accuracy": post["sensitivity"] * 0.5 + post["specificity"] * 0.5,
        "delta_balanced_accuracy": (post["sensitivity"] + post["specificity"] - base["sensitivity"] - base["specificity"]) * 0.5,
        "deesc_n": int(deesc_sub.sum()),
        "deesc_events": int(y[deesc_sub].sum()),
        "deesc_event_rate": float(y[deesc_sub].mean()) if deesc_sub.any() else np.nan,
        "deesc_event_fisher_p": fisher_event_p(y, kept, deesc_sub, sex_mask),
        **paired_pvalues(y, clinical_pos, final_pos, sex_mask),
    }


def interaction_row(
    cohort: str,
    op: str,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    deesc: np.ndarray,
    male: np.ndarray,
    female: np.ndarray,
) -> dict:
    """남성과 여성 사이에 하향조정 효과가 다르게 나타나는지 확인하기 위해, 임상 위음성(진짜양성 중 임상양성)군에서의 민감도손실 하향조정률과 임상 진짜음성군에서의 특이도이득 하향조정률 각각에 대해 성별 간 Fisher 교호작용검정 p값을 계산."""
    # Specificity gain interaction: among clinical false positives, compare de-escalation rate by sex.
    tn_pool = clinical_pos & (y == 0)
    m_fp_removed = int(np.sum(male & tn_pool & deesc))
    m_fp_kept = int(np.sum(male & tn_pool & ~deesc))
    f_fp_removed = int(np.sum(female & tn_pool & deesc))
    f_fp_kept = int(np.sum(female & tn_pool & ~deesc))
    spec_interaction_p = (
        float(stats.fisher_exact([[m_fp_removed, m_fp_kept], [f_fp_removed, f_fp_kept]])[1])
        if (m_fp_removed + m_fp_kept and f_fp_removed + f_fp_kept)
        else np.nan
    )

    # Sensitivity loss interaction: among clinical true positives, compare de-escalation rate by sex.
    tp_pool = clinical_pos & (y == 1)
    m_tp_lost = int(np.sum(male & tp_pool & deesc))
    m_tp_kept = int(np.sum(male & tp_pool & ~deesc))
    f_tp_lost = int(np.sum(female & tp_pool & deesc))
    f_tp_kept = int(np.sum(female & tp_pool & ~deesc))
    sens_interaction_p = (
        float(stats.fisher_exact([[m_tp_lost, m_tp_kept], [f_tp_lost, f_tp_kept]])[1])
        if (m_tp_lost + m_tp_kept and f_tp_lost + f_tp_kept)
        else np.nan
    )
    return {
        "cohort": cohort,
        "operating_point": op,
        "male_fp_removed": m_fp_removed,
        "male_fp_clinical_positive": m_fp_removed + m_fp_kept,
        "female_fp_removed": f_fp_removed,
        "female_fp_clinical_positive": f_fp_removed + f_fp_kept,
        "spec_gain_interaction_fisher_p": spec_interaction_p,
        "male_tp_lost": m_tp_lost,
        "male_tp_clinical_positive": m_tp_lost + m_tp_kept,
        "female_tp_lost": f_tp_lost,
        "female_tp_clinical_positive": f_tp_lost + f_tp_kept,
        "sens_loss_interaction_fisher_p": sens_interaction_p,
    }


def summarize_ranges(detail: pd.DataFrame) -> pd.DataFrame:
    """코호트x성별 조합마다 5개 운영점에 걸친 각 지표의 최솟값/최댓값/평균 범위를 요약."""
    return (
        detail.groupby(["cohort", "sex"])
        .agg(
            n=("n", "first"),
            events=("events", "first"),
            min_clinical_sensitivity=("clinical_sensitivity", "min"),
            max_clinical_sensitivity=("clinical_sensitivity", "max"),
            min_clinical_specificity=("clinical_specificity", "min"),
            max_clinical_specificity=("clinical_specificity", "max"),
            min_post_sensitivity=("post_sensitivity", "min"),
            max_post_sensitivity=("post_sensitivity", "max"),
            min_post_specificity=("post_specificity", "min"),
            max_post_specificity=("post_specificity", "max"),
            min_sensitivity_loss=("sensitivity_loss", "min"),
            max_sensitivity_loss=("sensitivity_loss", "max"),
            worst_p_loss=("sensitivity_loss_p_exact", "min"),
            min_specificity_gain=("specificity_gain", "min"),
            max_specificity_gain=("specificity_gain", "max"),
            mean_specificity_gain=("specificity_gain", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            min_deesc_event_rate=("deesc_event_rate", "min"),
            max_deesc_event_rate=("deesc_event_rate", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_block_or_consensus_search류의 "중후반 구간" 4특징
    2-of-4 합의 하향조정 게이트가 남성과 여성에서 서로 다르게 작동하지 않는지 - 즉 성별에 따른
    공평성/일관성 문제가 있는지 확인):

    1. g1090(Gangnam)/sdata(Sinchon)를 로드하고 choose_feature_settings로 4개 특징의 폭·람다
       설정을 가져온 뒤, 임상점수·특징뱅크·표준화값·방향을 계산하고 Gangnam 기준 5개 운영점
       (S80~S90)의 임상 임계값을 구한다.
    2. 각 코호트·운영점마다 2-of-4 합의 게이트로 하향조정 여부를 정한 뒤, 환자 성별(PatientSex)로
       남성/여성 부분집합을 나누어 subgroup_row로 각 성별 내에서의 임상단독 대비 성능 변화를
       계산하고, interaction_row로 남녀 간 하향조정률 차이에 대한 Fisher 교호작용검정을 수행.
    3. 성별·운영점별 상세 성능표, 교호작용검정 결과표를 CSV로 저장.
    4. summarize_ranges로 코호트x성별별 5개 운영점에 걸친 지표 범위를 요약해 CSV로 저장하고,
       범위 요약과 교호작용검정 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    settings = choose_feature_settings()

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    cg, cs, _ = clinical_scores(g, s)
    direction = risk_direction(g["y"].astype(int), cg, xg)
    name_to_idx = {name: i for i, name in enumerate(names)}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], cg, target) for op, target in OPS}

    datasets = {
        "Gangnam internal": {"data": g, "clinical": cg, "features": xg},
        "Sinchon external": {"data": s, "clinical": cs, "features": xs},
    }

    detail_rows = []
    interaction_rows = []
    for cohort, obj in datasets.items():
        d = obj["data"]
        y = d["y"].astype(int)
        clinical = obj["clinical"]
        feature_matrix = obj["features"]
        sex = d["meta"]["PatientSex"].astype(str).str.upper().to_numpy()
        male = sex == "M"
        female = sex == "F"
        for op, _ in OPS:
            th = thresholds[op]
            clinical_pos = clinical >= th
            votes = np.zeros(len(y), dtype=int)
            for _, r in settings.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((clinical - th) / float(r["width"])) ** 2)
                gate = clinical + float(r["lambda"]) * boundary * feature_matrix[:, idx] * direction[idx]
                votes += (clinical_pos & (gate < th)).astype(int)
            deesc = clinical_pos & (votes >= 2)
            detail_rows.append(subgroup_row(cohort, "Male", op, y, clinical_pos, deesc, male))
            detail_rows.append(subgroup_row(cohort, "Female", op, y, clinical_pos, deesc, female))
            interaction_rows.append(interaction_row(cohort, op, y, clinical_pos, deesc, male, female))

    detail = pd.DataFrame(detail_rows)
    interactions = pd.DataFrame(interaction_rows)
    summary = summarize_ranges(detail)
    detail.to_csv(OUT_DIR / "sex_subgroup_operating_point_details.csv", index=False)
    interactions.to_csv(OUT_DIR / "sex_subgroup_interaction_tests.csv", index=False)
    summary.to_csv(OUT_DIR / "sex_subgroup_range_summary.csv", index=False)

    print("\nRANGE SUMMARY")
    print(summary.to_string(index=False))
    print("\nINTERACTION TESTS")
    print(interactions.to_string(index=False))


if __name__ == "__main__":
    main()
