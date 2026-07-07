from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    risk_direction,
    standardize_train_test,
)
from aec_sex_subgroup_gate_performance import choose_feature_settings  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_smoothing_normalization_robustness"
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def load_aec128_smoothed(path: Path, sigma: float) -> dict:
    """AEC_128 곡선을 가우시안 평활화(폭 sigma, 0이면 평활화 없음) 후 정규화하고, 라벨과 함께 반환."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    if sigma > 0:
        raw = ndimage.gaussian_filter1d(raw, sigma=sigma, axis=1, mode="nearest")
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": norm, "y": y}


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측(불리언)과 실제 라벨로부터 정확도·민감도·특이도·균형정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
    }


def exact_p(a: int, b: int) -> float:
    """두 카운트에 대한 이항 정확검정(양측) p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상 양성 판정을 최종 판정으로 바꿨을 때 민감도손실/특이도이득/정확도변화에 대한 대응 이항검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
        "tp_lost_n": sens_loss,
        "fp_removed_n": spec_gain,
    }


def fisher_event_p(y: np.ndarray, kept: np.ndarray, deesc: np.ndarray) -> float:
    """유지군과 하향조정군의 사건 발생률 차이에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(kept & (y == 1)))
    b = int(np.sum(kept & (y == 0)))
    c = int(np.sum(deesc & (y == 1)))
    d = int(np.sum(deesc & (y == 0)))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def metric_row(
    sigma: float,
    cohort: str,
    op: str,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    deesc: np.ndarray,
) -> dict:
    """평활화 폭(sigma)·코호트·민감도 목표별로, 임상 단독 대비 하향조정 후 민감도손실/특이도이득/정확도변화와 하향조정군 통계를 한 행으로 정리."""
    final_pos = clinical_pos & ~deesc
    base = counts(y, clinical_pos)
    post = counts(y, final_pos)
    return {
        "sigma": sigma,
        "cohort": cohort,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_event_p(y, final_pos, deesc),
        **paired_pvalues(y, clinical_pos, final_pos),
    }


def evaluate_sigma(sigma: float) -> pd.DataFrame:
    """한 평활화 폭(sigma)에 대해 g1090/sdata를 다시 로드·정규화하고, 이미 정해둔(choose_feature_settings)
    특징 게이트들의 다수결 투표(2표 이상)로 하향조정군을 정해, 코호트x민감도목표별 지표 표를 만듦."""
    g = load_aec128_smoothed(DATA_DIR / "g1090.xlsx", sigma)
    s = load_aec128_smoothed(DATA_DIR / "sdata.xlsx", sigma)
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)
    cg, cs, _ = clinical_scores(g, s)
    thresholds = {op: threshold_for_min_sensitivity(yg, cg, target) for op, target in OPS}

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(yg, cg, xg)
    x_by = {"Gangnam": xg * direction[None, :], "Sinchon": xs * direction[None, :]}
    y_by = {"Gangnam": yg, "Sinchon": ys}
    c_by = {"Gangnam": cg, "Sinchon": cs}
    name_to_idx = {name: i for i, name in enumerate(names)}
    settings = choose_feature_settings()

    rows = []
    for cohort in ["Gangnam", "Sinchon"]:
        for op, _ in OPS:
            th = thresholds[op]
            clinical_pos = c_by[cohort] >= th
            votes = np.zeros(len(y_by[cohort]), dtype=int)
            for _, r in settings.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((c_by[cohort] - th) / float(r["width"])) ** 2)
                gate = c_by[cohort] + float(r["lambda"]) * boundary * x_by[cohort][:, idx]
                votes += (clinical_pos & (gate < th)).astype(int)
            deesc = clinical_pos & (votes >= 2)
            rows.append(metric_row(sigma, cohort, op, y_by[cohort], clinical_pos, deesc))
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    """평활화 폭x코호트별로 여러 민감도 목표에 걸친 최소/평균 지표(최악의 경우 포함)를 요약."""
    return (
        details.groupby(["sigma", "cohort"])
        .agg(
            min_p_loss=("sensitivity_loss_p_exact", "min"),
            max_sens_loss=("sensitivity_loss", "max"),
            min_spec_gain=("specificity_gain", "min"),
            mean_spec_gain=("specificity_gain", "mean"),
            min_delta_ba=("delta_balanced_accuracy", "min"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            min_deesc_event_rate=("deesc_event_rate", "min"),
            max_deesc_event_rate=("deesc_event_rate", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 6/30에 고른 특징 게이트 조합(choose_feature_settings)이,
    AEC 곡선을 얼마나 평활화하느냐(가우시안 필터 폭)에 따라 결과가 크게 흔들리는가? — 전처리
    선택에 대한 강건성 점검):

    1. SIGMAS(0, 0.5, 1.0, 1.5, 2.0)의 평활화 폭마다 evaluate_sigma를 실행: g1090/sdata를 그 폭으로
       다시 평활화·정규화해 로드하고, 사전에 정해둔 특징 게이트들의 다수결 투표로 임상 양성군 중
       하향조정군을 정해, 코호트(강남/신촌) x 5개 민감도 목표(S80~S90)별 지표를 계산한다.
    2. 모든 sigma의 결과를 합쳐 상세 표로 CSV 저장.
    3. sigma x 코호트별로 여러 민감도 목표에 걸친 최소/평균 지표(최악의 경우 포함)를 요약해 CSV 저장
       — 평활화 정도를 바꿔도 하향조정 규칙의 안전성(민감도손실 유의성, 특이도이득)이 유지되는지 확인.
    4. 요약표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details = pd.concat([evaluate_sigma(sigma) for sigma in SIGMAS], ignore_index=True)
    summary = summarize(details)
    details.to_csv(OUT_DIR / "smoothing_sigma_deescalation_details.csv", index=False)
    summary.to_csv(OUT_DIR / "smoothing_sigma_range_summary.csv", index=False)
    print("\nSMOOTHING SIGMA RANGE SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 여러 평활화 폭(sigma)에 대해 evaluate_sigma로 하향조정 지표를 계산하고,
    # 그 결과를 상세/요약 CSV로 저장한 뒤 요약을 콘솔에 출력하는 강건성 점검 파이프라인이 수행된다.
    main()
