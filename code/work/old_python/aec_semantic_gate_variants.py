from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    SIGMA,
    auc_with_p,
    clinical_scores,
    counts,
    deesc_metric_row,
    load_dataset,
    plot_locked,
    risk_direction,
    standardize_train_test,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_semantic_gate_variants"
WIDTHS = [0.50, 0.70, 0.90]
LAMBDAS = [0.40, 0.55, 0.70]


def seg_stat(x: np.ndarray, start: int, end: int, stat: str) -> np.ndarray:
    """지정 구간(start~end)에서 평균/표준편차/최소/최대 중 하나를 계산."""
    block = x[:, start - 1 : end]
    if stat == "mean":
        return np.mean(block, axis=1)
    if stat == "sd":
        return np.std(block, axis=1)
    if stat == "min":
        return np.min(block, axis=1)
    if stat == "max":
        return np.max(block, axis=1)
    raise ValueError(stat)


def build_semantic_features(norm: np.ndarray) -> pd.DataFrame:
    """다른 스크립트들의 미세 최적화된 좁은 윈도우 대신, 후반부(80~120 부근)를 폭넓게 아우르는 "의미론적으로 이해하기 쉬운" 후반부 동역학(기울기/곡률 변동성) 대안 특징들을 계산."""
    d1 = np.diff(norm, axis=1)
    d2 = np.diff(d1, axis=1)
    rows: dict[str, np.ndarray] = {}
    # Broad semantic alternatives to the optimized tiny windows.
    rows["late_dynamics_slope_sd_080_096"] = seg_stat(d1, 80, 96, "sd")
    rows["late_dynamics_abs_slope_max_090_104"] = seg_stat(np.abs(d1), 90, 104, "max")
    rows["late_dynamics_curv_mean_096_112"] = seg_stat(d2, 96, 112, "mean")
    rows["late_dynamics_curv_sd_092_108"] = seg_stat(d2, 92, 108, "sd")

    rows["late_dynamics_slope_sd_080_110"] = seg_stat(d1, 80, 110, "sd")
    rows["late_dynamics_abs_slope_mean_080_110"] = seg_stat(np.abs(d1), 80, 110, "mean")
    rows["late_dynamics_abs_slope_max_080_110"] = seg_stat(np.abs(d1), 80, 110, "max")
    rows["late_dynamics_curv_sd_080_110"] = seg_stat(d2, 80, 110, "sd")
    rows["late_dynamics_abs_curv_mean_080_110"] = seg_stat(np.abs(d2), 80, 110, "mean")

    rows["late_dynamics_slope_sd_082_120"] = seg_stat(d1, 82, 120, "sd")
    rows["late_dynamics_abs_slope_mean_082_120"] = seg_stat(np.abs(d1), 82, 120, "mean")
    rows["late_dynamics_curv_sd_082_120"] = seg_stat(d2, 82, 120, "sd")

    # A single simple morphology score candidate: overall late dynamic vitality.
    df = pd.DataFrame(rows)
    return df


def make_deesc(clinical_z: np.ndarray, feature_z: np.ndarray, th: float, width: float, lam: float) -> np.ndarray:
    """단일 특징 기반 de-escalation 게이트: 임계값 근방 가우시안 가중치로 특징을 반영한 게이트 점수가 임계값 아래면 강등."""
    cpos = clinical_z >= th
    boundary = np.exp(-0.5 * ((clinical_z - th) / width) ** 2)
    gate = clinical_z + lam * boundary * feature_z
    return cpos & (gate < th)


def eval_rule(rule_label: str, feature_names: list[str], xg: np.ndarray, xs: np.ndarray, names: list[str], g: dict, s: dict, c_g: np.ndarray, c_s: np.ndarray, thresholds: dict[str, float], width: float, lam: float, k: int | None = None) -> list[dict]:
    """지정된 의미론적 특징 묶음을, k가 None이면 "평균 점수" 방식, k가 주어지면 "k표 이상 합의" 방식으로 de-escalation 게이트를 적용해 내부/외부 x 모든 운영점 결과 행을 계산."""
    idx = [names.index(f) for f in feature_names]
    rows = []
    for dataset, d, c, x in [("g1090_internal", g, c_g, xg), ("sdata_external", s, c_s, xs)]:
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c >= th
            sigs = [make_deesc(c, x[:, j], th, width, lam) for j in idx]
            if k is None:
                score = np.mean(x[:, idx], axis=1)
                deesc = make_deesc(c, score, th, width, lam)
                label = rule_label
            else:
                deesc = cpos & (np.vstack(sigs).sum(axis=0) >= k)
                label = f"{k}-of-{len(idx)} {rule_label}"
            rows.append(deesc_metric_row(dataset, label, " + ".join(feature_names), op, d["y"], cpos, deesc))
    return rows


def summarize(rows: list[dict], dataset: str) -> dict:
    """지정된 데이터셋의 여러 운영점 결과 중 가장 나쁜 경우(최소 p값, 최대 민감도손실 등)를 요약통계로 압축."""
    sub = [r for r in rows if r["dataset"] == dataset]
    return {
        f"{dataset}_min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in sub])),
        f"{dataset}_max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in sub])),
        f"{dataset}_min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in sub])),
        f"{dataset}_mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        f"{dataset}_max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in sub])),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_lock_smoothed_deesc_gate가 찾은 미세 최적화된 좁은 윈도우
    대신, 임상의가 이해하기 쉬운 "후반부(80~120 부근) 넓은 구간의 기울기/곡률 변동성" 같은 광의적
    특징 묶음으로도 비슷한 de-escalation 효과를 낼 수 있는가? — 해석 가능성과 성능의 트레이드오프 탐색):

    1. g1090/sdata를 로드하고 3가지 폭의 "후반부 동역학" 의미론적 특징 묶음(4개/4개/3개 구성요소)을
       만들어 위험 방향으로 정렬.
    2. 각 특징 묶음 x 게이트 폭(WIDTHS) x 람다(LAMBDAS) x 방식(평균점수 또는 k표 합의) 조합을 모두
       평가해, 내부+외부 최소 특이도이득/최대 민감도손실 기반 점수로 순위를 매김.
    3. 전체 조합 요약과 상세 결과를 CSV로 저장하고, 가장 점수가 높은 조합을 골라 그 상세 결과만 별도
       CSV로 저장.
    4. 최고 조합의 운영점별 효과를 plot_locked로 시각화하고, 요약을 JSON으로 저장한 뒤 콘솔에 상위
       20개 조합과 최고 조합의 상세 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_semantic_features(g["norm"])
    fs = build_semantic_features(s["norm"])
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]

    feature_sets = {
        "broad_4_component_late_vitality": [
            "late_dynamics_slope_sd_080_096",
            "late_dynamics_abs_slope_max_090_104",
            "late_dynamics_curv_mean_096_112",
            "late_dynamics_curv_sd_092_108",
        ],
        "very_broad_4_component_080_110": [
            "late_dynamics_slope_sd_080_110",
            "late_dynamics_abs_slope_mean_080_110",
            "late_dynamics_abs_slope_max_080_110",
            "late_dynamics_curv_sd_080_110",
        ],
        "ultra_broad_3_component_082_120": [
            "late_dynamics_slope_sd_082_120",
            "late_dynamics_abs_slope_mean_082_120",
            "late_dynamics_curv_sd_082_120",
        ],
    }
    summary_rows = []
    detail_rows = []
    for label, feats in feature_sets.items():
        for width in WIDTHS:
            for lam in LAMBDAS:
                for mode in ["mean_score", "consensus"]:
                    k = None
                    if mode == "consensus":
                        k = 2 if len(feats) >= 4 else 2
                    rows = eval_rule(label + "_" + mode, feats, xg, xs, names, g, s, c_g, c_s, thresholds, width, lam, k)
                    sg = summarize(rows, "g1090_internal")
                    ss = summarize(rows, "sdata_external")
                    score = (
                        2.0 * min(sg["g1090_internal_min_spec_gain"], ss["sdata_external_min_spec_gain"])
                        + min(sg["g1090_internal_mean_spec_gain"], ss["sdata_external_mean_spec_gain"])
                        - 0.4 * max(sg["g1090_internal_max_sens_loss"], ss["sdata_external_max_sens_loss"])
                    )
                    summary_rows.append({"label": label, "mode": mode, "width": width, "lambda": lam, "score": score, **sg, **ss})
                    for r in rows:
                        rr = dict(r)
                        rr["label"] = label
                        rr["mode"] = mode
                        rr["width"] = width
                        rr["lambda"] = lam
                        detail_rows.append(rr)
    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    details = pd.DataFrame(detail_rows)
    summary.to_csv(OUT_DIR / "semantic_gate_variant_summary.csv", index=False)
    details.to_csv(OUT_DIR / "semantic_gate_variant_details.csv", index=False)
    best = summary.iloc[0]
    best_details = details[
        details["label"].eq(best["label"])
        & details["mode"].eq(best["mode"])
        & details["width"].eq(best["width"])
        & details["lambda"].eq(best["lambda"])
    ].copy()
    best_details.to_csv(OUT_DIR / "semantic_gate_best_details.csv", index=False)
    plot_locked(best_details, OUT_DIR / "semantic_gate_best_plot.png")
    (OUT_DIR / "semantic_gate_best_summary.json").write_text(json.dumps(best.to_dict(), indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))
    print("\nBEST DETAILS")
    show = [
        "dataset",
        "operating_point",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    print(best_details[show].to_string(index=False))


if __name__ == "__main__":
    # 미세 최적화된 좁은 윈도우 대신 해석하기 쉬운 후반부 광의적 특징 묶음으로 de-escalation
    # 게이트 대안을 비교하는 파이프라인을 실행한다.
    main()
