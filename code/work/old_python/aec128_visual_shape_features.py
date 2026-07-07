from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

from aec128_common_shape_feature import FILES, OUT_DIR as COMMON_OUT_DIR, feature_stats, load_aec128, summarize_feature


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_visual_shape_features"
MID_SLICE = slice(41, 78)  # 1-based 42:78
LATE_SLICE = slice(99, 128)  # 1-based 100:128


def smooth_curves(x: np.ndarray) -> np.ndarray:
    """Savitzky-Golay 필터로 점 노이즈만 제거 (골짜기/봉우리 위치는 그대로 유지되도록 작은 윈도우 사용)."""
    # Window 9 is deliberately small: enough to remove point noise without moving the visible trough/peak.
    return savgol_filter(x, window_length=9, polyorder=2, axis=1, mode="interp")


def weighted_centroid(indices_1based: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """가중치(weights)로 가중평균한 위치(무게중심)를 계산 (가중치 합이 0이면 NaN)."""
    denom = np.sum(weights, axis=1)
    out = np.full(weights.shape[0], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = (weights[ok] @ indices_1based) / denom[ok]
    return out


def extract_visual_features(x_norm: np.ndarray) -> pd.DataFrame:
    """평활화된 곡선의 중간 구간(42:78) 최저점과 후반 구간(100:128) 최고점을 찾아, 골짜기 깊이/봉우리
    높이/반등 폭·기울기/결핍·초과 면적 및 무게중심 등, 눈으로 본 모양을 수치화한 특징들을 계산."""
    s = smooth_curves(x_norm)
    idx = np.arange(1, 129, dtype=float)

    mid = s[:, MID_SLICE]
    late = s[:, LATE_SLICE]
    mid_idx = idx[MID_SLICE]
    late_idx = idx[LATE_SLICE]

    mid_min_local = np.argmin(mid, axis=1)
    late_max_local = np.argmax(late, axis=1)
    mid_valley_pos = mid_idx[mid_min_local]
    late_peak_pos = late_idx[late_max_local]
    mid_valley_value = mid[np.arange(mid.shape[0]), mid_min_local]
    late_peak_value = late[np.arange(late.shape[0]), late_max_local]

    mid_deficit = np.maximum(1.0 - mid, 0.0)
    late_excess = np.maximum(late - 1.0, 0.0)

    mid_deficit_area = np.mean(mid_deficit, axis=1)
    late_excess_area = np.mean(late_excess, axis=1)
    mid_deficit_centroid = weighted_centroid(mid_idx, mid_deficit)
    late_excess_centroid = weighted_centroid(late_idx, late_excess)

    rebound_interval = late_peak_pos - mid_valley_pos
    rebound_height = late_peak_value - mid_valley_value
    rebound_slope = rebound_height / np.maximum(rebound_interval, 1.0)

    mid_mean = np.mean(mid, axis=1)
    late_mean = np.mean(late, axis=1)

    visual_rebound_score = (
        (1.0 - mid_valley_value)
        + (late_peak_value - 1.0)
        + (late_mean - mid_mean)
        + mid_deficit_area
        + late_excess_area
    )

    return pd.DataFrame(
        {
            "aec128_mid_valley_value": mid_valley_value,
            "aec128_mid_valley_depth_below_1": 1.0 - mid_valley_value,
            "aec128_mid_valley_pos": mid_valley_pos,
            "aec128_late_peak_value": late_peak_value,
            "aec128_late_peak_excess_above_1": late_peak_value - 1.0,
            "aec128_late_peak_pos": late_peak_pos,
            "aec128_rebound_height_peak_minus_valley": rebound_height,
            "aec128_rebound_interval_points": rebound_interval,
            "aec128_rebound_slope_per_point": rebound_slope,
            "aec128_mid_deficit_area_below_1": mid_deficit_area,
            "aec128_late_excess_area_above_1": late_excess_area,
            "aec128_mid_deficit_centroid": mid_deficit_centroid,
            "aec128_late_excess_centroid": late_excess_centroid,
            "aec128_late_minus_mid_mean": late_mean - mid_mean,
            "aec128_visual_rebound_score": visual_rebound_score,
        }
    )


def summarize_visual_features(features_by_cohort: dict[str, pd.DataFrame], datasets: dict[str, dict]) -> pd.DataFrame:
    """각 시각 특징에 대해 코호트별 통계와 "두 코호트에서 방향이 일치하는지" 여부를 계산하고, pooled Mann-Whitney p값 기준으로 정렬."""
    rows = []
    for name in features_by_cohort["g1090"].columns:
        vals = {cohort: df[name].to_numpy(dtype=float) for cohort, df in features_by_cohort.items()}
        for row in summarize_feature(name, vals, datasets):
            row["direction_consistent"] = np.sign(
                feature_stats(vals["g1090"], datasets["g1090"]["y"])["delta_low_minus_nonlow"]
            ) == np.sign(feature_stats(vals["sdata"], datasets["sdata"]["y"])["delta_low_minus_nonlow"])
            rows.append(row)
    out = pd.DataFrame(rows)
    pooled = out[out["cohort"].eq("pooled")].copy()
    order = pooled.sort_values("mannwhitney_p")["feature"].tolist()
    out["feature_order"] = out["feature"].map({f: i for i, f in enumerate(order)})
    return out.sort_values(["feature_order", "cohort"]).drop(columns=["feature_order"])


def plot_schematic(datasets: dict[str, dict], features_by_cohort: dict[str, pd.DataFrame]) -> None:
    """골짜기/봉우리/반등 개념을 화살표와 음영으로 표시한 설명용 그래프와, 두 코호트의 최종 rebound 점수 분포 히스토그램을 PNG로 저장."""
    pooled_x = np.vstack([d["x"] for d in datasets.values()])
    pooled_y = np.concatenate([d["y"] for d in datasets.values()])
    smooth = smooth_curves(pooled_x)
    low_mean = smooth[pooled_y].mean(axis=0)
    non_mean = smooth[~pooled_y].mean(axis=0)
    xgrid = np.arange(1, 129)

    mid = low_mean[MID_SLICE]
    late = low_mean[LATE_SLICE]
    mid_idx = xgrid[MID_SLICE]
    late_idx = xgrid[LATE_SLICE]
    valley_pos = int(mid_idx[np.argmin(mid)])
    peak_pos = int(late_idx[np.argmax(late)])
    valley_y = float(low_mean[valley_pos - 1])
    peak_y = float(low_mean[peak_pos - 1])

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.plot(xgrid, non_mean, color="#2F6F73", lw=2.2, label="Non-low SMI mean, smoothed")
    ax.plot(xgrid, low_mean, color="#C84630", lw=2.4, label="Low SMI mean, smoothed")
    ax.axhline(1.0, color="#555555", lw=1.0, ls="--", alpha=0.75, label="Patient mean baseline")
    ax.axvspan(42, 78, color="#2F6F73", alpha=0.12, label="Mid valley search window 42:78")
    ax.axvspan(100, 128, color="#C84630", alpha=0.12, label="Late peak search window 100:128")
    ax.scatter([valley_pos, peak_pos], [valley_y, peak_y], color=["#2F6F73", "#C84630"], s=65, zorder=5)
    ax.annotate(
        "mid valley",
        xy=(valley_pos, valley_y),
        xytext=(valley_pos - 20, valley_y - 0.065),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        fontsize=10,
    )
    ax.annotate(
        "late peak",
        xy=(peak_pos, peak_y),
        xytext=(peak_pos - 24, peak_y + 0.055),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        fontsize=10,
    )
    ax.plot([valley_pos, peak_pos], [valley_y, peak_y], color="#333333", lw=1.5, ls=":")
    ax.text(
        (valley_pos + peak_pos) / 2,
        (valley_y + peak_y) / 2 + 0.018,
        "rebound height / interval = slope",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )
    ax.fill_between(mid_idx, low_mean[MID_SLICE], 1.0, where=low_mean[MID_SLICE] < 1.0, color="#2F6F73", alpha=0.20)
    ax.fill_between(late_idx, 1.0, low_mean[LATE_SLICE], where=low_mean[LATE_SLICE] > 1.0, color="#C84630", alpha=0.20)
    ax.set_xlabel("AEC_128 point index")
    ax.set_ylabel("AEC / patient mean AEC")
    ax.set_title("Visual AEC_128 shape features: valley, rebound, and late excess", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_visual_feature_schematic.png", dpi=200)
    plt.close(fig)

    # Distribution of the final visual score in both cohorts.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    for ax, cohort in zip(axes, ["g1090", "sdata"]):
        y = datasets[cohort]["y"]
        score = features_by_cohort[cohort]["aec128_visual_rebound_score"].to_numpy(dtype=float)
        ax.hist(score[~y], bins=36, density=True, alpha=0.55, color="#2F6F73", label="Non-low SMI")
        ax.hist(score[y], bins=24, density=True, alpha=0.58, color="#C84630", label="Low SMI")
        ax.set_title(cohort, loc="left", fontweight="bold")
        ax.set_xlabel("AEC_128 visual rebound score")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_visual_rebound_score_distribution.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec128_common_shape_feature에서 찾은 "중간 저점/후반 고점"
    패턴을, 딱딱한 구간 평균이 아니라 "눈으로 보는 골짜기·봉우리·반등" 형태로 다시 정의해도
    똑같이 저근감소증군과 관련이 있는가?):

    1. g1090/sdata를 load_aec128로 로드하고, extract_visual_features로 골짜기 깊이/봉우리 높이/
       반등 폭·기울기/결핍·초과 면적 등 15개 시각적 특징을 계산해 환자별 CSV로 저장.
    2. summarize_visual_features로 각 특징의 코호트별 통계와 방향 일치 여부를 계산해 CSV로 저장.
    3. plot_schematic으로 평균 곡선 위에 골짜기/봉우리/반등을 화살표로 표시한 설명 그래프와,
       최종 종합 점수(visual_rebound_score)의 두 코호트 분포 히스토그램을 저장.
    4. 정규화·평활화 방법, 탐색 구간, 각 특징의 수식 정의를 JSON으로 저장하고, 핵심 특징들의
       통계표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    features_by_cohort = {cohort: extract_visual_features(d["x"]) for cohort, d in datasets.items()}

    for cohort, df in features_by_cohort.items():
        out = df.copy()
        out.insert(0, "low_smi", datasets[cohort]["y"].astype(int))
        out.to_csv(OUT_DIR / f"{cohort}_aec128_visual_shape_features_patient_level.csv", index=False)

    summary_df = summarize_visual_features(features_by_cohort, datasets)
    summary_df.to_csv(OUT_DIR / "aec128_visual_shape_feature_stats.csv", index=False)
    plot_schematic(datasets, features_by_cohort)

    definitions = {
        "normalization": "N_i(j) = AEC_i(j) / mean(AEC_i(1:128))",
        "smoothing": "Savitzky-Golay smoothing, window_length=9, polyorder=2, applied after patient-level normalization.",
        "mid_window": "AEC_128 points 42:78, selected because low SMI is lower than non-low SMI in both cohorts.",
        "late_window": "AEC_128 points 100:128, selected because low SMI is higher than non-low SMI in both cohorts.",
        "visual_features": {
            "mid_valley_depth": "1 - min(smoothed N_i(j), j in 42:78)",
            "late_peak_excess": "max(smoothed N_i(j), j in 100:128) - 1",
            "rebound_height": "late_peak_value - mid_valley_value",
            "rebound_slope": "rebound_height / (late_peak_position - mid_valley_position)",
            "mid_deficit_area": "mean(max(1 - smoothed N_i(j), 0), j in 42:78)",
            "late_excess_area": "mean(max(smoothed N_i(j) - 1, 0), j in 100:128)",
            "visual_rebound_score": "mid_valley_depth + late_peak_excess + late_minus_mid_mean + mid_deficit_area + late_excess_area",
        },
    }
    with open(OUT_DIR / "aec128_visual_shape_feature_definitions.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, ensure_ascii=False, indent=2)

    selected = summary_df[
        summary_df["feature"].isin(
            [
                "aec128_mid_valley_depth_below_1",
                "aec128_late_peak_excess_above_1",
                "aec128_rebound_height_peak_minus_valley",
                "aec128_rebound_slope_per_point",
                "aec128_mid_deficit_area_below_1",
                "aec128_late_excess_area_above_1",
                "aec128_late_minus_mid_mean",
                "aec128_visual_rebound_score",
            ]
        )
    ]
    print(selected.to_string(index=False))
    print(OUT_DIR / "aec128_visual_feature_schematic.png")
    print(OUT_DIR / "aec128_visual_shape_feature_stats.csv")


if __name__ == "__main__":
    main()
