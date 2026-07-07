from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from aec128_common_shape_feature import FILES, feature_stats, load_aec128, summarize_feature


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_cylindrical_features"


def smooth_log_profile(x_norm: np.ndarray) -> np.ndarray:
    """정규화된 곡선을 로그 스케일(상대적 반경 편차)로 바꾼 뒤 Savitzky-Golay로 평활화."""
    log_profile = np.log(np.clip(x_norm, 1e-6, None))
    return savgol_filter(log_profile, window_length=9, polyorder=2, axis=1, mode="interp")


def weighted_centroid(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """가중치 w로 축 위치 z의 가중평균(무게중심)을 계산."""
    denom = np.sum(w, axis=1)
    out = np.full(w.shape[0], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = (w[ok] @ z) / denom[ok]
    return out


def weighted_spread(z: np.ndarray, w: np.ndarray, center: np.ndarray) -> np.ndarray:
    """가중치 w로 무게중심(center) 기준 가중 표준편차(퍼짐 정도)를 계산."""
    out = np.full(w.shape[0], np.nan, dtype=float)
    denom = np.sum(w, axis=1)
    ok = denom > 0
    if np.any(ok):
        out[ok] = np.sqrt(np.sum(w[ok] * ((z[None, :] - center[ok, None]) ** 2), axis=1) / denom[ok])
    return out


def interval_mean(profile: np.ndarray, start_1: int, end_1: int) -> np.ndarray:
    """1-based 구간 [start_1, end_1]의 로그 프로파일 평균."""
    return np.mean(profile[:, start_1 - 1 : end_1], axis=1)


def interval_area_positive(profile: np.ndarray, start_1: int, end_1: int) -> np.ndarray:
    """1-based 구간에서 양수 부분(0 이상)만 취한 평균 (양의 초과 면적)."""
    return np.mean(np.maximum(profile[:, start_1 - 1 : end_1], 0.0), axis=1)


def interval_area_negative(profile: np.ndarray, start_1: int, end_1: int) -> np.ndarray:
    """1-based 구간에서 음수 부분을 양수로 뒤집어 취한 평균 (음의 결핍 면적)."""
    return np.mean(np.maximum(-profile[:, start_1 - 1 : end_1], 0.0), axis=1)


def extract_cylindrical_features(x_norm: np.ndarray) -> pd.DataFrame:
    """AEC 곡선을 원기둥의 축 방향 로그-반경 편차 프로파일로 보고, 양/음 질량·무게중심·퍼짐,
    쌍극자/사중극자 모멘트, 부피·표면적 근사, 중간결핍+후반초과 등 29개 "원기둥 형태" 특징을 계산."""
    p = smooth_log_profile(x_norm)
    z = np.linspace(0.0, 1.0, p.shape[1])
    dz = 1.0 / (p.shape[1] - 1)

    pos = np.maximum(p, 0.0)
    neg = np.maximum(-p, 0.0)
    absdev = np.abs(p)

    pos_mass = np.mean(pos, axis=1)
    neg_mass = np.mean(neg, axis=1)
    abs_mass = np.mean(absdev, axis=1)
    signed_mass = np.mean(p, axis=1)

    pos_centroid = weighted_centroid(z, pos)
    neg_centroid = weighted_centroid(z, neg)
    abs_centroid = weighted_centroid(z, absdev)
    pos_spread = weighted_spread(z, pos, pos_centroid)
    neg_spread = weighted_spread(z, neg, neg_centroid)

    # Axial moments: shape as a signed attenuation-radius deviation over a cylinder axis.
    centered_z = z - 0.5
    dipole = p @ centered_z / p.shape[1]
    quadrupole = p @ ((centered_z**2) - np.mean(centered_z**2)) / p.shape[1]

    d1 = np.gradient(p, dz, axis=1)
    d2 = np.gradient(d1, dz, axis=1)

    # Surface-of-revolution proxies. Scale is arbitrary; these are shape-only because p is log-normalized.
    radius_proxy = np.exp(p)
    rel_volume = np.mean(radius_proxy**2, axis=1)
    rel_surface = np.mean(radius_proxy * np.sqrt(1.0 + np.gradient(radius_proxy, dz, axis=1) ** 2), axis=1)

    mid_negative = interval_area_negative(p, 42, 78)
    late_positive = interval_area_positive(p, 100, 128)
    early_positive = interval_area_positive(p, 1, 32)
    late_negative = interval_area_negative(p, 100, 128)

    mid_mean = interval_mean(p, 42, 78)
    late_mean = interval_mean(p, 100, 128)
    early_mean = interval_mean(p, 1, 32)

    # The "cylindrical dipole" is a visually interpretable axial mass separation:
    # mid deficit plus caudal/late excess.
    caudal_deficit_rebound = late_positive + mid_negative
    caudal_vs_mid_signed = late_mean - mid_mean
    posterior_shift_of_positive_mass = pos_centroid
    positive_negative_separation = pos_centroid - neg_centroid

    return pd.DataFrame(
        {
            "cyl_log_pos_mass": pos_mass,
            "cyl_log_neg_mass": neg_mass,
            "cyl_log_abs_mass": abs_mass,
            "cyl_log_signed_mass": signed_mass,
            "cyl_positive_centroid_z": pos_centroid,
            "cyl_negative_centroid_z": neg_centroid,
            "cyl_abs_deviation_centroid_z": abs_centroid,
            "cyl_positive_spread_z": pos_spread,
            "cyl_negative_spread_z": neg_spread,
            "cyl_positive_negative_centroid_separation": positive_negative_separation,
            "cyl_axial_dipole_moment": dipole,
            "cyl_axial_quadrupole_moment": quadrupole,
            "cyl_mid_negative_area_42_78": mid_negative,
            "cyl_late_positive_area_100_128": late_positive,
            "cyl_late_positive_minus_mid_negative": late_positive - mid_negative,
            "cyl_late_positive_plus_mid_negative": caudal_deficit_rebound,
            "cyl_late_mean_minus_mid_mean_log": caudal_vs_mid_signed,
            "cyl_early_mean_log": early_mean,
            "cyl_mid_mean_log": mid_mean,
            "cyl_late_mean_log": late_mean,
            "cyl_early_positive_area_1_32": early_positive,
            "cyl_late_negative_area_100_128": late_negative,
            "cyl_max_upstroke_78_110": np.max(d1[:, 77:110], axis=1),
            "cyl_mean_upstroke_78_110": np.mean(d1[:, 77:110], axis=1),
            "cyl_curvature_energy": np.mean(d2**2, axis=1),
            "cyl_relative_volume_proxy": rel_volume,
            "cyl_relative_surface_proxy": rel_surface,
            "cyl_surface_to_volume_proxy": rel_surface / rel_volume,
            "cyl_posterior_shifted_positive_mass": posterior_shift_of_positive_mass * pos_mass,
        }
    )


def summarize_cyl_features(features_by_cohort: dict[str, pd.DataFrame], datasets: dict[str, dict]) -> pd.DataFrame:
    """각 원기둥 특징의 코호트별 통계, 방향 일치 여부, 두 코호트 중 더 약한 쪽의 AUC 거리(min_abs_auc_distance)를 계산해 정렬."""
    rows = []
    for name in features_by_cohort["g1090"].columns:
        vals = {cohort: df[name].to_numpy(dtype=float) for cohort, df in features_by_cohort.items()}
        stat_g = feature_stats(vals["g1090"], datasets["g1090"]["y"])
        stat_s = feature_stats(vals["sdata"], datasets["sdata"]["y"])
        direction_consistent = np.sign(stat_g["delta_low_minus_nonlow"]) == np.sign(stat_s["delta_low_minus_nonlow"])
        min_auc_distance = min(abs(stat_g["auc_if_higher_predicts_low_smi"] - 0.5), abs(stat_s["auc_if_higher_predicts_low_smi"] - 0.5))
        for row in summarize_feature(name, vals, datasets):
            row["direction_consistent"] = bool(direction_consistent)
            row["min_abs_auc_distance_g1090_sdata"] = float(min_auc_distance)
            rows.append(row)
    out = pd.DataFrame(rows)
    score = (
        out[out["cohort"].eq("pooled")]
        .assign(rank_key=lambda d: d["feature"].map(
            out[out["cohort"].eq("g1090")].set_index("feature")["min_abs_auc_distance_g1090_sdata"].to_dict()
        ))
        .sort_values(["direction_consistent", "rank_key", "mannwhitney_p"], ascending=[False, False, True])
    )
    order = score["feature"].tolist()
    out["feature_order"] = out["feature"].map({f: i for i, f in enumerate(order)})
    return out.sort_values(["feature_order", "cohort"]).drop(columns=["feature_order"])


def plot_cylindrical_signature(datasets: dict[str, dict], features_by_cohort: dict[str, pd.DataFrame]) -> None:
    """대표 특징(원기둥 반등 질량)의 코호트별 분포 히스토그램과, 평균 로그-반경 프로파일 위에 중간결핍/후반초과 영역을 표시한 그래프를 PNG로 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    feature = "cyl_late_positive_plus_mid_negative"
    for ax, cohort in zip(axes, ["g1090", "sdata"]):
        y = datasets[cohort]["y"]
        val = features_by_cohort[cohort][feature].to_numpy(dtype=float)
        ax.hist(val[~y], bins=36, density=True, color="#2F6F73", alpha=0.55, label="Non-low SMI")
        ax.hist(val[y], bins=24, density=True, color="#C84630", alpha=0.58, label="Low SMI")
        ax.set_title(cohort, loc="left", fontweight="bold")
        ax.set_xlabel("Cylindrical rebound mass: mid deficit + late excess")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cylindrical_rebound_mass_distribution.png", dpi=200)
    plt.close(fig)

    # Show the log-radius interpretation on the common mean profile.
    pooled_x = np.vstack([d["x"] for d in datasets.values()])
    pooled_y = np.concatenate([d["y"] for d in datasets.values()])
    profile = smooth_log_profile(pooled_x)
    low = profile[pooled_y].mean(axis=0)
    non = profile[~pooled_y].mean(axis=0)
    z = np.arange(1, 129)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(z, non, color="#2F6F73", lw=2.2, label="Non-low SMI mean log profile")
    ax.plot(z, low, color="#C84630", lw=2.4, label="Low SMI mean log profile")
    ax.axhline(0.0, color="#555555", ls="--", lw=1.0, label="Patient mean radius baseline")
    ax.fill_between(z[41:78], low[41:78], 0.0, where=low[41:78] < 0, color="#2F6F73", alpha=0.22, label="Mid negative area")
    ax.fill_between(z[99:128], 0.0, low[99:128], where=low[99:128] > 0, color="#C84630", alpha=0.22, label="Late positive area")
    ax.set_xlabel("AEC_128 point index")
    ax.set_ylabel("log(AEC / patient mean AEC)")
    ax.set_title("AEC_128 as a cylindrical axial attenuation-radius profile", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cylindrical_log_radius_profile.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: AEC 축 프로파일을 "원기둥의 축 방향 감쇠 반경 편차"라는
    물리적 은유로 재해석한 형태 특징들도, 앞서 찾은 중간결핍/후반초과 패턴을 재현하는가?):

    1. g1090/sdata를 load_aec128로 로드하고, extract_cylindrical_features로 29개 원기둥 형태
       특징(질량/무게중심/퍼짐/모멘트/부피·표면적 근사 등)을 계산해 환자별 CSV로 저장.
    2. summarize_cyl_features로 각 특징의 코호트별 통계, 방향 일치 여부, AUC 강도로 정렬한 표를 CSV로 저장.
    3. plot_cylindrical_signature로 대표 특징(cyl_late_positive_plus_mid_negative)의 분포와,
       평균 로그-반경 프로파일 위에 중간결핍/후반초과 영역을 표시한 그래프를 저장.
    4. 원기둥 해석 방식과 주요/보조 특징의 수식·의미를 JSON으로 저장하고, 선택된 8개 특징의
       통계를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    features_by_cohort = {cohort: extract_cylindrical_features(d["x"]) for cohort, d in datasets.items()}

    for cohort, df in features_by_cohort.items():
        out = df.copy()
        out.insert(0, "low_smi", datasets[cohort]["y"].astype(int))
        out.to_csv(OUT_DIR / f"{cohort}_aec128_cylindrical_features_patient_level.csv", index=False)

    stats_df = summarize_cyl_features(features_by_cohort, datasets)
    stats_df.to_csv(OUT_DIR / "aec128_cylindrical_feature_stats.csv", index=False)
    plot_cylindrical_signature(datasets, features_by_cohort)

    definitions = {
        "interpretation": "AEC_128 is treated as an axial profile on a cylinder, not a full z-theta cylinder surface. log(AEC/patient mean AEC) is used as a relative effective attenuation-radius deviation.",
        "profile": "P_i(j) = smooth(log(AEC_i(j) / mean(AEC_i(1:128))))",
        "main_feature": {
            "name": "cyl_late_positive_plus_mid_negative",
            "formula": "mean(max(-P_i(j),0), j=42:78) + mean(max(P_i(j),0), j=100:128)",
            "meaning": "Total cylindrical rebound mass: mid-cylinder inward deficit plus late-cylinder outward excess.",
        },
        "secondary_feature": {
            "name": "cyl_positive_negative_centroid_separation",
            "formula": "centroid_z(max(P_i,0)) - centroid_z(max(-P_i,0))",
            "meaning": "How far the positive attenuation mass is shifted later than the negative attenuation mass.",
        },
    }
    with open(OUT_DIR / "aec128_cylindrical_feature_definitions.json", "w", encoding="utf-8") as f:
        json.dump(definitions, f, ensure_ascii=False, indent=2)

    selected_names = [
        "cyl_late_positive_plus_mid_negative",
        "cyl_late_mean_minus_mid_mean_log",
        "cyl_positive_negative_centroid_separation",
        "cyl_posterior_shifted_positive_mass",
        "cyl_mid_negative_area_42_78",
        "cyl_late_positive_area_100_128",
        "cyl_max_upstroke_78_110",
        "cyl_surface_to_volume_proxy",
    ]
    selected = stats_df[stats_df["feature"].isin(selected_names)]
    print(selected.to_string(index=False))
    print(OUT_DIR / "aec128_cylindrical_feature_stats.csv")
    print(OUT_DIR / "cylindrical_log_radius_profile.png")
    print(OUT_DIR / "cylindrical_rebound_mass_distribution.png")


if __name__ == "__main__":
    main()
