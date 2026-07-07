from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

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


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_waviness_feature_test"
RANKED_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_midrange_feature_refit" / "midrange_feature_search_train_ranked.csv"
FEATURE_SHORTS = [
    "visual_trough_depth__early_041_056__mid_053_076__tail_101_128",
    "norm_slope_085_096_mean",
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
]
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def moving_average_edge(x: np.ndarray, width: int) -> np.ndarray:
    """가장자리를 반복(edge padding)으로 채운 뒤 지정한 폭의 이동평균을 각 행(환자)마다 계산."""
    pad = width // 2
    kernel = np.ones(width, dtype=float) / width
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, xp)


def waviness_features(x: np.ndarray) -> pd.DataFrame:
    """정규화 AEC 곡선을 치골-간(pubis-liver) 직선(centerline)에서 뺀 잔차(waviness)를 바탕으로, 구간별 RMS/총변동/곡률, 이동평균 대비 잔차, 골-회복 진폭, 저주파/고주파 FFT 에너지 등 다양한 "굴곡" 특징을 계산."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    linear = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    resid = x - linear
    smooth = moving_average_edge(x, 31)
    smooth_resid = x - smooth
    d1 = np.diff(resid, axis=1)
    d2 = np.diff(x, axis=1, n=2)

    def seg(a: int, b: int) -> slice:
        return slice(a - 1, b)

    rows = {
        "wave_linear_rms_041_118": np.sqrt(np.mean(resid[:, seg(41, 118)] ** 2, axis=1)),
        "wave_linear_rms_053_118": np.sqrt(np.mean(resid[:, seg(53, 118)] ** 2, axis=1)),
        "wave_linear_abs_tv_041_118": np.mean(np.abs(d1[:, seg(41, 117)]), axis=1),
        "wave_linear_curv_abs_041_118": np.mean(np.abs(d2[:, seg(41, 116)]), axis=1),
        "wave_smooth_rms_041_118": np.sqrt(np.mean(smooth_resid[:, seg(41, 118)] ** 2, axis=1)),
        "wave_trough_rebound_amp_053_118": np.max(resid[:, seg(85, 118)], axis=1)
        - np.min(resid[:, seg(53, 76)], axis=1),
        "wave_mid_trough_depth_linear_053_076": -np.mean(resid[:, seg(53, 76)], axis=1),
        "wave_upper_recovery_linear_085_118": np.mean(resid[:, seg(85, 118)], axis=1),
        "wave_regional_contrast_sum": -np.mean(resid[:, seg(53, 76)], axis=1)
        + np.mean(resid[:, seg(85, 118)], axis=1),
    }

    centered = x - x.mean(axis=1, keepdims=True)
    coeff = np.fft.rfft(centered, axis=1)
    power = np.abs(coeff) ** 2
    rows["wave_fft_midfreq_energy_03_12"] = np.sqrt(np.sum(power[:, 3:13], axis=1))
    rows["wave_fft_highfreq_energy_13_32"] = np.sqrt(np.sum(power[:, 13:33], axis=1))
    return pd.DataFrame(rows)


def choose_gate_settings() -> pd.DataFrame:
    """aec_midrange_feature_refit의 순위표에서 미리 지정한 4개 "중후반 구간" 특징(FEATURE_SHORTS)의 최적 폭·람다 설정을 가져온다."""
    ranked = pd.read_csv(RANKED_PATH)
    ranked["feature_short"] = (
        ranked["feature"].astype(str).str.replace("bank_norm__", "", regex=False).str.replace("midrange__", "", regex=False)
    )
    rows = []
    for short in FEATURE_SHORTS:
        rows.append(ranked[ranked["feature_short"].eq(short)].iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def gate_memberships(g: dict, s: dict) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """두 코호트 x 5개 운영점 전체에 대해, 4개 특징의 2-of-4 합의 게이트로 임상양성군을 유지(kept)/하향조정(deesc)으로 나누고, 임상양성 마스크(cpos)와 함께 반환."""
    settings = choose_gate_settings()
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    cg, cs, _ = clinical_scores(g, s)
    direction = risk_direction(g["y"].astype(int), cg, xg)
    name_to_idx = {name: i for i, name in enumerate(names)}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], cg, target) for op, target in OPS}
    datasets = {
        "Gangnam": {"y": g["y"].astype(int), "clinical": cg, "features": xg},
        "Sinchon": {"y": s["y"].astype(int), "clinical": cs, "features": xs},
    }
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cohort, d in datasets.items():
        for op, _ in OPS:
            th = thresholds[op]
            cpos = d["clinical"] >= th
            votes = np.zeros(len(d["y"]), dtype=int)
            for _, r in settings.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((d["clinical"] - th) / float(r["width"])) ** 2)
                gate = d["clinical"] + float(r["lambda"]) * boundary * d["features"][:, idx] * direction[idx]
                votes += (cpos & (gate < th)).astype(int)
            deesc = cpos & (votes >= 2)
            kept = cpos & ~deesc
            out[(cohort, op)] = (cpos, kept, deesc)
    return out


def auc_nonlow(values: np.ndarray, y: np.ndarray) -> float:
    """특징값이 비저근감소증(non-low)을 얼마나 잘 구분하는지 AUC를 계산하고, 방향이 반대여도(AUC<0.5) 상관없도록 1-AUC와 비교해 더 큰 값을 반환."""
    nonlow = 1 - y.astype(int)
    try:
        auc = roc_auc_score(nonlow, values)
    except ValueError:
        return np.nan
    return float(max(auc, 1.0 - auc))


def comparison_rows(
    cohort: str,
    op: str,
    y: np.ndarray,
    features: pd.DataFrame,
    cpos: np.ndarray,
    kept: np.ndarray,
    deesc: np.ndarray,
) -> list[dict]:
    """한 코호트·운영점에 대해 각 굴곡 특징이 (하향조정 vs 유지) 및 (임상양성 내 비저근감소증 vs 저근감소증)을 얼마나 다르게 가르는지 평균차·Mann-Whitney p값·AUC로 비교한 행 목록을 만든다."""
    rows = []
    for name in features.columns:
        v = features[name].to_numpy(dtype=float)
        kd = v[kept]
        dd = v[deesc]
        nonlow = v[cpos & (y == 0)]
        low = v[cpos & (y == 1)]
        mw_gate = stats.mannwhitneyu(dd, kd, alternative="two-sided").pvalue if len(dd) and len(kd) else np.nan
        mw_low = stats.mannwhitneyu(nonlow, low, alternative="two-sided").pvalue if len(nonlow) and len(low) else np.nan
        rows.append(
            {
                "cohort": cohort,
                "operating_point": op,
                "feature": name,
                "aec_negative_deesc_mean": float(np.nanmean(dd)),
                "aec_positive_kept_mean": float(np.nanmean(kd)),
                "deesc_minus_kept": float(np.nanmean(dd) - np.nanmean(kd)),
                "deesc_vs_kept_mannwhitney_p": float(mw_gate),
                "clinical_positive_nonlow_mean": float(np.nanmean(nonlow)),
                "clinical_positive_low_mean": float(np.nanmean(low)),
                "nonlow_minus_low": float(np.nanmean(nonlow) - np.nanmean(low)),
                "nonlow_vs_low_mannwhitney_p": float(mw_low),
                "auc_for_nonlow_within_clinical_positive_best_direction": auc_nonlow(v[cpos], y[cpos]),
                "clinical_positive_n": int(cpos.sum()),
                "deesc_n": int(deesc.sum()),
                "kept_n": int(kept.sum()),
            }
        )
    return rows


def add_spans(ax: plt.Axes) -> None:
    """그래프에 4개 관심 구간을 색칠한 배경 영역으로 표시."""
    for a, b, color in [
        (41, 56, "#D6EFD8"),
        (53, 76, "#F8D6D2"),
        (85, 96, "#D6E4F5"),
        (103, 118, "#E8DCF5"),
    ]:
        ax.axvspan(a, b, color=color, alpha=0.24, lw=0)


def plot_residuals(datasets: dict[str, dict], memberships: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """S85 운영점에서 두 코호트의 (곡선 - 치골간 직선) 잔차 평균곡선을 유지군 대 하향조정군으로 겹쳐 그려, "굴곡"이 실제로 눈에 보이는 곡선 모양 차이인지 시각적으로 확인하는 그림을 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    z = np.arange(1, 129)
    for ax, (cohort, d) in zip(axes, datasets.items()):
        x = d["norm"]
        linear = x[:, [0]] * (1.0 - np.linspace(0, 1, 128)) + x[:, [-1]] * np.linspace(0, 1, 128)
        resid = x - linear
        _, kept, deesc = memberships[(cohort, "S85")]
        m_keep = resid[kept].mean(axis=0)
        m_deesc = resid[deesc].mean(axis=0)
        ax.axhline(0, color="#777777", lw=1.0, ls="--")
        add_spans(ax)
        ax.plot(z, m_keep, color="#B23A48", lw=2.2, label="AEC(+): kept")
        ax.plot(z, m_deesc, color="#2F6F9F", lw=2.2, label="AEC(-): de-escalated")
        ax.set_title(f"{cohort} S85: residual from pubis-liver centerline", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(1, 128)
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Craniocaudal index: 1 pubis -> 128 liver dome")
    axes[0].set_ylabel("Normalized AEC minus straight centerline")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("AEC Waviness as Centerline Residual", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "s85_centerline_residual_waviness_aec_gate.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 개별 곡률/기울기 특징을 넘어서, 곡선 전체의 "굴곡짐
    (waviness)" 자체 - 직선에서 얼마나 벗어나는지, 진폭이 얼마나 큰지, 저/고주파 성분이 얼마나
    되는지 - 를 특징화하면, 임상양성군 내에서 하향조정군과 유지군을 구분하거나 저근감소증
    여부를 구분하는 데 도움이 되는가?):

    1. g1090(Gangnam)/sdata(Sinchon)를 로드하고, waviness_features로 각 코호트의 정규화 곡선에서
       9개의 굴곡 특징(직선잔차 RMS/총변동/곡률, 평활잔차 RMS, 골-회복 진폭, FFT 중간/고주파
       에너지 등)을 계산.
    2. gate_memberships로 aec_block_or_consensus_search류 4특징 2-of-4 게이트를 이용해 두 코호트
       x 5개 운영점 전체에서 임상양성군을 유지/하향조정으로 분류.
    3. 각 코호트·운영점·굴곡특징 조합에 대해 comparison_rows로 (하향조정 vs 유지) 평균차·
       Mann-Whitney p값과 (임상양성 내 비저근감소증 vs 저근감소증) 평균차·p값·AUC를 계산해 CSV로 저장.
    4. S85 운영점만 골라 코호트 간 최악(max) p값과 평균 AUC 기준으로 특징 순위를 매긴 요약표를 저장.
    5. plot_residuals로 두 코호트의 S85 잔차 평균곡선(유지 vs 하향조정)을 그려 저장하고, 요약표
       상위 24행과 저장 경로를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    datasets = {
        "Gangnam": {"y": g["y"].astype(int), "norm": g["norm"]},
        "Sinchon": {"y": s["y"].astype(int), "norm": s["norm"]},
    }
    feature_tables = {cohort: waviness_features(d["norm"]) for cohort, d in datasets.items()}
    memberships = gate_memberships(g, s)

    rows = []
    for cohort, d in datasets.items():
        for op, _ in OPS:
            cpos, kept, deesc = memberships[(cohort, op)]
            rows.extend(comparison_rows(cohort, op, d["y"], feature_tables[cohort], cpos, kept, deesc))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "waviness_feature_clinical_positive_comparisons.csv", index=False)

    s85 = detail[detail["operating_point"].eq("S85")].copy()
    s85["min_gate_p_across_cohorts"] = s85.groupby("feature")["deesc_vs_kept_mannwhitney_p"].transform("max")
    s85["min_low_p_across_cohorts"] = s85.groupby("feature")["nonlow_vs_low_mannwhitney_p"].transform("max")
    s85["mean_auc_nonlow"] = s85.groupby("feature")["auc_for_nonlow_within_clinical_positive_best_direction"].transform("mean")
    summary = (
        s85.sort_values(["mean_auc_nonlow", "min_gate_p_across_cohorts"], ascending=[False, True])
        [
            [
                "feature",
                "cohort",
                "deesc_minus_kept",
                "deesc_vs_kept_mannwhitney_p",
                "nonlow_minus_low",
                "nonlow_vs_low_mannwhitney_p",
                "auc_for_nonlow_within_clinical_positive_best_direction",
                "mean_auc_nonlow",
            ]
        ]
        .reset_index(drop=True)
    )
    summary.to_csv(OUT_DIR / "waviness_feature_S85_ranked_summary.csv", index=False)
    plot_residuals(datasets, memberships)

    print(summary.head(24).to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
