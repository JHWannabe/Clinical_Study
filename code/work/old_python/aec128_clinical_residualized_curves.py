from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aec128_common_shape_feature import FILES, load_aec128


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_clinical_residualized_curves"
COHORT_LABELS = {
    "g1090": "Gangnam g1090",
    "sdata": "Sinchon sdata",
}


def clinical_covariates(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """나이/성별/키/몸무게를 결측 중앙값 대체 후 표준화하고 절편을 붙인 설계행렬을 만듦."""
    age = pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float)
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    height = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float)
    x = np.column_stack([age, sex_m, height, weight])
    names = ["age", "sex_male", "height", "weight"]
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    xz = (x - mu) / sd
    return np.column_stack([np.ones(xz.shape[0]), xz]), ["intercept"] + names


def residualize_aec_by_clinical(x_norm: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """각 위치별로 AEC 값을 임상변수에 최소제곱 회귀시켜, 잔차(residual)와 전체평균을 더한 보정값(adjusted)을 계산 — 임상변수의 선형 효과를 곡선에서 제거."""
    design, names = clinical_covariates(meta)
    beta, *_ = np.linalg.lstsq(design, x_norm, rcond=None)
    fitted = design @ beta
    residual = x_norm - fitted
    adjusted = residual + x_norm.mean(axis=0, keepdims=True)
    return adjusted, residual, beta, names


def mean_ci(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """위치별 평균과 95% 신뢰구간(정규근사)을 계산."""
    mean = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return mean, mean - 1.96 * se, mean + 1.96 * se


def build_rows(cohort: str, y: np.ndarray, adjusted: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
    """보정값/잔차 두 버전 각각에 대해 저근감소증/정상군 평균곡선과, 두 군의 차이 곡선까지 위치별로 표로 정리."""
    rows = []
    for data_name, mat in [("adjusted_normalized_aec", adjusted), ("residualized_aec", residual)]:
        for group_name, mask in [("low_smi", y), ("normal", ~y)]:
            mean, lo, hi = mean_ci(mat[mask])
            for i in range(mat.shape[1]):
                rows.append(
                    {
                        "cohort": cohort,
                        "data": data_name,
                        "group": group_name,
                        "point_1_to_128": i + 1,
                        "position_0_to_1": i / 127,
                        "n": int(mask.sum()),
                        "mean": mean[i],
                        "ci95_low": lo[i],
                        "ci95_high": hi[i],
                    }
                )
    diff_adj = adjusted[y].mean(axis=0) - adjusted[~y].mean(axis=0)
    diff_res = residual[y].mean(axis=0) - residual[~y].mean(axis=0)
    for data_name, diff in [("adjusted_low_minus_normal", diff_adj), ("residual_low_minus_normal", diff_res)]:
        for i in range(len(diff)):
            rows.append(
                {
                    "cohort": cohort,
                    "data": data_name,
                    "group": "low_minus_normal",
                    "point_1_to_128": i + 1,
                    "position_0_to_1": i / 127,
                    "n": int(len(y)),
                    "mean": diff[i],
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_curves(results: dict[str, dict]) -> None:
    """코호트별로 (보정값 평균곡선 2그룹) + (잔차 차이곡선) 2x2 패널 그래프를 그려 PNG로 저장."""
    colors = {"low_smi": "#C84630", "normal": "#2F6F73"}
    labels = {"low_smi": "Low SMI", "normal": "Normal"}
    x = np.arange(1, 129)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.4), sharex=True)
    for col, cohort in enumerate(["g1090", "sdata"]):
        res = results[cohort]
        y = res["y"]
        adjusted = res["adjusted"]
        residual = res["residual"]

        ax = axes[0, col]
        for group_name, mask in [("low_smi", y), ("normal", ~y)]:
            mean, lo, hi = mean_ci(adjusted[mask])
            ax.plot(x, mean, lw=2.2, color=colors[group_name], label=f"{labels[group_name]} (n={int(mask.sum())})")
            ax.fill_between(x, lo, hi, color=colors[group_name], alpha=0.14, linewidth=0)
        ax.axhline(1.0, color="#555555", lw=0.9, ls="--", alpha=0.65)
        ax.set_title(f"{COHORT_LABELS[cohort]}: adjusted normalized AEC", loc="left", fontweight="bold")
        ax.set_ylabel("AEC / patient mean, adjusted")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)

        ax = axes[1, col]
        low_mean = residual[y].mean(axis=0)
        normal_mean = residual[~y].mean(axis=0)
        diff = low_mean - normal_mean
        ax.plot(x, diff, lw=2.3, color="#333333", label="Low SMI - normal")
        ax.fill_between(x, 0, diff, where=diff >= 0, color="#C84630", alpha=0.20)
        ax.fill_between(x, 0, diff, where=diff < 0, color="#2F6F73", alpha=0.20)
        ax.axhline(0.0, color="#555555", lw=0.9, ls="--")
        ax.axvspan(42, 78, color="#2F6F73", alpha=0.10)
        ax.axvspan(100, 128, color="#C84630", alpha=0.10)
        ax.set_title(f"{COHORT_LABELS[cohort]}: residual difference", loc="left", fontweight="bold")
        ax.set_xlabel("AEC_128 point index")
        ax.set_ylabel("Residual mean difference")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("AEC_128 after removing age, sex, height, and weight", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "aec128_age_sex_height_weight_residualized_curves.png", dpi=200)
    plt.close(fig)


def plot_pooled_comparison(summary: pd.DataFrame) -> None:
    """두 코호트의 잔차 차이곡선(저근감소증-정상)을 한 그래프에 겹쳐 그려, 임상변수 제거 후에도 패턴이 일관되는지 비교하는 PNG를 저장."""
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for cohort, color in [("g1090", "#4C78A8"), ("sdata", "#F58518")]:
        sub = summary[
            (summary["cohort"] == cohort)
            & (summary["data"] == "residual_low_minus_normal")
            & (summary["group"] == "low_minus_normal")
        ]
        ax.plot(sub["point_1_to_128"], sub["mean"], lw=2.2, color=color, label=COHORT_LABELS[cohort])
    ax.axhline(0.0, color="#555555", lw=0.9, ls="--")
    ax.axvspan(42, 78, color="#2F6F73", alpha=0.10, label="Common mid-low window")
    ax.axvspan(100, 128, color="#C84630", alpha=0.10, label="Common late-high window")
    ax.set_xlabel("AEC_128 point index")
    ax.set_ylabel("Adjusted residual difference: low SMI - normal")
    ax.set_title("Covariate-removed AEC_128 low-SMI signature", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_residual_low_minus_normal_gangnam_sinchon.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 나이/성별/키/몸무게의 선형 효과를 곡선에서 완전히
    제거해도(residualize), 앞서 발견한 저근감소증군의 AEC 모양 차이가 여전히 남아있는가?):

    1. g1090/sdata 각각 load_aec128로 로드하고, residualize_aec_by_clinical로 각 위치의 AEC 값을
       임상변수(나이/성별/키/몸무게)에 회귀시켜 잔차(residual)와 보정값(adjusted = 잔차+전체평균)을 계산.
    2. build_rows로 코호트별 보정값·잔차 각각의 두 그룹(저근감소증/정상) 평균곡선과 차이곡선을 표로 만들어 CSV로 저장.
    3. 위치별 임상변수 계수(어느 위치가 나이/성별/키/몸무게에 얼마나 민감한지)도 CSV로 저장.
    4. plot_curves로 코호트별 2x2(보정값 두 그룹 + 잔차 차이) 그래프를 저장.
    5. plot_pooled_comparison으로 두 코호트의 잔차 차이곡선을 겹쳐 그려, 임상변수를 걷어낸 뒤에도
       공통 중간저점/후반고점 패턴이 유지되는지 시각적으로 확인하는 그래프를 저장.
    6. 코호트별 표본수를 CSV로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    summary_frames = []
    coef_rows = []
    for cohort, path in FILES.items():
        data = load_aec128(path)
        adjusted, residual, beta, names = residualize_aec_by_clinical(data["x"], data["meta"])
        results[cohort] = {"y": data["y"], "adjusted": adjusted, "residual": residual, "meta": data["meta"]}
        summary_frames.append(build_rows(cohort, data["y"], adjusted, residual))
        for cov_i, cov_name in enumerate(names):
            for j in range(beta.shape[1]):
                coef_rows.append(
                    {
                        "cohort": cohort,
                        "covariate": cov_name,
                        "point_1_to_128": j + 1,
                        "coefficient": beta[cov_i, j],
                    }
                )

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "aec128_age_sex_height_weight_residualized_curve_summary.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT_DIR / "aec128_clinical_covariate_coefficients_by_point.csv", index=False)
    plot_curves(results)
    plot_pooled_comparison(summary)

    counts = []
    for cohort, res in results.items():
        y = res["y"]
        counts.append({"cohort": cohort, "n": int(len(y)), "low_smi": int(y.sum()), "normal": int((~y).sum())})
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(OUT_DIR / "cohort_counts.csv", index=False)
    print(counts_df.to_string(index=False))
    print(OUT_DIR / "aec128_age_sex_height_weight_residualized_curves.png")
    print(OUT_DIR / "aec128_residual_low_minus_normal_gangnam_sinchon.png")
    print(OUT_DIR / "aec128_age_sex_height_weight_residualized_curve_summary.csv")


if __name__ == "__main__":
    main()
