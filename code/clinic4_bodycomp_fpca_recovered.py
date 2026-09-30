from __future__ import annotations

# clinic4_aec_logistic_regression.py의 recover_curve_coefficients/save_recovered_curve_plot(AEC 곡선 전용
# FPCA 계수 -> 128-slice logit 기여도 역변환)과 완전히 같은 수식을 clinic4_aec_bodycomp_logistic.py가 학습한
# aec_bodycomp 모델(outputs/clinic4/coefficients.xlsx의 "aec_bodycomp" 시트)의 VAT/SAT/LAMA/NAMA/IMATA
# FPCA 계수에 대해 반복한다. PCA는 gangnam raw curve로 이 스크립트에서 새로 fit(원 모델 학습때와 동일하게
# frozen PCA를 재현하기 위해 seed/n_components를 BODYCOMP_CURVES와 맞춤)하고, coefficients.xlsx에 저장된
# 로지스틱 계수(표준화 스케일)를 그대로 대입해 역변환한다. 원리는 clinic4_aec_logistic_regression.py의
# recover_curve_coefficients 주석 참조: PCA는 선형변환이라 score_k=(curve-mean).component_k, 로지스틱
# 표준화 계수 beta_k를 대입하면 logit 기여분 = curve.[sum_k (beta_k/score_std_k)*component_k]+const로 재전개.

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clinic4_logistic_regression import DATA_XLSX, DISEASES, PROJECT_ROOT, SEED, load_data
from clinic4_aec_bodycomp_logistic import (
    AEC_FPCA_N, AEC_PREFIX, AEC_SHEET, BODYCOMP_CURVES, CLINICAL_BASE_COLS, EXTRA_COLS,
    add_curve_features, clinical_matrix, curve_cols, load_data_with_curves,
)

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic4" / "aec_bodycomp"
COEF_XLSX = PROJECT_ROOT / "outputs" / "clinic4" / "coefficients.xlsx"
N_SLICES = 128


# aec_bodycomp 모델(clinic4_aec_bodycomp_logistic.py)이 실제로 학습에 썼던 것과 동일한 gangnam PCA/scaler를
# 재현한다(frozen PCA가 저장돼 있지 않으므로 같은 seed/n_components로 다시 fit - PCA는 데이터+seed가
# 같으면 항상 같은 components_가 나오는 결정론적 알고리즘이라 재현에 문제없음). scaler도 EXTRA_COLS 순서로
# 다시 fit해야 coefficients.xlsx의 표준화 계수를 정확히 역변환할 수 있다.
def refit_pcas_and_scaler() -> tuple[dict[str, PCA], StandardScaler, pd.DataFrame]:
    df = load_data_with_curves(DATA_XLSX)
    df, pcas = add_curve_features(df)
    _, scaler = clinical_matrix(df)
    return pcas, scaler, df


# VAT/SAT/LAMA/NAMA/IMATA 5개 곡선 각각에 대해, 질병별 FPCA 계수(표준화 스케일)를 128-slice 곡선 축으로
# 역변환. 수식은 clinic4_aec_logistic_regression.recover_curve_coefficients와 동일(PCA 선형성 이용):
# score_k=(curve-pca.mean_).component_k, logit=...+beta_k*(score_k-score_mean_k)/score_std_k이므로
# logit 기여분 = curve.[sum_k (beta_k/score_std_k)*component_k]+const. raw curve slice별 표준편차를 곱해
# "그 slice가 +1SD 증가할 때 logit 변화량"으로 정규화.
def recover_bodycomp_curves(coef_df: pd.DataFrame, pcas: dict[str, PCA], scaler: StandardScaler,
                             df: pd.DataFrame) -> pd.DataFrame:
    cols = CLINICAL_BASE_COLS + EXTRA_COLS  # aec_bodycomp 모델의 scaler fit 컬럼 순서(clinical_matrix와 동일)
    rows = []
    for name, (sheet, prefix, n_pc) in BODYCOMP_CURVES.items():
        curve = df[curve_cols(prefix)].to_numpy(dtype=float)
        curve_std = curve.std(axis=0, ddof=1)
        pca = pcas[name]
        fpca_cols = [f"{name}_fpca_pc{i}" for i in range(1, n_pc + 1)]
        for disease in DISEASES:
            d = coef_df[coef_df["disease"] == disease].set_index("term")
            w_curve = np.zeros(N_SLICES)
            for i, col in enumerate(fpca_cols):
                beta = d.loc[col, "coefficient"]
                score_std = scaler.scale_[cols.index(col)]
                w_curve += (beta / score_std) * pca.components_[i]
            w_curve_per_1sd = w_curve * curve_std
            for slice_i in range(1, N_SLICES + 1):
                rows.append({"curve": name, "disease": disease, "slice": slice_i,
                             "logit_per_1sd_raw_curve": w_curve_per_1sd[slice_i - 1]})
    return pd.DataFrame(rows)


# 5개 체성분 곡선(행) x 3개 질병(열) 그리드로 recovered curve(logit 기여도 vs slice)를 그려 저장.
# 기존 aec/fpca_recovered_curve.png는 곡선 하나(AEC)라 질병을 한 그래프에 겹쳐 그렸지만, 여기는 곡선이
# 5개라 질병을 겹쳐 그리면 5개 subplot 각각에서 스케일이 달라 비교가 어려우므로 grid로 분리해 질병별
# 색상은 통일하고 곡선(행)마다 y축 스케일을 자유롭게 둔다(체성분별 logit 기여도 크기가 서로 다름).
def save_recovered_curve_grid(recovered: pd.DataFrame, out_path: Path) -> None:
    curve_names = list(BODYCOMP_CURVES.keys())
    disease_colors = {"HTN": "tab:blue", "DM": "tab:orange", "CKD": "tab:green"}
    fig, axes = plt.subplots(len(curve_names), len(DISEASES), figsize=(12.2, 4.16 / 3 * len(curve_names) * 1.3),
                              sharex=True)
    for row, name in enumerate(curve_names):
        for col, disease in enumerate(DISEASES):
            ax = axes[row, col]
            d = recovered[(recovered["curve"] == name) & (recovered["disease"] == disease)]
            ax.plot(d["slice"], d["logit_per_1sd_raw_curve"], color=disease_colors[disease], linewidth=1.3)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(disease)
            if col == 0:
                ax.set_ylabel(f"{name.upper()}\nlogit / +1SD")
            if row == len(curve_names) - 1:
                ax.set_xlabel("slice index (liver -> pubis)")
    fig.suptitle("FPCA Coefficient Inverse Transform — Slice-wise Logit Contribution (체성분)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_df = pd.read_excel(COEF_XLSX, sheet_name="aec_bodycomp")

    pcas, scaler, df = refit_pcas_and_scaler()
    recovered = recover_bodycomp_curves(coef_df, pcas, scaler, df)

    recovered.to_csv(OUTPUT_DIR / "fpca_recovered_curve_bodycomp_coefficients.csv", index=False)
    save_recovered_curve_grid(recovered, OUTPUT_DIR / "fpca_recovered_curve_bodycomp.png")
    print(f"Saved fpca_recovered_curve_bodycomp_coefficients.csv, fpca_recovered_curve_bodycomp.png to {OUTPUT_DIR}")

    # 요약: 곡선x질병별 |logit_per_1sd| 최대값(어디가 가장 크게 기여하는지 빠르게 확인)
    summary = (recovered.assign(abs_logit=recovered["logit_per_1sd_raw_curve"].abs())
               .groupby(["curve", "disease"])["abs_logit"].max().reset_index()
               .sort_values("abs_logit", ascending=False))
    print("\n=== curve x disease별 최대 |logit per +1SD| (내림차순) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
