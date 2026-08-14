from __future__ import annotations

# AEC-128 곡선의 FPCA(등간격 128포인트이므로 표준 PCA가 이산 근사) 최적 컴포넌트 수를 결정하고,
# 각 컴포넌트가 곡선의 어느 구간을 어떻게 변형시키는지 시각화한다.
# n_components 후보(1~20, 특히 2~5)별 internal OOF R^2와 분산설명비율(%)을 함께 비교해 best n을
# 선택하고(내부 CV로만 선택 -> external은 미사용), mean curve ± 1·SD·PC loading으로 PC1~4를 그린다.
# (step3의 라벨 이상판정 cutoff이 mean±1SD라, PPT에서 다른 SD배수로 오해되지 않도록 동일하게 맞춤)

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step1_fpca"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COMPONENT_CANDIDATES = list(range(1, 21))
FPCA_VARIANCE_CANDIDATES = list(range(1, 11))  # 분산설명비율 표에 표시할 n_components 후보(2~5 포함)
N_CURVE_COMPONENTS_SHOWN = 4  # PC 시각화에 그릴 컴포넌트 수 상한

# step2_clinic_aec_ratio.py와 동일한 지방계열 output feature 9종(절대값 3 + 비율 6) -> 파일명 slug.
# 근육계열(TAMA/NAMA/LAMA/IMATA)은 라벨변수와 상관이 높아 순환논리가 되므로 제외([[feedback_no_circular_label_feature]])
FEATURES = {
    "SAT(피하지방)_SUM": "sat",
    "VAT(내장지방)_SUM": "vat",
    "Total Fat_SUM": "total_fat",
    "VAT_SAT_ratio": "vat_sat",
    "VAT_TotalFat_ratio": "vat_total_fat",
    "VAT_TAMA_ratio": "vat_tama",
    "SAT_TotalFat_ratio": "sat_total_fat",
    "SAT_TAMA_ratio": "sat_tama",
    "TotalFat_TAMA_ratio": "total_fat_tama",
}


# VAT/SAT, VAT/Total Fat, VAT/TAMA, SAT/Total Fat, SAT/TAMA, Total Fat/TAMA 6개 비율 컬럼을 원본 SUM 컬럼으로부터
# 산출해 추가 (step2_clinic_aec_ratio.py/step3_clinic_aec_logistic.py와 동일)
def add_ratio_features(meta: pd.DataFrame) -> pd.DataFrame:
    vat = pd.to_numeric(meta["VAT(내장지방)_SUM"], errors="coerce")
    sat = pd.to_numeric(meta["SAT(피하지방)_SUM"], errors="coerce")
    tama = pd.to_numeric(meta["TAMA_SUM"], errors="coerce")
    total_fat = pd.to_numeric(meta["Total Fat_SUM"], errors="coerce")

    meta = meta.copy()
    meta["VAT_SAT_ratio"] = vat / sat
    meta["VAT_TotalFat_ratio"] = vat / total_fat
    meta["VAT_TAMA_ratio"] = vat / tama
    meta["SAT_TotalFat_ratio"] = sat / total_fat
    meta["SAT_TAMA_ratio"] = sat / tama
    meta["TotalFat_TAMA_ratio"] = total_fat / tama
    return meta


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합, 체성분 비율 컬럼을 추가
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return add_ratio_features(merged)


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + (있으면) FPCA score 결합.
# 성별을 고정한 남/여 개별 실행에서는 sex가 상수가 되어 계수가 무의미해지므로 include_sex=False로 제외
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     include_sex: bool = True) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    x = clinic if aec_extra is None else np.column_stack([clinic, aec_extra])
    return x, scaler


# FPCA_COMPONENT_CANDIDATES 각각에 대해 clinic4+FPCA 모델의 internal OOF R^2를 target_features 전체 평균으로
# 산출하고, 그 평균이 가장 높은 n_components를 best로 선택한다. external은 전혀 사용하지 않음(내부 CV로만
# 모델 선택 -> 동결 모델을 external로 1회만 최종 검증하는 원칙을 FPCA 컴포넌트 수 선택에도 동일 적용).
# PCA(고유함수 추정)는 반드시 각 fold의 학습 데이터에서만 fit하고 검증 fold는 그 고유함수에 transform만 해야
# 함 - 전체 데이터로 먼저 PCA를 fit하면 검증 fold의 곡선 정보가 고유함수 추정에 이미 들어가 OOF R^2가
# 낙관적으로 부풀려지는 data leakage가 됨. 그래서 PCA.fit을 fold 루프 안(train_idx에만)으로 옮긴다
def select_best_fpca_n(meta_int: pd.DataFrame, aec_int_raw: np.ndarray, include_sex: bool,
                        target_features: list[str], cv: KFold) -> tuple[int, pd.DataFrame]:
    records = []
    for feat in target_features:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        mask_int = np.isfinite(y_int_all)
        y_int = y_int_all[mask_int]
        aec_feat = aec_int_raw[mask_int]
        meta_feat = meta_int.loc[mask_int].reset_index(drop=True)

        for n in FPCA_COMPONENT_CANDIDATES:
            oof = np.empty(len(y_int))
            for train_idx, test_idx in cv.split(aec_feat):
                pca = PCA(n_components=n, random_state=SEED).fit(aec_feat[train_idx])
                fpca_train = pca.transform(aec_feat[train_idx])
                fpca_test = pca.transform(aec_feat[test_idx])

                x_train, scaler = clinical_matrix(meta_feat.iloc[train_idx], fpca_train, include_sex=include_sex)
                x_test, _ = clinical_matrix(meta_feat.iloc[test_idx], fpca_test, scaler=scaler,
                                             include_sex=include_sex)

                model = LinearRegression().fit(x_train, y_int[train_idx])
                oof[test_idx] = model.predict(x_test)
            records.append({"n_components": n, "feature": feat, "r2_internal": r2_score(y_int, oof)})

    search_df = pd.DataFrame(records)
    mean_by_n = search_df.groupby("n_components")["r2_internal"].mean()
    best_n = int(mean_by_n.idxmax())
    print(f"[FPCA] n_components 후보별 internal OOF 평균 R^2:\n{mean_by_n.round(4)}")
    print(f"[FPCA] 선택된 best n_components = {best_n} (internal OOF 평균 R^2={mean_by_n[best_n]:.4f})")
    return best_n, search_df


# FPCA_VARIANCE_CANDIDATES(1~10, 2~5 포함) 각각의 개별/누적 분산설명비율(%)과 해당 n의 internal OOF 평균 R^2를
# 정리한 표를 산출. 컴포넌트 순서는 n_components를 몇으로 잡든 동일하므로 최대 후보 수로 한 번만 fit하면 충분
def fpca_variance_table(pca_full: PCA, search_df: pd.DataFrame) -> pd.DataFrame:
    ratios = pca_full.explained_variance_ratio_
    mean_r2_by_n = search_df.groupby("n_components")["r2_internal"].mean()

    rows = []
    for n in FPCA_VARIANCE_CANDIDATES:
        if n > len(ratios):
            continue
        rows.append({
            "n_components": n,
            "explained_variance_pct": float(ratios[n - 1] * 100),
            "cumulative_variance_pct": float(ratios[:n].sum() * 100),
            "mean_r2_internal": float(mean_r2_by_n.get(n, np.nan)),
        })
    return pd.DataFrame(rows)


# n_components 후보별 internal OOF R^2(평균 및 feature별 범위)를 선으로 그리고 선택된 best n을 표시.
# 오른쪽 보조축에 누적 분산설명비율(%)을 겹쳐 그려 "몇 개를 써야 분산을 얼마나 설명하는지"를 R²와 함께 확인
def plot_fpca_component_search(search_df: pd.DataFrame, best_n: int, variance_table: pd.DataFrame,
                                out_path: Path) -> None:
    grouped = search_df.groupby("n_components")["r2_internal"]
    mean_r2, min_r2, max_r2 = grouped.mean(), grouped.min(), grouped.max()

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.fill_between(mean_r2.index, min_r2.values, max_r2.values, alpha=0.2, color="#2a78d6",
                     label="feature별 범위(min-max)")
    ax.plot(mean_r2.index, mean_r2.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="전체 feature 평균 R²")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"선택된 best n={best_n}")

    ax.set_xticks(list(mean_r2.index))
    ax.set_xlabel("FPCA n_components", fontsize=30)
    ax.set_ylabel("Internal OOF R²", fontsize=30)
    ax.set_title("FPCA 컴포넌트 수별 internal R² (clinic4+FPCA)", fontsize=30, fontweight="bold",
                 color="#161616", pad=20)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)

    var_sorted = variance_table.sort_values("n_components")
    ax2 = ax.twinx()
    ax2.plot(var_sorted["n_components"], var_sorted["cumulative_variance_pct"], marker="s", markersize=14,
              linewidth=4, linestyle=":", color="#9b59b6", label="누적 분산설명비율(%)")
    ax2.set_ylabel("누적 분산설명비율(%)", fontsize=30, color="#9b59b6")
    ax2.tick_params(axis="y", labelsize=30, labelcolor="#9b59b6")
    ax2.set_ylim(0, 100)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=2, fontsize=30, frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA component search plot to {out_path}")


# PC 하나(mean ± 2·sqrt(eigenvalue)·loading)만 그려서 자체 파일로 저장 — PC별로 개별 그림을 남겨
# 어느 PC를 봐도 그 PC가 곡선의 어느 구간을 어떻게 변형시키는지 독립적으로 확인 가능하게 한다
def plot_single_fpca_component_curve(pca: PCA, mean_curve: np.ndarray, x_axis: np.ndarray, i: int,
                                      out_path: Path) -> None:
    component = pca.components_[i]
    scale = 1 * np.sqrt(pca.explained_variance_[i])

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.plot(x_axis, mean_curve + scale * component, color="#2a78d6", linewidth=3, linestyle="--",
            label="평균 + 1·SD·PC 방향")
    ax.plot(x_axis, mean_curve - scale * component, color="#e2622e", linewidth=3, linestyle="--",
            label="평균 - 1·SD·PC 방향")
    ax.set_title(f"PC{i + 1} (분산설명 {pca.explained_variance_ratio_[i] * 100:.1f}%)", fontsize=36,
                 fontweight="bold", color="#161616", pad=20)
    ax.set_xlabel("AEC 슬라이스 위치", fontsize=30)
    ax.set_ylabel("AEC 값", fontsize=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(fontsize=30, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA PC{i + 1} curve plot to {out_path}")


# PC1~n_show 각각을 개별 파일(fpca_pc{i}_curve.png)로 저장하고, 한눈에 비교할 수 있는 결합 그림도 함께 남긴다
def plot_fpca_component_curves(pca: PCA, aec_int_raw: np.ndarray, n_show: int, output_dir: Path) -> None:
    mean_curve = aec_int_raw.mean(axis=0)
    x_axis = np.arange(1, N_SLICES + 1)

    for i in range(n_show):
        plot_single_fpca_component_curve(pca, mean_curve, x_axis, i, output_dir / f"fpca_pc{i + 1}_curve.png")

    fig, axes = plt.subplots(1, n_show, figsize=(13 * n_show, 11), squeeze=False)
    out_path = output_dir / "fpca_component_curves_overview.png"
    for i, ax in enumerate(axes[0]):
        component = pca.components_[i]
        scale = 1 * np.sqrt(pca.explained_variance_[i])
        ax.plot(x_axis, mean_curve + scale * component, color="#2a78d6", linewidth=3, linestyle="--",
                label="평균 + 1·SD·PC 방향")
        ax.plot(x_axis, mean_curve - scale * component, color="#e2622e", linewidth=3, linestyle="--",
                label="평균 - 1·SD·PC 방향")
        ax.set_title(f"PC{i + 1} (분산설명 {pca.explained_variance_ratio_[i] * 100:.1f}%)", fontsize=36,
                     fontweight="bold", color="#161616", pad=20)
        ax.set_xlabel("AEC 슬라이스 위치", fontsize=20)
        if i == 0:
            ax.set_ylabel("AEC 값", fontsize=20)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=26)

    y_min = min(ax.get_ylim()[0] for ax in axes[0])
    y_max = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(y_min, y_max)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=30,
               frameon=False)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA component curve plot to {out_path}")


# PC1만, PC2만, PC1+PC2를 함께 썼을 때 재구성 곡선이 실제 원본과 얼마나 비슷한지 비교 -> PC1+PC2 기준
# 재구성 오차(환자별 SSE)가 중앙값에 가장 가까운 환자 1명을 골라(극단적 이상치가 아닌 대표적인 재구성 품질을
# 보여줌) 원본·PC1 재구성·PC2 재구성·PC1+PC2 재구성 네 곡선을 겹쳐 그린다 -> 각 성분의 개별 기여를 시각적으로 확인
def plot_pc1_pc2_reconstruction(pca: PCA, aec_int_raw: np.ndarray, mean_curve: np.ndarray, x_axis: np.ndarray,
                                 out_path: Path) -> None:
    scores = pca.transform(aec_int_raw)
    recon_pc1 = mean_curve[None, :] + np.outer(scores[:, 0], pca.components_[0])
    recon_pc2 = mean_curve[None, :] + np.outer(scores[:, 1], pca.components_[1])
    recon_pc12 = recon_pc1 + np.outer(scores[:, 1], pca.components_[1])

    per_patient_sse = ((aec_int_raw - recon_pc12) ** 2).sum(axis=1)
    idx = int(np.argmin(np.abs(per_patient_sse - np.median(per_patient_sse))))
    raw_curve = aec_int_raw[idx]
    recon_pc1_curve, recon_pc2_curve, recon_pc12_curve = recon_pc1[idx], recon_pc2[idx], recon_pc12[idx]

    ss_tot = float(((aec_int_raw - mean_curve[None, :]) ** 2).sum())
    r2_pc1 = 1 - float(((aec_int_raw - recon_pc1) ** 2).sum()) / ss_tot
    r2_pc2 = 1 - float(((aec_int_raw - recon_pc2) ** 2).sum()) / ss_tot
    r2_pc12 = 1 - float(((aec_int_raw - recon_pc12) ** 2).sum()) / ss_tot
    var_pct_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pct_pc2 = pca.explained_variance_ratio_[1] * 100
    var_pct_pc12 = pca.explained_variance_ratio_[:2].sum() * 100

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.plot(x_axis, raw_curve, color="#161616", linewidth=4, label="원본 곡선")
    ax.plot(x_axis, recon_pc1_curve, color="#e2622e", linewidth=3, linestyle="--",
            label=f"PC1만 재구성 (분산설명 {var_pct_pc1:.1f}%, 전체 환자 R²={r2_pc1:.3f})")
    ax.plot(x_axis, recon_pc2_curve, color="#2e8b57", linewidth=3, linestyle="--",
            label=f"PC2만 재구성 (분산설명 {var_pct_pc2:.1f}%, 전체 환자 R²={r2_pc2:.3f})")
    ax.plot(x_axis, recon_pc12_curve, color="#2a78d6", linewidth=3, linestyle="--",
            label=f"PC1+PC2 재구성 (분산설명 {var_pct_pc12:.1f}%, 전체 환자 R²={r2_pc12:.3f})")
    ax.set_title("PC1만·PC2만·PC1+PC2 재구성 비교", fontsize=32, fontweight="bold", color="#161616", pad=20)
    ax.set_xlabel("AEC 슬라이스 위치", fontsize=30)
    ax.set_ylabel("AEC 값", fontsize=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(fontsize=24, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PC1/PC2/PC1+PC2 reconstruction plot to {out_path}")


# n_components 탐색 -> 분산설명비율 표 -> best n 곡선 시각화까지 한 코호트/성별 scope에 대해 전부 수행
def run(meta_int: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()

    best_n, search_df = select_best_fpca_n(meta_int, aec_int_raw, include_sex, list(FEATURES.keys()), cv)

    pca_full = PCA(n_components=max(FPCA_COMPONENT_CANDIDATES), random_state=SEED).fit(aec_int_raw)
    variance_table = fpca_variance_table(pca_full, search_df)
    variance_table.to_csv(output_dir / "fpca_variance_table.csv", index=False)
    print(f"[FPCA] n_components별 분산설명비율(%) 및 internal R² 요약:\n{variance_table.round(3)}")

    search_df.to_csv(output_dir / "fpca_component_search.csv", index=False)
    plot_fpca_component_search(search_df, best_n, variance_table, output_dir / "fpca_component_search.png")

    pca_best = PCA(n_components=best_n, random_state=SEED).fit(aec_int_raw)
    print(f"[FPCA] explained variance ratio (PC1-{best_n}): {pca_best.explained_variance_ratio_.round(4)}")
    plot_fpca_component_curves(pca_best, aec_int_raw, min(best_n, N_CURVE_COMPONENTS_SHOWN), output_dir)

    mean_curve = aec_int_raw.mean(axis=0)
    x_axis = np.arange(1, N_SLICES + 1)
    plot_pc1_pc2_reconstruction(pca_best, aec_int_raw, mean_curve, x_axis,
                                 output_dir / "fpca_pc1_pc2_reconstruction.png")


# internal(Gangnam) 코호트를 로드/전처리 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행.
# n_components 선택은 내부 CV로만 하므로 external 코호트는 이 스크립트에서 사용하지 않는다
def main() -> None:
    meta_int = load_cohort(INTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]
    vals = meta_int[clinical_cols].apply(pd.to_numeric, errors="coerce")
    mask_clinical = vals.notna().all(axis=1).to_numpy()
    print(f"Clinical input 결측 제외: internal {(~mask_clinical).sum()}/{len(mask_clinical)}")
    meta_int = meta_int[mask_clinical].reset_index(drop=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()

    run(meta_int, OUTPUT_DIR / "total", include_sex=True)
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True), sub_dir, include_sex=False)


if __name__ == "__main__":
    main()
