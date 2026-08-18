from __future__ import annotations

# AEC-128 곡선의 FPCA(등간격 128포인트이므로 표준 PCA가 이산 근사) 최적 컴포넌트 수를 결정하고,
# 각 컴포넌트가 곡선의 어느 구간을 어떻게 변형시키는지 시각화한다.
# n_components는 internal 코호트 AEC 곡선의 scree curve에서 elbow(Kneedle 방식: 축을 0~1로 정규화한 뒤
# 첫점-끝점을 잇는 직선까지 수직거리가 최대인 지점)로 결정한다(사용자 확인: "R제곱값 말고 누적분산비율로
# 확인해" 이후 "elbow로 교체해서 재확인" - 다운스트림 예측성능이 아니라 FPCA/PCA 표준 scree test 관행대로
# 분산 감소 패턴만으로 n을 정함, 육안 판단 대신 기하학적 규칙으로 재현 가능하게 함).
# mean curve ± 1·SD·PC loading으로 PC1~4를 그린다(step3의 라벨 이상판정 cutoff이 mean±1SD라, PPT에서
# 다른 SD배수로 오해되지 않도록 동일하게 맞춤).

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0821" / "step1_fpca"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COMPONENT_CANDIDATES_MAX = 20  # elbow 탐색에 쓸 n_components 상한
N_CURVE_COMPONENTS_SHOWN = 4  # PC 시각화에 그릴 컴포넌트 수 상한


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# scree curve(컴포넌트별 개별 explained variance ratio)를 구하고, 축을 0~1로 정규화한 뒤 첫점-끝점을
# 잇는 직선(chord)까지의 수직거리를 계산한다. 거리가 최대인 지점이 elbow(Satopaa et al. 2011 Kneedle
# 알고리즘). 축 정규화 없이 원래 스케일(n_components 1~20 vs 분산비율 0~1)로 거리를 재면 축 단위 차이
# 때문에 결과가 왜곡되므로 정규화가 필수다
def _scree_and_elbow_distance(cum_var: pd.Series) -> tuple[pd.Series, pd.Series]:
    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    return scree, pd.Series(dist, index=scree.index)


# internal 코호트 AEC-128 곡선의 scree curve에서 elbow(첫점-끝점 직선까지 수직거리가 최대인 지점)를
# n_components로 선택한다. 다운스트림 예측성능(AUC/R^2)으로 n을 고르면 그 성능 자체가 선택 기준에 쓰인
# 데이터로 최적화되어 낙관적으로 부풀려질 위험이 있어, FPCA/PCA 표준 scree test 관행대로 분산 감소
# 패턴만으로 n을 정한다
def select_best_fpca_n(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))
    _, dist = _scree_and_elbow_distance(cum_var)
    best_n = int(dist.idxmax())

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] 선택된 elbow n_components = {best_n} (누적분산비율={cum_var[best_n]:.4f}, "
          f"chord-거리={dist[best_n]:.4f})")
    return best_n, cum_var


# n_components별 누적/개별 explained variance ratio, elbow 판단에 쓴 chord-거리, 선택된 best_n을 엑셀로
# 저장(그래프의 원자료)
def save_cum_var_excel(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values,
                        "individual_variance_ratio": scree.values, "elbow_chord_distance": dist.values})
    df["selected_best_n"] = df["n_components"] == best_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# 좌: n_components별 누적 explained variance ratio, 우: scree curve+chord+elbow 판단 근거를 나란히 표시
def plot_fpca_component_search(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, _ = _scree_and_elbow_distance(cum_var)
    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    ax = axes[0]
    ax.plot(cum_var.index, cum_var.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="누적 explained variance ratio")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"선택된 elbow n={best_n}")
    ax.set_xticks(list(cum_var.index))
    ax.set_xlabel("FPCA n_components", fontsize=42)
    ax.set_ylabel("누적 explained variance ratio", fontsize=42)
    ax.set_title("누적분산비율", fontsize=40, fontweight="bold", color="#161616", pad=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(loc="lower right", fontsize=26, frameon=False)

    ax = axes[1]
    ax.plot(scree.index, scree.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="개별 explained variance ratio (scree)")
    ax.plot([scree.index[0], scree.index[-1]], [scree.values[0], scree.values[-1]],
            color="#898781", linestyle=":", linewidth=3, label="첫점-끝점 직선(chord)")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"선택된 elbow n={best_n}")
    ax.set_xticks(list(scree.index))
    ax.set_xlabel("FPCA n_components", fontsize=42)
    ax.set_ylabel("개별 explained variance ratio", fontsize=42)
    ax.set_title("Scree curve (elbow 판단 근거)", fontsize=40, fontweight="bold", color="#161616", pad=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(loc="upper right", fontsize=26, frameon=False)

    fig.suptitle("FPCA 컴포넌트 수 선택 (internal 코호트, elbow 방식)", fontsize=44, fontweight="bold", y=1.03)
    fig.tight_layout()
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


# n_components 탐색(elbow) -> best n 곡선 시각화까지 한 코호트/성별 scope에 대해 전부 수행
def run(meta_int: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()

    best_n, cum_var = select_best_fpca_n(aec_int_raw)
    save_cum_var_excel(cum_var, best_n, output_dir / "fpca_cumulative_variance.xlsx")
    plot_fpca_component_search(cum_var, best_n, output_dir / "fpca_component_search.png")

    pca_best = PCA(n_components=best_n, random_state=SEED).fit(aec_int_raw)
    print(f"[FPCA] explained variance ratio (PC1-{best_n}): {pca_best.explained_variance_ratio_.round(4)}")
    plot_fpca_component_curves(pca_best, aec_int_raw, min(best_n, N_CURVE_COMPONENTS_SHOWN), output_dir)

    if best_n < 2:
        print(f"[스킵] best_n={best_n} < 2라 PC1+PC2 재구성 비교를 생략합니다.")
        return
    mean_curve = aec_int_raw.mean(axis=0)
    x_axis = np.arange(1, N_SLICES + 1)
    plot_pc1_pc2_reconstruction(pca_best, aec_int_raw, mean_curve, x_axis,
                                 output_dir / "fpca_pc1_pc2_reconstruction.png")


# internal(Gangnam) 코호트를 로드/전처리 후 전체/남성만/여성만 3가지로 나눠 run()을 각각 실행.
# n_components 선택은 AEC 곡선 자체의 scree curve elbow로만 하므로 external 코호트는 이 스크립트에서 사용하지 않는다
def main() -> None:
    meta_int = load_cohort(INTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]
    vals = meta_int[clinical_cols].apply(pd.to_numeric, errors="coerce")
    mask_clinical = vals.notna().all(axis=1).to_numpy()
    print(f"Clinical input 결측 제외: internal {(~mask_clinical).sum()}/{len(mask_clinical)}")
    meta_int = meta_int[mask_clinical].reset_index(drop=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()

    run(meta_int, OUTPUT_DIR / "total")
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True), sub_dir)


if __name__ == "__main__":
    main()
