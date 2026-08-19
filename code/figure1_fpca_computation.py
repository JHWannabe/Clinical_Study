from __future__ import annotations

# m&m 초안(docx/md)의 Figure 1(내부 코호트 AEC-128 곡선에 대한 FPCA 계산 과정)을 실제로 그리는 스크립트.
# 사용자 확인 2026-08-19: "Figure 1을 생성하는 py파일도 만들어줘". 대표 환자는 "PC1-3 성분점수가 모두 크고
# 재구성 적합도가 상위 25% 이내"라는 캡션/본문 조건을 만족하는 환자를 코드로 직접 탐색해 선택하며(하드코딩
# PatientID 없음), FPCA_N_FIXED=3은 step2_clinic_aec_disease_select.py와 동일한 고정값을 그대로 따른다.
# 영문 논문에 들어갈 그림이라 그림 안 텍스트(제목/축/범례)는 전부 영문으로 표기한다(사용자 확인 2026-08-19:
# "figure에 들어가는 텍스트는 모두 영문으로 표기해"). 패널 성격이 다르면 하나의 합성 Figure로 묶지 않고
# 각각 별도 파일로 저장한다(사용자 확인 2026-08-19: "굳이 하나의 figure로 나타내지 않아도 돼. 성격이 다르면
# 따로 figure를 생성하도록해") - 다만 공분산행렬 원본/근사 두 패널은 서로 비교하는 목적이라 한 파일에 나란히
# 둔다. 통합본(figure1_combined.png)에서는 원래 있던 "대표 환자의 편차곡선 단독" 패널을 제거했다(사용자
# 확인 2026-08-19: "D가 있기 때문에 B도 필요 없을거 같다는 생각인데" - Projection 패널이 d_i(z) 곡선을
# 이미 그대로 포함하고 있어 별도 패널로 다시 보여주는 게 정보 중복이라는 지적, 동의하여 제거). 이후 두 차례
# 재배치를 거쳤다: 먼저 알파벳을 등장 순서대로 재부여했고(사용자 확인 2026-08-19: "ABCD 이름 순서도
# 변경해"), 그다음 "곡선 계산(표본→편차/투영→scree)" 행과 "공분산 구조(원본/근사→eigenfunction)" 행으로
# 나뉘도록 공분산행렬 쌍과 편차/scree 쌍의 행 위치를 맞바꿨다(사용자 확인 2026-08-19: "b,c랑 d,e 배치를
# 변경하는게 어때?"). 개별 패널 파일(figure1a~f)에도 통합본과 동일한 알파벳을 사용해 두 출력물 간 라벨이
# 어긋나지 않게 했다. 캡션 문구는 docs/m&m 초안.md의 Figure 1 캡션에도 동일하게 반영해야 한다.
#
# 사용자 확인 2026-08-19(2차): 완성된 초안에서 "SVD에 대한 개념이 안 들어가 있다"는 지적을 받아, D(원본
# 공분산행렬)와 E(rank-3 근사) 사이에 SVD 분해(Φ·Λ·Φᵀ) 자체를 시각화하는 패널을 한 차례 추가했었다. 그러나
# 곧이어 "figure는 굳이 생성 안해도 됐을거 같아"라는 재검토를 받아 그 그림 패널은 다시 제거했다 - 최종
# 결론은 본문(Method, Functional Principal Component Analysis 단락)에 C ≈ ΦΛΦᵀ 수식만 명시하고, 그림은
# 원래의 6패널(A-F) 구성을 그대로 유지하는 것. 그림 쪽 코드는 이번 절 전과 동일하다.
#
# 패널 구성(통합본 기준, 개별 파일도 동일 알파벳) 및 저장 파일(사용자 확인 2026-08-19: 논문 내 Figure 2에
# 해당하므로 outputs/figure의 다른 파일들과 마찬가지로 fig2 접두사로 저장):
#   A(fig2a_sample_curves_mean.png): 무작위 표본 곡선(회색) + 평균곡선 μ(z)(검정)
#   B(fig2b_projection_score.png): 대표 환자의 편차곡선 d_i(z) = x_i(z) - μ(z)를 φ1(z)에 투영해
#       score_i,1을 얻는 과정(편차곡선 자체도 이 패널에서 함께 보여줌)
#   C(fig2c_scree_elbow.png): eigenvalue scree plot과 elbow(k=3) 판단 근거
#   D/E(fig2de_covariance_matrix.png): 128x128 공분산행렬 원본(D)과 rank-3 근사(E) 비교
#   F(fig2f_eigenfunctions.png): eigenfunction φ1-φ3를 μ(z) ± √λ_k·φ_k(z)로 표시

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949가 μ/φ/₁ 등을 인코딩 못 해 print에서 죽는 것 방지

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_N_FIXED = 3  # step2_clinic_aec_disease_select.py와 동일한 고정값(elbow=3과 일치)
FPCA_COMPONENT_CANDIDATES_MAX = 20  # elbow 탐색용 n_components 상한(step1_aec_fpca.py와 동일)
N_SAMPLE_CURVES = 40  # 패널 A에 겹쳐 그릴 무작위 표본 곡선 수
RECON_R2_TOP_QUANTILE = 0.75  # "재구성 적합도가 상위 25% 이내" 기준
# 본문(para 34)/Figure 1 캡션(패널 B)에 이미 보고된 대표 환자의 component score(PC1 +1,318, PC2 +1,175,
# PC3 -247) - 대표 환자를 다시 특정하기 위해 PatientID를 하드코딩하는 대신, 이 공개된 score 좌표에 가장
# 가까운 환자를 코드로 탐색해 재현한다(select_representative_patient 참고)
TARGET_SCORE = np.array([1318.0, 1175.0, -247.0])


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# scree curve(개별 explained variance ratio)를 구하고, 축을 0~1로 정규화한 뒤 첫점-끝점을 잇는 직선(chord)
# 까지의 수직거리를 계산한다. 거리가 최대인 지점이 elbow(Satopaa et al. 2011 Kneedle 알고리즘) -
# step2_clinic_aec_disease_select.py의 _scree_and_elbow_distance와 동일 로직
def scree_and_elbow_distance(cum_var: pd.Series) -> tuple[pd.Series, pd.Series]:
    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    return scree, pd.Series(dist, index=scree.index)


# n_components=1..max로 PCA를 적합해 누적 explained variance ratio를 구하고 elbow(패널 C 판단 근거)를 계산
def compute_scree(aec_raw: np.ndarray) -> tuple[pd.Series, pd.Series, int]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_raw.shape[0], aec_raw.shape[1])
    pca_full = PCA(n_components=max_components, random_state=SEED).fit(aec_raw)
    cum_var = pd.Series(np.cumsum(pca_full.explained_variance_ratio_), index=range(1, max_components + 1))
    scree, dist = scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    return cum_var, scree, elbow_n


# TARGET_SCORE(본문에 이미 보고된 대표 환자의 PC1-3 score)에 가장 가까운 환자를 찾고, 그 환자가 캡션의
# 정성적 조건("PC1-3 성분점수가 모두 크고 재구성 적합도가 상위 25% 이내")을 실제로 만족하는지 검증한다
def select_representative_patient(scores: np.ndarray, r2_per_patient: np.ndarray) -> int:
    rep_idx = int(np.argmin(np.linalg.norm(scores - TARGET_SCORE, axis=1)))

    score_pct = np.array([(np.abs(scores[:, k]) < np.abs(scores[rep_idx, k])).mean() for k in range(scores.shape[1])])
    r2_pct = (r2_per_patient < r2_per_patient[rep_idx]).mean()
    assert np.all(score_pct >= 0.75), f"대표 환자의 PC score가 '모두 큼' 조건(상위 25%)을 만족하지 않음: {score_pct}"
    assert r2_pct >= RECON_R2_TOP_QUANTILE, f"대표 환자의 재구성 R²가 상위 25% 조건을 만족하지 않음: {r2_pct:.3f}"
    return rep_idx


# (A) 무작위 표본 곡선(회색) + 평균곡선(검정) - 원곡선들의 형태와 그 평균을 보여주는 패널이라 다른 패널과
# 성격이 달라 별도 파일로 저장(사용자 확인 2026-08-19: "성격이 다르면 따로 figure를 생성하도록해")
def plot_panel_a_sample_and_mean(aec_raw: np.ndarray, mean_curve: np.ndarray, out_path: Path) -> None:
    x_axis = np.arange(1, N_SLICES + 1)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(aec_raw.shape[0], size=min(N_SAMPLE_CURVES, aec_raw.shape[0]), replace=False)

    fig, ax = plt.subplots(figsize=(16, 11))
    for i in sample_idx:
        ax.plot(x_axis, aec_raw[i], color="#c9c7c1", linewidth=1, alpha=0.6)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=3, label="mean curve μ(z)")
    ax.set_title("Figure 1A. Sample curves and mean curve", fontsize=32, fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("AEC value", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=22, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1A to {out_path}")


# (B) 대표 환자의 편차곡선 d_i(z) = x_i(z) - μ(z)를 φ1에 투영해 component score를 얻는 과정
# (d_i(z)와 φ1(z)를 twin-axis로 겹쳐 표시; 편차곡선 자체도 이 패널에서 함께 보여준다)
def plot_panel_b_projection(d_rep: np.ndarray, phi1: np.ndarray, score1: float, out_path: Path) -> None:
    x_axis = np.arange(1, N_SLICES + 1)
    fig, ax = plt.subplots(figsize=(16, 11))
    ax2 = ax.twinx()
    ax.plot(x_axis, d_rep, color="#2a78d6", linewidth=3, label="d_i(z)")
    ax2.plot(x_axis, phi1, color="#e2622e", linewidth=3, linestyle="--", label="φ1(z)")
    ax.set_title("Figure 1B. Deviation curve d_i(z), projected onto φ1 → component score", fontsize=28,
                 fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("d_i(z) = x_i(z) - μ(z)", fontsize=26, color="#2a78d6")
    ax2.set_ylabel("φ1(z)", fontsize=26, color="#e2622e")
    ax.tick_params(labelsize=22)
    ax2.tick_params(labelsize=22)
    ax.text(0.03, 0.05, f"score_i,1 = Σ d_i(z)·φ1(z)\n= {score1:,.1f}", transform=ax.transAxes, fontsize=22,
             va="bottom", ha="left", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#161616"})
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=20, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1B to {out_path}")


# (C) eigenvalue scree plot과 elbow(k=3) 판단 근거 - 진단성 차트라 곡선/행렬 패널들과 성격이 달라 별도 저장
def plot_panel_c_scree_elbow(scree: pd.Series, elbow_n: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.plot(scree.index, scree.values, marker="o", markersize=10, linewidth=3, color="#161616",
            label="individual explained variance ratio")
    ax.plot([scree.index[0], scree.index[-1]], [scree.values[0], scree.values[-1]], color="#898781",
            linestyle=":", linewidth=2.5, label="first-to-last point line (chord)")
    ax.axvline(elbow_n, color="#e2622e", linestyle="--", linewidth=3, label=f"elbow k={elbow_n}")
    ax.set_xticks(list(scree.index))
    ax.set_title("Figure 1C. Scree plot and elbow-based component selection", fontsize=28, fontweight="bold",
                 pad=20)
    ax.set_xlabel("component index", fontsize=26)
    ax.set_ylabel("individual explained variance ratio", fontsize=26)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=20, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1C to {out_path}")


# (D, E) 128x128 공분산행렬 - 원본(D)과 상위 3개 eigenfunction만으로 재구성한 rank-3 근사(E)를 같은
# 컬러스케일로 나란히 배치해 SVD 저차원 근사의 품질을 직접 비교(둘 다 "공분산행렬 히트맵"으로 성격이 같아 하나로 묶음)
def plot_panel_de_covariance_matrices(cov_matrix: np.ndarray, cov_rank3: np.ndarray, out_path: Path) -> None:
    vmin, vmax = float(cov_matrix.min()), float(cov_matrix.max())
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))

    im = axes[0].imshow(cov_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Figure 1D. Covariance matrix (original)", fontsize=28, fontweight="bold", pad=18)
    axes[0].set_xlabel("z", fontsize=24)
    axes[0].set_ylabel("z", fontsize=24)
    axes[0].tick_params(labelsize=18)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(cov_rank3, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Figure 1E. Covariance matrix (rank-{FPCA_N_FIXED} approximation)", fontsize=28,
                       fontweight="bold", pad=18)
    axes[1].set_xlabel("z", fontsize=24)
    axes[1].set_ylabel("z", fontsize=24)
    axes[1].tick_params(labelsize=18)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1D/E to {out_path}")


# (F) eigenfunction phi1-phi3를 mean(z) +- sqrt(eigenvalue_k)*phi_k(z)로 표시
def plot_panel_f_eigenfunctions(mean_curve: np.ndarray, pca3: PCA, out_path: Path) -> None:
    x_axis = np.arange(1, N_SLICES + 1)
    colors = ["#2a78d6", "#e2622e", "#1baf7a"]

    fig, ax = plt.subplots(figsize=(20, 11))
    for k in range(FPCA_N_FIXED):
        scale = np.sqrt(pca3.explained_variance_[k])
        upper, lower = mean_curve + scale * pca3.components_[k], mean_curve - scale * pca3.components_[k]
        ax.fill_between(x_axis, lower, upper, color=colors[k], alpha=0.18)
        ax.plot(x_axis, upper, color=colors[k], linewidth=2.5,
                 label=f"φ{k + 1} (variance explained {pca3.explained_variance_ratio_[k] * 100:.1f}%)")
        ax.plot(x_axis, lower, color=colors[k], linewidth=2.5)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=3, linestyle=":", label="mean curve μ(z)")
    ax.set_title("Figure 1F. Eigenfunctions φ1-φ3 (μ(z) ± √λ_k·φ_k(z))", fontsize=30, fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("AEC value", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=20, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1F to {out_path}")


# A-F를 2행 3열 그리드 하나에 합친 통합본도 별도로 저장(사용자 확인 2026-08-19: "figure1으로 통합한것도
# 저장해") - 개별 패널 파일(figure1a~f)과 별개로 outputs에 함께 남긴다. 패널별 개별 함수와 문구/스타일을
# 동일하게 맞추되, 여러 패널이 한 화면에 들어가는 만큼 폰트 크기만 축소한다. row0은 "곡선 계산(표본/평균
# → 편차/투영 → scree/elbow)", row1은 "공분산 구조(원본/근사→eigenfunction)"로 나뉘도록 배치했다(사용자
# 확인 2026-08-19: "b,c랑 d,e 배치를 변경하는게 어때?"). 공분산행렬 두 패널(D, E)은 같은 행에 나란히
# 배치해 직접 비교되도록 하고, imshow에 aspect="auto"를 줘서 컬러바로 인한 폭 축소가 세로 여백으로 번져
# 다른 패널보다 낮아 보이는 문제를 방지한다(사용자 확인 2026-08-19: "c랑f가... 높이를 모두 동일한 크기로
# 맞춰줘").
def plot_figure1_combined(aec_raw: np.ndarray, mean_curve: np.ndarray, cov_matrix: np.ndarray, cov_rank3: np.ndarray,
                           pca3: PCA, d_rep: np.ndarray, score1: float, scree: pd.Series, elbow_n: int,
                           out_path: Path) -> None:
    x_axis = np.arange(1, N_SLICES + 1)
    vmin, vmax = float(cov_matrix.min()), float(cov_matrix.max())
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(aec_raw.shape[0], size=min(N_SAMPLE_CURVES, aec_raw.shape[0]), replace=False)

    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for i in sample_idx:
        ax.plot(x_axis, aec_raw[i], color="#c9c7c1", linewidth=1, alpha=0.6)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=3, label="mean curve μ(z)")
    ax.set_title("(A) Sample curves and mean curve", fontsize=20, fontweight="bold")
    ax.set_xlabel("AEC slice position z", fontsize=15)
    ax.set_ylabel("AEC value", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc="best")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    ax.plot(x_axis, d_rep, color="#2a78d6", linewidth=2.5, label="d_i(z)")
    ax2.plot(x_axis, pca3.components_[0], color="#e2622e", linewidth=2.5, linestyle="--", label="φ1(z)")
    ax.set_title("(B) Deviation curve, projected onto φ1 → score", fontsize=18, fontweight="bold")
    ax.set_xlabel("AEC slice position z", fontsize=15)
    ax.set_ylabel("d_i(z) = x_i(z) - μ(z)", fontsize=15, color="#2a78d6")
    ax2.set_ylabel("φ1(z)", fontsize=15, color="#e2622e")
    ax.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax.text(0.03, 0.05, f"score_i,1 = Σ d_i(z)·φ1(z)\n= {score1:,.1f}", transform=ax.transAxes, fontsize=14,
             va="bottom", ha="left", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#161616"})
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(scree.index, scree.values, marker="o", markersize=7, linewidth=2.5, color="#161616",
            label="individual explained variance ratio")
    ax.plot([scree.index[0], scree.index[-1]], [scree.values[0], scree.values[-1]], color="#898781",
            linestyle=":", linewidth=2, label="first-to-last point line (chord)")
    ax.axvline(elbow_n, color="#e2622e", linestyle="--", linewidth=2.5, label=f"elbow k={elbow_n}")
    ax.set_xticks(list(scree.index))
    ax.set_title("(C) Scree plot / elbow", fontsize=20, fontweight="bold")
    ax.set_xlabel("component index", fontsize=15)
    ax.set_ylabel("individual explained variance ratio", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cov_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title("(D) Covariance matrix (original)", fontsize=20, fontweight="bold")
    ax.set_xlabel("z", fontsize=15)
    ax.set_ylabel("z", fontsize=15)
    ax.tick_params(labelsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(cov_rank3, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(f"(E) Covariance matrix (rank-{FPCA_N_FIXED} approximation)", fontsize=20, fontweight="bold")
    ax.set_xlabel("z", fontsize=15)
    ax.set_ylabel("z", fontsize=15)
    ax.tick_params(labelsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 2])
    colors = ["#2a78d6", "#e2622e", "#1baf7a"]
    for k in range(FPCA_N_FIXED):
        scale = np.sqrt(pca3.explained_variance_[k])
        upper, lower = mean_curve + scale * pca3.components_[k], mean_curve - scale * pca3.components_[k]
        ax.fill_between(x_axis, lower, upper, color=colors[k], alpha=0.18)
        ax.plot(x_axis, upper, color=colors[k], linewidth=2,
                 label=f"φ{k + 1} ({pca3.explained_variance_ratio_[k] * 100:.1f}%)")
        ax.plot(x_axis, lower, color=colors[k], linewidth=2)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=2.5, linestyle=":", label="mean curve μ(z)")
    ax.set_title("(F) Eigenfunctions φ1-φ3 (μ(z) ± √λ_k·φ_k(z))", fontsize=18, fontweight="bold")
    ax.set_xlabel("AEC slice position z", fontsize=15)
    ax.set_ylabel("AEC value", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)

    #fig.suptitle("Figure 1. Computation of FPCA on the internal-cohort AEC-128 curves", fontsize=26,
    #             fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined Figure 1 to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_cohort(INTERNAL_XLSX)
    aec_raw = meta[AEC_COLS].astype(float).to_numpy()
    mean_curve = aec_raw.mean(axis=0)

    cum_var, scree, elbow_n = compute_scree(aec_raw)
    print(f"[FPCA] 누적 explained variance ratio(PC1-{FPCA_N_FIXED}): "
          f"{cum_var.loc[:FPCA_N_FIXED].round(4).to_dict()}, elbow k={elbow_n}")
    assert elbow_n == FPCA_N_FIXED, f"elbow({elbow_n}) != FPCA_N_FIXED({FPCA_N_FIXED}) - 캡션 k=3 근거 재확인 필요"

    pca3 = PCA(n_components=FPCA_N_FIXED, random_state=SEED).fit(aec_raw)
    scores3 = pca3.transform(aec_raw)
    recon3 = mean_curve[None, :] + scores3 @ pca3.components_
    ss_res = ((aec_raw - recon3) ** 2).sum(axis=1)
    ss_tot = ((aec_raw - mean_curve[None, :]) ** 2).sum(axis=1)
    r2_per_patient = 1 - ss_res / ss_tot

    rep_idx = select_representative_patient(scores3, r2_per_patient)
    print(f"[대표 환자] PatientID={meta['PatientID'].iloc[rep_idx]}, "
          f"score(PC1-{FPCA_N_FIXED})={scores3[rep_idx].round(1)}, R²={r2_per_patient[rep_idx]:.3f}")

    cov_matrix = np.cov((aec_raw - mean_curve[None, :]).T)
    cov_rank3 = (pca3.components_.T * pca3.explained_variance_) @ pca3.components_
    d_rep = aec_raw[rep_idx] - mean_curve

    plot_panel_a_sample_and_mean(aec_raw, mean_curve, OUTPUT_DIR / "fig2a_sample_curves_mean.png")
    plot_panel_b_projection(d_rep, pca3.components_[0], float(scores3[rep_idx, 0]),
                             OUTPUT_DIR / "fig2b_projection_score.png")
    plot_panel_c_scree_elbow(scree.loc[:20], elbow_n, OUTPUT_DIR / "fig2c_scree_elbow.png")
    plot_panel_de_covariance_matrices(cov_matrix, cov_rank3, OUTPUT_DIR / "fig2de_covariance_matrix.png")
    plot_panel_f_eigenfunctions(mean_curve, pca3, OUTPUT_DIR / "fig2f_eigenfunctions.png")
    plot_figure1_combined(aec_raw, mean_curve, cov_matrix, cov_rank3, pca3, d_rep, float(scores3[rep_idx, 0]),
                           scree.loc[:20], elbow_n, OUTPUT_DIR / "fig2_combined.png")


if __name__ == "__main__":
    main()
