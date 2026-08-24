from __future__ import annotations

# 논문에 들어가는 두 Figure(Figure 1: 코호트 선정 흐름도 / Figure 2: FPCA 계산 과정)를 만들던
# figure1_patient_selection_flow.py와 figure1_fpca_computation.py를 하나로 합친 파일(사용자 요청
# 2026-08-24: "두 파일을 합쳐줘"). 두 스크립트 모두 outputs/figure에 결과를 저장하는 논문용 Figure
# 생성 스크립트라 파일을 분리해 둘 이유가 없었다. 코드 내용은 각 원본 파일 그대로이며, main()만
# run_patient_selection_flow()/run_fpca_computation() 두 단계를 순서대로 호출하도록 합쳤다.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.decomposition import PCA

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949가 μ/φ/₁ 등을 인코딩 못 해 print에서 죽는 것 방지

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure"

INTERNAL_XLSX = DATA_DIR / "gangnam_원본.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_원본.xlsx"
AGE_CUTOFF = 20
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_N_FIXED = 3  # step_disease_logistic.py와 동일한 elbow 기반 고정값
FPCA_COMPONENT_CANDIDATES_MAX = 20  # elbow 탐색용 n_components 상한
N_SAMPLE_CURVES = 40  # 패널 A에 겹쳐 그릴 무작위 표본 곡선 수
RECON_R2_TOP_QUANTILE = 0.75  # "재구성 적합도가 상위 25% 이내" 기준
SCORE_TOP_QUANTILE = 0.75  # "PC1-3 성분점수가 모두 큼(절댓값 기준)" 기준

INTERNAL_SITE = "Gangnam Severance Hospital"
EXTERNAL_SITE = "Sinchon Severance Hospital"
STATE_WHITE = "white"
FINAL_GREEN = "#d9ead3"
BORDER = "#161616"


# ==================================================================================
# Figure 1: Materials 섹션의 코호트 선정 절차를 실제 데이터로 재현하는 patient-selection flow diagram.
# data/{gangnam,sinchon}_원본.xlsx에서 연령<20 제외만 적용하며(스캐너/kVp 제한 없음, kVp는 원본 전체가
# 100kVp뿐이라 해제 불가), 결과 인원(internal 1,088명 / external 925명)이 원본에서 매 실행마다
# 다시 계산된다(하드코딩 아님).
# ==================================================================================

def load_raw(site_key: str) -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / f"{site_key}_원본.xlsx", sheet_name="metadata", engine="openpyxl")


# 연령<20 제외만 적용한 단일 단계 필터링(스캐너/벤더 제한 없음)
def compute_flow(raw: pd.DataFrame) -> dict:
    n_enroll = len(raw)
    after_age = raw[raw["PatientAge"] >= AGE_CUTOFF]
    return {"n_enroll": n_enroll, "n_age_excluded": n_enroll - len(after_age), "n_final": len(after_age)}


# bbox_inches="tight" 저장 결과는 실제 그려진 픽셀(텍스트/박스)의 좌우 비대칭 여백을 그대로 남긴다.
# 저장 후 non-white 컨텐츠 bbox를 찾아 좌우/상하 여백이 동일하도록 다시 잘라 중앙 정렬한다.
def center_pad_png(path: Path, pad_px: int = 40) -> None:
    im = Image.open(path).convert("RGB")
    arr = np.array(im.convert("L"))
    mask = arr < 250
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    cropped = im.crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))
    canvas = Image.new("RGB", (cropped.width + 2 * pad_px, cropped.height + 2 * pad_px), "white")
    canvas.paste(cropped, (pad_px, pad_px))
    canvas.save(path)


def box(ax, renderer, cx, cy, w, h, text, facecolor, fontsize=24.0, fontweight="normal", min_fontsize=16.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=facecolor, edgecolor=BORDER, linewidth=1.6, zorder=2))
    txt = ax.text(cx, cy, text, ha="center", va="center", fontweight=fontweight, zorder=3, linespacing=1.4)

    (x0, y0) = ax.transData.transform((cx - w / 2, cy - h / 2))
    (x1, y1) = ax.transData.transform((cx + w / 2, cy + h / 2))
    box_w_px, box_h_px = abs(x1 - x0) * 0.92, abs(y1 - y0) * 0.88

    fs = fontsize
    while fs > min_fontsize:
        txt.set_fontsize(fs)
        bbox = txt.get_window_extent(renderer=renderer)
        if bbox.width <= box_w_px and bbox.height <= box_h_px:
            break
        fs -= 0.5
    txt.set_fontsize(fs)


def arrow(ax, xy_from, xy_to, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
                                  color=BORDER, zorder=1))


# 한 코호트 열(Enrollment -> 배제박스(연령<20) -> 최종 Inclusion box) — 스캐너 배제 단계 없음
def draw_column(ax, renderer, cx_main, cx_excl, flow: dict, cohort_label: str, site: str, main_w=7.0, excl_w=4.8):
    y_enroll, h_enroll = 8.6, 2.3
    y_excl, h_excl = 6.2, 1.6
    y_final, h_final = 3.6, 2.0

    box(ax, renderer, cx_main, y_enroll, main_w, h_enroll,
        f"{flow['n_enroll']:,} patients\nfrom {site}\n({cohort_label})", STATE_WHITE, fontweight="bold")

    arrow(ax, (cx_main, y_enroll - h_enroll / 2), (cx_main, y_final + h_final / 2 + 0.05))
    box(ax, renderer, cx_excl, y_excl, excl_w, h_excl,
        f"{flow['n_age_excluded']} excluded\nAge <{AGE_CUTOFF} years", STATE_WHITE, fontsize=20)
    arrow(ax, (cx_main + 0.08, y_excl), (cx_excl - excl_w / 2 - 0.08, y_excl), lw=1.2)

    box(ax, renderer, cx_main, y_final, main_w, h_final,
        f"{flow['n_final']:,} patients\n({cohort_label})", FINAL_GREEN, fontweight="bold")


def plot_diagram(flow_internal: dict, flow_external: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 9.5))
    xlim, ylim = (-0.5, 26.0), (1.7, 11.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cx_internal_main, cx_internal_excl = 3.7, 9.9
    cx_external_main, cx_external_excl = 16.3, 22.5

    draw_column(ax, renderer, cx_internal_main, cx_internal_excl, flow_internal, "internal cohort", INTERNAL_SITE)
    draw_column(ax, renderer, cx_external_main, cx_external_excl, flow_external, "external cohort", EXTERNAL_SITE)

    ax.text((cx_internal_main + cx_external_main) / 2, 10.65,
             "Both cohorts: CT examinations Jan 2018–Jun 2020 (internal) / 2019 (external); clinical data + abdominal CT at 100 kVp available for the same patient.\n"
             "No CT scanner-model restriction applied (all vendors/models retained); every CT examination in the source dataset was acquired at 100 kVp,\n"
             "so a tube-voltage restriction could not be relaxed within the available data.",
             ha="center", va="center", fontsize=18, style="italic", color="#3a3a3a")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    center_pad_png(out_path)
    print(f"Saved patient selection flow diagram to {out_path}")


def run_patient_selection_flow() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_g, raw_s = load_raw("gangnam"), load_raw("sinchon")
    flow_g, flow_s = compute_flow(raw_g), compute_flow(raw_s)

    print("Internal (Gangnam):", flow_g)
    print("External (Sinchon):", flow_s)

    plot_diagram(flow_g, flow_s, OUTPUT_DIR / "fig1_patient_selection_flow.png")


# ==================================================================================
# Figure 2: 내부 코호트 AEC-128 곡선에 대한 FPCA 계산 과정. 대표 환자 선택은 캡션 조건("PC1-3
# 성분점수가 모두 크고(상위 25% 이내) 재구성 적합도도 상위 25% 이내")을 만족하는 환자 중 네
# percentile(|PC1|,|PC2|,|PC3|,R²)의 최솟값이 가장 큰(=조건을 가장 여유있게 만족하는) 환자를
# 결정론적으로 선택한다(select_representative_patient 참고, 하드코딩 PatientID 없음).
# 영문 논문에 들어갈 그림이라 그림 안 텍스트(제목/축/범례)는 전부 영문으로 표기한다.
#
# 패널 구성(통합본 기준, 개별 파일도 동일 알파벳) 및 저장 파일(m&m 초안 Figure 2에 해당하므로 fig2 접두사로 저장):
#   A(fig2a_sample_curves_mean.png): 무작위 표본 곡선(회색) + 평균곡선 μ(z)(검정)
#   B(fig2b_projection_score.png): 대표 환자의 편차곡선 d_i(z) = x_i(z) - μ(z)를 φ1(z)에 투영해
#       score_i,1을 얻는 과정(편차곡선 자체도 이 패널에서 함께 보여줌)
#   C(fig2c_scree_elbow.png): eigenvalue scree plot과 elbow(k=3) 판단 근거
#   D/E(fig2de_covariance_matrix.png): 128x128 공분산행렬 원본(D)과 rank-3 근사(E) 비교
#   F(fig2f_eigenfunctions.png): eigenfunction φ1-φ3를 μ(z) ± √λ_k·φ_k(z)로 표시
# ==================================================================================

# 원본 metadata에서 연령<20만 제외(스캐너/벤더 제한 없음)한 뒤 aec_128 원시곡선을 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# scree curve(개별 explained variance ratio)를 구하고, 축을 0~1로 정규화한 뒤 첫점-끝점을 잇는 직선(chord)
# 까지의 수직거리를 계산한다. 거리가 최대인 지점이 elbow(Satopaa et al. 2011 Kneedle 알고리즘) -
# step_disease_logistic.py의 select_fpca_n_by_elbow와 동일 로직
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


# 캡션 조건("PC1-3 성분점수가 모두 크고 재구성 적합도가 상위 25% 이내")을 만족하는 환자 중, 네 percentile
# (|PC1|,|PC2|,|PC3|,R²)의 최솟값이 가장 큰(=조건을 가장 여유있게 만족하는) 환자를 결정론적으로 선택
def select_representative_patient(scores: np.ndarray, r2_per_patient: np.ndarray) -> int:
    score_pct = np.column_stack([pd.Series(np.abs(scores[:, k])).rank(pct=True).to_numpy()
                                  for k in range(scores.shape[1])])
    r2_pct = pd.Series(r2_per_patient).rank(pct=True).to_numpy()
    all_pct = np.column_stack([score_pct, r2_pct])

    qualifies = np.all(score_pct >= SCORE_TOP_QUANTILE, axis=1) & (r2_pct >= RECON_R2_TOP_QUANTILE)
    assert qualifies.any(), "캡션 조건(PC1-3 절댓값 상위 25% & 재구성 R² 상위 25%)을 만족하는 환자가 없음"

    min_pct = all_pct.min(axis=1)
    min_pct_masked = np.where(qualifies, min_pct, -np.inf)
    rep_idx = int(np.argmax(min_pct_masked))

    score_pct_rep = np.array([(np.abs(scores[:, k]) < np.abs(scores[rep_idx, k])).mean() for k in range(scores.shape[1])])
    r2_pct_rep = (r2_per_patient < r2_per_patient[rep_idx]).mean()
    assert np.all(score_pct_rep >= 0.75), f"대표 환자의 PC score가 '모두 큼' 조건(상위 25%)을 만족하지 않음: {score_pct_rep}"
    assert r2_pct_rep >= RECON_R2_TOP_QUANTILE, f"대표 환자의 재구성 R²가 상위 25% 조건을 만족하지 않음: {r2_pct_rep:.3f}"
    return rep_idx


# (A) 무작위 표본 곡선(회색) + 평균곡선(검정)
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


# (C) eigenvalue scree plot과 elbow(k=3) 판단 근거
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
    ax.legend(fontsize=15, loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1C to {out_path}")


# (D, E) 128x128 공분산행렬 - 원본(D)과 상위 3개 eigenfunction만으로 재구성한 rank-3 근사(E)
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


# A-F를 2행 3열 그리드 하나에 합친 통합본(docx Figure 2에 실제로 삽입되는 파일)
def plot_figure1_combined(aec_raw: np.ndarray, mean_curve: np.ndarray, cov_matrix: np.ndarray, cov_rank3: np.ndarray,
                           pca3: PCA, d_rep: np.ndarray, score1: float, scree: pd.Series, elbow_n: int,
                           out_path: Path) -> None:
    x_axis = np.arange(1, N_SLICES + 1)
    vmin, vmax = float(cov_matrix.min()), float(cov_matrix.max())
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(aec_raw.shape[0], size=min(N_SAMPLE_CURVES, aec_raw.shape[0]), replace=False)

    fig = plt.figure(figsize=(19, 26))
    gs = fig.add_gridspec(3, 2, hspace=0.7, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0])
    for i in sample_idx:
        ax.plot(x_axis, aec_raw[i], color="#c9c7c1", linewidth=1, alpha=0.6)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=3, label="mean curve μ(z)")
    ax.set_title("(A) Sample curves and mean curve", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("AEC value", fontsize=26)
    ax.tick_params(labelsize=21)
    ax.legend(fontsize=22, loc="best")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    ax.plot(x_axis, d_rep, color="#2a78d6", linewidth=2.5, label="d_i(z)")
    ax2.plot(x_axis, pca3.components_[0], color="#e2622e", linewidth=2.5, linestyle="--", label="φ1(z)")
    ax.set_title("(B) Deviation curve, projected onto φ1 → score", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("d_i(z) = x_i(z) - μ(z)", fontsize=26, color="#2a78d6")
    ax2.set_ylabel("φ1(z)", fontsize=26, color="#e2622e")
    ax.tick_params(labelsize=21)
    ax2.tick_params(labelsize=21)
    ax.text(0.03, 0.05, f"score_i,1 = Σ d_i(z)·φ1(z)\n= {score1:,.1f}", transform=ax.transAxes, fontsize=24,
             va="bottom", ha="left", bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#161616"})
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=21, loc="upper right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(scree.index, scree.values, marker="o", markersize=7, linewidth=2.5, color="#161616",
            label="individual explained variance ratio")
    ax.plot([scree.index[0], scree.index[-1]], [scree.values[0], scree.values[-1]], color="#898781",
            linestyle=":", linewidth=2, label="first-to-last point line (chord)")
    ax.axvline(elbow_n, color="#e2622e", linestyle="--", linewidth=2.5, label=f"elbow k={elbow_n}")
    ax.set_xticks(list(scree.index))
    ax.set_title("(C) Scree plot / elbow", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("component index", fontsize=26)
    ax.set_ylabel("individual explained variance ratio", fontsize=26)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=13, loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=True)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(cov_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title("(D) Covariance matrix (original)", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("z", fontsize=26)
    ax.set_ylabel("z", fontsize=26)
    ax.tick_params(labelsize=21)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=19)

    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(cov_rank3, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(f"(E) Covariance matrix (rank-{FPCA_N_FIXED} approx.)", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("z", fontsize=26)
    ax.set_ylabel("z", fontsize=26)
    ax.tick_params(labelsize=21)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=19)

    ax = fig.add_subplot(gs[2, 1])
    colors = ["#2a78d6", "#e2622e", "#1baf7a"]
    for k in range(FPCA_N_FIXED):
        scale = np.sqrt(pca3.explained_variance_[k])
        upper, lower = mean_curve + scale * pca3.components_[k], mean_curve - scale * pca3.components_[k]
        ax.fill_between(x_axis, lower, upper, color=colors[k], alpha=0.18)
        ax.plot(x_axis, upper, color=colors[k], linewidth=2,
                 label=f"φ{k + 1} ({pca3.explained_variance_ratio_[k] * 100:.1f}%)")
        ax.plot(x_axis, lower, color=colors[k], linewidth=2)
    ax.plot(x_axis, mean_curve, color="#161616", linewidth=2.5, linestyle=":", label="mean curve μ(z)")
    ax.set_title("(F) Eigenfunctions φ1-φ3 (μ(z) ± √λ_k·φ_k(z))", fontsize=24, fontweight="bold", pad=20)
    ax.set_xlabel("AEC slice position z", fontsize=26)
    ax.set_ylabel("AEC value", fontsize=26)
    ax.tick_params(labelsize=21)
    ax.legend(fontsize=20, loc="best")
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined Figure 1 to {out_path}")


def run_fpca_computation() -> None:
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


def main() -> None:
    run_patient_selection_flow()
    run_fpca_computation()


if __name__ == "__main__":
    main()
