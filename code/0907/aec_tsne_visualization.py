from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0907" / "aec_tsne"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
COHORTS = {"internal": INTERNAL_XLSX, "external": EXTERNAL_XLSX}

AGE_CUTOFF = 20
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
EXCLUDED_PATIENT_IDS: set[int] = set()
# perplexity 상한(사용자 확인 없이 임의 대형값을 쓰면 소표본 cohort에서 과도해지므로 표본수에 비례해 상한 적용)
PERPLEXITY = 30

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

_REF_AVG_DIM = (16.0 + 11.0) / 2
_LABEL_FS_RATIO = 32.5 / _REF_AVG_DIM
_TICK_FS_RATIO = 30.0 / _REF_AVG_DIM
_LEGEND_FS_RATIO = 27.5 / _REF_AVG_DIM


def scaled_fontsizes(width: float, height: float) -> tuple[float, float, float]:
    avg_dim = (width + height) / 2
    return _LABEL_FS_RATIO * avg_dim, _TICK_FS_RATIO * avg_dim, _LEGEND_FS_RATIO * avg_dim


def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[~meta["PatientID"].isin(EXCLUDED_PATIENT_IDS)].reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# patient-wise z-score(사용자 요청 2026-09-07: "raw AEC, patient-wise 모두"). code/03_aec_deep_learning/
# fusion/aec_fusion_common.py의 prepare_curve(mode="patient_zscore")와 동일 정의 — 환자별 행(row) 단위
# mean/std만 사용하고 cohort 통계는 전혀 참조하지 않는다([[feedback_aec_preprocessing_methods]])
def patient_wise_zscore(curve: np.ndarray) -> np.ndarray:
    mean = curve.mean(axis=1, keepdims=True)
    std = curve.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (curve - mean) / std


def run_tsne(curve: np.ndarray) -> np.ndarray:
    perplexity = min(PERPLEXITY, max(5, (len(curve) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=SEED, init="pca").fit_transform(curve)


# FPCA는 선형 투영이라 곡선 간 비선형 유사도는 놓칠 수 있다는 우려(사용자 질문 2026-09-07: "fpca말고 t-SNE로
# 확인하는 방법도 있을까?")에 대한 순수 시각화 진단 — t-SNE는 out-of-sample transform이 없어 internal-fit/
# external-frozen 검증 체계([[feedback_internal_external_validation_discipline]])에는 쓸 수 없으므로 로지스틱
# 회귀 feature가 아니라 "질환군별로 곡선이 조금이라도 뭉치는가"를 눈으로 보는 보조 진단으로만 사용
def plot_mode(mode_label: str, embeddings: dict[str, np.ndarray], meta_by_cohort: dict[str, pd.DataFrame],
              out_path: Path) -> None:
    cohorts = list(COHORTS)
    features = list(FEATURES)
    panel_w, panel_h = 6.0, 6.0
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    fig, axes = plt.subplots(len(cohorts), len(features),
                              figsize=(panel_w * len(features), panel_h * len(cohorts)))
    for row, cohort in enumerate(cohorts):
        emb = embeddings[cohort]
        meta = meta_by_cohort[cohort]
        for col, feat in enumerate(features):
            ax = axes[row, col]
            y = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y)
            neg, pos = mask & (y == 0), mask & (y == 1)
            ax.scatter(emb[neg, 0], emb[neg, 1], s=10, c="#c7c7c7", alpha=0.6,
                       label=f"{feat}(-) n={int(neg.sum())}")
            ax.scatter(emb[pos, 0], emb[pos, 1], s=16, c="#e2622e", alpha=0.85,
                       label=f"{feat}(+) n={int(pos.sum())}")
            ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
            ax.set_xlabel("t-SNE 1", fontsize=label_fs)
            ax.set_ylabel("t-SNE 2", fontsize=label_fs)
            ax.tick_params(labelsize=tick_fs)
            ax.legend(fontsize=legend_fs * 0.6, loc="best", frameon=False)
            ax.grid(alpha=0.3)
    fig.suptitle(f"AEC-128 t-SNE embedding ({mode_label})", fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_by_cohort = {name: load_cohort(path) for name, path in COHORTS.items()}
    for name, meta in meta_by_cohort.items():
        print(f"[{name}] n={len(meta)}")

    modes = {
        "raw": lambda c: c,
        "patient_wise": patient_wise_zscore,
    }
    for mode_name, prep_fn in modes.items():
        embeddings: dict[str, np.ndarray] = {}
        for name, meta in meta_by_cohort.items():
            curve = prep_fn(meta[AEC_COLS].astype(float).to_numpy())
            print(f"[{mode_name} / {name}] running t-SNE on n={len(curve)} curves (dim={curve.shape[1]})...")
            embeddings[name] = run_tsne(curve)
        plot_mode(mode_name, embeddings, meta_by_cohort, OUTPUT_DIR / f"aec128_tsne_{mode_name}.png")


if __name__ == "__main__":
    main()
