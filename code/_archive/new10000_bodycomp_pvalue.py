from __future__ import annotations

# new10000_final_dataset의 체성분 128구간 곡선(AEC/VAT/SAT/LAMA/NAMA/IMATA, liver->pubis)을 질환(HTN/DM)별
# 양성/음성 두 그룹의 평균곡선 ± 95% CI 리본으로 겹쳐 그리고, 128개 slice를 하나의 벡터로 보는 전체-곡선
# RMSD permutation test로 단일 p-value를 구해 주석으로 표시한다.
# (docs/260729_Stratified Analysis of AEC.pptx 슬라이드 9~11, code/baseline/aec_curve_comparison.py의
# plot_curve_comparison/curve_diff_test와 동일한 스타일·방법론을 재사용 - slice별 개별 t-test 대신
# 곡선 전체를 하나의 검정통계량(RMSD)으로 보고 라벨을 섞는 permutation으로 유의성을 본다)
# CKD는 new10000에 컬럼이 없어 제외. raw(원본값, 체격 차이가 confound일 수 있음)와 patient-normalized
# (환자 자신의 128-slice 평균으로 나눠 체격을 지운 뒤 곡선 "형태"만 비교) 두 버전을 모두 만든다
# (code/baseline/aec_curve_comparison.py load_data의 patient-normalized 정의와 동일: curve/patient_mean).

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트(없으면 그래프 한글 라벨이 깨짐)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = PROJECT_ROOT / "data" / "new10000_final_dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic4" / "new10000_pvalue"

N_SLICES = 128
DISEASES = ["HTN", "DM"]  # new10000엔 CKD 컬럼 없음
# name -> (sheet, prefix)
CURVES: dict[str, tuple[str, str]] = {
    "AEC": ("aec_128", "aec"), "VAT": ("VFA_128", "VFA"), "SAT": ("SFA_128", "SFA"),
    "LAMA": ("LAMA_128", "LAMA"), "NAMA": ("NAMA_128", "NAMA"), "IMATA": ("IMATA_128", "IMATA"),
}
N_PERM = 2000
SEED = 42

# code/baseline/aec_curve_comparison.py와 동일한 팔레트
COL_POS, COL_NEG = "#af1b1b", "#2d2ad6"  # 양성=red, 음성=blue
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def smooth(mat_mean: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(mat_mean).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


# 그룹 하나(mask)의 slice별 평균곡선(smoothed)과 표준편차(std, smoothed) 반환 - 개별 환자값이
# 얼마나 퍼져있는지를 보여줌(평균의 신뢰구간인 SEM*1.96과는 다름)
def group_curve_stats(curve: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    sub = curve[mask]
    mean = sub.mean(axis=0)
    sd = sub.std(axis=0, ddof=1)
    return smooth(mean), smooth(sd), int(mask.sum())


def plot_curve_comparison(ax, curve: np.ndarray, y: np.ndarray, title: str, ylabel: str) -> None:
    x = np.arange(1, N_SLICES + 1)
    for val, label, color in [(1, "Positive", COL_POS), (0, "Negative", COL_NEG)]:
        mean, sd, n = group_curve_stats(curve, y == val)
        ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Slice index (1-128, liver -> pubis)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    style_axes(ax)


# 128 slice를 개별 포인트로 따로 검정하지 않고, 두 그룹 평균곡선 사이의 RMSD를 하나의 전역
# 검정통계량으로 삼아 라벨을 섞는 permutation test로 유의성을 본다(code/baseline/aec_curve_comparison.py
# curve_diff_test와 동일 로직)
def curve_diff_test(curve: np.ndarray, y: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    def curve_stat(labels: np.ndarray) -> tuple[float, np.ndarray]:
        mean_pos = curve[labels == 1].mean(axis=0)
        mean_neg = curve[labels == 0].mean(axis=0)
        deviation = mean_pos - mean_neg
        return float(np.sqrt(np.mean(deviation**2))), deviation

    obs_stat, obs_deviation = curve_stat(y)
    peak_idx = int(np.argmax(np.abs(obs_deviation)))

    rng = np.random.default_rng(seed)
    perm_labels = y.copy()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(perm_labels)
        perm_stats[i] = curve_stat(perm_labels)[0]
    p = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)

    return {"rmsd": obs_stat, "p_value": float(p), "peak_slice": peak_idx + 1,
            "peak_deviation": float(obs_deviation[peak_idx]), "n_perm": n_perm}


def diff_note(r: dict) -> str:
    return (f"RMSD={r['rmsd']:.4f}, perm p={r['p_value']:.3g} (n_perm={r['n_perm']}; "
            f"peak Δ={r['peak_deviation']:+.3f} @ slice {r['peak_slice']})")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_excel(DATA_XLSX, sheet_name="metadata")[["PatientID"] + DISEASES]

    curve_data: dict[str, pd.DataFrame] = {}
    for name, (sheet, prefix) in CURVES.items():
        df = pd.read_excel(DATA_XLSX, sheet_name=sheet)[["PatientID"] + curve_cols(prefix)]
        curve_data[name] = meta.merge(df, on="PatientID", how="inner")

    # (normalization 이름, 곡선 변환 함수, 제목/파일명용 라벨, y축 라벨 접미사)
    NORMALIZATIONS = [
        ("raw", lambda c: c, "raw", ""),
        ("patient_norm", lambda c: c / c.mean(axis=1, keepdims=True), "patient-normalized",
         " (patient-normalized)"),
    ]

    summary_rows = []
    for disease in DISEASES:
        for name, (_, prefix) in CURVES.items():
            df = curve_data[name].dropna(subset=[disease])
            y = df[disease].to_numpy(dtype=int)
            curve_raw = df[curve_cols(prefix)].to_numpy(dtype=float)

            for norm_key, transform, norm_label, ylabel_suffix in NORMALIZATIONS:
                curve = transform(curve_raw)
                r = curve_diff_test(curve, y)
                fig, ax = plt.subplots(figsize=(8, 5.5))
                plot_curve_comparison(ax, curve, y,
                                      f"new10000 — {disease}: {name}-128 {norm_label}, Positive vs Negative",
                                      f"{name}{ylabel_suffix}")
                ax.text(0.02, 0.02, diff_note(r), transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
                fig.patch.set_facecolor(SURFACE)
                fig.tight_layout()
                out_path = OUTPUT_DIR / disease / f"{name}_{norm_key}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
                plt.close(fig)

                print(f"[{disease}] {name} ({norm_key}) n={len(y)} pos={int(y.sum())} {diff_note(r)}")
                summary_rows.append({"disease": disease, "curve": name, "normalization": norm_key,
                                     "n": len(y), "n_pos": int(y.sum()), **r})

    summary = pd.DataFrame(summary_rows)
    summary["significant_p<0.05"] = summary["p_value"] < 0.05
    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {summary_path}")

    # self-check: 질환수 x 곡선수 x 정규화방식수만큼 이미지 파일이 실제로 저장됐는지 확인
    n_expected = len(DISEASES) * len(CURVES) * len(NORMALIZATIONS)
    saved_files = list(OUTPUT_DIR.rglob("*.png"))
    assert len(saved_files) == n_expected, f"expected {n_expected} files, found {len(saved_files)}"
    print(f"OK: {len(saved_files)} PNG files exist on disk")


if __name__ == "__main__":
    main()
