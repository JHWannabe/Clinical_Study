from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_2dcnn_deep_dive import (  # noqa: E402
    DATA_DIR,
    WORK_DATA_DIR,
    build_curves,
    d1,
    d2,
    load_dataset,
    midlate_mask,
    residual,
)
from aec_conditional_value import row_norm  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_2dcnn_deep_dive_extremes"
SCORE_PATH = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_2dcnn_deep_dive" / "cnn_deep_dive_scores.csv"
EXPERIMENT = "company_midlate_resid_slope_curv"
MODEL_ARCH = "tiny"
REGIME = "strong_reg"
N = 12


def add_index(scores: pd.DataFrame) -> pd.DataFrame:
    """실험x정규화강도x아키텍처x코호트 그룹 내에서 순서대로 0부터 매긴 row_index를 추가."""
    out = scores.copy()
    out["row_index"] = out.groupby(["experiment", "regime", "model_arch", "cohort"]).cumcount()
    return out


def selected_scores() -> pd.DataFrame:
    """aec_2dcnn_deep_dive가 저장한 점수 CSV에서, 가장 유망했던 실험(EXPERIMENT/MODEL_ARCH/REGIME) 하나만 골라냄."""
    scores = add_index(pd.read_csv(SCORE_PATH))
    return scores[
        scores["experiment"].eq(EXPERIMENT)
        & scores["model_arch"].eq(MODEL_ARCH)
        & scores["regime"].eq(REGIME)
    ].copy()


def extreme_table(scores: pd.DataFrame, data_by: dict[str, dict]) -> pd.DataFrame:
    """선택된 실험의 CNN 점수 중 코호트별로 가장 높은/낮은 12명씩을 뽑아 실제 라벨·임상변수와 함께 표로 정리."""
    rows = []
    for cohort in ["Gangnam", "Sinchon"]:
        sub = scores[scores["cohort"].eq(cohort)].copy()
        for group, selected in [
            ("cnn_high_low_smi_likely", sub.nlargest(N, "aec_2dcnn")),
            ("cnn_low_low_smi_unlikely", sub.nsmallest(N, "aec_2dcnn")),
        ]:
            meta = data_by[cohort]["meta"]
            for rank, (_, r) in enumerate(selected.iterrows(), start=1):
                i = int(r["row_index"])
                m = meta.iloc[i]
                rows.append(
                    {
                        "cohort": cohort,
                        "cnn_group": group,
                        "rank": rank,
                        "row_index": i,
                        "true_low_smi": int(r["y"]),
                        "cnn_score": float(r["aec_2dcnn"]),
                        "clinical_score": float(r["clinical"]),
                        "sex": str(m.get("PatientSex", "")),
                        "age": float(m.get("PatientAge", np.nan)),
                        "bmi": float(m.get("BMI", np.nan)),
                        "smi": float(m.get("SMI", np.nan)),
                        "manufacturer": str(m.get("Manufacturer", "")),
                    }
                )
    return pd.DataFrame(rows)


def group_idx(table: pd.DataFrame, cohort: str, group: str) -> np.ndarray:
    """코호트x그룹(CNN 고점/저점)에 해당하는 원본 데이터 행 인덱스들을 뽑아냄."""
    return table[(table["cohort"].eq(cohort)) & (table["cnn_group"].eq(group))]["row_index"].to_numpy(dtype=int)


def plot_contrast(table: pd.DataFrame, data_by: dict[str, dict], curves_by: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    """코호트x4채널(원곡선/잔차/기울기/곡률, 중반후반 마스크 적용) 2x4 그리드에서, CNN 고점군 vs
    저점군의 평균±표준오차 곡선을 비교해 PNG로 저장 (최고 성능 실험이 실제로 무엇을 보는지 확인)."""
    curve_g, curve_s = curves_by["mean_norm_company_harmonized"]
    orig_g = row_norm(ndimage.gaussian_filter1d(data_by["Gangnam"]["raw"], sigma=1.0, axis=1, mode="nearest"))
    orig_s = row_norm(ndimage.gaussian_filter1d(data_by["Sinchon"]["raw"], sigma=1.0, axis=1, mode="nearest"))
    mats = {
        "Gangnam": {
            "smoothed mean-norm curve": orig_g,
            "company-harmonized residual": residual(curve_g),
            "company-harmonized slope": d1(curve_g),
            "company-harmonized curvature": d2(curve_g),
        },
        "Sinchon": {
            "smoothed mean-norm curve": orig_s,
            "company-harmonized residual": residual(curve_s),
            "company-harmonized slope": d1(curve_s),
            "company-harmonized curvature": d2(curve_s),
        },
    }
    mask = midlate_mask()[None, :]
    for cohort in ["Gangnam", "Sinchon"]:
        mats[cohort]["company-harmonized residual"] = mats[cohort]["company-harmonized residual"] * mask
        mats[cohort]["company-harmonized slope"] = mats[cohort]["company-harmonized slope"] * mask
        mats[cohort]["company-harmonized curvature"] = mats[cohort]["company-harmonized curvature"] * mask

    xs = np.arange(1, 129)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), sharex=True)
    colors = {"cnn_high_low_smi_likely": "#b2182b", "cnn_low_low_smi_unlikely": "#2166ac"}
    for row, cohort in enumerate(["Gangnam", "Sinchon"]):
        for col, (name, mat) in enumerate(mats[cohort].items()):
            ax = axes[row, col]
            for group, color in colors.items():
                idx = group_idx(table, cohort, group)
                mean = mat[idx].mean(axis=0)
                se = mat[idx].std(axis=0, ddof=1) / np.sqrt(len(idx))
                label = "CNN high" if group.startswith("cnn_high") else "CNN low"
                ax.plot(xs, mean, color=color, lw=2.0, label=label)
                ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
            ax.axhline(1.0 if col == 0 else 0.0, color="0.65", ls="--", lw=0.8)
            ax.axvspan(41, 118, color="0.86", alpha=0.35, linewidth=0)
            ax.set_title(f"{cohort}: {name}", fontsize=9, fontweight="bold")
            ax.grid(alpha=0.17)
            if row == 0 and col == 3:
                ax.legend(frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("AEC index: 1 inferior pubic margin -> 128 liver dome")
    fig.suptitle(f"2D CNN deep-dive extremes: {EXPERIMENT}, {MODEL_ARCH}/{REGIME}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """코호트x그룹별 표본수, 실제 사건율, CNN/임상 점수 평균, SMI/BMI, 여성 비율을 요약."""
    return (
        table.groupby(["cohort", "cnn_group"])
        .agg(
            n=("row_index", "size"),
            true_low_smi_n=("true_low_smi", "sum"),
            true_low_smi_rate=("true_low_smi", "mean"),
            cnn_score_mean=("cnn_score", "mean"),
            clinical_score_mean=("clinical_score", "mean"),
            smi_mean=("smi", "mean"),
            bmi_mean=("bmi", "mean"),
            female_rate=("sex", lambda s: float(np.mean(s.astype(str).str.upper().eq("F")))),
        )
        .reset_index()
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_2dcnn_deep_dive의 36개 조합 중 가장 좋았던
    "company_midlate_resid_slope_curv / tiny / strong_reg" 조합이, 실제로 어떤 사례에서
    어떤 곡선 모양을 근거로 고/저 점수를 매기는가? — 최고 성능 모델에 대한 사례 기반 검토):

    1. g1090/sdata를 로드하고 build_curves로 곡선 표현들을 재구성, 저장된 점수 CSV에서 최고
       성능 실험 하나만 골라 코호트별 상위/하위 12명씩을 뽑는다.
    2. plot_contrast로 고점군 vs 저점군의 (원곡선/잔차/기울기/곡률, 중반후반 마스크 적용) 평균
       곡선을 2x4 그래프로 비교해 PNG로 저장.
    3. 그룹별 요약통계(사건율, 점수평균, SMI/BMI, 여성비율)를 CSV로 저장.
    4. 요약과, 그룹별 제조사 분포(장비 편향이 있는지 확인)를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    data_by = {"Gangnam": load_dataset(g_path), "Sinchon": load_dataset(s_path)}
    curves_by = build_curves(data_by["Gangnam"], data_by["Sinchon"])
    table = extreme_table(selected_scores(), data_by)
    summary = summarize(table)
    table.to_csv(OUT_DIR / "best_midlate_cnn_top_bottom_cases.csv", index=False)
    summary.to_csv(OUT_DIR / "best_midlate_cnn_top_bottom_summary.csv", index=False)
    plot_contrast(table, data_by, curves_by, OUT_DIR / "best_midlate_cnn_high_vs_low_contrast.png")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nMANUFACTURERS")
    for cohort in ["Gangnam", "Sinchon"]:
        for group in ["cnn_high_low_smi_likely", "cnn_low_low_smi_unlikely"]:
            sub = table[(table["cohort"].eq(cohort)) & (table["cnn_group"].eq(group))]
            print(f"\n{cohort} {group}")
            print(sub["manufacturer"].value_counts().to_string())
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 최고 성능 CNN 실험(company_midlate_resid_slope_curv/tiny/strong_reg)의 코호트별 상위/하위
    # 12명 사례를 곡선 기반으로 대조 분석하는 파이프라인을 실행한다.
    main()
