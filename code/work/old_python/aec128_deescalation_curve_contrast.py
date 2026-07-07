from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    clinical_estimator,
    clinical_matrix,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    zfit_apply,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec128_deescalation_curve_contrast"
SEED = 20260630


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 원시행렬·행정규화 곡선·모델용(-1 오프셋) 곡선과 저근감소증 라벨을 함께 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": norm, "x_model": norm - 1.0, "y": y}


def aec128_model(seed: int) -> Pipeline:
    """결측대체→표준화→상위 64개 특징 선택→선형 SVM으로 이어지는 AEC 전용 분류 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=64)),
            ("svm", LinearSVC(C=0.2, class_weight="balanced", max_iter=20000, random_state=seed)),
        ]
    )


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """위치별 평균과 95% 신뢰구간(정규근사)을 계산."""
    mean = np.nanmean(x, axis=0)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])
    return mean, mean - 1.96 * se, mean + 1.96 * se


def group_row(cohort: str, label: str, y: np.ndarray, mask: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray) -> dict:
    """한 그룹(유지/하향조정)의 표본수·이벤트·유병률과 평균 임상/AEC z점수를 한 행으로 정리."""
    n = int(mask.sum())
    events = int(y[mask].sum())
    return {
        "cohort": cohort,
        "group": label,
        "n": n,
        "low_smi_events": events,
        "low_smi_prevalence": events / n if n else np.nan,
        "clinical_z_mean": float(np.mean(clinical_score[mask])) if n else np.nan,
        "aec128_score_z_mean": float(np.mean(aec_score[mask])) if n else np.nan,
    }


def plot_curves(curve_df: pd.DataFrame, curve_type: str, path: Path) -> None:
    """코호트별로 (AEC-유지 vs AEC-하향조정 평균곡선) + (두 그룹 차이곡선) 2x2 패널 그래프를 그려 PNG로 저장."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6), sharex=True)
    colors = {"AEC-kept": "#4C78A8", "AEC-de-escalated": "#F58518"}
    for row_i, cohort in enumerate(["g1090_oof", "sdata_external"]):
        ax = axes[row_i, 0]
        for label in ["AEC-kept", "AEC-de-escalated"]:
            sub = curve_df[(curve_df["cohort"].eq(cohort)) & (curve_df["curve_type"].eq(curve_type)) & (curve_df["group"].eq(label))]
            ax.plot(sub["point"], sub["mean"], lw=2.3, color=colors[label], label=label)
            ax.fill_between(sub["point"], sub["ci95_low"], sub["ci95_high"], color=colors[label], alpha=0.14, lw=0)
        ax.set_title(f"{cohort}: clinical-positive groups", loc="left", fontweight="bold")
        ax.set_ylabel("Raw AEC" if curve_type == "raw" else "Patient-normalized AEC")
        ax.grid(alpha=0.24)
        if row_i == 0:
            ax.legend(frameon=False)

        axd = axes[row_i, 1]
        diff = curve_df[
            (curve_df["cohort"].eq(cohort)) & (curve_df["curve_type"].eq(curve_type)) & (curve_df["group"].eq("deesc_minus_kept"))
        ]
        axd.plot(diff["point"], diff["mean"], lw=2.4, color="#C84630")
        axd.fill_between(diff["point"], diff["ci95_low"], diff["ci95_high"], color="#C84630", alpha=0.15, lw=0)
        axd.axhline(0, color="#555555", lw=1, ls="--")
        axd.set_title(f"{cohort}: de-escalated - kept", loc="left", fontweight="bold")
        axd.set_ylabel("Difference")
        axd.grid(alpha=0.24)
    axes[1, 0].set_xlabel("AEC_128 point")
    axes[1, 1].set_xlabel("AEC_128 point")
    fig.suptitle(
        f"Clinical-positive AEC de-escalation contrast ({curve_type})",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_segments(diff: np.ndarray, threshold: float = 0.02) -> list[dict]:
    """차이곡선(diff)에서 절댓값이 threshold를 넘는 연속 구간(길이 3 이상)을 찾아, 방향·위치·평균차·최대절댓값차를 정리."""
    rows = []
    for direction, mask in [("deesc_higher", diff > threshold), ("deesc_lower", diff < -threshold)]:
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            if start is not None and ((not val) or i == len(mask) - 1):
                end = i if val and i == len(mask) - 1 else i - 1
                if end - start + 1 >= 3:
                    rows.append(
                        {
                            "direction": direction,
                            "start_point": start + 1,
                            "end_point": end + 1,
                            "length": end - start + 1,
                            "mean_diff": float(np.mean(diff[start : end + 1])),
                            "max_abs_diff": float(np.max(np.abs(diff[start : end + 1]))),
                        }
                    )
                start = None
    return rows


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 95% 민감도 기준 양성 환자 중, AEC 가우시안 게이트로
    "유지" vs "하향조정"으로 나뉜 두 그룹은 AEC 곡선 자체의 모양이 실제로 다르게 생겼는가?):

    1. g1090/sdata를 로드하고 임상 단독 모델과 AEC128 단독(SVM) 모델의 OOF/외부 점수를 표준화.
    2. 임상 95% 민감도 임계값을 고정하고, 폭0.40·람다0.25의 가우시안 경계 게이트로 임상 양성군을
       "유지(kept)"와 "하향조정(deesc)" 두 그룹으로 나눈다.
    3. 두 그룹의 표본수·사건율·평균 점수를 표로 정리.
    4. 정규화 곡선과 원시 곡선 각각에 대해, 두 그룹의 위치별 평균(+95%CI)과 차이곡선(하향조정-유지)을
       계산하고, summarize_segments로 차이가 뚜렷한 연속 구간들을 찾아 표로 저장.
    5. 그룹별 평균곡선+차이곡선 그래프를 정규화/원시 버전 각각 PNG로 저장.
    6. 그룹 요약표와 눈에 띄는 차이 구간을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    xclin_g, xclin_s, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"], 5)

    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_g, g["y"], xclin_s, folds)
    aec_oof, aec_ext = oof_and_external(lambda seed: aec128_model(seed), g["x_model"], g["y"], s["x_model"], folds)
    c_g, c_s, _, _ = zfit_apply(clinical_oof, clinical_ext)
    a_g, a_s, _, _ = zfit_apply(aec_oof, aec_ext)
    if np.corrcoef(a_g, g["y"])[0, 1] < 0:
        a_g = -a_g
        a_s = -a_s

    t95 = (threshold_for_min_sensitivity(g["y"], clinical_oof, 0.95) - np.mean(clinical_oof)) / np.std(clinical_oof)
    width = 0.40
    lam = 0.25

    rows = []
    curve_rows = []
    segment_rows = []
    split_masks = {}
    for cohort, d, c, a in [("g1090_oof", g, c_g, a_g), ("sdata_external", s, c_s, a_s)]:
        boundary = np.exp(-0.5 * ((c - t95) / width) ** 2)
        gate_score = c + lam * boundary * a
        clinical_pos = c >= t95
        deesc = clinical_pos & (gate_score < t95)
        kept = clinical_pos & (gate_score >= t95)
        split_masks[cohort] = {"kept": kept, "deesc": deesc}

        rows.extend(
            [
                group_row(cohort, "AEC-kept", d["y"], kept, c, a),
                group_row(cohort, "AEC-de-escalated", d["y"], deesc, c, a),
            ]
        )

        for curve_type, mat in [("normalized", d["norm"]), ("raw", d["raw"])]:
            kept_mean, kept_lo, kept_hi = mean_ci(mat[kept])
            de_mean, de_lo, de_hi = mean_ci(mat[deesc])
            diff = de_mean - kept_mean
            # Approximate CI for difference by independent SEs.
            kept_se = np.nanstd(mat[kept], axis=0, ddof=1) / np.sqrt(np.sum(kept))
            de_se = np.nanstd(mat[deesc], axis=0, ddof=1) / np.sqrt(np.sum(deesc))
            diff_se = np.sqrt(kept_se**2 + de_se**2)
            for j in range(128):
                point = j + 1
                curve_rows.extend(
                    [
                        {
                            "cohort": cohort,
                            "curve_type": curve_type,
                            "group": "AEC-kept",
                            "point": point,
                            "mean": kept_mean[j],
                            "ci95_low": kept_lo[j],
                            "ci95_high": kept_hi[j],
                        },
                        {
                            "cohort": cohort,
                            "curve_type": curve_type,
                            "group": "AEC-de-escalated",
                            "point": point,
                            "mean": de_mean[j],
                            "ci95_low": de_lo[j],
                            "ci95_high": de_hi[j],
                        },
                        {
                            "cohort": cohort,
                            "curve_type": curve_type,
                            "group": "deesc_minus_kept",
                            "point": point,
                            "mean": diff[j],
                            "ci95_low": diff[j] - 1.96 * diff_se[j],
                            "ci95_high": diff[j] + 1.96 * diff_se[j],
                        },
                    ]
                )
            threshold = 0.02 if curve_type == "normalized" else float(0.1 * np.nanstd(mat))
            for seg in summarize_segments(diff, threshold=threshold):
                segment_rows.append({"cohort": cohort, "curve_type": curve_type, **seg})

    group_df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curve_rows)
    segment_df = pd.DataFrame(segment_rows)
    group_df.to_csv(OUT_DIR / "clinical_positive_deescalation_group_summary.csv", index=False)
    curve_df.to_csv(OUT_DIR / "clinical_positive_deescalation_mean_curves.csv", index=False)
    segment_df.to_csv(OUT_DIR / "clinical_positive_deescalation_difference_segments.csv", index=False)
    plot_curves(curve_df, "normalized", OUT_DIR / "clinical_positive_deescalation_normalized_curves.png")
    plot_curves(curve_df, "raw", OUT_DIR / "clinical_positive_deescalation_raw_curves.png")

    print("Group summary")
    print(group_df.to_string(index=False))
    print("\nLargest normalized deesc-kept segments")
    norm_seg = segment_df[segment_df["curve_type"].eq("normalized")].sort_values(["cohort", "max_abs_diff"], ascending=[True, False])
    print(norm_seg.to_string(index=False) if not norm_seg.empty else "None")
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
