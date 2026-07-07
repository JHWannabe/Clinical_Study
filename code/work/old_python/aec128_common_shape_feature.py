from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aec_conditional_value import DATA_DIR


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_common_shape_feature"
FILES = {
    "g1090": DATA_DIR / "g1090.xlsx",
    "sdata": DATA_DIR / "sdata.xlsx",
}


def aec_cols(df: pd.DataFrame) -> list[str]:
    """데이터프레임에서 'aec_'로 시작하는 컬럼명을 뽑아 번호 순서로 정렬."""
    return sorted([c for c in df.columns if str(c).startswith("aec_")], key=lambda c: int(str(c).split("_")[1]))


def fill_row_nan(x: np.ndarray) -> np.ndarray:
    """각 행의 결측값을 같은 행의 값으로 선형보간해서 채움 (유효값이 하나면 그 값으로, 하나도 없으면 NaN 유지)."""
    out = x.copy().astype(float)
    grid = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        row = out[i]
        ok = np.isfinite(row)
        if ok.all():
            continue
        if ok.sum() == 0:
            out[i] = np.nan
        elif ok.sum() == 1:
            out[i, ~ok] = row[ok][0]
        else:
            out[i, ~ok] = np.interp(grid[~ok], grid[ok], row[ok])
    return out


def load_aec128(path: Path) -> dict:
    """엑셀 파일에서 aec_128 시트(128개 컬럼 검증)를 읽어 결측 보간→행 평균 정규화하고, 저근감소증 라벨(y)과 메타데이터를 함께 반환."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    aec = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")
    cols = aec_cols(aec)
    if len(cols) != 128:
        raise ValueError(f"{path.name} has {len(cols)} AEC columns, expected 128")

    x = aec[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x = fill_row_nan(x)
    row_mean = np.nanmean(x, axis=1)
    row_mean[~np.isfinite(row_mean) | (row_mean == 0)] = 1.0
    x_norm = x / row_mean[:, None]

    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / (height_m**2)
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(bool)
    return {"x": x_norm, "y": y, "meta": meta}


def contiguous_segments(mask: np.ndarray, min_len: int = 4) -> list[tuple[int, int]]:
    """불리언 배열에서 True가 연속되는 구간(길이 min_len 이상)들을 (시작, 끝) 인덱스로 찾음."""
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if start is not None and (not val or i == len(mask) - 1):
            end = i if val and i == len(mask) - 1 else i - 1
            if end - start + 1 >= min_len:
                segments.append((start, end))
            start = None
    return segments


def feature_stats(feature: np.ndarray, y: np.ndarray) -> dict:
    """저근감소증군 vs 비저근감소증군의 특징값 평균차·Cohen's d·AUC·Welch t검정·Mann-Whitney U검정 결과를 계산."""
    low = feature[y]
    non = feature[~y]
    delta = float(np.mean(low) - np.mean(non))
    pooled_sd = np.sqrt(((low.size - 1) * np.var(low, ddof=1) + (non.size - 1) * np.var(non, ddof=1)) / (low.size + non.size - 2))
    d = float(delta / pooled_sd) if pooled_sd > 0 else np.nan
    t_p = float(stats.ttest_ind(low, non, equal_var=False, nan_policy="omit").pvalue)
    mw = stats.mannwhitneyu(low, non, alternative="two-sided")
    auc_high_low = float(mw.statistic / (low.size * non.size))
    return {
        "low_mean": float(np.mean(low)),
        "non_low_mean": float(np.mean(non)),
        "low_sd": float(np.std(low, ddof=1)),
        "non_low_sd": float(np.std(non, ddof=1)),
        "delta_low_minus_nonlow": delta,
        "cohen_d": d,
        "auc_if_higher_predicts_low_smi": auc_high_low,
        "welch_p": t_p,
        "mannwhitney_p": float(mw.pvalue),
    }


def summarize_feature(name: str, values_by_cohort: dict[str, np.ndarray], datasets: dict[str, dict]) -> list[dict]:
    """한 특징에 대해 코호트별(및 pooled) feature_stats 결과를 한 행씩 모아 리스트로 반환."""
    rows = []
    for cohort, values in values_by_cohort.items():
        row = {"feature": name, "cohort": cohort, "n": int(values.size), "events": int(datasets[cohort]["y"].sum())}
        row.update(feature_stats(values, datasets[cohort]["y"]))
        rows.append(row)
    pooled_values = np.concatenate(list(values_by_cohort.values()))
    pooled_y = np.concatenate([d["y"] for d in datasets.values()])
    row = {"feature": name, "cohort": "pooled", "n": int(pooled_values.size), "events": int(pooled_y.sum())}
    row.update(feature_stats(pooled_values, pooled_y))
    rows.append(row)
    return rows


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: g1090과 sdata 양쪽에서 공통으로 "저근감소증군이 더
    낮은/더 높은" 부호를 보이는 AEC_128 위치 구간이 있는가? — 이후 여러 스크립트가 재사용하는
    기초 특징을 여기서 "발견"한다):

    1. g1090/sdata 각각 load_aec128로 정규화된 128포인트 곡선과 라벨을 로드.
    2. 각 코호트에서 위치별 "저근감소증 평균 - 비저근감소증 평균" 차이 곡선(diff)을 계산.
    3. 두 코호트 모두에서 diff가 같은 부호(둘 다 양수 or 둘 다 음수)인 위치들을 찾고,
       contiguous_segments로 길이 4 이상 연속 구간만 골라낸다 (common_low_higher / common_low_lower).
    4. 그 구간들 중 가장 길고 뚜렷한 "중간 저점(mid-low)" 구간과 "후반 고점(late-high)" 구간을
       하나씩 골라, 그 구간 평균으로 3개 특징(mid_trough_mean, late_excess_mean,
       late_minus_mid_contrast)을 정의한다 — 이 3개가 이후 다른 여러 스크립트에서 재사용되는
       "발견된" 핵심 특징이다.
    5. summarize_feature로 이 3개 특징 각각의 코호트별/풀링 통계(Cohen's d, AUC, p값)를 계산해 CSV로 저장.
    6. 두 코호트의 차이 곡선을 겹쳐 그리고, 발견된 두 구간을 음영으로 표시한 그래프를 PNG로 저장.
    7. 정규화 방식, 발견된 구간 위치, 추천 특징의 공식과 해석을 JSON으로 저장.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}

    diffs = {}
    point_rows = []
    for cohort, d in datasets.items():
        x = d["x"]
        y = d["y"]
        low_mean = np.nanmean(x[y], axis=0)
        non_mean = np.nanmean(x[~y], axis=0)
        diff = low_mean - non_mean
        diffs[cohort] = diff
        for j in range(128):
            point_rows.append(
                {
                    "cohort": cohort,
                    "point_1_to_128": j + 1,
                    "position_0_to_1": j / 127,
                    "low_mean": low_mean[j],
                    "non_low_mean": non_mean[j],
                    "diff_low_minus_nonlow": diff[j],
                }
            )

    common_positive = (diffs["g1090"] > 0) & (diffs["sdata"] > 0)
    common_negative = (diffs["g1090"] < 0) & (diffs["sdata"] < 0)
    pos_segments = contiguous_segments(common_positive, min_len=4)
    neg_segments = contiguous_segments(common_negative, min_len=4)

    segment_rows = []
    for sign_name, segments in [("common_low_higher", pos_segments), ("common_low_lower", neg_segments)]:
        for start, end in segments:
            segment_rows.append(
                {
                    "segment_type": sign_name,
                    "start_point_1_to_128": start + 1,
                    "end_point_1_to_128": end + 1,
                    "length": end - start + 1,
                    "start_position_0_to_1": start / 127,
                    "end_position_0_to_1": end / 127,
                    "g1090_mean_diff": float(np.mean(diffs["g1090"][start : end + 1])),
                    "sdata_mean_diff": float(np.mean(diffs["sdata"][start : end + 1])),
                    "pooled_abs_min_diff": float(min(abs(np.mean(diffs["g1090"][start : end + 1])), abs(np.mean(diffs["sdata"][start : end + 1])))),
                }
            )

    seg_df = pd.DataFrame(segment_rows).sort_values(["segment_type", "length"], ascending=[True, False])
    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(OUT_DIR / "aec128_pointwise_low_minus_nonlow.csv", index=False)
    seg_df.to_csv(OUT_DIR / "aec128_common_sign_segments.csv", index=False)

    # Longest common mid-low and late-high segments are the most interpretable stable shape signature.
    neg = seg_df[seg_df["segment_type"].eq("common_low_lower")].sort_values(["length", "pooled_abs_min_diff"], ascending=False).iloc[0]
    pos = seg_df[seg_df["segment_type"].eq("common_low_higher")].sort_values(["length", "pooled_abs_min_diff"], ascending=False).iloc[0]
    mid_idx = np.arange(int(neg["start_point_1_to_128"]) - 1, int(neg["end_point_1_to_128"]))
    late_idx = np.arange(int(pos["start_point_1_to_128"]) - 1, int(pos["end_point_1_to_128"]))

    values = {
        "aec128_mid_trough_mean": {cohort: d["x"][:, mid_idx].mean(axis=1) for cohort, d in datasets.items()},
        "aec128_late_excess_mean": {cohort: d["x"][:, late_idx].mean(axis=1) for cohort, d in datasets.items()},
        "aec128_late_minus_mid_contrast": {
            cohort: d["x"][:, late_idx].mean(axis=1) - d["x"][:, mid_idx].mean(axis=1) for cohort, d in datasets.items()
        },
    }

    feature_rows = []
    for name, vals in values.items():
        feature_rows.extend(summarize_feature(name, vals, datasets))
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(OUT_DIR / "aec128_common_shape_feature_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    xgrid = np.arange(1, 129)
    for cohort, color in [("g1090", "#4C78A8"), ("sdata", "#F58518")]:
        ax.plot(xgrid, diffs[cohort], lw=2.2, color=color, label=f"{cohort}: low - non-low")
    ax.axhline(0.0, color="#555555", lw=1.0, ls="--")
    ax.axvspan(mid_idx[0] + 1, mid_idx[-1] + 1, color="#2F6F73", alpha=0.14, label="Common mid-low segment")
    ax.axvspan(late_idx[0] + 1, late_idx[-1] + 1, color="#C84630", alpha=0.14, label="Common late-high segment")
    ax.set_xlabel("AEC_128 point index")
    ax.set_ylabel("Mean normalized AEC difference: low SMI - non-low SMI")
    ax.set_title("AEC_128 common low-SMI shape signature in both cohorts", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_common_shape_signature.png", dpi=200)
    plt.close(fig)

    summary = {
        "normalization": "For each patient, normalized_aec_j = raw_aec_128_j / mean(raw_aec_128_1..raw_aec_128_128).",
        "common_low_lower_segment": {
            "start_point_1_to_128": int(mid_idx[0] + 1),
            "end_point_1_to_128": int(mid_idx[-1] + 1),
            "definition": "mid_trough_mean = mean(normalized_aec_j over this segment)",
        },
        "common_low_higher_segment": {
            "start_point_1_to_128": int(late_idx[0] + 1),
            "end_point_1_to_128": int(late_idx[-1] + 1),
            "definition": "late_excess_mean = mean(normalized_aec_j over this segment)",
        },
        "recommended_feature": {
            "name": "aec128_late_minus_mid_contrast",
            "formula": "mean(normalized_aec_j in late-high segment) - mean(normalized_aec_j in mid-low segment)",
            "interpretation": "Higher values indicate a deeper mid-curve trough followed by a higher late-curve rebound, the common low-SMI pattern across g1090 and sdata.",
        },
    }
    pd.Series(summary).to_json(OUT_DIR / "aec128_common_shape_feature_definition.json", force_ascii=False, indent=2)

    print("Common sign segments")
    print(seg_df.to_string(index=False))
    print("\nFeature stats")
    print(feature_df.to_string(index=False))
    print("\nRecommended feature definition")
    print(summary)
    print(OUT_DIR / "aec128_common_shape_signature.png")


if __name__ == "__main__":
    main()
