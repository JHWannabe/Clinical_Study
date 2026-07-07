from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
)


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "aec_region_search_surrogate"
TARGET_OPS = [("S80", 0.80), ("S85", 0.85), ("S90", 0.90)]
COARSE_STEP = 8
FINE_STEP = 4
COARSE_LENGTHS = [16, 24, 32]
FINE_LENGTHS = [12, 16, 20, 24, 28, 32]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55, 0.70]


def d1(x: np.ndarray) -> np.ndarray:
    """각 행(환자)에 대해 1차 차분(기울기)을 계산. 맨 앞 값은 첫 차분값으로 채워 길이를 원본과 동일하게 유지."""
    v = np.diff(x, axis=1)
    return np.column_stack([v[:, :1], v])


def d2(x: np.ndarray) -> np.ndarray:
    """1차 차분(d1)의 차분을 계산해 2차 차분(곡률)을 구함. 맨 앞 값은 첫 값으로 채워 길이를 유지."""
    v = np.diff(d1(x), axis=1)
    return np.column_stack([v[:, :1], v])


def candidate_windows(step: int, lengths: list[int], lo: int = 1, hi: int = 128) -> list[tuple[int, int]]:
    """지정된 길이들과 간격(step)으로 [lo, hi] 범위 안에서 가능한 모든 (시작, 끝) 윈도우 좌표 후보를 생성해 중복 제거 후 정렬된 리스트로 반환."""
    out = []
    for length in lengths:
        for start in range(lo, hi - length + 2, step):
            out.append((start, start + length - 1))
    return sorted(set(out))


def window_features(norm: np.ndarray, windows: list[tuple[int, int]]) -> pd.DataFrame:
    """정규화된 AEC 곡선에서 각 윈도우 구간마다 레벨(평균/표준편차/최소/최대/끝점차/선형기울기)과 기울기·곡률 통계를 계산해 특징 DataFrame을 만듦."""
    slope = d1(norm)
    curv = d2(norm)
    rows: dict[str, np.ndarray] = {}
    grid = np.arange(norm.shape[1], dtype=float)
    for start, end in windows:
        sl = slice(start - 1, end)
        tag = f"{start:03d}_{end:03d}"
        block = norm[:, sl]
        sb = slope[:, sl]
        cb = curv[:, sl]
        rows[f"win_{tag}__level_mean"] = block.mean(axis=1)
        rows[f"win_{tag}__level_sd"] = block.std(axis=1)
        rows[f"win_{tag}__level_min"] = block.min(axis=1)
        rows[f"win_{tag}__level_max"] = block.max(axis=1)
        rows[f"win_{tag}__endpoint_delta"] = block[:, -1] - block[:, 0]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        rows[f"win_{tag}__linear_slope"] = ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        rows[f"win_{tag}__slope_mean"] = sb.mean(axis=1)
        rows[f"win_{tag}__slope_sd"] = sb.std(axis=1)
        rows[f"win_{tag}__abs_slope_mean"] = np.abs(sb).mean(axis=1)
        rows[f"win_{tag}__abs_slope_max"] = np.abs(sb).max(axis=1)
        rows[f"win_{tag}__curv_mean"] = cb.mean(axis=1)
        rows[f"win_{tag}__curv_sd"] = cb.std(axis=1)
        rows[f"win_{tag}__abs_curv_mean"] = np.abs(cb).mean(axis=1)
        rows[f"win_{tag}__abs_curv_max"] = np.abs(cb).max(axis=1)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def z_train_apply(xg: pd.DataFrame, xs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """내부(xg) 데이터의 중앙값으로 결측을 채우고 내부 데이터 기준 평균/표준편차로 z-표준화한 뒤, 같은 통계를 외부(xs) 데이터에도 적용."""
    names = list(xg.columns)
    a = xg.to_numpy(dtype=float)
    b = xs.to_numpy(dtype=float)
    med = np.nanmedian(a, axis=0)
    a = np.where(np.isfinite(a), a, med[None, :])
    b = np.where(np.isfinite(b), b, med[None, :])
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (a - mu) / sd, (b - mu) / sd, names


def parse_window(feature_name: str) -> tuple[int, int, str]:
    """"win_시작_끝__설명" 형식의 특징 이름 문자열을 파싱해 (시작, 끝, 설명자) 튜플로 분해."""
    parts = feature_name.split("__")
    win = parts[0].replace("win_", "")
    start, end = [int(x) for x in win.split("_")]
    return start, end, parts[1]


def evaluate_feature(
    name: str,
    z_g: np.ndarray,
    z_s: np.ndarray,
    sign: int,
    width: float,
    lam: float,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """부호가 적용된 단일 특징 하나를 대리(surrogate) 신호로 써서 내부/외부 데이터셋의 각 목표 운영점에서 단일-특징 de-escalation 게이트를 만들고 성능 지표를 계산."""
    rows = []
    feat_g = sign * z_g
    feat_s = sign * z_s
    for dataset, d, clinical_z, feat_z in [
        ("g1090_internal", g, c_g, feat_g),
        ("sdata_external", s, c_s, feat_s),
    ]:
        for op, _ in TARGET_OPS:
            th = float(thresholds[op])
            deesc = make_single_deesc(clinical_z, feat_z, th, width, lam)
            rows.append(
                deesc_metric_row(
                    dataset,
                    f"{name}__sign{sign:+d}__w{width:.2f}__lam{lam:.2f}",
                    name,
                    op,
                    d["y"].astype(int),
                    clinical_z >= th,
                    deesc,
                )
            )
    return pd.DataFrame(rows)


def pass_internal(detail: pd.DataFrame) -> bool:
    """내부(g1090) 데이터셋이 최소 de-escalation 건수, 민감도 손실 유의성, 특이도/정확도 증가 조건을 모두 만족하는지 판정."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    return bool(
        (gi["deesc_n"] >= 10).all()
        and (gi["sensitivity_loss_p_exact"].fillna(1.0) >= 0.05).all()
        and (gi["specificity_gain"] > 0).all()
        and (gi["accuracy_delta"] > 0).all()
    )


def pass_external(detail: pd.DataFrame) -> bool:
    """외부(sdata) 데이터셋이 최소 de-escalation 건수, 민감도 손실 유의성, 특이도/정확도 증가 조건을 모두 만족하는지 판정."""
    se = detail[detail["dataset"].eq("sdata_external")]
    return bool(
        (se["deesc_n"] >= 10).all()
        and (se["sensitivity_loss_p_exact"].fillna(1.0) >= 0.05).all()
        and (se["specificity_gain"] > 0).all()
        and (se["accuracy_delta"] > 0).all()
    )


def summary_row(stage: str, name: str, sign: int, width: float, lam: float, detail: pd.DataFrame) -> dict:
    """한 특징(윈도우+부호+width+lambda 조합)의 세부 지표 DataFrame을 요약해, 통과 여부와 내부/외부 평균·최소 성능, 선택 점수를 담은 한 행짜리 딕셔너리로 만듦."""
    start, end, descriptor = parse_window(name)
    gi = detail[detail["dataset"].eq("g1090_internal")]
    se = detail[detail["dataset"].eq("sdata_external")]
    row = {
        "stage": stage,
        "feature": name,
        "window_start": start,
        "window_end": end,
        "window_len": end - start + 1,
        "descriptor": descriptor,
        "sign": sign,
        "width": width,
        "lambda": lam,
        "internal_pass": pass_internal(detail),
        "external_pass": pass_external(detail),
        "internal_mean_accuracy": float(gi["post_accuracy"].mean()),
        "internal_mean_accuracy_gain": float(gi["accuracy_delta"].mean()),
        "internal_min_accuracy_gain": float(gi["accuracy_delta"].min()),
        "internal_mean_specificity_gain": float(gi["specificity_gain"].mean()),
        "internal_min_specificity_gain": float(gi["specificity_gain"].min()),
        "internal_max_sensitivity_loss": float(gi["sensitivity_loss"].max()),
        "internal_min_sensitivity_loss_p": float(gi["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "external_mean_accuracy": float(se["post_accuracy"].mean()),
        "external_mean_accuracy_gain": float(se["accuracy_delta"].mean()),
        "external_min_accuracy_gain": float(se["accuracy_delta"].min()),
        "external_mean_specificity_gain": float(se["specificity_gain"].mean()),
        "external_min_specificity_gain": float(se["specificity_gain"].min()),
        "external_max_sensitivity_loss": float(se["sensitivity_loss"].max()),
        "external_min_sensitivity_loss_p": float(se["sensitivity_loss_p_exact"].fillna(1.0).min()),
    }
    if row["internal_pass"]:
        row["internal_selection_score"] = (
            row["internal_mean_accuracy"]
            + 0.45 * row["internal_min_accuracy_gain"]
            + 0.20 * row["internal_min_specificity_gain"]
            - 0.20 * row["internal_max_sensitivity_loss"]
        )
    else:
        row["internal_selection_score"] = -1e6
    return row


def scan_stage(
    stage: str,
    windows: list[tuple[int, int]],
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """주어진 윈도우 목록에서 특징을 만들고 표준화한 뒤, 특징×부호×width×lambda 전 조합을 평가해 요약 결과와 내부 통과 규칙들의 상세표를 반환."""
    xg_df = window_features(g["norm"], windows)
    xs_df = window_features(s["norm"], windows)
    xg, xs, names = z_train_apply(xg_df, xs_df)
    rows = []
    details: dict[str, pd.DataFrame] = {}
    total = len(names) * 2 * len(WIDTHS) * len(LAMBDAS)
    done = 0
    for j, name in enumerate(names):
        for sign in [-1, 1]:
            for width in WIDTHS:
                for lam in LAMBDAS:
                    detail = evaluate_feature(name, xg[:, j], xs[:, j], sign, width, lam, g, s, c_g, c_s, thresholds)
                    row = summary_row(stage, name, sign, width, lam, detail)
                    rows.append(row)
                    if row["internal_pass"]:
                        key = f"{name}__sign{sign:+d}__w{width:.2f}__lam{lam:.2f}"
                        details[key] = detail
                    done += 1
        if (j + 1) % 25 == 0 or j + 1 == len(names):
            print(f"{stage}: features {j + 1}/{len(names)}; eval {done}/{total}", flush=True)
    return pd.DataFrame(rows), details


def fine_windows_from_top(coarse: pd.DataFrame, flank: int = 16) -> list[tuple[int, int]]:
    """거친(coarse) 탐색에서 내부 통과(또는 없으면 점수 상위) 상위 30개 윈도우 주변에 flank만큼 여유를 두고, 세밀한(fine) 재탐색용 윈도우 후보를 생성."""
    top = coarse[coarse["internal_pass"]].sort_values("internal_selection_score", ascending=False).head(30)
    if top.empty:
        top = coarse.sort_values("internal_selection_score", ascending=False).head(30)
    windows: set[tuple[int, int]] = set()
    for r in top.itertuples():
        lo = max(1, int(r.window_start) - flank)
        hi = min(128, int(r.window_end) + flank)
        windows.update(candidate_windows(FINE_STEP, FINE_LENGTHS, lo, hi))
    return sorted(windows)


def plot_top(df: pd.DataFrame, out_path: Path) -> None:
    """내부 통과(또는 없으면 점수 상위) 상위 40개 특징에 대해, 내부 평균 정확도 증가(막대)와 외부 평균 정확도 증가(점)를 가로 막대그래프로 비교해 저장."""
    sub = df[df["internal_pass"]].sort_values("internal_selection_score", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    if sub.empty:
        sub = df.sort_values("internal_selection_score", ascending=False).head(40)
    y = np.arange(len(sub))
    labels = [f"{r.window_start}-{r.window_end} {r.descriptor}" for r in sub.itertuples()]
    ax.barh(y, sub["internal_mean_accuracy_gain"] * 100, color="#2f6b9a", alpha=0.85, label="Internal mean acc gain")
    ax.scatter(sub["external_mean_accuracy_gain"] * 100, y, color="#c54e2c", label="External mean acc gain", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("percentage points")
    ax.set_title("Top region-surrogate features", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """전체 128포인트 AEC 곡선을 거친 격자로 스캔한 뒤, 내부 통과 상위 구간 주변을 세밀하게 재탐색하여 단일-특징 대리(surrogate)
    de-escalation 규칙 후보들을 랭킹하고, 통과 후보 목록/그래프/요약 JSON을 저장. 결과는 최종 CNN 모델이 아니라 유망 구간을 고르기 위한 빠른 사전 탐색."""
    # 결과 출력 폴더 준비 (outputs/0701/aec_region_search_surrogate).
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 내부(g1090) / 외부(sdata) 데이터셋 로드. 각각 dict로 AEC 곡선("norm"), 라벨("y") 등을 담고 있음.
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")

    # 임상 점수 기반 baseline 계산: c_g/c_s는 내부/외부 각 환자의 임상 점수(z-표준화),
    # thresholds는 S80/S85/S90 목표 민감도에 대응하는 임계값. 이 임계값이 이후 단일-특징
    # de-escalation 게이트(make_single_deesc)의 기준선으로 쓰임. oof/ext 예측값 자체는 여기서는 안 씀.
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)

    # --- 1단계: Coarse(거친) 스캔 ---
    # step=8 간격, 길이 16/24/32 조합으로 곡선 전체(1~128)에서 가능한 모든 윈도우 좌표를 생성.
    # Coarse scan covers the whole 128-point AEC curve. Fine scan is local around internally promising windows.
    coarse_windows = candidate_windows(COARSE_STEP, COARSE_LENGTHS, 1, 128)
    print(f"coarse windows={len(coarse_windows)}", flush=True)
    # 각 윈도우에서 특징을 뽑고, 특징×부호×width×lambda 전 조합을 평가해 요약표(coarse)와
    # 내부 통과 규칙들의 상세표(coarse_details)를 얻음. coarse_details는 main()에서는 더 쓰이지 않음(부산물).
    coarse, coarse_details = scan_stage("coarse8", coarse_windows, g, s, c_g, c_s, thresholds)
    # 내부 통과 여부(internal_pass) 우선, 그 다음 선택 점수(internal_selection_score) 내림차순으로 정렬.
    coarse = coarse.sort_values(["internal_pass", "internal_selection_score"], ascending=[False, False])
    coarse.to_csv(OUT_DIR / "region_surrogate_coarse8_all.csv", index=False)

    # --- 2단계: Fine(세밀) 재탐색 ---
    # coarse 결과 중 내부 통과 상위 30개(없으면 점수 상위 30개) 윈도우 주변 ±flank(16)를 확장해,
    # step=4, 길이 12~32(4단위)로 더 촘촘한 후보 윈도우를 다시 생성.
    fine_windows = fine_windows_from_top(coarse)
    print(f"fine windows={len(fine_windows)}", flush=True)
    # 좁혀진 후보군에 대해 동일한 평가 파이프라인(scan_stage)을 재실행.
    fine, fine_details = scan_stage("fine4", fine_windows, g, s, c_g, c_s, thresholds)

    # --- 3단계: coarse+fine 결과 통합 및 랭킹 ---
    # 두 단계 결과를 합치고, 동일한 (feature, sign, width, lambda) 조합 중복은 제거(fine 재탐색 시
    # coarse와 겹치는 윈도우가 다시 나올 수 있음). 이후 다시 internal_pass 우선 + 점수 내림차순 정렬.
    all_df = pd.concat([coarse, fine], ignore_index=True)
    all_df = all_df.drop_duplicates(["feature", "sign", "width", "lambda"]).sort_values(
        ["internal_pass", "internal_selection_score"], ascending=[False, False]
    )
    all_df.to_csv(OUT_DIR / "region_surrogate_all_ranked.csv", index=False)

    # 내부 데이터에서 안전 기준(최소 건수/민감도 손실 없음/특이도·정확도 증가)을 통과한 후보만 추림.
    # 이 목록이 "guided CNN 지역(region) 후보"를 사람이 고를 때 참고하는 1차 필터링 결과.
    passing = all_df[all_df["internal_pass"]].copy()
    passing.to_csv(OUT_DIR / "region_surrogate_internal_passing_ranked.csv", index=False)

    # 내부뿐 아니라 외부(sdata)에서도 같은 안전 기준을 통과한 후보만 다시 추려, 외부 성능
    # (외부 평균 정확도 → 외부 평균 정확도 증가 → 내부 선택 점수) 순으로 정렬. 가장 신뢰도 높은 후보군.
    both = passing[passing["external_pass"]].sort_values(
        ["external_mean_accuracy", "external_mean_accuracy_gain", "internal_selection_score"], ascending=False
    )
    both.to_csv(OUT_DIR / "region_surrogate_internal_external_passing_ranked.csv", index=False)

    # 내부 통과(또는 상위) 40개 후보의 내부/외부 정확도 증가를 막대+점 그래프로 시각화해 저장.
    plot_top(all_df, OUT_DIR / "region_surrogate_top.png")

    # 탐색 설정(격자 크기 등)과 상위 10개 후보(내부 통과 / 내부+외부 통과)를 요약 JSON으로 저장.
    # purpose 필드에 명시된 대로, 이 결과는 최종 모델이 아니라 이후 guided CNN에 쓸 구간을 고르기 위한 사전 탐색용.
    with (OUT_DIR / "region_surrogate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "purpose": "Fast region search surrogate for choosing guided CNN regions. This is not the final CNN model.",
                "grid": {
                    "coarse": {"step": COARSE_STEP, "lengths": COARSE_LENGTHS, "range": "1-128"},
                    "fine": {"step": FINE_STEP, "lengths": FINE_LENGTHS, "range": "around top internal coarse windows"},
                },
                "target_ops": [op for op, _ in TARGET_OPS],
                "n_coarse_windows": len(coarse_windows),
                "n_fine_windows": len(fine_windows),
                "n_candidates": int(len(all_df)),
                "n_internal_passing": int(len(passing)),
                "n_internal_external_passing": int(len(both)),
                "top_internal": passing.head(10).to_dict(orient="records"),
                "top_internal_external": both.head(10).to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    # 콘솔에 요약 출력할 열 목록 (원본 데이터프레임의 일부 컬럼만 선택).
    show = [
        "stage",
        "window_start",
        "window_end",
        "window_len",
        "descriptor",
        "sign",
        "width",
        "lambda",
        "internal_mean_accuracy_gain",
        "internal_min_accuracy_gain",
        "internal_mean_specificity_gain",
        "internal_max_sensitivity_loss",
        "internal_min_sensitivity_loss_p",
        "external_pass",
        "external_mean_accuracy_gain",
        "external_mean_specificity_gain",
        "external_max_sensitivity_loss",
        "external_min_sensitivity_loss_p",
    ]
    # 내부 통과 후보 상위 20개, 내부+외부 모두 통과 후보 상위 20개를 각각 콘솔에 표로 출력.
    print("\nTOP INTERNAL")
    print(passing[show].head(20).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTOP INTERNAL+EXTERNAL")
    print(both[show].head(20).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# AEC 곡선 전체를 거친 격자로 스캔하고 유망 구간 주변을 세밀히 재탐색해, guided CNN에 쓸 지역(region) 후보를 고르기 위한
# 단일-특징 de-escalation 대리 규칙들을 평가·랭킹하여 outputs 폴더에 결과를 저장하는 파이프라인을 실행.
if __name__ == "__main__":
    main()
