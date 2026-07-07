from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_2of3_s80_90_visualization"
CONFIG_PATH = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics" / "selected_individual_feature_configs.csv"
)
FEATURES = ["norm_slope_013_016_sd", "norm_curv_007_010_min", "dct_log_17"]
OPS = [
    ("S80", 0.80),
    ("S82.5", 0.825),
    ("S85", 0.85),
    ("S87.5", 0.875),
    ("S90", 0.90),
]


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측 양성/음성과 실제 결과로부터 tp/fp/fn/tn, 정확도, 민감도, 특이도, 균형정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (sens + spec),
    }


def exact_pair_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 2/3 투표 게이트 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    clinical_correct = clinical_pos == yy
    final_correct = final_pos == yy
    acc_gain = int(np.sum(~clinical_correct & final_correct))
    acc_loss = int(np.sum(clinical_correct & ~final_correct))
    return {
        "sensitivity_loss_p_exact": exact_pair_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_pair_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_pair_p(acc_gain, acc_loss),
    }


def fisher_deesc_p(y: np.ndarray, final_pos: np.ndarray, deesc: np.ndarray) -> float:
    """최종 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[final_pos] == 1))
    b = int(np.sum(y[final_pos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def bootstrap_delta_ba_p(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray, seed: int) -> float:
    """부트스트랩 재표본추출로 균형정확도 변화량(게이트 후 - 임상단독)의 분포를 만들고, 그 값이 0 이하일 확률(부트스트랩 p값)을 계산."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(5000):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(counts(y[idx], final_pos[idx])["balanced_accuracy"] - counts(y[idx], clinical_pos[idx])["balanced_accuracy"])
    vals = np.asarray(vals)
    return float((np.sum(vals <= 0.0) + 1) / (len(vals) + 1)) if len(vals) else np.nan


def p_fmt(x: float) -> str:
    """p값을 소수점 4자리(또는 매우 작으면 지수표기)로 사람이 읽기 쉽게 문자열로 변환."""
    if pd.isna(x):
        return "NA"
    if x < 1e-4:
        return f"{x:.1e}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def feature_signals(
    c: np.ndarray,
    x: np.ndarray,
    threshold: float,
    feature_info: list[tuple[str, int, float, float]],
) -> dict[str, np.ndarray]:
    """임상양성군 내에서 각 특징별 가우시안 게이트를 적용해, 그 특징이 "하향조정 신호"를 내는 표본을 특징이름별 불리언 배열로 반환."""
    cpos = c >= threshold
    signals = {}
    for label, idx, width, lam in feature_info:
        boundary = np.exp(-0.5 * ((c - threshold) / width) ** 2)
        gate = c + lam * boundary * x[:, idx]
        signals[label] = cpos & (gate < threshold)
    return signals


def metrics_for_dataset(
    dataset: str,
    y: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    thresholds: dict[str, float],
    feature_info: list[tuple[str, int, float, float]],
) -> pd.DataFrame:
    """한 데이터셋(Gangnam 또는 Sinchon)에서 5개 운영점(S80~S90) 각각에 대해, 3개 특징 중 2개 이상이 하향조정 신호를 낸 표본만 실제로 하향조정하는 "2-of-3 합의규칙"의 성능을 계산."""
    rows = []
    for i, (op, target) in enumerate(OPS):
        th = thresholds[op]
        clinical_pos = c >= th
        signals = feature_signals(c, x, th, feature_info)
        votes = np.column_stack([signals[f] for f in FEATURES]).sum(axis=1)
        deesc = clinical_pos & (votes >= 2)
        final_pos = clinical_pos & ~deesc
        base = counts(y, clinical_pos)
        post = counts(y, final_pos)
        rows.append(
            {
                "dataset": dataset,
                "operating_point": op,
                "target_training_sensitivity": target,
                "clinical_threshold_z": th,
                "clinical_balanced_accuracy": base["balanced_accuracy"],
                "post_balanced_accuracy": post["balanced_accuracy"],
                "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
                "balanced_accuracy_delta_p_bootstrap": bootstrap_delta_ba_p(y, clinical_pos, final_pos, 20260630 + i + (0 if dataset.startswith("Gangnam") else 100)),
                "clinical_accuracy": base["accuracy"],
                "post_accuracy": post["accuracy"],
                "delta_accuracy": post["accuracy"] - base["accuracy"],
                "clinical_sensitivity": base["sensitivity"],
                "post_sensitivity": post["sensitivity"],
                "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
                "clinical_specificity": base["specificity"],
                "post_specificity": post["specificity"],
                "specificity_gain": post["specificity"] - base["specificity"],
                "deesc_n": int(deesc.sum()),
                "deesc_events": int(y[deesc].sum()),
                "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
                "deesc_event_fisher_p": fisher_deesc_p(y, final_pos, deesc),
                **paired_pvalues(y, clinical_pos, final_pos),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_meaning(
    g: dict,
    s: dict,
    feature_z: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
    c_by: dict[str, np.ndarray],
    x_by: dict[str, np.ndarray],
    feature_info: list[tuple[str, int, float, float]],
    out_path: Path,
) -> None:
    """2x2 패널 그림으로 (1) 저근감소증군 vs 비저근감소증군의 정규화 AEC 평균곡선과 관심 구간, (2) dct_log_17 기저함수 모양, (3) 3개 특징의 정규화 z-score 분포 박스플롯, (4) S85 운영점에서 투표수별 저근감소증 발생률 막대그래프를 그려 PNG로 저장."""
    x_axis = np.arange(1, 129)
    pooled_norm = np.vstack([g["norm"], s["norm"]])
    pooled_y = np.concatenate([g["y"], s["y"]]).astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0))
    ax = axes[0, 0]
    ax.plot(x_axis, pooled_norm[~pooled_y].mean(axis=0), color="#4C78A8", lw=2.2, label="Non-low SMI mean")
    ax.plot(x_axis, pooled_norm[pooled_y].mean(axis=0), color="#D95F02", lw=2.2, label="Low SMI mean")
    ax.axvspan(7, 12, color="#E45756", alpha=0.16, label="curv 007-010")
    ax.axvspan(13, 17, color="#54A24B", alpha=0.18, label="slope 013-016")
    ax.set_title("Where the local AEC features look", loc="left", fontweight="bold")
    ax.set_xlabel("AEC point index")
    ax.set_ylabel("Patient-normalized AEC")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    n = np.arange(128)
    basis17 = np.cos(np.pi * (n + 0.5) * 17 / 128)
    ax.plot(x_axis, basis17, color="#7A5195", lw=2.0)
    ax.axhline(0, color="#555555", lw=1)
    ax.set_title("dct_log_17 basis pattern", loc="left", fontweight="bold")
    ax.set_xlabel("AEC point index")
    ax.set_ylabel("DCT basis value")
    ax.grid(alpha=0.22)

    ax = axes[1, 0]
    labels = []
    data = []
    positions = []
    pos = 1
    colors = []
    short = {
        "norm_slope_013_016_sd": "slope\n013-016\nsd",
        "norm_curv_007_010_min": "curv\n007-010\nmin",
        "dct_log_17": "DCT\nlog 17",
    }
    for feat in FEATURES:
        for low in [False, True]:
            vals = np.concatenate([feature_z["Gangnam internal OOF"][feat], feature_z["Sinchon external"][feat]])
            yy = np.concatenate([g["y"], s["y"]]).astype(bool)
            data.append(vals[yy == low])
            positions.append(pos)
            labels.append(f"{short[feat]}\n{'low' if low else 'non-low'}")
            colors.append("#D95F02" if low else "#4C78A8")
            pos += 1
        pos += 0.7
    bp = ax.boxplot(data, positions=positions, widths=0.56, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Oriented feature z-score")
    ax.set_title("Feature distributions after Gangnam standardization", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    op = "S85"
    bars = []
    xt = []
    bar_labels = []
    xpos = 0
    for dname, d in [("Gangnam", g), ("Sinchon", s)]:
        key = "Gangnam internal OOF" if dname == "Gangnam" else "Sinchon external"
        c = c_by[key]
        x = x_by[key]
        th = thresholds[op]
        cpos = c >= th
        signals_by_feature = feature_signals(c, x, th, feature_info)
        signals = [signals_by_feature[feat] for feat in FEATURES]
        votes = np.column_stack(signals).sum(axis=1)
        for v in [0, 1, 2, 3]:
            m = cpos & (votes == v)
            rate = float(d["y"][m].mean()) if m.any() else np.nan
            bars.append(rate * 100 if np.isfinite(rate) else np.nan)
            xt.append(xpos)
            bar_labels.append(f"{dname}\n{v} votes\nn={int(m.sum())}")
            xpos += 1
        xpos += 0.6
    ax.bar(xt, bars, color=["#B9C4D0", "#8FB4DC", "#F2A65A", "#D95F02"] * 2, width=0.76)
    ax.set_xticks(xt)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel("Low SMI rate (%)")
    ax.set_title("Clinical-positive S85: event rate by low-risk feature votes", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Meaning of the 2-of-3 AEC consensus features", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_operating_points(metrics: pd.DataFrame, out_path: Path) -> None:
    """Gangnam(internal OOF)과 Sinchon(external) 두 데이터셋에 대해, S80~S90 운영점 전체에서 특이도이득/민감도손실 추이와 하향조정군 사건비율 추이를 나란히 그려 PNG로 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7))
    xs = np.array([80, 82.5, 85, 87.5, 90])
    for dataset, color in [("Gangnam internal OOF", "#4C78A8"), ("Sinchon external", "#D95F02")]:
        sub = metrics[metrics["dataset"].eq(dataset)].copy()
        sub["x"] = sub["operating_point"].str.replace("S", "", regex=False).astype(float)
        sub = sub.sort_values("x")
        axes[0].plot(sub["x"], sub["specificity_gain"] * 100, marker="o", color=color, lw=2.2, label=f"{dataset} specificity gain")
        axes[0].plot(sub["x"], sub["sensitivity_loss"] * 100, marker="s", color=color, lw=1.8, ls="--", label=f"{dataset} sensitivity loss")
        axes[1].plot(sub["x"], sub["deesc_event_rate"] * 100, marker="o", color=color, lw=2.2, label=dataset)
    axes[0].axhline(0, color="#555555", lw=1)
    axes[0].set_xticks(xs)
    axes[0].set_xlabel("Clinical target sensitivity (%)")
    axes[0].set_ylabel("Change after AEC gate (%p)")
    axes[0].set_title("Tradeoff across S80-S90", loc="left", fontweight="bold")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("Clinical target sensitivity (%)")
    axes[1].set_ylabel("Event rate among de-escalated (%)")
    axes[1].set_title("De-escalated group risk", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 단일 특징 하나에 의존하는 대신, 3개의 서로 다른 성격의
    특징(초반 기울기 변동성, 초반 곡률 최소값, 중간주파수 DCT 성분) 중 2개 이상이 동시에
    "하향조정" 신호를 낼 때만 실제로 하향조정하는 "2-of-3 합의규칙"이 S80~S90 전 구간에서
    일관되게 잘 작동하는지, 그리고 그 의미를 시각적으로 이해하기 쉽게 보여주는 목적):

    1. g1090(Gangnam)/sdata(Sinchon)를 로드하고 임상점수를 계산한 뒤, g1090 기준 5개 운영점
       (S80~S90, 2.5%p 간격)의 임상 임계값을 threshold_for_min_sensitivity로 구한다.
    2. 후보 특징뱅크를 표준화·방향고정하고, 3개 특징(FEATURES)에 대해 aec_individual_feature_full
       _metrics가 저장해둔 폭·람다 설정을 읽어온다.
    3. 두 데이터셋 각각에서 metrics_for_dataset으로 2-of-3 합의규칙의 성능(임상단독 대비 특이도
       이득/민감도손실/균형정확도변화 및 각종 p값)을 5개 운영점에 걸쳐 계산해 CSV로 저장.
    4. 3개 특징에 대한 설명(신호 정의, 대략적인 곡선구간, 시각적 의미)을 표로 정리해 CSV로 저장.
    5. plot_feature_meaning으로 특징들이 곡선의 어느 부분을 보는지, 분포가 어떻게 갈리는지,
       투표수별 발생률이 어떻게 다른지를 종합한 그림을 그리고, plot_operating_points로 S80~S90
       전체에 걸친 트레이드오프 추이 그래프를 그려 각각 PNG로 저장.
    6. 전체 성능표와 저장된 그림 경로를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    g["y"] = g["y"].astype(int)
    s["y"] = s["y"].astype(int)
    c_g, c_s, _ = clinical_scores(g, s)
    thresholds = {label: threshold_for_min_sensitivity(g["y"], c_g, target) for label, target in OPS}

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]
    name_to_idx = {name: i for i, name in enumerate(names)}

    config = pd.read_csv(CONFIG_PATH)
    feature_info = []
    feature_z = {"Gangnam internal OOF": {}, "Sinchon external": {}}
    for label in FEATURES:
        r = config[config["label"].eq(label)].iloc[0]
        idx = name_to_idx[str(r["feature"])]
        feature_info.append((label, idx, float(r["width"]), float(r["lambda"])))
        feature_z["Gangnam internal OOF"][label] = xg[:, idx]
        feature_z["Sinchon external"][label] = xs[:, idx]

    metrics = pd.concat(
        [
            metrics_for_dataset("Gangnam internal OOF", g["y"], c_g, xg, thresholds, feature_info),
            metrics_for_dataset("Sinchon external", s["y"], c_s, xs, thresholds, feature_info),
        ],
        ignore_index=True,
    )
    metrics.to_csv(OUT_DIR / "metrics_2of3_s80_s82p5_s85_s87p5_s90.csv", index=False)

    feature_defs = pd.DataFrame(
        [
            {
                "feature": "norm_slope_013_016_sd",
                "signal": "SD of first differences of normalized AEC over derivative positions 13-16",
                "approx_curve_region": "AEC point transitions 13->14 through 16->17",
                "visual_meaning": "early local roughness / uneven slope",
            },
            {
                "feature": "norm_curv_007_010_min",
                "signal": "minimum second difference of normalized AEC over curvature positions 7-10",
                "approx_curve_region": "early bend centered around AEC points 8-11",
                "visual_meaning": "sharpest early concavity / local bend",
            },
            {
                "feature": "dct_log_17",
                "signal": "DCT-II coefficient 17 of log normalized AEC over all 128 points",
                "approx_curve_region": "global 128-point curve",
                "visual_meaning": "medium-high-frequency ripple / oscillatory shape",
            },
        ]
    )
    feature_defs.to_csv(OUT_DIR / "feature_definitions.csv", index=False)

    plot_feature_meaning(
        g,
        s,
        feature_z,
        thresholds,
        {"Gangnam internal OOF": c_g, "Sinchon external": c_s},
        {"Gangnam internal OOF": xg, "Sinchon external": xs},
        feature_info,
        OUT_DIR / "feature_meaning_2of3_consensus.png",
    )
    plot_operating_points(metrics, OUT_DIR / "operating_points_2of3_s80_to_s90.png")
    print(metrics.to_string(index=False))
    print(OUT_DIR / "feature_meaning_2of3_consensus.png")
    print(OUT_DIR / "operating_points_2of3_s80_to_s90.png")


if __name__ == "__main__":
    main()
