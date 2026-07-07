from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_signal_audit"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_alternative_shape_features"
SEED = 20260629


def pseg(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """1-based 구간 [a, b]의 평균값(행마다)을 계산."""
    return x[:, a - 1 : b].mean(axis=1)


def dseg(d: np.ndarray, a: int, b: int) -> np.ndarray:
    """도함수 배열에서 1-based 구간 [a, b]의 평균값을 계산."""
    # Interval a means point a -> a+1. For 128 points, intervals are 1..127.
    return d[:, a - 1 : b].mean(axis=1)


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """분모가 0에 가까우면 부호를 살린 작은 값으로 대체해 0 나눗셈을 방지하는 나눗셈."""
    return num / np.where(np.abs(den) < eps, np.sign(den) * eps + (den == 0) * eps, den)


def first_crossing_down(x: np.ndarray, level: float = 1.0, start: int = 35, end: int = 80) -> np.ndarray:
    """구간 [start,end] 안에서 곡선이 level을 위→아래로 처음 통과하는 위치를 선형보간으로 찾음 (없으면 중앙값으로 대체)."""
    out = np.full(x.shape[0], np.nan)
    for i, row in enumerate(x):
        seg = row[start - 1 : end]
        above = seg[:-1] >= level
        below_next = seg[1:] < level
        idx = np.flatnonzero(above & below_next)
        if idx.size:
            j = idx[0] + start - 1
            y0, y1 = row[j], row[j + 1]
            frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
            out[i] = (j + 1) + frac
    med = np.nanmedian(out)
    out[~np.isfinite(out)] = med if np.isfinite(med) else float((start + end) / 2)
    return out


def first_crossing_up(x: np.ndarray, level: float = 1.0, start: int = 75, end: int = 128) -> np.ndarray:
    """구간 [start,end] 안에서 곡선이 level을 아래→위로 처음 통과하는 위치를 선형보간으로 찾음 (없으면 중앙값으로 대체)."""
    out = np.full(x.shape[0], np.nan)
    for i, row in enumerate(x):
        seg = row[start - 1 : end]
        below = seg[:-1] <= level
        above_next = seg[1:] > level
        idx = np.flatnonzero(below & above_next)
        if idx.size:
            j = idx[0] + start - 1
            y0, y1 = row[j], row[j + 1]
            frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
            out[i] = (j + 1) + frac
    med = np.nanmedian(out)
    out[~np.isfinite(out)] = med if np.isfinite(med) else float((start + end) / 2)
    return out


def count_runs(mask: np.ndarray) -> np.ndarray:
    """각 행에서 True가 연속되는 구간(run)이 몇 번 나타나는지 개수를 세어 반환."""
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        runs = 0
        in_run = False
        for v in row:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        out[i] = runs
    return out


def longest_run(mask: np.ndarray) -> np.ndarray:
    """각 행에서 True가 가장 길게 연속되는 구간의 길이를 반환."""
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        best = cur = 0
        for v in row:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[i] = best
    return out


def build_features(x: np.ndarray) -> pd.DataFrame:
    """구간 레벨/대비, 극값과 위치, level=1 교차 위치, 곡률·거칠기·평탄구간·부호전환 등 손으로 설계한
    대안적(=이전 스크립트들과 다른 방식의) AEC_128 모양 특징 수십 개를 계산."""
    x = np.asarray(x, dtype=float)
    logx = np.log(np.clip(x, 1e-6, None))
    d1 = np.diff(x, axis=1)
    dlog = np.diff(logx, axis=1)
    d2 = np.diff(d1, axis=1)

    feats: dict[str, np.ndarray] = {}

    # Level and area contrasts.
    early = pseg(x, 1, 32)
    early_mid = pseg(x, 1, 58)
    transition = pseg(x, 58, 74)
    trough_band = pseg(x, 70, 85)
    recovery = pseg(x, 81, 113)
    late = pseg(x, 91, 113)
    tail = pseg(x, 114, 128)
    tail2 = pseg(x, 120, 128)
    trough_min = np.min(x[:, 60 - 1 : 95], axis=1)
    trough_arg = np.argmin(x[:, 60 - 1 : 95], axis=1) + 60
    early_peak = np.max(x[:, 1 - 1 : 40], axis=1)
    early_peak_arg = np.argmax(x[:, 1 - 1 : 40], axis=1) + 1
    tail_peak = np.max(x[:, 91 - 1 : 128], axis=1)
    tail_peak_arg = np.argmax(x[:, 91 - 1 : 128], axis=1) + 91

    feats["level_tail_114_128"] = tail
    feats["level_tail_120_128"] = tail2
    feats["level_recovery_81_113"] = recovery
    feats["level_trough_70_85"] = trough_band
    feats["contrast_tail114_128_minus_early1_32"] = tail - early
    feats["contrast_tail120_128_minus_early1_32"] = tail2 - early
    feats["contrast_recovery81_113_minus_trough70_85"] = recovery - trough_band
    feats["contrast_late91_113_minus_trough70_85"] = late - trough_band
    feats["contrast_tail114_128_minus_trough70_85"] = tail - trough_band
    feats["contrast_tailpeak91_128_minus_troughmin60_95"] = tail_peak - trough_min
    feats["contrast_earlypeak1_40_minus_troughmin60_95"] = early_peak - trough_min
    feats["contrast_transition58_74_minus_early1_58"] = transition - early_mid
    feats["recovery_fraction_tail114_128"] = safe_div(tail - trough_min, early_peak - trough_min)
    feats["recovery_fraction_tail120_128"] = safe_div(tail2 - trough_min, early_peak - trough_min)
    feats["recovery_fraction_late91_113"] = safe_div(late - trough_min, early_peak - trough_min)
    feats["relative_rebound_tail_vs_trough"] = safe_div(tail - trough_band, early - trough_band)
    feats["relative_rebound_late_vs_trough"] = safe_div(late - trough_band, early - trough_band)

    # Extrema and positions.
    feats["trough_min_60_95"] = trough_min
    feats["trough_position_60_95"] = trough_arg.astype(float)
    feats["early_peak_1_40"] = early_peak
    feats["early_peak_position_1_40"] = early_peak_arg.astype(float)
    feats["tail_peak_91_128"] = tail_peak
    feats["tail_peak_position_91_128"] = tail_peak_arg.astype(float)
    feats["peak_distance_tail_minus_early"] = tail_peak_arg.astype(float) - early_peak_arg.astype(float)
    feats["trough_to_tail_peak_distance"] = tail_peak_arg.astype(float) - trough_arg.astype(float)

    # Zero crossing features.
    down = first_crossing_down(x, 1.0, 35, 80)
    up = first_crossing_up(x, 1.0, 75, 128)
    feats["crossing_down_1_at_35_80"] = down
    feats["crossing_up_1_at_75_128"] = up
    feats["below_one_duration_down_to_up"] = up - down

    # Curvature and recovery geometry, not simple mean slope.
    for a, b in [(50, 65), (58, 74), (65, 80), (75, 90), (81, 113), (91, 113), (114, 126)]:
        feats[f"curvature_mean_d2_{a}_{b}"] = d2[:, a - 1 : min(b, d2.shape[1])].mean(axis=1)
        feats[f"curvature_max_d2_{a}_{b}"] = d2[:, a - 1 : min(b, d2.shape[1])].max(axis=1)
        feats[f"curvature_min_d2_{a}_{b}"] = d2[:, a - 1 : min(b, d2.shape[1])].min(axis=1)
        feats[f"roughness_abs_d2_{a}_{b}"] = np.abs(d2[:, a - 1 : min(b, d2.shape[1])]).mean(axis=1)

    # Plateau and roughness by region. These are different from exact global flat counts.
    for a, b in [(1, 32), (33, 58), (58, 74), (75, 90), (81, 113), (114, 127), (1, 58), (81, 127)]:
        dd = d1[:, a - 1 : b]
        feats[f"roughness_abs_d1_{a}_{b}"] = np.abs(dd).mean(axis=1)
        feats[f"roughness_sd_d1_{a}_{b}"] = dd.std(axis=1)
        feats[f"near_flat_count_absdiff_lt_0p001_{a}_{b}"] = (np.abs(dd) < 0.001).sum(axis=1).astype(float)
        feats[f"near_flat_run_count_absdiff_lt_0p001_{a}_{b}"] = count_runs(np.abs(dd) < 0.001)
        feats[f"near_flat_longest_run_absdiff_lt_0p001_{a}_{b}"] = longest_run(np.abs(dd) < 0.001)
        signs = np.sign(dd)
        feats[f"slope_sign_change_count_{a}_{b}"] = np.sum(np.diff(signs, axis=1) != 0, axis=1).astype(float)

    # Log-space analogues for a few robust ratio-like quantities.
    log_early = pseg(logx, 1, 32)
    log_trough = pseg(logx, 70, 85)
    log_recovery = pseg(logx, 81, 113)
    log_late = pseg(logx, 91, 113)
    log_tail = pseg(logx, 114, 128)
    feats["log_contrast_recovery81_113_minus_trough70_85"] = log_recovery - log_trough
    feats["log_contrast_late91_113_minus_trough70_85"] = log_late - log_trough
    feats["log_contrast_tail114_128_minus_trough70_85"] = log_tail - log_trough
    feats["log_contrast_tail114_128_minus_early1_32"] = log_tail - log_early
    feats["log_tail_114_128"] = log_tail
    feats["log_roughness_abs_d1_81_113"] = np.abs(dlog[:, 81 - 1 : 113]).mean(axis=1)

    return pd.DataFrame(feats)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """값들의 평균에 대한 부트스트랩 95% 신뢰구간을 계산."""
    rng = np.random.default_rng(SEED)
    vals = np.asarray(values, dtype=float)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots[i] = np.mean(vals[idx])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarize_feature(cohort: str, feature: str, values: np.ndarray, y: np.ndarray, pairs: pd.DataFrame) -> dict:
    """한 특징에 대해 전체 표본 기준 통계(AUC·평균차·Mann-Whitney)와, 임상점수로 매칭된 쌍 기준
    대응비교 통계(대응차 평균·부트스트랩CI·대응t검정)를 함께 계산."""
    yb = y.astype(bool)
    low = values[yb]
    non = values[~yb]
    auc_high = roc_auc_score(y.astype(int), values)
    mw = stats.mannwhitneyu(low, non, alternative="two-sided")
    low_idx = pairs["low_index"].to_numpy(dtype=int)
    non_idx = pairs["matched_nonlow_index"].to_numpy(dtype=int)
    pdiff = values[low_idx] - values[non_idx]
    ci_low, ci_high = bootstrap_ci(pdiff)
    paired_t = stats.ttest_rel(values[low_idx], values[non_idx], nan_policy="omit")
    return {
        "cohort": cohort,
        "feature": feature,
        "n": int(len(values)),
        "events": int(np.sum(y)),
        "low_mean": float(np.mean(low)),
        "nonlow_mean": float(np.mean(non)),
        "diff_low_minus_nonlow": float(np.mean(low) - np.mean(non)),
        "auc_if_higher_predicts_low": float(auc_high),
        "auc_best_direction": float(max(auc_high, 1 - auc_high)),
        "mannwhitney_p": float(mw.pvalue),
        "matched_pairs_n": int(len(pdiff)),
        "matched_low_mean": float(np.mean(values[low_idx])),
        "matched_nonlow_mean": float(np.mean(values[non_idx])),
        "matched_diff_low_minus_nonlow": float(np.mean(pdiff)),
        "matched_diff_ci2.5": ci_low,
        "matched_diff_ci97.5": ci_high,
        "paired_t_p": float(paired_t.pvalue),
    }


def plot_top_features(patient_df: pd.DataFrame, selected: pd.DataFrame) -> None:
    """공통방향으로 선택된 상위 8개 특징에 대해, 코호트별 저근감소증/비저근감소증 분포를 바이올린플롯+산점도로 그려 PNG로 저장."""
    top = selected.head(8)["feature"].tolist()
    if not top:
        return
    fig, axes = plt.subplots(len(top), 2, figsize=(11.5, 2.2 * len(top)), squeeze=False)
    for r, feat in enumerate(top):
        for c, cohort in enumerate(["g1090", "sdata"]):
            ax = axes[r, c]
            sub = patient_df[patient_df["cohort"].eq(cohort)]
            vals0 = sub[sub["y_low_smi"].eq(0)][feat].to_numpy(float)
            vals1 = sub[sub["y_low_smi"].eq(1)][feat].to_numpy(float)
            parts = ax.violinplot([vals0, vals1], positions=[0, 1], widths=0.75, showmeans=True, showextrema=False)
            for body, color in zip(parts["bodies"], ["#4C78A8", "#C84630"]):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.24)
            parts["cmeans"].set_color("#222222")
            ax.scatter(np.zeros(len(vals0)), vals0, s=5, alpha=0.12, color="#4C78A8", linewidths=0)
            ax.scatter(np.ones(len(vals1)), vals1, s=7, alpha=0.18, color="#C84630", linewidths=0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["non-low", "low"])
            ax.set_title(f"{cohort}: {feat}", loc="left", fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alternative_shape_top_common_feature_distributions.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 지금까지와 전혀 다른 방식으로 손수 설계한 특징들
    (구간대비, 극값위치, level=1 교차점, 곡률/거칠기, 평탄구간 등)로도, 두 코호트 모두에서
    + 임상점수로 매칭된 쌍에서도 일관된 저근감소증 신호가 나오는가?):

    1. g1090/sdata 각각 build_features로 수십 개의 "대안적" 모양 특징을 계산하고, 이전 신호감사
       (run_aec128_signal_audit)에서 만든 임상점수 매칭쌍 CSV를 불러온다.
    2. 각 특징마다 summarize_feature로 코호트 내 전체표본 통계와 매칭쌍 대응비교 통계를 모두 계산해
       CSV로 저장.
    3. 두 코호트를 합친 pooled 데이터에서도 각 특징의 전체 AUC/평균차/p값을 계산.
    4. 한 특징이 "선택"되려면: (a) g1090과 sdata 전체표본에서 차이 방향이 같고, (b) 매칭쌍에서도
       두 코호트 방향이 같아야 한다 — 이 이중 조건을 만족하는 특징만 골라 판별력·매칭차이 크기로 정렬.
    5. 선택된 특징 목록을 CSV로 저장하고, 상위 8개의 코호트별 분포를 바이올린플롯으로 그려 저장.
    6. 최종 선택된 공통방향 특징 목록을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_tables = []
    summary_rows = []

    for cohort, path in FILES.items():
        d = load_aec128(path)
        x = d["x"].astype(float)
        y = d["y"].astype(int)
        features = build_features(x)
        features.insert(0, "y_low_smi", y)
        features.insert(0, "patient_index", np.arange(len(y)))
        features.insert(0, "cohort", cohort)
        pairs = pd.read_csv(BASE_DIR / f"{cohort}_clinical_matched_pairs.csv")
        pairs = pairs[pairs["within_caliper"]].copy()

        for feature in [c for c in features.columns if c not in {"cohort", "patient_index", "y_low_smi"}]:
            summary_rows.append(summarize_feature(cohort, feature, features[feature].to_numpy(float), y, pairs))
        patient_tables.append(features)

    patient_df = pd.concat(patient_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    patient_df.to_csv(OUT_DIR / "alternative_shape_patient_features.csv", index=False)
    summary_df.to_csv(OUT_DIR / "alternative_shape_feature_stats_by_cohort.csv", index=False)

    # Pooled stats, then require common direction across cohorts and in matched pairs.
    pooled_rows = []
    feature_cols = [c for c in patient_df.columns if c not in {"cohort", "patient_index", "y_low_smi"}]
    for feature in feature_cols:
        values = patient_df[feature].to_numpy(float)
        y = patient_df["y_low_smi"].to_numpy(int)
        low = values[y == 1]
        non = values[y == 0]
        auc_high = roc_auc_score(y, values)
        mw = stats.mannwhitneyu(low, non, alternative="two-sided")
        pooled_rows.append(
            {
                "feature": feature,
                "pooled_low_mean": float(np.mean(low)),
                "pooled_nonlow_mean": float(np.mean(non)),
                "pooled_diff_low_minus_nonlow": float(np.mean(low) - np.mean(non)),
                "pooled_auc_if_higher_predicts_low": float(auc_high),
                "pooled_auc_best_direction": float(max(auc_high, 1 - auc_high)),
                "pooled_mannwhitney_p": float(mw.pvalue),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows)

    wide_all = summary_df.pivot(index="feature", columns="cohort", values="diff_low_minus_nonlow")
    wide_pair = summary_df.pivot(index="feature", columns="cohort", values="matched_diff_low_minus_nonlow")
    selected_rows = []
    for _, row in pooled_df.iterrows():
        feat = row["feature"]
        if feat not in wide_all.index or feat not in wide_pair.index:
            continue
        g_all, s_all = float(wide_all.loc[feat, "g1090"]), float(wide_all.loc[feat, "sdata"])
        g_pair, s_pair = float(wide_pair.loc[feat, "g1090"]), float(wide_pair.loc[feat, "sdata"])
        same_all = np.sign(g_all) == np.sign(s_all) and g_all != 0 and s_all != 0
        same_pair = np.sign(g_pair) == np.sign(s_pair) and g_pair != 0 and s_pair != 0
        if same_all and same_pair:
            selected = row.to_dict()
            selected.update(
                {
                    "g1090_all_diff": g_all,
                    "sdata_all_diff": s_all,
                    "g1090_matched_diff": g_pair,
                    "sdata_matched_diff": s_pair,
                    "min_abs_all_diff": float(min(abs(g_all), abs(s_all))),
                    "min_abs_matched_diff": float(min(abs(g_pair), abs(s_pair))),
                    "g1090_auc_best": float(
                        summary_df[(summary_df["cohort"].eq("g1090")) & (summary_df["feature"].eq(feat))][
                            "auc_best_direction"
                        ].iloc[0]
                    ),
                    "sdata_auc_best": float(
                        summary_df[(summary_df["cohort"].eq("sdata")) & (summary_df["feature"].eq(feat))][
                            "auc_best_direction"
                        ].iloc[0]
                    ),
                }
            )
            selected_rows.append(selected)

    selected_df = pd.DataFrame(selected_rows)
    if not selected_df.empty:
        selected_df = selected_df.sort_values(
            ["pooled_auc_best_direction", "min_abs_matched_diff", "min_abs_all_diff"],
            ascending=[False, False, False],
        )
    selected_df.to_csv(OUT_DIR / "alternative_shape_common_direction_features.csv", index=False)
    plot_top_features(patient_df, selected_df)

    print("TOP COMMON DIRECTION ALTERNATIVE FEATURES")
    print(selected_df.head(25).to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
