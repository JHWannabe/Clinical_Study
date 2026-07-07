from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec_offset_score import clinical_raw  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_signal_audit"
SEED = 20260629
N_PERM = 3000
T_THRESHOLD = 2.0


def smi_from_meta(meta: pd.DataFrame) -> np.ndarray:
    """메타데이터에 SMI 컬럼이 있고 대부분 유효하면 그대로 쓰고, 아니면 TAMA/키^2로 SMI를 직접 계산."""
    if "SMI" in meta.columns:
        smi = pd.to_numeric(meta["SMI"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(smi).mean() > 0.95:
            return smi
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    return tama / (height_m**2)


def clinical_logit_pipeline() -> Pipeline:
    """결측대체→표준화→로지스틱 회귀(정규화 거의 없음)로 이어지는 임상 모델 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def smi_pipeline() -> Pipeline:
    """결측대체→표준화→선형회귀로 이어지는, 임상변수로 SMI(연속값)를 예측하는 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lin", LinearRegression()),
        ]
    )


def crossfit_clinical_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold 교차검증으로 임상 로짓 모델을 학습해 train 전체에 대한 out-of-fold 점수를 만듦."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = np.zeros(len(y), dtype=float)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        model = clinical_logit_pipeline()
        model.fit(x[tr_idx], y[tr_idx])
        out[va_idx] = model.decision_function(x[va_idx])
    return out


def crossfit_smi_residual(x: np.ndarray, smi: np.ndarray) -> np.ndarray:
    """SMI를 임상변수로 예측하는 모델을 SMI 5분위 층화 5-fold로 학습해, train 전체의 out-of-fold 잔차(실제-예측)를 계산."""
    # Stratify by low-SMI-like quantiles so residuals are not dominated by one fold's SMI range.
    bins = pd.qcut(pd.Series(smi).rank(method="first"), q=5, labels=False).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 11)
    pred = np.zeros(len(smi), dtype=float)
    for tr_idx, va_idx in skf.split(x, bins):
        model = smi_pipeline()
        model.fit(x[tr_idx], smi[tr_idx])
        pred[va_idx] = model.predict(x[va_idx])
    return smi - pred


def fit_external_clinical_and_smi(
    xtr: np.ndarray,
    ytr: np.ndarray,
    smitr: np.ndarray,
    xte: np.ndarray,
    smite: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train 전체로 임상 로짓 모델과 SMI 예측 모델을 학습해, train·외부 데이터의 임상점수와 SMI 잔차를 모두 계산."""
    logit = clinical_logit_pipeline()
    logit.fit(xtr, ytr)
    ctr = logit.decision_function(xtr)
    cte = logit.decision_function(xte)

    smi_model = smi_pipeline()
    smi_model.fit(xtr, smitr)
    rtr = smitr - smi_model.predict(xtr)
    rte = smite - smi_model.predict(xte)
    return ctr, cte, rtr, rte


def covariate_design(meta: pd.DataFrame) -> np.ndarray:
    """임상 원시변수를 결측대체+표준화해 회귀 보정용 공변량 행렬을 만듦."""
    x = clinical_raw(meta)
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(x))


def residualize_curve_within_cohort(curve: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """코호트 내에서 곡선의 각 위치를 임상 공변량에 최소제곱 회귀시켜 잔차만 남김 (임상변수의 선형 효과 제거)."""
    z = covariate_design(meta)
    z = np.column_stack([np.ones(z.shape[0]), z])
    beta = np.linalg.lstsq(z, curve, rcond=None)[0]
    return curve - z @ beta


def t_curve_binary(resid: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """이진 라벨(y)에 대해 위치별 평균차와 Welch식 t통계량 곡선을 계산."""
    y = y.astype(bool)
    a = resid[y]
    b = resid[~y]
    diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    va = np.nanvar(a, axis=0, ddof=1)
    vb = np.nanvar(b, axis=0, ddof=1)
    se = np.sqrt(va / max(1, a.shape[0]) + vb / max(1, b.shape[0]))
    se[~np.isfinite(se) | (se < 1e-12)] = np.nan
    return diff, diff / se


def t_curve_continuous(resid: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """연속형 목표변수(target)에 대해 위치별 피어슨 상관계수와 그에 대응하는 t통계량 곡선을 계산."""
    target = np.asarray(target, dtype=float)
    x = resid - np.nanmean(resid, axis=0, keepdims=True)
    y = target - np.nanmean(target)
    denom = np.sqrt(np.sum(x**2, axis=0) * np.sum(y**2))
    r = np.sum(x * y[:, None], axis=0) / np.maximum(denom, 1e-12)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((len(target) - 2) / np.maximum(1e-12, 1 - r**2))
    return r, t


def find_clusters(stat: np.ndarray, threshold: float = T_THRESHOLD) -> list[dict]:
    """t통계량 곡선에서 |t|>=threshold인 연속 구간(클러스터)을 찾아, 위치·길이·질량(절댓값 합)·최고점을 기록."""
    clusters = []
    for sign_name, mask in [("positive", stat >= threshold), ("negative", stat <= -threshold)]:
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            if start is not None and ((not val) or i == len(mask) - 1):
                end = i if val and i == len(mask) - 1 else i - 1
                seg = stat[start : end + 1]
                clusters.append(
                    {
                        "sign": sign_name,
                        "start_point": start + 1,
                        "end_point": end + 1,
                        "length": end - start + 1,
                        "mass": float(np.sum(np.abs(seg))),
                        "peak_abs_stat": float(np.max(np.abs(seg))),
                        "peak_point": int(start + np.argmax(np.abs(seg)) + 1),
                    }
                )
                start = None
    return clusters


def max_cluster_mass(stat: np.ndarray, threshold: float = T_THRESHOLD) -> float:
    """t통계량 곡선에서 발견된 클러스터들 중 가장 큰 질량(mass)을 반환 (순열분포 구축용)."""
    clusters = find_clusters(stat, threshold=threshold)
    if not clusters:
        return 0.0
    return float(max(c["mass"] for c in clusters))


def cluster_permutation_binary(resid: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """이진 라벨 y를 n_perm회 무작위로 섞어 최대 클러스터 질량의 순열분포를 만들고, 관측된 각 클러스터의
    family-wise error rate(FWER) p값을 계산해 클러스터표와 위치별 통계표를 반환."""
    rng = np.random.default_rng(SEED + 101)
    diff, t_obs = t_curve_binary(resid, y)
    clusters = find_clusters(t_obs)
    max_masses = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        yp = rng.permutation(y)
        _, tp = t_curve_binary(resid, yp)
        max_masses[i] = max_cluster_mass(tp)
    rows = []
    for c in clusters:
        row = dict(c)
        idx = np.arange(row["start_point"] - 1, row["end_point"])
        row["mean_diff_low_minus_nonlow"] = float(np.mean(diff[idx]))
        row["cluster_fwer_p"] = float((np.sum(max_masses >= row["mass"]) + 1) / (n_perm + 1))
        rows.append(row)
    point_df = pd.DataFrame(
        {
            "point": np.arange(1, resid.shape[1] + 1),
            "mean_diff_low_minus_nonlow": diff,
            "t_stat": t_obs,
        }
    )
    return pd.DataFrame(rows), point_df


def cluster_permutation_continuous(resid: np.ndarray, target: np.ndarray, n_perm: int = N_PERM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """연속형 목표변수를 n_perm회 무작위로 섞어 최대 클러스터 질량의 순열분포를 만들고, 관측된 각 클러스터의
    FWER p값을 계산해 클러스터표와 위치별 통계표를 반환."""
    rng = np.random.default_rng(SEED + 202)
    r_obs, t_obs = t_curve_continuous(resid, target)
    clusters = find_clusters(t_obs)
    max_masses = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        tp_target = rng.permutation(target)
        _, tp = t_curve_continuous(resid, tp_target)
        max_masses[i] = max_cluster_mass(tp)
    rows = []
    for c in clusters:
        row = dict(c)
        idx = np.arange(row["start_point"] - 1, row["end_point"])
        row["mean_r_aec_residual_vs_smi_residual"] = float(np.mean(r_obs[idx]))
        row["cluster_fwer_p"] = float((np.sum(max_masses >= row["mass"]) + 1) / (n_perm + 1))
        rows.append(row)
    point_df = pd.DataFrame(
        {
            "point": np.arange(1, resid.shape[1] + 1),
            "r_aec_residual_vs_smi_residual": r_obs,
            "t_stat": t_obs,
        }
    )
    return pd.DataFrame(rows), point_df


def nearest_clinical_pairs(y: np.ndarray, clinical_score: np.ndarray, caliper_sd: float = 0.20) -> pd.DataFrame:
    """각 저근감소증 환자를 임상점수가 가장 가까운 비저근감소증 환자와 1:1 매칭하고(가장 가까운 이웃),
    거리가 caliper(0.20 표준편차) 이내인지 표시."""
    y = y.astype(bool)
    low_idx = np.flatnonzero(y)
    non_idx = np.flatnonzero(~y)
    non_scores = clinical_score[non_idx]
    caliper = float(caliper_sd * np.nanstd(clinical_score))
    rows = []
    order = np.argsort(non_scores)
    sorted_non_idx = non_idx[order]
    sorted_non_scores = non_scores[order]
    for i in low_idx:
        pos = int(np.searchsorted(sorted_non_scores, clinical_score[i]))
        candidates = []
        if pos < len(sorted_non_idx):
            candidates.append(sorted_non_idx[pos])
        if pos > 0:
            candidates.append(sorted_non_idx[pos - 1])
        if not candidates:
            continue
        j = min(candidates, key=lambda jj: abs(clinical_score[i] - clinical_score[jj]))
        distance = float(abs(clinical_score[i] - clinical_score[j]))
        rows.append(
            {
                "low_index": int(i),
                "matched_nonlow_index": int(j),
                "clinical_score_low": float(clinical_score[i]),
                "clinical_score_nonlow": float(clinical_score[j]),
                "abs_clinical_score_distance": distance,
                "within_caliper": bool(distance <= caliper),
                "caliper": caliper,
            }
        )
    return pd.DataFrame(rows)


def matched_curve_summary(resid: np.ndarray, pairs: pd.DataFrame, n_boot: int = 3000) -> pd.DataFrame:
    """caliper 내 매칭쌍들의 위치별 잔차 차이(저근감소증-매칭된 비저근감소증) 평균과 부트스트랩 95% 신뢰구간을 계산."""
    use = pairs[pairs["within_caliper"]].copy()
    diffs = resid[use["low_index"].to_numpy(dtype=int)] - resid[use["matched_nonlow_index"].to_numpy(dtype=int)]
    rng = np.random.default_rng(SEED + 303)
    boot = np.zeros((n_boot, diffs.shape[1]), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, diffs.shape[0], size=diffs.shape[0])
        boot[i] = np.mean(diffs[idx], axis=0)
    mean = np.mean(diffs, axis=0)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    return pd.DataFrame(
        {
            "point": np.arange(1, resid.shape[1] + 1),
            "matched_low_minus_nonlow_mean": mean,
            "ci2.5": lo,
            "ci97.5": hi,
            "n_pairs": int(diffs.shape[0]),
        }
    )


def contiguous_segments(mask: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """불리언 배열에서 True가 연속되는 구간(길이 min_len 이상)들을 (시작, 끝) 인덱스로 찾음."""
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if start is not None and ((not val) or i == len(mask) - 1):
            end = i if val and i == len(mask) - 1 else i - 1
            if end - start + 1 >= min_len:
                segments.append((start, end))
            start = None
    return segments


def common_direction_segments(point_tables: dict[str, pd.DataFrame], col: str, min_len: int = 4) -> pd.DataFrame:
    """g1090과 sdata 양쪽에서 같은 부호(둘 다 양수/음수)인 연속 구간(길이 min_len 이상)을 찾아 길이·최소절댓값 기준으로 정렬."""
    a = point_tables["g1090"][col].to_numpy(dtype=float)
    b = point_tables["sdata"][col].to_numpy(dtype=float)
    rows = []
    for name, mask in [("common_positive", (a > 0) & (b > 0)), ("common_negative", (a < 0) & (b < 0))]:
        for start, end in contiguous_segments(mask, min_len=min_len):
            rows.append(
                {
                    "direction": name,
                    "start_point": start + 1,
                    "end_point": end + 1,
                    "length": end - start + 1,
                    "g1090_mean": float(np.mean(a[start : end + 1])),
                    "sdata_mean": float(np.mean(b[start : end + 1])),
                    "min_abs_mean": float(min(abs(np.mean(a[start : end + 1])), abs(np.mean(b[start : end + 1])))),
                }
            )
    return pd.DataFrame(rows).sort_values(["length", "min_abs_mean"], ascending=[False, False])


def feature_summary_from_segments(
    datasets: dict[str, dict],
    residuals: dict[str, np.ndarray],
    smi_residuals: dict[str, np.ndarray],
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """공통 방향 구간 상위 8개 각각을 구간평균 특징으로 만들어, 코호트별 저근감소증 연관성(AUC·평균차)과 SMI 잔차와의 상관을 계산."""
    if segments.empty:
        return pd.DataFrame()
    selected = segments.head(8).copy()
    rows = []
    for _, seg in selected.iterrows():
        idx = np.arange(int(seg["start_point"]) - 1, int(seg["end_point"]))
        feature_name = f"{seg['direction']}_{int(seg['start_point'])}_{int(seg['end_point'])}"
        for cohort in ["g1090", "sdata"]:
            vals = residuals[cohort][:, idx].mean(axis=1)
            y = datasets[cohort]["y"].astype(bool)
            smir = smi_residuals[cohort]
            r, rp = stats.pearsonr(vals, smir)
            mw = stats.mannwhitneyu(vals[y], vals[~y], alternative="two-sided")
            auc_high = float(mw.statistic / (np.sum(y) * np.sum(~y)))
            rows.append(
                {
                    "feature": feature_name,
                    "cohort": cohort,
                    "direction": seg["direction"],
                    "start_point": int(seg["start_point"]),
                    "end_point": int(seg["end_point"]),
                    "n": int(len(vals)),
                    "events": int(np.sum(y)),
                    "low_mean": float(np.mean(vals[y])),
                    "nonlow_mean": float(np.mean(vals[~y])),
                    "diff_low_minus_nonlow": float(np.mean(vals[y]) - np.mean(vals[~y])),
                    "auc_if_higher_predicts_low_smi": auc_high,
                    "auc_best_direction": float(max(auc_high, 1 - auc_high)),
                    "pearson_r_with_smi_residual": float(r),
                    "pearson_p_with_smi_residual": float(rp),
                }
            )
    return pd.DataFrame(rows)


def plot_matched_curves(matched_points: dict[str, pd.DataFrame]) -> None:
    """임상점수로 매칭된 쌍들의 위치별 잔차 차이(+95%CI)를 코호트별로 겹쳐 그려 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"g1090": "#4C78A8", "sdata": "#F58518"}
    for cohort, df in matched_points.items():
        x = df["point"].to_numpy()
        y = df["matched_low_minus_nonlow_mean"].to_numpy()
        lo = df["ci2.5"].to_numpy()
        hi = df["ci97.5"].to_numpy()
        ax.plot(x, y, lw=2.2, color=colors[cohort], label=f"{cohort} matched pairs")
        ax.fill_between(x, lo, hi, color=colors[cohort], alpha=0.18, lw=0)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel("AEC_128 point")
    ax.set_ylabel("Clinical-matched low - non-low residual log(AEC)")
    ax.set_title("Clinical-Matched AEC Residual Difference", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clinical_matched_residual_aec_difference.png", dpi=220)
    plt.close(fig)


def plot_cluster_binary(point_tables: dict[str, pd.DataFrame], cluster_tables: dict[str, pd.DataFrame]) -> None:
    """코호트별 위치별 t통계량 곡선을 그리고, FWER p<=0.10인 유의 클러스터 구간을 음영+p값 표시로 강조한 PNG를 저장."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    colors = {"g1090": "#4C78A8", "sdata": "#F58518"}
    for ax, cohort in zip(axes, ["g1090", "sdata"]):
        df = point_tables[cohort]
        ax.plot(df["point"], df["t_stat"], color=colors[cohort], lw=2.0, label=cohort)
        ax.axhline(T_THRESHOLD, color="#555555", lw=1, ls="--")
        ax.axhline(-T_THRESHOLD, color="#555555", lw=1, ls="--")
        ax.axhline(0, color="#999999", lw=0.8)
        clusters = cluster_tables[cohort]
        if not clusters.empty:
            for _, row in clusters.iterrows():
                if row["cluster_fwer_p"] <= 0.10:
                    ax.axvspan(row["start_point"], row["end_point"], color="#54A24B", alpha=0.14)
                    ax.text(
                        (row["start_point"] + row["end_point"]) / 2,
                        ax.get_ylim()[1] * 0.86,
                        f"p={row['cluster_fwer_p']:.3f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                    )
        ax.set_ylabel("t-stat")
        ax.set_title(f"{cohort}: cluster permutation on clinical-adjusted AEC residuals", loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("AEC_128 point")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_permutation_residual_aec_low_smi.png", dpi=220)
    plt.close(fig)


def plot_smi_correlation(point_tables: dict[str, pd.DataFrame], cluster_tables: dict[str, pd.DataFrame]) -> None:
    """코호트별 위치별 (AEC 잔차 vs SMI 잔차) 피어슨 상관 곡선을 겹쳐 그려 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"g1090": "#4C78A8", "sdata": "#F58518"}
    for cohort, df in point_tables.items():
        ax.plot(df["point"], df["r_aec_residual_vs_smi_residual"], color=colors[cohort], lw=2.1, label=cohort)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel("AEC_128 point")
    ax.set_ylabel("Pearson r: AEC residual vs SMI residual")
    ax.set_title("Continuous SMI Residual Signal Along AEC Curve", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "smi_residual_correlation_aec.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 지금까지의 발견들이 순전히 임상변수 차이나 우연한 패턴
    매칭 때문이 아니라, 통계적으로 엄격하게 검증해도 살아남는 "진짜" 신호인가? — 클러스터 순열검정
    + 임상점수 매칭이라는 두 가지 엄격한 방법으로 감사):

    1. g1090/sdata를 로드하고, 각 코호트 내에서 로그 정규화 곡선을 임상변수(나이/성별/키/몸무게)에
       회귀시켜 잔차 곡선(residual_curve)을 만든다. 임상 로짓 점수(OOF/외부)와 SMI 잔차
       (임상변수로 설명 안 되는 SMI 부분)도 함께 계산한다.
    2. [임상점수 매칭 검증] nearest_clinical_pairs로 각 저근감소증 환자를 임상점수가 가장 가까운
       비저근감소증 환자와 1:1 매칭하고, matched_curve_summary로 매칭쌍들의 위치별 잔차 차이와
       부트스트랩 신뢰구간을 계산한다 — "임상적으로 거의 동일한 사람들끼리 비교해도 AEC가 다른가?"
    3. [클러스터 순열검정] cluster_permutation_binary/continuous로, 저근감소증 여부(이진) 및
       SMI 잔차(연속) 각각에 대해 위치별 t통계량 곡선을 만들고, 라벨을 수천 번 무작위로 섞어
       "우연히 이 정도 크기의 연속 구간이 나올 확률"(FWER로 다중비교 보정된 p값)을 계산한다.
    4. common_direction_segments로 매칭곡선/잔차곡선/SMI상관곡선 각각에서 두 코호트 모두 같은
       방향인 공통 구간을 찾고, feature_summary_from_segments로 그 구간들의 판별력(AUC)과
       SMI 잔차와의 상관을 정리한다.
    5. 매칭곡선 그래프, 클러스터 유의구간 표시 그래프, SMI 상관 그래프를 PNG로 저장.
    6. 코호트별 표본수·매칭 성공률·유의 클러스터 개수 등을 요약 CSV로, 방법론 설명 전체를 JSON으로
       저장하고, 클러스터·공통구간 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    for d in datasets.values():
        d["y"] = d["y"].astype(int)
        d["smi"] = smi_from_meta(d["meta"])
        d["clinical_x"] = clinical_raw(d["meta"])
        d["log_curve"] = np.log(np.clip(d["x"], 1e-6, None))
        d["residual_curve"] = residualize_curve_within_cohort(d["log_curve"], d["meta"])

    g = datasets["g1090"]
    s = datasets["sdata"]
    g["clinical_score_oof"] = crossfit_clinical_score(g["clinical_x"], g["y"])
    g_full_score, s_score, g_smi_resid_full, s_smi_resid = fit_external_clinical_and_smi(
        g["clinical_x"], g["y"], g["smi"], s["clinical_x"], s["smi"]
    )
    g["clinical_score_full"] = g_full_score
    s["clinical_score_external"] = s_score
    g["smi_residual_oof"] = crossfit_smi_residual(g["clinical_x"], g["smi"])
    g["smi_residual_full"] = g_smi_resid_full
    s["smi_residual_external"] = s_smi_resid

    matched_pairs = {
        "g1090": nearest_clinical_pairs(g["y"], g["clinical_score_oof"]),
        "sdata": nearest_clinical_pairs(s["y"], s["clinical_score_external"]),
    }
    matched_points = {
        cohort: matched_curve_summary(datasets[cohort]["residual_curve"], pairs)
        for cohort, pairs in matched_pairs.items()
    }
    for cohort, pairs in matched_pairs.items():
        pairs.to_csv(OUT_DIR / f"{cohort}_clinical_matched_pairs.csv", index=False)
        matched_points[cohort].to_csv(OUT_DIR / f"{cohort}_clinical_matched_curve_summary.csv", index=False)

    binary_clusters = {}
    binary_points = {}
    continuous_clusters = {}
    continuous_points = {}
    smi_residuals = {"g1090": g["smi_residual_oof"], "sdata": s["smi_residual_external"]}
    for cohort in ["g1090", "sdata"]:
        clusters, points = cluster_permutation_binary(datasets[cohort]["residual_curve"], datasets[cohort]["y"])
        clusters.insert(0, "cohort", cohort)
        points.insert(0, "cohort", cohort)
        binary_clusters[cohort] = clusters
        binary_points[cohort] = points

        c_clusters, c_points = cluster_permutation_continuous(datasets[cohort]["residual_curve"], smi_residuals[cohort])
        c_clusters.insert(0, "cohort", cohort)
        c_points.insert(0, "cohort", cohort)
        continuous_clusters[cohort] = c_clusters
        continuous_points[cohort] = c_points

    binary_cluster_df = pd.concat(binary_clusters.values(), ignore_index=True)
    binary_point_df = pd.concat(binary_points.values(), ignore_index=True)
    cont_cluster_df = pd.concat(continuous_clusters.values(), ignore_index=True)
    cont_point_df = pd.concat(continuous_points.values(), ignore_index=True)

    binary_cluster_df.to_csv(OUT_DIR / "cluster_permutation_low_smi_clusters.csv", index=False)
    binary_point_df.to_csv(OUT_DIR / "cluster_permutation_low_smi_point_stats.csv", index=False)
    cont_cluster_df.to_csv(OUT_DIR / "cluster_permutation_smi_residual_clusters.csv", index=False)
    cont_point_df.to_csv(OUT_DIR / "cluster_permutation_smi_residual_point_stats.csv", index=False)

    matched_common = common_direction_segments(matched_points, "matched_low_minus_nonlow_mean", min_len=4)
    residual_common = common_direction_segments(
        {k: v for k, v in binary_points.items()}, "mean_diff_low_minus_nonlow", min_len=4
    )
    smi_common = common_direction_segments(
        {k: v for k, v in continuous_points.items()}, "r_aec_residual_vs_smi_residual", min_len=4
    )
    matched_common.to_csv(OUT_DIR / "common_direction_segments_matched_pairs.csv", index=False)
    residual_common.to_csv(OUT_DIR / "common_direction_segments_residual_low_smi.csv", index=False)
    smi_common.to_csv(OUT_DIR / "common_direction_segments_smi_residual.csv", index=False)

    feature_df = feature_summary_from_segments(datasets, {k: v["residual_curve"] for k, v in datasets.items()}, smi_residuals, residual_common)
    feature_df.to_csv(OUT_DIR / "common_segment_feature_summary.csv", index=False)

    plot_matched_curves(matched_points)
    plot_cluster_binary(binary_points, binary_clusters)
    plot_smi_correlation(continuous_points, continuous_clusters)

    summary_rows = []
    for cohort in ["g1090", "sdata"]:
        pairs = matched_pairs[cohort]
        summary_rows.append(
            {
                "cohort": cohort,
                "n": int(len(datasets[cohort]["y"])),
                "events": int(np.sum(datasets[cohort]["y"])),
                "event_rate": float(np.mean(datasets[cohort]["y"])),
                "clinical_auc": float(
                    roc_auc_score(
                        datasets[cohort]["y"],
                        datasets[cohort]["clinical_score_oof"] if cohort == "g1090" else datasets[cohort]["clinical_score_external"],
                    )
                ),
                "matched_low_patients": int(len(pairs)),
                "matched_pairs_within_caliper": int(np.sum(pairs["within_caliper"])),
                "matched_pair_median_abs_clinical_distance": float(np.median(pairs["abs_clinical_score_distance"])),
                "matched_pair_caliper": float(pairs["caliper"].iloc[0]) if len(pairs) else np.nan,
                "n_low_smi_clusters_p_le_0.10": int(np.sum(binary_clusters[cohort].get("cluster_fwer_p", pd.Series(dtype=float)) <= 0.10)),
                "n_smi_residual_clusters_p_le_0.10": int(
                    np.sum(continuous_clusters[cohort].get("cluster_fwer_p", pd.Series(dtype=float)) <= 0.10)
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "aec128_signal_audit_summary.csv", index=False)

    summary = {
        "method": {
            "curve": "log(patient-normalized AEC_128)",
            "residualization": "Within each cohort, each AEC point was linearly adjusted for age, sex, height, and weight before signal tests.",
            "clinical_matching": "Each low-SMI patient was nearest-neighbor matched to a non-low-SMI patient by clinical logit; pairs within 0.20 SD clinical logit were used for curve summaries.",
            "cluster_permutation": f"Pointwise t-statistics thresholded at |t| >= {T_THRESHOLD}; familywise p from {N_PERM} label permutations using maximum cluster mass.",
            "continuous_endpoint": "SMI residual after clinical prediction; g1090 uses OOF residuals, sdata uses g1090-trained external residuals.",
        },
        "outputs": {
            "summary_csv": str(OUT_DIR / "aec128_signal_audit_summary.csv"),
            "matched_curve_plot": str(OUT_DIR / "clinical_matched_residual_aec_difference.png"),
            "cluster_plot": str(OUT_DIR / "cluster_permutation_residual_aec_low_smi.png"),
            "smi_residual_plot": str(OUT_DIR / "smi_residual_correlation_aec.png"),
            "low_smi_clusters_csv": str(OUT_DIR / "cluster_permutation_low_smi_clusters.csv"),
            "smi_residual_clusters_csv": str(OUT_DIR / "cluster_permutation_smi_residual_clusters.csv"),
            "common_segment_features_csv": str(OUT_DIR / "common_segment_feature_summary.csv"),
        },
    }
    (OUT_DIR / "aec128_signal_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Summary")
    print(summary_df)
    print("\nLow-SMI residual AEC clusters")
    print(binary_cluster_df.sort_values(["cohort", "cluster_fwer_p", "start_point"]).head(20))
    print("\nContinuous SMI residual clusters")
    print(cont_cluster_df.sort_values(["cohort", "cluster_fwer_p", "start_point"]).head(20))
    print("\nCommon residual low-SMI direction segments")
    print(residual_common.head(12))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
