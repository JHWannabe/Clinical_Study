from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    counts,
    gate_metrics,
    load_aec128,
    risk_direction,
    standardize_train_test,
    summarize_train_metrics,
)

# NOTE: the original aec_individual_feature_full_metrics.py (and its output CSVs) was
# lost with no recoverable backup. This is a reconstruction, reverse-engineered from
# the exact column names and gate formula expected by its four surviving consumers
# (aec_deesc_curve_vs_clinical_auc.py, aec_single_operating_points_no_integral.py,
# aec_2of3_s80_90_visualize.py, aec_block_or_consensus_search.py) and from the
# feature-bank / width-lambda-grid / gate machinery shared via aec_midrange_feature_refit.py.
# The TOP_K shortlist size is an approximation of the original scope; REQUIRED_FEATURES
# guarantees the specific features every consumer script names by hand are always present.

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics"
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55]
TOP_K = 40
REQUIRED_FEATURES = [
    "norm_curv_010_025_max",
    "norm_curv_010_021_max",
    "norm_slope_013_016_sd",
    "norm_curv_007_010_min",
    "dct_log_17",
    "norm_curv_055_058_mean",
    # aec_block_or_consensus_search.py's MIDLATE4 block expects these two from this
    # config table specifically -- they are absent from top3000_individual_screen_summary.csv.
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
]


def exact_paired_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 게이트 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
    yy = y.astype(bool)
    clinical_pos = clinical_pos.astype(bool)
    final_pos = final_pos.astype(bool)

    pos = yy
    sens_loss = int(np.sum(pos & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(pos & ~clinical_pos & final_pos))

    neg = ~yy
    spec_gain = int(np.sum(neg & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(neg & ~clinical_pos & final_pos))

    clinical_correct = clinical_pos == yy
    final_correct = final_pos == yy
    acc_loss = int(np.sum(clinical_correct & ~final_correct))
    acc_gain = int(np.sum(~clinical_correct & final_correct))

    return {
        "sensitivity_loss_p_exact": exact_paired_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_paired_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_paired_p(acc_gain, acc_loss),
    }


def bootstrap_delta_auc_p(
    y: np.ndarray,
    clinical_pos: np.ndarray,
    final_pos: np.ndarray,
    n_boot: int = 5000,
    seed: int = 20260630,
) -> float:
    """부트스트랩 재표본추출로 이진(문턱값 기반) AUC 변화량(게이트 후 - 임상단독)의 분포를 만들고, 그 값이 0 이하일 확률을 계산."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        c = counts(y[idx], clinical_pos[idx])
        f = counts(y[idx], final_pos[idx])
        vals.append(0.5 * (f["sensitivity"] + f["specificity"]) - 0.5 * (c["sensitivity"] + c["specificity"]))
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals <= 0.0))


def row_metrics(
    feature_label: str,
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk_z: np.ndarray,
    op: str,
    threshold_z: float,
    width: float,
    lam: float,
) -> dict:
    """한 특징·한 운영점에 대해 임상단독 vs 게이트적용 후 이진 AUC(=균형정확도)와 각종 p값을 모두 계산해 하나의 결과 행으로 반환."""
    boundary = np.exp(-0.5 * ((clinical_z - threshold_z) / width) ** 2)
    gate = clinical_z + lam * boundary * aec_risk_z
    clinical_pos = clinical_z >= threshold_z
    final_pos = clinical_pos & (gate >= threshold_z)

    clinical = counts(y, clinical_pos)
    post = counts(y, final_pos)
    pv = pvalues(y, clinical_pos, final_pos)

    deesc = clinical_pos & ~final_pos
    keep = final_pos
    a = int(np.sum(y[keep] == 1))
    b = int(np.sum(y[keep] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    fisher_p = stats.fisher_exact([[a, b], [c, d]])[1] if (a + b) and (c + d) else np.nan

    clinical_auc_binary = 0.5 * (clinical["sensitivity"] + clinical["specificity"])
    post_auc_binary = 0.5 * (post["sensitivity"] + post["specificity"])

    return {
        "feature_label": feature_label,
        "operating_point": op,
        "clinical_auc_binary": clinical_auc_binary,
        "post_auc_binary": post_auc_binary,
        "delta_auc_binary": post_auc_binary - clinical_auc_binary,
        "auc_delta_p_bootstrap": bootstrap_delta_auc_p(y, clinical_pos, final_pos),
        "clinical_sensitivity": clinical["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "delta_sensitivity": post["sensitivity"] - clinical["sensitivity"],
        "clinical_specificity": clinical["specificity"],
        "post_specificity": post["specificity"],
        "delta_specificity": post["specificity"] - clinical["specificity"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": c,
        "deesc_event_rate": c / (c + d) if c + d else np.nan,
        "fp_removed": d,
        "tp_lost": c,
        "deesc_event_fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
        **pv,
    }


def best_config_per_feature(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, names: list[str], thresholds: dict) -> pd.DataFrame:
    """특징마다 폭(width) 3종 x 람다(lambda) 3종 조합 중 train 선택 점수가 가장 높은 하나를 골라 반환 (train/OOF 전용, 외부데이터 미사용)."""
    rows = []
    for j, name in enumerate(names):
        z = x[:, j]
        best_row = None
        for width in WIDTHS:
            for lam in LAMBDAS:
                metrics = []
                for label, cfg in thresholds.items():
                    m = gate_metrics(y, clinical_z, z, cfg["clinical_z"], width, lam)
                    metrics.append({"operating_point": label, **m})
                row = {"feature": name, "width": width, "lambda": lam}
                row.update(summarize_train_metrics(metrics))
                if best_row is None or row["train_selection_score"] > best_row["train_selection_score"]:
                    best_row = row
        rows.append(best_row)
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 페어/조합이 아닌 "개별" 후보 특징 하나하나에 대해, 각자에게
    최적인 게이트 폭·람다를 하나씩 골라주면, 외부데이터에서 임상단독 대비 이진 AUC를 얼마나
    끌어올리고 그게 통계적으로 유의한지, 5개 운영점 전체에 걸쳐 남김없이 정리):

    1. g1090/sdata를 로드해 임상점수·5개 운영점(youden/sens80~95) 임계값을 준비.
    2. build_candidate_bank로 초대형 후보 특징 테이블을 만들고 표준화·위험방향 통일.
    3. 특징마다 g1090 OOF에서 폭 3종 x 람다 3종 중 train 선택 점수가 가장 높은 조합을 골라
       "선택된 개별 특징 설정" 표로 저장.
    4. train 선택 점수 상위 특징 + 이후 스크립트들이 이름으로 지정하는 특징들을 합쳐, 각각의
       선택된 폭·람다로 sdata(external)에서 5개 운영점 전체의 이진 AUC/민감도/특이도 변화와
       각종 p값(Fisher, 부호검정, 부트스트랩)을 계산해 전체 지표 표로 저장.
    5. 저장한 표들의 크기를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]

    search_df = best_config_per_feature(g["y"], c_g, xg, names, thresholds)
    search_df = search_df.sort_values("train_selection_score", ascending=False)
    search_df["label"] = (
        search_df["feature"].str.replace("bank_norm__", "", regex=False).str.replace("midrange__", "", regex=False)
    )

    shortlist_labels = list(dict.fromkeys(search_df.head(TOP_K)["label"].tolist() + REQUIRED_FEATURES))
    shortlist = search_df[search_df["label"].isin(shortlist_labels)].drop_duplicates("label").copy()

    config_cols = ["label", "feature", "width", "lambda", "train_selection_score"]
    shortlist[config_cols].to_csv(OUT_DIR / "selected_individual_feature_configs.csv", index=False)

    name_to_idx = {name: i for i, name in enumerate(names)}
    y_s = s["y"].astype(int)
    metric_rows = []
    for _, r in shortlist.iterrows():
        label = str(r["label"])
        idx = name_to_idx[str(r["feature"])]
        width = float(r["width"])
        lam = float(r["lambda"])
        for op, cfg in thresholds.items():
            metric_rows.append(
                row_metrics(
                    feature_label=label,
                    y=y_s,
                    clinical_z=c_s,
                    aec_risk_z=xs[:, idx],
                    op=op,
                    threshold_z=cfg["clinical_z"],
                    width=width,
                    lam=lam,
                )
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT_DIR / "individual_features_all_metrics_with_pvalues.csv", index=False)

    print(f"selected_individual_feature_configs.csv: {len(shortlist)} features -> {OUT_DIR}")
    print(f"individual_features_all_metrics_with_pvalues.csv: {len(metrics_df)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
