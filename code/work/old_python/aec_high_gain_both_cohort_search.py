from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import build_candidate_bank, clinical_scores, load_aec128, risk_direction, standardize_train_test  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_high_gain_both_cohort_search"
SCREEN_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search" / "top3000_individual_screen_summary.csv"
BOTH_SUMMARY_PATH = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search" / "top3000_individual_both_cohort_survival_summary.csv"
)
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측 양성/음성과 실제 결과로부터 민감도, 특이도, 균형정확도, 정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": 0.5 * (sens + spec), "accuracy": (tp + tn) / len(y)}


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvals(y: np.ndarray, cpos: np.ndarray, fpos: np.ndarray) -> dict:
    """임상단독 판정과 k-of-m 규칙 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & cpos & ~fpos))
    sens_gain = int(np.sum(yy & ~cpos & fpos))
    spec_gain = int(np.sum(~yy & cpos & ~fpos))
    spec_loss = int(np.sum(~yy & ~cpos & fpos))
    cc = cpos == yy
    fc = fpos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
    }


def fisher_p(y: np.ndarray, fpos: np.ndarray, deesc: np.ndarray) -> float:
    """최종 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[fpos] == 1))
    b = int(np.sum(y[fpos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    return np.nan if not (a + b and c + d) else float(stats.fisher_exact([[a, b], [c, d]])[1])


def metric_row(dataset: str, rule: str, features: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    """한 데이터셋·규칙·운영점 조합에 대해 임상단독 대비 최종판정의 성능 변화와 하향조정군 사건비율·Fisher p값을 계산."""
    fpos = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, fpos)
    return {
        "dataset": dataset,
        "rule": rule,
        "features": features,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_p(y, fpos, deesc),
        **paired_pvals(y, cpos, fpos),
    }


def finite_min(vals: list[float], default: float = 0.0) -> float:
    """리스트에서 결측이 아닌 값들만 골라 최솟값을 구하고, 모두 결측이면 default를 반환."""
    arr = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    return float(arr.min()) if arr.size else default


def finite_max(vals: list[float], default: float = 1.0) -> float:
    """리스트에서 결측이 아닌 값들만 골라 최댓값을 구하고, 모두 결측이면 default를 반환."""
    arr = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    return float(arr.max()) if arr.size else default


def summarize(rows: list[dict], dataset: str) -> dict:
    """지정된 데이터셋(Gangnam/Sinchon)에서 5개 운영점에 걸친 metric_row 결과를 모아 각 지표의 최솟값/최댓값/평균을 계산."""
    sub = [r for r in rows if r["dataset"] == dataset]
    return {
        f"{dataset}_min_p_loss": finite_min([r["sensitivity_loss_p_exact"] for r in sub]),
        f"{dataset}_max_sens_loss": finite_max([r["sensitivity_loss"] for r in sub]),
        f"{dataset}_min_spec_gain": finite_min([r["specificity_gain"] for r in sub]),
        f"{dataset}_mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        f"{dataset}_min_delta_ba": finite_min([r["delta_balanced_accuracy"] for r in sub]),
        f"{dataset}_mean_delta_ba": float(np.nanmean([r["delta_balanced_accuracy"] for r in sub])),
        f"{dataset}_max_fisher": finite_max([r["deesc_event_fisher_p"] for r in sub]),
    }


def build_pool() -> pd.DataFrame:
    """aec_top3000_kofn_fine_search가 저장한 "두 코호트 생존 요약"과 "개별 특징 스크리닝" 표를 합쳐, 두 코호트 모두에서 특이도이득이 확실하고(>1.5%p) Fisher p가 낮으며 민감도손실이 작은(≤7.5%) 특징들만 골라 high_gain_score로 정렬한 뒤 다양성을 고려해 상위 22개 탐색 풀을 구성."""
    both = pd.read_csv(BOTH_SUMMARY_PATH)
    screen = pd.read_csv(SCREEN_PATH)
    # One concrete width/lambda setting per short feature, preferring the best pool_score.
    if "pool_score" not in screen.columns:
        screen["pool_score"] = 0.0
    configs = screen.sort_values("pool_score", ascending=False).drop_duplicates("feature_short")
    both = both.merge(
        configs[["feature_short", "feature", "width", "lambda", "pool_score"]],
        left_on="feature",
        right_on="feature_short",
        how="inner",
        suffixes=("_metric", ""),
    )
    both = both.rename(columns={"feature_metric": "feature_label"})
    both["min_both_spec"] = both[["g_min_spec_gain", "s_min_spec_gain"]].min(axis=1)
    both["mean_both_spec"] = both[["g_mean_spec_gain", "s_mean_spec_gain"]].min(axis=1)
    both["max_both_loss"] = both[["g_max_sens_loss", "s_max_sens_loss"]].max(axis=1)
    both["high_gain_score"] = 2.4 * both["min_both_spec"] + both["mean_both_spec"] - 0.25 * both["max_both_loss"]
    base = both[
        (both["g_min_spec_gain"] > 0.015)
        & (both["s_min_spec_gain"] > 0.015)
        & (both["g_max_fisher"] < 0.08)
        & (both["s_max_fisher"] < 0.08)
        & (both["max_both_loss"] <= 0.075)
    ].copy()
    parts = [
        base.sort_values("high_gain_score", ascending=False).head(34),
        base.sort_values("g_min_spec_gain", ascending=False).head(16),
        base.sort_values("s_min_spec_gain", ascending=False).head(16),
        base.sort_values("min_both_spec", ascending=False).head(20),
    ]
    pool = pd.concat(parts, ignore_index=True).drop_duplicates("feature_short")
    pool = pool.sort_values("high_gain_score", ascending=False).head(22).reset_index(drop=True)
    return pool


def evaluate_rules(pool: pd.DataFrame, max_m: int, top_m5: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """탐색 풀에서 크기 1~max_m의 모든 부분집합(5개 조합은 top_m5로 제한 가능) x 과반수 이상 k값 조합에 대해 k-of-m 규칙 성능을 계산하고, Gangnam·Sinchon 두 코호트 모두에서 생존조건을 만족하는지와 score로 정렬한 요약표, 생존한 고성능 규칙의 상세표를 반환."""
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    y_by = {"Gangnam": g["y"].astype(int), "Sinchon": s["y"].astype(int)}
    c_g, c_s, _ = clinical_scores(g, s)
    c_by = {"Gangnam": c_g, "Sinchon": c_s}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], c_g, target) for op, target in OPS}
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    x_by = {"Gangnam": xg * direction[None, :], "Sinchon": xs * direction[None, :]}
    name_to_idx = {name: i for i, name in enumerate(names)}

    sig: dict[tuple[str, str], np.ndarray] = {}
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset in ["Gangnam", "Sinchon"]:
        for op, _ in OPS:
            th = thresholds[op]
            cp = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cp
            mat = np.zeros((len(pool), len(y_by[dataset])), dtype=np.int8)
            for i, r in pool.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((c_by[dataset] - th) / float(r["width"])) ** 2)
                gate = c_by[dataset] + float(r["lambda"]) * boundary * x_by[dataset][:, idx]
                mat[i] = (cp & (gate < th)).astype(np.int8)
            sig[(dataset, op)] = mat

    summary_rows = []
    detail_rows = []
    pool_indices_for_m = {
        2: list(range(len(pool))),
        3: list(range(len(pool))),
        4: list(range(len(pool))),
        5: list(range(min(top_m5 or len(pool), len(pool)))),
    }
    for m in range(1, max_m + 1):
        idxs = pool_indices_for_m.get(m, list(range(len(pool))))
        if m == 1:
            k_values = [1]
        else:
            k_values = [k for k in range((m + 1) // 2, m + 1) if k >= 2]
        for subset in itertools.combinations(idxs, m):
            features = " + ".join(pool.iloc[list(subset)]["feature_short"].astype(str))
            for k in k_values:
                rule = f"{k}-of-{m}"
                rows = []
                for dataset in ["Gangnam", "Sinchon"]:
                    for op, _ in OPS:
                        votes = sig[(dataset, op)][list(subset)].sum(axis=0)
                        deesc = cpos_by[(dataset, op)] & (votes >= k)
                        rows.append(metric_row(dataset, rule, features, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
                gs = summarize(rows, "Gangnam")
                ss = summarize(rows, "Sinchon")
                survives = (
                    gs["Gangnam_min_p_loss"] >= 0.05
                    and ss["Sinchon_min_p_loss"] >= 0.05
                    and gs["Gangnam_min_spec_gain"] > 0
                    and ss["Sinchon_min_spec_gain"] > 0
                    and gs["Gangnam_max_fisher"] < 0.05
                    and ss["Sinchon_max_fisher"] < 0.05
                )
                score = (
                    2.2 * min(gs["Gangnam_min_spec_gain"], ss["Sinchon_min_spec_gain"])
                    + 0.9 * min(gs["Gangnam_mean_spec_gain"], ss["Sinchon_mean_spec_gain"])
                    + 0.6 * min(gs["Gangnam_min_delta_ba"], ss["Sinchon_min_delta_ba"])
                    - 0.25 * max(gs["Gangnam_max_sens_loss"], ss["Sinchon_max_sens_loss"])
                )
                summary_rows.append({"rule": rule, "features": features, "m": m, "k": k, "survives_both": survives, "score": score, **gs, **ss})
                if survives and score > 0.17:
                    detail_rows.extend(rows)
    summary = pd.DataFrame(summary_rows).sort_values(["survives_both", "score"], ascending=False)
    details = pd.DataFrame(detail_rows)
    return summary, details


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_top3000_kofn_fine_search가 이미 계산해둔 "두 코호트
    각각에서 안전한지" 결과를 바탕으로, 특이도이득이 유난히 큰(high-gain) 특징들만 골라 더
    작고 정예한 풀을 만들고, 그 안에서 1~4개 조합의 k-of-m 규칙을 탐색하면 Gangnam·Sinchon
    양쪽에서 더 큰 특이도이득을 내는 규칙을 찾을 수 있는가?):

    1. build_pool로 두 코호트 모두에서 특이도이득이 크고 안전한 특징들만 골라 22개 탐색 풀을
       구성해 CSV로 저장.
    2. evaluate_rules로 이 풀에서 크기 1~4의 모든 부분집합 x k값 조합에 대해 Gangnam·Sinchon
       양쪽 성능을 계산하고, 두 코호트 모두 생존조건을 만족하는지와 score로 정렬한 요약표와
       고성능 생존 규칙의 상세표를 만든다.
    3. 두 표를 CSV로 저장하고, 생존한 상위 25개 규칙을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    pool.to_csv(OUT_DIR / "high_gain_search_pool.csv", index=False)
    print(f"Pool size: {len(pool)}")
    summary, details = evaluate_rules(pool, max_m=4, top_m5=None)
    summary.to_csv(OUT_DIR / "high_gain_kofn_summary.csv", index=False)
    details.to_csv(OUT_DIR / "high_gain_survivor_details.csv", index=False)
    print(summary[summary["survives_both"]].head(25)[[
        "rule",
        "features",
        "Gangnam_min_p_loss",
        "Gangnam_max_sens_loss",
        "Gangnam_min_spec_gain",
        "Gangnam_mean_spec_gain",
        "Sinchon_min_p_loss",
        "Sinchon_max_sens_loss",
        "Sinchon_min_spec_gain",
        "Sinchon_mean_spec_gain",
        "score",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
