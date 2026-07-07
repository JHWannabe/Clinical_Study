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


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search"
POOL_PATH = OUT_DIR / "search_pool_from_top3000.csv"
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
POOL_N = 28


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


def pvals(y: np.ndarray, cpos: np.ndarray, fpos: np.ndarray) -> dict:
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


def fisher(y: np.ndarray, fpos: np.ndarray, deesc: np.ndarray) -> float:
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
        "deesc_event_fisher_p": fisher(y, fpos, deesc),
        **pvals(y, cpos, fpos),
    }


def summarize(rows: list[dict], dataset: str) -> dict:
    """지정된 데이터셋(Gangnam/Sinchon/Pooled)에서 5개 운영점에 걸친 metric_row 결과를 모아 각 지표의 최솟값/최댓값/평균과 생존조건(민감도손실 p≥0.05, 특이도이득>0, Fisher p<0.05) 충족 여부를 계산."""
    sub = [r for r in rows if r["dataset"] == dataset]
    p_loss = np.asarray([r["sensitivity_loss_p_exact"] for r in sub], dtype=float)
    spec_gain = np.asarray([r["specificity_gain"] for r in sub], dtype=float)
    fisher_p = np.asarray([r["deesc_event_fisher_p"] for r in sub], dtype=float)
    sens_loss = np.asarray([r["sensitivity_loss"] for r in sub], dtype=float)
    delta_ba = np.asarray([r["delta_balanced_accuracy"] for r in sub], dtype=float)
    return {
        f"{dataset}_min_p_loss": float(np.nanmin(p_loss)),
        f"{dataset}_max_sens_loss": float(np.nanmax(sens_loss)),
        f"{dataset}_min_spec_gain": float(np.nanmin(spec_gain)),
        f"{dataset}_mean_spec_gain": float(np.nanmean(spec_gain)),
        f"{dataset}_min_delta_ba": float(np.nanmin(delta_ba)),
        f"{dataset}_mean_delta_ba": float(np.nanmean(delta_ba)),
        f"{dataset}_max_fisher_p": float(np.nanmax(fisher_p)),
        f"{dataset}_survives": bool(np.nanmin(p_loss) >= 0.05 and np.nanmin(spec_gain) > 0 and np.nanmax(fisher_p) < 0.05),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_top3000_kofn_fine_search가 이미 골라둔 34개 탐색
    풀을 재사용해, Gangnam과 Sinchon을 "합친(pooled)" 기준이 아니라 두 코호트 각각을 독립적인
    기준으로 놓고 k-of-m 규칙을 더 빠르게, 더 엄격한 기준(양쪽 코호트 모두 생존)으로 재평가):

    1. aec_top3000_kofn_fine_search가 저장한 탐색 풀(search_pool_from_top3000.csv)에서 상위
       28개(POOL_N)만 사용하고, g1090(Gangnam)/sdata(Sinchon)를 로드해 임상점수·특징뱅크를 계산.
    2. 두 데이터셋 x 5개 운영점(S80~S90) 전체에 대해 풀의 28개 후보 각각의 하향조정 신호를
       미리 계산해둔다.
    3. 풀에서 크기 2~5의 모든 부분집합 x 과반수 이상 k값 조합에 대해, Gangnam/Sinchon/Pooled
       3가지 기준 각각의 성능을 요약(summarize)하고, Gangnam과 Sinchon 코호트 "각각 독립적으로"
       생존조건을 만족하는지(survives_both_cohorts)와 score로 순위를 매긴다.
    4. 전체/생존 규칙 요약표와, 생존 규칙들의 상세 성능표를 CSV로 저장하고 상위 15개 생존
       규칙을 콘솔에 출력.
    """
    pool = pd.read_csv(POOL_PATH).head(POOL_N).copy().reset_index(drop=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    y_by = {"Gangnam": g["y"].astype(int), "Sinchon": s["y"].astype(int)}
    cg, cs, _ = clinical_scores(g, s)
    c_by = {"Gangnam": cg, "Sinchon": cs}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], cg, target) for op, target in OPS}
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], cg, xg)
    x_by = {"Gangnam": xg * direction[None, :], "Sinchon": xs * direction[None, :]}
    name_to_idx = {name: i for i, name in enumerate(names)}

    sig_by: dict[tuple[str, str], np.ndarray] = {}
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset in ["Gangnam", "Sinchon"]:
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cpos
            sig = np.zeros((len(pool), len(y_by[dataset])), dtype=np.int8)
            for i, r in pool.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((c_by[dataset] - th) / float(r["width"])) ** 2)
                gate = c_by[dataset] + float(r["lambda"]) * boundary * x_by[dataset][:, idx]
                sig[i] = (cpos & (gate < th)).astype(np.int8)
            sig_by[(dataset, op)] = sig

    summaries = []
    detail_for_survivors = []
    for m in range(2, min(5, len(pool)) + 1):
        k_values = list(range((m + 1) // 2, m + 1))
        k_values = [k for k in k_values if k >= 2]
        for subset in itertools.combinations(range(len(pool)), m):
            feature_label = " + ".join(pool.iloc[list(subset)]["feature_short"].astype(str).tolist())
            for k in k_values:
                rule = f"{k}-of-{m}"
                rows = []
                for dataset in ["Gangnam", "Sinchon"]:
                    for op, _ in OPS:
                        votes = sig_by[(dataset, op)][list(subset)].sum(axis=0)
                        deesc = cpos_by[(dataset, op)] & (votes >= k)
                        rows.append(metric_row(dataset, rule, feature_label, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
                pooled_rows = []
                for op, _ in OPS:
                    y = np.concatenate([y_by["Gangnam"], y_by["Sinchon"]])
                    cpos = np.concatenate([cpos_by[("Gangnam", op)], cpos_by[("Sinchon", op)]])
                    deesc = np.concatenate(
                        [
                            cpos_by[("Gangnam", op)] & (sig_by[("Gangnam", op)][list(subset)].sum(axis=0) >= k),
                            cpos_by[("Sinchon", op)] & (sig_by[("Sinchon", op)][list(subset)].sum(axis=0) >= k),
                        ]
                    )
                    pooled_rows.append(metric_row("Pooled", rule, feature_label, op, y, cpos, deesc))
                rows_all = rows + pooled_rows
                summary = {
                    "rule": rule,
                    "features": feature_label,
                    "m": m,
                    "k": k,
                    **summarize(rows_all, "Gangnam"),
                    **summarize(rows_all, "Sinchon"),
                    **summarize(rows_all, "Pooled"),
                }
                summary["survives_both_cohorts"] = bool(summary["Gangnam_survives"] and summary["Sinchon_survives"])
                summary["score"] = (
                    2.0 * min(summary["Gangnam_min_spec_gain"], summary["Sinchon_min_spec_gain"])
                    + min(summary["Gangnam_mean_spec_gain"], summary["Sinchon_mean_spec_gain"])
                    + 0.5 * min(summary["Gangnam_min_delta_ba"], summary["Sinchon_min_delta_ba"])
                    - 0.2 * max(summary["Gangnam_max_sens_loss"], summary["Sinchon_max_sens_loss"])
                )
                summaries.append(summary)
                if summary["survives_both_cohorts"]:
                    detail_for_survivors.extend(rows_all)

    out = pd.DataFrame(summaries).sort_values(["survives_both_cohorts", "score"], ascending=False)
    out.to_csv(OUT_DIR / "fast_kofn_summary_pool28.csv", index=False)
    surv = out[out["survives_both_cohorts"]].copy()
    surv.to_csv(OUT_DIR / "fast_kofn_survivors_pool28.csv", index=False)
    pd.DataFrame(detail_for_survivors).to_csv(OUT_DIR / "fast_kofn_survivor_details_pool28.csv", index=False)
    print(surv.head(15)[[
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
