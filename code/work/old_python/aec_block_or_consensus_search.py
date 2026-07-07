from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import build_candidate_bank, clinical_scores, load_aec128, risk_direction, standardize_train_test  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_block_or_consensus_search"
SELECTED_CONFIG = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics" / "selected_individual_feature_configs.csv"
)
TOP3000_SUMMARY = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search" / "top3000_individual_screen_summary.csv"
)
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]

EARLY4 = [
    "norm_curv_010_025_max",
    "norm_slope_013_016_sd",
    "norm_curv_010_021_max",
    "norm_curv_007_010_min",
]
EARLY5 = EARLY4 + ["dct_log_17"]
MIDLATE4 = [
    "visual_trough_depth__early_041_056__mid_053_076__tail_101_128",
    "norm_slope_085_096_mean",
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
]


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
    return {
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (sens + spec),
        "accuracy": (tp + tn) / len(y),
    }


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def pvals(y: np.ndarray, cpos: np.ndarray, fpos: np.ndarray) -> dict:
    """임상단독 판정과 블록 OR/AND 규칙 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
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


def metric_row(dataset: str, rule: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    """한 데이터셋·규칙·운영점 조합에 대해 임상단독 대비 최종판정의 성능 변화와 하향조정군 사건비율·Fisher p값을 계산."""
    fpos = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, fpos)
    return {
        "dataset": dataset,
        "rule": rule,
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
    """지정된 데이터셋(Gangnam/Sinchon)에서 5개 운영점에 걸친 metric_row 결과를 모아 각 지표의 최솟값/최댓값/평균을 계산."""
    sub = [r for r in rows if r["dataset"] == dataset]
    p_loss = np.asarray([r["sensitivity_loss_p_exact"] for r in sub], dtype=float)
    spec_gain = np.asarray([r["specificity_gain"] for r in sub], dtype=float)
    sens_loss = np.asarray([r["sensitivity_loss"] for r in sub], dtype=float)
    delta_ba = np.asarray([r["delta_balanced_accuracy"] for r in sub], dtype=float)
    fisher_p = np.asarray([r["deesc_event_fisher_p"] for r in sub], dtype=float)
    return {
        f"{dataset}_min_p_loss": float(np.nanmin(p_loss)) if np.isfinite(p_loss).any() else 0.0,
        f"{dataset}_max_sens_loss": float(np.nanmax(sens_loss)),
        f"{dataset}_min_spec_gain": float(np.nanmin(spec_gain)),
        f"{dataset}_mean_spec_gain": float(np.nanmean(spec_gain)),
        f"{dataset}_min_delta_ba": float(np.nanmin(delta_ba)),
        f"{dataset}_mean_delta_ba": float(np.nanmean(delta_ba)),
        f"{dataset}_max_fisher_p": float(np.nanmax(fisher_p)) if np.isfinite(fisher_p).any() else 1.0,
    }


def load_feature_config() -> pd.DataFrame:
    """aec_individual_feature_full_metrics와 aec_top3000_kofn_fine_search가 저장해둔 설정표에서, EARLY5+MIDLATE4 특징들의 폭·람다 설정을 모아 하나의 표로 합친다."""
    selected = pd.read_csv(SELECTED_CONFIG)[["label", "feature", "width", "lambda"]].rename(columns={"label": "feature_short"})
    top = pd.read_csv(TOP3000_SUMMARY)[["feature_short", "feature", "width", "lambda"]]
    cfg = pd.concat([selected, top], ignore_index=True)
    wanted = EARLY5 + MIDLATE4
    cfg = cfg[cfg["feature_short"].isin(wanted)].drop_duplicates("feature_short")
    missing = sorted(set(wanted) - set(cfg["feature_short"]))
    if missing:
        raise RuntimeError(f"Missing feature configs: {missing}")
    return cfg


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "초반 구간" 특징 블록(EARLY4/EARLY5)과 "중후반 구간"
    특징 블록(MIDLATE4)을 각각 투표 형태로 묶은 뒤, 두 블록을 OR 또는 AND로 결합하면 단일
    블록 규칙보다 더 넓은 범위에서 안전하게 살아남는 하향조정 규칙을 찾을 수 있는가?):

    1. load_feature_config로 EARLY5+MIDLATE4 총 9개 특징의 폭·람다 설정을 모으고, g1090
       (Gangnam)/sdata(Sinchon)를 로드해 임상점수·특징뱅크·표준화값을 계산한 뒤, Gangnam 기준
       5개 운영점(S80~S90)의 임상 임계값을 구한다.
    2. 두 데이터셋 x 5개 운영점 전체에 대해 9개 특징 각각의 하향조정 신호를 미리 계산해둔다.
    3. early_block(early4 또는 early5) x early 투표 임계값(n) x midlate 투표 임계값(o) x
       결합방식(OR/AND) 전체 조합에 대해, "early 블록 투표 >= n"과 "midlate 블록 투표 >= o"를
       OR 또는 AND로 합쳐 하향조정 여부를 정하고, Gangnam·Sinchon 각각의 성능을 요약한 뒤
       두 코호트 모두에서 생존조건(민감도손실 p≥0.05, 특이도이득>0, Fisher p<0.05)을
       만족하는지와 score로 순위를 매긴다.
    4. 전체 규칙 요약표, 생존 규칙의 상세 성능표, 규칙별 특징 정의표를 CSV로 저장하고, 생존한
       규칙 상위 20개를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_feature_config()
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

    signals: dict[tuple[str, str, str], np.ndarray] = {}
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset in ["Gangnam", "Sinchon"]:
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cpos
            for _, r in cfg.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((c_by[dataset] - th) / float(r["width"])) ** 2)
                gate = c_by[dataset] + float(r["lambda"]) * boundary * x_by[dataset][:, idx]
                signals[(dataset, op, str(r["feature_short"]))] = cpos & (gate < th)

    summary_rows = []
    detail_rows = []
    rule_def_rows = []
    block_defs = [("early4", EARLY4), ("early5", EARLY5)]
    for early_name, early_features in block_defs:
        mid_features = MIDLATE4
        for n in range(1, len(early_features) + 1):
            for o in range(1, len(mid_features) + 1):
                for mode in ["OR", "AND"]:
                    rule = f"{early_name}>={n} {mode} midlate4>={o}"
                    rows = []
                    for dataset in ["Gangnam", "Sinchon"]:
                        for op, _ in OPS:
                            early_votes = np.column_stack([signals[(dataset, op, f)] for f in early_features]).sum(axis=1)
                            mid_votes = np.column_stack([signals[(dataset, op, f)] for f in mid_features]).sum(axis=1)
                            if mode == "OR":
                                deesc = cpos_by[(dataset, op)] & ((early_votes >= n) | (mid_votes >= o))
                            else:
                                deesc = cpos_by[(dataset, op)] & ((early_votes >= n) & (mid_votes >= o))
                            rows.append(metric_row(dataset, rule, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
                    gsum = summarize(rows, "Gangnam")
                    ssum = summarize(rows, "Sinchon")
                    survives = (
                        gsum["Gangnam_min_p_loss"] >= 0.05
                        and ssum["Sinchon_min_p_loss"] >= 0.05
                        and gsum["Gangnam_min_spec_gain"] > 0
                        and ssum["Sinchon_min_spec_gain"] > 0
                        and gsum["Gangnam_max_fisher_p"] < 0.05
                        and ssum["Sinchon_max_fisher_p"] < 0.05
                    )
                    score = (
                        2.0 * min(gsum["Gangnam_min_spec_gain"], ssum["Sinchon_min_spec_gain"])
                        + min(gsum["Gangnam_mean_spec_gain"], ssum["Sinchon_mean_spec_gain"])
                        + 0.5 * min(gsum["Gangnam_min_delta_ba"], ssum["Sinchon_min_delta_ba"])
                        - 0.2 * max(gsum["Gangnam_max_sens_loss"], ssum["Sinchon_max_sens_loss"])
                    )
                    summary_rows.append(
                        {
                            "rule": rule,
                            "early_block": early_name,
                            "early_n": n,
                            "midlate_o": o,
                            "mode": mode,
                            "survives_both": survives,
                            "score": score,
                            **gsum,
                            **ssum,
                        }
                    )
                    rule_def_rows.append(
                        {
                            "rule": rule,
                            "early_features": " + ".join(early_features),
                            "midlate_features": " + ".join(mid_features),
                        }
                    )
                    if survives:
                        detail_rows.extend(rows)

    summary = pd.DataFrame(summary_rows).sort_values(["survives_both", "score"], ascending=False)
    summary.to_csv(OUT_DIR / "block_or_and_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "block_or_and_survivor_details.csv", index=False)
    pd.DataFrame(rule_def_rows).drop_duplicates("rule").to_csv(OUT_DIR / "block_rule_definitions.csv", index=False)
    print(summary[summary["survives_both"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
