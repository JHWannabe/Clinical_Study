from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import build_candidate_bank, clinical_scores, load_aec128  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_sex_specific_feature_search"
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55]
TOP_UNIVARIATE_N = 140
POOL_N = 16
MAX_M = 3
SEED = 20260630
ALWAYS_INCLUDE_SHORTS = {
    "visual_trough_depth__early_041_056__mid_053_076__tail_101_128",
    "norm_slope_085_096_mean",
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
    "norm_slope_013_016_sd",
    "norm_curv_007_010_min",
    "dct_log_17",
}


def short_name(name: str) -> str:
    """특징 이름 앞의 "bank_norm__"/"midrange__" 접두어를 제거해 사람이 읽기 쉬운 짧은 이름으로 변환."""
    return name.replace("bank_norm__", "").replace("midrange__", "")


def standardize_by_train_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """train 성별 부분집합만으로 원본(raw) 특징을 제외한 정규화 형태 특징들의 결측대체값·상하한 클리핑·표준화 파라미터를 정하고, train/test 부분집합 양쪽에 동일하게 적용."""
    # Keep normalized shape features only. Raw AEC level/range can be scanner/protocol dominated.
    cols = [
        c
        for c in train_df.columns
        if not short_name(str(c)).startswith("raw_")
        and not short_name(str(c)).startswith("rawlog_")
        and "raw_" not in short_name(str(c))
    ]
    xtr = train_df.loc[train_mask, cols].to_numpy(dtype=float)
    xte = test_df.loc[test_mask, cols].to_numpy(dtype=float)
    med = np.nanmedian(xtr, axis=0)
    med[~np.isfinite(med)] = 0.0
    xtr = np.where(np.isfinite(xtr), xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    lo = np.nanquantile(xtr, 0.01, axis=0)
    hi = np.nanquantile(xtr, 0.99, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    xtr[:, ok] = np.clip(xtr[:, ok], lo[ok], hi[ok])
    xte[:, ok] = np.clip(xte[:, ok], lo[ok], hi[ok])
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    keep = np.isfinite(sd) & (sd > 1e-10)
    xtr = (xtr[:, keep] - mu[keep]) / sd[keep]
    xte = (xte[:, keep] - mu[keep]) / sd[keep]
    names = [str(c) for c, k in zip(cols, keep) if k]
    return xtr, xte, names


def risk_direction(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상점수의 예측 잔차와 각 특징의 상관 부호를 구해, 값이 클수록 위험이 커지도록 특징별 방향(+1/-1)을 정한다(성별 부분집합에서 별도로 계산)."""
    y = y.astype(int)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)
        model.fit(clinical_z.reshape(-1, 1), y)
        pred = model.predict_proba(clinical_z.reshape(-1, 1))[:, 1]
        resid = y.astype(float) - pred
    except Exception:
        resid = y.astype(float) - y.mean()
    score = x.T @ resid
    direction = np.sign(score)
    fallback = np.sign(x.T @ (y.astype(float) - y.mean()))
    direction[direction == 0] = fallback[direction == 0]
    direction[direction == 0] = 1.0
    return direction.astype(float)


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
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
    }


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 게이트적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값과 놓친/제거된 표본수를 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
        "tp_lost_n": sens_loss,
        "fp_removed_n": spec_gain,
    }


def fisher_p(y: np.ndarray, final_pos: np.ndarray, deesc: np.ndarray) -> float:
    """최종 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[final_pos] == 1))
    b = int(np.sum(y[final_pos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def metric_row(dataset: str, sex: str, rule: str, features: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    """한 데이터셋·성별·규칙·운영점 조합에 대해 임상단독 대비 최종판정의 성능 변화와 하향조정군 사건비율·Fisher p값을 계산."""
    fpos = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, fpos)
    return {
        "dataset": dataset,
        "sex": sex,
        "rule": rule,
        "features": features,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_p(y, fpos, deesc),
        **paired_pvalues(y, cpos, fpos),
    }


def summarize(rows: list[dict], dataset: str) -> dict:
    """지정된 데이터셋(Gangnam train 또는 Sinchon external)에서 5개 운영점에 걸친 metric_row 결과를 모아 각 지표의 최솟값/최댓값/평균과 생존조건(민감도손실 p≥0.05, 특이도이득>0) 충족 여부를 계산."""
    sub = [r for r in rows if r["dataset"] == dataset]
    p_loss = np.asarray([r["sensitivity_loss_p_exact"] for r in sub], dtype=float)
    spec_gain = np.asarray([r["specificity_gain"] for r in sub], dtype=float)
    sens_loss = np.asarray([r["sensitivity_loss"] for r in sub], dtype=float)
    delta_ba = np.asarray([r["delta_balanced_accuracy"] for r in sub], dtype=float)
    fisher = np.asarray([r["deesc_event_fisher_p"] for r in sub], dtype=float)
    deesc_rate = np.asarray([r["deesc_event_rate"] for r in sub], dtype=float)
    return {
        f"{dataset}_min_p_loss": float(np.nanmin(p_loss)),
        f"{dataset}_max_sens_loss": float(np.nanmax(sens_loss)),
        f"{dataset}_min_spec_gain": float(np.nanmin(spec_gain)),
        f"{dataset}_mean_spec_gain": float(np.nanmean(spec_gain)),
        f"{dataset}_min_delta_ba": float(np.nanmin(delta_ba)),
        f"{dataset}_mean_delta_ba": float(np.nanmean(delta_ba)),
        f"{dataset}_max_fisher_p": float(np.nanmax(fisher)),
        f"{dataset}_mean_deesc_event_rate": float(np.nanmean(deesc_rate)),
        f"{dataset}_survives": bool(np.nanmin(p_loss) >= 0.05 and np.nanmin(spec_gain) > 0),
    }


def individual_candidate_pool(
    sex: str,
    y: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    names: list[str],
    direction: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """지정된 성별 train 부분집합에서, 결과와의 상관크기로 빠르게 상위 140개(+항상 포함할 특징들) 후보를 추리고, 각 후보에 대해 폭 3종 x 람다 3종 조합 중 candidate_score가 가장 좋은 설정을 골라 게이트 성능이 양호한 특징들로 구성된 상위 16개 탐색 풀을 만든다."""
    cpos_by = {op: c >= th for op, th in thresholds.items()}
    rows = []
    # Fast pre-screen: rank shape features by sex-specific marginal association.
    # The final gate metrics still use clinical-positive de-escalation, not this score.
    assoc = np.abs(x.T @ (y.astype(float) - y.mean())) / max(1, len(y))
    order = list(np.argsort(assoc)[::-1][:TOP_UNIVARIATE_N])
    for idx, name in enumerate(names):
        if short_name(name) in ALWAYS_INCLUDE_SHORTS and idx not in order:
            order.append(idx)

    for idx in order:
        name = names[idx]
        xr = x[:, idx] * direction[idx]
        best = None
        for width in WIDTHS:
            for lam in LAMBDAS:
                metric_rows = []
                for op, th in thresholds.items():
                    cpos = cpos_by[op]
                    boundary = np.exp(-0.5 * ((c - th) / width) ** 2)
                    gate = c + lam * boundary * xr
                    deesc = cpos & (gate < th)
                    metric_rows.append(metric_row("Gangnam train", sex, "1-of-1", short_name(name), op, y, cpos, deesc))
                sm = summarize(metric_rows, "Gangnam train")
                # Soft score for candidate pool. Do not require individual survival;
                # consensus can rescue sensitivity loss.
                score = (
                    2.0 * sm["Gangnam train_min_spec_gain"]
                    + sm["Gangnam train_mean_spec_gain"]
                    + 0.5 * sm["Gangnam train_min_delta_ba"]
                    - 0.30 * sm["Gangnam train_max_sens_loss"]
                    - 0.01 * max(0.0, np.log10(max(sm["Gangnam train_max_fisher_p"], 1e-12)) + 2.0)
                )
                if sm["Gangnam train_min_spec_gain"] <= 0:
                    score -= 2.0
                if sm["Gangnam train_min_p_loss"] < 0.01:
                    score -= 1.0
                candidate = {
                    "sex": sex,
                    "feature": name,
                    "feature_short": short_name(name),
                    "idx": idx,
                    "width": width,
                    "lambda": lam,
                    "direction": direction[idx],
                    "candidate_score": float(score),
                    **sm,
                }
                if best is None or candidate["candidate_score"] > best["candidate_score"]:
                    best = candidate
        if best is not None:
            rows.append(best)
    cand = pd.DataFrame(rows)
    near = cand[
        (cand["Gangnam train_min_spec_gain"] > 0.0)
        & (cand["Gangnam train_max_sens_loss"] <= 0.10)
        & (cand["Gangnam train_mean_spec_gain"] > 0.01)
    ].copy()
    if len(near) < POOL_N:
        near = cand.sort_values("candidate_score", ascending=False).head(max(POOL_N, len(near))).copy()
    return near.sort_values("candidate_score", ascending=False).head(POOL_N).reset_index(drop=True)


def eval_rule(
    sex: str,
    rule: str,
    feature_label: str,
    subset: list[int],
    k: int,
    datasets: dict[str, dict],
    cpos_by: dict[tuple[str, str], np.ndarray],
    sig_by: dict[tuple[str, str], np.ndarray],
) -> list[dict]:
    """선택된 특징 부분집합(subset)에 대해 k-of-m 투표 규칙으로 하향조정 여부를 정하고, train/external 두 데이터셋 x 5개 운영점 전체의 성능 행을 계산."""
    rows = []
    for dataset_name, obj in datasets.items():
        y = obj["y"]
        for op, _ in OPS:
            votes = sig_by[(dataset_name, op)][subset].sum(axis=0)
            deesc = cpos_by[(dataset_name, op)] & (votes >= k)
            rows.append(metric_row(dataset_name, sex, rule, feature_label, op, y, cpos_by[(dataset_name, op)], deesc))
    return rows


def search_sex_specific(
    sex: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    g: dict,
    s: dict,
    fg: pd.DataFrame,
    fs: pd.DataFrame,
    cg: np.ndarray,
    cs: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """한 성별에 대해 전체 탐색 파이프라인을 실행: train 부분집합만으로 표준화·방향·임계값을 정하고, individual_candidate_pool로 16개 특징 풀을 만든 뒤 크기 1~3의 모든 부분집합 x k값 조합에 대해 train_selection_score로 순위를 매겨 상위 20개 규칙의 상세 성능을 계산하고, 후보풀·전체요약·상세표를 반환."""
    xtr, xte, names = standardize_by_train_subset(fg, fs, train_mask, test_mask)
    ytr = g["y"].astype(int)[train_mask]
    yte = s["y"].astype(int)[test_mask]
    ctr = cg[train_mask]
    cte = cs[test_mask]
    direction = risk_direction(ytr, ctr, xtr)
    thresholds = {op: threshold_for_min_sensitivity(ytr, ctr, target) for op, target in OPS}

    pool = individual_candidate_pool(sex, ytr, ctr, xtr, names, direction, thresholds)
    pool.to_csv(OUT_DIR / f"{sex.lower()}_candidate_pool.csv", index=False)

    datasets = {
        "Gangnam train": {"y": ytr, "c": ctr, "x": xtr},
        "Sinchon external": {"y": yte, "c": cte, "x": xte},
    }
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    sig_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset_name, obj in datasets.items():
        for op, th in thresholds.items():
            cpos = obj["c"] >= th
            cpos_by[(dataset_name, op)] = cpos
            sig = np.zeros((len(pool), len(obj["y"])), dtype=np.int8)
            for i, r in pool.iterrows():
                idx = int(r["idx"])
                width = float(r["width"])
                lam = float(r["lambda"])
                xr = obj["x"][:, idx] * float(r["direction"])
                boundary = np.exp(-0.5 * ((obj["c"] - th) / width) ** 2)
                gate = obj["c"] + lam * boundary * xr
                sig[i] = (cpos & (gate < th)).astype(np.int8)
            sig_by[(dataset_name, op)] = sig

    summaries = []
    detail_rows = []
    for m in range(1, min(MAX_M, len(pool)) + 1):
        k_values = [1] if m == 1 else list(range((m + 1) // 2, m + 1))
        k_values = [k for k in k_values if k >= 2 or m == 1]
        for subset in itertools.combinations(range(len(pool)), m):
            feature_label = " + ".join(pool.iloc[list(subset)]["feature_short"].astype(str).tolist())
            for k in k_values:
                rule = f"{k}-of-{m}"
                rows = eval_rule(sex, rule, feature_label, list(subset), k, datasets, cpos_by, sig_by)
                train_sum = summarize(rows, "Gangnam train")
                external_sum = summarize(rows, "Sinchon external")
                summary = {
                    "sex": sex,
                    "rule": rule,
                    "features": feature_label,
                    "m": m,
                    "k": k,
                    **train_sum,
                    **external_sum,
                }
                summary["train_selection_score"] = (
                    2.0 * train_sum["Gangnam train_min_spec_gain"]
                    + train_sum["Gangnam train_mean_spec_gain"]
                    + 0.5 * train_sum["Gangnam train_min_delta_ba"]
                    - 0.25 * train_sum["Gangnam train_max_sens_loss"]
                    - 0.02 * min(train_sum["Gangnam train_max_fisher_p"], 1.0)
                )
                if train_sum["Gangnam train_min_p_loss"] < 0.05:
                    summary["train_selection_score"] -= 5.0
                if train_sum["Gangnam train_min_spec_gain"] <= 0:
                    summary["train_selection_score"] -= 5.0
                if train_sum["Gangnam train_max_fisher_p"] > 0.20:
                    summary["train_selection_score"] -= 0.1
                summary["external_survives"] = external_sum["Sinchon external_survives"]
                summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values("train_selection_score", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / f"{sex.lower()}_combo_summary.csv", index=False)

    top = summary_df.head(20)
    for _, row in top.iterrows():
        feat_list = str(row["features"]).split(" + ")
        lookup = {f: i for i, f in enumerate(pool["feature_short"].astype(str))}
        subset = [lookup[f] for f in feat_list]
        k = int(str(row["rule"]).split("-of-")[0])
        detail_rows.extend(eval_rule(sex, str(row["rule"]), str(row["features"]), subset, k, datasets, cpos_by, sig_by))
    details = pd.DataFrame(detail_rows)
    details.to_csv(OUT_DIR / f"{sex.lower()}_top20_details.csv", index=False)
    return pool, summary_df, details


def semantic_label(feature: str) -> str:
    """특징 이름의 키워드(trough_depth, slope, haar, curv, level, dct/fft 등)를 보고 사람이 이해하기 쉬운 해부학적/형태학적 의미 라벨을 붙인다."""
    f = feature.lower()
    if "trough_depth" in f:
        return "mid-abdominal trough / rebound contrast"
    if "slope" in f and any(s in f for s in ["085", "088", "091", "094", "097", "100", "103", "106", "109"]):
        return "upper-abdominal recovery or tail slope"
    if "slope" in f:
        return "local slope / transition"
    if "haar" in f:
        return "local step/edge contrast"
    if "curv" in f:
        return "local curvature / bend"
    if "level" in f:
        return "regional normalized level"
    if "dct" in f or "fft" in f:
        return "spectral waviness"
    return "shape descriptor"


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_sex_subgroup_gate_performance가 "공통" 게이트를
    성별로 나눠 평가했다면, 이 스크립트는 반대로 남성과 여성 각각에 대해 처음부터 완전히
    독립적으로 특징을 탐색하고 게이트를 최적화하면, 성별 특이적으로 더 잘 맞는 하향조정 규칙을
    찾을 수 있는가?):

    1. g1090(train)/sdata(external)를 로드하고 임상점수를 계산한 뒤, 환자 성별(PatientSex)로
       여성/남성 부분집합을 나눈다.
    2. 각 성별에 대해 search_sex_specific을 실행: 그 성별의 train 부분집합만으로 특징을
       표준화·방향고정하고, 16개 후보 특징 풀을 구성한 뒤 크기 1~3의 모든 조합 x k값에 대해
       train에서 가장 좋은 하향조정 규칙(train_selection_score 기준)을 찾고, external에서도
       생존하는지 확인.
    3. 각 성별의 최상위 규칙과 그 규칙을 구성하는 특징들의 의미론적 라벨(semantic_label)을
       수집해 CSV로 저장.
    4. 성별별 최상위 규칙 비교표와 의미론적 라벨표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    cg, cs, _ = clinical_scores(g, s)
    sex_g = g["meta"]["PatientSex"].astype(str).str.upper().to_numpy()
    sex_s = s["meta"]["PatientSex"].astype(str).str.upper().to_numpy()

    all_rows = []
    semantic_rows = []
    for sex, code in [("Female", "F"), ("Male", "M")]:
        train_mask = sex_g == code
        test_mask = sex_s == code
        pool, summary, details = search_sex_specific(sex, train_mask, test_mask, g, s, fg, fs, cg, cs)
        top = summary.iloc[0].to_dict()
        all_rows.append(top)
        for f in str(top["features"]).split(" + "):
            semantic_rows.append({"sex": sex, "selected_feature": f, "semantic_label": semantic_label(f)})

    top_df = pd.DataFrame(all_rows)
    top_df.to_csv(OUT_DIR / "sex_specific_train_selected_top_rules.csv", index=False)
    pd.DataFrame(semantic_rows).to_csv(OUT_DIR / "sex_specific_top_rule_semantics.csv", index=False)

    print("\nSEX-SPECIFIC TRAIN-SELECTED TOP RULES")
    cols = [
        "sex",
        "rule",
        "features",
        "Gangnam train_min_p_loss",
        "Gangnam train_max_sens_loss",
        "Gangnam train_min_spec_gain",
        "Gangnam train_mean_spec_gain",
        "Gangnam train_max_fisher_p",
        "Sinchon external_min_p_loss",
        "Sinchon external_max_sens_loss",
        "Sinchon external_min_spec_gain",
        "Sinchon external_mean_spec_gain",
        "Sinchon external_max_fisher_p",
        "external_survives",
    ]
    print(top_df[cols].to_string(index=False))
    print("\nSEMANTIC LABELS")
    print(pd.DataFrame(semantic_rows).to_string(index=False))


if __name__ == "__main__":
    main()
