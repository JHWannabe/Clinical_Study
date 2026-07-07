from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aec_conditional_value import (
    DATA_DIR,
    aec_estimator,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    make_folds,
    oof_and_external,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "clinical90_deescalation"
SEED = 20260629
RNG = np.random.default_rng(SEED)


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> tuple[float, dict]:
    """Highest score threshold that keeps training sensitivity at least target."""
    rows = []
    for th in np.unique(score):
        pred = score >= th
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum(~pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = int(np.sum(~pred & (y == 0)))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        rows.append((float(th), sens, spec, tp, fp, fn, tn))

    feasible = [r for r in rows if r[1] >= target]
    if not feasible:
        chosen = min(rows, key=lambda r: abs(r[1] - target))
    else:
        chosen = max(feasible, key=lambda r: r[0])

    th, sens, spec, tp, fp, fn, tn = chosen
    detail = {
        "target_sensitivity": float(target),
        "threshold": float(th),
        "achieved_train_sensitivity": float(sens),
        "achieved_train_specificity": float(spec),
        "train_tp": int(tp),
        "train_fp": int(fp),
        "train_fn": int(fn),
        "train_tn": int(tn),
    }
    return float(th), detail


def wilson_ci(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간을 계산."""
    if n <= 0:
        return np.nan, np.nan
    p = events / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def summarize_mask(y: np.ndarray, mask: np.ndarray) -> dict:
    """mask로 선택된 부분집합의 표본수·이벤트수·사건율(Wilson 신뢰구간 포함)·음성예측도(NPV)를 계산."""
    n = int(np.sum(mask))
    events = int(np.sum(y[mask]))
    nonevents = int(n - events)
    lo, hi = wilson_ci(events, n)
    return {
        "n": n,
        "events": events,
        "non_events": nonevents,
        "event_rate": float(events / n) if n else np.nan,
        "event_rate_wilson_95ci_low": lo,
        "event_rate_wilson_95ci_high": hi,
        "npv_if_negative": float(nonevents / n) if n else np.nan,
    }


def paired_error_changes(y: np.ndarray, clinical_pos: np.ndarray, gated_pos: np.ndarray) -> dict:
    """임상 양성 판정을 AEC 게이트로 바꿨을 때 새로 생기는/없어지는 위양성(FP)과 놓치는/얻는 진양성(TP)
    개수를 세고, 각각 대응 이항검정(binomial exact test)으로 유의성을 계산."""
    non_event = y == 0
    event = y == 1

    fp_removed = int(np.sum(clinical_pos & ~gated_pos & non_event))
    fp_added = int(np.sum(~clinical_pos & gated_pos & non_event))
    tp_lost = int(np.sum(clinical_pos & ~gated_pos & event))
    tp_gained = int(np.sum(~clinical_pos & gated_pos & event))

    fp_total = fp_removed + fp_added
    tp_total = tp_lost + tp_gained
    return {
        "false_positives_removed": fp_removed,
        "false_positives_added": fp_added,
        "false_positive_net_change_gated_minus_clinical": fp_added - fp_removed,
        "false_positive_paired_exact_p": float(stats.binomtest(min(fp_removed, fp_added), fp_total, 0.5).pvalue)
        if fp_total
        else np.nan,
        "true_positives_lost": tp_lost,
        "true_positives_gained": tp_gained,
        "true_positive_net_change_gated_minus_clinical": tp_gained - tp_lost,
        "true_positive_paired_exact_p": float(stats.binomtest(min(tp_lost, tp_gained), tp_total, 0.5).pvalue)
        if tp_total
        else np.nan,
    }


def reclassification_table(y: np.ndarray, clinical_pos: np.ndarray, gated_pos: np.ndarray) -> pd.DataFrame:
    """임상 양성/음성 x AEC 게이트 양성/음성 4개 그룹별로 표본수·사건율 요약을 표로 만듦."""
    rows = []
    for label, mask in [
        ("clinical+ / AEC-kept+", clinical_pos & gated_pos),
        ("clinical+ / AEC-deescalated", clinical_pos & ~gated_pos),
        ("clinical- / AEC-escalated", ~clinical_pos & gated_pos),
        ("clinical- / AEC-kept-", ~clinical_pos & ~gated_pos),
    ]:
        rows.append({"group": label, **summarize_mask(y, mask)})
    return pd.DataFrame(rows)


def conditional_stats(y: np.ndarray, clinical_pos: np.ndarray, gated_pos: np.ndarray) -> dict:
    """임상 양성군을 "AEC로 유지(kept)"와 "AEC로 하향조정(deescalated)"으로 나눠 사건율 차이와 Fisher 정확검정을 계산."""
    kept = clinical_pos & gated_pos
    deesc = clinical_pos & ~gated_pos

    kept_sum = summarize_mask(y, kept)
    deesc_sum = summarize_mask(y, deesc)
    table = np.array(
        [
            [kept_sum["events"], kept_sum["non_events"]],
            [deesc_sum["events"], deesc_sum["non_events"]],
        ],
        dtype=int,
    )
    fisher_or, fisher_p = stats.fisher_exact(table)
    return {
        "clinical_positive_aec_kept": kept_sum,
        "clinical_positive_aec_deescalated": deesc_sum,
        "absolute_event_rate_difference_kept_minus_deescalated": float(kept_sum["event_rate"] - deesc_sum["event_rate"]),
        "aec_kept_vs_deescalated_fisher_exact_or": float(fisher_or),
        "aec_kept_vs_deescalated_fisher_exact_p": float(fisher_p),
        "table_rows": ["clinical+_AEC_kept", "clinical+_AEC_deescalated"],
        "table_cols": ["low_smi_event", "non_event"],
        "table": table.tolist(),
    }


def make_scores(target: float, data: dict) -> dict:
    """목표 민감도(target)에서 train 임계값을 정하고, 그 임계값을 고정한 채 가우시안/하드존 두 여성
    AEC 게이트를 train·외부 데이터 각각에 적용해 지표·조건부통계·재분류표를 모두 계산."""
    ytr = data["ytr"]
    t_raw, threshold_detail = threshold_for_min_sensitivity(ytr, data["clinical_oof"], target)
    c_z, c_te_z = data["c_z"], data["c_te_z"]
    a_z, a_te_z = data["a_z"], data["a_te_z"]
    t_z = (t_raw - data["clinical_oof_mean"]) / data["clinical_oof_sd"]

    out = {
        "threshold_detail": {**threshold_detail, "threshold_z": float(t_z)},
        "train": {},
        "external": {},
    }
    for split, y, cscore, ascore, female in [
        ("train", data["ytr"], c_z, a_z, data["female_tr"]),
        ("external", data["yte"], c_te_z, a_te_z, data["female_te"]),
    ]:
        clinical_pos = cscore >= t_z
        boundary = np.exp(-0.5 * ((cscore - t_z) / 0.75) ** 2)
        gaussian_score = cscore + 0.25 * female * boundary * ascore
        gaussian_pos = gaussian_score >= t_z

        hard_zone = (np.abs(cscore - t_z) <= 0.50).astype(float)
        hard_score = cscore + 0.25 * female * hard_zone * ascore
        hard_pos = hard_score >= t_z

        out[split] = {
            "clinical_metrics": binary_metrics(y, cscore, t_z),
            "gaussian_metrics_same_threshold": binary_metrics(y, gaussian_score, t_z),
            "hard_zone_metrics_same_threshold": binary_metrics(y, hard_score, t_z),
            "gaussian_conditional": conditional_stats(y, clinical_pos, gaussian_pos),
            "hard_zone_conditional": conditional_stats(y, clinical_pos, hard_pos),
            "gaussian_paired_error_changes": paired_error_changes(y, clinical_pos, gaussian_pos),
            "hard_zone_paired_error_changes": paired_error_changes(y, clinical_pos, hard_pos),
            "gaussian_reclassification": reclassification_table(y, clinical_pos, gaussian_pos),
            "hard_zone_reclassification": reclassification_table(y, clinical_pos, hard_pos),
        }
    return out


def bootstrap_external_90(y: np.ndarray, cscore: np.ndarray, ascore: np.ndarray, female: np.ndarray, t_z: float, n_boot: int = 5000) -> dict:
    """90% 민감도 임계값 고정 하에, 하향조정군/유지군 사건율과 민감도손실·특이도이득·FP/TP 변화를 5000회 부트스트랩으로 신뢰구간과 함께 추정."""
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yy = y[idx]
        cc = cscore[idx]
        aa = ascore[idx]
        ff = female[idx]
        clinical_pos = cc >= t_z
        boundary = np.exp(-0.5 * ((cc - t_z) / 0.75) ** 2)
        gated_score = cc + 0.25 * ff * boundary * aa
        gated_pos = gated_score >= t_z
        clinical_m = binary_metrics(yy, cc, t_z)
        gated_m = binary_metrics(yy, gated_score, t_z)
        deesc = clinical_pos & ~gated_pos
        kept = clinical_pos & gated_pos
        deesc_sum = summarize_mask(yy, deesc)
        kept_sum = summarize_mask(yy, kept)
        changes = paired_error_changes(yy, clinical_pos, gated_pos)
        rows.append(
            {
                "deescalated_n": deesc_sum["n"],
                "deescalated_event_rate": deesc_sum["event_rate"],
                "kept_event_rate": kept_sum["event_rate"],
                "event_rate_difference_kept_minus_deescalated": kept_sum["event_rate"] - deesc_sum["event_rate"],
                "sensitivity_loss": clinical_m["sensitivity"] - gated_m["sensitivity"],
                "specificity_gain": gated_m["specificity"] - clinical_m["specificity"],
                "ppv_gain": gated_m["ppv"] - clinical_m["ppv"],
                "fp_removed": changes["false_positives_removed"],
                "fp_added": changes["false_positives_added"],
                "tp_lost": changes["true_positives_lost"],
                "tp_gained": changes["true_positives_gained"],
            }
        )

    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        out[col] = {
            "mean": float(np.mean(vals)),
            "ci2.5": float(np.quantile(vals, 0.025)),
            "ci97.5": float(np.quantile(vals, 0.975)),
        }
    return out


def flatten_result_rows(target_results: dict) -> pd.DataFrame:
    """여러 민감도 목표(target)별 결과 딕셔너리를, (목표, 데이터분할, 모델)별 지표 한 행씩으로 펼침."""
    rows = []
    for target_label, result in target_results.items():
        target = float(target_label)
        for split in ["train", "external"]:
            for model_key in ["clinical_metrics", "gaussian_metrics_same_threshold", "hard_zone_metrics_same_threshold"]:
                row = {"target_sensitivity": target, "split": split, "model": model_key}
                row.update(result[split][model_key])
                rows.append(row)
    return pd.DataFrame(rows)


def conditional_rows(target_results: dict) -> pd.DataFrame:
    """여러 민감도 목표·게이트 종류별 조건부 통계(유지 vs 하향조정 그룹 비교)를 한 행씩 펼쳐 표로 만듦."""
    rows = []
    for target_label, result in target_results.items():
        target = float(target_label)
        for split in ["train", "external"]:
            for gate_name in ["gaussian", "hard_zone"]:
                cond = result[split][f"{gate_name}_conditional"]
                for group_key, group_label in [
                    ("clinical_positive_aec_kept", "clinical+ / AEC-kept+"),
                    ("clinical_positive_aec_deescalated", "clinical+ / AEC-deescalated"),
                ]:
                    rows.append(
                        {
                            "target_sensitivity": target,
                            "split": split,
                            "gate": gate_name,
                            "group": group_label,
                            **cond[group_key],
                            "kept_minus_deescalated_event_rate_difference": cond[
                                "absolute_event_rate_difference_kept_minus_deescalated"
                            ],
                            "fisher_or": cond["aec_kept_vs_deescalated_fisher_exact_or"],
                            "fisher_p": cond["aec_kept_vs_deescalated_fisher_exact_p"],
                        }
                    )
    return pd.DataFrame(rows)


def save_reclassification_tables(target_results: dict) -> None:
    """민감도 목표/데이터분할/게이트 종류별 재분류표를 각각 CSV 파일로 저장."""
    for target_label, result in target_results.items():
        pct = int(round(float(target_label) * 100))
        for split in ["train", "external"]:
            for gate_name in ["gaussian", "hard_zone"]:
                df = result[split][f"{gate_name}_reclassification"]
                df.insert(0, "target_sensitivity", float(target_label))
                df.insert(1, "split", split)
                df.insert(2, "gate", gate_name)
                df.to_csv(OUT_DIR / f"{split}_{pct}_{gate_name}_reclassification.csv", index=False)


def plot_external_90(target_results: dict) -> None:
    """90% 민감도 목표에서 외부 데이터의 (AEC 유지 vs 하향조정) 사건율 막대그래프와, 재분류 흐름(막대) 그래프를 PNG로 저장."""
    result = target_results["0.9"]
    cond = result["external"]["gaussian_conditional"]
    labels = ["AEC-kept+", "AEC-deescalated"]
    groups = [cond["clinical_positive_aec_kept"], cond["clinical_positive_aec_deescalated"]]
    rates = [g["event_rate"] for g in groups]
    ci_low = [g["event_rate"] - g["event_rate_wilson_95ci_low"] for g in groups]
    ci_high = [g["event_rate_wilson_95ci_high"] - g["event_rate"] for g in groups]

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(labels, rates, yerr=[ci_low, ci_high], capsize=5, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, max(0.45, max(ci_high[i] + rates[i] for i in range(len(rates))) + 0.05))
    ax.set_ylabel("Observed low-SMI event rate")
    ax.set_title("External sdata: clinical-positive patients at 90% train sensitivity")
    ax.grid(axis="y", alpha=0.25)
    for bar, g in zip(bars, groups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            g["event_rate_wilson_95ci_high"] + 0.015,
            f"{g['events']}/{g['n']}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_90_gaussian_deescalation_event_rate.png", dpi=180)
    plt.close(fig)

    reclass = result["external"]["gaussian_reclassification"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y_pos = np.arange(len(reclass))
    ax.barh(y_pos, reclass["non_events"], color="#72B7B2", label="Non-event")
    ax.barh(y_pos, reclass["events"], left=reclass["non_events"], color="#C84630", label="Low SMI event")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reclass["group"])
    ax.invert_yaxis()
    ax.set_xlabel("Patients")
    ax.set_title("External sdata 90% rule: reclassification by AEC gate")
    ax.legend(frameon=False, loc="lower right")
    for i, row in reclass.iterrows():
        ax.text(row["n"] + 4, i, f"n={int(row['n'])}, event={row['event_rate']:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_90_gaussian_reclassification_flow.png", dpi=180)
    plt.close(fig)


def plot_target_sweep(cond_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """민감도 목표(85/90/95%)를 바꿔가며 하향조정군 사건율과 순 위양성 감소량이 어떻게 변하는지 이중 y축 그래프로 저장 (임계값 선택에 대한 강건성 확인)."""
    ext_gauss = cond_df[
        (cond_df["split"] == "external")
        & (cond_df["gate"] == "gaussian")
        & (cond_df["group"] == "clinical+ / AEC-deescalated")
    ].sort_values("target_sensitivity")
    ext_metrics = metrics_df[
        (metrics_df["split"] == "external") & (metrics_df["model"].isin(["clinical_metrics", "gaussian_metrics_same_threshold"]))
    ]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(ext_gauss["target_sensitivity"], ext_gauss["event_rate"], marker="o", color="#F58518", label="De-escalated event rate")
    ax1.set_xlabel("g1090 clinical sensitivity target")
    ax1.set_ylabel("AEC-deescalated event rate")
    ax1.set_ylim(0, max(0.25, ext_gauss["event_rate"].max() + 0.05))
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    pivot = ext_metrics.pivot(index="target_sensitivity", columns="model", values="fp").sort_index()
    fp_removed = pivot["clinical_metrics"] - pivot["gaussian_metrics_same_threshold"]
    ax2.plot(fp_removed.index, fp_removed.values, marker="s", color="#4C78A8", label="Net FP reduction")
    ax2.set_ylabel("External net false-positive reduction")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.set_title("Robustness across fixed clinical sensitivity targets")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_target_sweep_deescalation.png", dpi=180)
    plt.close(fig)


def json_ready(obj):
    """중첩된 결과 구조에서 DataFrame은 제거하고 numpy 타입을 JSON 직렬화 가능한 파이썬 기본 타입으로 재귀 변환."""
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "임상 90% 민감도 기준 양성"으로 나온 환자 중, AEC 게이트로
    "하향조정(deescalate)"된 사람들은 실제로 저위험군인가? — 임계값을 고정한 임상적 활용 시나리오):

    1. train(g1090)/test(sdata)를 로드하고, 임상 단독/AEC 단독 모델의 OOF·외부 점수를 표준화해 준비.
    2. make_scores로 85%/90%/95% 세 가지 train 민감도 목표 각각에 대해:
       - threshold_for_min_sensitivity로 그 민감도를 유지하는 가장 높은(=가장 엄격한) 임계값을 고른다.
       - 그 임계값은 그대로 둔 채, 경계 근처 여성에게만 AEC를 얹는 가우시안 게이트와 하드존 게이트로
         "임상 양성"이었던 사람 일부를 "AEC 상 음성(하향조정)"으로 재분류한다.
       - conditional_stats/paired_error_changes/reclassification_table로 하향조정군과 유지군의
         사건율 차이, 새로 생기는/없어지는 FP·TP, Fisher 검정 결과를 모두 계산한다.
    3. 세 목표의 결과를 표로 펼쳐 CSV로 저장하고, 재분류표들도 각각 CSV로 저장한다.
    4. 90% 목표에 한해 부트스트랩(bootstrap_external_90)으로 하향조정군 사건율과 민감도손실/
       특이도이득의 신뢰구간을 외부 데이터에서 추정한다.
    5. 90% 목표의 외부 결과를 사건율 막대그래프와 재분류 흐름 그래프로 시각화하고, 목표 민감도를
       85→95%로 바꿔가며 결과가 얼마나 안정적인지(target sweep) 그래프로 확인한다.
    6. 방법론 설명(임계값을 g1090에서 고정하고 sdata에는 그대로 적용, 게이트 수식 등), 코호트 정보,
       목표별 전체 결과, 부트스트랩 결과를 모두 JSON으로 저장하고, 90% 목표의 핵심 결과만 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(ytr, 5)
    clinical_oof, clinical_test = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    aec_oof, aec_test = oof_and_external(lambda seed: aec_estimator(train["aec"].shape[1], seed), train["aec"], ytr, test["aec"], folds)

    c_z, c_te_z, c_mu, c_sd = zfit_apply(clinical_oof, clinical_test)
    a_z, a_te_z, _, _ = zfit_apply(aec_oof, aec_test)
    data = {
        "ytr": ytr,
        "yte": yte,
        "clinical_oof": clinical_oof,
        "clinical_test": clinical_test,
        "c_z": c_z,
        "c_te_z": c_te_z,
        "a_z": a_z,
        "a_te_z": a_te_z,
        "clinical_oof_mean": c_mu,
        "clinical_oof_sd": c_sd,
        "female_tr": (train["sex"] == "F").astype(float),
        "female_te": (test["sex"] == "F").astype(float),
    }

    target_results = {str(target): make_scores(target, data) for target in [0.85, 0.9, 0.95]}
    metrics_df = flatten_result_rows(target_results)
    cond_df = conditional_rows(target_results)
    metrics_df.to_csv(OUT_DIR / "sensitivity_target_metrics.csv", index=False)
    cond_df.to_csv(OUT_DIR / "clinical_positive_aec_deescalation_by_target.csv", index=False)
    save_reclassification_tables(target_results)

    result90 = target_results["0.9"]
    t_z90 = result90["threshold_detail"]["threshold_z"]
    bootstrap90 = bootstrap_external_90(yte, c_te_z, a_te_z, data["female_te"], t_z90)
    pd.DataFrame(bootstrap90).T.to_csv(OUT_DIR / "external_90_gaussian_bootstrap_ci.csv")

    plot_external_90(target_results)
    plot_target_sweep(cond_df, metrics_df)

    summary = {
        "method": {
            "clinical_high_risk_definition": "clinical-only logistic model age + sex + height + weight; high-risk threshold selected in g1090 OOF as the highest threshold retaining at least 90% sensitivity",
            "external_validation": "same locked threshold applied to sdata; no sdata threshold optimization",
            "aec_gate": "frozen female-boundary Gaussian gate: clinical_z + 0.25 * female * exp(-0.5*((clinical_z-threshold_z)/0.75)^2) * AEC_z; same clinical threshold used for final positive/negative decision",
            "hard_zone_sensitivity_analysis": "same but boundary weight is 1 only when abs(clinical_z-threshold_z) <= 0.50",
        },
        "cohorts": {
            "g1090_n": int(len(ytr)),
            "g1090_events": int(np.sum(ytr)),
            "g1090_event_rate": float(np.mean(ytr)),
            "sdata_n": int(len(yte)),
            "sdata_events": int(np.sum(yte)),
            "sdata_event_rate": float(np.mean(yte)),
        },
        "targets": json_ready(target_results),
        "external_90_gaussian_bootstrap": bootstrap90,
        "output_files": {
            "metrics_csv": str(OUT_DIR / "sensitivity_target_metrics.csv"),
            "conditional_csv": str(OUT_DIR / "clinical_positive_aec_deescalation_by_target.csv"),
            "bootstrap_csv": str(OUT_DIR / "external_90_gaussian_bootstrap_ci.csv"),
            "event_rate_plot": str(OUT_DIR / "external_90_gaussian_deescalation_event_rate.png"),
            "flow_plot": str(OUT_DIR / "external_90_gaussian_reclassification_flow.png"),
            "target_sweep_plot": str(OUT_DIR / "external_target_sweep_deescalation.png"),
        },
    }
    with open(OUT_DIR / "clinical90_deescalation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    short = {
        "threshold_90": result90["threshold_detail"],
        "external_clinical_90": result90["external"]["clinical_metrics"],
        "external_gaussian_90": result90["external"]["gaussian_metrics_same_threshold"],
        "external_gaussian_conditional_90": result90["external"]["gaussian_conditional"],
        "external_gaussian_paired_error_changes_90": result90["external"]["gaussian_paired_error_changes"],
        "train_gaussian_conditional_90": result90["train"]["gaussian_conditional"],
    }
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
