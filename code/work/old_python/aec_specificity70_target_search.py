from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    OUT_DIR as MIDRANGE_OUT_DIR,
    adjusted_deesc_p,
    build_candidate_bank,
    clinical_scores,
    counts,
    gate_metrics,
    load_aec128,
    risk_direction,
    standardize_train_test,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_specificity70_target_search"
PRIMARY_OPS = ["youden", "sens80", "sens85"]
ALL_OPS = ["youden", "sens80", "sens85", "sens90", "sens95"]
TARGET_SPECS = [0.68, 0.70, 0.72]
MAX_SENS_LOSSES = [0.03, 0.05, 0.075]


def add_train_rule_columns(train_ranked: pd.DataFrame, y: np.ndarray, clinical_z: np.ndarray, thresholds: dict[str, dict]) -> pd.DataFrame:
    """이전 스크립트가 저장해둔 특징 순위표에, 3개 주요 운영점의 "게이트 적용 후 실제 특이도/민감도" 절대값 컬럼들을 추가로 계산해 붙임."""
    df = train_ranked.copy()
    for op in PRIMARY_OPS:
        base = counts(y, clinical_z >= thresholds[op]["clinical_z"])
        df[f"{op}_clinical_specificity"] = base["specificity"]
        df[f"{op}_rule_specificity"] = base["specificity"] + df[f"{op}_spec_gain"]
        df[f"{op}_rule_sensitivity"] = base["sensitivity"] - df[f"{op}_sens_loss"]
    df["primary_min_rule_specificity"] = df[[f"{op}_rule_specificity" for op in PRIMARY_OPS]].min(axis=1)
    df["primary_avg_rule_specificity"] = df[[f"{op}_rule_specificity" for op in PRIMARY_OPS]].mean(axis=1)
    df["primary_max_sens_loss"] = df[[f"{op}_sens_loss" for op in PRIMARY_OPS]].max(axis=1)
    df["primary_avg_sens_loss"] = df[[f"{op}_sens_loss" for op in PRIMARY_OPS]].mean(axis=1)
    df["primary_min_balanced_gain"] = df[[f"{op}_balanced_gain" for op in PRIMARY_OPS]].min(axis=1)
    df["primary_max_fisher_p"] = df[[f"{op}_fisher_p" for op in PRIMARY_OPS]].max(axis=1)
    return df


def candidate_score(df: pd.DataFrame) -> pd.Series:
    """최소/평균 최종특이도, 최소균형이득, 평균민감도손실, 최대 Fisher p값을 가중합해, 목표 특이도를 만족하는 후보들 중 우선순위를 매기는 점수를 계산."""
    return (
        2.0 * df["primary_min_rule_specificity"]
        + 0.8 * df["primary_avg_rule_specificity"]
        + 0.7 * df["primary_min_balanced_gain"]
        - 0.6 * df["primary_avg_sens_loss"]
        - 0.01 * np.log10(np.clip(df["primary_max_fisher_p"], 1e-12, 1.0))
    )


def evaluate_candidate(
    row: pd.Series,
    names: list[str],
    datasets: dict[str, dict],
    clinical_scores_by_dataset: dict[str, np.ndarray],
    x_by_dataset: dict[str, np.ndarray],
    thresholds: dict[str, dict],
) -> list[dict]:
    """한 후보(특징+폭+람다)를 모든 데이터셋 x 모든 운영점(5개)에서 평가해 결과 행 목록으로 반환."""
    j = names.index(str(row["feature"]))
    rows = []
    for dataset_name, d in datasets.items():
        c = clinical_scores_by_dataset[dataset_name]
        z = x_by_dataset[dataset_name][:, j]
        for op in ALL_OPS:
            m = gate_metrics(d["y"], c, z, thresholds[op]["clinical_z"], float(row["width"]), float(row["lambda"]))
            rows.append(
                {
                    "selection_label": row["selection_label"],
                    "selection_type": row["selection_type"],
                    "target_min_specificity": row["target_min_specificity"],
                    "max_allowed_sensitivity_loss": row["max_allowed_sensitivity_loss"],
                    "train_ranked_index": int(row["train_ranked_index"]),
                    "dataset": dataset_name,
                    "feature": row["feature"],
                    "width": float(row["width"]),
                    "lambda": float(row["lambda"]),
                    "operating_point": op,
                    **m,
                }
            )
    return rows


def summarize_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    """후보x데이터셋별로 3개 주요 운영점의 지표를 요약(최소/평균 특이도, 민감도손실, 균형이득 등)하고, 목표 특이도·손실 조건을 통과하는지(passes_target) 표시."""
    rows = []
    for keys, sub in eval_df.groupby(
        [
            "selection_label",
            "selection_type",
            "target_min_specificity",
            "max_allowed_sensitivity_loss",
            "dataset",
            "feature",
            "width",
            "lambda",
        ]
    ):
        primary = sub[sub["operating_point"].isin(PRIMARY_OPS)]
        rows.append(
            {
                **dict(
                    zip(
                        [
                            "selection_label",
                            "selection_type",
                            "target_min_specificity",
                            "max_allowed_sensitivity_loss",
                            "dataset",
                            "feature",
                            "width",
                            "lambda",
                        ],
                        keys,
                    )
                ),
                "primary_min_rule_specificity": float(primary["rule_specificity"].min()),
                "primary_avg_rule_specificity": float(primary["rule_specificity"].mean()),
                "primary_avg_spec_gain": float(primary["specificity_gain"].mean()),
                "primary_avg_sens_loss": float(primary["sensitivity_loss"].mean()),
                "primary_max_sens_loss": float(primary["sensitivity_loss"].max()),
                "primary_min_balanced_gain": float(primary["balanced_gain"].min()),
                "primary_avg_deesc_prevalence": float(primary["deesc_prevalence"].mean()),
                "primary_max_fisher_p": float(primary["fisher_p"].max()),
                "passes_target": bool(
                    (primary["rule_specificity"].min() >= keys[2])
                    and (primary["sensitivity_loss"].max() <= keys[3])
                ),
            }
        )
    return pd.DataFrame(rows)


def build_selection_table(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """목표 특이도(68/70/72%) x 최대허용 민감도손실(3/5/7.5%) 9개 조합마다, train에서 조건을 만족하는
    후보 중 candidate_score가 가장 높은 1개씩을 "엄격한 train 선택" 후보로 뽑음."""
    selections = []
    counts_rows = []
    ranked = train_df.reset_index(names="train_ranked_index").copy()
    ranked["candidate_priority"] = candidate_score(ranked)
    for target in TARGET_SPECS:
        for max_loss in MAX_SENS_LOSSES:
            ok = ranked[
                (ranked["primary_min_rule_specificity"] >= target)
                & (ranked["primary_max_sens_loss"] <= max_loss)
                & (ranked["train_primary_min_deesc_n"] >= 20)
                & (~ranked["train_constraint_fail"].astype(bool))
            ].copy()
            counts_rows.append(
                {
                    "target_min_specificity": target,
                    "max_allowed_sensitivity_loss": max_loss,
                    "n_train_candidates": int(len(ok)),
                }
            )
            label = f"spec{int(round(target*100))}_loss{max_loss*100:.1f}"
            if ok.empty:
                continue
            strict = ok.sort_values(
                ["candidate_priority", "primary_min_rule_specificity", "primary_min_balanced_gain"],
                ascending=False,
            ).head(1)
            strict = strict.assign(
                selection_label=label,
                selection_type="strict_train_selected",
                target_min_specificity=target,
                max_allowed_sensitivity_loss=max_loss,
            )
            selections.append(strict)
    if selections:
        return pd.concat(selections, ignore_index=True), pd.DataFrame(counts_rows)
    return pd.DataFrame(), pd.DataFrame(counts_rows)


def pick_external_references(
    train_df: pd.DataFrame,
    external_eval_all: pd.DataFrame,
) -> pd.DataFrame:
    """목표 조합마다 train 조건을 통과한 모든 후보의 외부 성능을 비교해, 외부에서 가장 좋은 후보를
    "외부 참고용(oracle 성격)"으로 별도로 뽑음 — train 선택과 별개로 상한선 참고용."""
    rows = []
    ranked = train_df.reset_index(names="train_ranked_index").copy()
    for target in TARGET_SPECS:
        for max_loss in MAX_SENS_LOSSES:
            label = f"spec{int(round(target*100))}_loss{max_loss*100:.1f}"
            ok_train = ranked[
                (ranked["primary_min_rule_specificity"] >= target)
                & (ranked["primary_max_sens_loss"] <= max_loss)
                & (ranked["train_primary_min_deesc_n"] >= 20)
                & (~ranked["train_constraint_fail"].astype(bool))
            ][["train_ranked_index"]]
            if ok_train.empty:
                continue
            sub = external_eval_all.merge(ok_train, on="train_ranked_index", how="inner")
            primary = sub[(sub["dataset"].eq("sdata_external")) & (sub["operating_point"].isin(PRIMARY_OPS))]
            if primary.empty:
                continue
            summary = (
                primary.groupby("train_ranked_index", as_index=False)
                .agg(
                    ext_min_rule_spec=("rule_specificity", "min"),
                    ext_avg_rule_spec=("rule_specificity", "mean"),
                    ext_avg_spec_gain=("specificity_gain", "mean"),
                    ext_avg_sens_loss=("sensitivity_loss", "mean"),
                    ext_max_sens_loss=("sensitivity_loss", "max"),
                    ext_min_balanced_gain=("balanced_gain", "min"),
                    ext_max_fisher_p=("fisher_p", "max"),
                )
                .sort_values(["ext_min_rule_spec", "ext_min_balanced_gain", "ext_avg_rule_spec"], ascending=False)
            )
            best = summary.head(1).merge(ranked, on="train_ranked_index", how="left")
            best = best.assign(
                selection_label=label,
                selection_type="external_reference",
                target_min_specificity=target,
                max_allowed_sensitivity_loss=max_loss,
            )
            rows.append(best)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def plot_primary(summary: pd.DataFrame, path: Path) -> None:
    """목표 조합별로 최종 특이도와 최대 민감도손실을 막대그래프로 비교하고 70% 기준선을 표시해 PNG로 저장."""
    ext = summary[(summary["dataset"].eq("sdata_external")) & (summary["selection_type"].eq("strict_train_selected"))].copy()
    if ext.empty:
        return
    ext = ext.sort_values(["target_min_specificity", "max_allowed_sensitivity_loss"])
    labels = [f"{int(t*100)}%/{l*100:.1f}%" for t, l in zip(ext["target_min_specificity"], ext["max_allowed_sensitivity_loss"])]
    x = np.arange(len(ext))
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.bar(x - 0.18, ext["primary_min_rule_specificity"] * 100, width=0.36, color="#4C78A8", label="Min final specificity")
    ax.bar(x + 0.18, ext["primary_max_sens_loss"] * 100, width=0.36, color="#D95F02", label="Max sensitivity loss")
    ax.axhline(70, color="#333333", lw=1.1, ls="--", label="70% specificity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.set_xlabel("Train target specificity / max sensitivity loss")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "특이도 70% 이상, 민감도손실 X% 이하"라는 구체적인 임상
    목표를 정해두고, aec_midrange_feature_refit의 후보군 중 그 목표를 만족하는 특징+게이트를
    train에서 골라내면, 외부에서도 목표가 지켜지는가?):

    1. aec_midrange_feature_refit이 저장해둔 train 순위표를 불러와 각 후보의 3개 주요 운영점
       실제 특이도·민감도손실을 계산해 붙인다.
    2. 목표 특이도(68/70/72%) x 최대허용 민감도손실(3/5/7.5%) 9개 조합마다, train에서 조건을
       만족하는 후보 수를 세고, 그중 candidate_score가 가장 높은 1개를 "엄격한 train 선택"으로 뽑는다.
    3. 선택된 후보들을 모든 데이터셋 x 5개 운영점에서 평가해 CSV로 저장하고 요약.
    4. 비교를 위해, train 조건을 통과하는 모든 후보를 외부에서도 한 번씩 평가해두고, 외부 성능이
       가장 좋은 것을 "외부 참고용(oracle)"으로 별도 표시 — 이건 선택 기준이 아니라 상한선 참고용.
    5. 선택된 후보(엄격한 train 선택 + 외부 참고용)에 대해 스캐너(제조사) 통제 하 하향조정 유의성을 확인.
    6. 목표 조합별 특이도/민감도손실 막대그래프를 저장하고, 결합 요약·목표 정의·산출물 경로를
       JSON으로 저장한 뒤 train 후보 수·외부 요약·조정된 p값을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked_path = MIDRANGE_OUT_DIR / "midrange_feature_search_train_ranked.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Run aec_midrange_feature_refit.py first: {ranked_path}")

    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)
    train_ranked = pd.read_csv(ranked_path)
    train_df = add_train_rule_columns(train_ranked, g["y"], c_g, thresholds)
    train_df.to_csv(OUT_DIR / "specificity_target_train_candidates_all.csv", index=False)
    selections, counts_df = build_selection_table(train_df)
    counts_df.to_csv(OUT_DIR / "specificity_target_train_candidate_counts.csv", index=False)

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]
    datasets = {"g1090_oof": g, "sdata_external": s}
    clinical_by_dataset = {"g1090_oof": c_g, "sdata_external": c_s}
    x_by_dataset = {"g1090_oof": xg, "sdata_external": xs}

    if selections.empty:
        raise RuntimeError("No train candidates passed the specificity/sensitivity target grid.")
    eval_rows = []
    for _, row in selections.iterrows():
        eval_rows.extend(evaluate_candidate(row, names, datasets, clinical_by_dataset, x_by_dataset, thresholds))
    strict_eval = pd.DataFrame(eval_rows)
    strict_eval.to_csv(OUT_DIR / "strict_train_selected_external_eval.csv", index=False)
    strict_summary = summarize_eval(strict_eval)
    strict_summary.to_csv(OUT_DIR / "strict_train_selected_summary.csv", index=False)

    # Evaluate every train-passing candidate once, so the external-reference table is explicit
    # about its exploratory/oracle nature.
    all_train_ok = train_df.reset_index(names="train_ranked_index")
    all_train_ok = all_train_ok[
        (all_train_ok["primary_min_rule_specificity"] >= min(TARGET_SPECS))
        & (all_train_ok["primary_max_sens_loss"] <= max(MAX_SENS_LOSSES))
        & (all_train_ok["train_primary_min_deesc_n"] >= 20)
        & (~all_train_ok["train_constraint_fail"].astype(bool))
    ].copy()
    all_train_ok = all_train_ok.assign(
        selection_label="all_train_pass_for_external_reference",
        selection_type="all_train_pass",
        target_min_specificity=min(TARGET_SPECS),
        max_allowed_sensitivity_loss=max(MAX_SENS_LOSSES),
    )
    all_rows = []
    for _, row in all_train_ok.iterrows():
        all_rows.extend(evaluate_candidate(row, names, datasets, clinical_by_dataset, x_by_dataset, thresholds))
    all_eval = pd.DataFrame(all_rows)
    all_eval.to_csv(OUT_DIR / "all_train_pass_candidates_external_eval.csv", index=False)
    external_refs = pick_external_references(train_df, all_eval)
    ref_eval = pd.DataFrame()
    if not external_refs.empty:
        ref_rows = []
        for _, row in external_refs.iterrows():
            ref_rows.extend(evaluate_candidate(row, names, datasets, clinical_by_dataset, x_by_dataset, thresholds))
        ref_eval = pd.DataFrame(ref_rows)
        ref_eval.to_csv(OUT_DIR / "external_reference_selected_eval.csv", index=False)
        summarize_eval(ref_eval).to_csv(OUT_DIR / "external_reference_selected_summary.csv", index=False)

    selected_for_adjustment = pd.concat(
        [
            selections.assign(adjustment_source="strict_train_selected"),
            external_refs.assign(adjustment_source="external_reference") if not external_refs.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    adj_rows = []
    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    seen = set()
    for _, row in selected_for_adjustment.iterrows():
        key = (row["selection_label"], row["selection_type"], row["feature"], float(row["width"]), float(row["lambda"]))
        if key in seen:
            continue
        seen.add(key)
        j = names.index(str(row["feature"]))
        for include_clinical in [False, True]:
            adj = adjusted_deesc_p(
                s["y"],
                c_s,
                xs[:, j],
                scanner_s,
                thresholds,
                float(row["width"]),
                float(row["lambda"]),
                include_clinical=include_clinical,
            )
            adj.insert(0, "selection_label", row["selection_label"])
            adj.insert(1, "selection_type", row["selection_type"])
            adj.insert(2, "feature", row["feature"])
            adj_rows.append(adj)
    adjusted = pd.concat(adj_rows, ignore_index=True) if adj_rows else pd.DataFrame()
    adjusted.to_csv(OUT_DIR / "selected_external_adjusted_pvalues.csv", index=False)

    combined_summary = pd.concat(
        [strict_summary.assign(summary_source="strict_train_selected"), summarize_eval(ref_eval).assign(summary_source="external_reference") if not ref_eval.empty else pd.DataFrame()],
        ignore_index=True,
    )
    combined_summary.to_csv(OUT_DIR / "specificity70_combined_summary.csv", index=False)
    plot_primary(strict_summary, OUT_DIR / "strict_train_selected_specificity_targets.png")

    print("Train candidate counts")
    print(counts_df.to_string(index=False))
    print("\nStrict train-selected external summary")
    print(strict_summary[strict_summary["dataset"].eq("sdata_external")].to_string(index=False))
    if not ref_eval.empty:
        ref_summary = summarize_eval(ref_eval)
        print("\nExternal-reference summary")
        print(ref_summary[ref_summary["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nAdjusted p-values")
    print(adjusted.to_string(index=False))
    (OUT_DIR / "specificity70_target_search_summary.json").write_text(
        json.dumps(
            {
                "primary_operating_points": PRIMARY_OPS,
                "target_specificities": TARGET_SPECS,
                "max_sensitivity_losses": MAX_SENS_LOSSES,
                "n_train_ranked_rows": int(len(train_df)),
                "n_train_pass_min_grid": int(len(all_train_ok)),
                "outputs": {
                    "strict_summary": str(OUT_DIR / "strict_train_selected_summary.csv"),
                    "combined_summary": str(OUT_DIR / "specificity70_combined_summary.csv"),
                    "adjusted_pvalues": str(OUT_DIR / "selected_external_adjusted_pvalues.csv"),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
