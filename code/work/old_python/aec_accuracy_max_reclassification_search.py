from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_all_reclassification import build_aec_all_matrices  # noqa: E402
from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, clinical_scores, deesc_metric_row, load_dataset  # noqa: E402
from aec_oof_auc_max_search import Candidate, crossfit_candidate  # noqa: E402
from aec_region_cnn_pattern_gate import codes_from_prob, pattern_mask_to_text  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_accuracy_max_reclassification_search"
PROB_PATH = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_pattern_gate" / "direct_vote_probabilities.npz"
PATTERN_SUMMARY_FILES = [
    Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_pattern_gate" / "direct_vote_balanced_pattern_search_summary.csv",
    Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_pattern_gate" / "direct_vote_guarded_pattern_search_summary.csv",
]
LOSS_BUDGETS = [0.02, 0.04, 0.06, 0.08]
MIN_DEESC_N = 10


@dataclass
class DeescCandidate:
    name: str
    kind: str
    op: str
    deesc_g: np.ndarray
    deesc_s: np.ndarray
    detail: str


def pass_constraints(row: dict, budget: float) -> bool:
    """재분류 후보 지표(row)가 최소 재분류 표본 수, 양(+)의 특이도 증가, 예산 이내 민감도 손실, 손실 유의성(p>=0.05) 조건을 모두 만족하는지 확인한다."""
    p_loss = row.get("sensitivity_loss_p_exact", np.nan)
    if not np.isfinite(p_loss):
        p_loss = 1.0
    return (
        row["deesc_n"] >= MIN_DEESC_N
        and row["specificity_gain"] > 0
        and row["sensitivity_loss"] <= budget + 1e-12
        and p_loss >= 0.05
    )


def score_cutoffs(score: np.ndarray, cpos: np.ndarray) -> np.ndarray:
    """임상 양성(cpos) 대상자의 점수 분포에서 분위수 기반 후보 컷오프 값들을 생성한다."""
    vals = np.asarray(score[cpos], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float)
    qs = np.unique(np.r_[np.linspace(0.01, 0.90, 90), np.linspace(0.91, 0.99, 9)])
    return np.unique(np.quantile(vals, qs))


def score_model_candidates(g: dict, s: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """aec_all 특징에 대해 다양한 k(SelectKBest 개수)와 C(정규화 강도)를 조합한 L2/L1 로지스틱 회귀 후보들을 학습해
    각 후보의 이름, 내부 OOF 점수, 외부 재적합 점수를 담은 목록을 반환한다."""
    xg, xs = build_aec_all_matrices(g, s)
    specs: list[Candidate] = []
    for k in [20, 40, 80, 100, 150, 250, 400, 600]:
        for c in [0.03, 0.1, 0.3]:
            specs.append(Candidate(f"aec_all_l2_k{k}_C{c}", "aec_all", "logit_l2", k=k, c=c))
    for k in [40, 100, 250]:
        for c in [0.03, 0.1]:
            specs.append(Candidate(f"aec_all_l1_k{k}_C{c}", "aec_all", "logit_l1", k=k, c=c))
    out = []
    for i, cand in enumerate(specs, start=1):
        print(f"[score {i}/{len(specs)}] {cand.name}", flush=True)
        score_g, _score_s_fold, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)
        out.append((cand.name, score_g, score_s))
    return out


def pattern_candidates(cpos_g: dict[str, np.ndarray], cpos_s: dict[str, np.ndarray]) -> list[DeescCandidate]:
    """direct-vote CNN 확률과 패턴 탐색 요약 파일들에서 (설정, 임계값, 패턴 마스크) 조합을 불러와, 각 운영점에서 임상 양성이면서
    선택된 코드 패턴에 해당하는 대상자를 재분류(de-escalate) 후보로 만든 DeescCandidate 목록을 반환한다."""
    data = np.load(PROB_PATH, allow_pickle=True)
    probs = {
        str(name): (
            np.asarray(data[f"{str(name)}_prob_g"], dtype=float),
            np.asarray(data[f"{str(name)}_prob_s"], dtype=float),
        )
        for name in data["configs"]
    }
    rows = []
    for path in PATTERN_SUMMARY_FILES:
        df = pd.read_csv(path).head(80)
        rows.append(df)
    summary = pd.concat(rows, ignore_index=True)
    summary = summary.drop_duplicates(["config", "threshold_R1", "threshold_R2", "threshold_R3", "threshold_R4", "pattern_mask"])
    out: list[DeescCandidate] = []
    for idx, row in summary.iterrows():
        config = str(row["config"])
        if config not in probs:
            continue
        th = row[["threshold_R1", "threshold_R2", "threshold_R3", "threshold_R4"]].to_numpy(dtype=float)
        mask = int(row["pattern_mask"])
        selected_codes = [k for k in range(16) if mask & (1 << k)]
        prob_g, prob_s = probs[config]
        code_g = codes_from_prob(prob_g, th)
        code_s = codes_from_prob(prob_s, th)
        for op_idx, (op, _) in enumerate(OPS):
            deesc_g = cpos_g[op] & np.isin(code_g[:, op_idx], selected_codes)
            deesc_s = cpos_s[op] & np.isin(code_s[:, op_idx], selected_codes)
            out.append(
                DeescCandidate(
                    name=f"pattern_{config}_{idx}_op{op}",
                    kind="pattern",
                    op=op,
                    deesc_g=deesc_g,
                    deesc_s=deesc_s,
                    detail=f"{config}; th={','.join(f'{x:.2f}' for x in th)}; patterns={pattern_mask_to_text(mask)}",
                )
            )
    return out


def score_threshold_candidates(
    score_models: list[tuple[str, np.ndarray, np.ndarray]],
    cpos_g: dict[str, np.ndarray],
    cpos_s: dict[str, np.ndarray],
) -> list[DeescCandidate]:
    """각 점수 모델과 운영점에 대해 여러 후보 컷오프를 적용, 임상 양성이면서 점수가 컷오프 미만인 대상자를 재분류 후보로 만든 DeescCandidate 목록을 반환한다."""
    out: list[DeescCandidate] = []
    for name, score_g, score_s in score_models:
        for op, _ in OPS:
            for cut in score_cutoffs(score_g, cpos_g[op]):
                deesc_g = cpos_g[op] & (score_g < cut)
                deesc_s = cpos_s[op] & (score_s < cut)
                out.append(
                    DeescCandidate(
                        name=f"{name}_cut{cut:.5g}_op{op}",
                        kind="score",
                        op=op,
                        deesc_g=deesc_g,
                        deesc_s=deesc_s,
                        detail=f"{name}; cutoff={cut:.6g}",
                    )
                )
    return out


def metric_for_candidate(dataset: str, d: dict, cpos: np.ndarray, cand: DeescCandidate) -> dict:
    """지정된 데이터셋(dataset)에 대해 재분류 후보(cand)의 de-escalation 마스크를 이용해 재분류 성능 지표 한 행을 계산한다."""
    row = deesc_metric_row(
        dataset,
        cand.name,
        cand.detail,
        cand.op,
        d["y"].astype(int),
        cpos,
        cand.deesc_g if dataset == "g1090_internal" else cand.deesc_s,
    )
    row["candidate_kind"] = cand.kind
    row["candidate_detail"] = cand.detail
    return row


def build_combo_candidates(
    op: str,
    base_candidates: list[DeescCandidate],
    internal_rows: pd.DataFrame,
    budget: float,
) -> list[DeescCandidate]:
    """제약 조건을 통과한 상위 점수/패턴 후보들끼리 OR(합집합) 및 AND(교집합) 조합을 만들어 새로운 결합 후보(DeescCandidate) 목록을 생성한다."""
    sub = internal_rows[internal_rows["operating_point"].eq(op)].copy()
    passing = sub[sub.apply(lambda r: pass_constraints(r.to_dict(), budget), axis=1)]
    if passing.empty:
        return []
    score_names = passing[passing["candidate_kind"].eq("score")].sort_values("accuracy_delta", ascending=False)["rule"].head(12).tolist()
    pattern_names = passing[passing["candidate_kind"].eq("pattern")].sort_values("accuracy_delta", ascending=False)["rule"].head(12).tolist()
    by_name = {c.name: c for c in base_candidates if c.op == op}
    combos = []
    for sn in score_names:
        for pn in pattern_names:
            if sn not in by_name or pn not in by_name:
                continue
            s = by_name[sn]
            p = by_name[pn]
            combos.append(
                DeescCandidate(
                    name=f"OR__{sn}__{pn}",
                    kind="score_OR_pattern",
                    op=op,
                    deesc_g=s.deesc_g | p.deesc_g,
                    deesc_s=s.deesc_s | p.deesc_s,
                    detail=f"OR({s.detail}) + ({p.detail})",
                )
            )
            combos.append(
                DeescCandidate(
                    name=f"AND__{sn}__{pn}",
                    kind="pattern_AND_score_guard",
                    op=op,
                    deesc_g=s.deesc_g & p.deesc_g,
                    deesc_s=s.deesc_s & p.deesc_s,
                    detail=f"AND({s.detail}) + ({p.detail})",
                )
            )
    return combos


def select_best(
    candidates: list[DeescCandidate],
    g: dict,
    s: dict,
    cpos_g: dict[str, np.ndarray],
    cpos_s: dict[str, np.ndarray],
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """기본 후보와 이들의 결합 후보를 모두 평가해 내부 지표를 구하고, 제약 조건을 통과하는 후보 중 정확도 위주 선택 점수가 가장 높은 후보를
    운영점별로 선택한 뒤 외부 데이터에도 적용해 (내부 선택 결과, 외부 적용 결과) 데이터프레임 쌍을 반환한다."""
    internal = []
    for cand in candidates:
        internal.append(metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand))
    internal_df = pd.DataFrame(internal)
    combo_candidates = []
    for op, _ in OPS:
        combo_candidates.extend(build_combo_candidates(op, candidates, internal_df, budget))
    combo_rows = []
    for cand in combo_candidates:
        combo_rows.append(metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand))
    if combo_rows:
        combo_df = pd.DataFrame(combo_rows)
        internal_df = pd.concat([internal_df, combo_df], ignore_index=True)
        all_candidates = candidates + combo_candidates
    else:
        all_candidates = candidates
    by_name = {c.name: c for c in all_candidates}
    selected = []
    external = []
    for op, _ in OPS:
        sub = internal_df[internal_df["operating_point"].eq(op)].copy()
        sub = sub[sub.apply(lambda r: pass_constraints(r.to_dict(), budget), axis=1)]
        if sub.empty:
            continue
        sub["selection_score"] = (
            sub["accuracy_delta"]
            + 0.15 * sub["specificity_gain"]
            - 0.10 * sub["sensitivity_loss"]
            - 0.001 * sub["deesc_event_rate"].fillna(0)
        )
        best = sub.sort_values(["selection_score", "accuracy_delta", "specificity_gain"], ascending=False).iloc[0]
        cand = by_name[str(best["rule"])]
        int_row = best.to_dict()
        int_row["loss_budget"] = budget
        selected.append(int_row)
        ext_row = metric_for_candidate("sdata_external", s, cpos_s[op], cand)
        ext_row["loss_budget"] = budget
        external.append(ext_row)
    return pd.DataFrame(selected), pd.DataFrame(external)


def plot_external(external: pd.DataFrame, out_path: Path) -> None:
    """민감도 손실 예산별로 외부 데이터셋의 정확도 증가, 민감도 손실, 특이도 증가를 운영점에 따라 3개 선그래프로 그려 저장한다."""
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {0.02: "#4c78a8", 0.04: "#f58518", 0.06: "#54a24b", 0.08: "#e45756"}
    for budget, color in colors.items():
        sub = external[external["loss_budget"].eq(budget)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=color, label=f"budget {budget:.0%}")
        axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, label=f"budget {budget:.0%}")
        axes[2].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, label=f"budget {budget:.0%}")
    for ax, title in [(axes[0], "Accuracy gain"), (axes[1], "Sensitivity loss"), (axes[2], "Specificity gain")]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """AEC 점수 임계값 후보와 CNN 패턴 후보(및 이들의 OR/AND 결합)를 생성해, 여러 민감도 손실 예산 하에서 정확도를 최대화하는
    재분류 규칙을 내부 데이터로 선택하고 외부 데이터에 적용한 결과를 CSV/그래프/JSON으로 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, _thresholds = clinical_scores(g, s)
    clinical_thresholds = {
        op: float(threshold_for_min_sensitivity(g["y"].astype(int), c_g, target))
        for op, target in OPS
    }
    cpos_g = {op: c_g >= clinical_thresholds[op] for op, _ in OPS}
    cpos_s = {op: c_s >= clinical_thresholds[op] for op, _ in OPS}
    score_models = score_model_candidates(g, s)
    base_candidates = score_threshold_candidates(score_models, cpos_g, cpos_s)
    base_candidates.extend(pattern_candidates(cpos_g, cpos_s))
    print(f"base candidates={len(base_candidates)}", flush=True)

    selected_rows = []
    external_rows = []
    for budget in LOSS_BUDGETS:
        print(f"selecting budget={budget:.2f}", flush=True)
        selected, external = select_best(base_candidates, g, s, cpos_g, cpos_s, budget)
        selected_rows.append(selected)
        external_rows.append(external)
    selected_all = pd.concat(selected_rows, ignore_index=True)
    external_all = pd.concat(external_rows, ignore_index=True)
    selected_all.to_csv(OUT_DIR / "accuracy_max_internal_selected.csv", index=False)
    external_all.to_csv(OUT_DIR / "accuracy_max_external_applied.csv", index=False)
    plot_external(external_all, OUT_DIR / "accuracy_max_external_tradeoff.png")
    with (OUT_DIR / "accuracy_max_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection": "Internal g1090 OOF chooses the de-escalation candidate maximizing accuracy gain under a sensitivity-loss budget, p_loss>=0.05, specificity_gain>0, and min de-escalation n>=10.",
                "candidate_types": ["AEC-all score thresholds across k/C", "direct-vote CNN pattern candidates", "score OR pattern", "pattern AND score guard"],
                "operating_points": [op for op, _ in OPS],
                "loss_budgets": LOSS_BUDGETS,
            },
            f,
            indent=2,
        )
    show = external_all[
        [
            "loss_budget",
            "operating_point",
            "candidate_kind",
            "clinical_accuracy",
            "post_accuracy",
            "accuracy_delta",
            "accuracy_delta_p_mcnemar",
            "clinical_sensitivity",
            "post_sensitivity",
            "sensitivity_loss",
            "sensitivity_loss_p_exact",
            "clinical_specificity",
            "post_specificity",
            "specificity_gain",
            "specificity_gain_p_exact",
            "deesc_n",
            "deesc_events",
            "deesc_event_rate",
        ]
    ]
    print("\nEXTERNAL ACCURACY-MAX RESULTS")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# AEC 점수/CNN 패턴 기반 재분류 후보들을 생성·결합해 민감도 손실 예산별로 정확도를 최대화하는 규칙을 탐색하고 내부/외부 결과를 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
