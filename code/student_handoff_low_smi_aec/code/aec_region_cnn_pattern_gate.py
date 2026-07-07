from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, deesc_metric_row, exact_p, load_dataset  # noqa: E402
from aec_region_constrained_cnn_gate import DEVICE, REGIONS, make_channels, standardize_channels_train_apply  # noqa: E402
from aec_region_cnn_direct_vote_gate import (  # noqa: E402
    CONFIGS,
    auc_table,
    crossfit_config,
    exact_feature_votes,
)
from aec_region_cnn_teacher_mimic_gate import locked_targets  # noqa: E402
from aec_lock_smoothed_deesc_gate import clinical_scores  # noqa: E402


OUT_DIR = Path(r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_region_cnn_pattern_gate")
PROB_CACHE = OUT_DIR / "direct_vote_probabilities.npz"


def pattern_str(code: int) -> str:
    return "".join("+" if code & (1 << j) else "-" for j in range(len(REGIONS)))


def pattern_mask_to_text(mask: int) -> str:
    return ",".join(pattern_str(code) for code in range(16) if mask & (1 << code))


def popcount(x: int) -> int:
    return int(bin(x).count("1"))


def codes_from_prob(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    votes = prob >= thresholds[None, None, :]
    code = np.zeros(votes.shape[:2], dtype=np.int16)
    for j in range(votes.shape[-1]):
        code += votes[..., j].astype(np.int16) * (1 << j)
    return code


def votes_to_codes(votes: np.ndarray) -> np.ndarray:
    code = np.zeros(votes.shape[:2], dtype=np.int16)
    for j in range(votes.shape[-1]):
        code += votes[..., j].astype(np.int16) * (1 << j)
    return code


def evaluate_pattern_gate(
    rule: str,
    pattern_mask: int,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for dataset, d, cpos, code in [
        ("g1090_internal", g, cpos_g, code_g),
        ("sdata_external", s, cpos_s, code_s),
    ]:
        for op_idx, (op, _) in enumerate(OPS):
            selected = np.isin(code[:, op_idx], [k for k in range(16) if pattern_mask & (1 << k)])
            deesc = cpos[:, op_idx] & selected
            rows.append(
                deesc_metric_row(
                    dataset,
                    rule,
                    pattern_mask_to_text(pattern_mask),
                    op,
                    d["y"].astype(int),
                    cpos[:, op_idx],
                    deesc,
                )
            )
    return pd.DataFrame(rows)


def summarize_internal(detail: pd.DataFrame) -> dict:
    gi = detail[detail["dataset"].eq("g1090_internal")]
    return {
        "internal_min_p_loss": float(gi["sensitivity_loss_p_exact"].min(skipna=True)),
        "internal_max_sens_loss": float(gi["sensitivity_loss"].max(skipna=True)),
        "internal_min_spec_gain": float(gi["specificity_gain"].min(skipna=True)),
        "internal_mean_spec_gain": float(gi["specificity_gain"].mean(skipna=True)),
        "internal_max_fisher_p": float(gi["deesc_event_fisher_p"].max(skipna=True)),
        "internal_min_deesc_n": int(gi["deesc_n"].min(skipna=True)),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean(skipna=True)),
    }


def fast_summary_internal(g: dict, cpos_g: np.ndarray, code_g: np.ndarray, pattern_mask: int) -> dict:
    selected_codes = [k for k in range(16) if pattern_mask & (1 << k)]
    y = g["y"].astype(bool)
    total_pos = max(int(y.sum()), 1)
    total_neg = max(int((~y).sum()), 1)
    p_loss = []
    sens_loss = []
    spec_gain = []
    deesc_n = []
    event_rate = []
    for op_idx, _ in enumerate(OPS):
        deesc = cpos_g[:, op_idx] & np.isin(code_g[:, op_idx], selected_codes)
        de_e = int(np.sum(deesc & y))
        de_ne = int(np.sum(deesc & ~y))
        n = de_e + de_ne
        p_loss.append(exact_p(de_e, 0))
        sens_loss.append(de_e / total_pos)
        spec_gain.append(de_ne / total_neg)
        deesc_n.append(n)
        event_rate.append(de_e / n if n else np.nan)
    return {
        "internal_min_p_loss": float(np.nanmin(p_loss)),
        "internal_max_sens_loss": float(np.nanmax(sens_loss)),
        "internal_min_spec_gain": float(np.nanmin(spec_gain)),
        "internal_mean_spec_gain": float(np.nanmean(spec_gain)),
        "internal_max_fisher_p": np.nan,
        "internal_min_deesc_n": int(np.nanmin(deesc_n)),
        "internal_mean_event_rate": float(np.nanmean(event_rate)),
    }


def internal_score(summary: dict) -> tuple[bool, float]:
    fisher_ok = not np.isfinite(summary.get("internal_max_fisher_p", np.nan)) or summary["internal_max_fisher_p"] < 0.05
    survives = (
        summary["internal_min_p_loss"] >= 0.05
        and summary["internal_max_sens_loss"] <= 0.08
        and summary["internal_min_spec_gain"] > 0
        and fisher_ok
        and summary["internal_min_deesc_n"] >= 25
        and summary["internal_mean_event_rate"] <= 0.12
    )
    score = (
        3.0 * summary["internal_min_spec_gain"]
        + 1.3 * summary["internal_mean_spec_gain"]
        - 0.9 * summary["internal_max_sens_loss"]
        - 0.25 * summary["internal_mean_event_rate"]
    )
    if np.isfinite(summary.get("internal_max_fisher_p", np.nan)):
        score -= 0.02 * summary["internal_max_fisher_p"]
    if not survives:
        score -= 10.0
    return survives, float(score)


def rank_single_patterns(g: dict, cpos_g: np.ndarray, code_g: np.ndarray) -> list[int]:
    rows = []
    dummy_s = g
    dummy_cpos_s = cpos_g
    dummy_code_s = code_g
    for code in range(16):
        mask = 1 << code
        summary = fast_summary_internal(g, cpos_g, code_g, mask)
        _, score = internal_score(summary)
        rows.append((score, code, summary["internal_min_deesc_n"], summary["internal_mean_event_rate"]))
    rows.sort(reverse=True)
    return [code for _, code, _, _ in rows[:6]]


def candidate_masks(top_codes: list[int]) -> list[int]:
    masks: set[int] = set()
    for code in range(16):
        masks.add(1 << code)
    for k in [1, 2, 3, 4]:
        m = 0
        for code in range(16):
            if popcount(code) >= k:
                m |= 1 << code
        masks.add(m)
    for k in [1, 2, 3, 4]:
        m = 0
        for code in range(16):
            if popcount(code) == k:
                m |= 1 << code
        masks.add(m)
    for r in range(2, min(3, len(top_codes)) + 1):
        for combo in itertools.combinations(top_codes, r):
            m = 0
            for code in combo:
                m |= 1 << code
            masks.add(m)
    return sorted(masks)


def threshold_vectors() -> list[np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    for p in np.round(np.arange(0.35, 0.86, 0.05), 2):
        rows.append((float(p), float(p), float(p), float(p)))
    for vals in itertools.product([0.55, 0.65, 0.75], repeat=4):
        rows.append(tuple(float(v) for v in vals))
    unique = sorted(set(rows))
    return [np.array(v, dtype=float) for v in unique]


def pattern_distribution_table(
    label: str,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for dataset, d, cpos, code in [
        ("g1090_internal", g, cpos_g, code_g),
        ("sdata_external", s, cpos_s, code_s),
    ]:
        y = d["y"].astype(bool)
        for op_idx, (op, _) in enumerate(OPS):
            cp = cpos[:, op_idx]
            for pat in range(16):
                idx = cp & (code[:, op_idx] == pat)
                rows.append(
                    {
                        "rule": label,
                        "dataset": dataset,
                        "operating_point": op,
                        "pattern_code": pat,
                        "pattern": pattern_str(pat),
                        "n": int(idx.sum()),
                        "events": int((idx & y).sum()),
                        "event_rate": float((idx & y).sum() / idx.sum()) if idx.sum() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def search_pattern_gate(
    config_name: str,
    prob_g: np.ndarray,
    prob_s: np.ndarray,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fast_rows = []
    dist_rows = []
    for thresholds in threshold_vectors():
        code_g = codes_from_prob(prob_g, thresholds)
        code_s = codes_from_prob(prob_s, thresholds)
        top_codes = rank_single_patterns(g, cpos_g, code_g)
        for mask in candidate_masks(top_codes):
            summary = fast_summary_internal(g, cpos_g, code_g, mask)
            survives, score = internal_score(summary)
            fast_rows.append(
                {
                    "config": config_name,
                    "threshold_R1": thresholds[0],
                    "threshold_R2": thresholds[1],
                    "threshold_R3": thresholds[2],
                    "threshold_R4": thresholds[3],
                    "pattern_mask": mask,
                    "patterns": pattern_mask_to_text(mask),
                    "n_patterns": popcount(mask),
                    "survives_internal_constraints": survives,
                    "internal_selection_score": score,
                    **summary,
                }
            )
    fast_df = pd.DataFrame(fast_rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    exact_rows = []
    exact_details: dict[tuple[float, float, float, float, int], pd.DataFrame] = {}
    for _, row in fast_df.head(300).iterrows():
        thresholds = row[["threshold_R1", "threshold_R2", "threshold_R3", "threshold_R4"]].to_numpy(dtype=float)
        mask = int(row["pattern_mask"])
        code_g = codes_from_prob(prob_g, thresholds)
        code_s = codes_from_prob(prob_s, thresholds)
        rule = f"{config_name}_patterns_t{'_'.join(f'{x:.2f}' for x in thresholds)}_m{mask}"
        detail = evaluate_pattern_gate(rule, mask, g, s, cpos_g, cpos_s, code_g, code_s)
        summary = summarize_internal(detail)
        survives, score = internal_score(summary)
        key = (*[float(x) for x in thresholds], mask)
        exact_details[key] = detail
        exact_rows.append(
            {
                "config": config_name,
                "threshold_R1": thresholds[0],
                "threshold_R2": thresholds[1],
                "threshold_R3": thresholds[2],
                "threshold_R4": thresholds[3],
                "pattern_mask": mask,
                "patterns": pattern_mask_to_text(mask),
                "n_patterns": popcount(mask),
                "survives_internal_constraints": survives,
                "internal_selection_score": score,
                **summary,
            }
        )
    summary_df = pd.DataFrame(exact_rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best = summary_df.iloc[0]
    best_thresholds = best[["threshold_R1", "threshold_R2", "threshold_R3", "threshold_R4"]].to_numpy(dtype=float)
    best_mask = int(best["pattern_mask"])
    best_code_g = codes_from_prob(prob_g, best_thresholds)
    best_code_s = codes_from_prob(prob_s, best_thresholds)
    best_rule = f"{config_name}_pattern_gate"
    best_detail = evaluate_pattern_gate(best_rule, best_mask, g, s, cpos_g, cpos_s, best_code_g, best_code_s)
    dist_rows.append(pattern_distribution_table(best_rule, g, s, cpos_g, cpos_s, best_code_g, best_code_s))
    return summary_df, best_detail, pd.concat(dist_rows, ignore_index=True)


def load_or_train_probabilities(g: dict, s: dict, c_g: np.ndarray, c_s: np.ndarray, thresholds: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    if PROB_CACHE.exists():
        data = np.load(PROB_CACHE, allow_pickle=True)
        configs = [str(x) for x in data["configs"]]
        out = {}
        for name in configs:
            out[name] = {"prob_g": data[f"{name}_prob_g"], "prob_s": data[f"{name}_prob_s"]}
        logs = pd.read_csv(OUT_DIR / "pattern_gate_training_log.csv") if (OUT_DIR / "pattern_gate_training_log.csv").exists() else pd.DataFrame()
        return out, logs

    feature_g, _, _ = locked_targets(g, s, c_g)
    target_g, cpos_g = exact_feature_votes(g["y"], c_g, thresholds, feature_g)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    threshold_vec = np.array([thresholds[op] for op, _ in OPS], dtype=np.float32)
    out = {}
    logs = []
    for cfg in CONFIGS:
        print(f"training {cfg.name}", flush=True)
        logits_g, logits_s, log_df = crossfit_config(cfg, xg, c_g, target_g, cpos_g, xs, c_s, g["y"], threshold_vec)
        out[cfg.name] = {
            "prob_g": 1.0 / (1.0 + np.exp(-logits_g)),
            "prob_s": 1.0 / (1.0 + np.exp(-logits_s)),
        }
        logs.append(log_df)
    logs_df = pd.concat(logs, ignore_index=True)
    np.savez_compressed(
        PROB_CACHE,
        configs=np.array(list(out.keys()), dtype=object),
        **{f"{name}_prob_g": v["prob_g"] for name, v in out.items()},
        **{f"{name}_prob_s": v["prob_s"] for name, v in out.items()},
    )
    return out, logs_df


def plot_best(detail: pd.DataFrame, dist: pd.DataFrame, out_path: Path) -> None:
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {"exact_locked_2of4": "#2c7fb8", "pattern_gate": "#d95f02"}
    detail_plot = detail.copy()
    detail_plot["plot_rule"] = np.where(detail_plot["rule"].eq("exact_locked_2of4"), "exact_locked_2of4", "pattern_gate")
    for rule in ["exact_locked_2of4", "pattern_gate"]:
        for dataset, ls in [("g1090_internal", "-"), ("sdata_external", "--")]:
            sub = detail_plot[detail_plot["plot_rule"].eq(rule) & detail_plot["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
            x = np.arange(len(labels))
            axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", ls=ls, color=colors[rule], label=f"{rule} {dataset}")
            axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="x", ls=ls, color=colors[rule], label=f"{rule} {dataset}")
    for ax, title in [(axes[0], "Specificity gain"), (axes[1], "Sensitivity loss")]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    sub = dist[dist["dataset"].eq("sdata_external") & dist["operating_point"].eq("S85")].copy()
    sub = sub.sort_values("n", ascending=False).head(8)
    axes[2].bar(np.arange(len(sub)), sub["event_rate"] * 100, color="#756bb1")
    axes[2].set_xticks(np.arange(len(sub)))
    axes[2].set_xticklabels(sub["pattern"].tolist(), rotation=45, ha="right")
    axes[2].set_ylabel("Low SMI %")
    axes[2].set_title("External S85 pattern event rate", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    feature_g, feature_s, _ = locked_targets(g, s, c_g)
    target_g, cpos_g = exact_feature_votes(g["y"], c_g, thresholds, feature_g)
    target_s, cpos_s = exact_feature_votes(s["y"], c_s, thresholds, feature_s)
    exact_mask = sum(1 << code for code in range(16) if popcount(code) >= 2)
    exact_detail = evaluate_pattern_gate(
        "exact_locked_2of4",
        exact_mask,
        g,
        s,
        cpos_g,
        cpos_s,
        votes_to_codes(target_g.astype(bool)),
        votes_to_codes(target_s.astype(bool)),
    )

    probs, logs = load_or_train_probabilities(g, s, c_g, c_s, thresholds)
    logs.to_csv(OUT_DIR / "pattern_gate_training_log.csv", index=False)

    all_summary = []
    best_detail_by_config = {}
    best_dist_by_config = {}
    for config_name, val in probs.items():
        print(f"searching patterns {config_name}", flush=True)
        summary, best_detail, dist = search_pattern_gate(config_name, val["prob_g"], val["prob_s"], g, s, cpos_g, cpos_s)
        summary.to_csv(OUT_DIR / f"{config_name}_pattern_search_summary.csv", index=False)
        all_summary.append(summary.assign(config=config_name))
        best_detail_by_config[config_name] = best_detail
        best_dist_by_config[config_name] = dist

    summary_all = pd.concat(all_summary, ignore_index=True).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best = summary_all.iloc[0]
    best_config = str(best["config"])
    best_detail = pd.concat([exact_detail, best_detail_by_config[best_config]], ignore_index=True)
    best_dist = best_dist_by_config[best_config]

    summary_all.to_csv(OUT_DIR / "pattern_gate_model_selection_summary.csv", index=False)
    best_detail.to_csv(OUT_DIR / "pattern_gate_best_deescalation_details.csv", index=False)
    best_dist.to_csv(OUT_DIR / "pattern_gate_best_pattern_distribution.csv", index=False)
    plot_best(best_detail, best_dist, OUT_DIR / "pattern_gate_best_plot.png")
    with (OUT_DIR / "pattern_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "regions_1_indexed_inclusive": REGIONS,
                "selected_config": best_config,
                "selected_thresholds": {
                    "R1": float(best["threshold_R1"]),
                    "R2": float(best["threshold_R2"]),
                    "R3": float(best["threshold_R3"]),
                    "R4": float(best["threshold_R4"]),
                },
                "selected_patterns": str(best["patterns"]),
                "rule": "CNN branch probabilities are thresholded into 16 +/- morphology patterns; internal-only pattern set is locked and applied externally.",
            },
            f,
            indent=2,
        )

    print("\nMODEL SUMMARY")
    print(summary_all.head(20).to_string(index=False))
    print("\nBEST DE-ESCALATION")
    show = [
        "rule",
        "dataset",
        "operating_point",
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
        "deesc_event_fisher_p",
        "features",
    ]
    print(best_detail[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    main()
