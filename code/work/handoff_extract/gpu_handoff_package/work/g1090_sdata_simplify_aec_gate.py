from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_simplified_aec_gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import load_dataset, threshold_youden  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import clinical_gate_base, expert_oof_test, feature_sets_no_scanner  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 20260626


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def eval_score(name: str, ytr: np.ndarray, yte: np.ndarray, tr_score: np.ndarray, te_score: np.ndarray) -> dict:
    th = threshold_youden(ytr, tr_score)
    m = metric_at_threshold(yte, te_score, th)
    return {
        "model": name,
        "threshold_source": "train_youden",
        "threshold": th,
        "cv_auc": auc_or_nan(ytr, tr_score),
        "test_auc": auc_or_nan(yte, te_score),
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "ppv": m["ppv"],
        "npv": m["npv"],
        "tp": m["tp"],
        "fn": m["fn"],
        "tn": m["tn"],
        "fp": m["fp"],
    }


def eval_binary(name: str, y: np.ndarray, pred: np.ndarray, extra: dict | None = None) -> dict:
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    out = {
        "rule": name,
        "n_positive": int(pred.sum()),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }
    if extra:
        out.update(extra)
    return out


def deescalation_threshold(
    y: np.ndarray,
    clinical_positive: np.ndarray,
    aec_score: np.ndarray,
    max_event_rate: float,
) -> float:
    scores = np.sort(np.unique(aec_score[clinical_positive]))
    chosen = None
    for th in scores:
        m = clinical_positive & (aec_score <= th)
        if m.sum() == 0:
            continue
        if float(y[m].mean()) <= max_event_rate:
            chosen = float(th)
    return float(scores.min() - 1e-12) if chosen is None else chosen


def high_priority_threshold(
    y: np.ndarray,
    clinical_positive: np.ndarray,
    aec_score: np.ndarray,
    min_ppv: float,
) -> float:
    scores = np.sort(np.unique(aec_score[clinical_positive]))
    for th in scores:
        m = clinical_positive & (aec_score >= th)
        if m.sum() == 0:
            continue
        if float(y[m].mean()) >= min_ppv:
            return float(th)
    return float(scores.max() + 1e-12)


def group_row(name: str, y: np.ndarray, mask: np.ndarray, extra: dict | None = None) -> dict:
    n = int(mask.sum())
    events = int(y[mask].sum())
    out = {
        "group": name,
        "n": n,
        "events": events,
        "non_events": n - events,
        "prevalence": events / n if n else np.nan,
    }
    if extra:
        out.update(extra)
    return out


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)

    clinical_tr, clinical_te, _, _ = clinical_oof_test(train, test, folds)
    feats = feature_sets_no_scanner(train, test)
    xtr = feats["direct_curve"][0].to_numpy(dtype=float)
    xte = feats["direct_curve"][1].to_numpy(dtype=float)
    aec_tr, aec_te = expert_oof_test(xtr, xte, ytr, folds, "linsvm_C0.2", 128)

    clinical_tr_z, clinical_te_z, _, _ = zfit_apply(clinical_tr, clinical_te)
    aec_tr_z, aec_te_z, _, _ = zfit_apply(aec_tr, aec_te)

    clinical_th = threshold_youden(ytr, clinical_tr)
    clinical_pos_tr = clinical_tr >= clinical_th
    clinical_pos_te = clinical_te >= clinical_th

    rows = []
    rows.append(eval_score("clinical_only", ytr, yte, clinical_tr, clinical_te))
    rows.append(eval_score("aec_expert_only_direct_curve_linsvm", ytr, yte, aec_tr, aec_te))

    # Simple two-score stack: no sex-specific term, no Gaussian boundary, no hand-set lambda.
    stack = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=4000)
    stack.fit(np.column_stack([clinical_tr_z, aec_tr_z]), ytr)
    stack_tr = stack.decision_function(np.column_stack([clinical_tr_z, aec_tr_z]))
    stack_te = stack.decision_function(np.column_stack([clinical_te_z, aec_te_z]))
    stack_row = eval_score("simple_logistic_stack_clinical_plus_aec", ytr, yte, stack_tr, stack_te)
    stack_row.update(
        {
            "coef_clinical_z": float(stack.coef_[0, 0]),
            "coef_aec_z": float(stack.coef_[0, 1]),
            "intercept": float(stack.intercept_[0]),
        }
    )
    rows.append(stack_row)

    sex_tr_arr = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_te_arr = test["meta"]["PatientSex"].astype(str).to_numpy()
    female_tr_float = (sex_tr_arr == "F").astype(float)
    female_te_float = (sex_te_arr == "F").astype(float)
    x_meta_tr = np.column_stack([clinical_tr_z, aec_tr_z, female_tr_float, aec_tr_z * female_tr_float])
    x_meta_te = np.column_stack([clinical_te_z, aec_te_z, female_te_float, aec_te_z * female_te_float])
    inter = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=4000)
    inter.fit(x_meta_tr, ytr)
    inter_tr = inter.decision_function(x_meta_tr)
    inter_te = inter.decision_function(x_meta_te)
    inter_row = eval_score("classic_meta_logistic_clinical_aec_sex_interaction", ytr, yte, inter_tr, inter_te)
    inter_row.update(
        {
            "coef_clinical_z": float(inter.coef_[0, 0]),
            "coef_aec_z": float(inter.coef_[0, 1]),
            "coef_female": float(inter.coef_[0, 2]),
            "coef_aec_x_female": float(inter.coef_[0, 3]),
            "intercept": float(inter.intercept_[0]),
        }
    )
    rows.append(inter_row)

    clinical_z_th = threshold_youden(ytr, clinical_tr_z)
    sex_tr = sex_tr_arr
    sex_te = sex_te_arr
    female_tr = sex_tr == "F"
    female_te = sex_te == "F"

    # More interpretable replacement for Gaussian weighting:
    # use AEC only inside a hard clinical indeterminate zone.
    for width in [0.50, 0.75, 1.00, 1.25]:
        zone_tr = np.abs(clinical_tr_z - clinical_z_th) <= width
        zone_te = np.abs(clinical_te_z - clinical_z_th) <= width
        rows.append(
            eval_score(
                f"hard_indeterminate_zone_all_sex_w{width:.2f}_lambda025",
                ytr,
                yte,
                clinical_tr_z + 0.25 * zone_tr * aec_tr_z,
                clinical_te_z + 0.25 * zone_te * aec_te_z,
            )
        )
        rows.append(
            eval_score(
                f"hard_indeterminate_zone_female_only_w{width:.2f}_lambda025",
                ytr,
                yte,
                clinical_tr_z + 0.25 * zone_tr * female_tr * aec_tr_z,
                clinical_te_z + 0.25 * zone_te * female_te * aec_te_z,
            )
        )

    # Previous best, included only as a benchmark for what the simpler approach gives up.
    gates = clinical_gate_base(ytr, clinical_tr, clinical_te, sex_tr, sex_te)
    gtr, gte = gates["female_boundary"]
    old_tr = clinical_tr_z + 0.25 * gtr * aec_tr_z
    old_te = clinical_te_z + 0.25 * gte * aec_te_z
    rows.append(eval_score("old_benchmark_female_gaussian_boundary_lambda025", ytr, yte, old_tr, old_te))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "score_model_comparison.csv", index=False, encoding="utf-8-sig")

    band_rows = []
    binary_rows = []

    band_rows.append(group_row("all_sdata", yte, np.ones_like(yte, dtype=bool)))
    band_rows.append(group_row("clinical_positive", yte, clinical_pos_te))
    band_rows.append(group_row("clinical_negative", yte, ~clinical_pos_te))

    binary_rows.append(eval_binary("clinical_positive_baseline", yte, clinical_pos_te))

    for cap in [0.03, 0.05, 0.075, 0.10, 0.125]:
        th = deescalation_threshold(ytr, clinical_pos_tr, aec_tr_z, cap)
        low_tr = clinical_pos_tr & (aec_tr_z <= th)
        low_te = clinical_pos_te & (aec_te_z <= th)
        pred = clinical_pos_te & ~low_te
        band_rows.append(
            group_row(
                f"simple_aec_low_risk_cap_{cap:.3f}",
                yte,
                low_te,
                {
                    "threshold": th,
                    "train_n": int(low_tr.sum()),
                    "train_events": int(ytr[low_tr].sum()),
                    "train_prevalence": float(ytr[low_tr].mean()) if low_tr.sum() else np.nan,
                    "clinical_positive_removed_frac": float(low_te.sum() / clinical_pos_te.sum()),
                },
            )
        )
        binary_rows.append(
            eval_binary(
                f"clinical_positive_minus_simple_aec_lowrisk_cap_{cap:.3f}",
                yte,
                pred,
                {
                    "threshold": th,
                    "removed_n": int(low_te.sum()),
                    "removed_events": int(yte[low_te].sum()),
                    "removed_prevalence": float(yte[low_te].mean()) if low_te.sum() else np.nan,
                },
            )
        )

    for target in [0.30, 0.35, 0.40, 0.45, 0.50]:
        th = high_priority_threshold(ytr, clinical_pos_tr, aec_tr_z, target)
        high_tr = clinical_pos_tr & (aec_tr_z >= th)
        high_te = clinical_pos_te & (aec_te_z >= th)
        band_rows.append(
            group_row(
                f"simple_aec_high_priority_ppv_{target:.2f}",
                yte,
                high_te,
                {
                    "threshold": th,
                    "train_n": int(high_tr.sum()),
                    "train_events": int(ytr[high_tr].sum()),
                    "train_prevalence": float(ytr[high_tr].mean()) if high_tr.sum() else np.nan,
                    "clinical_positive_kept_frac": float(high_te.sum() / clinical_pos_te.sum()),
                },
            )
        )
        binary_rows.append(
            eval_binary(
                f"simple_aec_high_priority_only_ppv_{target:.2f}",
                yte,
                high_te,
                {
                    "threshold": th,
                    "target_train_ppv": target,
                },
            )
        )

    # A clean three-band version for manuscript framing.
    low_th = deescalation_threshold(ytr, clinical_pos_tr, aec_tr_z, 0.10)
    high_th = high_priority_threshold(ytr, clinical_pos_tr, aec_tr_z, 0.45)
    low = clinical_pos_te & (aec_te_z <= low_th)
    high = clinical_pos_te & (aec_te_z >= high_th)
    mid = clinical_pos_te & ~(low | high)
    three = pd.DataFrame(
        [
            group_row("clinical_negative", yte, ~clinical_pos_te),
            group_row("clinical_positive_low_risk_by_aec", yte, low, {"low_threshold": low_th}),
            group_row("clinical_positive_intermediate_by_aec", yte, mid, {"low_threshold": low_th, "high_threshold": high_th}),
            group_row("clinical_positive_high_priority_by_aec", yte, high, {"high_threshold": high_th}),
        ]
    )
    three.to_csv(OUT_DIR / "simple_three_band_reclassification.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(band_rows).to_csv(OUT_DIR / "simple_aec_triage_bands.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(binary_rows).to_csv(OUT_DIR / "simple_aec_binary_rules.csv", index=False, encoding="utf-8-sig")

    print("\nScore model comparison")
    print(summary.to_string(index=False))
    print("\nSimple three-band reclassification")
    print(three.to_string(index=False))
    print("\nSimple AEC binary rules")
    print(pd.DataFrame(binary_rows).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
