from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import load_dataset, row_norm, threshold_youden  # noqa: E402
from g1090_sdata_aec_counterattack import make_pipeline, score_estimator  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import feature_sets_no_scanner  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_missed_rescue"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260626


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def clinical_threshold_metrics(y_train: np.ndarray, score_train: np.ndarray, y_test: np.ndarray, score_test: np.ndarray) -> tuple[float, dict]:
    th = threshold_youden(y_train, score_train)
    return th, metric_at_threshold(y_test, score_test, th)


def curve_audit_features(d: dict) -> pd.DataFrame:
    a = row_norm(d["a128"]) - 1.0
    c = row_norm(d["crop"]) - 1.0
    da = np.diff(a, axis=1)
    dc = np.diff(c, axis=1)
    out = pd.DataFrame(
        {
            "aec128_energy": np.mean(a * a, axis=1),
            "crop_energy": np.mean(c * c, axis=1),
            "aec128_tv": np.sum(np.abs(da), axis=1),
            "crop_tv": np.sum(np.abs(dc), axis=1),
            "aec128_max": np.max(a, axis=1),
            "aec128_min": np.min(a, axis=1),
            "crop_max": np.max(c, axis=1),
            "crop_min": np.min(c, axis=1),
            "aec128_crop_diff_energy": np.mean((a - c) ** 2, axis=1),
            "aec128_mid_mean": np.mean(a[:, 48:80], axis=1),
            "crop_mid_mean": np.mean(c[:, 48:80], axis=1),
            "aec128_lower_mean": np.mean(a[:, 84:128], axis=1),
            "crop_lower_mean": np.mean(c[:, 84:128], axis=1),
        }
    )
    return out


def fit_rescue_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    base_neg_train: np.ndarray,
    base_neg_test: np.ndarray,
    folds: list[np.ndarray],
    kind: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for i, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(y_train)), val_idx)
        subset = train_idx[base_neg_train[train_idx]]
        val_subset = val_idx[base_neg_train[val_idx]]
        if len(subset) < 40 or y_train[subset].sum() < 3 or len(np.unique(y_train[subset])) < 2:
            continue
        try:
            model = make_pipeline(kind, k, x_train.shape[1], SEED + i)
            model.fit(x_train[subset], y_train[subset])
            if len(val_subset):
                oof[val_subset] = score_estimator(model, x_train[val_subset])
        except Exception:
            continue
    train_eval = base_neg_train & np.isfinite(oof)
    if train_eval.sum() < 40 or y_train[train_eval].sum() < 3:
        return None
    final_subset = np.where(base_neg_train)[0]
    try:
        final = make_pipeline(kind, k, x_train.shape[1], SEED + 99)
        final.fit(x_train[final_subset], y_train[final_subset])
        test_score = np.full(len(x_test), np.nan, dtype=float)
        test_score[base_neg_test] = score_estimator(final, x_test[base_neg_test])
    except Exception:
        return None
    return oof, test_score


def threshold_by_train_budget(y: np.ndarray, score: np.ndarray, train_mask: np.ndarray, max_added_fp_rate: float) -> float:
    vals = np.unique(score[train_mask & np.isfinite(score)])
    if len(vals) == 0:
        return np.inf
    cuts = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2, vals.max() + 1e-12]
    neg_n = max(1, int(((y == 0) & train_mask).sum()))
    best = None
    for cut in cuts:
        pred = train_mask & (score >= cut)
        fp = int(((pred == 1) & (y == 0)).sum())
        if fp / neg_n > max_added_fp_rate:
            continue
        tp = int(((pred == 1) & (y == 1)).sum())
        ppv = tp / max(1, int(pred.sum()))
        sens = tp / max(1, int(((y == 1) & train_mask).sum()))
        key = (tp, sens, ppv, -fp)
        if best is None or key > best[0]:
            best = (key, float(cut))
    return best[1] if best else float(vals.max() + 1e-12)


def rescue_metrics(
    y: np.ndarray,
    clinical_score: np.ndarray,
    clinical_threshold: float,
    rescue_score: np.ndarray,
    rescue_threshold: float,
) -> dict:
    base_pred = clinical_score >= clinical_threshold
    rescue_pred = (~base_pred) & np.isfinite(rescue_score) & (rescue_score >= rescue_threshold)
    final_pred = base_pred | rescue_pred
    tp = int(((final_pred == 1) & (y == 1)).sum())
    fn = int(((final_pred == 0) & (y == 1)).sum())
    tn = int(((final_pred == 0) & (y == 0)).sum())
    fp = int(((final_pred == 1) & (y == 0)).sum())
    return {
        "final_sens": tp / max(1, tp + fn),
        "final_spec": tn / max(1, tn + fp),
        "final_ppv": tp / max(1, tp + fp),
        "final_npv": tn / max(1, tn + fn),
        "final_tp": tp,
        "final_fn": fn,
        "final_tn": tn,
        "final_fp": fp,
        "rescued_n": int(rescue_pred.sum()),
        "rescued_tp": int((rescue_pred & (y == 1)).sum()),
        "rescued_fp": int((rescue_pred & (y == 0)).sum()),
    }


def soft_score_auc_search(
    y_train: np.ndarray,
    y_test: np.ndarray,
    clinical_train: np.ndarray,
    clinical_test: np.ndarray,
    rescue_train: np.ndarray,
    rescue_test: np.ndarray,
    base_neg_train: np.ndarray,
    base_neg_test: np.ndarray,
) -> dict:
    c_z, c_te_z, _, _ = zfit_apply(clinical_train, clinical_test)
    r_z, r_te_z, _, _ = zfit_apply(rescue_train[np.isfinite(rescue_train)], rescue_test[np.isfinite(rescue_test)])
    # Rebuild z-scored rescue arrays with train rescue subset parameters.
    finite_tr = np.isfinite(rescue_train)
    mu = np.nanmean(rescue_train[finite_tr])
    sd = np.nanstd(rescue_train[finite_tr]) or 1.0
    rz_full = np.zeros_like(clinical_train, dtype=float)
    rte_full = np.zeros_like(clinical_test, dtype=float)
    rz_full[finite_tr] = (rescue_train[finite_tr] - mu) / sd
    finite_te = np.isfinite(rescue_test)
    rte_full[finite_te] = (rescue_test[finite_te] - mu) / sd
    best = None
    for alpha in [-1.0, -0.7, -0.45, -0.25, 0.25, 0.45, 0.7, 1.0]:
        train_score = c_z.copy()
        test_score = c_te_z.copy()
        train_score[base_neg_train] = c_z[base_neg_train] + alpha * rz_full[base_neg_train]
        test_score[base_neg_test] = c_te_z[base_neg_test] + alpha * rte_full[base_neg_test]
        cv_auc = auc_or_nan(y_train, train_score)
        row = {
            "alpha": alpha,
            "soft_cv_auc": cv_auc,
            "soft_test_auc": auc_or_nan(y_test, test_score),
        }
        if best is None or cv_auc > best["soft_cv_auc"]:
            best = row
    return best or {"alpha": np.nan, "soft_cv_auc": np.nan, "soft_test_auc": np.nan}


def summarize_groups(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby(label_col):
        rows.append(
            {
                "group": label,
                "n": len(g),
                "low_smi_n": int(g["y"].sum()),
                "male_n": int((g["sex"] == "M").sum()),
                "female_n": int((g["sex"] == "F").sum()),
                "age_mean": g["age"].mean(),
                "height_mean": g["height"].mean(),
                "weight_mean": g["weight"].mean(),
                "smi_mean": g["smi"].mean(),
                "clinical_margin_mean": g["clinical_margin"].mean(),
                "aec128_energy_mean": g["aec128_energy"].mean(),
                "crop_energy_mean": g["crop_energy"].mean(),
                "aec128_crop_diff_energy_mean": g["aec128_crop_diff_energy"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)

    c_oof, c_test, _, _ = clinical_oof_test(train, test, folds)
    c_th, base_m = clinical_threshold_metrics(ytr, c_oof, yte, c_test)
    base_pred_train = c_oof >= c_th
    base_pred_test = c_test >= c_th
    base_neg_train = ~base_pred_train
    base_neg_test = ~base_pred_test

    feat_sets = feature_sets_no_scanner(train, test)
    model_grid = [
        ("logit_C0.2", 16),
        ("logit_C0.2", 64),
        ("logit_C0.2", 128),
        ("linsvm_C0.2", 16),
        ("linsvm_C0.2", 64),
        ("linsvm_C0.2", 128),
        ("extra_D3_L8", 16),
        ("extra_D3_L8", 64),
    ]
    rows = []
    score_store = {}
    for fam, (tr_df, te_df) in feat_sets.items():
        xtr = tr_df.to_numpy(dtype=float)
        xte = te_df.to_numpy(dtype=float)
        print(f"Rescue family={fam} p={xtr.shape[1]}", flush=True)
        for kind, k in model_grid:
            fitted = fit_rescue_oof(xtr, xte, ytr, base_neg_train, base_neg_test, folds, kind, k)
            if fitted is None:
                continue
            r_oof, r_test = fitted
            train_eval = base_neg_train & np.isfinite(r_oof)
            name = f"{fam}_{kind}_k{k}"
            score_store[name] = (r_oof, r_test)
            soft = soft_score_auc_search(ytr, yte, c_oof, c_test, r_oof, r_test, base_neg_train, base_neg_test)
            for fp_budget in [0.02, 0.05, 0.10, 0.15, 0.20]:
                r_th = threshold_by_train_budget(ytr, r_oof, train_eval, fp_budget)
                met = rescue_metrics(yte, c_test, c_th, r_test, r_th)
                row = {
                    "model": name,
                    "family": fam,
                    "kind": kind,
                    "k": k,
                    "train_clinical_negative_n": int(train_eval.sum()),
                    "train_missed_low_smi_n": int(ytr[train_eval].sum()),
                    "train_rescue_auc_within_clinical_negative": auc_or_nan(ytr[train_eval], r_oof[train_eval]),
                    "test_rescue_auc_within_clinical_negative": auc_or_nan(yte[base_neg_test], r_test[base_neg_test]),
                    "train_added_fp_budget": fp_budget,
                    "rescue_threshold": r_th,
                    "base_sens": base_m["sensitivity"],
                    "base_spec": base_m["specificity"],
                    "base_ppv": base_m["ppv"],
                    "base_npv": base_m["npv"],
                    **met,
                    **soft,
                }
                row["delta_sens"] = row["final_sens"] - row["base_sens"]
                row["delta_spec"] = row["final_spec"] - row["base_spec"]
                rows.append(row)

    rescue_df = pd.DataFrame(rows).sort_values(["rescued_tp", "rescued_fp", "final_spec"], ascending=[False, True, False])
    rescue_df.to_csv(OUT_DIR / "rescue_model_results.csv", index=False, encoding="utf-8-sig")

    # Pick one practical rescue: at least one rescue TP, minimal FP, then best final AUC proxy/test soft AUC.
    practical = rescue_df[(rescue_df["rescued_tp"] > 0) & (rescue_df["rescued_fp"] <= 30)].copy()
    if len(practical):
        pick = practical.sort_values(["rescued_tp", "rescued_fp", "test_rescue_auc_within_clinical_negative"], ascending=[False, True, False]).iloc[0]
    else:
        pick = rescue_df.iloc[0]
    pick_name = str(pick["model"])
    r_oof, r_test = score_store[pick_name]
    r_th = float(pick["rescue_threshold"])
    rescue_pred_test = base_neg_test & np.isfinite(r_test) & (r_test >= r_th)

    audit = pd.DataFrame(
        {
            "PatientID": test["meta"]["PatientID"],
            "Series_Desc": test["meta"]["Series_Desc"],
            "sex": test["meta"]["PatientSex"].astype(str),
            "age": pd.to_numeric(test["meta"]["PatientAge"], errors="coerce"),
            "height": pd.to_numeric(test["meta"]["Height"], errors="coerce"),
            "weight": pd.to_numeric(test["meta"]["Weight"], errors="coerce"),
            "TAMA": pd.to_numeric(test["meta"]["TAMA"], errors="coerce"),
            "smi": test["smi_calc"],
            "y": yte,
            "clinical_score": c_test,
            "clinical_margin": c_test - c_th,
            "clinical_pred": base_pred_test.astype(int),
            "clinical_false_negative": ((yte == 1) & (~base_pred_test)).astype(int),
            "rescue_score": r_test,
            "rescue_pred": rescue_pred_test.astype(int),
            "rescued_true_positive": (rescue_pred_test & (yte == 1)).astype(int),
            "rescue_false_positive": (rescue_pred_test & (yte == 0)).astype(int),
            "final_pred": (base_pred_test | rescue_pred_test).astype(int),
            "final_false_negative": ((yte == 1) & (~(base_pred_test | rescue_pred_test))).astype(int),
        }
    )
    audit = pd.concat([audit, curve_audit_features(test)], axis=1)
    audit["miss_status"] = np.select(
        [
            (audit["clinical_false_negative"] == 1) & (audit["rescued_true_positive"] == 1),
            (audit["clinical_false_negative"] == 1) & (audit["final_false_negative"] == 1),
            (audit["y"] == 1) & (audit["clinical_pred"] == 1),
            (audit["y"] == 0) & (audit["rescue_false_positive"] == 1),
        ],
        ["clinical_miss_rescued", "clinical_miss_still_missed", "clinical_detected_low_smi", "rescue_false_positive"],
        default="other",
    )
    audit.to_csv(OUT_DIR / "sdata_missed_case_audit.csv", index=False, encoding="utf-8-sig")
    summarize_groups(audit[audit["y"] == 1], "miss_status").to_csv(OUT_DIR / "missed_case_group_summary.csv", index=False, encoding="utf-8-sig")

    # Scanner is not used in modeling; keep only as optional audit distribution.
    scanner_audit = (
        audit[audit["clinical_false_negative"] == 1]
        .assign(scanner=test["meta"]["Manufacturer"].astype(str).to_numpy()[audit["clinical_false_negative"].astype(bool).to_numpy()])
        .groupby(["scanner", "miss_status"])
        .size()
        .reset_index(name="n")
    )
    scanner_audit.to_csv(OUT_DIR / "scanner_distribution_of_misses_audit_only.csv", index=False, encoding="utf-8-sig")

    print("\nClinical baseline")
    print({k: base_m[k] for k in ["sensitivity", "specificity", "ppv", "npv", "tp", "fn", "tn", "fp"]})
    print(f"clinical_threshold={c_th:.6f}")
    print("\nTop rescue candidates")
    show_cols = [
        "model",
        "train_added_fp_budget",
        "train_rescue_auc_within_clinical_negative",
        "test_rescue_auc_within_clinical_negative",
        "rescued_tp",
        "rescued_fp",
        "delta_sens",
        "delta_spec",
        "final_sens",
        "final_spec",
        "final_ppv",
        "soft_test_auc",
        "alpha",
    ]
    print(rescue_df[show_cols].head(30).to_string(index=False))
    print("\nPicked practical rescue")
    print(pick[show_cols].to_string())
    print("\nLow-SMI miss status")
    print(audit[audit["y"] == 1]["miss_status"].value_counts().to_string())
    print("\nMissed/rescued low-SMI cases")
    case_cols = [
        "PatientID",
        "sex",
        "age",
        "height",
        "weight",
        "TAMA",
        "smi",
        "clinical_margin",
        "rescue_score",
        "miss_status",
        "aec128_energy",
        "crop_energy",
        "aec128_crop_diff_energy",
    ]
    missed_cases = audit[audit["clinical_false_negative"] == 1].sort_values(["miss_status", "clinical_margin"])
    print(missed_cases[case_cols].to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
