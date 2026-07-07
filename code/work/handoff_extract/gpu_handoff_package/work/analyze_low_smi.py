from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def logit_fit(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-5,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float) * ridge
    penalty[0] = 0.0
    for _ in range(max_iter):
        eta = x @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        z = eta + (y - p) / w
        xw = x * w[:, None]
        h = x.T @ xw + np.diag(penalty)
        rhs = x.T @ (w * z)
        try:
            beta_new = np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(h) @ rhs
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def log_likelihood(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    p = np.clip(sigmoid(x @ beta), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def design_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_index: np.ndarray | None = None,
    levels: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict]:
    if train_index is None:
        train_index = np.arange(len(df))
    if levels is None:
        levels = {}
        for c in categorical_cols:
            vals = sorted([str(v) for v in df.iloc[train_index][c].dropna().unique()])
            levels[c] = vals

    pieces = [np.ones((len(df), 1), dtype=float)]
    meta = {"numeric": {}, "levels": levels}
    for c in numeric_cols:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        tr = values[train_index]
        mu = float(np.nanmean(tr))
        sd = float(np.nanstd(tr))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        filled = np.where(np.isfinite(values), values, mu)
        pieces.append(((filled - mu) / sd)[:, None])
        meta["numeric"][c] = {"mean": mu, "sd": sd}

    for c in categorical_cols:
        vals = df[c].astype(str).fillna("__NA__").to_numpy()
        cats = levels[c]
        # Drop the first category to avoid exact collinearity with intercept.
        for cat in cats[1:]:
            pieces.append((vals == cat).astype(float)[:, None])

    return np.hstack(pieces), meta


def transform_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    meta: dict,
) -> np.ndarray:
    pieces = [np.ones((len(df), 1), dtype=float)]
    for c in numeric_cols:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mu = meta["numeric"][c]["mean"]
        sd = meta["numeric"][c]["sd"]
        filled = np.where(np.isfinite(values), values, mu)
        pieces.append(((filled - mu) / sd)[:, None])
    for c in categorical_cols:
        vals = df[c].astype(str).fillna("__NA__").to_numpy()
        cats = meta["levels"][c]
        for cat in cats[1:]:
            pieces.append((vals == cat).astype(float)[:, None])
    return np.hstack(pieces)


def make_stratified_folds(y: np.ndarray, k: int = 5, seed: int = 20260621) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]


def cv_predict_logit(
    df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    folds: list[np.ndarray],
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    all_idx = np.arange(len(y))
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx)
        x_all, meta = design_matrix(df, numeric_cols, categorical_cols, train_idx)
        x_train = x_all[train_idx]
        x_test = x_all[test_idx]
        beta = logit_fit(x_train, y[train_idx])
        pred[test_idx] = sigmoid(x_test @ beta)
    return pred


def fit_full_lr_p(
    df: pd.DataFrame,
    y: np.ndarray,
    base_numeric: list[str],
    base_cat: list[str],
    feature_col: str,
) -> tuple[float, float]:
    x0, _ = design_matrix(df, base_numeric, base_cat)
    b0 = logit_fit(x0, y)
    ll0 = log_likelihood(x0, y, b0)
    x1, _ = design_matrix(df, base_numeric + [feature_col], base_cat)
    b1 = logit_fit(x1, y)
    ll1 = log_likelihood(x1, y, b1)
    lr = max(0.0, 2.0 * (ll1 - ll0))
    # chi-square(df=1) survival function.
    p = math.erfc(math.sqrt(lr / 2.0))
    return lr, p


def metric_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "pred_positive_rate": float(np.mean(pred)),
    }


def youden_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    vals = np.unique(score[np.isfinite(score)])
    candidates = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
    best = None
    for thr in candidates:
        m = metric_at_threshold(y, score, float(thr))
        j = m["sensitivity"] + m["specificity"] - 1.0
        if best is None or j > best[0]:
            best = (j, thr, m)
    return float(best[1]), best[2]


def best_threshold_with_spec(
    y: np.ndarray,
    score: np.ndarray,
    min_spec: float,
) -> tuple[float, dict]:
    vals = np.unique(score[np.isfinite(score)])
    candidates = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
    best = None
    for thr in candidates:
        m = metric_at_threshold(y, score, float(thr))
        if m["specificity"] + 1e-12 < min_spec:
            continue
        key = (m["sensitivity"], m["specificity"], -m["pred_positive_rate"])
        if best is None or key > best[0]:
            best = (key, thr, m)
    if best is None:
        # Fall back to the most conservative threshold if no candidate clears min_spec.
        thr = vals.max() + 1e-12
        return float(thr), metric_at_threshold(y, score, float(thr))
    return float(best[1]), best[2]


def logsumexp(logps: Iterable[float]) -> float:
    arr = np.array(list(logps), dtype=float)
    if len(arr) == 0:
        return -math.inf
    m = float(np.max(arr))
    return m + math.log(float(np.sum(np.exp(arr - m))))


def binom_tail(k: int, n: int, side: str) -> float:
    if n <= 0:
        return 1.0
    if side == "upper":
        rng = range(k, n + 1)
    else:
        rng = range(0, k + 1)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in rng
    ]
    return min(1.0, math.exp(logsumexp(logs)))


def mcnemar_one_sided_gain(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    # For sensitivity: arrays are positive-call indicators among true positives.
    # For specificity: arrays are correct-negative indicators among true negatives.
    gain = int(np.sum((reference == 0) & (candidate == 1)))
    loss = int(np.sum((reference == 1) & (candidate == 0)))
    n = gain + loss
    p = binom_tail(gain, n, "upper") if n else 1.0
    return gain, loss, p


def mcnemar_one_sided_loss(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int, float]:
    gain = int(np.sum((reference == 0) & (candidate == 1)))
    loss = int(np.sum((reference == 1) & (candidate == 0)))
    n = gain + loss
    p = binom_tail(loss, n, "upper") if n else 1.0
    return gain, loss, p


def bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    q = np.ones(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        i = n - rank + 1
        val = min(prev, pvals[idx] * n / i)
        q[idx] = val
        prev = val
    return q.tolist()


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks for ties.
    values, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for g in np.flatnonzero(counts > 1):
            idx = np.flatnonzero(inv == g)
            ranks[idx] = ranks[idx].mean()
    rpos = ranks[y == 1].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def add_curve_features(meta: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    aec_cols = [c for c in curve.columns if c.startswith("aec_")]
    x = curve[aec_cols].to_numpy(dtype=float)
    feat = pd.DataFrame({"PatientID": curve["PatientID"].to_numpy()})
    feat["aec_mean"] = np.nanmean(x, axis=1)
    feat["aec_median"] = np.nanmedian(x, axis=1)
    feat["aec_sd"] = np.nanstd(x, axis=1)
    feat["aec_cv"] = feat["aec_sd"] / feat["aec_mean"]
    feat["aec_min"] = np.nanmin(x, axis=1)
    feat["aec_max"] = np.nanmax(x, axis=1)
    feat["aec_range"] = feat["aec_max"] - feat["aec_min"]
    feat["aec_p10"] = np.nanpercentile(x, 10, axis=1)
    feat["aec_p25"] = np.nanpercentile(x, 25, axis=1)
    feat["aec_p75"] = np.nanpercentile(x, 75, axis=1)
    feat["aec_p90"] = np.nanpercentile(x, 90, axis=1)
    feat["aec_iqr"] = feat["aec_p75"] - feat["aec_p25"]
    feat["aec_min_pos"] = np.nanargmin(x, axis=1) + 1
    feat["aec_max_pos"] = np.nanargmax(x, axis=1) + 1
    feat["aec_first16_mean"] = np.nanmean(x[:, :16], axis=1)
    feat["aec_q1_mean"] = np.nanmean(x[:, :32], axis=1)
    feat["aec_q2_mean"] = np.nanmean(x[:, 32:64], axis=1)
    feat["aec_q3_mean"] = np.nanmean(x[:, 64:96], axis=1)
    feat["aec_q4_mean"] = np.nanmean(x[:, 96:], axis=1)
    feat["aec_last16_mean"] = np.nanmean(x[:, -16:], axis=1)
    feat["aec_mid64_mean"] = np.nanmean(x[:, 32:96], axis=1)
    feat["aec_early_late_diff"] = feat["aec_first16_mean"] - feat["aec_last16_mean"]
    feat["aec_early_late_ratio"] = feat["aec_first16_mean"] / feat["aec_last16_mean"]
    feat["aec_mid_to_global"] = feat["aec_mid64_mean"] / feat["aec_mean"]
    feat["aec_drop_max_to_min"] = feat["aec_max"] - feat["aec_min"]
    feat["aec_min_to_mean"] = feat["aec_min"] / feat["aec_mean"]
    feat["aec_max_to_mean"] = feat["aec_max"] / feat["aec_mean"]

    for nseg in [8, 16]:
        width = x.shape[1] // nseg
        for i in range(nseg):
            block = x[:, i * width : (i + 1) * width]
            col = f"aec_seg{nseg}_{i+1:02d}_mean"
            feat[col] = np.nanmean(block, axis=1)
            feat[f"{col}_norm"] = feat[col] / feat["aec_mean"]

    for i, c in enumerate(aec_cols, start=1):
        feat[f"aec_point_{i:03d}"] = x[:, i - 1]
        feat[f"aec_point_{i:03d}_norm"] = x[:, i - 1] / feat["aec_mean"]

    if {"n_slices_cropped", "z_range"}.issubset(curve.columns):
        feat["n_slices_cropped"] = pd.to_numeric(curve["n_slices_cropped"], errors="coerce")
        feat["z_range"] = pd.to_numeric(curve["z_range"], errors="coerce")

    base = meta[
        [
            "PatientID",
            "PatientAge",
            "PatientSex",
            "ManufacturerModelName",
            "Height",
            "Weight",
            "BMI",
            "TAMA",
            "IMATA",
            "SMI",
            "mAs",
        ]
    ].copy()
    return base.merge(feat, on="PatientID", how="inner")


def compare_predictions(
    y: np.ndarray,
    ref_pred: np.ndarray,
    cand_pred: np.ndarray,
) -> dict:
    pos = y == 1
    neg = y == 0
    sens_gain, sens_loss, p_sens = mcnemar_one_sided_gain(ref_pred[pos], cand_pred[pos])
    spec_gain, spec_loss, p_spec_loss = mcnemar_one_sided_loss((~ref_pred[neg]).astype(int), (~cand_pred[neg]).astype(int))
    return {
        "sens_reclassified_gain_n": sens_gain,
        "sens_reclassified_loss_n": sens_loss,
        "p_sens_gain": p_sens,
        "spec_correct_gain_n": spec_gain,
        "spec_correct_loss_n": spec_loss,
        "p_spec_loss": p_spec_loss,
    }


def evaluate_score(
    y: np.ndarray,
    score: np.ndarray,
    ref_pred: np.ndarray,
    ref_metrics: dict,
    min_spec: float,
) -> tuple[dict, np.ndarray]:
    thr, m = best_threshold_with_spec(y, score, min_spec)
    pred = score >= thr
    comp = compare_predictions(y, ref_pred, pred)
    return {**m, **comp, "auc": auc_rank(y, score)}, pred


def evaluate_or_reclassification(
    y: np.ndarray,
    ref_pred: np.ndarray,
    feature_score: np.ndarray,
    min_spec: float,
) -> tuple[dict, np.ndarray, float]:
    vals = np.unique(feature_score[np.isfinite(feature_score)])
    candidates = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
    best = None
    for thr in candidates:
        pred = ref_pred | (feature_score >= thr)
        tp = int(np.sum(pred & (y == 1)))
        fn = int(np.sum((~pred) & (y == 1)))
        tn = int(np.sum((~pred) & (y == 0)))
        fp = int(np.sum(pred & (y == 0)))
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        if spec + 1e-12 < min_spec:
            continue
        key = (sens, spec, -np.mean(pred))
        if best is None or key > best[0]:
            m = {
                "threshold": float(thr),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "npv": tn / (tn + fn) if tn + fn else np.nan,
                "pred_positive_rate": float(np.mean(pred)),
            }
            best = (key, m, pred, float(thr))
    if best is None:
        thr = vals.max() + 1e-12
        pred = ref_pred | (feature_score >= thr)
        m = metric_at_threshold(y, pred.astype(float), 0.5)
        best = (None, m, pred, float(thr))
    comp = compare_predictions(y, ref_pred, best[2])
    return {**best[1], **comp}, best[2], best[3]


def top_pair_or_rules(
    df: pd.DataFrame,
    y: np.ndarray,
    ref_pred: np.ndarray,
    top_features: list[str],
    min_spec: float,
    orientations: dict[str, int],
    max_pairs: int = 60,
) -> list[dict]:
    rows = []
    # Precompute per-feature candidate thresholds that keep at least min_spec when ORed with baseline.
    per_feature_rules = {}
    for f in top_features:
        score = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float) * orientations[f]
        vals = np.unique(score[np.isfinite(score)])
        candidates = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
        rules = []
        for thr in candidates:
            add = score >= thr
            pred = ref_pred | add
            tn = np.sum((~pred) & (y == 0))
            fp = np.sum(pred & (y == 0))
            spec = tn / (tn + fp)
            if spec + 1e-12 >= min_spec:
                sens = np.sum(pred & (y == 1)) / np.sum(y == 1)
                rules.append((sens, spec, float(thr), add))
        rules = sorted(rules, key=lambda x: (x[0], x[1]), reverse=True)[:6]
        per_feature_rules[f] = rules

    count = 0
    for i, f1 in enumerate(top_features):
        for f2 in top_features[i + 1 :]:
            if count >= max_pairs:
                break
            count += 1
            best = None
            for r1 in per_feature_rules.get(f1, []):
                for r2 in per_feature_rules.get(f2, []):
                    pred = ref_pred | r1[3] | r2[3]
                    tp = int(np.sum(pred & (y == 1)))
                    fn = int(np.sum((~pred) & (y == 1)))
                    tn = int(np.sum((~pred) & (y == 0)))
                    fp = int(np.sum(pred & (y == 0)))
                    sens = tp / (tp + fn)
                    spec = tn / (tn + fp)
                    if spec + 1e-12 < min_spec:
                        continue
                    key = (sens, spec, -np.mean(pred))
                    if best is None or key > best[0]:
                        m = {
                            "tp": tp,
                            "fn": fn,
                            "tn": tn,
                            "fp": fp,
                            "sensitivity": sens,
                            "specificity": spec,
                            "ppv": tp / (tp + fp) if tp + fp else np.nan,
                            "npv": tn / (tn + fn) if tn + fn else np.nan,
                            "pred_positive_rate": float(np.mean(pred)),
                        }
                        best = (key, m, pred, r1[2], r2[2])
            if best is not None:
                comp = compare_predictions(y, ref_pred, best[2])
                rows.append(
                    {
                        "family": "pair_or_reclass",
                        "candidate": f"{f1} OR {f2}",
                        "feature_1": f1,
                        "feature_2": f2,
                        "orientation_1": "higher risk" if orientations[f1] == 1 else "lower risk",
                        "orientation_2": "higher risk" if orientations[f2] == 1 else "lower risk",
                        "threshold_1_oriented": best[3],
                        "threshold_2_oriented": best[4],
                        **best[1],
                        **comp,
                    }
                )
    return rows


def analyze_endpoint(name: str, df: pd.DataFrame, y: np.ndarray, spec_margin: float = 0.02) -> dict:
    folds = make_stratified_folds(y)
    baselines = {
        "clinical": (["PatientAge", "Weight", "Height"], ["PatientSex"]),
        "clinical_acquisition": (
            ["PatientAge", "Weight", "Height"],
            ["PatientSex", "ManufacturerModelName"],
        ),
    }
    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "PatientID",
            "PatientAge",
            "PatientSex",
            "ManufacturerModelName",
            "Height",
            "Weight",
            "BMI",
            "TAMA",
            "IMATA",
            "SMI",
        }
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 2
    ]
    baseline_rows = []
    candidate_rows = []
    for base_name, (base_num, base_cat) in baselines.items():
        base_score = cv_predict_logit(df, y, base_num, base_cat, folds)
        base_thr, base_m = youden_threshold(y, base_score)
        base_pred = base_score >= base_thr
        base_row = {
            "endpoint": name,
            "baseline": base_name,
            "n": len(y),
            "events": int(np.sum(y)),
            "prevalence": float(np.mean(y)),
            "threshold": base_thr,
            **base_m,
            "auc": auc_rank(y, base_score),
        }
        baseline_rows.append(base_row)
        min_spec = max(0.0, base_m["specificity"] - spec_margin)

        # Single feature rules and models.
        for f in feature_cols:
            raw = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
            if np.nanstd(raw) == 0:
                continue
            for orient, label in [(1, "higher risk"), (-1, "lower risk")]:
                score = raw * orient
                m, _ = evaluate_score(y, score, base_pred, base_m, min_spec)
                candidate_rows.append(
                    {
                        "endpoint": name,
                        "baseline": base_name,
                        "family": "single_feature_rule",
                        "candidate": f,
                        "orientation": label,
                        "baseline_sensitivity": base_m["sensitivity"],
                        "baseline_specificity": base_m["specificity"],
                        "delta_sensitivity": m["sensitivity"] - base_m["sensitivity"],
                        "delta_specificity": m["specificity"] - base_m["specificity"],
                        "lr_stat": np.nan,
                        "lr_p": np.nan,
                        **m,
                    }
                )

            # Logistic baseline + feature.
            pred = cv_predict_logit(df, y, base_num + [f], base_cat, folds)
            m, _ = evaluate_score(y, pred, base_pred, base_m, min_spec)
            try:
                lr_stat, lr_p = fit_full_lr_p(df, y, base_num, base_cat, f)
            except Exception:
                lr_stat, lr_p = np.nan, np.nan
            candidate_rows.append(
                {
                    "endpoint": name,
                    "baseline": base_name,
                    "family": "baseline_plus_feature_model",
                    "candidate": f,
                    "orientation": "model",
                    "baseline_sensitivity": base_m["sensitivity"],
                    "baseline_specificity": base_m["specificity"],
                    "delta_sensitivity": m["sensitivity"] - base_m["sensitivity"],
                    "delta_specificity": m["specificity"] - base_m["specificity"],
                    "lr_stat": lr_stat,
                    "lr_p": lr_p,
                    **m,
                }
            )

            # Reclassification: keep all baseline positives, add feature positives.
            for orient, label in [(1, "higher risk"), (-1, "lower risk")]:
                score = raw * orient
                m, _, _ = evaluate_or_reclassification(y, base_pred, score, min_spec)
                candidate_rows.append(
                    {
                        "endpoint": name,
                        "baseline": base_name,
                        "family": "or_reclassification",
                        "candidate": f,
                        "orientation": label,
                        "baseline_sensitivity": base_m["sensitivity"],
                        "baseline_specificity": base_m["specificity"],
                        "delta_sensitivity": m["sensitivity"] - base_m["sensitivity"],
                        "delta_specificity": m["specificity"] - base_m["specificity"],
                        "lr_stat": np.nan,
                        "lr_p": np.nan,
                        **m,
                    }
                )

        # Pairwise OR reclassification among promising single-feature OR rules.
        single_or = [
            r
            for r in candidate_rows
            if r["endpoint"] == name
            and r["baseline"] == base_name
            and r["family"] == "or_reclassification"
            and r["delta_sensitivity"] > 0
            and r["specificity"] >= min_spec
        ]
        best_by_feature = {}
        for r in sorted(single_or, key=lambda d: (d["p_sens_gain"], -d["delta_sensitivity"])):
            best_by_feature.setdefault(r["candidate"], r)
        top = sorted(
            best_by_feature.values(),
            key=lambda d: (d["p_sens_gain"], -d["delta_sensitivity"], -d["specificity"]),
        )[:12]
        orientations = {r["candidate"]: 1 if r["orientation"] == "higher risk" else -1 for r in top}
        pair_rows = top_pair_or_rules(
            df,
            y,
            base_pred,
            [r["candidate"] for r in top],
            min_spec,
            orientations,
        )
        for r in pair_rows:
            r.update(
                {
                    "endpoint": name,
                    "baseline": base_name,
                    "baseline_sensitivity": base_m["sensitivity"],
                    "baseline_specificity": base_m["specificity"],
                    "delta_sensitivity": r["sensitivity"] - base_m["sensitivity"],
                    "delta_specificity": r["specificity"] - base_m["specificity"],
                    "lr_stat": np.nan,
                    "lr_p": np.nan,
                }
            )
        candidate_rows.extend(pair_rows)

    cand = pd.DataFrame(candidate_rows)
    if not cand.empty:
        cand["q_sens_gain_all"] = bh_fdr(cand["p_sens_gain"].fillna(1.0).tolist())
        cand["comparable_specificity"] = (
            (cand["delta_specificity"] >= -spec_margin - 1e-12)
            & ((cand["p_spec_loss"] >= 0.05) | (cand["delta_specificity"] >= 0))
        )
        cand["significant_gain"] = (
            cand["comparable_specificity"]
            & (cand["delta_sensitivity"] > 0)
            & (cand["p_sens_gain"] < 0.05)
            & (cand["q_sens_gain_all"] < 0.10)
        )
    return {
        "baselines": pd.DataFrame(baseline_rows),
        "candidates": cand,
    }


def main() -> None:
    xlsx = sorted(DATA_DIR.glob("*features.xlsx"))[0]
    metadata = pd.read_excel(xlsx, sheet_name="metadata")
    curve128 = pd.read_excel(xlsx, sheet_name="aec_128")
    df = add_curve_features(metadata, curve128)

    smi_q1_by_sex = df.groupby("PatientSex")["SMI"].quantile(0.25).to_dict()
    tama_q1_by_sex = df.groupby("PatientSex")["TAMA"].quantile(0.25).to_dict()
    imata_q4_by_sex = df.groupby("PatientSex")["IMATA"].quantile(0.75).to_dict()
    y_smi_q1 = np.array([row.SMI <= smi_q1_by_sex[row.PatientSex] for row in df.itertuples()], dtype=int)
    y_tama_q1 = np.array([row.TAMA <= tama_q1_by_sex[row.PatientSex] for row in df.itertuples()], dtype=int)
    y_imata_q4 = np.array([row.IMATA >= imata_q4_by_sex[row.PatientSex] for row in df.itertuples()], dtype=int)
    endpoints = {
        "low_smi_sex_q1": {
            "y": y_smi_q1,
            "definition": "SMI <= sex-specific dataset Q1",
            "thresholds": {str(k): float(v) for k, v in smi_q1_by_sex.items()},
        },
        "low_tama_sex_q1": {
            "y": y_tama_q1,
            "definition": "TAMA <= sex-specific dataset Q1",
            "thresholds": {str(k): float(v) for k, v in tama_q1_by_sex.items()},
        },
        "high_imata_sex_q4": {
            "y": y_imata_q4,
            "definition": "IMATA >= sex-specific dataset Q4",
            "thresholds": {str(k): float(v) for k, v in imata_q4_by_sex.items()},
        },
    }

    all_baselines = []
    all_candidates = []
    endpoint_meta = []
    for name, spec in endpoints.items():
        print(f"Analyzing {name}: events={int(np.sum(spec['y']))}/{len(spec['y'])}")
        res = analyze_endpoint(name, df, spec["y"])
        all_baselines.append(res["baselines"])
        all_candidates.append(res["candidates"])
        endpoint_meta.append(
            {
                "endpoint": name,
                "definition": spec["definition"],
                "events": int(np.sum(spec["y"])),
                "n": int(len(spec["y"])),
                "prevalence": float(np.mean(spec["y"])),
                "thresholds_json": json.dumps(spec["thresholds"], ensure_ascii=False),
            }
        )

    baselines = pd.concat(all_baselines, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    sig = candidates[candidates["significant_gain"]].copy()
    sig = sig.sort_values(
        ["baseline", "p_sens_gain", "delta_sensitivity", "specificity"],
        ascending=[True, True, False, False],
    )
    top = candidates.sort_values(
        ["baseline", "comparable_specificity", "p_sens_gain", "delta_sensitivity"],
        ascending=[True, False, True, False],
    ).groupby(["endpoint", "baseline", "family"], as_index=False).head(20)

    pd.DataFrame(endpoint_meta).to_csv(OUT_DIR / "endpoint_meta.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(OUT_DIR / "baselines.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_DIR / "all_candidates.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(OUT_DIR / "significant_candidates.csv", index=False, encoding="utf-8-sig")
    top.to_csv(OUT_DIR / "top_candidates_by_family.csv", index=False, encoding="utf-8-sig")
    print("Baselines")
    print(baselines.to_string(index=False))
    print("\nSignificant candidates", len(sig))
    cols = [
        "endpoint",
        "baseline",
        "family",
        "candidate",
        "orientation",
        "baseline_sensitivity",
        "baseline_specificity",
        "sensitivity",
        "specificity",
        "delta_sensitivity",
        "delta_specificity",
        "p_sens_gain",
        "q_sens_gain_all",
        "p_spec_loss",
        "lr_p",
    ]
    present = [c for c in cols if c in sig.columns]
    print(sig[present].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
