from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, compare_predictions, make_stratified_folds, metric_at_threshold, sigmoid  # noqa: E402


DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260626
DERSTINE_CUTOFF = {"M": 45.4, "F": 34.4}


def logit_fit_weighted(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float = 1.0,
    weights: np.ndarray | None = None,
    max_iter: int = 80,
) -> np.ndarray:
    y = y.astype(float)
    if weights is None:
        weights = np.ones_like(y, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float) * ridge
    penalty[0] = 0.0
    for _ in range(max_iter):
        eta = x @ beta
        p = sigmoid(eta)
        var = np.clip(p * (1 - p), 1e-6, None)
        w = weights * var
        z = eta + (y - p) / var
        h = x.T @ (x * w[:, None]) + np.diag(penalty)
        rhs = x.T @ (w * z)
        try:
            beta_new = np.linalg.solve(h, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(h) @ rhs
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    return beta


def class_weights(y: np.ndarray, positive_boost: float = 1.0) -> np.ndarray:
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    w = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg)).astype(float)
    return np.where(y == 1, w * positive_boost, w)


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def clean_fit_apply(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    med = np.nanmedian(x_train, axis=0)
    med[~np.isfinite(med)] = 0.0
    tr = np.where(np.isfinite(x_train), x_train, med)
    te = np.where(np.isfinite(x_test), x_test, med)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.clip((tr - mu) / sd, -8, 8), np.clip((te - mu) / sd, -8, 8), mu, sd


def clean_apply(x: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(x), x, med)
    return np.clip((z - mu) / sd, -8, 8)


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return auc_rank(y, score)


def threshold_youden(y: np.ndarray, score: np.ndarray) -> float:
    vals = np.unique(score[np.isfinite(score)])
    if len(vals) == 0:
        return 0.5
    cuts = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
    best = None
    for cut in cuts:
        m = metric_at_threshold(y, score, float(cut))
        j = m["sensitivity"] + m["specificity"] - 1
        key = (j, m["sensitivity"], m["specificity"])
        if best is None or key > best[0]:
            best = (key, cut)
    return float(best[1])


def threshold_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    vals = np.unique(score[np.isfinite(score)])
    if len(vals) == 0:
        return 0.5
    cuts = np.r_[vals.min() - 1e-12, (vals[:-1] + vals[1:]) / 2.0, vals.max() + 1e-12]
    best = None
    for cut in cuts:
        m = metric_at_threshold(y, score, float(cut))
        if m["sensitivity"] + 1e-12 < target:
            continue
        key = (m["specificity"], m["sensitivity"], -m["pred_positive_rate"])
        if best is None or key > best[0]:
            best = (key, cut)
    return float(best[1]) if best else float(vals.min() - 1e-12)


def metric_dict(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    m = metric_at_threshold(y, score, threshold)
    m["auc"] = auc_or_nan(y, score)
    return m


def smi_derstine(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / np.where(height_m > 0, height_m**2, np.nan)
    sex = meta["PatientSex"].astype(str).to_numpy()
    y = (((sex == "M") & (smi < DERSTINE_CUTOFF["M"])) | ((sex == "F") & (smi < DERSTINE_CUTOFF["F"]))).astype(int)
    return smi, y


def aec_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        s = str(c)
        if s.startswith("aec_"):
            try:
                idx = int(s.split("_")[1])
            except Exception:
                continue
            cols.append((idx, c))
    return [c for _, c in sorted(cols)]


def matrix_from_sheet(sheet: pd.DataFrame) -> np.ndarray:
    cols = aec_columns(sheet)
    x = sheet[cols].to_numpy(dtype=float)
    global_med = float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    col_med = np.nanmedian(x, axis=0)
    col_med[~np.isfinite(col_med)] = global_med
    inds = np.where(~np.isfinite(x))
    if len(inds[0]):
        x[inds] = np.take(col_med, inds[1])
    x[~np.isfinite(x)] = global_med
    return x


def resample_rows(x: np.ndarray, n: int = 128) -> np.ndarray:
    old = np.linspace(0, 1, x.shape[1])
    new = np.linspace(0, 1, n)
    out = np.zeros((x.shape[0], n), dtype=float)
    for i in range(x.shape[0]):
        out[i] = np.interp(new, old, x[i])
    return out


def row_norm(x: np.ndarray) -> np.ndarray:
    m = np.mean(x, axis=1)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m[:, None]


def slope(block: np.ndarray) -> np.ndarray:
    w = block.shape[1]
    idx = np.arange(w, dtype=float)
    idx = (idx - idx.mean()) / (idx.std() if idx.std() else 1.0)
    return np.mean((block - block.mean(axis=1)[:, None]) * idx[None, :], axis=1)


def curve_feature_frame(x: np.ndarray, prefix: str) -> pd.DataFrame:
    n, p = x.shape
    norm = row_norm(x)
    cent = norm - 1.0
    d1 = np.diff(norm, axis=1)
    d2 = np.diff(norm, n=2, axis=1)
    pos = np.linspace(0, 1, p)
    feat: dict[str, np.ndarray] = {}
    for tag, mat in [("raw", x), ("norm", norm), ("cent", cent)]:
        feat[f"{prefix}_{tag}_mean"] = mat.mean(axis=1)
        feat[f"{prefix}_{tag}_sd"] = mat.std(axis=1)
        feat[f"{prefix}_{tag}_p05"] = np.percentile(mat, 5, axis=1)
        feat[f"{prefix}_{tag}_p25"] = np.percentile(mat, 25, axis=1)
        feat[f"{prefix}_{tag}_p50"] = np.percentile(mat, 50, axis=1)
        feat[f"{prefix}_{tag}_p75"] = np.percentile(mat, 75, axis=1)
        feat[f"{prefix}_{tag}_p95"] = np.percentile(mat, 95, axis=1)
        feat[f"{prefix}_{tag}_iqr"] = feat[f"{prefix}_{tag}_p75"] - feat[f"{prefix}_{tag}_p25"]
        feat[f"{prefix}_{tag}_range"] = mat.max(axis=1) - mat.min(axis=1)
        feat[f"{prefix}_{tag}_min_pos"] = np.argmin(mat, axis=1) / max(1, p - 1)
        feat[f"{prefix}_{tag}_max_pos"] = np.argmax(mat, axis=1) / max(1, p - 1)
        weights = np.clip(mat - mat.min(axis=1)[:, None] + 1e-6, 1e-6, None)
        feat[f"{prefix}_{tag}_centroid"] = (weights * pos[None, :]).sum(axis=1) / weights.sum(axis=1)
    feat[f"{prefix}_d1_abs_mean"] = np.abs(d1).mean(axis=1)
    feat[f"{prefix}_d1_sd"] = d1.std(axis=1)
    feat[f"{prefix}_d1_max_rise"] = d1.max(axis=1)
    feat[f"{prefix}_d1_max_drop"] = d1.min(axis=1)
    feat[f"{prefix}_d2_abs_mean"] = np.abs(d2).mean(axis=1)
    feat[f"{prefix}_d2_sd"] = d2.std(axis=1)
    feat[f"{prefix}_total_variation"] = np.abs(d1).sum(axis=1)
    # 1D texture/radiomic-like summaries on quantized normalized curve.
    q = np.clip(np.floor((norm - np.percentile(norm, 5, axis=1)[:, None]) / (np.percentile(norm, 95, axis=1)[:, None] - np.percentile(norm, 5, axis=1)[:, None] + 1e-6) * 8), 0, 7).astype(int)
    feat[f"{prefix}_texture_transition_rate"] = (np.diff(q, axis=1) != 0).mean(axis=1)
    feat[f"{prefix}_texture_high_run_frac"] = (q >= 6).mean(axis=1)
    feat[f"{prefix}_texture_low_run_frac"] = (q <= 1).mean(axis=1)
    probs = np.stack([(q == k).mean(axis=1) for k in range(8)], axis=1)
    feat[f"{prefix}_texture_entropy"] = -np.sum(np.where(probs > 0, probs * np.log(probs), 0), axis=1) / math.log(8)
    feat[f"{prefix}_texture_energy"] = np.sum(probs**2, axis=1)

    for nseg in [4, 8, 16, 32]:
        edges = np.linspace(0, p, nseg + 1).astype(int)
        for i in range(nseg):
            a, b = edges[i], edges[i + 1]
            for tag, mat in [("raw", x), ("norm", norm)]:
                block = mat[:, a:b]
                name = f"{prefix}_{tag}_seg{nseg:02d}_{i+1:02d}"
                feat[f"{name}_mean"] = block.mean(axis=1)
                feat[f"{name}_sd"] = block.std(axis=1)
                if b - a >= 4:
                    feat[f"{name}_slope"] = slope(block)
    for width in [4, 8, 12, 16, 24, 32, 48]:
        step = max(2, width // 2)
        for a in range(0, p - width + 1, step):
            b = a + width
            for tag, mat in [("raw", x), ("norm", norm)]:
                block = mat[:, a:b]
                name = f"{prefix}_{tag}_w{width:03d}_p{a+1:03d}_{b:03d}"
                feat[f"{name}_mean"] = block.mean(axis=1)
                feat[f"{name}_sd"] = block.std(axis=1)
                if width >= 8:
                    feat[f"{name}_slope"] = slope(block)
    idx = np.arange(p)
    for k in range(1, 17):
        basis = np.cos(np.pi * k * (idx + 0.5) / p)
        feat[f"{prefix}_dct{k:02d}"] = (cent @ basis) / p
    fft = np.fft.rfft(cent, axis=1)
    for k in range(1, min(9, fft.shape[1])):
        feat[f"{prefix}_fft{k:02d}_amp"] = np.abs(fft[:, k]) / p
        feat[f"{prefix}_fft{k:02d}_phase"] = np.angle(fft[:, k])
    return pd.DataFrame(feat).replace([np.inf, -np.inf], np.nan)


def conv1d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        out[:, i] = np.sum(xp[:, i : i + len(kernel)] * kernel[None, :], axis=1)
    return out


def pool_features(mat: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    feat = {
        f"{prefix}_mean": mat.mean(axis=1),
        f"{prefix}_sd": mat.std(axis=1),
        f"{prefix}_abs_mean": np.abs(mat).mean(axis=1),
        f"{prefix}_max": mat.max(axis=1),
        f"{prefix}_min": mat.min(axis=1),
        f"{prefix}_p95": np.percentile(mat, 95, axis=1),
    }
    edges = np.linspace(0, mat.shape[1], 8 + 1).astype(int)
    for i in range(8):
        block = mat[:, edges[i] : edges[i + 1]]
        feat[f"{prefix}_seg8_{i+1:02d}_mean"] = block.mean(axis=1)
        feat[f"{prefix}_seg8_{i+1:02d}_sd"] = block.std(axis=1)
    return feat


def cnn_feature_frame(x: np.ndarray, prefix: str) -> pd.DataFrame:
    norm = row_norm(x)
    kernels = {
        "edge2": np.array([-1, 1], dtype=float),
        "edge3": np.array([-1, 0, 1], dtype=float),
        "lap3": np.array([1, -2, 1], dtype=float),
        "smooth3": np.ones(3) / 3,
        "smooth5": np.ones(5) / 5,
        "smooth9": np.ones(9) / 9,
        "dog5": np.array([-1, -1, 0, 1, 1], dtype=float),
        "wave7": np.array([-1, -1, 0, 2, 0, -1, -1], dtype=float),
    }
    feat: dict[str, np.ndarray] = {}
    for name, k in kernels.items():
        k = k / (np.linalg.norm(k) if np.linalg.norm(k) else 1)
        out = conv1d_same(norm, k)
        feat.update(pool_features(out, f"{prefix}_cnn_{name}"))
    return pd.DataFrame(feat).replace([np.inf, -np.inf], np.nan)


def resnet_feature_frame(x: np.ndarray, prefix: str) -> pd.DataFrame:
    norm = row_norm(x)
    feat: dict[str, np.ndarray] = {}
    current = norm.copy()
    for block_id, width in enumerate([3, 5, 9, 17], start=1):
        smooth = conv1d_same(current, np.ones(width) / width)
        resid = current - smooth
        feat.update(pool_features(resid, f"{prefix}_res_block{block_id}_w{width}"))
        current = current + 0.5 * resid
    feat.update(pool_features(current - norm, f"{prefix}_res_total_delta"))
    return pd.DataFrame(feat).replace([np.inf, -np.inf], np.nan)


def transformer_feature_frame(x: np.ndarray, prefix: str) -> pd.DataFrame:
    norm = row_norm(x)
    n, p = norm.shape
    n_tokens = 16
    edges = np.linspace(0, p, n_tokens + 1).astype(int)
    token_list = []
    feat: dict[str, np.ndarray] = {}
    for i in range(n_tokens):
        block = norm[:, edges[i] : edges[i + 1]]
        t = np.column_stack([block.mean(axis=1), block.std(axis=1), slope(block), np.abs(np.diff(block, axis=1)).mean(axis=1)])
        token_list.append(t)
        for j, nm in enumerate(["mean", "sd", "slope", "rough"]):
            feat[f"{prefix}_tok{i+1:02d}_{nm}"] = t[:, j]
    tokens = np.stack(token_list, axis=1)  # n, token, dim
    # Self-attention-like summaries from token similarity.
    centered = tokens - tokens.mean(axis=1, keepdims=True)
    sim = np.einsum("ntd,nsd->nts", centered, centered) / math.sqrt(tokens.shape[2])
    sim = sim - sim.max(axis=2, keepdims=True)
    att = np.exp(sim)
    att = att / att.sum(axis=2, keepdims=True)
    pos = np.linspace(0, 1, n_tokens)
    attended_pos = np.einsum("nts,s->nt", att, pos)
    entropy = -np.sum(att * np.log(np.clip(att, 1e-9, None)), axis=2) / math.log(n_tokens)
    feat[f"{prefix}_att_pos_mean"] = attended_pos.mean(axis=1)
    feat[f"{prefix}_att_pos_sd"] = attended_pos.std(axis=1)
    feat[f"{prefix}_att_entropy_mean"] = entropy.mean(axis=1)
    feat[f"{prefix}_att_entropy_min"] = entropy.min(axis=1)
    feat[f"{prefix}_att_diag_mass"] = np.mean(np.diagonal(att, axis1=1, axis2=2), axis=1)
    feat[f"{prefix}_att_upper_lower_mass"] = att[:, :8, 8:].mean(axis=(1, 2)) - att[:, 8:, :8].mean(axis=(1, 2))
    return pd.DataFrame(feat).replace([np.inf, -np.inf], np.nan)


def clinical_matrix(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = ["PatientAge", "Height", "Weight"]
    tr_parts, te_parts, names = [], [], []
    for c in cols:
        tr = pd.to_numeric(train_meta[c], errors="coerce").to_numpy(dtype=float)
        te = pd.to_numeric(test_meta[c], errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(tr)
        tr = np.where(np.isfinite(tr), tr, med)
        te = np.where(np.isfinite(te), te, med)
        tr_parts.append(tr[:, None])
        te_parts.append(te[:, None])
        names.append(c)
    tr_sex = (train_meta["PatientSex"].astype(str).to_numpy() == "M").astype(float)
    te_sex = (test_meta["PatientSex"].astype(str).to_numpy() == "M").astype(float)
    tr_parts.append(tr_sex[:, None])
    te_parts.append(te_sex[:, None])
    names.append("sex_M")
    return np.hstack(tr_parts), np.hstack(te_parts), names


def cross_attention_features(aec_feat: pd.DataFrame, clinical_raw: np.ndarray, prefix: str) -> pd.DataFrame:
    # Compact clinical-AEC interaction block; later standardized inside the model.
    token_cols = [c for c in aec_feat.columns if ("_tok" in c and (c.endswith("_mean") or c.endswith("_sd") or c.endswith("_rough")))]
    token_cols = token_cols[:64]
    x = aec_feat[token_cols].to_numpy(dtype=float)
    age, height, weight, sex = clinical_raw[:, 0], clinical_raw[:, 1], clinical_raw[:, 2], clinical_raw[:, 3]
    clinical = {
        "age": age,
        "height": height,
        "weight": weight,
        "sexM": sex,
        "age_weight": age * weight,
        "height_weight": height * weight,
    }
    out: dict[str, np.ndarray] = {}
    for i, c in enumerate(token_cols):
        base = x[:, i]
        for nm, cv in clinical.items():
            out[f"{prefix}_cross_{c}_{nm}"] = base * cv
    # Gated token pooling proxies.
    risk_proxy = (age - np.nanmean(age)) / (np.nanstd(age) or 1) - (weight - np.nanmean(weight)) / (np.nanstd(weight) or 1)
    for i, c in enumerate(token_cols):
        out[f"{prefix}_gated_{c}"] = x[:, i] * risk_proxy
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def load_dataset(path: Path, label: str) -> dict:
    meta = pd.read_excel(path, sheet_name="metadata")
    a128 = pd.read_excel(path, sheet_name="aec_128")
    crop = pd.read_excel(path, sheet_name="aec_cropped")
    smi_calc, y = smi_derstine(meta)
    a128_mat = resample_rows(matrix_from_sheet(a128), 128)
    crop_mat = resample_rows(matrix_from_sheet(crop), 128)
    out = {
        "label": label,
        "meta": meta.copy(),
        "smi_calc": smi_calc,
        "y": y,
        "a128": a128_mat,
        "crop": crop_mat,
    }
    return out


def assemble_feature_families(train: dict, test: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    families: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    tr128_rad = curve_feature_frame(train["a128"], "a128")
    te128_rad = curve_feature_frame(test["a128"], "a128")
    trcrop_rad = curve_feature_frame(train["crop"], "crop")
    tecrop_rad = curve_feature_frame(test["crop"], "crop")
    tr128_cnn = cnn_feature_frame(train["a128"], "a128")
    te128_cnn = cnn_feature_frame(test["a128"], "a128")
    trcrop_cnn = cnn_feature_frame(train["crop"], "crop")
    tecrop_cnn = cnn_feature_frame(test["crop"], "crop")
    tr128_res = resnet_feature_frame(train["a128"], "a128")
    te128_res = resnet_feature_frame(test["a128"], "a128")
    tr128_trans = transformer_feature_frame(train["a128"], "a128")
    te128_trans = transformer_feature_frame(test["a128"], "a128")
    tr_clin, te_clin, _ = clinical_matrix(train["meta"], test["meta"])
    tr_cross = cross_attention_features(tr128_trans, tr_clin, "a128")
    te_cross = cross_attention_features(te128_trans, te_clin, "a128")
    families["radiomic_128"] = (tr128_rad, te128_rad)
    families["radiomic_crop"] = (trcrop_rad, tecrop_rad)
    families["radiomic_both"] = (pd.concat([tr128_rad, trcrop_rad], axis=1), pd.concat([te128_rad, tecrop_rad], axis=1))
    families["cnn_128"] = (tr128_cnn, te128_cnn)
    families["cnn_both"] = (pd.concat([tr128_cnn, trcrop_cnn], axis=1), pd.concat([te128_cnn, tecrop_cnn], axis=1))
    families["resnet_128"] = (tr128_res, te128_res)
    families["transformer_128"] = (tr128_trans, te128_trans)
    families["crossattention_128"] = (pd.concat([tr128_trans, tr_cross], axis=1), pd.concat([te128_trans, te_cross], axis=1))
    families["ml_all_aec"] = (
        pd.concat([tr128_rad, trcrop_rad, tr128_cnn, trcrop_cnn, tr128_res, tr128_trans], axis=1),
        pd.concat([te128_rad, tecrop_rad, te128_cnn, tecrop_cnn, te128_res, te128_trans], axis=1),
    )
    return families


def topk_indices(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if k >= x.shape[1]:
        return np.arange(x.shape[1])
    y = y.astype(int)
    pos = x[y == 1]
    neg = x[y == 0]
    diff = np.nanmean(pos, axis=0) - np.nanmean(neg, axis=0)
    sd = np.nanstd(x, axis=0)
    score = np.abs(diff) / np.where(sd == 0, 1, sd)
    score[~np.isfinite(score)] = 0.0
    return np.argsort(score)[::-1][:k]


def fit_predict_params(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_apply_raw: np.ndarray,
    params: dict,
) -> np.ndarray:
    # Top-k selection happens after rough median imputation on the current train partition.
    tr0, ap0, _, _ = clean_fit_apply(x_train_raw, x_apply_raw)
    idx = topk_indices(tr0, y_train, int(params["topk"]))
    tr = tr0[:, idx]
    ap = ap0[:, idx]
    w = class_weights(y_train, positive_boost=float(params["pos_boost"])) if params["weighted"] else None
    beta = logit_fit_weighted(add_intercept(tr), y_train, ridge=float(params["ridge"]), weights=w)
    return sigmoid(add_intercept(ap) @ beta)


def cv_oof_score(x: np.ndarray, y: np.ndarray, params: dict, folds: list[np.ndarray]) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    all_idx = np.arange(len(y))
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx)
        pred[test_idx] = fit_predict_params(x[train_idx], y[train_idx], x[test_idx], params)
    return pred


def score_at_sens(y: np.ndarray, score: np.ndarray, target: float) -> float:
    thr = threshold_min_sensitivity(y, score, target)
    return metric_at_threshold(y, score, thr)["specificity"]


def train_model_family(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    grids: list[dict],
    folds: list[np.ndarray],
) -> dict:
    best = None
    rows = []
    for params in grids:
        oof = cv_oof_score(x_train, y_train, params, folds)
        auc = auc_rank(y_train, oof)
        spec90 = score_at_sens(y_train, oof, 0.90)
        composite = auc + 0.10 * spec90
        row = {"model": name, **params, "cv_auc": auc, "cv_spec_at_sens90": spec90, "cv_composite": composite}
        rows.append(row)
        key = (composite, auc, spec90)
        if best is None or key > best[0]:
            best = (key, params, oof)
    params = best[1]
    test_score = fit_predict_params(x_train, y_train, x_test, params)
    return {
        "name": name,
        "params": params,
        "oof": best[2],
        "test": test_score,
        "cv_grid": pd.DataFrame(rows).sort_values("cv_composite", ascending=False),
    }


def rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[order] = np.arange(len(score), dtype=float)
    return r / max(1, len(score) - 1)


def train_clinical(train: dict, test: dict, folds: list[np.ndarray]) -> dict:
    xtr, xte, names = clinical_matrix(train["meta"], test["meta"])
    grids = [
        {"topk": xtr.shape[1], "ridge": r, "weighted": w, "pos_boost": pb}
        for r in [0.001, 0.01, 0.1, 1.0]
        for w in [False, True]
        for pb in ([1.0, 1.5] if w else [1.0])
    ]
    return train_model_family("clinical", xtr, train["y"], xte, grids, folds)


def make_model_matrices(families: dict[str, tuple[pd.DataFrame, pd.DataFrame]], train: dict, test: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    for name, (trdf, tedf) in families.items():
        tr = trdf.to_numpy(dtype=float)
        te = tedf.to_numpy(dtype=float)
        mats[f"aec_only_{name}"] = (tr, te)
        mats[f"clinical_plus_{name}"] = (np.hstack([xclin_tr, tr]), np.hstack([xclin_te, te]))
    return mats


def evaluate_model(name: str, y_train: np.ndarray, oof: np.ndarray, y_test: np.ndarray, test_score: np.ndarray) -> dict:
    thr_youden = threshold_youden(y_train, oof)
    thr_sens90 = threshold_min_sensitivity(y_train, oof, 0.90)
    out = {
        "model": name,
        "cv_auc": auc_rank(y_train, oof),
        "cv_spec_at_sens90": metric_at_threshold(y_train, oof, thr_sens90)["specificity"],
        "test_auc": auc_or_nan(y_test, test_score),
    }
    for label, thr in [("youden", thr_youden), ("sens90", thr_sens90)]:
        m = metric_at_threshold(y_test, test_score, thr)
        for k, v in m.items():
            out[f"test_{label}_{k}"] = v
        out[f"{label}_threshold_train"] = thr
    return out


def subgroup_metrics(model_scores: dict[str, np.ndarray], thresholds: dict[str, float], test: dict, selected_models: list[str]) -> pd.DataFrame:
    meta = test["meta"]
    y = test["y"]
    groups: list[tuple[str, np.ndarray]] = [("Overall", np.ones(len(y), dtype=bool))]
    for sex in ["M", "F"]:
        groups.append((f"Sex={sex}", meta["PatientSex"].astype(str).to_numpy() == sex))
    for scanner, count in meta["Manufacturer"].value_counts().items():
        groups.append((f"Scanner={scanner}", meta["Manufacturer"].astype(str).to_numpy() == str(scanner)))
    rows = []
    for model in selected_models:
        score = model_scores[model]
        thr = thresholds[model]
        pred = score >= thr
        for gname, mask in groups:
            if int(mask.sum()) == 0:
                continue
            yy = y[mask]
            pp = pred[mask]
            ss = score[mask]
            tp = int(np.sum((pp == 1) & (yy == 1)))
            fn = int(np.sum((pp == 0) & (yy == 1)))
            tn = int(np.sum((pp == 0) & (yy == 0)))
            fp = int(np.sum((pp == 1) & (yy == 0)))
            rows.append(
                {
                    "model": model,
                    "subgroup": gname,
                    "n": int(mask.sum()),
                    "events": int(yy.sum()),
                    "prevalence": float(yy.mean()) if len(yy) else np.nan,
                    "auc": auc_or_nan(yy, ss),
                    "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                    "specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "ppv": tp / (tp + fp) if tp + fp else np.nan,
                    "npv": tn / (tn + fn) if tn + fn else np.nan,
                    "accuracy": (tp + tn) / len(yy) if len(yy) else np.nan,
                    "tp": tp,
                    "fn": fn,
                    "tn": tn,
                    "fp": fp,
                }
            )
    return pd.DataFrame(rows)


def clinical_weakness_rows(clinical_score: np.ndarray, aec_score: np.ndarray, y_train: np.ndarray, train_clin: np.ndarray, test: dict, model_name: str) -> dict:
    # Thresholds set from train: clinical high-sensitivity negatives rescued by AEC ensemble.
    clin_thr = threshold_min_sensitivity(y_train, train_clin, 0.90)
    aec_thr = threshold_min_sensitivity(y_train[train_clin < clin_thr], aec_score["train"][train_clin < clin_thr], 0.35) if np.sum(train_clin < clin_thr) > 20 else np.percentile(aec_score["train"], 75)
    y = test["y"]
    base_pred = clinical_score >= clin_thr
    rescue = (~base_pred) & (aec_score["test"] >= aec_thr)
    final_pred = base_pred | rescue
    base_m = metric_at_threshold(y, clinical_score, clin_thr)
    final_m = {
        "tp": int(np.sum((final_pred == 1) & (y == 1))),
        "fn": int(np.sum((final_pred == 0) & (y == 1))),
        "tn": int(np.sum((final_pred == 0) & (y == 0))),
        "fp": int(np.sum((final_pred == 1) & (y == 0))),
    }
    final_m["sensitivity"] = final_m["tp"] / (final_m["tp"] + final_m["fn"])
    final_m["specificity"] = final_m["tn"] / (final_m["tn"] + final_m["fp"])
    final_m["ppv"] = final_m["tp"] / (final_m["tp"] + final_m["fp"]) if final_m["tp"] + final_m["fp"] else np.nan
    final_m["npv"] = final_m["tn"] / (final_m["tn"] + final_m["fn"]) if final_m["tn"] + final_m["fn"] else np.nan
    comp = compare_predictions(y, base_pred, final_pred)
    return {
        "strategy": model_name,
        "clinical_threshold": clin_thr,
        "aec_rescue_threshold": aec_thr,
        "base_sensitivity": base_m["sensitivity"],
        "base_specificity": base_m["specificity"],
        **{f"final_{k}": v for k, v in final_m.items()},
        "rescued_n": int(rescue.sum()),
        "rescued_tp": int(np.sum(rescue & (y == 1))),
        "rescued_fp": int(np.sum(rescue & (y == 0))),
        "delta_sensitivity": final_m["sensitivity"] - base_m["sensitivity"],
        "delta_specificity": final_m["specificity"] - base_m["specificity"],
        **comp,
    }


def main() -> None:
    train_path = DATA_DIR / "g1090.xlsx"
    test_path = DATA_DIR / "sdata.xlsx"
    train = load_dataset(train_path, "g1090")
    test = load_dataset(test_path, "sdata")
    folds = make_stratified_folds(train["y"], k=5, seed=SEED)

    print(f"Train g1090: n={len(train['y'])}, events={int(train['y'].sum())} ({train['y'].mean():.1%})")
    print(f"Test sdata: n={len(test['y'])}, events={int(test['y'].sum())} ({test['y'].mean():.1%})")

    families = assemble_feature_families(train, test)
    matrices = make_model_matrices(families, train, test)

    model_results = []
    model_scores_train: dict[str, np.ndarray] = {}
    model_scores_test: dict[str, np.ndarray] = {}
    grid_rows = []

    clinical = train_clinical(train, test, folds)
    model_scores_train["clinical"] = clinical["oof"]
    model_scores_test["clinical"] = clinical["test"]
    model_results.append(evaluate_model("clinical", train["y"], clinical["oof"], test["y"], clinical["test"]))
    clinical["cv_grid"].to_csv(OUT_DIR / "grid_clinical.csv", index=False, encoding="utf-8-sig")

    # Family-specific grids keep dimensions moderate and selection fully inside g1090 CV.
    for name, (xtr, xte) in matrices.items():
        max_p = xtr.shape[1]
        topks = sorted(set([min(max_p, k) for k in [8, 16, 32, 64, 128, 256]]))
        grids = [
            {"topk": k, "ridge": r, "weighted": w, "pos_boost": pb}
            for k in topks
            for r in [0.01, 0.1, 1.0, 10.0]
            for w in [False, True]
            for pb in ([1.0, 1.5, 2.0] if w else [1.0])
        ]
        print(f"Training {name}: n_features={max_p}, grid={len(grids)}")
        res = train_model_family(name, xtr, train["y"], xte, grids, folds)
        model_scores_train[name] = res["oof"]
        model_scores_test[name] = res["test"]
        model_results.append(evaluate_model(name, train["y"], res["oof"], test["y"], res["test"]))
        best_grid = res["cv_grid"].head(10).copy()
        best_grid["selected_model"] = name
        grid_rows.append(best_grid)

    model_df = pd.DataFrame(model_results).sort_values(["cv_auc", "cv_spec_at_sens90"], ascending=False)

    # CV-selected ensembles, not selected on sdata.
    aec_only_candidates = [m for m in model_scores_train if m.startswith("aec_only_")]
    clin_plus_candidates = [m for m in model_scores_train if m.startswith("clinical_plus_")]
    cv_rank = model_df.set_index("model")
    top_aec = sorted(aec_only_candidates, key=lambda m: (cv_rank.loc[m, "cv_auc"] + 0.10 * cv_rank.loc[m, "cv_spec_at_sens90"]), reverse=True)[:5]
    top_cp = sorted(clin_plus_candidates, key=lambda m: (cv_rank.loc[m, "cv_auc"] + 0.10 * cv_rank.loc[m, "cv_spec_at_sens90"]), reverse=True)[:5]
    ensembles = {
        "aec_only_cv_top5_ensemble": top_aec,
        "clinical_plus_aec_cv_top5_ensemble": top_cp,
        "clinical_plus_aec_and_clinical_ensemble": ["clinical", *top_cp],
    }
    for ename, members in ensembles.items():
        tr = np.mean([rank01(model_scores_train[m]) for m in members], axis=0)
        te = np.mean([rank01(model_scores_test[m]) for m in members], axis=0)
        model_scores_train[ename] = tr
        model_scores_test[ename] = te
        row = evaluate_model(ename, train["y"], tr, test["y"], te)
        row["members"] = ";".join(members)
        model_results.append(row)
    model_df = pd.DataFrame(model_results).sort_values(["test_auc", "cv_auc"], ascending=False)
    model_df.to_csv(OUT_DIR / "model_summary_train_cv_sdata_test.csv", index=False, encoding="utf-8-sig")
    if grid_rows:
        pd.concat(grid_rows, ignore_index=True).to_csv(OUT_DIR / "model_grid_top10_by_family.csv", index=False, encoding="utf-8-sig")

    # Thresholds from train OOF Youden for subgroup table.
    thresholds = {m: threshold_youden(train["y"], s) for m, s in model_scores_train.items()}
    selected = [
        "clinical",
        "aec_only_cv_top5_ensemble",
        "clinical_plus_aec_cv_top5_ensemble",
        "clinical_plus_aec_and_clinical_ensemble",
    ]
    # Add best individual AEC and best individual clinical+AEC by CV composite.
    selected.append(top_aec[0])
    selected.append(top_cp[0])
    selected = list(dict.fromkeys(selected))
    sub = subgroup_metrics(model_scores_test, thresholds, test, selected)
    sub.to_csv(OUT_DIR / "sdata_subgroup_metrics_selected_models.csv", index=False, encoding="utf-8-sig")

    # Clinical weakness: rescue clinical high-sensitivity false negatives by AEC-only ensemble.
    weakness = []
    weakness.append(
        clinical_weakness_rows(
            model_scores_test["clinical"],
            {"train": model_scores_train["aec_only_cv_top5_ensemble"], "test": model_scores_test["aec_only_cv_top5_ensemble"]},
            train["y"],
            model_scores_train["clinical"],
            test,
            "clinical_sens90_plus_aec_only_ensemble_rescue",
        )
    )
    weakness.append(
        clinical_weakness_rows(
            model_scores_test["clinical"],
            {"train": model_scores_train[top_aec[0]], "test": model_scores_test[top_aec[0]]},
            train["y"],
            model_scores_train["clinical"],
            test,
            f"clinical_sens90_plus_{top_aec[0]}_rescue",
        )
    )
    weak_df = pd.DataFrame(weakness)
    weak_df.to_csv(OUT_DIR / "clinical_weakness_aec_rescue_sdata.csv", index=False, encoding="utf-8-sig")

    # Dataset summaries.
    dataset_rows = []
    for d in [train, test]:
        meta = d["meta"]
        for name, mask in [
            ("Overall", np.ones(len(d["y"]), dtype=bool)),
            ("Sex=M", meta["PatientSex"].astype(str).to_numpy() == "M"),
            ("Sex=F", meta["PatientSex"].astype(str).to_numpy() == "F"),
        ]:
            dataset_rows.append(
                {
                    "dataset": d["label"],
                    "subgroup": name,
                    "n": int(mask.sum()),
                    "events": int(d["y"][mask].sum()),
                    "prevalence": float(d["y"][mask].mean()),
                    "age_mean": float(pd.to_numeric(meta.loc[mask, "PatientAge"], errors="coerce").mean()),
                    "height_mean": float(pd.to_numeric(meta.loc[mask, "Height"], errors="coerce").mean()),
                    "weight_mean": float(pd.to_numeric(meta.loc[mask, "Weight"], errors="coerce").mean()),
                    "smi_calc_mean": float(np.nanmean(d["smi_calc"][mask])),
                }
            )
        for scanner, count in meta["Manufacturer"].value_counts().items():
            mask = meta["Manufacturer"].astype(str).to_numpy() == str(scanner)
            dataset_rows.append(
                {
                    "dataset": d["label"],
                    "subgroup": f"Scanner={scanner}",
                    "n": int(mask.sum()),
                    "events": int(d["y"][mask].sum()),
                    "prevalence": float(d["y"][mask].mean()),
                    "age_mean": float(pd.to_numeric(meta.loc[mask, "PatientAge"], errors="coerce").mean()),
                    "height_mean": float(pd.to_numeric(meta.loc[mask, "Height"], errors="coerce").mean()),
                    "weight_mean": float(pd.to_numeric(meta.loc[mask, "Weight"], errors="coerce").mean()),
                    "smi_calc_mean": float(np.nanmean(d["smi_calc"][mask])),
                }
            )
    pd.DataFrame(dataset_rows).to_csv(OUT_DIR / "dataset_subgroup_descriptives.csv", index=False, encoding="utf-8-sig")

    print("\nTop models by external sdata AUC")
    show_cols = [
        "model",
        "cv_auc",
        "cv_spec_at_sens90",
        "test_auc",
        "test_youden_sensitivity",
        "test_youden_specificity",
        "test_sens90_sensitivity",
        "test_sens90_specificity",
        "members",
    ]
    print(model_df[[c for c in show_cols if c in model_df.columns]].head(25).to_string(index=False))
    print("\nSelected subgroup metrics")
    print(sub[sub["subgroup"].isin(["Overall", "Sex=M", "Sex=F"])].to_string(index=False))
    print("\nClinical weakness / AEC rescue")
    print(weak_df.to_string(index=False))
    print("\nSaved outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
