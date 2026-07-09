from __future__ import annotations

# Standalone 4-stage AEC low-SMI pipeline (run: python code/run_from_raw_standalone.py).
# Inputs: data/g1090.xlsx, data/sdata.xlsx. All helper modules are inlined (no aec_*.py
# deps); per-module globals are prefixed (COND_, LOCK_, ...) to avoid name collisions.
# Stages: LOCK_main (lock features) -> PATTERN_main (CNN branch probs) ->
# BOOST_main (AEC scores) -> FINAL_main (phenotype enrichment). All thresholds are
# locked on the internal cohort before touching the external cohort.

import itertools
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Pin thread pools to 1 (+ n_jobs=1 below) so BLAS/OpenMP reductions sum in a fixed
# order, keeping BOOST_main's scores exactly reproducible run-to-run on a fixed seed.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by torch's deterministic CUDA algorithms

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from typing import cast

import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, stats
from scipy.fft import dct
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC

# exact_p() returns NaN for candidates that never de-escalate anyone (n=0), which is
# expected during screening and already rejected downstream by the deesc_n>=25 check.
warnings.filterwarnings("ignore", message="All-NaN (slice|axis) encountered")

# CONFIG: paths derived from this script's location so the project is portable.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "run_from_raw_standalone"
DRY_RUN = False

FILES = {"g1090": DATA_DIR / "g1090.xlsx", "sdata": DATA_DIR / "sdata.xlsx"}
INTERNAL_XLSX = DATA_DIR / "g1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sdata.xlsx"

LOCK_OUT_DIR = OUTPUT_ROOT / "aec_lock_smoothed_deesc_gate"
LOCK_DIR = LOCK_OUT_DIR

PATTERN_OUT_DIR = OUTPUT_ROOT / "aec_region_cnn_pattern_gate"
PROB_CACHE = PATTERN_OUT_DIR / "direct_vote_probabilities.npz"
PROB_PATH = PROB_CACHE

BOOST_OUT_DIR = OUTPUT_ROOT / "aec_direct_vote_auc_boost"
DIRECT_VOTE_SCORE_CSV = BOOST_OUT_DIR / "direct_vote_auc_boost_scores.csv"

FINAL_OUT_DIR = OUTPUT_ROOT / "aec_final_global_quintile_phenotype"
AEC_SCORE_COLUMN = "vote_only_logit_l1"
PRIMARY_Q = 0.20
SENSITIVITY_Q = 0.25

# Fallback scripts used by run_if_needed() to rebuild DIRECT_VOTE_SCORE_CSV if missing.
LOCK_SCRIPT = PROJECT_ROOT / "work" / "aec_lock_smoothed_deesc_gate.py"
PATTERN_GATE_SCRIPT = PROJECT_ROOT / "work" / "aec_region_cnn_pattern_gate.py"
DIRECT_VOTE_AUC_SCRIPT = PROJECT_ROOT / "work" / "aec_direct_vote_auc_boost.py"

COND_SEED = 20260629
RNG = np.random.default_rng(COND_SEED)

def aec_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if str(c).startswith("aec_")], key=lambda c: int(str(c).split("_")[1]))

def matrix_from_sheet(df: pd.DataFrame) -> np.ndarray:
    x = df[aec_columns(df)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    global_med = float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    col_med = np.nanmedian(x, axis=0)
    col_med[~np.isfinite(col_med)] = global_med
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(col_med, np.where(bad)[1])
    x[~np.isfinite(x)] = global_med
    return x

def clinical_matrix(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = ["PatientAge", "Height", "Weight", "sex_M"]

    def raw(meta: pd.DataFrame) -> np.ndarray:
        x = np.column_stack([pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float), pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float), pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float), (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)])
        return x

    tr = raw(train_meta)
    te = raw(test_meta)
    med = np.nanmedian(tr, axis=0)
    tr = np.where(np.isfinite(tr), tr, med)
    te = np.where(np.isfinite(te), te, med)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (tr - mu) / sd, (te - mu) / sd, names

def make_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]

def clinical_estimator() -> LogisticRegression:
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)

def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)

def oof_and_external(model_factory, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(ytr), dtype=float)
    all_idx = np.arange(len(ytr))
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = model_factory(COND_SEED + fold_id)
        model.fit(xtr[tr_idx], ytr[tr_idx])
        oof[val_idx] = score_model(model, xtr[val_idx])
    final = model_factory(COND_SEED + 99)
    final.fit(xtr, ytr)
    return oof, score_model(final, xte)

def zfit_apply(train_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu = float(np.mean(train_score))
    sd = float(np.std(train_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (train_score - mu) / sd, (test_score - mu) / sd, mu, sd

def confusion_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(pred, dtype=bool)
    pos = np.asarray(y).astype(bool)
    return int(np.sum(pred & pos)), int(np.sum(pred & ~pos)), int(np.sum(~pred & pos)), int(np.sum(~pred & ~pos))

def COND_binary_metrics(y: np.ndarray, score: np.ndarray, th: float) -> dict:
    tp, fp, fn, tn = confusion_counts(y, score >= th)
    return {"threshold": float(th), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan, "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan}

def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    best = None
    for th in np.unique(score):
        m = COND_binary_metrics(y, score, float(th))
        if m["sensitivity"] >= target and (best is None or m["specificity"] > best[1]):
            best = (float(th), m["specificity"])
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return best[0]

def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return num / np.where(np.abs(den) < eps, eps * np.sign(den + eps), den)

def pseg(x: np.ndarray, a: int, b: int) -> np.ndarray:
    return x[:, a - 1 : b].mean(axis=1)

def MASS_add_window_stats(out: dict[str, np.ndarray], signal: np.ndarray, prefix: str, lengths: list[int], step: int, stats_list: tuple[str, ...] = ("mean",)) -> None:
    n = signal.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            block = signal[:, start0:end0]
            tag = f"{prefix}_{start0 + 1:03d}_{end0:03d}"
            if "mean" in stats_list:
                out[f"{tag}_mean"] = block.mean(axis=1)
            if "sd" in stats_list:
                out[f"{tag}_sd"] = block.std(axis=1)
            if "min" in stats_list:
                out[f"{tag}_min"] = block.min(axis=1)
            if "max" in stats_list:
                out[f"{tag}_max"] = block.max(axis=1)

def add_haar_edges(out: dict[str, np.ndarray], signal: np.ndarray, prefix: str, blocks: list[int], step: int) -> None:
    n = signal.shape[1]
    for block in blocks:
        length = 2 * block
        for start0 in range(0, n - length + 1, step):
            mid0 = start0 + block
            end0 = start0 + length
            out[f"{prefix}_haar_b{block:02d}_{start0 + 1:03d}_{end0:03d}"] = signal[:, mid0:end0].mean(axis=1) - signal[:, start0:mid0].mean(axis=1)

def longest_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        best = cur = 0
        for val in row:
            if val:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[i] = best
    return out

def run_count(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        runs = 0
        in_run = False
        for val in row:
            if val and not in_run:
                runs += 1
                in_run = True
            elif not val:
                in_run = False
        out[i] = runs
    return out

def autocorr_features(x: np.ndarray, prefix: str, lags: list[int]) -> dict[str, np.ndarray]:
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered**2, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    out = {}
    for lag in lags:
        out[f"{prefix}_autocorr_lag_{lag:02d}"] = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / denom
    return out

def build_feature_bank(x_norm: np.ndarray) -> pd.DataFrame:
    x = np.asarray(x_norm, dtype=float)
    logx = np.log(np.clip(x, 1e-6, None))
    d1 = np.diff(x, axis=1)
    dlog = np.diff(logx, axis=1)
    d2 = np.diff(d1, axis=1)
    coeff = np.asarray(dct(logx, type=2, norm="ortho", axis=1))
    fft_mag = np.abs(np.fft.rfft(logx - logx.mean(axis=1, keepdims=True), axis=1))

    rows: dict[str, np.ndarray] = {}

    MASS_add_window_stats(rows, x, "norm_level", [4, 8, 12, 16, 24, 32, 48, 64], step=4, stats_list=("mean",))
    MASS_add_window_stats(rows, logx, "log_level", [4, 8, 12, 16, 24, 32, 48, 64], step=4, stats_list=("mean",))
    MASS_add_window_stats(rows, d1, "norm_slope", [4, 8, 12, 16, 24, 32], step=3, stats_list=("mean", "sd"))
    MASS_add_window_stats(rows, dlog, "log_slope", [4, 8, 12, 16, 24, 32], step=3, stats_list=("mean", "sd"))
    MASS_add_window_stats(rows, d2, "norm_curv", [4, 8, 12, 16, 24], step=3, stats_list=("mean", "sd", "min", "max"))
    add_haar_edges(rows, x, "norm", [2, 4, 8, 12, 16, 24], step=2)
    add_haar_edges(rows, logx, "log", [2, 4, 8, 12, 16, 24], step=2)

    segs = {"early_001_032": (1, 32), "earlymid_001_058": (1, 58), "pretrough_050_065": (50, 65), "transition_058_074": (58, 74), "trough_060_095": (60, 95), "troughcore_070_085": (70, 85), "recover_075_090": (75, 90), "recover_081_113": (81, 113), "late_091_113": (91, 113), "tail_114_128": (114, 128), "tail_120_128": (120, 128)}
    level_vals = {name: pseg(x, a, b) for name, (a, b) in segs.items()}
    log_vals = {f"log_{name}": pseg(logx, a, b) for name, (a, b) in segs.items()}
    rows.update({f"level_{k}": v for k, v in level_vals.items()})
    rows.update(log_vals)
    keys = list(level_vals.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            rows[f"contrast_{k2}_minus_{k1}"] = level_vals[k2] - level_vals[k1]
            rows[f"ratio_{k2}_over_{k1}"] = safe_div(level_vals[k2], level_vals[k1])
    log_keys = list(log_vals.keys())
    for i, k1 in enumerate(log_keys):
        for k2 in log_keys[i + 1 :]:
            rows[f"logcontrast_{k2}_minus_{k1}"] = log_vals[k2] - log_vals[k1]

    ranges = {"early_001_040": (1, 40), "mid_040_080": (40, 80), "trough_060_095": (60, 95), "late_091_128": (91, 128), "tail_114_128": (114, 128)}
    extrema = {}
    for name, (a, b) in ranges.items():
        block = x[:, a - 1 : b]
        extrema[f"{name}_min"] = block.min(axis=1)
        extrema[f"{name}_max"] = block.max(axis=1)
        extrema[f"{name}_argmin"] = np.argmin(block, axis=1).astype(float) + a
        extrema[f"{name}_argmax"] = np.argmax(block, axis=1).astype(float) + a
        extrema[f"{name}_range"] = block.max(axis=1) - block.min(axis=1)
    rows.update(extrema)
    rows["late_rebound_height_max91_128_minus_min60_95"] = extrema["late_091_128_max"] - extrema["trough_060_095_min"]
    rows["tail_rebound_height_max114_128_minus_min60_95"] = extrema["tail_114_128_max"] - extrema["trough_060_095_min"]
    rows["early_to_trough_drop_max1_40_minus_min60_95"] = extrema["early_001_040_max"] - extrema["trough_060_095_min"]
    rows["tailpeak_minus_earlypeak"] = extrema["late_091_128_max"] - extrema["early_001_040_max"]
    rows["trough_to_latepeak_distance"] = extrema["late_091_128_argmax"] - extrema["trough_060_095_argmin"]
    rows["earlypeak_to_trough_distance"] = extrema["trough_060_095_argmin"] - extrema["early_001_040_argmax"]
    rows["recovery_fraction_late"] = safe_div(extrema["late_091_128_max"] - extrema["trough_060_095_min"], extrema["early_001_040_max"] - extrema["trough_060_095_min"])

    for name, (a, b) in {"early_001_032": (1, 32), "transition_058_074": (58, 74), "recover_075_090": (75, 90), "recover_081_113": (81, 113), "tail_114_127": (114, 127), "late_081_127": (81, 127)}.items():
        dd = d1[:, a - 1 : b]
        rows[f"rough_abs_d1_{name}"] = np.abs(dd).mean(axis=1)
        rows[f"rough_sd_d1_{name}"] = dd.std(axis=1)
        rows[f"slope_pos_fraction_{name}"] = (dd > 0).mean(axis=1)
        rows[f"slope_sign_change_count_{name}"] = np.sum(np.diff(np.sign(dd), axis=1) != 0, axis=1).astype(float)
        flat = np.abs(dd) < 0.001
        rows[f"nearflat_count_{name}"] = flat.sum(axis=1).astype(float)
        rows[f"nearflat_run_count_{name}"] = run_count(flat)
        rows[f"nearflat_longest_run_{name}"] = longest_run(flat)

    rows["global_norm_sd"] = x.std(axis=1)
    rows["global_log_sd"] = logx.std(axis=1)
    rows["global_norm_range"] = x.max(axis=1) - x.min(axis=1)
    rows["global_norm_skew"] = stats.skew(x, axis=1)
    rows["global_norm_kurtosis"] = stats.kurtosis(x, axis=1)
    rows["global_abs_slope_mean"] = np.abs(d1).mean(axis=1)
    rows["global_abs_curv_mean"] = np.abs(d2).mean(axis=1)

    for i in range(1, 49):
        rows[f"dct_log_{i:02d}"] = coeff[:, i]
    for i in range(1, min(49, fft_mag.shape[1])):
        rows[f"fftmag_log_{i:02d}"] = fft_mag[:, i]
    rows.update(autocorr_features(logx, "log", [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64]))

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

LOCK_SEED = 20260701
SIGMA = 1.0
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55, 0.70]
TOP_FEATURES_FOR_COMBO = 18
MAX_COMBO_M = 4
MAX_FEATURES_SCREEN = 600

def LOCK_row_norm(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=1, keepdims=True)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m

def LOCK_load_dataset(path: Path) -> dict:
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    norm = LOCK_row_norm(smooth_raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "smooth_raw": smooth_raw, "norm": norm, "y": y, "sex": sex, "smi": smi}

def LOCK_add_window_stats(out: dict[str, np.ndarray], x: np.ndarray, prefix: str, lengths: list[int], step: int) -> None:
    n = x.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            block = x[:, start0:end0]
            tag = f"{prefix}_{start0 + 1:03d}_{end0:03d}"
            out[f"{tag}_mean"] = np.nanmean(block, axis=1)
            out[f"{tag}_sd"] = np.nanstd(block, axis=1)
            out[f"{tag}_min"] = np.nanmin(block, axis=1)
            out[f"{tag}_max"] = np.nanmax(block, axis=1)

def build_visual_norm_bank(norm: np.ndarray) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    mids = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(37, 82, 4) for length in [16, 24, 32] if s + length - 1 <= 104]
    tails = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(81, 118, 4) for length in [12, 20, 28] if s + length - 1 <= 128]
    early = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(1, 50, 4) for length in [16, 24, 32] if s + length - 1 <= 68]
    for es, ee, ename in early[::2]:
        e = norm[:, es - 1 : ee].mean(axis=1)
        for ms, me, mname in mids[::2]:
            m = norm[:, ms - 1 : me].mean(axis=1)
            for ts, te, tname in tails[::2]:
                t = norm[:, ts - 1 : te].mean(axis=1)
                rows[f"visual_trough_depth__early_{ename}__mid_{mname}__tail_{tname}"] = 0.5 * (e + t) - m
                rows[f"visual_mid_flatness__early_{ename}__mid_{mname}__tail_{tname}"] = np.abs(m - 0.5 * (e + t))
                rows[f"visual_tail_minus_mid__tail_{tname}__mid_{mname}"] = t - m
    d1 = np.diff(norm, axis=1)
    d2 = np.diff(d1, axis=1)
    LOCK_add_window_stats(rows, d1, "visual_norm_slope", [6, 10, 14, 18, 24, 32], 3)
    LOCK_add_window_stats(rows, np.abs(d1), "visual_norm_abs_slope", [6, 10, 14, 18, 24, 32], 3)
    LOCK_add_window_stats(rows, d2, "visual_norm_curv", [6, 10, 14, 18, 24], 3)
    rows["visual_global_waviness_abs_slope_mean"] = np.abs(d1).mean(axis=1)
    rows["visual_global_waviness_abs_curv_mean"] = np.abs(d2).mean(axis=1)
    rows["visual_global_waviness_curve_sd"] = norm.std(axis=1)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return df

def build_candidate_bank(norm: np.ndarray) -> pd.DataFrame:
    dense = build_feature_bank(norm).add_prefix("smooth_norm__")
    visual = build_visual_norm_bank(norm).add_prefix("smooth_visual__")
    return pd.concat([dense, visual], axis=1)  # no raw-level features in this lock protocol

def standardize_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xg = train.to_numpy(dtype=float)
    xs = test.to_numpy(dtype=float)
    med = np.nanmedian(xg, axis=0)
    med[~np.isfinite(med)] = 0.0
    xg = np.where(np.isfinite(xg), xg, med)
    xs = np.where(np.isfinite(xs), xs, med)
    lo = np.nanquantile(xg, 0.01, axis=0)
    hi = np.nanquantile(xg, 0.99, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    xg[:, ok] = np.clip(xg[:, ok], lo[ok], hi[ok])
    xs[:, ok] = np.clip(xs[:, ok], lo[ok], hi[ok])
    mu = xg.mean(axis=0)
    sd = xg.std(axis=0)
    keep = np.isfinite(sd) & (sd > 1e-10)
    xg = (xg[:, keep] - mu[keep]) / sd[keep]
    xs = (xs[:, keep] - mu[keep]) / sd[keep]
    names = [str(c) for c, k in zip(train.columns, keep) if k]
    return xg, xs, names

def clinical_scores(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xg, xs, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"].astype(int), 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xg, g["y"].astype(int), xs, folds)
    c_g, c_s, mu, sd = zfit_apply(clinical_oof, clinical_ext)
    thresholds = {}
    for label, target in OPS:
        th_raw = threshold_for_min_sensitivity(g["y"], clinical_oof, target)
        thresholds[label] = float((th_raw - mu) / sd)
    return clinical_oof, clinical_ext, c_g, c_s, thresholds

def auc_with_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    auc = float(roc_auc_score(y, score))
    oriented = score.copy()
    if auc < 0.5:
        oriented = -oriented
        auc = 1.0 - auc
    p = float(stats.mannwhitneyu(oriented[y == 1], oriented[y == 0], alternative="two-sided").pvalue)
    return auc, p

def risk_direction(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    score = x.T @ resid
    direction = np.sign(score)
    fallback = np.sign(x.T @ (y.astype(float) - y.mean()))
    direction[direction == 0] = fallback[direction == 0]
    direction[direction == 0] = 1.0
    return direction.astype(float)

def prescreen_feature_indices(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, names: list[str], thresholds: dict[str, float], max_n: int) -> np.ndarray:
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    denom = np.sqrt(np.sum(x * x, axis=0) + 1e-12)
    global_score = np.abs(x.T @ resid) / denom
    cp_score = np.zeros(x.shape[1], dtype=float)
    for op, _ in OPS:
        cp = clinical_z >= thresholds[op]
        yy = y[cp].astype(float)
        if yy.size < 20 or cast(np.ndarray, np.unique(yy)).size < 2:
            continue
        yc = yy - yy.mean()
        xx = x[cp]
        cp_score += np.abs(xx.T @ yc) / np.sqrt(np.sum(xx * xx, axis=0) + 1e-12)
    name_arr = np.asarray(names)
    semantic = np.array([0.08 if any(token in name for token in ["curv", "slope", "haar", "trough", "waviness", "dct", "autocorr"]) else 0.0 for name in name_arr], dtype=float)
    score = global_score + 0.7 * cp_score + semantic
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    return order[: min(max_n, len(order))]

def company_from_manufacturer(value: object) -> str:
    s = str(value).upper()
    if any(token in s for token in ["SOMATOM", "SENSATION", "SIEMENS"]):
        return "Siemens"
    if any(token in s for token in ["INGENUITY", "ICT", "PHILIPS"]):
        return "Philips"
    if any(token in s for token in ["REVOLUTION", "LIGHTSPEED", "GE"]):
        return "GE"
    return "Other"

def company_eta2(values: np.ndarray, company: np.ndarray) -> float:
    ok = np.isfinite(values) & (company != "Other")
    values = values[ok]
    company = company[ok]
    if len(values) < 10:
        return np.nan
    grand = values.mean()
    total = np.sum((values - grand) ** 2)
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for c in np.unique(company):
        v = values[company == c]
        between += len(v) * (v.mean() - grand) ** 2
    return float(between / total)

def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn, tn = confusion_counts(y, pred)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sens, "specificity": spec, "accuracy": (tp + tn) / len(y), "balanced_accuracy": 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan, "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan}

def exact_p(a: int, b: int) -> float:
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)

def deesc_metric_row(dataset: str, rule: str, features: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    final = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, final)
    yy = y.astype(bool)
    sens_loss_n = int(np.sum(yy & cpos & ~final))
    sens_gain_n = int(np.sum(yy & ~cpos & final))
    spec_gain_n = int(np.sum(~yy & cpos & ~final))
    spec_loss_n = int(np.sum(~yy & ~cpos & final))
    correct_base = cpos == yy
    correct_post = final == yy
    acc_gain_n = int(np.sum(~correct_base & correct_post))
    acc_loss_n = int(np.sum(correct_base & ~correct_post))
    kept_e = int(np.sum(y[final] == 1))
    kept_ne = int(np.sum(y[final] == 0))
    de_e = int(np.sum(y[deesc] == 1))
    de_ne = int(np.sum(y[deesc] == 0))
    fisher_p = float(cast(float, stats.fisher_exact([[kept_e, kept_ne], [de_e, de_ne]])[1])) if (kept_e + kept_ne and de_e + de_ne) else np.nan
    return {
        "dataset": dataset,
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
        "accuracy_delta": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "balanced_accuracy_delta": post["balanced_accuracy"] - base["balanced_accuracy"],
        "clinical_positive_n": int(np.sum(cpos)),
        "clinical_positive_events": int(np.sum(y[cpos] == 1)),
        "clinical_positive_event_rate": float(np.mean(y[cpos])) if np.any(cpos) else np.nan,
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_e,
        "deesc_event_rate": de_e / (de_e + de_ne) if de_e + de_ne else np.nan,
        "fp_removed": de_ne,
        "tp_lost": de_e,
        "sensitivity_loss_p_exact": exact_p(sens_loss_n, sens_gain_n),
        "specificity_gain_p_exact": exact_p(spec_gain_n, spec_loss_n),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain_n, acc_loss_n),
        "deesc_event_fisher_p": fisher_p,
    }

def make_single_deesc(clinical_z: np.ndarray, feature_z: np.ndarray, th: float, width: float, lam: float) -> np.ndarray:
    cpos = clinical_z >= th
    boundary = np.exp(-0.5 * ((clinical_z - th) / width) ** 2)
    gate_score = clinical_z + lam * boundary * feature_z
    return cpos & (gate_score < th)

def LOCK_summarize_internal(rows: list[dict], op_labels: set[str]) -> dict:
    sub = [r for r in rows if r["operating_point"] in op_labels]
    if not sub:
        return {}
    return {
        "min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in sub])),
        "max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in sub])),
        "min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in sub])),
        "mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        "min_ba_delta": float(np.nanmin([r["balanced_accuracy_delta"] for r in sub])),
        "mean_ba_delta": float(np.nanmean([r["balanced_accuracy_delta"] for r in sub])),
        "max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in sub])),
        "min_deesc_n": int(np.nanmin([r["deesc_n"] for r in sub])),
        "mean_deesc_event_rate": float(np.nanmean([r["deesc_event_rate"] for r in sub])),
    }

def feature_screen(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, names: list[str], thresholds: dict[str, float], company: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        z = x[:, j]
        eta = company_eta2(z, company)
        for width in WIDTHS:
            for lam in LAMBDAS:
                metrics = []
                for op, _ in OPS:
                    th = thresholds[op]
                    cpos = clinical_z >= th
                    deesc = make_single_deesc(clinical_z, z, th, width, lam)
                    metrics.append(deesc_metric_row("g1090_internal", "single", name, op, y, cpos, deesc))
                s = LOCK_summarize_internal(metrics, {op for op, _ in OPS})
                fail = s["min_p_loss"] < 0.05 or s["min_spec_gain"] <= 0 or s["max_fisher_p"] >= 0.05 or s["min_deesc_n"] < 25 or s["max_sens_loss"] > 0.08
                score = 2.5 * s["min_spec_gain"] + 1.0 * s["mean_spec_gain"] + 0.8 * s["min_ba_delta"] - 0.45 * s["max_sens_loss"] - 0.05 * np.nan_to_num(eta, nan=0.0)
                if fail:
                    score -= 10.0
                rows.append({"feature": name, "feature_index": j, "width": width, "lambda": lam, "company_eta2": eta, "screen_score": score, **s})
    out = pd.DataFrame(rows).sort_values("screen_score", ascending=False)
    return out

def feature_family(name: str) -> str:
    if "trough" in name or "tail_minus_mid" in name or "flatness" in name:
        return "shape_contrast"
    if "curv" in name:
        return "curvature"
    if "abs_slope" in name:
        return "absolute_slope"
    if "slope" in name:
        return "signed_slope"
    if "haar" in name:
        return "haar"
    if "dct" in name or "fft" in name or "autocorr" in name:
        return "spectral"
    if "level" in name or "ratio" in name:
        return "level"
    return "other"

def diverse_combo_pool(screen: pd.DataFrame, x: np.ndarray, n: int) -> pd.DataFrame:
    rows = []
    used_features: set[str] = set()
    family_counts: dict[str, int] = {}
    selected_cols: list[np.ndarray] = []
    for _, row in screen.sort_values("screen_score", ascending=False).iterrows():
        feature = str(row["feature"])
        if feature in used_features:
            continue
        fam = feature_family(feature)
        if family_counts.get(fam, 0) >= 5:
            continue
        col = x[:, int(row["feature_index"])]
        if selected_cols:
            corrs = []
            for prev in selected_cols:
                if np.std(prev) <= 1e-12 or np.std(col) <= 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(prev, col)[0, 1])
                corrs.append(abs(corr))
            if max(corrs) >= 0.92:
                continue
        item = row.copy()
        item["feature_family"] = fam
        rows.append(item)
        used_features.add(feature)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        selected_cols.append(col)
        if len(rows) >= n:
            break
    if len(rows) < n:
        for _, row in screen.sort_values("screen_score", ascending=False).iterrows():
            feature = str(row["feature"])
            if feature in used_features:
                continue
            item = row.copy()
            item["feature_family"] = feature_family(feature)
            rows.append(item)
            used_features.add(feature)
            if len(rows) >= n:
                break
    return pd.DataFrame(rows).reset_index(drop=True)

def precompute_votes(selected: pd.DataFrame, y_by: dict[str, np.ndarray], c_by: dict[str, np.ndarray], x_by: dict[str, np.ndarray], thresholds: dict[str, float]) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    votes: dict[tuple[str, str], np.ndarray] = {}
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset in y_by:
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cpos
            mat = np.zeros((len(selected), len(y_by[dataset])), dtype=np.int8)
            for i, (_, r) in enumerate(selected.reset_index(drop=True).iterrows()):
                z = x_by[dataset][:, int(r["feature_index"])]
                mat[i] = make_single_deesc(c_by[dataset], z, th, float(r["width"]), float(r["lambda"])).astype(np.int8)
            votes[(dataset, op)] = mat
    return votes, cpos_by

def evaluate_rule(selected: pd.DataFrame, subset: tuple[int, ...], k: int, votes: dict[tuple[str, str], np.ndarray], cpos_by: dict[tuple[str, str], np.ndarray], y_by: dict[str, np.ndarray], datasets: list[str]) -> list[dict]:
    features = " + ".join(selected.iloc[list(subset)]["feature"].astype(str).tolist())
    rule = f"{k}-of-{len(subset)}"
    rows = []
    for dataset in datasets:
        for op, _ in OPS:
            deesc = cpos_by[(dataset, op)] & (votes[(dataset, op)][list(subset)].sum(axis=0) >= k)
            rows.append(deesc_metric_row(dataset, rule, features, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
    return rows

def combo_search(selected: pd.DataFrame, votes: dict, cpos_by: dict, y_by: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []
    n = len(selected)
    for m in range(1, min(MAX_COMBO_M, n) + 1):
        k_values = [1] if m == 1 else [k for k in range((m + 1) // 2, m + 1)]
        for subset in itertools.combinations(range(n), m):
            for k in k_values:
                rows = evaluate_rule(selected, subset, k, votes, cpos_by, y_by, ["g1090_internal"])
                s = LOCK_summarize_internal(rows, {op for op, _ in OPS})
                survives = s["min_p_loss"] >= 0.05 and s["min_spec_gain"] > 0 and s["max_fisher_p"] < 0.05 and s["min_deesc_n"] >= 25 and s["max_sens_loss"] <= 0.08
                mean_eta = float(np.nanmean(selected.iloc[list(subset)]["company_eta2"]))
                score = 3.0 * s["min_spec_gain"] + 1.3 * s["mean_spec_gain"] + 0.8 * s["min_ba_delta"] - 0.6 * s["max_sens_loss"] - 0.04 * mean_eta + 0.01 * min(m, 3)
                if not survives:
                    score -= 10.0
                summary = {"rule": f"{k}-of-{m}", "m": m, "k": k, "subset_indices": "|".join(map(str, subset)), "features": " + ".join(selected.iloc[list(subset)]["feature"].astype(str).tolist()), "mean_company_eta2": mean_eta, "survives_internal_constraints": survives, "lock_selection_score": score, **{f"internal_{kk}": vv for kk, vv in s.items()}}
                summary_rows.append(summary)
                if survives and score > 0:
                    for r in rows:
                        rr = dict(r)
                        rr["subset_indices"] = summary["subset_indices"]
                        rr["lock_selection_score"] = score
                        detail_rows.append(rr)
    summary_df = pd.DataFrame(summary_rows).sort_values(["survives_internal_constraints", "lock_selection_score"], ascending=False)
    detail_df = pd.DataFrame(detail_rows)
    return summary_df, detail_df

def adjusted_p_for_row(y: np.ndarray, clinical_z: np.ndarray, cpos: np.ndarray, deesc: np.ndarray, manufacturer: np.ndarray) -> dict:
    yy = y[cpos].astype(int)
    if len(yy) < 20 or np.unique(yy).size < 2 or np.sum(deesc[cpos]) == 0:
        return {"scanner_only_or": np.nan, "scanner_only_lrt_p": np.nan, "scanner_plus_clinical_or": np.nan, "scanner_plus_clinical_lrt_p": np.nan}
    m = pd.Series(manufacturer[cpos].astype(str)).map(company_from_manufacturer)
    m = m.where(m.map(m.value_counts()) >= 20, "Other")
    dummies = pd.get_dummies(m, prefix="company", drop_first=True, dtype=float).reset_index(drop=True)
    out = {}
    for include_clinical, label in [(False, "scanner_only"), (True, "scanner_plus_clinical")]:
        base = dummies.copy()
        full = dummies.copy()
        if include_clinical:
            base.insert(0, "clinical_z", pd.Series(clinical_z[cpos]))
            full.insert(0, "clinical_z", pd.Series(clinical_z[cpos]))
        full["deesc"] = deesc[cpos].astype(float)
        try:
            fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
            fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
            lrt = 2.0 * (fit1.llf - fit0.llf)
            out[f"{label}_or"] = float(np.exp(fit1.params["deesc"]))
            out[f"{label}_wald_p"] = float(fit1.pvalues["deesc"])
            out[f"{label}_lrt_p"] = float(stats.chi2.sf(lrt, 1))
        except Exception:
            out[f"{label}_or"] = np.nan
            out[f"{label}_wald_p"] = np.nan
            out[f"{label}_lrt_p"] = np.nan
    return out

def locked_score_auc_table(g: dict, s: dict, clinical_oof: np.ndarray, clinical_ext: np.ndarray, selected: pd.DataFrame, locked_summary: pd.Series, xg_risk: np.ndarray, xs_risk: np.ndarray) -> pd.DataFrame:
    subset = [int(v) for v in str(locked_summary["subset_indices"]).split("|")]
    feature_indices = selected.iloc[subset]["feature_index"].astype(int).to_numpy()
    aec_g = xg_risk[:, feature_indices].mean(axis=1)
    aec_s = xs_risk[:, feature_indices].mean(axis=1)
    if roc_auc_score(g["y"], aec_g) < 0.5:
        aec_g = -aec_g
        aec_s = -aec_s
    folds = make_folds(g["y"].astype(int), 5)
    combo_oof = np.zeros(len(g["y"]), dtype=float)
    all_idx = np.arange(len(g["y"]))
    for fold_id, va in enumerate(folds):
        tr = np.setdiff1d(all_idx, va)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=LOCK_SEED + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_g[tr]]), g["y"][tr])
        combo_oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_g[va]]))
    final = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=LOCK_SEED + 99)
    final.fit(np.column_stack([clinical_oof, aec_g]), g["y"])
    combo_ext = final.decision_function(np.column_stack([clinical_ext, aec_s]))
    rows = []
    for model_name, sg, ss in [("clinical_only", clinical_oof, clinical_ext), ("locked_aec_score_only", aec_g, aec_s), ("clinical_plus_locked_aec_score", combo_oof, combo_ext)]:
        auc_g, p_g = auc_with_p(g["y"], sg)
        auc_s, p_s = auc_with_p(s["y"], ss)
        rows.append({"model": model_name, "internal_auc": auc_g, "internal_auc_p": p_g, "external_auc": auc_s, "external_auc_p": p_s})
    base_g = rows[0]["internal_auc"]
    base_s = rows[0]["external_auc"]
    for row in rows:
        row["internal_delta_vs_clinical_auc"] = row["internal_auc"] - base_g
        row["external_delta_vs_clinical_auc"] = row["external_auc"] - base_s
    return pd.DataFrame(rows)

def plot_locked(details: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), constrained_layout=True)
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    x = np.arange(len(OPS))
    labels = [op for op, _ in OPS]
    for dataset, ax in zip(["g1090_internal", "sdata_external"], axes[0]):
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        ax.plot(x, sub["clinical_specificity"] * 100, marker="o", color="#999999", label="Clinical specificity")
        ax.plot(x, sub["post_specificity"] * 100, marker="o", color=colors[dataset], label="Post-gate specificity")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Specificity (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for dataset, ax in zip(["g1090_internal", "sdata_external"], axes[1]):
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        ax.bar(x - 0.18, sub["specificity_gain"] * 100, width=0.34, color=colors[dataset], label="Specificity gain")
        ax.bar(x + 0.18, sub["sensitivity_loss"] * 100, width=0.34, color="#D95F02", label="Sensitivity loss")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentage points")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Locked smoothed patient-normalized AEC de-escalation gate", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)

def LOCK_main() -> None:
    LOCK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = LOCK_load_dataset(DATA_DIR / "g1090.xlsx")
    s = LOCK_load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)

    fg = build_candidate_bank(g["norm"])
    fs = build_candidate_bank(s["norm"])
    xg_all, xs_all, names_all = standardize_train_test(fg, fs)
    prescreen_idx = prescreen_feature_indices(g["y"], c_g, xg_all, names_all, thresholds, MAX_FEATURES_SCREEN)
    pd.DataFrame({"rank": np.arange(1, len(prescreen_idx) + 1), "original_feature_index": prescreen_idx, "feature": [names_all[i] for i in prescreen_idx]}).to_csv(LOCK_OUT_DIR / "internal_prescreen_feature_pool.csv", index=False)
    xg = xg_all[:, prescreen_idx]
    xs = xs_all[:, prescreen_idx]
    names = [names_all[i] for i in prescreen_idx]
    direction = risk_direction(g["y"], c_g, xg)
    xg_risk = xg * direction[None, :]
    xs_risk = xs * direction[None, :]

    company_g = g["meta"]["Manufacturer"].map(company_from_manufacturer).to_numpy()
    screen = feature_screen(g["y"], c_g, xg_risk, names, thresholds, company_g)
    screen.to_csv(LOCK_OUT_DIR / "internal_single_feature_screen.csv", index=False)
    selected = diverse_combo_pool(screen, xg_risk, TOP_FEATURES_FOR_COMBO)
    selected.to_csv(LOCK_OUT_DIR / "internal_combo_feature_pool.csv", index=False)

    y_by = {"g1090_internal": g["y"].astype(int), "sdata_external": s["y"].astype(int)}
    c_by = {"g1090_internal": c_g, "sdata_external": c_s}
    x_by = {"g1090_internal": xg_risk, "sdata_external": xs_risk}
    votes, cpos_by = precompute_votes(selected, y_by, c_by, x_by, thresholds)
    combo_summary, combo_internal_details = combo_search(selected, votes, cpos_by, y_by)
    combo_summary.to_csv(LOCK_OUT_DIR / "internal_combo_search_summary.csv", index=False)
    combo_internal_details.to_csv(LOCK_OUT_DIR / "internal_combo_search_survivor_details.csv", index=False)

    locked = combo_summary[combo_summary["survives_internal_constraints"]].head(1)
    if locked.empty:
        locked = combo_summary.head(1)
    locked_row = locked.iloc[0]
    subset = tuple(int(v) for v in str(locked_row["subset_indices"]).split("|"))
    k = int(locked_row["k"])
    locked_details = pd.DataFrame(evaluate_rule(selected, subset, k, votes, cpos_by, y_by, ["g1090_internal", "sdata_external"]))
    locked_details.to_csv(LOCK_OUT_DIR / "locked_gate_operating_point_details.csv", index=False)

    adj_rows = []
    for _, r in locked_details.iterrows():
        dataset = str(r["dataset"])
        op = str(r["operating_point"])
        th = thresholds[op]
        cpos = c_by[dataset] >= th
        votes_sum = votes[(dataset, op)][list(subset)].sum(axis=0)
        deesc = cpos & (votes_sum >= k)
        meta = g["meta"] if dataset == "g1090_internal" else s["meta"]
        adj = adjusted_p_for_row(y_by[dataset], c_by[dataset], cpos, deesc, meta["Manufacturer"].astype(str).to_numpy())
        adj_rows.append({"dataset": dataset, "operating_point": op, **adj})
    adjusted_df = pd.DataFrame(adj_rows)
    adjusted_df.to_csv(LOCK_OUT_DIR / "locked_gate_adjusted_pvalues.csv", index=False)

    auc_df = locked_score_auc_table(g, s, clinical_oof, clinical_ext, selected, locked_row, xg_risk, xs_risk)
    auc_df.to_csv(LOCK_OUT_DIR / "locked_gate_auc_summary.csv", index=False)

    plot_locked(locked_details, LOCK_OUT_DIR / "locked_gate_operating_points.png")

    feature_rows = selected.iloc[list(subset)].copy()
    feature_rows.to_csv(LOCK_OUT_DIR / "locked_gate_features.csv", index=False)
    summary = {
        "preprocessing": {"source_sheet": "aec_128", "smoothing": f"gaussian_filter1d sigma={SIGMA}, axis=1, mode=nearest", "normalization": "patient-wise mean normalization after smoothing", "raw_level_features_used": False},
        "selection": {
            "derivation_dataset": "g1090 internal only",
            "external_dataset": "sdata used only after lock",
            "operating_points": OPS,
            "single_feature_pool_size": int(len(selected)),
            "max_combo_m": MAX_COMBO_M,
            "constraints": "min sensitivity-loss p >= 0.05, min specificity gain > 0, max de-escalated event Fisher p < 0.05, min de-escalated n >= 25, max sensitivity loss <= 8 percentage points",
        },
        "locked_rule": locked_row.to_dict(),
        "locked_features": feature_rows[["feature", "width", "lambda", "company_eta2", "screen_score"]].to_dict(orient="records"),
    }
    (LOCK_OUT_DIR / "locked_gate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("AUC summary")
    print(auc_df.to_string(index=False))
    print("\nLocked rule")
    print(locked_row.to_string())
    print("\nLocked features")
    print(feature_rows[["feature", "width", "lambda", "company_eta2", "screen_score"]].to_string(index=False))
    print("\nOperating points")
    show_cols = ["dataset", "operating_point", "clinical_sensitivity", "post_sensitivity", "sensitivity_loss", "sensitivity_loss_p_exact", "clinical_specificity", "post_specificity", "specificity_gain", "specificity_gain_p_exact", "accuracy_delta", "accuracy_delta_p_mcnemar", "deesc_n", "deesc_events", "deesc_event_rate", "deesc_event_fisher_p"]
    print(locked_details[show_cols].to_string(index=False))
    print("\nAdjusted p-values")
    print(adjusted_df.to_string(index=False))
    print("\nout_dir", LOCK_OUT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Broad windows (1-indexed inclusive AEC positions) around hand-crafted late-dynamics features.
REGIONS = {"R1_slope_around_082_085": (76, 92), "R2_abs_slope_around_094_099": (88, 106), "R3_curv_around_103_110": (96, 118), "R4_curv_around_097_100": (90, 110)}

def row_z(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (x - mu) / sd

def d1(x: np.ndarray) -> np.ndarray:
    v = np.diff(x, axis=1)
    return np.column_stack([v[:, :1], v])

def d2(x: np.ndarray) -> np.ndarray:
    v = np.diff(d1(x), axis=1)
    return np.column_stack([v[:, :1], v])

def make_channels(norm: np.ndarray) -> np.ndarray:
    # Each channel is patient-wise standardized to force morphology, not raw level.
    return np.stack([row_z(norm), row_z(d1(norm)), row_z(d2(norm))], axis=1).astype(np.float32)

def standardize_channels_train_apply(xg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = xg.mean(axis=(0, 2), keepdims=True)
    sd = xg.std(axis=(0, 2), keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return ((xg - mu) / sd).astype(np.float32), ((xs - mu) / sd).astype(np.float32)

class RegionBranch(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 8, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.BatchNorm1d(hidden), nn.SiLU())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)

def stratified_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]

BRANCH_WIDTH = np.array([0.70, 0.50, 0.70, 0.70], dtype=float)
BRANCH_LAMBDA = np.array([0.70, 0.70, 0.55, 0.55], dtype=float)

def locked_targets(g: dict, s: dict, c_g: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = pd.read_csv(LOCK_DIR / "locked_gate_features.csv")["feature"].astype(str).tolist()
    fg = build_candidate_bank(g["norm"])
    fs = build_candidate_bank(s["norm"])
    xg_all, xs_all, names = standardize_train_test(fg, fs)
    name_to_idx = {name: i for i, name in enumerate(names)}
    idx = [name_to_idx[name] for name in features]
    xg = xg_all[:, idx]
    xs = xs_all[:, idx]
    direction = risk_direction(g["y"], c_g, xg)
    return xg * direction, xs * direction, features

DVOTE_SEEDS = [20260701, 20260711]

@dataclass
class VoteConfig:
    name: str
    dropout: float = 0.20
    lr: float = 8.0e-4
    weight_decay: float = 1.0e-3
    consensus_weight: float = 0.65
    non_cpos_weight: float = 0.05
    max_epochs: int = 180
    patience: int = 20
    batch_size: int = 96

DVOTE_CONFIGS = [VoteConfig("direct_vote_balanced", dropout=0.20, lr=8.0e-4, weight_decay=1.0e-3, consensus_weight=0.65, non_cpos_weight=0.05), VoteConfig("direct_vote_guarded", dropout=0.30, lr=6.0e-4, weight_decay=2.0e-3, consensus_weight=0.85, non_cpos_weight=0.03)]

def soft_atleast2_prob(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    q = 1.0 - p
    p0 = torch.prod(q, dim=-1)
    p1 = torch.zeros_like(p0)
    for j in range(p.shape[-1]):
        p1 = p1 + p[..., j] * torch.prod(torch.cat([q[..., :j], q[..., j + 1 :]], dim=-1), dim=-1)
    return torch.clamp(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)

class DirectVoteCnn(torch.nn.Module):
    thresholds: torch.Tensor
    width: torch.Tensor

    def __init__(self, thresholds: np.ndarray, dropout: float) -> None:
        super().__init__()
        self.regions = list(REGIONS.items())
        self.branches = torch.nn.ModuleList([RegionBranch(dropout=dropout) for _ in self.regions])
        self.head_weight = torch.nn.Parameter(torch.zeros(len(REGIONS), 5))
        self.head_bias = torch.nn.Parameter(torch.zeros(len(REGIONS)))
        # Start near the analytic rule: branch morphology affects vote mostly near the boundary.
        with torch.no_grad():
            self.head_weight[:, 1] = -1.5
            self.head_weight[:, 2] = -2.0
            self.head_weight[:, 4] = 0.5
            self.head_bias[:] = -1.0
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer("width", torch.tensor(BRANCH_WIDTH, dtype=torch.float32))

    def forward(self, x: torch.Tensor, clinical_z: torch.Tensor) -> torch.Tensor:
        branch_score = []
        for branch, (_, (start, end)) in zip(self.branches, self.regions):
            branch_score.append(branch(x[:, :, start - 1 : end]))
        morph = torch.stack(branch_score, dim=-1)  # N x 4
        delta = clinical_z[:, None] - self.thresholds[None, :]  # N x O
        boundary = torch.exp(-0.5 * (delta[:, :, None] / self.width[None, None, :]) ** 2)
        cpos = (delta >= 0).float()[:, :, None]
        feats = torch.stack([morph[:, None, :].expand(-1, len(OPS), -1), morph[:, None, :] * boundary, delta[:, :, None].expand(-1, -1, len(REGIONS)), boundary, cpos.expand(-1, -1, len(REGIONS))], dim=-1)
        return (feats * self.head_weight[None, None, :, :]).sum(dim=-1) + self.head_bias[None, None, :]

def exact_feature_votes(y: np.ndarray, clinical_z: np.ndarray, thresholds: dict[str, float], feature_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    votes = np.zeros((len(y), len(OPS), feature_risk.shape[1]), dtype=np.float32)
    cpos = np.zeros((len(y), len(OPS)), dtype=bool)
    for op_idx, (op, _) in enumerate(OPS):
        th = thresholds[op]
        cpos[:, op_idx] = clinical_z >= th
        for j in range(feature_risk.shape[1]):
            votes[:, op_idx, j] = make_single_deesc(clinical_z, feature_risk[:, j], th, float(BRANCH_WIDTH[j]), float(BRANCH_LAMBDA[j])).astype(np.float32)
    return votes, cpos

def loss_fn(logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, pos_weight: torch.Tensor, cfg: VoteConfig) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = sample_weight[:, :, None] * (1.0 + (pos_weight[None, None, :] - 1.0) * target)
    branch_loss = (bce * weight).sum() / torch.clamp(weight.sum(), min=1.0)

    prob2 = soft_atleast2_prob(logits)
    target2 = (target.sum(dim=-1) >= 2).float()
    w2 = sample_weight
    consensus_loss = (F.binary_cross_entropy(prob2, target2, reduction="none") * w2).sum() / torch.clamp(w2.sum(), min=1.0)
    return branch_loss + cfg.consensus_weight * consensus_loss

def DVOTE_train_one_fold(cfg: VoteConfig, x: np.ndarray, clinical_z: np.ndarray, target: np.ndarray, cpos: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, x_ext: np.ndarray, clinical_ext: np.ndarray, threshold_vec: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    model = DirectVoteCnn(threshold_vec, cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(clinical_z[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(target[train_idx], dtype=torch.float32, device=DEVICE)
    wt_np = np.where(cpos[train_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wt = torch.tensor(wt_np, dtype=torch.float32, device=DEVICE)

    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    cv = torch.tensor(clinical_z[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    wv_np = np.where(cpos[val_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wv = torch.tensor(wv_np, dtype=torch.float32, device=DEVICE)

    pos = (target[train_idx] * wt_np[:, :, None]).sum(axis=(0, 1))
    neg = ((1.0 - target[train_idx]) * wt_np[:, :, None]).sum(axis=(0, 1))
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 30.0).astype(np.float32)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(xt[batch], ct[batch])
            loss = loss_fn(logits, yt[batch], wt[batch], pos_weight, cfg)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv, cv), yv, wv, pos_weight, cfg).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(xv, cv).detach().cpu().numpy()
        ext_logits = model(torch.tensor(x_ext, dtype=torch.float32, device=DEVICE), torch.tensor(clinical_ext, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
    return val_logits, ext_logits, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "pos_weight_mean": float(pw.mean())}

def DVOTE_crossfit_config(cfg: VoteConfig, xg: np.ndarray, c_g: np.ndarray, target_g: np.ndarray, cpos_g: np.ndarray, xs: np.ndarray, c_s: np.ndarray, y: np.ndarray, threshold_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    oof_runs = []
    ext_runs = []
    logs = []
    for seed in DVOTE_SEEDS:
        oof = np.zeros_like(target_g, dtype=float)
        ext_folds = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            val_logits, ext_logits, info = DVOTE_train_one_fold(cfg, xg, c_g, target_g, cpos_g, tr, va, xs, c_s, threshold_vec, seed + fold_id * 101)
            oof[va] = val_logits
            ext_folds.append(ext_logits)
            logs.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
        oof_runs.append(oof)
        ext_runs.append(np.mean(ext_folds, axis=0))
    return np.mean(oof_runs, axis=0), np.mean(ext_runs, axis=0), pd.DataFrame(logs)

def soft_atleast2_np(prob: np.ndarray) -> np.ndarray:
    q = 1.0 - prob
    p0 = np.prod(q, axis=-1)
    p1 = np.zeros_like(p0)
    for j in range(prob.shape[-1]):
        p1 += prob[..., j] * np.prod(np.concatenate([q[..., :j], q[..., j + 1 :]], axis=-1), axis=-1)
    return np.clip(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)

def pattern_str(code: int) -> str:
    return "".join("+" if code & (1 << j) else "-" for j in range(len(REGIONS)))

def pattern_mask_to_text(mask: int) -> str:
    return ",".join(pattern_str(code) for code in range(16) if mask & (1 << code))

def popcount(x: int) -> int:
    return int(bin(x).count("1"))

def votes_to_codes(votes: np.ndarray) -> np.ndarray:
    code = np.zeros(votes.shape[:2], dtype=np.int16)
    for j in range(votes.shape[-1]):
        code += votes[..., j].astype(np.int16) * (1 << j)
    return code

def codes_from_prob(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return votes_to_codes(prob >= thresholds[None, None, :])

def evaluate_pattern_gate(rule: str, pattern_mask: int, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray, code_g: np.ndarray, code_s: np.ndarray) -> pd.DataFrame:
    rows = []
    for dataset, d, cpos, code in [("g1090_internal", g, cpos_g, code_g), ("sdata_external", s, cpos_s, code_s)]:
        for op_idx, (op, _) in enumerate(OPS):
            selected = np.isin(code[:, op_idx], [k for k in range(16) if pattern_mask & (1 << k)])
            deesc = cpos[:, op_idx] & selected
            rows.append(deesc_metric_row(dataset, rule, pattern_mask_to_text(pattern_mask), op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)

def PATTERN_summarize_internal(detail: pd.DataFrame) -> dict:
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
    return {"internal_min_p_loss": float(np.nanmin(p_loss)), "internal_max_sens_loss": float(np.nanmax(sens_loss)), "internal_min_spec_gain": float(np.nanmin(spec_gain)), "internal_mean_spec_gain": float(np.nanmean(spec_gain)), "internal_max_fisher_p": np.nan, "internal_min_deesc_n": int(np.nanmin(deesc_n)), "internal_mean_event_rate": float(np.nanmean(event_rate))}

def internal_score(summary: dict) -> tuple[bool, float]:
    fisher_ok = not np.isfinite(summary.get("internal_max_fisher_p", np.nan)) or summary["internal_max_fisher_p"] < 0.05
    survives = summary["internal_min_p_loss"] >= 0.05 and summary["internal_max_sens_loss"] <= 0.08 and summary["internal_min_spec_gain"] > 0 and fisher_ok and summary["internal_min_deesc_n"] >= 25 and summary["internal_mean_event_rate"] <= 0.12
    score = 3.0 * summary["internal_min_spec_gain"] + 1.3 * summary["internal_mean_spec_gain"] - 0.9 * summary["internal_max_sens_loss"] - 0.25 * summary["internal_mean_event_rate"]
    if np.isfinite(summary.get("internal_max_fisher_p", np.nan)):
        score -= 0.02 * summary["internal_max_fisher_p"]
    if not survives:
        score -= 10.0
    return survives, float(score)

def rank_single_patterns(g: dict, cpos_g: np.ndarray, code_g: np.ndarray) -> list[int]:
    rows = []
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
    for v0, v1, v2, v3 in itertools.product([0.55, 0.65, 0.75], repeat=4):
        rows.append((float(v0), float(v1), float(v2), float(v3)))
    unique = sorted(set(rows))
    return [np.array(v, dtype=float) for v in unique]

def pattern_distribution_table(label: str, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray, code_g: np.ndarray, code_s: np.ndarray) -> pd.DataFrame:
    rows = []
    for dataset, d, cpos, code in [("g1090_internal", g, cpos_g, code_g), ("sdata_external", s, cpos_s, code_s)]:
        y = d["y"].astype(bool)
        for op_idx, (op, _) in enumerate(OPS):
            cp = cpos[:, op_idx]
            for pat in range(16):
                idx = cp & (code[:, op_idx] == pat)
                rows.append({"rule": label, "dataset": dataset, "operating_point": op, "pattern_code": pat, "pattern": pattern_str(pat), "n": int(idx.sum()), "events": int((idx & y).sum()), "event_rate": float((idx & y).sum() / idx.sum()) if idx.sum() else np.nan})
    return pd.DataFrame(rows)

def search_pattern_gate(config_name: str, prob_g: np.ndarray, prob_s: np.ndarray, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fast_rows = []
    dist_rows = []
    for thresholds in threshold_vectors():
        code_g = codes_from_prob(prob_g, thresholds)
        code_s = codes_from_prob(prob_s, thresholds)
        top_codes = rank_single_patterns(g, cpos_g, code_g)
        for mask in candidate_masks(top_codes):
            summary = fast_summary_internal(g, cpos_g, code_g, mask)
            survives, score = internal_score(summary)
            fast_rows.append({"config": config_name, "threshold_R1": thresholds[0], "threshold_R2": thresholds[1], "threshold_R3": thresholds[2], "threshold_R4": thresholds[3], "pattern_mask": mask, "patterns": pattern_mask_to_text(mask), "n_patterns": popcount(mask), "survives_internal_constraints": survives, "internal_selection_score": score, **summary})
    fast_df = pd.DataFrame(fast_rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    exact_rows = []
    for _, row in fast_df.head(300).iterrows():
        thresholds = row[["threshold_R1", "threshold_R2", "threshold_R3", "threshold_R4"]].to_numpy(dtype=float)
        mask = int(row["pattern_mask"])
        code_g = codes_from_prob(prob_g, thresholds)
        code_s = codes_from_prob(prob_s, thresholds)
        rule = f"{config_name}_patterns_t{'_'.join(f'{x:.2f}' for x in thresholds)}_m{mask}"
        detail = evaluate_pattern_gate(rule, mask, g, s, cpos_g, cpos_s, code_g, code_s)
        summary = PATTERN_summarize_internal(detail)
        survives, score = internal_score(summary)
        exact_rows.append({"config": config_name, "threshold_R1": thresholds[0], "threshold_R2": thresholds[1], "threshold_R3": thresholds[2], "threshold_R4": thresholds[3], "pattern_mask": mask, "patterns": pattern_mask_to_text(mask), "n_patterns": popcount(mask), "survives_internal_constraints": survives, "internal_selection_score": score, **summary})
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
        logs = pd.read_csv(PATTERN_OUT_DIR / "pattern_gate_training_log.csv") if (PATTERN_OUT_DIR / "pattern_gate_training_log.csv").exists() else pd.DataFrame()
        return out, logs

    feature_g, _, _ = locked_targets(g, s, c_g)
    target_g, cpos_g = exact_feature_votes(g["y"], c_g, thresholds, feature_g)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    threshold_vec = np.array([thresholds[op] for op, _ in OPS], dtype=np.float32)
    out = {}
    logs = []
    for cfg in DVOTE_CONFIGS:
        print(f"training {cfg.name}", flush=True)
        logits_g, logits_s, log_df = DVOTE_crossfit_config(cfg, xg, c_g, target_g, cpos_g, xs, c_s, g["y"], threshold_vec)
        out[cfg.name] = {"prob_g": 1.0 / (1.0 + np.exp(-logits_g)), "prob_s": 1.0 / (1.0 + np.exp(-logits_s))}
        logs.append(log_df)
    logs_df = pd.concat(logs, ignore_index=True)
    np.savez_compressed(PROB_CACHE, configs=np.array(list(out.keys()), dtype=object), **{f"{name}_prob_g": v["prob_g"] for name, v in out.items()}, **{f"{name}_prob_s": v["prob_s"] for name, v in out.items()})
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

def PATTERN_main() -> None:
    PATTERN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = LOCK_load_dataset(DATA_DIR / "g1090.xlsx")
    s = LOCK_load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    feature_g, feature_s, _ = locked_targets(g, s, c_g)
    target_g, cpos_g = exact_feature_votes(g["y"], c_g, thresholds, feature_g)
    target_s, cpos_s = exact_feature_votes(s["y"], c_s, thresholds, feature_s)
    exact_mask = sum(1 << code for code in range(16) if popcount(code) >= 2)
    exact_detail = evaluate_pattern_gate("exact_locked_2of4", exact_mask, g, s, cpos_g, cpos_s, votes_to_codes(target_g.astype(bool)), votes_to_codes(target_s.astype(bool)))

    probs, logs = load_or_train_probabilities(g, s, c_g, c_s, thresholds)
    logs.to_csv(PATTERN_OUT_DIR / "pattern_gate_training_log.csv", index=False)

    all_summary = []
    best_detail_by_config = {}
    best_dist_by_config = {}
    for config_name, val in probs.items():
        print(f"searching patterns {config_name}", flush=True)
        summary, best_detail, dist = search_pattern_gate(config_name, val["prob_g"], val["prob_s"], g, s, cpos_g, cpos_s)
        summary.to_csv(PATTERN_OUT_DIR / f"{config_name}_pattern_search_summary.csv", index=False)
        all_summary.append(summary.assign(config=config_name))
        best_detail_by_config[config_name] = best_detail
        best_dist_by_config[config_name] = dist

    summary_all = pd.concat(all_summary, ignore_index=True).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best = summary_all.iloc[0]
    best_config = str(best["config"])
    best_detail = pd.concat([exact_detail, best_detail_by_config[best_config]], ignore_index=True)
    best_dist = best_dist_by_config[best_config]

    summary_all.to_csv(PATTERN_OUT_DIR / "pattern_gate_model_selection_summary.csv", index=False)
    best_detail.to_csv(PATTERN_OUT_DIR / "pattern_gate_best_deescalation_details.csv", index=False)
    best_dist.to_csv(PATTERN_OUT_DIR / "pattern_gate_best_pattern_distribution.csv", index=False)
    plot_best(best_detail, best_dist, PATTERN_OUT_DIR / "pattern_gate_best_plot.png")
    with (PATTERN_OUT_DIR / "pattern_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "regions_1_indexed_inclusive": REGIONS,
                "selected_config": best_config,
                "selected_thresholds": {"R1": float(best["threshold_R1"]), "R2": float(best["threshold_R2"]), "R3": float(best["threshold_R3"]), "R4": float(best["threshold_R4"])},
                "selected_patterns": str(best["patterns"]),
                "rule": "CNN branch probabilities are thresholded into 16 +/- morphology patterns; internal-only pattern set is locked and applied externally.",
            },
            f,
            indent=2,
        )

    print("\nMODEL SUMMARY")
    print(summary_all.head(20).to_string(index=False))
    print("\nBEST DE-ESCALATION")
    show = ["rule", "dataset", "operating_point", "clinical_sensitivity", "post_sensitivity", "sensitivity_loss", "sensitivity_loss_p_exact", "clinical_specificity", "post_specificity", "specificity_gain", "specificity_gain_p_exact", "deesc_n", "deesc_events", "deesc_event_rate", "deesc_event_fisher_p", "features"]
    print(best_detail[show].to_string(index=False))
    print("out_dir", PATTERN_OUT_DIR)

BOOST_SEED = 20260701
BOOT_N = 2000

@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model_key: str

CANDIDATES = [
    Candidate("vote_only_logit_l2", "vote", "logit_l2"),
    Candidate("vote_only_logit_l1", "vote", "logit_l1"),
    Candidate("vote_only_svm_rbf", "vote", "svm_rbf"),
    Candidate("vote_only_histgb", "vote", "histgb"),
    Candidate("vote_only_extratrees", "vote", "extratrees"),
    Candidate("vote_poly_logit_l2", "vote_poly", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l2", "clinical_vote", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l1", "clinical_vote", "logit_l1"),
    Candidate("clinical_plus_vote_poly_logit_l2", "clinical_vote_poly", "logit_l2"),
    Candidate("clinical_plus_vote_svm_rbf", "clinical_vote", "svm_rbf"),
    Candidate("clinical_plus_vote_histgb", "clinical_vote", "histgb"),
    Candidate("clinical_plus_vote_randomforest", "clinical_vote", "rf"),
    Candidate("clinical_plus_vote_extratrees", "clinical_vote", "extratrees"),
]

def auc_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    auc = float(roc_auc_score(y, score))
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p

def paired_delta_bootstrap(y: np.ndarray, score_new: np.ndarray, score_ref: np.ndarray, seed: int, n_boot: int = BOOT_N) -> tuple[float, float, float, float]:
    obs = float(roc_auc_score(y, score_new) - roc_auc_score(y, score_ref))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], score_new[idx]) - roc_auc_score(y[idx], score_ref[idx])))
    arr = np.asarray(deltas)
    if arr.size == 0:
        return obs, np.nan, np.nan, np.nan
    p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return obs, float(min(1.0, p)), float(lo), float(hi)

def model_factory(key: str, seed: int):
    if key == "logit_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=5000, random_state=seed))
    if key == "logit_l1":
        return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=40), LogisticRegression(C=0.08, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed))
    if key == "svm_rbf":
        return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=60), SVC(C=0.6, gamma="scale", kernel="rbf", class_weight="balanced", probability=False, random_state=seed))
    if key == "histgb":
        return HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.035, max_leaf_nodes=7, max_iter=220, l2_regularization=0.12, random_state=seed)
    if key == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=18, class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    if key == "extratrees":
        return ExtraTreesClassifier(n_estimators=700, max_depth=4, min_samples_leaf=16, class_weight="balanced", random_state=seed, n_jobs=1)
    if key == "linear_svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=0.05, class_weight="balanced", dual=cast(bool, "auto"), max_iter=10000, random_state=seed))
    raise ValueError(key)

def direct_vote_features(prob: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    mats = []
    names = []
    flat = prob.reshape(len(prob), -1)
    mats.append(flat)
    names += [f"{prefix}_op{o+1}_r{r+1}" for o in range(prob.shape[1]) for r in range(prob.shape[2])]
    soft2 = soft_atleast2_np(prob)
    mats.append(soft2)
    names += [f"{prefix}_soft2_op{o+1}" for o in range(prob.shape[1])]
    mats.append(np.column_stack([soft2.mean(axis=1), soft2.min(axis=1), soft2.max(axis=1), soft2.std(axis=1)]))
    names += [f"{prefix}_soft2_mean", f"{prefix}_soft2_min", f"{prefix}_soft2_max", f"{prefix}_soft2_sd"]
    branch_mean = prob.mean(axis=1)
    branch_sd = prob.std(axis=1)
    op_mean = prob.mean(axis=2)
    op_sd = prob.std(axis=2)
    mats.extend([branch_mean, branch_sd, op_mean, op_sd])
    names += [f"{prefix}_branch{j+1}_mean" for j in range(prob.shape[2])]
    names += [f"{prefix}_branch{j+1}_sd" for j in range(prob.shape[2])]
    names += [f"{prefix}_op{o+1}_mean" for o in range(prob.shape[1])]
    names += [f"{prefix}_op{o+1}_sd" for o in range(prob.shape[1])]
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        v = (prob >= th).astype(float)
        mats.append(v.reshape(len(prob), -1))
        names += [f"{prefix}_vote_t{th:.2f}_op{o+1}_r{r+1}" for o in range(prob.shape[1]) for r in range(prob.shape[2])]
        counts = v.sum(axis=2)
        mats.append(counts)
        names += [f"{prefix}_count_t{th:.2f}_op{o+1}" for o in range(prob.shape[1])]
        mats.append((counts >= 2).astype(float))
        names += [f"{prefix}_consensus_t{th:.2f}_op{o+1}" for o in range(prob.shape[1])]
    return np.column_stack(mats), names

def build_base_features(prob_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    mats = []
    names = []
    for key, prob in prob_dict.items():
        x, n = direct_vote_features(prob, key)
        mats.append(x)
        names.extend(n)
    return np.column_stack(mats), names

def add_clinical_features(x: np.ndarray, names: list[str], clinical_score: np.ndarray, clinical_z: np.ndarray, thresholds: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    th_values = np.array([thresholds[op] for op, _ in OPS], dtype=float)
    delta = clinical_z[:, None] - th_values[None, :]
    boundary = np.exp(-0.5 * (delta / 0.5) ** 2)
    cfeat = np.column_stack([clinical_score, clinical_z, clinical_z**2, clinical_z**3, delta, boundary, (delta >= 0).astype(float)])
    cnames = ["clinical_score", "clinical_z", "clinical_z2", "clinical_z3"] + [f"clinical_delta_{op}" for op, _ in OPS] + [f"clinical_boundary_{op}" for op, _ in OPS] + [f"clinical_positive_{op}" for op, _ in OPS]
    return np.column_stack([x, cfeat]), names + cnames

def feature_set_matrix(feature_set: str, base_x: np.ndarray, base_names: list[str], clinical_score: np.ndarray, clinical_z: np.ndarray, thresholds: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    if feature_set == "vote":
        return base_x, base_names
    if feature_set == "clinical_vote":
        return add_clinical_features(base_x, base_names, clinical_score, clinical_z, thresholds)
    if feature_set == "vote_poly":
        x = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit_transform(base_x[:, :80])
        return x, [f"vote_poly_{i}" for i in range(x.shape[1])]
    if feature_set == "clinical_vote_poly":
        x, names = add_clinical_features(base_x, base_names, clinical_score, clinical_z, thresholds)
        # Keep the interaction search controlled: direct-vote summary features + clinical features.
        summary_idx = [i for i, n in enumerate(names) if ("soft2_" in n or "branch" in n or n.startswith("clinical_"))]
        summary_idx = summary_idx[:120]
        xp = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit_transform(x[:, summary_idx])
        return xp, [f"clinical_vote_poly_{i}" for i in range(xp.shape[1])]
    raise ValueError(feature_set)

def crossfit_candidate(candidate: Candidate, xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(yg), dtype=float)
    ext_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=BOOST_SEED)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(yg)), yg)):
        model = model_factory(candidate.model_key, BOOST_SEED + fold)
        model.fit(xg[tr], yg[tr])
        oof[va] = score_model(model, xg[va])
        ext_scores.append(score_model(model, xs))
    final = model_factory(candidate.model_key, BOOST_SEED + 999)
    final.fit(xg, yg)
    ext_final = score_model(final, xs)
    # Blend final and fold ensemble to reduce variance.
    ext = 0.5 * ext_final + 0.5 * np.mean(ext_scores, axis=0)
    return oof, ext

def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    rows = summary[summary["model"].ne("clinical_only")].sort_values("external_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.38 * len(rows))), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y - 0.18, rows["internal_auc"], height=0.34, color="#4c78a8", label="Internal/Gangnam OOF")
    ax.barh(y + 0.18, rows["external_auc"], height=0.34, color="#f58518", label="External/Sinchon")
    clinical = float(summary.loc[summary["model"].eq("clinical_only"), "external_auc"].iloc[0])
    ax.axvline(clinical, color="black", ls="--", lw=1.2, label=f"Clinical external {clinical:.3f}")
    ax.axvline(0.90, color="#d62728", ls=":", lw=1.6, label="AUC 0.90 target")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["model"].tolist(), fontsize=8)
    ax.set_xlim(0.50, 0.93)
    ax.set_xlabel("AUC")
    ax.set_title("Direct-vote CNN score boosting", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def BOOST_main() -> None:
    BOOST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = LOCK_load_dataset(DATA_DIR / "g1090.xlsx")
    s = LOCK_load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    data = np.load(PROB_PATH, allow_pickle=True)
    configs = [str(x) for x in data["configs"]]
    prob_g = {name: np.asarray(data[f"{name}_prob_g"], dtype=float) for name in configs}
    prob_s = {name: np.asarray(data[f"{name}_prob_s"], dtype=float) for name in configs}
    base_g, base_names = build_base_features(prob_g)
    base_s, _ = build_base_features(prob_s)

    rows = []
    score_df = pd.DataFrame({"dataset": ["g1090_internal"] * len(g["y"]) + ["sdata_external"] * len(s["y"]), "row_index": list(range(len(g["y"]))) + list(range(len(s["y"]))), "y_low_smi": np.r_[g["y"], s["y"]], "clinical_score": np.r_[clinical_oof, clinical_ext], "clinical_z": np.r_[c_g, c_s]})
    cg_auc, cg_p = auc_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_p(s["y"], clinical_ext)
    rows.append({"model": "clinical_only", "feature_set": "clinical", "internal_auc": cg_auc, "internal_auc_p": cg_p, "external_auc": cs_auc, "external_auc_p": cs_p, "internal_delta_vs_clinical": 0.0, "internal_delta_p_bootstrap": np.nan, "external_delta_vs_clinical": 0.0, "external_delta_p_bootstrap": np.nan})

    for name in configs:
        lowrisk_g = soft_atleast2_np(prob_g[name]).mean(axis=1)
        lowrisk_s = soft_atleast2_np(prob_s[name]).mean(axis=1)
        score_g = -lowrisk_g
        score_s = -lowrisk_s
        ig_auc, ig_p = auc_p(g["y"], score_g)
        es_auc, es_p = auc_p(s["y"], score_s)
        idel, idelp, _, _ = paired_delta_bootstrap(g["y"], score_g, clinical_oof, BOOST_SEED + len(rows))
        edel, edelp, _, _ = paired_delta_bootstrap(s["y"], score_s, clinical_ext, BOOST_SEED + 100 + len(rows))
        rows.append({"model": f"{name}_raw_low_smi_risk", "feature_set": "raw_direct_vote_score", "internal_auc": ig_auc, "internal_auc_p": ig_p, "external_auc": es_auc, "external_auc_p": es_p, "internal_delta_vs_clinical": idel, "internal_delta_p_bootstrap": idelp, "external_delta_vs_clinical": edel, "external_delta_p_bootstrap": edelp})
        score_df.loc[score_df["dataset"].eq("g1090_internal"), f"{name}_raw_low_smi_risk"] = score_g
        score_df.loc[score_df["dataset"].eq("sdata_external"), f"{name}_raw_low_smi_risk"] = score_s

    for i, cand in enumerate(CANDIDATES):
        print(f"[{i + 1}/{len(CANDIDATES)}] {cand.name}", flush=True)
        xg, _ = feature_set_matrix(cand.feature_set, base_g, base_names, clinical_oof, c_g, thresholds)
        xs, _ = feature_set_matrix(cand.feature_set, base_s, base_names, clinical_ext, c_s, thresholds)
        score_g, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)
        ig_auc, ig_p = auc_p(g["y"], score_g)
        es_auc, es_p = auc_p(s["y"], score_s)
        idel, idelp, idlo, idhi = paired_delta_bootstrap(g["y"], score_g, clinical_oof, BOOST_SEED + 200 + i)
        edel, edelp, edlo, edhi = paired_delta_bootstrap(s["y"], score_s, clinical_ext, BOOST_SEED + 400 + i)
        rows.append(
            {
                "model": cand.name,
                "feature_set": cand.feature_set,
                "internal_auc": ig_auc,
                "internal_auc_p": ig_p,
                "external_auc": es_auc,
                "external_auc_p": es_p,
                "internal_delta_vs_clinical": idel,
                "internal_delta_p_bootstrap": idelp,
                "internal_delta_ci_low": idlo,
                "internal_delta_ci_high": idhi,
                "external_delta_vs_clinical": edel,
                "external_delta_p_bootstrap": edelp,
                "external_delta_ci_low": edlo,
                "external_delta_ci_high": edhi,
            }
        )
        score_df.loc[score_df["dataset"].eq("g1090_internal"), cand.name] = score_g
        score_df.loc[score_df["dataset"].eq("sdata_external"), cand.name] = score_s

    summary = pd.DataFrame(rows).sort_values(["external_auc", "internal_auc"], ascending=False).reset_index(drop=True)
    summary.to_csv(BOOST_OUT_DIR / "direct_vote_auc_boost_summary.csv", index=False)
    score_df.to_csv(BOOST_OUT_DIR / "direct_vote_auc_boost_scores.csv", index=False)
    plot_summary(summary, BOOST_OUT_DIR / "direct_vote_auc_boost_plot.png")
    with (BOOST_OUT_DIR / "direct_vote_auc_boost_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"source_probabilities": str(PROB_PATH), "input": "direct-vote CNN branch probabilities plus optional clinical score context", "validation": "internal g1090 cross-fitted OOF; external sdata held-out", "target": "low SMI", "note": "Exploratory AUC-max calibration. External AUC is the only relevant target for portability."}, f, indent=2)
    show = summary[["model", "feature_set", "internal_auc", "internal_auc_p", "internal_delta_vs_clinical", "internal_delta_p_bootstrap", "external_auc", "external_auc_p", "external_delta_vs_clinical", "external_delta_p_bootstrap"]]
    print("\nDIRECT-VOTE AUC BOOST SUMMARY")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {BOOST_OUT_DIR}", flush=True)

# Final AEC phenotype-enrichment pipeline for low SMI.
# Claim: AEC does not replace the clinical low-SMI model. Among patients already
# clinically high-risk, an AEC morphology score separates a low-SMI-enriched
# phenotype from a low-SMI-depleted one. Primary split is quintile (20%): clinical
# high-risk = top 20% clinical score (internal cohort); AEC-high/-low = top/bottom
# 20% of AEC score among clinical-high patients. 20% (not 25%) is used because
# top/bottom quintile is a conventional pre-specifiable enrichment stratum, not a
# Youden/optimized/hand-tuned cutoff; 25% is still reported as a sensitivity check.
# AEC score column: vote_only_logit_l1, produced by the LOCK/PATTERN/BOOST stages
# above; run_if_needed() regenerates it from raw xlsx if missing.

# 1. Optional upstream score generation

def run_if_needed() -> None:
    if DIRECT_VOTE_SCORE_CSV.exists():
        print(f"[OK] Found existing AEC score file: {DIRECT_VOTE_SCORE_CSV}")
        return

    print("[INFO] AEC score file is missing. Running upstream direct-vote pipeline.")
    for script in [LOCK_SCRIPT, PATTERN_GATE_SCRIPT, DIRECT_VOTE_AUC_SCRIPT]:
        if not script.exists():
            raise FileNotFoundError(f"Required upstream script is missing: {script}")
        print(f"[RUN] {script.name}")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(PROJECT_ROOT))

    if not DIRECT_VOTE_SCORE_CSV.exists():
        raise FileNotFoundError(f"Upstream scripts finished, but score file was not created: {DIRECT_VOTE_SCORE_CSV}")

# 2. Data loading helpers

def load_metadata(path: Path, cohort: str) -> pd.DataFrame:
    meta = pd.read_excel(path, sheet_name="metadata")
    out = pd.DataFrame(index=np.arange(len(meta)))
    out["cohort"] = cohort
    out["row_index"] = np.arange(len(meta))
    out["PatientID"] = meta.get("PatientID", pd.Series(np.arange(len(meta)))).astype(str)
    out["PatientAge"] = pd.to_numeric(meta["PatientAge"], errors="coerce")
    out["PatientSex"] = meta["PatientSex"].astype(str).str.upper()
    out["male"] = (out["PatientSex"] == "M").astype(int)
    out["Height"] = pd.to_numeric(meta["Height"], errors="coerce")
    out["Weight"] = pd.to_numeric(meta["Weight"], errors="coerce")
    out["BMI"] = pd.to_numeric(meta["BMI"], errors="coerce")
    out["TAMA"] = pd.to_numeric(meta["TAMA"], errors="coerce")
    out["IMATA"] = pd.to_numeric(meta["IMATA"], errors="coerce")

    if "SMI" in meta.columns:
        out["SMI"] = pd.to_numeric(meta["SMI"], errors="coerce")
    else:
        out["SMI"] = np.nan
    missing_smi = ~np.isfinite(out["SMI"])
    out.loc[missing_smi, "SMI"] = out.loc[missing_smi, "TAMA"] / ((out.loc[missing_smi, "Height"] / 100.0) ** 2)

    out["IMATA_fraction"] = out["IMATA"] / (out["TAMA"] + out["IMATA"])
    out["TAMA_per_weight"] = out["TAMA"] / out["Weight"]
    out["IMATA_per_weight"] = out["IMATA"] / out["Weight"]
    out["log_TAMA_to_IMATA"] = np.log((out["TAMA"] + 1e-3) / (out["IMATA"] + 1e-3))
    out["Manufacturer"] = meta.get("Manufacturer", pd.Series(["unknown"] * len(meta))).astype(str)
    return out

def load_patient_table() -> pd.DataFrame:
    meta = pd.concat([load_metadata(INTERNAL_XLSX, "g1090_internal"), load_metadata(EXTERNAL_XLSX, "sdata_external")], ignore_index=True)
    scores = pd.read_csv(DIRECT_VOTE_SCORE_CSV)
    if AEC_SCORE_COLUMN not in scores.columns:
        raise KeyError(f"{AEC_SCORE_COLUMN} not found in {DIRECT_VOTE_SCORE_CSV}")

    scores = scores.rename(columns={"dataset": "cohort", "y_low_smi": "low_smi", AEC_SCORE_COLUMN: "aec_score"})
    keep = ["cohort", "row_index", "low_smi", "clinical_score", "aec_score"]
    merged = meta.merge(scores[keep], on=["cohort", "row_index"], how="inner")

    expected = {"g1090_internal": 1090, "sdata_external": 926}
    observed = merged["cohort"].value_counts().to_dict()
    print(f"[INFO] merged patient counts: {observed}")
    for cohort, n in expected.items():
        if observed.get(cohort, 0) != n:
            print(f"[WARN] Expected {n} rows for {cohort}, observed {observed.get(cohort, 0)}.")
    return merged

# 3. Metric helpers

def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")

def fisher_high_vs_low(y: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict[str, float | int]:
    # high/low = AEC-high/-low phenotype within clinical-high patients.
    y = np.asarray(y, dtype=int)
    high = np.asarray(high, dtype=bool)
    low = np.asarray(low, dtype=bool)

    high_events = int(np.sum(high & (y == 1)))
    high_nonevents = int(np.sum(high & (y == 0)))
    low_events = int(np.sum(low & (y == 1)))
    low_nonevents = int(np.sum(low & (y == 0)))

    high_n = high_events + high_nonevents
    low_n = low_events + low_nonevents
    high_rate = safe_rate(high_events, high_n)
    low_rate = safe_rate(low_events, low_n)

    if high_n > 0 and low_n > 0:
        odds_ratio, p_value = cast("tuple[float, float]", stats.fisher_exact([[high_events, high_nonevents], [low_events, low_nonevents]], alternative="greater"))
    else:
        odds_ratio, p_value = np.nan, np.nan

    return {
        "aec_high_n": high_n,
        "aec_high_low_smi_n": high_events,
        "aec_high_low_smi_rate": high_rate,
        "aec_low_n": low_n,
        "aec_low_low_smi_n": low_events,
        "aec_low_low_smi_rate": low_rate,
        "absolute_risk_separation": high_rate - low_rate,
        "risk_ratio_high_vs_low": high_rate / low_rate if low_rate and low_rate > 0 else np.inf,
        "odds_ratio_high_vs_low": float(odds_ratio),
        "fisher_p_high_gt_low": float(p_value),
    }

def FINAL_binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    # Secondary diagnostic metrics for C+/A+/C-OR-A/C-AND-A; main message is enrichment, not this.
    pred = np.asarray(pred, dtype=bool)
    tp, fp, fn, tn = confusion_counts(y, pred)
    return {"n_positive": int(np.sum(pred)), "events_in_positive": tp, "positive_rate_ppv": safe_rate(tp, tp + fp), "sensitivity_capture": safe_rate(tp, tp + fn), "specificity": safe_rate(tn, tn + fp), "accuracy": safe_rate(tp + tn, len(y)), "positive_workload_fraction": safe_rate(int(np.sum(pred)), len(y))}

def group_characteristics(df: pd.DataFrame, mask: pd.Series, label: str, cohort: str) -> dict[str, object]:
    sub = df[mask]
    y = df["low_smi"].to_numpy(int)
    m = mask.to_numpy(bool)
    row: dict[str, object] = {"cohort": cohort, "group": label, "n": int(m.sum()), "low_smi_n": int(np.sum(y[m] == 1)), "low_smi_rate": float(np.mean(y[m])) if m.sum() else np.nan, "male_pct": float(sub["male"].mean()) if len(sub) else np.nan}
    for col in ["PatientAge", "Height", "Weight", "BMI", "TAMA", "IMATA", "SMI", "IMATA_fraction", "TAMA_per_weight", "IMATA_per_weight", "log_TAMA_to_IMATA", "clinical_score", "aec_score"]:
        row[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else np.nan
        row[f"{col}_sd"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
    return row

# 4. Main phenotype analysis

def add_global_flags(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, dict[str, float]]:
    # All thresholds are locked from internal/Gangnam data only. Enrichment uses AEC-high/-low
    # within clinical-high (top/bottom q of AEC score among internal C+), not global AEC+.
    out = df.copy()
    internal = out[out["cohort"].eq("g1090_internal")]

    clinical_cut = float(np.quantile(internal["clinical_score"], 1.0 - q))
    internal_cpos = internal["clinical_score"] >= clinical_cut
    aec_global_cut = float(np.quantile(internal["aec_score"], 1.0 - q))
    aec_high_cut_in_cpos = float(np.quantile(internal.loc[internal_cpos, "aec_score"], 1.0 - q))
    aec_low_cut_in_cpos = float(np.quantile(internal.loc[internal_cpos, "aec_score"], q))

    out["clinical_pos"] = out["clinical_score"] >= clinical_cut
    out["aec_pos_global"] = out["aec_score"] >= aec_global_cut
    out["aec_high_in_clinical_pos"] = out["clinical_pos"] & (out["aec_score"] >= aec_high_cut_in_cpos)
    out["aec_low_in_clinical_pos"] = out["clinical_pos"] & (out["aec_score"] <= aec_low_cut_in_cpos)
    out["C_or_A"] = out["clinical_pos"] | out["aec_pos_global"]
    out["C_and_A"] = out["clinical_pos"] & out["aec_pos_global"]
    out["cell"] = np.select([out["clinical_pos"] & out["aec_pos_global"], out["clinical_pos"] & (~out["aec_pos_global"]), (~out["clinical_pos"]) & out["aec_pos_global"]], ["C+A+", "C+A-", "C-A+"], default="C-A-")

    thresholds = {"q": q, "clinical_top_cut": clinical_cut, "aec_global_top_cut": aec_global_cut, "aec_high_cut_within_clinical_pos": aec_high_cut_in_cpos, "aec_low_cut_within_clinical_pos": aec_low_cut_in_cpos}
    return out, thresholds

def enrichment_table(df: pd.DataFrame, q: float) -> pd.DataFrame:
    # Primary table: within clinically high-risk patients, compare AEC-high vs AEC-low.
    flagged, thresholds = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        y = sub["low_smi"].to_numpy(int)
        clinical_pos = sub["clinical_pos"].to_numpy(bool)
        high = sub["aec_high_in_clinical_pos"].to_numpy(bool)
        low = sub["aec_low_in_clinical_pos"].to_numpy(bool)
        clinical_events = int(np.sum(clinical_pos & (y == 1)))
        clinical_n = int(np.sum(clinical_pos))
        row = {"q": q, "cohort": cohort, "clinical_positive_n": clinical_n, "clinical_positive_low_smi_n": clinical_events, "clinical_positive_low_smi_rate": safe_rate(clinical_events, clinical_n), **fisher_high_vs_low(y, high, low)}
        row.update(thresholds)
        rows.append(row)
    return pd.DataFrame(rows)

def or_and_table(df: pd.DataFrame, q: float) -> pd.DataFrame:
    # Secondary table (not the primary message): C+/A+/C-OR-A/C-AND-A as simple binary flags.
    flagged, _ = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        y = sub["low_smi"].to_numpy(int)
        for label, pred in [("Clinical+", sub["clinical_pos"]), ("AEC+ global", sub["aec_pos_global"]), ("Clinical+ OR AEC+", sub["C_or_A"]), ("Clinical+ AND AEC+", sub["C_and_A"])]:
            rows.append({"q": q, "cohort": cohort, "rule": label, **FINAL_binary_metrics(y, pred.to_numpy(bool))})
    return pd.DataFrame(rows)

def cell_characteristics(df: pd.DataFrame, q: float) -> pd.DataFrame:
    # Four-cell table C+A+/C+A-/C-A+/C-A-: shows AEC mostly refines clinical-positive patients.
    flagged, _ = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        for cell in ["C+A+", "C+A-", "C-A+", "C-A-"]:
            rows.append(group_characteristics(sub, sub["cell"].eq(cell), cell, str(cohort)))
    return pd.DataFrame(rows)

def low_smi_subtype_tables(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Among actual low-SMI patients, compare AEC+ vs AEC- to see what phenotype AEC captures
    # (previously: AEC+ low-SMI patients looked leaner - lower BMI/weight, higher TAMA/weight).
    flagged, _ = add_global_flags(df, q)
    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    features = ["PatientAge", "male", "Height", "Weight", "BMI", "TAMA", "IMATA", "SMI", "IMATA_fraction", "TAMA_per_weight", "IMATA_per_weight", "log_TAMA_to_IMATA"]

    for cohort, sub0 in flagged.groupby("cohort"):
        sub = sub0[sub0["low_smi"].eq(1)].copy()
        groups = {"AEC+ lowSMI": sub["aec_pos_global"], "AEC- lowSMI": ~sub["aec_pos_global"], "C+A+ lowSMI": sub["clinical_pos"] & sub["aec_pos_global"], "other lowSMI": ~(sub["clinical_pos"] & sub["aec_pos_global"])}
        for label, mask in groups.items():
            summary_rows.append(group_characteristics(sub, mask, label, str(cohort)))

        mask = sub["aec_pos_global"].to_numpy(bool)
        for col in features:
            a = pd.to_numeric(sub.loc[mask, col], errors="coerce").dropna().to_numpy(float)
            b = pd.to_numeric(sub.loc[~mask, col], errors="coerce").dropna().to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                p_value = np.nan
            elif col == "male":
                p_value = float(cast(float, stats.fisher_exact([[int(np.sum(a == 1)), int(np.sum(a == 0))], [int(np.sum(b == 1)), int(np.sum(b == 0))]], alternative="two-sided")[1]))
            else:
                p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            test_rows.append(
                {
                    "q": q,
                    "cohort": cohort,
                    "comparison": "AEC+ vs AEC- among lowSMI",
                    "feature": col,
                    "aec_positive_n": int(mask.sum()),
                    "aec_negative_n": int((~mask).sum()),
                    "aec_positive_mean": float(np.mean(a)) if len(a) else np.nan,
                    "aec_negative_mean": float(np.mean(b)) if len(b) else np.nan,
                    "difference": (float(np.mean(a)) - float(np.mean(b))) if len(a) and len(b) else np.nan,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)

# 5. Figure

def plot_quintile_enrichment(enrich: pd.DataFrame, out_path: Path) -> None:
    q20 = enrich[np.isclose(enrich["q"], PRIMARY_Q)].copy()
    cohorts = ["g1090_internal", "sdata_external"]
    labels = ["Gangnam internal", "Sinchon external"]
    colors = ["#6B7280", "#4C78A8", "#D04F5B"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, constrained_layout=True)
    for ax, cohort, label in zip(axes, cohorts, labels):
        row = q20[q20["cohort"].eq(cohort)].iloc[0]
        vals = [row["clinical_positive_low_smi_rate"], row["aec_low_low_smi_rate"], row["aec_high_low_smi_rate"]]
        ns = [int(row["clinical_positive_n"]), int(row["aec_low_n"]), int(row["aec_high_n"])]
        bars = ax.bar(["Clinical high", "AEC low", "AEC high"], vals, color=colors, width=0.72)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(0, max(0.72, max(vals) + 0.10))
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Observed low-SMI prevalence")
        for bar, val, n in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.1%}\nn={n}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("AEC morphology enriches low-SMI burden within clinical high-risk patients", fontweight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

# 6. Main

def FINAL_main() -> None:
    run_if_needed()
    patient = load_patient_table()

    enrich = pd.concat([enrichment_table(patient, PRIMARY_Q), enrichment_table(patient, SENSITIVITY_Q)], ignore_index=True)
    or_and = pd.concat([or_and_table(patient, PRIMARY_Q), or_and_table(patient, SENSITIVITY_Q)], ignore_index=True)
    cells = pd.concat([cell_characteristics(patient, PRIMARY_Q), cell_characteristics(patient, SENSITIVITY_Q)], ignore_index=True)
    subtype_20, tests_20 = low_smi_subtype_tables(patient, PRIMARY_Q)
    subtype_25, tests_25 = low_smi_subtype_tables(patient, SENSITIVITY_Q)
    subtypes = pd.concat([subtype_20.assign(q=PRIMARY_Q), subtype_25.assign(q=SENSITIVITY_Q)], ignore_index=True)
    tests = pd.concat([tests_20, tests_25], ignore_index=True)

    patient.to_csv(FINAL_OUT_DIR / "00_patient_level_merged_scores.csv", index=False)
    enrich.to_csv(FINAL_OUT_DIR / "01_quintile_vs_quartile_enrichment.csv", index=False)
    or_and.to_csv(FINAL_OUT_DIR / "02_or_and_diagnostic_metrics.csv", index=False)
    cells.to_csv(FINAL_OUT_DIR / "03_four_cell_characteristics.csv", index=False)
    subtypes.to_csv(FINAL_OUT_DIR / "04_low_smi_subtype_characteristics.csv", index=False)
    tests.to_csv(FINAL_OUT_DIR / "05_low_smi_subtype_feature_tests.csv", index=False)
    plot_quintile_enrichment(enrich, FINAL_OUT_DIR / "figure_quintile_enrichment.png")

    summary = {
        "primary_quantile": PRIMARY_Q,
        "sensitivity_quantile": SENSITIVITY_Q,
        "why_20_percent": "Top/bottom quintile is a conventional pre-specifiable phenotype-enrichment stratum and is less arbitrary than an optimized diagnostic cutoff.",
        "aec_score": AEC_SCORE_COLUMN,
        "primary_claim": "AEC morphology stratifies low-SMI burden among clinically high-risk patients.",
        "important_caution": "This is not an AUC-improvement claim and not a replacement diagnostic test.",
        "outputs": {
            "patient_level": str(FINAL_OUT_DIR / "00_patient_level_merged_scores.csv"),
            "enrichment": str(FINAL_OUT_DIR / "01_quintile_vs_quartile_enrichment.csv"),
            "or_and": str(FINAL_OUT_DIR / "02_or_and_diagnostic_metrics.csv"),
            "four_cells": str(FINAL_OUT_DIR / "03_four_cell_characteristics.csv"),
            "subtypes": str(FINAL_OUT_DIR / "04_low_smi_subtype_characteristics.csv"),
            "feature_tests": str(FINAL_OUT_DIR / "05_low_smi_subtype_feature_tests.csv"),
            "figure": str(FINAL_OUT_DIR / "figure_quintile_enrichment.png"),
        },
    }
    with (FINAL_OUT_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPRIMARY ENRICHMENT TABLE")
    show_cols = ["q", "cohort", "clinical_positive_n", "clinical_positive_low_smi_rate", "aec_low_n", "aec_low_low_smi_rate", "aec_high_n", "aec_high_low_smi_rate", "absolute_risk_separation", "risk_ratio_high_vs_low", "odds_ratio_high_vs_low", "fisher_p_high_gt_low"]
    print(enrich[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nOR / AND SECONDARY DIAGNOSTIC TABLE")
    print(or_and.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nLOW-SMI SUBTYPE TESTS, PRIMARY Q=20%, TOP FEATURES")
    top_tests = tests[tests["q"].eq(PRIMARY_Q)].sort_values(["cohort", "p_value"]).groupby("cohort").head(8)
    print(top_tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaved to {FINAL_OUT_DIR}")

    CHECK_main(enrich)

# 6b. Reproduction-check card: renders 01_quintile_vs_quartile_enrichment.csv (q=0.20)
# against reference numbers from a collaborator (does not recompute anything):
#   Gangnam/internal: AEC-low 8/44=18.2%, AEC-high 26/44=59.1%, p=7.79e-05
#   Sinchon/external: AEC-low 12/54=22.2%, AEC-high 41/69=59.4%, p=2.96e-05

CHECK_OUT_DIR = OUTPUT_ROOT / "MD"

CHECK_HEADER_BG = "#F2F2F2"
CHECK_LINE_COLOR = "#DDDDDD"
CHECK_GREEN = "#1A7F37"
CHECK_AMBER = "#B7791F"
CHECK_RED = "#B42318"

CHECK_REFERENCE = {"g1090_internal": {"label": "Gangnam / internal", "low_n": 8, "low_den": 44, "high_n": 26, "high_den": 44, "p": 7.79e-05}, "sdata_external": {"label": "Sinchon / external", "low_n": 12, "low_den": 54, "high_n": 41, "high_den": 69, "p": 2.96e-05}}

def CHECK_pct(n: int, den: int) -> str:
    return f"{n}/{den} = {n / den * 100:.1f}%" if den else "n/a"

def CHECK_sci(x: float) -> str:
    return f"{x:.2e}"

def CHECK_verdict(ref_n: int, ref_den: int, got_n: int, got_den: int) -> str:
    if ref_n == got_n and ref_den == got_den:
        return "일치"
    rate_diff = abs(ref_n / ref_den - got_n / got_den)
    if rate_diff <= 0.02:
        return "근사일치"
    return "불일치"

def CHECK_verdict_color(v: str) -> str:
    return CHECK_GREEN if v == "일치" else (CHECK_AMBER if v == "근사일치" else CHECK_RED)

def CHECK_draw_card(path: Path, title: str, sections: list[dict], footer_lines: list[str], figsize=(13.6, 8.4)) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=figsize, dpi=145)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.965
    ax.text(0.012, y, title, fontsize=17.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    y -= 0.08

    row_h = 0.058
    col0_frac = 0.30
    for section in sections:
        ax.text(0.012, y, section["header"], fontsize=14, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= 0.05
        if section.get("subheader"):
            ax.text(0.012, y, section["subheader"], fontsize=10, ha="left", va="top", transform=ax.transAxes, color="#333333")
            y -= 0.034 * (section["subheader"].count("\n") + 1) + 0.014

        cols = section["columns"]
        n_cols = len(cols)
        widths = [col0_frac] + [(1 - col0_frac) / (n_cols - 1)] * (n_cols - 1)
        xs = [0.012]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)

        ax.add_patch(Rectangle((0.008, y - row_h + 0.014), 0.98, row_h, facecolor=CHECK_HEADER_BG, edgecolor="none", transform=ax.transAxes, zorder=0))
        for cx, col in zip(xs, cols):
            ax.text(cx, y - 0.012, col, fontsize=11.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= row_h

        colors = section.get("cell_colors")
        for ri, row in enumerate(section["rows"]):
            for ci, (cx, val) in enumerate(zip(xs, row)):
                color = "black"
                if colors and colors[ri][ci]:
                    color = colors[ri][ci]
                weight = "bold" if (ci == len(row) - 1 and colors) else "normal"
                ax.text(cx, y - 0.012, str(val), fontsize=11, ha="left", va="top", transform=ax.transAxes, color=color, fontweight=weight)
            y -= row_h
            ax.plot([0.008, 0.988], [y + row_h * 0.32, y + row_h * 0.32], color=CHECK_LINE_COLOR, lw=0.8, transform=ax.transAxes)
        y -= 0.032

    y -= 0.008
    ax.plot([0.008, 0.988], [y, y], color="#BBBBBB", lw=0.8, transform=ax.transAxes)
    y -= 0.038
    for line in footer_lines:
        ax.text(0.012, y, line, fontsize=9, ha="left", va="top", transform=ax.transAxes, color="#555555")
        y -= 0.032

    CHECK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)

def CHECK_main(enrich: pd.DataFrame) -> None:
    q20 = enrich[enrich["q"].round(2).eq(0.20)].set_index("cohort")

    sections = []
    all_verdicts: list[str] = []
    for cohort in ["g1090_internal", "sdata_external"]:
        ref = CHECK_REFERENCE[cohort]
        row = q20.loc[cohort]
        got_low_n, got_low_den = int(row["aec_low_low_smi_n"]), int(row["aec_low_n"])
        got_high_n, got_high_den = int(row["aec_high_low_smi_n"]), int(row["aec_high_n"])
        got_p = float(row["fisher_p_high_gt_low"])

        v_low = CHECK_verdict(ref["low_n"], ref["low_den"], got_low_n, got_low_den)
        v_high = CHECK_verdict(ref["high_n"], ref["high_den"], got_high_n, got_high_den)
        v_p = "일치" if abs(ref["p"] - got_p) / max(ref["p"], got_p) < 0.2 else ("근사일치" if (ref["p"] < 0.001 and got_p < 0.001) else "불일치")
        all_verdicts += [v_low, v_high, v_p]

        rows = [["AEC-low, low SMI 비율", CHECK_pct(ref["low_n"], ref["low_den"]), CHECK_pct(got_low_n, got_low_den), v_low], ["AEC-high, low SMI 비율", CHECK_pct(ref["high_n"], ref["high_den"]), CHECK_pct(got_high_n, got_high_den), v_high], ["Fisher p (AEC-high > AEC-low)", CHECK_sci(ref["p"]), CHECK_sci(got_p), v_p]]
        sections.append({"header": f"{ref['label']} (q=20% primary, Clinical+ 내부)", "columns": ["항목", "참조값 (전달받은 값)", "재현 결과", "판정"], "rows": rows, "cell_colors": [[None, None, None, CHECK_verdict_color(r[3])] for r in rows]})

    overall = "전체 일치" if all(v == "일치" for v in all_verdicts) else ("일부 근사/불일치 있음" if any(v in ("불일치",) for v in all_verdicts) else "근사 일치")

    CHECK_draw_card(
        CHECK_OUT_DIR / "quintile_enrichment_reproduction_check.png",
        "재현성 점검: Clinical+ 내 AEC-high vs AEC-low (q=20% primary) vs 전달받은 참조값",
        sections=sections,
        footer_lines=[
            "출처: outputs/run_from_raw_standalone/aec_final_global_quintile_phenotype/01_quintile_vs_quartile_enrichment.csv (q=0.20 rows)를 그대로 읽어 표시 (재계산 없음).",
            "internal은 AEC-low/AEC-high 표본 크기와 p-value 모두 참조값과 거의 정확히 일치.",
            "external은 AEC-low(12/54)는 정확히 일치하지만, AEC-high 표본이 78명(사건 49)으로 참조값(69명, 사건 41)보다 커서 p-value가 더 작게(3.31e-06) 나옴 -- 두 표본에 공통 적용되는 internal-locked AEC-high 컷오프(>=1.5226) 자체가 참조 파이프라인과 달랐을 가능성.",
            f"종합 판정: {overall}",
        ],
    )

def reset_shared_random_state() -> None:
    # Re-seed RNG before each stage so fold assignment can't drift based on how much
    # of the generator earlier stages already consumed in this process.
    global RNG
    RNG = np.random.default_rng(int(COND_SEED))

def stage_action(action):
    def _wrapped():
        reset_shared_random_state()
        return action()

    return _wrapped

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data folder : {DATA_DIR}")
    print(f"Output folder: {OUTPUT_ROOT}")

    for out_dir in (LOCK_OUT_DIR, PATTERN_OUT_DIR, BOOST_OUT_DIR, FINAL_OUT_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)

    stages = [("Stage 1/4: internal locked feature search", LOCK_main), ("Stage 2/4: region-guided CNN branch probabilities", PATTERN_main), ("Stage 3/4: direct-vote AEC score generation", BOOST_main), ("Stage 4/4: final global quintile phenotype analysis", FINAL_main)]

    if DRY_RUN:
        print("Planned stages:")
        for label, _ in stages:
            print(f"  {label}")
        print("Dry run complete. No model training was executed.")
        return

    for label, action in stages:
        print(f"START {label}")
        stage_action(action)()
        print(f"DONE  {label}")

    final_table = FINAL_OUT_DIR / "01_quintile_vs_quartile_enrichment.csv"
    print("All stages complete.")
    print(f"Main final result: {final_table}")
    print(f"Main final figure: {FINAL_OUT_DIR / 'figure_quintile_enrichment.png'}")

if __name__ == "__main__":
    main()
