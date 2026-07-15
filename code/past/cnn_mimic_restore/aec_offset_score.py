from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.fft import dct
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, binary_metrics, load_dataset, threshold_youden  # noqa: E402


OUT_DIR = Path(r"C:\Users\user\OneDrive\Dokumen\radiation\outputs\aec_offset_score")
SEED = 20260629


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def clinical_raw(meta: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float),
            (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float),
            pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float),
        ]
    )


def fit_clinical(x: np.ndarray, y: np.ndarray) -> Pipeline:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )
    model.fit(x, y)
    return model


def apply_clinical(model: Pipeline, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    return model.decision_function(x)


def curve_features(curve: np.ndarray, prefix: str, n_dct: int = 12) -> tuple[np.ndarray, list[str]]:
    curve = np.asarray(curve, dtype=float)
    n, p = curve.shape
    x = curve - curve.mean(axis=1, keepdims=True)
    coeff = dct(x, type=2, norm="ortho", axis=1)[:, 1 : n_dct + 1]
    names = [f"{prefix}_dct_{i:02d}" for i in range(1, n_dct + 1)]

    pos = np.linspace(-1.0, 1.0, p)
    denom = float(np.sum(pos**2)) or 1.0
    slope = (x @ pos) / denom
    d1 = np.diff(curve, axis=1)
    d2 = np.diff(curve, n=2, axis=1)
    third = max(1, p // 3)
    early = curve[:, :third].mean(axis=1)
    mid = curve[:, third : 2 * third].mean(axis=1)
    late = curve[:, 2 * third :].mean(axis=1)
    q05 = np.quantile(curve, 0.05, axis=1)
    q95 = np.quantile(curve, 0.95, axis=1)
    max_pos = np.argmax(curve, axis=1) / max(1, p - 1)
    min_pos = np.argmin(curve, axis=1) / max(1, p - 1)
    stats_block = np.column_stack(
        [
            curve.std(axis=1),
            q95 - q05,
            np.abs(d1).mean(axis=1),
            d1.std(axis=1),
            np.abs(d2).mean(axis=1),
            slope,
            early,
            mid,
            late,
            mid - early,
            late - mid,
            late - early,
            max_pos,
            min_pos,
            max_pos - min_pos,
        ]
    )
    stat_names = [
        "sd",
        "p95_p05_range",
        "abs_d1_mean",
        "d1_sd",
        "abs_d2_mean",
        "global_slope",
        "early_mean",
        "mid_mean",
        "late_mean",
        "mid_minus_early",
        "late_minus_mid",
        "late_minus_early",
        "max_pos",
        "min_pos",
        "max_minus_min_pos",
    ]
    names.extend([f"{prefix}_{name}" for name in stat_names])
    return np.column_stack([coeff, stats_block]), names


def fixed_aec_features(aec: np.ndarray, sex: np.ndarray, sex_modified: bool = False) -> tuple[np.ndarray, list[str]]:
    a128, a128_names = curve_features(aec[:, :128], "a128")
    crop, crop_names = curve_features(aec[:, 128:], "crop")
    base = np.column_stack([a128, crop])
    names = a128_names + crop_names
    if not sex_modified:
        return base, names
    female = (np.asarray(sex).astype(str) == "F").astype(float)[:, None]
    inter = base * female
    return np.column_stack([base, inter]), names + [f"female_x_{name}" for name in names]


def standardize_train_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    xtr_s = (np.where(np.isfinite(xtr), xtr, mu) - mu) / sd
    xte_s = (np.where(np.isfinite(xte), xte, mu) - mu) / sd
    return xtr_s, xte_s, mu, sd


def fit_offset_ridge(x: np.ndarray, y: np.ndarray, offset: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    p = x.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = theta[0]
        beta = theta[1:]
        eta = offset + alpha + x @ beta
        loss = np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * lam * np.sum(beta**2)
        pr = sigmoid(eta)
        resid = pr - y
        grad_alpha = np.sum(resid)
        grad_beta = x.T @ resid + lam * beta
        return float(loss), np.r_[grad_alpha, grad_beta]

    theta0 = np.zeros(p + 1, dtype=float)
    opt = minimize(lambda th: objective(th), theta0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    theta = opt.x
    return float(theta[0]), theta[1:]


def choose_lambda_nested(
    x: np.ndarray,
    y: np.ndarray,
    clinical_x: np.ndarray,
    lambdas: list[float],
    n_splits: int = 4,
    seed: int = SEED,
) -> tuple[float, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for lam in lambdas:
        probs = np.zeros(len(y), dtype=float)
        scores = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(x, y):
            clinical = fit_clinical(clinical_x[tr_idx], y[tr_idx])
            c_tr = apply_clinical(clinical, clinical_x[tr_idx])
            c_va = apply_clinical(clinical, clinical_x[va_idx])
            xtr_s, xva_s, _, _ = standardize_train_apply(x[tr_idx], x[va_idx])
            alpha, beta = fit_offset_ridge(xtr_s, y[tr_idx], c_tr, lam)
            scores[va_idx] = c_va + alpha + xva_s @ beta
            probs[va_idx] = sigmoid(scores[va_idx])
        rows.append(
            {
                "lambda": lam,
                "cv_auc": float(roc_auc_score(y, scores)),
                "cv_average_precision": float(average_precision_score(y, scores)),
                "cv_log_loss": float(log_loss(y, probs)),
                "cv_brier": float(brier_score_loss(y, probs)),
            }
        )
    df = pd.DataFrame(rows)
    best = df.sort_values(["cv_log_loss", "cv_brier"], ascending=True).iloc[0]
    return float(best["lambda"]), df


def crossfit_offset_score(
    x: np.ndarray,
    y: np.ndarray,
    clinical_x: np.ndarray,
    lambdas: list[float],
    outer_splits: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    clinical_oof = np.zeros(len(y), dtype=float)
    aec_oof = np.zeros(len(y), dtype=float)
    combined_oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        best_lam, cv_df = choose_lambda_nested(x[tr_idx], y[tr_idx], clinical_x[tr_idx], lambdas, seed=seed + fold_id)
        clinical = fit_clinical(clinical_x[tr_idx], y[tr_idx])
        c_tr = apply_clinical(clinical, clinical_x[tr_idx])
        c_va = apply_clinical(clinical, clinical_x[va_idx])
        xtr_s, xva_s, _, _ = standardize_train_apply(x[tr_idx], x[va_idx])
        alpha, beta = fit_offset_ridge(xtr_s, y[tr_idx], c_tr, best_lam)
        aec_va = alpha + xva_s @ beta
        clinical_oof[va_idx] = c_va
        aec_oof[va_idx] = aec_va
        combined_oof[va_idx] = c_va + aec_va
        fold_rows.append(
            {
                "fold": fold_id,
                "selected_lambda": best_lam,
                "inner_best_cv_log_loss": float(cv_df.loc[cv_df["lambda"].eq(best_lam), "cv_log_loss"].iloc[0]),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
            }
        )
    return clinical_oof, aec_oof, combined_oof, pd.DataFrame(fold_rows)


def train_external_offset_score(
    xtr: np.ndarray,
    ytr: np.ndarray,
    clinical_xtr: np.ndarray,
    xte: np.ndarray,
    clinical_xte: np.ndarray,
    lambdas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    best_lam, cv_df = choose_lambda_nested(xtr, ytr, clinical_xtr, lambdas, seed=SEED + 200)
    clinical = fit_clinical(clinical_xtr, ytr)
    c_tr = apply_clinical(clinical, clinical_xtr)
    c_te = apply_clinical(clinical, clinical_xte)
    xtr_s, xte_s, _, _ = standardize_train_apply(xtr, xte)
    alpha, beta = fit_offset_ridge(xtr_s, ytr, c_tr, best_lam)
    aec_tr = alpha + xtr_s @ beta
    aec_te = alpha + xte_s @ beta
    return c_tr, c_te, aec_tr, aec_te, best_lam, cv_df


def lrt_score_test(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray) -> dict:
    c = (clinical_score - clinical_score.mean()) / (clinical_score.std() or 1.0)
    a = (aec_score - aec_score.mean()) / (aec_score.std() or 1.0)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    chi2 = 2 * (m1.llf - m0.llf)
    return {
        "chi2_1df": float(chi2),
        "p": float(stats.chi2.sf(chi2, 1)),
        "aec_beta_per_sd": float(m1.params[2]),
        "aec_or_per_sd": float(np.exp(m1.params[2])),
        "aec_wald_p": float(m1.pvalues[2]),
    }


def metric_row(model: str, y: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict:
    prob = sigmoid(score)
    row = {
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    if threshold is not None:
        row.update(binary_metrics(y, score, threshold))
    return row


def bootstrap_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 2500) -> dict:
    rng = np.random.default_rng(SEED + 500)
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        rows.append(
            [
                roc_auc_score(yy, score_b[idx]) - roc_auc_score(yy, score_a[idx]),
                average_precision_score(yy, score_b[idx]) - average_precision_score(yy, score_a[idx]),
                log_loss(yy, sigmoid(score_a[idx])) - log_loss(yy, sigmoid(score_b[idx])),
                brier_score_loss(yy, sigmoid(score_a[idx])) - brier_score_loss(yy, sigmoid(score_b[idx])),
            ]
        )
    arr = np.asarray(rows)
    out = {}
    for i, name in enumerate(["delta_auc", "delta_average_precision", "log_loss_reduction", "brier_reduction"]):
        vals = arr[:, i]
        out[name] = {
            "mean": float(vals.mean()),
            "ci2.5": float(np.quantile(vals, 0.025)),
            "ci97.5": float(np.quantile(vals, 0.975)),
            "p_le_0": float(np.mean(vals <= 0)),
        }
    return out


def run_model(train: dict, test: dict, sex_modified: bool) -> dict:
    label = "sex_modified_offset_aec" if sex_modified else "main_offset_aec"
    xtr, names = fixed_aec_features(train["aec"], train["sex"], sex_modified=sex_modified)
    xte, _ = fixed_aec_features(test["aec"], test["sex"], sex_modified=sex_modified)
    clinical_xtr = clinical_raw(train["meta"])
    clinical_xte = clinical_raw(test["meta"])
    ytr = train["y"]
    yte = test["y"]
    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    clinical_oof, aec_oof, combined_oof, fold_df = crossfit_offset_score(xtr, ytr, clinical_xtr, lambdas)
    c_tr, c_te, aec_tr, aec_te, best_lam, cv_df = train_external_offset_score(xtr, ytr, clinical_xtr, xte, clinical_xte, lambdas)
    combined_te = c_te + aec_te

    clinical_th = threshold_youden(ytr, clinical_oof)
    combined_th = threshold_youden(ytr, combined_oof)
    train_metrics = [
        metric_row("clinical_oof", ytr, clinical_oof, clinical_th),
        metric_row(f"{label}_combined_oof", ytr, combined_oof, combined_th),
        metric_row(f"{label}_aec_oof_only", ytr, aec_oof),
    ]
    external_metrics = [
        metric_row("clinical_external", yte, c_te, clinical_th),
        metric_row(f"{label}_combined_external", yte, combined_te, combined_th),
        metric_row(f"{label}_aec_external_only", yte, aec_te),
    ]

    fold_df.to_csv(OUT_DIR / f"{label}_outer_fold_lambdas.csv", index=False)
    cv_df.to_csv(OUT_DIR / f"{label}_final_lambda_cv.csv", index=False)
    pd.DataFrame(train_metrics).to_csv(OUT_DIR / f"{label}_train_metrics.csv", index=False)
    pd.DataFrame(external_metrics).to_csv(OUT_DIR / f"{label}_external_metrics.csv", index=False)

    return {
        "label": label,
        "sex_modified": sex_modified,
        "n_aec_features": len(names),
        "feature_names": names,
        "outer_fold_lambdas": fold_df.to_dict(orient="records"),
        "final_selected_lambda": best_lam,
        "train_metrics": train_metrics,
        "external_metrics": external_metrics,
        "train_lrt_aec_score_added_to_clinical": lrt_score_test(ytr, clinical_oof, aec_oof),
        "external_lrt_aec_score_added_to_clinical": lrt_score_test(yte, c_te, aec_te),
        "external_delta_combined_vs_clinical": bootstrap_delta(yte, c_te, combined_te),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    results = [run_model(train, test, sex_modified=False), run_model(train, test, sex_modified=True)]
    with open(OUT_DIR / "aec_offset_score_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Fixed low-dimensional AEC features with clinical-logit offset ridge logistic. No outcome-based window or landmark selection.",
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rows = []
    for r in results:
        for metric in r["external_metrics"]:
            rows.append({"label": r["label"], **metric})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "aec_offset_score_external_summary_table.csv", index=False)

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
