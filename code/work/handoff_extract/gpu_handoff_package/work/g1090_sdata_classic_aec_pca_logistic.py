from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_classic_aec_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import clinical_matrix, load_dataset, threshold_youden  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import feature_sets_no_scanner  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 20260626


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def score_model(model: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(model.named_steps["clf"], "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def oof_test_pipeline(xtr: np.ndarray, y: np.ndarray, xte: np.ndarray, model_factory, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_scores = []
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        model = model_factory(SEED + fold_id)
        model.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = score_model(model, xtr[val_idx])
        test_scores.append(score_model(model, xte))
    final = model_factory(SEED + 99)
    final.fit(xtr, y)
    return oof, score_model(final, xte)


def eval_score(name: str, ytr: np.ndarray, yte: np.ndarray, tr_score: np.ndarray, te_score: np.ndarray) -> dict:
    th = threshold_youden(ytr, tr_score)
    m = metric_at_threshold(yte, te_score, th)
    return {
        "model": name,
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


def group_row(name: str, y: np.ndarray, mask: np.ndarray, extra: dict | None = None) -> dict:
    n = int(mask.sum())
    events = int(y[mask].sum())
    out = {"group": name, "n": n, "events": events, "prevalence": events / n if n else np.nan}
    if extra:
        out.update(extra)
    return out


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)

    clin_tr, clin_te, clin_names = clinical_matrix(train["meta"], test["meta"])
    direct_tr, direct_te = feature_sets_no_scanner(train, test)["direct_curve"]
    aec_tr = direct_tr.to_numpy(dtype=float)
    aec_te = direct_te.to_numpy(dtype=float)

    def logit_factory(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=4000, random_state=seed)),
            ]
        )

    rows = []
    c_oof, c_test = oof_test_pipeline(clin_tr, ytr, clin_te, logit_factory, folds)
    rows.append(eval_score("classic_clinical_logistic_age_sex_height_weight", ytr, yte, c_oof, c_test))

    for n_pc in [3, 5, 8, 10, 16, 24]:
        def aec_pca_factory(seed: int, n_pc=n_pc) -> Pipeline:
            return Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=n_pc, random_state=seed)),
                    ("clf", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=4000, random_state=seed)),
                ]
            )

        a_oof, a_test = oof_test_pipeline(aec_tr, ytr, aec_te, aec_pca_factory, folds)
        rows.append(eval_score(f"classic_aec_pca{n_pc}_logistic", ytr, yte, a_oof, a_test))

        xtr_combo = np.column_stack([clin_tr, aec_tr])
        xte_combo = np.column_stack([clin_te, aec_te])

        def combo_factory(seed: int, n_pc=n_pc) -> Pipeline:
            # PCA is applied to all standardized columns here. This is intentionally simple,
            # not a preferred clinical model; included only to test a fully classic approach.
            return Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("pca", PCA(n_components=n_pc + 4, random_state=seed)),
                    ("clf", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=4000, random_state=seed)),
                ]
            )

        combo_oof, combo_test = oof_test_pipeline(xtr_combo, ytr, xte_combo, combo_factory, folds)
        rows.append(eval_score(f"classic_clinical_plus_aec_pca{n_pc}_logistic", ytr, yte, combo_oof, combo_test))

        # Classic two-stage analysis: within clinical-positive patients, divide by AEC PCA score tertiles
        # using train-derived tertile cutpoints.
        if n_pc == 5:
            clin_th = threshold_youden(ytr, c_oof)
            cp_tr = c_oof >= clin_th
            cp_te = c_test >= clin_th
            q1, q2 = np.quantile(a_oof[cp_tr], [1 / 3, 2 / 3])
            groups = [
                group_row("clinical_negative", yte, ~cp_te),
                group_row("clinical_positive_aec_pca5_low_tertile", yte, cp_te & (a_test <= q1), {"q1": q1, "q2": q2}),
                group_row("clinical_positive_aec_pca5_middle_tertile", yte, cp_te & (a_test > q1) & (a_test <= q2), {"q1": q1, "q2": q2}),
                group_row("clinical_positive_aec_pca5_high_tertile", yte, cp_te & (a_test > q2), {"q1": q1, "q2": q2}),
            ]
            pd.DataFrame(groups).to_csv(OUT_DIR / "classic_two_stage_clinical_positive_aec_pca5_tertiles.csv", index=False, encoding="utf-8-sig")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "classic_pca_logistic_model_comparison.csv", index=False, encoding="utf-8-sig")
    print(res.sort_values("test_auc", ascending=False).to_string(index=False))
    p = OUT_DIR / "classic_two_stage_clinical_positive_aec_pca5_tertiles.csv"
    if p.exists():
        print("\nTwo-stage AEC PCA5 tertiles among clinical-positive")
        print(pd.read_csv(p).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
