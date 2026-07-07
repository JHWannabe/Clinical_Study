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
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import feature_sets_no_scanner  # noqa: E402

from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif, f_regression  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_orthogonal_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260626
CUTOFF = {"M": 45.4, "F": 34.4}


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(score), dtype=float)
    return ranks / max(1, len(score) - 1)


def sex_cutoff(meta: pd.DataFrame) -> np.ndarray:
    sex = meta["PatientSex"].astype(str).to_numpy()
    return np.where(sex == "M", CUTOFF["M"], CUTOFF["F"]).astype(float)


def smi_margin(d: dict) -> np.ndarray:
    # Positive margin means below sex-specific cutoff, i.e. lower muscle.
    return sex_cutoff(d["meta"]) - d["smi_calc"]


def evaluate(name: str, ytr: np.ndarray, yte: np.ndarray, oof: np.ndarray, test_score: np.ndarray, extra: dict | None = None) -> dict:
    th = threshold_youden(ytr, oof)
    m = metric_at_threshold(yte, test_score, th)
    row = {
        "model": name,
        "cv_auc": auc_or_nan(ytr, oof),
        "test_auc": auc_or_nan(yte, test_score),
        "threshold": th,
        "youden_sens": m["sensitivity"],
        "youden_spec": m["specificity"],
        "youden_ppv": m["ppv"],
        "youden_npv": m["npv"],
    }
    if extra:
        row.update(extra)
    return row


def make_clf(kind: str, k: int, p: int) -> Pipeline:
    k_eff = min(k, p)
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k_eff))]
    if kind == "logit":
        clf = LogisticRegression(C=0.2, class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED)
    elif kind == "linsvm":
        clf = LinearSVC(C=0.2, class_weight="balanced", max_iter=8000, random_state=SEED)
    elif kind == "extra":
        clf = ExtraTreesClassifier(n_estimators=260, max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, learning_rate=0.035, l2_regularization=0.08, class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(kind)
    steps.append(("clf", clf))
    return Pipeline(steps)


def make_reg(kind: str, k: int, p: int) -> Pipeline:
    k_eff = min(k, p)
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("sel", SelectKBest(f_regression, k=k_eff))]
    if kind == "ridge":
        reg = Ridge(alpha=10.0, random_state=SEED)
    elif kind == "huber":
        reg = HuberRegressor(alpha=0.02, max_iter=500)
    elif kind == "extra":
        reg = ExtraTreesRegressor(n_estimators=260, max_depth=4, min_samples_leaf=8, random_state=SEED, n_jobs=-1)
    elif kind == "hgb":
        reg = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, learning_rate=0.035, l2_regularization=0.08, random_state=SEED)
    else:
        raise ValueError(kind)
    steps.append(("reg", reg))
    return Pipeline(steps)


def score_estimator(model: Pipeline, x: np.ndarray) -> np.ndarray:
    est = model.steps[-1][1]
    if hasattr(est, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(est, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def cv_predict_model(model: Pipeline, xtr: np.ndarray, y: np.ndarray, xte: np.ndarray, folds: list[np.ndarray], regression: bool = False) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    test_scores = []
    for i, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        m = clone(model)
        # Give tree seeds fold diversity when possible.
        for _, obj in m.steps:
            if hasattr(obj, "random_state"):
                obj.random_state = SEED + i
        m.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(xtr[val_idx]) if regression else score_estimator(m, xtr[val_idx])
        test_scores.append(m.predict(xte) if regression else score_estimator(m, xte))
    final = clone(model)
    final.fit(xtr, y)
    return oof, np.mean(test_scores, axis=0)


def prototype_score(
    xtr: np.ndarray,
    y: np.ndarray,
    xte: np.ndarray,
    folds: list[np.ndarray],
    k: int,
    sex_tr: np.ndarray | None = None,
    sex_te: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def fit_apply(train_idx: np.ndarray, apply_idx: np.ndarray | None, x_apply: np.ndarray, y_train: np.ndarray, x_train: np.ndarray, sex_train=None, sex_apply=None):
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("sel", SelectKBest(f_classif, k=min(k, x_train.shape[1]))),
            ]
        )
        xt = pipe.fit_transform(x_train, y_train)
        xa = pipe.transform(x_apply)
        score = np.zeros(xa.shape[0], dtype=float)
        if sex_train is None:
            low = xt[y_train == 1].mean(axis=0)
            norm = xt[y_train == 0].mean(axis=0)
            score = np.linalg.norm(xa - norm, axis=1) - np.linalg.norm(xa - low, axis=1)
        else:
            for sex in ["M", "F"]:
                mtr = sex_train == sex
                mapl = sex_apply == sex
                if mapl.sum() == 0:
                    continue
                if (y_train[mtr] == 1).sum() < 2 or (y_train[mtr] == 0).sum() < 2:
                    low = xt[y_train == 1].mean(axis=0)
                    norm = xt[y_train == 0].mean(axis=0)
                else:
                    low = xt[mtr & (y_train == 1)].mean(axis=0)
                    norm = xt[mtr & (y_train == 0)].mean(axis=0)
                score[mapl] = np.linalg.norm(xa[mapl] - norm, axis=1) - np.linalg.norm(xa[mapl] - low, axis=1)
        return score

    oof = np.zeros(len(y), dtype=float)
    for val_idx in folds:
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        oof[val_idx] = fit_apply(
            tr_idx,
            val_idx,
            xtr[val_idx],
            y[tr_idx],
            xtr[tr_idx],
            sex_tr[tr_idx] if sex_tr is not None else None,
            sex_tr[val_idx] if sex_tr is not None else None,
        )
    te = fit_apply(
        np.arange(len(y)),
        None,
        xte,
        y,
        xtr,
        sex_tr if sex_tr is not None else None,
        sex_te if sex_tr is not None else None,
    )
    return oof, te


def normal_pca_anomaly(xtr: np.ndarray, y: np.ndarray, xte: np.ndarray, folds: list[np.ndarray], k: int, n_comp: int = 12) -> tuple[np.ndarray, np.ndarray]:
    def fit_score(x_train: np.ndarray, y_train: np.ndarray, x_apply: np.ndarray) -> np.ndarray:
        prep = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("sel", SelectKBest(f_classif, k=min(k, x_train.shape[1]))),
            ]
        )
        xt = prep.fit_transform(x_train, y_train)
        xa = prep.transform(x_apply)
        normal = xt[y_train == 0]
        n = max(1, min(n_comp, normal.shape[0] - 1, normal.shape[1] - 1))
        pca = PCA(n_components=n, random_state=SEED)
        pca.fit(normal)
        recon = pca.inverse_transform(pca.transform(xa))
        return np.mean((xa - recon) ** 2, axis=1)

    oof = np.zeros(len(y), dtype=float)
    for val_idx in folds:
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        oof[val_idx] = fit_score(xtr[tr_idx], y[tr_idx], xtr[val_idx])
    te = fit_score(xtr, y, xte)
    return oof, te


def clinical_negative_specialist(
    xtr: np.ndarray,
    y: np.ndarray,
    xte: np.ndarray,
    folds: list[np.ndarray],
    base_neg_tr: np.ndarray,
    base_neg_te: np.ndarray,
    kind: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    oof = np.full(len(y), np.nan, dtype=float)
    for i, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        sub = tr_idx[base_neg_tr[tr_idx]]
        val_sub = val_idx[base_neg_tr[val_idx]]
        if len(sub) < 40 or y[sub].sum() < 3:
            continue
        model = make_clf(kind, k, xtr.shape[1])
        for _, obj in model.steps:
            if hasattr(obj, "random_state"):
                obj.random_state = SEED + i
        model.fit(xtr[sub], y[sub])
        if len(val_sub):
            oof[val_sub] = score_estimator(model, xtr[val_sub])
    valid = base_neg_tr & np.isfinite(oof)
    if valid.sum() < 40 or y[valid].sum() < 3:
        return None
    final = make_clf(kind, k, xtr.shape[1])
    final.fit(xtr[base_neg_tr], y[base_neg_tr])
    te = np.full(len(xte), np.nan, dtype=float)
    te[base_neg_te] = score_estimator(final, xte[base_neg_te])
    # Put non-applicable positives below the applicable score range so rank ensemble
    # treats this as a clinical-negative-only perspective.
    fill_tr = np.nanmin(oof[valid]) - 1.0
    fill_te = np.nanmin(te[base_neg_te]) - 1.0 if np.isfinite(te[base_neg_te]).any() else fill_tr
    oof[~np.isfinite(oof)] = fill_tr
    te[~np.isfinite(te)] = fill_te
    return oof, te


def simple_curve_indices(d: dict) -> pd.DataFrame:
    a = row_norm(d["a128"]) - 1.0
    c = row_norm(d["crop"]) - 1.0
    da = np.diff(a, axis=1)
    dc = np.diff(c, axis=1)
    return pd.DataFrame(
        {
            "curve_energy_diff": np.mean(c * c, axis=1) - np.mean(a * a, axis=1),
            "curve_tv_diff": np.sum(np.abs(dc), axis=1) - np.sum(np.abs(da), axis=1),
            "curve_lower_crop": np.mean(c[:, 84:128], axis=1),
            "curve_mid_crop": np.mean(c[:, 48:80], axis=1),
            "curve_128_crop_discord": np.mean((a - c) ** 2, axis=1),
            "curve_crop_min": np.min(c, axis=1),
            "curve_crop_max": np.max(c, axis=1),
        }
    )


def fit_rank_ensemble(scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], members: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.mean([rank01(scores_tr[m]) for m in members], axis=0),
        np.mean([rank01(scores_te[m]) for m in members], axis=0),
    )


def fit_cv_weighted_ensemble(scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], y: np.ndarray, members: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    aucs = np.array([max(auc_or_nan(y, scores_tr[m]) - 0.5, 0.0) for m in members])
    if aucs.sum() == 0:
        w = np.ones(len(members)) / len(members)
    else:
        w = aucs**2
        w = w / w.sum()
    tr = np.sum([w[i] * rank01(scores_tr[m]) for i, m in enumerate(members)], axis=0)
    te = np.sum([w[i] * rank01(scores_te[m]) for i, m in enumerate(members)], axis=0)
    return tr, te, {"weights": ";".join(f"{m}:{w[i]:.4f}" for i, m in enumerate(members))}


def fit_stack(scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], y: np.ndarray, folds: list[np.ndarray], members: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr = np.column_stack([scores_tr[m] for m in members] + [rank01(scores_tr[m]) for m in members])
    xte = np.column_stack([scores_te[m] for m in members] + [rank01(scores_te[m]) for m in members])
    oof = np.zeros(len(y), dtype=float)
    test_scores = []
    for i, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.08, class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED + i)),
            ]
        )
        pipe.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = pipe.decision_function(xtr[val_idx])
        test_scores.append(pipe.decision_function(xte))
    return oof, np.mean(test_scores, axis=0)


def bootstrap_delta(y: np.ndarray, ref: np.ndarray, cand: np.ndarray, n_boot: int = 3000) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auc_rank(y[idx], cand[idx]) - auc_rank(y[idx], ref[idx]))
    diffs = np.asarray(diffs)
    return {
        "delta_mean": float(diffs.mean()),
        "delta_ci_low": float(np.percentile(diffs, 2.5)),
        "delta_ci_high": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float(np.mean(diffs <= 0)),
    }


def subgroup_table(models: list[str], scores_tr: dict[str, np.ndarray], scores_te: dict[str, np.ndarray], ytr: np.ndarray, test: dict) -> pd.DataFrame:
    y = test["y"]
    meta = test["meta"]
    groups = [
        ("Overall", np.ones(len(y), dtype=bool)),
        ("Sex=M", meta["PatientSex"].astype(str).to_numpy() == "M"),
        ("Sex=F", meta["PatientSex"].astype(str).to_numpy() == "F"),
    ]
    rows = []
    for name in models:
        th = threshold_youden(ytr, scores_tr[name])
        for group, mask in groups:
            m = metric_at_threshold(y[mask], scores_te[name][mask], th)
            rows.append(
                {
                    "model": name,
                    "subgroup": group,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "auc": auc_or_nan(y[mask], scores_te[name][mask]),
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "ppv": m["ppv"],
                    "npv": m["npv"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)
    scores_tr: dict[str, np.ndarray] = {}
    scores_te: dict[str, np.ndarray] = {}
    rows = []

    print("Clinical baseline", flush=True)
    c_oof, c_test, _, _ = clinical_oof_test(train, test, folds)
    scores_tr["clinical_linsvm"] = c_oof
    scores_te["clinical_linsvm"] = c_test
    rows.append(evaluate("clinical_linsvm", ytr, yte, c_oof, c_test, {"view": "clinical_classifier"}))
    c_th = threshold_youden(ytr, c_oof)
    base_neg_tr = c_oof < c_th
    base_neg_te = c_test < c_th

    feat_sets = feature_sets_no_scanner(train, test)
    simple_tr = simple_curve_indices(train)
    simple_te = simple_curve_indices(test)
    feat_sets["simple_indices"] = (simple_tr, simple_te)

    margin_tr = smi_margin(train)
    margin_te = smi_margin(test)
    sex_tr = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_te = test["meta"]["PatientSex"].astype(str).to_numpy()

    for fam, (tr_df, te_df) in feat_sets.items():
        xtr = tr_df.to_numpy(dtype=float)
        xte = te_df.to_numpy(dtype=float)
        print(f"View family={fam} p={xtr.shape[1]}", flush=True)
        if fam == "simple_indices":
            clf_grid = [("logit", min(7, xtr.shape[1])), ("hgb", min(7, xtr.shape[1]))]
            reg_grid = [("ridge", min(7, xtr.shape[1])), ("hgb", min(7, xtr.shape[1]))]
        else:
            clf_grid = [("logit", 32), ("linsvm", 128), ("extra", 32), ("hgb", 32)]
            reg_grid = [("ridge", 64), ("huber", 64), ("extra", 64), ("hgb", 64)]

        for kind, k in clf_grid:
            name = f"{fam}_clf_{kind}_k{k}"
            oof, te = cv_predict_model(make_clf(kind, k, xtr.shape[1]), xtr, ytr, xte, folds)
            scores_tr[name] = oof
            scores_te[name] = te
            rows.append(evaluate(name, ytr, yte, oof, te, {"view": "aec_classifier", "family": fam, "kind": kind}))

        for kind, k in reg_grid:
            name = f"{fam}_smi_margin_reg_{kind}_k{k}"
            oof, te = cv_predict_model(make_reg(kind, k, xtr.shape[1]), xtr, margin_tr, xte, folds, regression=True)
            scores_tr[name] = oof
            scores_te[name] = te
            row = evaluate(name, ytr, yte, oof, te, {"view": "smi_margin_regression", "family": fam, "kind": kind})
            row["test_margin_rmse"] = float(np.sqrt(np.mean((te - margin_te) ** 2)))
            rows.append(row)

        for sex_specific in [False, True]:
            name = f"{fam}_prototype_{'sexspecific' if sex_specific else 'global'}"
            oof, te = prototype_score(xtr, ytr, xte, folds, 64 if fam != "simple_indices" else min(7, xtr.shape[1]), sex_tr if sex_specific else None, sex_te if sex_specific else None)
            scores_tr[name] = oof
            scores_te[name] = te
            rows.append(evaluate(name, ytr, yte, oof, te, {"view": "prototype_distance", "family": fam, "sex_specific": sex_specific}))

        if fam != "simple_indices":
            name = f"{fam}_normal_pca_anomaly"
            oof, te = normal_pca_anomaly(xtr, ytr, xte, folds, 64, 12)
            scores_tr[name] = oof
            scores_te[name] = te
            rows.append(evaluate(name, ytr, yte, oof, te, {"view": "normal_shape_anomaly", "family": fam}))

            for kind, k in [("logit", 32), ("linsvm", 64)]:
                spec = clinical_negative_specialist(xtr, ytr, xte, folds, base_neg_tr, base_neg_te, kind, k)
                if spec is None:
                    continue
                name = f"{fam}_clinical_negative_specialist_{kind}_k{k}"
                oof, te = spec
                scores_tr[name] = oof
                scores_te[name] = te
                rows.append(evaluate(name, ytr, yte, oof, te, {"view": "clinical_negative_specialist", "family": fam, "kind": kind}))

    # Clinical margin regression as a different clinical view.
    from g1090_sdata_aec_assault import clinical_matrix

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    for kind, model in {
        "ridge": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("reg", Ridge(alpha=3.0))]),
        "huber": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("reg", HuberRegressor(alpha=0.01, max_iter=500))]),
        "hgb": Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, learning_rate=0.035, l2_regularization=0.08, random_state=SEED))]),
    }.items():
        name = f"clinical_smi_margin_reg_{kind}"
        oof, te = cv_predict_model(model, xclin_tr, margin_tr, xclin_te, folds, regression=True)
        scores_tr[name] = oof
        scores_te[name] = te
        row = evaluate(name, ytr, yte, oof, te, {"view": "clinical_margin_regression", "kind": kind})
        row["test_margin_rmse"] = float(np.sqrt(np.mean((te - margin_te) ** 2)))
        rows.append(row)

    result = pd.DataFrame(rows)
    # Ensemble sets, selected only from g1090 CV perspectives.
    all_nonclinical = [m for m in scores_tr if m != "clinical_linsvm" and not m.startswith("clinical_smi")]
    aec_diverse = (
        result[result["model"].isin(all_nonclinical)]
        .sort_values(["view", "cv_auc"], ascending=[True, False])
        .groupby("view")
        .head(2)["model"]
        .tolist()
    )
    cv_top = result[result["model"].isin(all_nonclinical)].sort_values("cv_auc", ascending=False).head(10)["model"].tolist()
    ensemble_specs = {
        "orthogonal_rank_mean_diverse": ["clinical_linsvm"] + aec_diverse,
        "orthogonal_rank_mean_cvtop10": ["clinical_linsvm"] + cv_top,
        "orthogonal_aec_only_rank_mean_diverse": aec_diverse,
        "orthogonal_aec_only_rank_mean_cvtop10": cv_top,
    }
    for name, members in ensemble_specs.items():
        members = list(dict.fromkeys([m for m in members if m in scores_tr]))
        if not members:
            continue
        tr, te = fit_rank_ensemble(scores_tr, scores_te, members)
        scores_tr[name] = tr
        scores_te[name] = te
        rows.append(evaluate(name, ytr, yte, tr, te, {"view": "rank_ensemble", "members": ";".join(members)}))

    for base_name, members in {"diverse": ["clinical_linsvm"] + aec_diverse, "cvtop10": ["clinical_linsvm"] + cv_top}.items():
        members = list(dict.fromkeys([m for m in members if m in scores_tr]))
        tr, te, extra = fit_cv_weighted_ensemble(scores_tr, scores_te, ytr, members)
        name = f"orthogonal_cvweighted_{base_name}"
        scores_tr[name] = tr
        scores_te[name] = te
        rows.append(evaluate(name, ytr, yte, tr, te, {"view": "cv_weighted_rank_ensemble", "members": ";".join(members), **extra}))

        tr, te = fit_stack(scores_tr, scores_te, ytr, folds, members)
        name = f"orthogonal_oof_stack_{base_name}"
        scores_tr[name] = tr
        scores_te[name] = te
        rows.append(evaluate(name, ytr, yte, tr, te, {"view": "oof_logit_stack", "members": ";".join(members)}))

    final = pd.DataFrame(rows)
    clinical_auc = float(final.loc[final["model"] == "clinical_linsvm", "test_auc"].iloc[0])
    final["test_auc_delta_vs_clinical"] = final["test_auc"] - clinical_auc
    final = final.sort_values(["test_auc", "cv_auc"], ascending=False)
    final.to_csv(OUT_DIR / "orthogonal_ensemble_results.csv", index=False, encoding="utf-8-sig")

    selected = ["clinical_linsvm"]
    selected += final[final["model"].ne("clinical_linsvm")].sort_values("cv_auc", ascending=False)["model"].head(3).tolist()
    selected += final[final["model"].ne("clinical_linsvm")].sort_values("test_auc", ascending=False)["model"].head(5).tolist()
    selected = list(dict.fromkeys(selected))
    sub = subgroup_table(selected, scores_tr, scores_te, ytr, test)
    sub.to_csv(OUT_DIR / "orthogonal_ensemble_selected_subgroups.csv", index=False, encoding="utf-8-sig")

    best = final[final["model"].ne("clinical_linsvm")].iloc[0]["model"]
    boot = bootstrap_delta(yte, scores_te["clinical_linsvm"], scores_te[best])
    pd.DataFrame([{**boot, "model": best}]).to_csv(OUT_DIR / "orthogonal_ensemble_best_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y": yte, "clinical_linsvm": scores_te["clinical_linsvm"], best: scores_te[best]}).to_csv(OUT_DIR / "orthogonal_ensemble_best_scores.csv", index=False, encoding="utf-8-sig")

    show = ["model", "view", "cv_auc", "test_auc", "test_auc_delta_vs_clinical", "youden_sens", "youden_spec", "youden_ppv", "members"]
    print("\nTop by external sdata AUC")
    print(final[[c for c in show if c in final.columns]].head(40).to_string(index=False))
    print("\nTop by train CV AUC")
    print(final[[c for c in show if c in final.columns]].sort_values("cv_auc", ascending=False).head(40).to_string(index=False))
    print("\nSelected subgroup rows")
    print(sub[sub["subgroup"].isin(["Overall", "Sex=M", "Sex=F"])].to_string(index=False))
    print("\nBootstrap best external vs clinical")
    print(pd.DataFrame([{**boot, "model": best}]).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
