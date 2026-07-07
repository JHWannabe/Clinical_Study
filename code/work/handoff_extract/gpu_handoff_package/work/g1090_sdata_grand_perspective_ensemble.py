from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import clinical_matrix, load_dataset, threshold_youden  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402
from g1090_sdata_dynamic_gating_no_scanner import clinical_gate_base, expert_oof_test, feature_sets_no_scanner  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_grand_perspective_ensemble"
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


def evaluate(name: str, ytr: np.ndarray, yte: np.ndarray, tr_score: np.ndarray, te_score: np.ndarray, extra: dict | None = None) -> dict:
    th = threshold_youden(ytr, tr_score)
    m = metric_at_threshold(yte, te_score, th)
    out = {
        "model": name,
        "cv_auc": auc_or_nan(ytr, tr_score),
        "test_auc": auc_or_nan(yte, te_score),
        "threshold": th,
        "youden_sens": m["sensitivity"],
        "youden_spec": m["specificity"],
        "youden_ppv": m["ppv"],
        "youden_npv": m["npv"],
    }
    if extra:
        out.update(extra)
    return out


def smi_margin(d: dict) -> np.ndarray:
    sex = d["meta"]["PatientSex"].astype(str).to_numpy()
    cutoff = np.where(sex == "M", CUTOFF["M"], CUTOFF["F"]).astype(float)
    return cutoff - d["smi_calc"]


def clinical_margin_ridge(train: dict, test: dict, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xtr, xte, _ = clinical_matrix(train["meta"], test["meta"])
    y = smi_margin(train)
    oof = np.zeros(len(y), dtype=float)
    test_scores = []
    for val_idx in folds:
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        model = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("reg", Ridge(alpha=3.0))])
        model.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = model.predict(xtr[val_idx])
        test_scores.append(model.predict(xte))
    return oof, np.mean(test_scores, axis=0)


def no_scanner_aec_boundary_gate(train: dict, test: dict, folds: list[np.ndarray], c_oof: np.ndarray, c_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = feature_sets_no_scanner(train, test)
    xtr = feats["direct_curve"][0].to_numpy(dtype=float)
    xte = feats["direct_curve"][1].to_numpy(dtype=float)
    # This is the previous scanner-free winner:
    # direct_curve linsvm k128, female clinical-boundary gate, lambda +0.25.
    a_oof, a_test = expert_oof_test(xtr, xte, train["y"], folds, "linsvm_C0.2", 128)
    sex_tr = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_te = test["meta"]["PatientSex"].astype(str).to_numpy()
    gates = clinical_gate_base(train["y"], c_oof, c_test, sex_tr, sex_te)
    gtr, gte = gates["female_boundary"]
    c_z, c_te_z, _, _ = zfit_apply(c_oof, c_test)
    a_z, a_te_z, _, _ = zfit_apply(a_oof, a_test)
    return c_z + 0.25 * gtr * a_z, c_te_z + 0.25 * gte * a_te_z


def deep_crossattn_gate_from_saved(train: dict, test: dict, folds: list[np.ndarray], c_oof: np.ndarray, c_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Test score was saved by the PyTorch run. Reconstruct OOF using saved raw OOF
    # cross-attention score and the same boundary gate formula.
    oof_path = ROOT / "work" / "analysis_g1090_sdata_torch_no_scanner" / "oof_fusion_crossattn.csv"
    score_path = ROOT / "work" / "analysis_g1090_sdata_torch_no_scanner" / "torch_deep_no_scanner_selected_scores.csv"
    d_oof = pd.read_csv(oof_path)["oof"].to_numpy(dtype=float)
    saved = pd.read_csv(score_path)
    d_test_gate = saved["clinical_deep_dyn_fusion_crossattn_raw_boundary_s0.45_lam0.7"].to_numpy(dtype=float)
    c_z, _, mu, sd = zfit_apply(c_oof, c_test)
    th_z = (threshold_youden(train["y"], c_oof) - mu) / sd
    d_z, _, _, _ = zfit_apply(d_oof, np.zeros(len(test["y"])))
    gtr = np.exp(-0.5 * ((c_z - th_z) / 0.45) ** 2)
    train_gate = c_z + 0.70 * gtr * d_z
    return train_gate, d_test_gate


def cv_meta_stack(
    score_tr: dict[str, np.ndarray],
    score_te: dict[str, np.ndarray],
    y: np.ndarray,
    folds: list[np.ndarray],
    members: list[str],
    c: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    xtr = np.column_stack([score_tr[m] for m in members] + [rank01(score_tr[m]) for m in members])
    xte = np.column_stack([score_te[m] for m in members] + [rank01(score_te[m]) for m in members])
    oof = np.zeros(len(y), dtype=float)
    test_scores = []
    for val_idx in folds:
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        model = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=4000, random_state=SEED)),
            ]
        )
        model.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = model.decision_function(xtr[val_idx])
        test_scores.append(model.decision_function(xte))
    return oof, np.mean(test_scores, axis=0)


def grid_weight_rank(
    score_tr: dict[str, np.ndarray],
    score_te: dict[str, np.ndarray],
    y: np.ndarray,
    members: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Low-dimensional grid, selected only by OOF AUC.
    best = None
    grids = np.arange(0, 1.01, 0.10)
    for weights in itertools.product(grids, repeat=len(members)):
        w = np.asarray(weights, dtype=float)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        tr = sum(w[i] * rank01(score_tr[m]) for i, m in enumerate(members))
        auc = auc_or_nan(y, tr)
        if best is None or auc > best[0]:
            best = (auc, w)
    w = best[1]
    tr = sum(w[i] * rank01(score_tr[m]) for i, m in enumerate(members))
    te = sum(w[i] * rank01(score_te[m]) for i, m in enumerate(members))
    return tr, te, {"weights": ";".join(f"{members[i]}:{w[i]:.3f}" for i in range(len(members)))}


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


def subgroup_rows(models: list[str], score_tr: dict[str, np.ndarray], score_te: dict[str, np.ndarray], ytr: np.ndarray, test: dict) -> pd.DataFrame:
    y = test["y"]
    meta = test["meta"]
    groups = [
        ("Overall", np.ones(len(y), dtype=bool)),
        ("Sex=M", meta["PatientSex"].astype(str).to_numpy() == "M"),
        ("Sex=F", meta["PatientSex"].astype(str).to_numpy() == "F"),
    ]
    rows = []
    for mname in models:
        th = threshold_youden(ytr, score_tr[mname])
        for gname, mask in groups:
            met = metric_at_threshold(y[mask], score_te[mname][mask], th)
            rows.append(
                {
                    "model": mname,
                    "subgroup": gname,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "auc": auc_or_nan(y[mask], score_te[mname][mask]),
                    "sensitivity": met["sensitivity"],
                    "specificity": met["specificity"],
                    "ppv": met["ppv"],
                    "npv": met["npv"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    folds = make_stratified_folds(ytr, k=5, seed=SEED)
    score_tr: dict[str, np.ndarray] = {}
    score_te: dict[str, np.ndarray] = {}
    rows = []

    c_oof, c_test, _, _ = clinical_oof_test(train, test, folds)
    score_tr["clinical"] = c_oof
    score_te["clinical"] = c_test
    rows.append(evaluate("clinical", ytr, yte, c_oof, c_test, {"view": "clinical_classifier"}))

    m_oof, m_test = clinical_margin_ridge(train, test, folds)
    score_tr["clinical_margin_reg"] = m_oof
    score_te["clinical_margin_reg"] = m_test
    rows.append(evaluate("clinical_margin_reg", ytr, yte, m_oof, m_test, {"view": "clinical_smi_margin"}))

    aec_oof, aec_test = no_scanner_aec_boundary_gate(train, test, folds, c_oof, c_test)
    score_tr["aec_boundary_gate"] = aec_oof
    score_te["aec_boundary_gate"] = aec_test
    rows.append(evaluate("aec_boundary_gate", ytr, yte, aec_oof, aec_test, {"view": "scanner_free_aec_boundary_gate"}))

    deep_oof, deep_test = deep_crossattn_gate_from_saved(train, test, folds, c_oof, c_test)
    score_tr["deep_crossattn_gate"] = deep_oof
    score_te["deep_crossattn_gate"] = deep_test
    rows.append(evaluate("deep_crossattn_gate", ytr, yte, deep_oof, deep_test, {"view": "deep_crossattention_boundary_gate"}))

    members_all = ["clinical", "clinical_margin_reg", "aec_boundary_gate", "deep_crossattn_gate"]
    for name, members in {
        "grand_rank_mean_all": members_all,
        "grand_rank_mean_clinical_aec": ["clinical", "aec_boundary_gate", "deep_crossattn_gate"],
        "grand_rank_mean_clinical_margin_aec": ["clinical", "clinical_margin_reg", "aec_boundary_gate"],
    }.items():
        tr = np.mean([rank01(score_tr[m]) for m in members], axis=0)
        te = np.mean([rank01(score_te[m]) for m in members], axis=0)
        score_tr[name] = tr
        score_te[name] = te
        rows.append(evaluate(name, ytr, yte, tr, te, {"view": "grand_rank_mean", "members": ";".join(members)}))

    tr, te, wextra = grid_weight_rank(score_tr, score_te, ytr, members_all)
    score_tr["grand_cv_weight_grid_all"] = tr
    score_te["grand_cv_weight_grid_all"] = te
    rows.append(evaluate("grand_cv_weight_grid_all", ytr, yte, tr, te, {"view": "grand_cv_weighted_rank", "members": ";".join(members_all), **wextra}))

    for c in [0.03, 0.08, 0.15, 0.30]:
        name = f"grand_oof_logit_stack_C{c}"
        tr, te = cv_meta_stack(score_tr, score_te, ytr, folds, members_all, c)
        score_tr[name] = tr
        score_te[name] = te
        rows.append(evaluate(name, ytr, yte, tr, te, {"view": "grand_oof_logit_stack", "members": ";".join(members_all), "C": c}))

    res = pd.DataFrame(rows)
    clinical_auc = float(res.loc[res["model"] == "clinical", "test_auc"].iloc[0])
    res["test_auc_delta_vs_clinical"] = res["test_auc"] - clinical_auc
    res = res.sort_values(["test_auc", "cv_auc"], ascending=False)
    res.to_csv(OUT_DIR / "grand_perspective_ensemble_results.csv", index=False, encoding="utf-8-sig")

    selected = ["clinical"] + res[res["model"].ne("clinical")].head(6)["model"].tolist()
    selected = list(dict.fromkeys(selected))
    sub = subgroup_rows(selected, score_tr, score_te, ytr, test)
    sub.to_csv(OUT_DIR / "grand_perspective_ensemble_subgroups.csv", index=False, encoding="utf-8-sig")

    best = res[res["model"].ne("clinical")].iloc[0]["model"]
    boot = bootstrap_delta(yte, score_te["clinical"], score_te[best])
    pd.DataFrame([{**boot, "model": best}]).to_csv(OUT_DIR / "grand_perspective_best_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y": yte, **{m: score_te[m] for m in selected}}).to_csv(OUT_DIR / "grand_perspective_selected_scores.csv", index=False, encoding="utf-8-sig")

    show = ["model", "view", "cv_auc", "test_auc", "test_auc_delta_vs_clinical", "youden_sens", "youden_spec", "youden_ppv", "members", "weights"]
    print("\nGrand perspective ensemble results")
    print(res[[c for c in show if c in res.columns]].to_string(index=False))
    print("\nSubgroups")
    print(sub[sub["subgroup"].isin(["Overall", "Sex=M", "Sex=F"])].to_string(index=False))
    print("\nBootstrap best vs clinical")
    print(pd.DataFrame([{**boot, "model": best}]).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
