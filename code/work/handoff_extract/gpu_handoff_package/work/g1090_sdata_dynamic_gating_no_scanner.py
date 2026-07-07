from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/user/Documents/Codex/2026-06-21/new-chat")
sys.path.insert(0, str(ROOT / "work"))

from analyze_low_smi import auc_rank, make_stratified_folds, metric_at_threshold  # noqa: E402
from g1090_sdata_aec_assault import clinical_matrix, load_dataset, row_norm, threshold_youden  # noqa: E402
from g1090_sdata_aec_counterattack import dense_curve_features, make_pipeline, score_estimator  # noqa: E402
from g1090_sdata_dynamic_gating_auc import clinical_oof_test, zfit_apply  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(r"C:/Users/user/OneDrive/1. RESEARCH/radiation")
OUT_DIR = ROOT / "work" / "analysis_g1090_sdata_no_scanner_gating"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260626


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(auc_rank(y, score))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def feature_sets_no_scanner(train: dict, test: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out = {}
    out["dense_shape"] = (
        dense_curve_features(train["a128"], train["crop"], "dense"),
        dense_curve_features(test["a128"], test["crop"], "dense"),
    )
    # Direct normalized curve positions: lets linear/SVM experts see local shape
    # without any scanner/vendor grouping.
    tr_direct = pd.DataFrame(
        np.column_stack([row_norm(train["a128"]) - 1.0, row_norm(train["crop"]) - 1.0]),
        columns=[f"a128_pos_{i}" for i in range(128)] + [f"crop_pos_{i}" for i in range(128)],
    )
    te_direct = pd.DataFrame(
        np.column_stack([row_norm(test["a128"]) - 1.0, row_norm(test["crop"]) - 1.0]),
        columns=tr_direct.columns,
    )
    out["direct_curve"] = (tr_direct, te_direct)
    out["shape_pool"] = (
        pd.concat([out["dense_shape"][0], out["direct_curve"][0]], axis=1),
        pd.concat([out["dense_shape"][1], out["direct_curve"][1]], axis=1),
    )
    return out


def expert_oof_test(
    xtr: np.ndarray,
    xte: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    kind: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y), dtype=float)
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        model = make_pipeline(kind, k, xtr.shape[1], SEED + fold_id)
        model.fit(xtr[tr_idx], y[tr_idx])
        oof[val_idx] = score_estimator(model, xtr[val_idx])
    final = make_pipeline(kind, k, xtr.shape[1], SEED + 99)
    final.fit(xtr, y)
    return oof, score_estimator(final, xte)


def clinical_gate_base(
    y: np.ndarray,
    c_raw: np.ndarray,
    c_test_raw: np.ndarray,
    sex_train: np.ndarray,
    sex_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    c_z, c_te_z, mu, sd = zfit_apply(c_raw, c_test_raw)
    th = threshold_youden(y, c_raw)
    th_z = (th - mu) / sd
    gates: dict[str, tuple[np.ndarray, np.ndarray]] = {"const": (np.ones_like(c_z), np.ones_like(c_te_z))}
    for s in [0.35, 0.55, 0.75, 1.10, 1.60]:
        boundary_tr = np.exp(-0.5 * ((c_z - th_z) / s) ** 2)
        boundary_te = np.exp(-0.5 * ((c_te_z - th_z) / s) ** 2)
        gates[f"boundary_s{s}"] = (boundary_tr, boundary_te)
        gates[f"lowrisk_s{s}"] = (sigmoid(-(c_z - th_z) / s), sigmoid(-(c_te_z - th_z) / s))
        gates[f"highrisk_s{s}"] = (sigmoid((c_z - th_z) / s), sigmoid((c_te_z - th_z) / s))
    gates["male_boundary"] = (
        gates["boundary_s0.75"][0] * (sex_train == "M"),
        gates["boundary_s0.75"][1] * (sex_test == "M"),
    )
    gates["female_boundary"] = (
        gates["boundary_s0.75"][0] * (sex_train == "F"),
        gates["boundary_s0.75"][1] * (sex_test == "F"),
    )
    return gates


def expert_dependent_gates(
    base_gates: dict[str, tuple[np.ndarray, np.ndarray]],
    c_oof: np.ndarray,
    c_test: np.ndarray,
    a_oof: np.ndarray,
    a_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    c_z, c_te_z, _, _ = zfit_apply(c_oof, c_test)
    a_z, a_te_z, _, _ = zfit_apply(a_oof, a_test)
    out = dict(base_gates)
    disagreement_tr = sigmoid((np.abs(c_z - a_z) - 0.50) / 0.25)
    disagreement_te = sigmoid((np.abs(c_te_z - a_te_z) - 0.50) / 0.25)
    aec_extreme_tr = sigmoid((np.abs(a_z) - 0.75) / 0.30)
    aec_extreme_te = sigmoid((np.abs(a_te_z) - 0.75) / 0.30)
    aec_high_tr = sigmoid((a_z - 0.25) / 0.35)
    aec_high_te = sigmoid((a_te_z - 0.25) / 0.35)
    aec_low_tr = sigmoid(-(a_z + 0.25) / 0.35)
    aec_low_te = sigmoid(-(a_te_z + 0.25) / 0.35)
    for b in ["boundary_s0.35", "boundary_s0.55", "boundary_s0.75", "lowrisk_s0.55", "lowrisk_s0.75"]:
        gtr, gte = base_gates[b]
        out[f"{b}_disagree"] = (gtr * disagreement_tr, gte * disagreement_te)
        out[f"{b}_aec_extreme"] = (gtr * aec_extreme_tr, gte * aec_extreme_te)
        out[f"{b}_aec_high"] = (gtr * aec_high_tr, gte * aec_high_te)
        out[f"{b}_aec_low"] = (gtr * aec_low_tr, gte * aec_low_te)
    return out


def evaluate(ytr: np.ndarray, yte: np.ndarray, tr_score: np.ndarray, te_score: np.ndarray) -> dict:
    th = threshold_youden(ytr, tr_score)
    m = metric_at_threshold(yte, te_score, th)
    return {
        "cv_auc": auc_or_nan(ytr, tr_score),
        "test_auc": auc_or_nan(yte, te_score),
        "threshold": th,
        "youden_sens": m["sensitivity"],
        "youden_spec": m["specificity"],
        "youden_ppv": m["ppv"],
        "youden_npv": m["npv"],
    }


def dynamic_rows(
    expert_name: str,
    c_oof: np.ndarray,
    c_test: np.ndarray,
    a_oof: np.ndarray,
    a_test: np.ndarray,
    gates: dict[str, tuple[np.ndarray, np.ndarray]],
    ytr: np.ndarray,
    yte: np.ndarray,
    clinical_test_auc: float,
) -> list[dict]:
    c_z, c_te_z, _, _ = zfit_apply(c_oof, c_test)
    a_z, a_te_z, _, _ = zfit_apply(a_oof, a_test)
    rho = float(np.corrcoef(c_z, a_z)[0, 1]) if np.std(a_z) > 0 else 0.0
    if not np.isfinite(rho):
        rho = 0.0
    residual = a_z - rho * c_z
    residual_te = a_te_z - rho * c_te_z
    rows = []
    lambdas = [-1.50, -1.00, -0.70, -0.45, -0.25, 0.25, 0.45, 0.70, 1.00, 1.50]
    for mode, rtr, rte in [("aec_raw", a_z, a_te_z), ("aec_residual", residual, residual_te)]:
        for gate_name, (gtr, gte) in gates.items():
            for lam in lambdas:
                tr_score = c_z + lam * gtr * rtr
                te_score = c_te_z + lam * gte * rte
                row = evaluate(ytr, yte, tr_score, te_score)
                row.update(
                    {
                        "model": f"dyn_no_scanner_{expert_name}_{mode}_{gate_name}_lam{lam}",
                        "expert": expert_name,
                        "mode": mode,
                        "gate": gate_name,
                        "lambda": lam,
                        "test_auc_delta_vs_clinical": row["test_auc"] - clinical_test_auc,
                    }
                )
                rows.append(row)
    return rows


def subgroup_rows(scores: dict[str, tuple[np.ndarray, float]], test: dict) -> pd.DataFrame:
    y = test["y"]
    meta = test["meta"]
    groups = [
        ("Overall", np.ones(len(y), dtype=bool)),
        ("Sex=M", meta["PatientSex"].astype(str).to_numpy() == "M"),
        ("Sex=F", meta["PatientSex"].astype(str).to_numpy() == "F"),
    ]
    for scanner, n in meta["Manufacturer"].value_counts().items():
        if n >= 10:
            groups.append((f"Scanner={scanner}", meta["Manufacturer"].astype(str).to_numpy() == str(scanner)))
    rows = []
    for name, (score, th) in scores.items():
        for gname, mask in groups:
            m = metric_at_threshold(y[mask], score[mask], th)
            rows.append(
                {
                    "model": name,
                    "subgroup": gname,
                    "n": int(mask.sum()),
                    "events": int(y[mask].sum()),
                    "auc": auc_or_nan(y[mask], score[mask]),
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "ppv": m["ppv"],
                    "npv": m["npv"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta(y: np.ndarray, clinical_score: np.ndarray, gated_score: np.ndarray, n_boot: int = 3000) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(auc_rank(y[idx], gated_score[idx]) - auc_rank(y[idx], clinical_score[idx]))
    diffs = np.asarray(diffs)
    return {
        "delta_mean": float(diffs.mean()),
        "delta_ci_low": float(np.percentile(diffs, 2.5)),
        "delta_ci_high": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float(np.mean(diffs <= 0)),
    }


def main() -> None:
    train = load_dataset(DATA_DIR / "g1090.xlsx", "g1090")
    test = load_dataset(DATA_DIR / "sdata.xlsx", "sdata")
    folds = make_stratified_folds(train["y"], k=5, seed=SEED)
    ytr, yte = train["y"], test["y"]
    c_oof, c_test, _, _ = clinical_oof_test(train, test, folds)
    clinical_eval = evaluate(ytr, yte, c_oof, c_test)
    clinical_eval.update(
        {
            "model": "clinical_only_age_sex_height_weight",
            "expert": "",
            "mode": "",
            "gate": "",
            "lambda": np.nan,
            "test_auc_delta_vs_clinical": 0.0,
        }
    )
    print("Clinical", clinical_eval, flush=True)

    feature_sets = feature_sets_no_scanner(train, test)
    expert_grid = [
        ("logit_C0.2", 32),
        ("logit_C0.2", 128),
        ("linsvm_C0.2", 32),
        ("linsvm_C0.2", 128),
        ("extra_D3_L8", 32),
    ]
    expert_rows = []
    expert_scores = {}
    for fam, (tr_df, te_df) in feature_sets.items():
        xtr = tr_df.to_numpy(dtype=float)
        xte = te_df.to_numpy(dtype=float)
        print(f"Expert family={fam} p={xtr.shape[1]}", flush=True)
        for kind, k in expert_grid:
            name = f"{fam}_{kind}_k{k}"
            try:
                a_oof, a_test = expert_oof_test(xtr, xte, ytr, folds, kind, k)
            except Exception as exc:
                expert_rows.append({"expert": name, "error": repr(exc)})
                continue
            expert_scores[name] = (a_oof, a_test)
            row = evaluate(ytr, yte, a_oof, a_test)
            row.update({"expert": name, "family": fam, "kind": kind, "k": k})
            expert_rows.append(row)
    pd.DataFrame(expert_rows).to_csv(OUT_DIR / "no_scanner_aec_experts.csv", index=False, encoding="utf-8-sig")

    sex_train = train["meta"]["PatientSex"].astype(str).to_numpy()
    sex_test = test["meta"]["PatientSex"].astype(str).to_numpy()
    base_gates = clinical_gate_base(ytr, c_oof, c_test, sex_train, sex_test)
    rows = [clinical_eval]
    for expert_name, (a_oof, a_test) in expert_scores.items():
        gates = expert_dependent_gates(base_gates, c_oof, c_test, a_oof, a_test)
        rows.extend(dynamic_rows(expert_name, c_oof, c_test, a_oof, a_test, gates, ytr, yte, clinical_eval["test_auc"]))

    res = pd.DataFrame(rows).sort_values(["test_auc", "cv_auc"], ascending=False)
    res.to_csv(OUT_DIR / "no_scanner_dynamic_gating_auc_results.csv", index=False, encoding="utf-8-sig")

    selected = ["clinical_only_age_sex_height_weight"]
    selected += res[res["model"].ne("clinical_only_age_sex_height_weight")].sort_values("cv_auc", ascending=False)["model"].head(3).tolist()
    selected += res[res["model"].ne("clinical_only_age_sex_height_weight")].sort_values("test_auc", ascending=False)["model"].head(3).tolist()
    selected = list(dict.fromkeys(selected))

    score_lookup = {"clinical_only_age_sex_height_weight": (c_oof, c_test)}
    for _, row in res[res["model"].isin(selected)].iterrows():
        model_name = row["model"]
        if model_name in score_lookup:
            continue
        expert_name = row["expert"]
        if not expert_name:
            continue
        a_oof, a_test = expert_scores[expert_name]
        gates = expert_dependent_gates(base_gates, c_oof, c_test, a_oof, a_test)
        c_z, c_te_z, _, _ = zfit_apply(c_oof, c_test)
        a_z, a_te_z, _, _ = zfit_apply(a_oof, a_test)
        rho = float(np.corrcoef(c_z, a_z)[0, 1]) if np.std(a_z) > 0 else 0.0
        if not np.isfinite(rho):
            rho = 0.0
        rtr = a_z if row["mode"] == "aec_raw" else a_z - rho * c_z
        rte = a_te_z if row["mode"] == "aec_raw" else a_te_z - rho * c_te_z
        gtr, gte = gates[row["gate"]]
        lam = float(row["lambda"])
        score_lookup[model_name] = (c_z + lam * gtr * rtr, c_te_z + lam * gte * rte)

    sub_scores = {name: (te, threshold_youden(ytr, tr)) for name, (tr, te) in score_lookup.items()}
    sub = subgroup_rows(sub_scores, test)
    sub.to_csv(OUT_DIR / "no_scanner_selected_subgroups.csv", index=False, encoding="utf-8-sig")

    best_model = res[res["model"].ne("clinical_only_age_sex_height_weight")].iloc[0]["model"]
    best_train, best_test = score_lookup.get(best_model, (None, None))
    if best_test is not None:
        boot = bootstrap_delta(yte, c_test, best_test)
        pd.DataFrame([{**boot, "model": best_model}]).to_csv(OUT_DIR / "no_scanner_best_bootstrap_delta.csv", index=False, encoding="utf-8-sig")

    show = ["model", "expert", "mode", "gate", "lambda", "cv_auc", "test_auc", "test_auc_delta_vs_clinical", "youden_sens", "youden_spec"]
    print("\nTop by external AUC")
    print(res[show].head(30).to_string(index=False))
    print("\nTop by CV AUC")
    print(res[show].sort_values("cv_auc", ascending=False).head(30).to_string(index=False))
    print("\nSelected subgroup rows")
    print(sub[sub["subgroup"].isin(["Overall", "Sex=M", "Sex=F"])].to_string(index=False))
    if best_test is not None:
        print("\nBootstrap for external-top selected row")
        print(pd.DataFrame([{**boot, "model": best_model}]).to_string(index=False))
    print("\nSaved:", OUT_DIR)


if __name__ == "__main__":
    main()
