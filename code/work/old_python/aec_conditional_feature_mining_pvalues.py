from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_mass_feature_combinations import load_data  # noqa: E402
from aec_conditional_value import clinical_estimator, clinical_matrix, make_folds, oof_and_external, zfit_apply  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_conditional_feature_mining_pvalues"
SEED = 20260630


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg 절차로 p값 배열을 다중비교 보정한 q값(FDR)으로 변환."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m, dtype=float)
    out[order] = adj
    q[ok] = out
    return q


def impute_standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """train 중앙값으로 결측 대체, 1~99% 분위수로 이상치 클리핑, train 평균/표준편차로 표준화하고,
    중복된 컬럼명은 __dup 접미사를 붙여 구분되는 이름 목록도 함께 반환."""
    xtr = train.to_numpy(dtype=float)
    xte = test.to_numpy(dtype=float)
    med = np.nanmedian(xtr, axis=0)
    med[~np.isfinite(med)] = 0.0
    xtr = np.where(np.isfinite(xtr), xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    lo = np.nanquantile(xtr, 0.01, axis=0)
    hi = np.nanquantile(xtr, 0.99, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    xtr[:, ok] = np.clip(xtr[:, ok], lo[ok], hi[ok])
    xte[:, ok] = np.clip(xte[:, ok], lo[ok], hi[ok])
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    counts: dict[str, int] = {}
    names = []
    for col in train.columns:
        base = str(col)
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}__dup{counts[base]}")
    return (xtr - mu) / sd, (xte - mu) / sd, names


def clinical_scores(train: dict, test: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """임상 모델의 원시 OOF/외부 점수와, 그것을 z-표준화한 버전을 함께 계산."""
    xtr, xte, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(train["y"].astype(int), 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xtr, train["y"].astype(int), xte, folds)
    c_g, c_s, _, _ = zfit_apply(clinical_oof, clinical_ext)
    return clinical_oof, clinical_ext, c_g, c_s


def null_fit(y: np.ndarray, clinical_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """임상점수만 넣은 "귀무모델" 로지스틱을 적합해 예측확률과 설계행렬을 반환 (점수검정의 기준선)."""
    base = np.column_stack([np.ones(len(y)), clinical_z])
    model = sm.Logit(y, base).fit(disp=False, maxiter=1000)
    p = np.asarray(model.predict(base), dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p, base


def conditional_score_test_matrix(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    """수천 개 특징 전체에 대해 각각 개별 로지스틱을 적합하지 않고도, 귀무모델(임상점수만) 기준
    점수검정(score test)으로 카이제곱/p값/방향/부분상관을 한 번에 빠르게 계산 (대규모 특징 스캔용)."""
    y = y.astype(float)
    p, base = null_fit(y, clinical_z)
    resid = y - p
    w = p * (1.0 - p)
    bw = base * w[:, None]
    xtwx_inv = np.linalg.pinv(base.T @ bw)
    # Weighted projection of every feature on intercept + clinical score.
    beta = xtwx_inv @ (base.T @ (w[:, None] * x))
    x_res = x - base @ beta
    u = x_res.T @ resid
    info = np.sum(w[:, None] * x_res * x_res, axis=0)
    stat = np.where(info > 1e-12, (u * u) / info, np.nan)
    pval = stats.chi2.sf(stat, 1)
    direction = np.sign(u)
    # Weighted partial correlation is a useful effect-size-like companion.
    denom = np.sqrt(np.sum((resid**2)) * np.sum(x_res**2, axis=0))
    partial_r = np.where(denom > 1e-12, (x_res.T @ resid) / denom, np.nan)
    return pd.DataFrame(
        {
            "score_chi2": stat,
            "score_p": pval,
            "direction": direction,
            "partial_r": partial_r,
        }
    )


def full_logit_for_features(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, names: list[str], wanted: list[str]) -> pd.DataFrame:
    """지정된 특징들(wanted)만 골라, 점수검정 근사가 아닌 완전한 로지스틱 회귀(계수·오즈비·Wald p값·LRT)를 개별적으로 적합."""
    idx_map = {name: i for i, name in enumerate(names)}
    rows = []
    for name in wanted:
        if name not in idx_map:
            continue
        z = x[:, idx_map[name]]
        mat = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z, "feature_z": z}), has_constant="add")
        try:
            fit = sm.Logit(y, mat).fit(disp=False, maxiter=1000)
            null = sm.Logit(y, sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")).fit(disp=False, maxiter=1000)
            lrt = 2 * (fit.llf - null.llf)
            rows.append(
                {
                    "feature": name,
                    "coef": float(fit.params["feature_z"]),
                    "or_per_1sd": float(np.exp(fit.params["feature_z"])),
                    "wald_p": float(fit.pvalues["feature_z"]),
                    "lrt_chi2": float(lrt),
                    "lrt_p": float(stats.chi2.sf(lrt, 1)),
                }
            )
        except Exception:
            rows.append({"feature": name, "coef": np.nan, "or_per_1sd": np.nan, "wald_p": np.nan, "lrt_chi2": np.nan, "lrt_p": np.nan})
    return pd.DataFrame(rows)


def feature_family(name: str) -> str:
    """특징 이름의 접두사/키워드를 보고 레벨/대비비율/기울기·거칠기/곡률/Haar엣지/스펙트럼/자기상관/위치기하/위상학/기타 중 어느 계열인지 분류."""
    if name.startswith("norm_level") or name.startswith("log_level") or name.startswith("level_") or name.startswith("log_"):
        return "level"
    if "contrast" in name or "ratio" in name or "rebound" in name:
        return "contrast_ratio"
    if "slope" in name or "d1" in name:
        return "slope_roughness"
    if "curv" in name or "d2" in name:
        return "curvature"
    if "haar" in name:
        return "haar_edge"
    if "dct" in name or "fft" in name or "spectral" in name:
        return "spectral"
    if "autocorr" in name:
        return "autocorr"
    if "arg" in name or "position" in name or "distance" in name:
        return "geometry_position"
    if "run" in name or "flat" in name or "cross" in name:
        return "topology"
    return "other"


def per_threshold_positive_association(
    y: np.ndarray,
    clinical_z: np.ndarray,
    feature_z: np.ndarray,
    threshold_z: float,
) -> dict:
    """임상점수가 임계값 이상인 부분집합(임상 양성군)에서만, 한 특징의 계수·오즈비·p값을 로지스틱 회귀로 계산 (표본 20 미만이면 NaN)."""
    mask = clinical_z >= threshold_z
    yy = y[mask].astype(int)
    zz = feature_z[mask]
    if mask.sum() < 20 or np.unique(yy).size < 2:
        return {"n": int(mask.sum()), "coef": np.nan, "or_per_1sd": np.nan, "p": np.nan}
    mat = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z[mask], "feature_z": zz}), has_constant="add")
    try:
        fit = sm.Logit(yy, mat).fit(disp=False, maxiter=1000)
        return {
            "n": int(mask.sum()),
            "coef": float(fit.params["feature_z"]),
            "or_per_1sd": float(np.exp(fit.params["feature_z"])),
            "p": float(fit.pvalues["feature_z"]),
        }
    except Exception:
        return {"n": int(mask.sum()), "coef": np.nan, "or_per_1sd": np.nan, "p": np.nan}


def plot_top_feature(train: dict, test: dict, xg: np.ndarray, xs: np.ndarray, names: list[str], feature: str) -> None:
    """한 특징에 대해 코호트x저근감소증여부 4개 그룹의 값 분포를 박스플롯으로 그려 PNG로 저장."""
    idx = names.index(feature)
    rows = []
    for cohort, y, z in [("g1090", train["y"].astype(int), xg[:, idx]), ("sdata", test["y"].astype(int), xs[:, idx])]:
        rows.append(pd.DataFrame({"cohort": cohort, "low_smi": y.astype(bool), "feature_z": z}))
    df = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    positions = []
    data = []
    labels = []
    pos = 1
    for cohort in ["g1090", "sdata"]:
        for low in [False, True]:
            sub = df[(df["cohort"].eq(cohort)) & (df["low_smi"].eq(low))]["feature_z"].to_numpy()
            data.append(sub)
            positions.append(pos)
            labels.append(f"{cohort}\n{'low' if low else 'non-low'}")
            pos += 1
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    colors = ["#8FB4DC", "#D9735F", "#8FB4DC", "#D9735F"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Feature z-score")
    ax.set_title(feature, loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in feature)[:140]
    fig.savefig(OUT_DIR / f"top_feature_{safe}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec128_mass_feature_combinations가 만든 3125개짜리
    초대형 특징 은행 전체에 대해, 다중비교(FDR)를 제대로 보정하면서 "임상변수를 통제한 뒤에도
    유의한" 특징이 정말 남아있는가? — 통계적으로 가장 엄격한 대량 특징 스캔):

    1. aec128_mass_feature_combinations에서 데이터와 3125개 특징 은행을 그대로 불러오고, 임상
       점수를 표준화해 준비.
    2. conditional_score_test_matrix로 (개별 로지스틱을 3125번 적합하는 대신) 점수검정 근사로
       모든 특징의 카이제곱/p값/방향/부분상관을 g1090과 sdata 각각에서 빠르게 계산.
    3. bh_fdr로 각 코호트 내에서 Benjamini-Hochberg FDR 보정을 적용하고, 두 코호트 p값을 결합한
       Fisher 방법으로 combined p값도 계산.
    4. "두 코호트 방향이 같고, g1090 FDR<=10%, sdata p<=5%"인 "train에서 발견되고 외부에서 검증된"
       특징들과, "sdata FDR<=10%"인 "외부 탐색적" 특징들을 각각 골라 저장.
    5. 특징을 레벨/대비비율/기울기/곡률/Haar/스펙트럼/자기상관/기하/위상학 계열로 묶어 계열별 요약도 계산.
    6. 상위 30개 특징에 대해서는 점수검정 근사가 아닌 완전한 로지스틱 회귀(LRT)로 재확인.
    7. 상위 20개 특징에 대해, 임상 80/85/90/95% 민감도 임계값별로 "임상 양성군 내에서만"의 연관성도
       추가 검정하고, 모든 특징에 대한 임상양성군 내 스캔도 별도로 수행해 그 안에서도 강건한 특징을 추출.
    8. 상위 3개 특징의 코호트x라벨별 분포를 박스플롯으로 그려 저장.
    9. 특징 수, FDR 정책, 상위 발견 특징들을 JSON으로 저장하고, 계열 요약·상위 발견·임상양성군
       검증 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, ftr, fte = load_data()
    y_g = train["y"].astype(int)
    y_s = test["y"].astype(int)
    clinical_oof_raw, _clinical_ext_raw, c_g, c_s = clinical_scores(train, test)
    x_g, x_s, names = impute_standardize(ftr, fte)

    g_stats = conditional_score_test_matrix(y_g, c_g, x_g)
    s_stats = conditional_score_test_matrix(y_s, c_s, x_s)
    out = pd.DataFrame({"feature": names})
    for prefix, stats_df in [("g1090", g_stats), ("sdata", s_stats)]:
        out[f"{prefix}_score_chi2"] = stats_df["score_chi2"].to_numpy()
        out[f"{prefix}_score_p"] = stats_df["score_p"].to_numpy()
        out[f"{prefix}_direction"] = stats_df["direction"].to_numpy()
        out[f"{prefix}_partial_r"] = stats_df["partial_r"].to_numpy()
        out[f"{prefix}_score_q"] = bh_fdr(out[f"{prefix}_score_p"].to_numpy())
    out["same_direction"] = np.sign(out["g1090_direction"]) == np.sign(out["sdata_direction"])
    out["family"] = out["feature"].map(feature_family)
    out["min_neglog10_p"] = np.minimum(-np.log10(out["g1090_score_p"]), -np.log10(out["sdata_score_p"]))
    out["combined_fisher_chi2"] = -2 * (np.log(out["g1090_score_p"]) + np.log(out["sdata_score_p"]))
    out["combined_fisher_p"] = stats.chi2.sf(out["combined_fisher_chi2"], 4)
    out["combined_fisher_q"] = bh_fdr(out["combined_fisher_p"].to_numpy())
    out = out.sort_values(["same_direction", "combined_fisher_p"], ascending=[False, True])
    out.to_csv(OUT_DIR / "conditional_feature_score_tests_all.csv", index=False)

    train_discovered = out[
        (out["same_direction"])
        & (out["g1090_score_q"] <= 0.10)
        & (out["sdata_score_p"] <= 0.05)
    ].sort_values(["sdata_score_p", "g1090_score_q"])
    train_discovered.to_csv(OUT_DIR / "train_discovered_external_validated_features.csv", index=False)

    external_exploratory = out[
        (out["same_direction"])
        & (out["sdata_score_q"] <= 0.10)
    ].sort_values(["sdata_score_q", "g1090_score_p"])
    external_exploratory.to_csv(OUT_DIR / "external_exploratory_fdr_features.csv", index=False)

    family = (
        out.groupby("family")
        .agg(
            n_features=("feature", "size"),
            n_same_direction=("same_direction", "sum"),
            best_g1090_p=("g1090_score_p", "min"),
            best_sdata_p=("sdata_score_p", "min"),
            best_combined_p=("combined_fisher_p", "min"),
            n_train_q10_external_p05=("feature", lambda s: int(train_discovered["feature"].isin(s).sum())),
            n_external_q10=("feature", lambda s: int(external_exploratory["feature"].isin(s).sum())),
        )
        .reset_index()
        .sort_values(["n_train_q10_external_p05", "best_combined_p"], ascending=[False, True])
    )
    family.to_csv(OUT_DIR / "feature_family_summary.csv", index=False)

    top_features = (
        train_discovered["feature"].head(30).tolist()
        if not train_discovered.empty
        else out[out["same_direction"]]["feature"].head(30).tolist()
    )
    top_full_g = full_logit_for_features(y_g, c_g, x_g, names, top_features)
    top_full_s = full_logit_for_features(y_s, c_s, x_s, names, top_features)
    top_full = top_full_g.merge(top_full_s, on="feature", suffixes=("_g1090", "_sdata"))
    top_full["wald_p_g1090_q_within_top"] = bh_fdr(top_full["wald_p_g1090"].to_numpy())
    top_full["wald_p_sdata_q_within_top"] = bh_fdr(top_full["wald_p_sdata"].to_numpy())
    top_full.to_csv(OUT_DIR / "top_features_full_logit_pvalues.csv", index=False)

    thresholds_raw = {
        "sens80": threshold_for_min_sensitivity(y_g, clinical_oof_raw, 0.80),
        "sens85": threshold_for_min_sensitivity(y_g, clinical_oof_raw, 0.85),
        "sens90": threshold_for_min_sensitivity(y_g, clinical_oof_raw, 0.90),
        "sens95": threshold_for_min_sensitivity(y_g, clinical_oof_raw, 0.95),
    }
    c_mu = float(np.mean(clinical_oof_raw))
    c_sd = float(np.std(clinical_oof_raw)) or 1.0
    thresholds_z = {k: float((v - c_mu) / c_sd) for k, v in thresholds_raw.items()}
    cp_rows = []
    for feature in top_features[:20]:
        idx = names.index(feature)
        for threshold_name, threshold_z in thresholds_z.items():
            for dataset, y, clinical, xmat in [
                ("g1090", y_g, c_g, x_g),
                ("sdata", y_s, c_s, x_s),
            ]:
                r = per_threshold_positive_association(y, clinical, xmat[:, idx], threshold_z)
                cp_rows.append({"feature": feature, "dataset": dataset, "threshold": threshold_name, **r})
    cp = pd.DataFrame(cp_rows)
    cp["p_q_within_table"] = bh_fdr(cp["p"].to_numpy())
    cp.to_csv(OUT_DIR / "top_features_clinical_positive_assoc_by_threshold.csv", index=False)

    cp_scan_rows = []
    for threshold_name, threshold_z in thresholds_z.items():
        for dataset, y, clinical, xmat in [
            ("g1090", y_g, c_g, x_g),
            ("sdata", y_s, c_s, x_s),
        ]:
            mask = clinical >= threshold_z
            if np.unique(y[mask]).size < 2:
                continue
            st = conditional_score_test_matrix(y[mask], clinical[mask], xmat[mask, :])
            tmp = pd.DataFrame({"feature": names})
            tmp["dataset"] = dataset
            tmp["threshold"] = threshold_name
            tmp["n_clinical_positive"] = int(mask.sum())
            tmp["score_p"] = st["score_p"].to_numpy()
            tmp["score_q"] = bh_fdr(tmp["score_p"].to_numpy())
            tmp["direction"] = st["direction"].to_numpy()
            tmp["partial_r"] = st["partial_r"].to_numpy()
            tmp["family"] = tmp["feature"].map(feature_family)
            cp_scan_rows.append(tmp)
    cp_scan = pd.concat(cp_scan_rows, ignore_index=True)
    cp_scan.to_csv(OUT_DIR / "clinical_positive_all_feature_score_tests.csv", index=False)

    cp_valid_rows = []
    for threshold_name in thresholds_z:
        gtab = cp_scan[(cp_scan["dataset"].eq("g1090")) & (cp_scan["threshold"].eq(threshold_name))].copy()
        stab = cp_scan[(cp_scan["dataset"].eq("sdata")) & (cp_scan["threshold"].eq(threshold_name))].copy()
        merged = gtab.merge(stab, on="feature", suffixes=("_g1090", "_sdata"))
        merged["same_direction"] = np.sign(merged["direction_g1090"]) == np.sign(merged["direction_sdata"])
        valid = merged[
            (merged["same_direction"])
            & (merged["score_q_g1090"] <= 0.10)
            & (merged["score_p_sdata"] <= 0.05)
        ].copy()
        valid["threshold"] = threshold_name
        cp_valid_rows.append(valid)
    cp_valid = pd.concat(cp_valid_rows, ignore_index=True) if cp_valid_rows else pd.DataFrame()
    cp_valid.to_csv(OUT_DIR / "clinical_positive_train_discovered_external_validated_features.csv", index=False)

    for feature in top_features[:3]:
        plot_top_feature(train, test, x_g, x_s, names, feature)

    summary = {
        "n_features_tested": int(len(names)),
        "feature_policy": "AEC128 normalized mass feature bank; p-values are conditional score tests for y ~ clinical_score + feature_z.",
        "fdr_policy": "Benjamini-Hochberg FDR within the 3125 tested features for each cohort.",
        "n_train_discovered_external_validated": int(len(train_discovered)),
        "n_external_exploratory_fdr_features": int(len(external_exploratory)),
        "top_train_discovered_external_validated": train_discovered.head(20).to_dict(orient="records"),
        "top_external_exploratory": external_exploratory.head(20).to_dict(orient="records"),
        "outputs": {
            "all_score_tests": str(OUT_DIR / "conditional_feature_score_tests_all.csv"),
            "train_discovered_external_validated": str(OUT_DIR / "train_discovered_external_validated_features.csv"),
            "external_exploratory": str(OUT_DIR / "external_exploratory_fdr_features.csv"),
            "full_logit_top": str(OUT_DIR / "top_features_full_logit_pvalues.csv"),
            "clinical_positive_top": str(OUT_DIR / "top_features_clinical_positive_assoc_by_threshold.csv"),
            "clinical_positive_all_features": str(OUT_DIR / "clinical_positive_all_feature_score_tests.csv"),
            "clinical_positive_validated": str(OUT_DIR / "clinical_positive_train_discovered_external_validated_features.csv"),
            "family_summary": str(OUT_DIR / "feature_family_summary.csv"),
        },
    }
    (OUT_DIR / "conditional_feature_mining_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFeature family summary")
    print(family.to_string(index=False))
    print("\nTrain-discovered external-validated top")
    print(train_discovered.head(25).to_string(index=False) if not train_discovered.empty else "None")
    print("\nExternal exploratory FDR top")
    print(external_exploratory.head(25).to_string(index=False) if not external_exploratory.empty else "None")
    print("\nTop full logit")
    print(top_full.to_string(index=False) if not top_full.empty else "None")
    print("\nClinical-positive train-discovered external-validated")
    if cp_valid.empty:
        print("None")
    else:
        show_cols = [
            "threshold",
            "feature",
            "family_g1090",
            "score_p_g1090",
            "score_q_g1090",
            "score_p_sdata",
            "score_q_sdata",
            "same_direction",
        ]
        print(cp_valid.sort_values(["threshold", "score_p_sdata"])[show_cols].head(30).to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
