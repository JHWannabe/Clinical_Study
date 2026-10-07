from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# anchor 체성분 "비율" feature로 AUC/OR 비교 (Model1 = clinic4 + liver~pubis VAT 합 위에 추가). 비율은 전부 log((a+1)/(b+1)).
#  - anchor간: 같은 조직의 anchor / L3 (11 anchor x 5조직, VAT@pubis 제외 = 54개) -> 체격과 무관한 상대 분포
#  - anchor내 조직 비율: VAT/SAT, TAMA/(TAMA+VAT+SAT) (근육 점유율)를 anchor마다 (VAT/SAT@pubis 제외). 2026-10-02 데이터셋에서 NAMA/LAMA/IMATA가 TAMA로 통합되어 근육 질 비율은 불가
# (1) 미리 정한 묶음 모델 (2) 후보 55개 중 trainval 안 |z| 순위 top-k 스윕(hold-out) (3) 비율 하나씩 OR(+1SD, 두 코호트, BH-FDR).
# 모든 모델은 GridSearchCV(PARAM_GRID, valid 단일 fold)로 튜닝. 로딩/행렬/hold-out/DeLong은 clinic4_landmark_vat_auc, 스윕 보조함수는 anchor_k_sweep 재사용.

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from clinic4_landmark_vat_auc import holdout_fit, holdout_split, run_disease_holdout
from clinic4_landmark_anchor_k_sweep import N_FIXED
from clinic4_landmark_level_auc import DEAD
from clinic4_landmark_vat_auc import LM, TISSUES, delong_vs, load_cohort, matrix
from clinic4_logistic_regression import CLINIC4_DIR, DISEASES, PROJECT_ROOT, SEED, balanced_idx, f3, format_floats, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

OUT_XLSX = "landmark/landmark_anchor_ratio_auc.xlsx"
FIG = PROJECT_ROOT / "outputs" / "clinic4" / "landmark" / "anchor_ratio_k_curve.png"
REF = "L3_center"
KS = list(range(1, 11)) + [12, 15, 20, 25, 30, 40, 50, 55]  # 풀 크기 = anchor/L3 32 + 조직 비율 23 = 55


def lr(a, b):
    return np.log((a + 1) / (b + 1))


# 비율 컬럼 추가. 반환: df, {그룹명: [컬럼]}
def add_ratios(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    new, groups = {}, {}
    for t in TISSUES:
        cols = []
        for a in LM:
            if a == REF or (t, a) == DEAD:
                continue
            new[f"{t}@{a}/L3"] = lr(df[f"{t}@{a}"], df[f"{t}@{REF}"])
            cols.append(f"{t}@{a}/L3")
        groups[f"{t} anchor/L3"] = cols
    vs, ts = [], []
    for a in LM:
        if ("VAT", a) != DEAD:
            new[f"VAT/SAT@{a}"] = lr(df[f"VAT@{a}"], df[f"SAT@{a}"])
            vs.append(f"VAT/SAT@{a}")
        new[f"TAMA/(TAMA+VAT+SAT)@{a}"] = lr(df[f"TAMA@{a}"], df[f"VAT@{a}"] + df[f"SAT@{a}"])
        ts.append(f"TAMA/(TAMA+VAT+SAT)@{a}")
    groups |= {"VAT/SAT @anchors": vs, "TAMA/(TAMA+fat) @anchors": ts}
    return pd.concat([df, pd.DataFrame(new)], axis=1), groups


def model_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    anchor_all = sum((groups[f"{t} anchor/L3"] for t in TISSUES), [])
    within_all = groups["VAT/SAT @anchors"] + groups["TAMA/(TAMA+fat) @anchors"]
    sets = {"baseline": [], "Model1": []}
    sets |= {f"+{g}": c for g, c in groups.items()}
    sets |= {"+all3 anchor/L3": anchor_all, "+tissue ratios @all anchors": within_all, "+all ratios": anchor_all + within_all}
    return sets


def rank_pool(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    z, base = np.zeros(n), sm.add_constant(x[:, :N_FIXED])
    for j in range(n):
        try:
            z[j] = sm.Logit(y, np.column_stack([base, x[:, N_FIXED + j]])).fit(disp=0, maxiter=100).tvalues[-1]
        except Exception:
            z[j] = 0.0
    return np.argsort(-np.abs(z))


def run_k(k, x, y, split, order, x_ext, y_ext):
    cols = list(range(N_FIXED)) + [N_FIXED + i for i in order[:k]]
    out = holdout_fit(x, y, split, cols, x_ext)
    return k, roc_auc_score(y[split[2]], out["test_score"]), roc_auc_score(y_ext, out["ext_score"])


def single_or(df, df_ext, pool):
    rows = []
    for d in DISEASES:
        for coh, data in (("gangnam", df), ("sinchon", df_ext)):
            y = data[d].to_numpy(int)
            for f in pool:
                x, _ = matrix(data, ["VAT_sum", f])
                fit = sm.Logit(y, sm.add_constant(x)).fit(disp=0, maxiter=200)
                ci = np.exp(fit.conf_int())[-1]
                rows.append({"disease": d, "cohort": coh, "feature": f, "OR_per_SD": np.exp(fit.params[-1]), "ci_low": ci[0],
                             "ci_high": ci[1], "p_value": fit.pvalues[-1]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby(["disease", "cohort"])["p_value"].transform(lambda p: bh_fdr(p.to_numpy()))
    return out


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    (df, groups), (df_ext, _) = add_ratios(df), add_ratios(df_ext)
    pool = sum(groups.values(), [])
    print(f"비율 후보 {len(pool)}개: " + ", ".join(f"{g}={len(c)}" for g, c in groups.items()))

    # (1) 묶음 모델
    rows, preds = [], []
    sets = model_sets(groups)
    for name, extra in sets.items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, sc = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, sc)
        for d in DISEASES:
            row, pred = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            row["n_features"] = 4 if name == "baseline" else 5 + len(extra)
            rows.append(row)
            preds.append(pred)
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    stats = pd.concat([delong_vs(preds, "baseline"), delong_vs(preds, "Model1")], ignore_index=True)

    # (2) nested top-k 스윕
    x_all, sc = matrix(df, ["VAT_sum"] + pool)
    x_ext_all, _ = matrix(df_ext, ["VAT_sum"] + pool, sc)
    perf, ranks = [], []
    for d in DISEASES:
        rng = np.random.default_rng(SEED)
        y_full, y_ext_full = df[d].to_numpy(int), df_ext[d].to_numpy(int)
        idx, idx_ext = balanced_idx(y_full, rng), balanced_idx(y_ext_full, rng)
        x, y, x_ext, y_ext = x_all[idx], y_full[idx], x_ext_all[idx_ext], y_ext_full[idx_ext]
        split = holdout_split(y)
        idx_tv = np.concatenate([split[0], split[1]])
        order = rank_pool(x[idx_tv], y[idx_tv], len(pool))
        res = Parallel(n_jobs=-1)(delayed(run_k)(k, x, y, split, order, x_ext, y_ext) for k in KS)
        perf += [{"disease": d, "k": k, "n_features": N_FIXED + k, "internal_auc": a, "external_auc": e} for k, a, e in res]
        ranks += [{"disease": d, "rank": r + 1, "feature": pool[i]} for r, i in enumerate(rank_pool(x_all, df[d].to_numpy(int), len(pool)))]
        print(f"[{d}] ratio k-sweep done")
    perf = pd.DataFrame(perf)

    # (3) 단일 비율 OR
    ors = single_or(df, df_ext, pool)
    w = ors.pivot_table(index=["disease", "feature"], columns="cohort", values=["OR_per_SD", "p_value", "p_fdr"]).reset_index()
    w.columns = ["disease", "feature", "or_g", "or_s", "fdr_g", "fdr_s", "p_g", "p_s"]
    w["replicated"] = (w.p_g < 0.05) & (w.p_s < 0.05) & (np.sign(np.log(w.or_g)) == np.sign(np.log(w.or_s)))

    for sheet, t in (("predictions", preds), ("summary", format_floats(summary)), ("delong", format_floats(stats)),
                     ("nested_k_perf", format_floats(perf)), ("ranking_gangnam", pd.DataFrame(ranks)),
                     ("odds_ratio_single", format_floats(ors)), ("or_replication", format_floats(w))):
        save_sheet(t, OUT_XLSX, sheet)

    # 곡선: 비율 풀 vs (참고) 값 풀 k-sweep
    ref = pd.read_excel(CLINIC4_DIR / "landmark" / "landmark_vat_auc.xlsx", sheet_name="summary")
    lvl = pd.read_excel(CLINIC4_DIR / "landmark" / "landmark_anchor_k_sweep.xlsx", sheet_name="perf_by_k")
    for c in ("internal_auc", "external_auc"):
        ref[c] = pd.to_numeric(ref[c]); lvl[c] = pd.to_numeric(lvl[c])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, d in zip(axes, DISEASES):
        p, q = perf[perf.disease == d], lvl[lvl.disease == d]
        ax.plot(p.k, p.internal_auc, color="#1f77b4", label="Ratio pool, Internal")
        ax.plot(p.k, p.external_auc, color="#e07b22", label="Ratio pool, External")
        ax.plot(q.k, q.internal_auc, color="#1f77b4", alpha=.3, label="Value pool, Internal")
        ax.plot(q.k, q.external_auc, color="#e07b22", alpha=.3, label="Value pool, External")
        for m, ls in (("baseline", ":"), ("Model1", "--")):
            r = ref[(ref.disease == d) & (ref.model == m)].iloc[0]
            ax.axhline(r.internal_auc, color="#1f77b4", ls=ls, lw=1, alpha=.7)
            ax.axhline(r.external_auc, color="#e07b22", ls=ls, lw=1, alpha=.7)
        ax.set(title=d, xlabel="선택한 feature 개수 k (순위 = trainval 안 |z|)", ylabel="AUC")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, title="점선=baseline, 파선=Model1", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)

    cols = ["internal_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    order = [(m, d) for m in cols for d in DISEASES]
    names = list(sets)
    show = piv.loc[names][order].map(f3)
    show.insert(0, "#feat", summary.drop_duplicates("model").set_index("model").loc[names, "n_features"].astype(int))
    print("\n묶음 모델 AUC\n" + show.to_string())
    print("\ndelta vs Model1\n" + piv.sub(piv.loc["Model1"], axis=1).loc[names[2:]][order].map(f3).to_string())
    s = stats[(stats.ref == "Model1") & (pd.to_numeric(stats.p_value) < 0.05)]
    print("\nDeLong vs Model1 p<0.05\n" + s[["disease", "cohort", "model", "diff", "p_value", "p_fdr"]].to_string())
    pk = perf.pivot(index="k", columns="disease", values=cols)
    print("\nnested top-k AUC\n" + pk.map(f3).to_string())
    print("\n두 코호트 p<0.05 & 같은 방향 비율(단일)\n" + w[w.replicated].sort_values(["disease", "p_g"])[["disease", "feature", "or_g", "or_s", "p_g", "p_s", "fdr_g", "fdr_s"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
