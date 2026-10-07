from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 12 anchor x 5조직 체성분 값(60개, 단 VAT@pubis는 항상 0이라 실효 K_MAX개)에서 k=1..K_MAX개를 골라 Model1(clinic4 + VAT 합)에 더했을 때의
# (1) AUC: gangnam hold-out(7/1/2) test + sinchon 외부. 순위(clinic4+VAT합 보정 후 feature별 |z|)는 trainval 안에서만 계산 -> test 누수 없음
# (2) OR: gangnam 전체의 같은 순위 상위 k개를 동시에 넣은 로지스틱 회귀를 gangnam/sinchon 각각 적합(+1SD당 OR). gangnam OR은 같은 데이터로 선택했으므로
#     post-selection이고, sinchon이 독립 확인용이다.

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from clinic4_landmark_anchor_subset_auc import CANDIDATES
from clinic4_landmark_vat_auc import holdout_fit, holdout_split, load_cohort, matrix
from clinic4_logistic_regression import CLINIC4_DIR, DISEASES, PARAM_GRID, PROJECT_ROOT, SEED, balanced_idx, f3, format_floats, save_sheet

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

OUT_XLSX = "landmark/landmark_anchor_k_sweep.xlsx"
FIG = PROJECT_ROOT / "outputs" / "clinic4" / "landmark" / "anchor_k_curve.png"
N_FIXED = 5  # sex, age, height, weight, VAT_sum
K_MAX = len(CANDIDATES)  # 59


# 보정 변수(clinic4 + VAT 합)를 넣은 로지스틱에서 후보 feature 하나씩의 z값 -> |z| 내림차순 후보 인덱스(0..58)
def rank_candidates(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.zeros(K_MAX)
    base = sm.add_constant(x[:, :N_FIXED])
    for j in range(K_MAX):
        try:
            z[j] = sm.Logit(y, np.column_stack([base, x[:, N_FIXED + j]])).fit(disp=0, maxiter=100).tvalues[-1]
        except Exception:
            z[j] = 0.0
    return np.argsort(-np.abs(z))


# 한 (질환, k)의 hold-out test AUC와 외부 AUC. order = trainval(train+valid)에서만 계산한 순위(test는 선택에 쓰지 않음), 튜닝은 valid 단일 fold GridSearchCV
def run_k(k, x, y, split, order, x_ext, y_ext):
    cols = list(range(N_FIXED)) + [N_FIXED + i for i in order[:k]]
    out = holdout_fit(x, y, split, cols, x_ext)
    return k, roc_auc_score(y[split[2]], out["test_score"]), roc_auc_score(y_ext, out["ext_score"])


def or_rows(disease, cohort, df, feats, k):
    cols = ["VAT_sum"] + feats
    x, _ = matrix(df, cols)
    terms = ["sex_M", "age", "height", "weight"] + cols
    try:
        fit = sm.Logit(df[disease].to_numpy(int), sm.add_constant(x)).fit(disp=0, maxiter=300)
        ci, ok = np.exp(fit.conf_int()), bool(fit.mle_retvals["converged"])
    except Exception:
        return []
    cond = float(np.linalg.cond(x))
    return [{"disease": disease, "k": k, "cohort": cohort, "term": t, "rank": (feats.index(t) + 1 if t in feats else 0),
             "OR_per_SD": np.exp(fit.params[i + 1]), "ci_low": ci[i + 1, 0], "ci_high": ci[i + 1, 1], "p_value": fit.pvalues[i + 1],
             "converged": ok, "cond_number": cond} for i, t in enumerate(terms) if t in cols]


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    x_all, scaler = matrix(df, ["VAT_sum"] + CANDIDATES)
    x_ext_all, _ = matrix(df_ext, ["VAT_sum"] + CANDIDATES, scaler)

    perf, ors, ranks = [], [], []
    for d in DISEASES:
        rng = np.random.default_rng(SEED)
        y_full, y_ext_full = df[d].to_numpy(int), df_ext[d].to_numpy(int)
        idx, idx_ext = balanced_idx(y_full, rng), balanced_idx(y_ext_full, rng)  # run_disease_holdout과 동일한 추출 순서
        x, y, x_ext, y_ext = x_all[idx], y_full[idx], x_ext_all[idx_ext], y_ext_full[idx_ext]
        split = holdout_split(y)
        idx_tv = np.concatenate([split[0], split[1]])
        order = rank_candidates(x[idx_tv], y[idx_tv])
        res = Parallel(n_jobs=-1)(delayed(run_k)(k, x, y, split, order, x_ext, y_ext) for k in range(1, K_MAX + 1))
        perf += [{"disease": d, "k": k, "n_features": N_FIXED + k, "internal_auc": a, "external_auc": e} for k, a, e in res]
        print(f"[{d}] k-sweep done")

        # OR: 순위는 gangnam 전체(비균형) 기준
        order_ols = rank_candidates(x_all, df[d].to_numpy(int))
        feats_all = [CANDIDATES[i] for i in order_ols]
        ranks += [{"disease": d, "rank": r + 1, "feature": f} for r, f in enumerate(feats_all)]
        for k in range(1, K_MAX + 1):
            for coh, data in (("gangnam", df), ("sinchon", df_ext)):
                ors += or_rows(d, coh, data, feats_all[:k], k)

    perf, ors, ranks = pd.DataFrame(perf), pd.DataFrame(ors), pd.DataFrame(ranks)
    ref = pd.read_excel(CLINIC4_DIR / "landmark" / "landmark_vat_auc.xlsx", sheet_name="summary")
    ref = ref[ref.model.isin(["baseline", "Model1"])].copy()
    for c in ("internal_auc", "external_auc"):
        ref[c] = pd.to_numeric(ref[c])

    # k별 OR 요약: 상위 k개 중 sinchon에서도 p<0.05이고 gangnam과 같은 방향인 feature 수
    w = ors[ors["rank"] > 0].pivot_table(index=["disease", "k", "term"], columns="cohort", values=["OR_per_SD", "p_value"]).reset_index()
    w.columns = ["disease", "k", "term", "or_g", "or_s", "p_g", "p_s"]
    w["replicated"] = (w.p_g < 0.05) & (w.p_s < 0.05) & (np.sign(np.log(w.or_g)) == np.sign(np.log(w.or_s)))
    w["sinchon_sig"] = w.p_s < 0.05
    rep = w.groupby(["disease", "k"]).agg(n_replicated=("replicated", "sum"), n_sinchon_sig=("sinchon_sig", "sum")).reset_index()
    perf = perf.merge(rep, on=["disease", "k"], how="left")

    for sheet, t in (("perf_by_k", format_floats(perf)), ("or_by_k", format_floats(ors)), ("ranking_gangnam", ranks)):
        save_sheet(t, OUT_XLSX, sheet)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, d in zip(axes, DISEASES):
        p = perf[perf.disease == d]
        ax.plot(p.k, p.internal_auc, label="Internal (gangnam test)", color="#1f77b4")
        ax.plot(p.k, p.external_auc, label="External (sinchon)", color="#e07b22")
        for m, ls in (("baseline", ":"), ("Model1", "--")):
            r = ref[(ref.disease == d) & (ref.model == m)].iloc[0]
            ax.axhline(r.internal_auc, color="#1f77b4", ls=ls, lw=1, alpha=.7)
            ax.axhline(r.external_auc, color="#e07b22", ls=ls, lw=1, alpha=.7)
        ax.set(title=d, xlabel="선택한 anchor 체성분 값 개수 k (순위 = trainval 안 |z|)", ylabel="AUC")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8, title="점선=baseline, 파선=Model1", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)

    ks = [k for k in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 59) if k <= K_MAX] + ([K_MAX] if K_MAX not in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 59) else [])
    piv = perf.pivot(index="k", columns="disease", values=["internal_auc", "external_auc"])
    base = ref.pivot(index="model", columns="disease", values=["internal_auc", "external_auc"])
    print("\nreference\n" + base.map(f3).to_string())
    print("\nAUC by k\n" + piv.loc[ks].map(f3).to_string())
    best = perf.assign(mean_auc=(perf.internal_auc + perf.external_auc) / 2).sort_values("mean_auc", ascending=False).groupby("disease").head(3)
    print("\n질환별 (내부+외부 평균) 상위 3 k\n" + best[["disease", "k", "internal_auc", "external_auc"]].map(lambda v: f3(v) if isinstance(v, float) else v).to_string(index=False))
    print("\nk별 sinchon에서도 재현(p<0.05 & 같은 방향)된 feature 수\n" + rep.pivot(index="k", columns="disease", values="n_replicated").loc[ks].to_string())
    print("\n수렴 실패/조건수:\n", ors.groupby("k").agg(nonconv=("converged", lambda s: int((~s).sum())), cond=("cond_number", "max")).loc[ks].round(1).to_string())


if __name__ == "__main__":
    main()
