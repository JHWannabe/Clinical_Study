from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 성별(M/F)을 나눠 anchor 체성분 모델의 AUC(7/1/2 hold-out test + sinchon 외부)와 OR을 비교한다.
#  - 성별마다 gangnam/sinchon을 각각 그 성별로만 잘라 baseline(clinic4에서 성별 제외 = 나이/신장/체중), Model1(+VAT 합), 12 anchor 값 모델, 부위 부분집합 모델을 평가
#  - 성별로 나누면 test가 작아(예: gangnam 여성 CKD 18명) AUC에 부트스트랩 95% CI를 붙인다
#  - OR: anchor x 조직 하나씩(+VAT 합 보정, 성별별 +1SD당). 성별 간 이질성 = 두 성별 계수 차이 z검정(BH-FDR)
# 모델 정의는 clinic4_landmark_level_auc.model_sets + clinic4_landmark_anchor_subset_auc.fixed_sets 재사용.

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics import roc_auc_score

import clinic4_landmark_level_auc as lvl
from clinic4_landmark_anchor_subset_auc import fixed_sets
from clinic4_landmark_level_auc import DEAD
from clinic4_landmark_vat_auc import LM, TISSUES, delong_vs, load_cohort, matrix, run_disease_holdout
from clinic4_logistic_regression import DISEASES, PROJECT_ROOT, SEED, f3, format_floats, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

OUT_XLSX = "landmark/landmark_sex_stratified.xlsx"
FIG = PROJECT_ROOT / "outputs" / "clinic4" / "landmark" / "sex_stratified_delta_auc.png"
SEXES = ["M", "F"]
KEY = ["+VAT+SAT@all12", "+all3@all12", "VAT+SAT@pelvis(femoral,pubis)", "all3@pelvis(femoral,pubis)", "VAT+SAT@lowspine(L4,L5,S1)",
       "VAT+SAT@rep4(T12,L3,S1,femoral)"]  # 그림/요약에 쓰는 대표 모델(전체 데이터 분석에서 눈에 띈 것들 + 12 anchor 전체)
N_BOOT = 500


def all_models() -> dict[str, list[str]]:
    return {**lvl.model_sets(), **fixed_sets()}


# 성별 하나로 자른 코호트에서 모델 전부를 hold-out 평가(성별 컬럼은 상수라 제외)
def eval_sex(sex: str, df: pd.DataFrame, df_ext: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    d_s, e_s = df[df.PatientSex == sex].reset_index(drop=True), df_ext[df_ext.PatientSex == sex].reset_index(drop=True)
    rows, preds = [], []
    for name, extra in all_models().items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, sc = matrix(d_s, cols)
        x_ext, _ = matrix(e_s, cols, sc)
        for d in DISEASES:
            row, pred = run_disease_holdout(d, name, x[:, 1:], d_s, x_ext[:, 1:], e_s)
            row["sex"] = sex
            row["n_features"] = (3 if name == "baseline" else 4 + len(extra))  # 성별 제외 clinic4 3개(+VAT 합 +추가)
            rows.append(row)
            preds.append(pred.assign(sex=sex))
    return rows, preds


def boot_ci(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def add_cis(summary: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ci = {}
    for (d, sx, m, coh), g in preds.groupby(["disease", "sex", "model", "cohort"]):
        ci[(d, sx, m, coh)] = boot_ci(g.y.to_numpy(int), g.score.to_numpy(), rng)
    for coh, col in (("gangnam", "internal"), ("sinchon", "external")):
        summary[f"{col}_ci_low"] = [ci[(r.disease, r.sex, r.model, coh)][0] for r in summary.itertuples()]
        summary[f"{col}_ci_high"] = [ci[(r.disease, r.sex, r.model, coh)][1] for r in summary.itertuples()]
    return summary


# anchor x 조직 하나씩 + VAT 합: 성별별 OR(+1SD당), 성별 간 이질성 z검정
def or_by_sex(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for sex in SEXES:
        d_s = df[df.PatientSex == sex].reset_index(drop=True)
        feats = [("VAT_sum", "VAT_sum", "sum")] + [(f"{t}@{a}", a, t) for a in LM for t in TISSUES if (t, a) != DEAD]
        for term, a, t in feats:
            cols = ["VAT_sum"] + ([term] if term != "VAT_sum" else [])
            x, _ = matrix(d_s, cols)
            x = x[:, 1:]  # 성별 제외
            for d in DISEASES:
                try:
                    fit = sm.Logit(d_s[d].to_numpy(int), sm.add_constant(x)).fit(disp=0, maxiter=200)
                except Exception:
                    continue
                rows.append({"cohort": cohort, "disease": d, "sex": sex, "landmark": a, "tissue": t, "n": len(d_s), "n_pos": int(d_s[d].sum()),
                             "coef": fit.params[-1], "se": fit.bse[-1], "OR_per_SD": np.exp(fit.params[-1]), "p_value": fit.pvalues[-1]})
    o = pd.DataFrame(rows)
    w = o.pivot_table(index=["cohort", "disease", "landmark", "tissue"], columns="sex", values=["coef", "se", "p_value", "OR_per_SD"]).reset_index()
    w.columns = ["cohort", "disease", "landmark", "tissue"] + [f"{a}_{b}" for a, b in w.columns[4:]]
    w["z_diff"] = (w.coef_M - w.coef_F) / np.sqrt(w.se_M ** 2 + w.se_F ** 2)
    w["p_het"] = 2 * st.norm.sf(np.abs(w.z_diff))
    w["p_het_fdr"] = w.groupby("disease")["p_het"].transform(lambda p: bh_fdr(p.to_numpy()))
    return w


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    cnt = pd.concat([df.assign(cohort="gangnam"), df_ext.assign(cohort="sinchon")]).groupby(["cohort", "PatientSex"]).agg(
        n=("PatientID", "size"), HTN=("HTN", "sum"), DM=("DM", "sum"), CKD=("CKD", "sum")).reset_index()
    print("성별 코호트 구성\n" + cnt.to_string(index=False))

    rows, preds = [], []
    for sx in SEXES:
        r, p = eval_sex(sx, df, df_ext)
        rows += r
        preds += p
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    summary = add_cis(summary, preds)
    stats = pd.concat([delong_vs(preds[preds.sex == sx], ref).assign(sex=sx) for sx in SEXES for ref in ("baseline", "Model1")], ignore_index=True)

    het = pd.concat([or_by_sex(df, "gangnam"), or_by_sex(df_ext, "sinchon")], ignore_index=True)
    for sheet, t in (("counts", cnt), ("summary", format_floats(summary)), ("delong", format_floats(stats)), ("or_by_sex", format_floats(het)),
                     ("predictions", preds)):
        save_sheet(t, OUT_XLSX, sheet)

    # 요약 출력: 성별 x (baseline, Model1, 대표 모델) AUC [95% CI]
    show = ["baseline", "Model1"] + KEY
    for sx in SEXES:
        print(f"\n===== {sx} AUC  internal(test) [CI] | external [CI] =====")
        for d in DISEASES:
            g = summary[(summary.sex == sx) & (summary.disease == d)].set_index("model")
            print(f"[{d}] n_test={int(g.n_test.iloc[0])}, n_external={int(g.n_external.iloc[0])}")
            for m in show:
                r = g.loc[m]
                print(f"  {m:34s} {r.internal_auc:.3f} [{r.internal_ci_low:.2f},{r.internal_ci_high:.2f}] | {r.external_auc:.3f} [{r.external_ci_low:.2f},{r.external_ci_high:.2f}]")

    # 그림: Model1 대비 AUC 변화(성별 비교)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey="row")
    for r_i, (col, title) in enumerate((("internal_auc", "Internal (gangnam test)"), ("external_auc", "External (sinchon)"))):
        for c_i, d in enumerate(DISEASES):
            ax = axes[r_i, c_i]
            for k, (sx, color) in enumerate((("M", "#1f77b4"), ("F", "#e07b22"))):
                g = summary[(summary.sex == sx) & (summary.disease == d)].set_index("model")
                dv = [g.loc[m, col] - g.loc["Model1", col] for m in KEY]
                ax.bar(np.arange(len(KEY)) + (k - 0.5) * 0.38, dv, 0.38, color=color, label={"M": "남성", "F": "여성"}[sx])
            ax.axhline(0, color="gray", lw=1)
            ax.set_title(f"{d} - {title}", fontsize=11)
            ax.set_xticks(range(len(KEY)), [m.replace("(femoral,pubis)", "").replace("(L4,L5,S1)", "").replace("(T12,L3,S1,femoral)", "") for m in KEY],
                          rotation=60, ha="right", fontsize=8)
            ax.grid(alpha=.3, axis="y")
    axes[0, 0].set_ylabel("Δ AUC vs Model1")
    axes[1, 0].set_ylabel("Δ AUC vs Model1")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)

    # Model1 대비 변화 + DeLong
    print("\n===== Model1 대비 ΔAUC (대표 모델) =====")
    for sx in SEXES:
        for d in DISEASES:
            g = summary[(summary.sex == sx) & (summary.disease == d)].set_index("model")
            print(f"{sx} {d}: " + " | ".join(f"{m.split('@')[0] + '@' + m.split('@')[-1][:6]} {g.loc[m, 'internal_auc'] - g.loc['Model1', 'internal_auc']:+.3f}/{g.loc[m, 'external_auc'] - g.loc['Model1', 'external_auc']:+.3f}" for m in KEY))
    s = stats[(stats.ref == "Model1") & (pd.to_numeric(stats.p_value) < 0.05) & (pd.to_numeric(stats["diff"]) > 0)]
    print("\nDeLong vs Model1 p<0.05 (향상)\n" + s[["sex", "disease", "cohort", "model", "diff", "p_value", "p_fdr"]].round(3).to_string(index=False))

    # OR 성별 이질성: 같은 코호트에서 성별 계수가 다른 feature (p_het<0.05), 양 코호트에서 같은 방향 이질성이면 강조
    h = het[het.p_het < 0.05].sort_values("p_het")
    print(f"\n성별 이질성(p_het<0.05) feature 수: {len(h)} / {len(het)}, FDR<0.10: {int((het.p_het_fdr < 0.10).sum())}")
    print(h.head(25)[["cohort", "disease", "landmark", "tissue", "OR_per_SD_M", "OR_per_SD_F", "p_value_M", "p_value_F", "p_het", "p_het_fdr"]].round(3).to_string(index=False))
    w = het.pivot_table(index=["disease", "landmark", "tissue"], columns="cohort", values=["z_diff"]).dropna()
    w.columns = ["z_gangnam", "z_sinchon"]
    both = w[(np.sign(w.z_gangnam) == np.sign(w.z_sinchon)) & (w.z_gangnam.abs() > 1.96) & (w.z_sinchon.abs() > 1.96)]
    print("\n두 코호트 모두 성별 이질성(|z|>1.96, 같은 방향)\n" + both.round(2).to_string())


if __name__ == "__main__":
    main()
