from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# SAT/TAMA/AEC(+VAT 참조)에서 질환 유무와 연관된 feature를 여러 case로 탐색: 나이·성별·BMI 보정 로지스틱 OR(+1SD), 코호트x질환 안 BH-FDR
#  case: point(landmark 단면) / lm_ratio(landmark 간 비율) / tissue_ratio_pt(landmark 단면의 조직 간 비율) / range_mean(landmark~landmark 구간 평균)
#        / range_ratio(구간 평균의 조직 간 비율) / range_frac(구간 평균 / liver~pubis 평균) / pos(liver~pubis 128점 정규화 위치별 값, 위치별 조직 비율)
# 결과: outputs/data_distribution/landmark_feature_cases.xlsx (all, robust 시트) + pos 곡선 png
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import LM, load_cohort
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
LMN = [a.replace("_center", "") for a in LM]
N_POS = 128
lr = lambda a, b: np.log((a + 1) / (b + 1))  # 로그 비율(0 방지), 기존 landmark 스크립트와 동일


# 코호트의 QC 통과 환자(load_cohort 기준)에 대해 raw slice 곡선 4종과 landmark slice를 읽는다
def load_raw(c: str):
    meta = load_cohort(c)[0][["PatientID", "PatientSex", "PatientAge", "BMI"] + DIS]
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    cur = {m: pd.read_excel(path, sheet_name=f"{'aec' if m == 'AEC' else m}_total").set_index("PatientID").filter(regex=f"^{'aec' if m == 'AEC' else m}_\\d+$").loc[meta.PatientID].to_numpy(float)
           for m in MEAS}
    sl = np.stack([lm[f"{a}_slice"].to_numpy(int) for a in LM], axis=1)  # (n, 12), 1-indexed
    return meta.reset_index(drop=True), cur, sl


# 환자별 feature 사전 생성: 반환 (feature DataFrame, pos 곡선 {이름: (n,128)})
def build(cur, sl):
    n = len(sl)
    rng = list(itertools.combinations(range(len(LM)), 2))
    pt = {m: np.stack([np.nan_to_num(cur[m][np.arange(n), sl[:, k] - 1]) for k in range(len(LM))], 1) for m in MEAS}
    rm = {m: np.stack([np.nanmean(cur[m][i, sl[i, a] - 1:sl[i, b]]) if sl[i, b] > sl[i, a] else np.nan for (a, b) in rng for i in range(n)]).reshape(len(rng), n).T
          for m in MEAS}  # (n, 66) 구간 평균
    whole = {m: np.array([np.nanmean(cur[m][i, sl[i, 0] - 1:sl[i, -1]]) for i in range(n)]) for m in MEAS}  # liver_dome~pubis 평균
    F = {}
    T = {"TAMA/SAT": ("TAMA", "SAT"), "TAMA/(SAT+VAT)": ("TAMA", None), "SAT/(SAT+VAT)": ("SAT", None)}

    def tratio(name, g):  # g(m) -> 해당 단위의 조직값 배열
        a, b = T[name]
        return lr(g("TAMA"), g("SAT")) if name == "TAMA/SAT" else (lr(g("TAMA"), g("SAT") + g("VAT")) if name == "TAMA/(SAT+VAT)" else lr(g("SAT"), g("VAT")))

    for m in MEAS:
        for k, a in enumerate(LMN):
            F[("point", f"{m}@{a}", m)] = pt[m][:, k]
        for (i, j) in itertools.combinations(range(len(LM)), 2):
            F[("lm_ratio", f"{m}@{LMN[i]}/{LMN[j]}", m)] = lr(pt[m][:, i], pt[m][:, j])
        for r, (i, j) in enumerate(rng):
            F[("range_mean", f"{m}[{LMN[i]}~{LMN[j]}]", m)] = rm[m][:, r]
            F[("range_frac", f"{m}[{LMN[i]}~{LMN[j]}]/whole", m)] = rm[m][:, r] / np.where(whole[m] > 0, whole[m], np.nan)
    for name in T:
        for k, a in enumerate(LMN):
            F[("tissue_ratio_pt", f"{name}@{a}", name.split("/")[0])] = tratio(name, lambda m, k=k: pt[m][:, k])
        for r, (i, j) in enumerate(rng):
            F[("range_ratio", f"{name}[{LMN[i]}~{LMN[j]}]", name.split("/")[0])] = tratio(name, lambda m, r=r: rm[m][:, r])
    # 정규화 위치 128점(liver_dome~pubis)
    pos = {m: np.stack([np.interp(np.linspace(sl[i, 0], sl[i, -1], N_POS), np.arange(1, cur[m].shape[1] + 1), np.nan_to_num(cur[m][i])) for i in range(n)]) for m in MEAS}
    P = {m: pos[m] for m in MEAS}
    for name in T:
        P[name] = tratio(name, lambda m: pos[m])
    return pd.DataFrame({k: v for k, v in F.items()}), P


# 표준화한 x의 +1SD당 OR(나이/성별/BMI 보정). 결측/상수 feature는 None
def fit_or(x, y, cov):
    ok = np.isfinite(x)
    if ok.sum() < 100 or np.nanstd(x) == 0:
        return None
    X = sm.add_constant(np.column_stack([(x[ok] - x[ok].mean()) / x[ok].std(), cov[ok]]))
    try:
        f = sm.Logit(y[ok], X).fit(disp=0, maxiter=100)
    except Exception:
        return None
    ci = np.exp(f.conf_int()[1])
    return np.exp(f.params[1]), ci[0], ci[1], f.pvalues[1]


rows, prow = [], []
for c in ["gangnam", "sinchon"]:
    meta, cur, sl = load_raw(c)
    cov = np.column_stack([meta.PatientAge, (meta.PatientSex == "M").astype(float), meta.BMI]).astype(float)
    F, P = build(cur, sl)
    for d in DIS:
        y = meta[d].to_numpy(int)
        for (case, name, tis) in F.columns:
            if "VAT" in name and tis == "VAT" and "pubic" in name and case == "point":
                continue  # 치골 하단 VAT는 항상 0
            r = fit_or(F[(case, name, tis)].to_numpy(float), y, cov)
            if r:
                rows.append(dict(cohort=c, disease=d, case=case, feature=name, tissue=tis, OR=r[0], ci_low=r[1], ci_high=r[2], p=r[3]))
        for k, A in P.items():  # 위치별
            for p in range(N_POS):
                r = fit_or(A[:, p], y, cov)
                if r:
                    prow.append(dict(cohort=c, disease=d, measure=k, pos=p / (N_POS - 1), OR=r[0], p=r[3]))
    print(c, len(meta), F.shape)

r = pd.DataFrame(rows)
r["p_fdr"] = r.groupby(["cohort", "disease"])["p"].transform(lambda p: bh_fdr(p.to_numpy()))
w = r.pivot_table(index=["disease", "case", "feature", "tissue"], columns="cohort", values=["OR", "ci_low", "ci_high", "p_fdr"]).reset_index()
w.columns = ["_".join(x).strip("_") for x in w.columns]
w["robust"] = (np.log(w.OR_gangnam) * np.log(w.OR_sinchon) > 0) & (w.p_fdr_gangnam < .05) & (w.p_fdr_sinchon < .05)
w["min_abs_logOR"] = np.minimum(np.log(w.OR_gangnam).abs(), np.log(w.OR_sinchon).abs())
w = w.sort_values(["disease", "min_abs_logOR"], ascending=[True, False])
pr = pd.DataFrame(prow)
with pd.ExcelWriter(f"{D}/landmark_feature_cases.xlsx") as xw:
    w[w.robust].round(4).to_excel(xw, sheet_name="robust", index=False)
    w.round(4).to_excel(xw, sheet_name="all", index=False)
    pr.round(4).to_excel(xw, sheet_name="pos_curves", index=False)

# 위치별 OR 곡선: 질환 x 조직(비율 포함), 코호트 겹쳐 그리기 (OR 1 기준선, FDR 보정 없이 p<0.05 위치는 굵게)
ms = [m for m in pr.measure.unique()]
fig, axes = plt.subplots(3, len(ms), figsize=(3.0 * len(ms), 8), sharex=True)
for i, d in enumerate(DIS):
    for j, m in enumerate(ms):
        ax = axes[i, j]
        for c, col in [("gangnam", "tab:blue"), ("sinchon", "tab:red")]:
            g = pr[(pr.disease == d) & (pr.measure == m) & (pr.cohort == c)]
            ax.plot(g.pos, g.OR, color=col, lw=1.2, label=c)
            ax.scatter(g.pos[g.p < .05], g.OR[g.p < .05], s=4, color=col)
        ax.axhline(1, color="gray", lw=.6)
        ax.set_title(f"{d} {m}", fontsize=8)
axes[0, 0].legend(fontsize=7)
fig.suptitle("liver_dome(0)~pubis(1) 정규화 위치별 +1SD당 OR (나이·성별·BMI 보정), 점 = p<0.05", fontsize=10)
fig.tight_layout()
fig.savefig(f"{D}/landmark_pos_or_curves.png", dpi=150)

print(w[w.robust].groupby(["disease", "case"]).size().unstack(fill_value=0))
print(w[w.robust].groupby(["disease", "tissue"]).size().unstack(fill_value=0))
