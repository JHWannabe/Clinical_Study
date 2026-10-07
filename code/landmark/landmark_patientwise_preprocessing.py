from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# raw 곡선 vs patient-wise 전처리 곡선(환자 자신의 값으로 정규화)의 질환 연관성 비교.
#  전처리: raw / pw_mean(곡선 / 환자 평균) / pw_z(환자 내 z-score) / pw_minmax(환자 내 0~1) / pw_L3(곡선 / 환자 L3 값)
#  평가1) liver_dome~pubis 128 위치별 +1SD당 OR(나이·성별·BMI 보정, 코호트x질환 안 BH-FDR) -> 두 코호트 일관 위치 수, best 위치 OR
#  평가2) Gangnam에서 p 최소 위치 1개를 골라 clinical(나이·성별·BMI)에 더한 모델의 Sinchon 외부 AUC 변화(선택은 Gangnam만 사용, 누수 없음)
# 출력: outputs/data_distribution/landmark_patientwise_preprocessing.xlsx, landmark_patientwise_or_curves_{tissue}.png
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import load_cohort
from delong_utils import bh_fdr

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
VARS = ["raw", "pw_mean", "pw_z", "pw_minmax", "pw_L3"]
Z = np.linspace(0, 1, 128)


# 환자별 128점 곡선(raw) + L3 landmark 값 반환
def load(c: str):
    meta = load_cohort(c)[0][["PatientID", "PatientSex", "PatientAge", "BMI"] + DIS].reset_index(drop=True)
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    s, e, l3 = (lm[f"{k}_slice"].to_numpy(int) for k in ("liver_dome", "inferior_pubic_margin", "L3_center"))
    raw, l3v = {}, {}
    for m in MEAS:
        pre = "aec" if m == "AEC" else m
        a = pd.read_excel(path, sheet_name=f"{pre}_total").set_index("PatientID").filter(regex=f"^{pre}_\\d+$").loc[meta.PatientID].to_numpy(float)
        raw[m] = np.stack([np.interp(np.linspace(s[i], e[i], 128), np.arange(1, a.shape[1] + 1), np.nan_to_num(a[i])) for i in range(len(meta))])
        l3v[m] = np.nan_to_num(a[np.arange(len(meta)), l3 - 1])
    return meta, raw, l3v


# 환자 단위 전처리 적용: x (n,128), l3 (n,)
def prep(x, l3, v):
    if v == "raw":
        return x
    if v == "pw_mean":
        return x / (x.mean(1, keepdims=True) + 1e-6)
    if v == "pw_z":
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    if v == "pw_minmax":
        lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
        return (x - lo) / (hi - lo + 1e-6)
    return (x + 1) / (l3[:, None] + 1)  # pw_L3


def fit_or(x, y, cov):
    if np.nanstd(x) == 0 or not np.isfinite(x).all():
        return None
    X = sm.add_constant(np.column_stack([(x - x.mean()) / x.std(), cov]))
    try:
        f = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception:
        return None
    return np.exp(f.params[1]), f.pvalues[1]


def auc_gain(xg, yg, cg, xs, ys, cs, j):  # Gangnam에서 학습, Sinchon 평가: clinical vs clinical+feature j
    def fitpred(cols_g, cols_s):
        sc = StandardScaler().fit(cols_g)
        m = LogisticRegression(max_iter=1000).fit(sc.transform(cols_g), yg)
        return roc_auc_score(ys, m.predict_proba(sc.transform(cols_s))[:, 1])
    return fitpred(np.column_stack([cg, xg[:, j]]), np.column_stack([cs, xs[:, j]])) - fitpred(cg, cs)


data = {c: load(c) for c in ["gangnam", "sinchon"]}
rows, summ = [], []
for v in VARS:
    for m in MEAS:
        P = {c: prep(data[c][1][m], data[c][2][m], v) for c in data}
        for d in DIS:
            res = {}
            for c in data:
                meta = data[c][0]
                cov = np.column_stack([meta.PatientAge, (meta.PatientSex == "M").astype(float), meta.BMI]).astype(float)
                y = meta[d].to_numpy(int)
                r = [fit_or(P[c][:, p], y, cov) for p in range(128)]
                res[c] = pd.DataFrame({"OR": [x[0] if x else np.nan for x in r], "p": [x[1] if x else np.nan for x in r]})
                res[c]["p_fdr"] = bh_fdr(res[c].p.fillna(1).to_numpy())
                rows += [dict(variant=v, measure=m, disease=d, cohort=c, pos=Z[p], OR=res[c].OR[p], p=res[c].p[p], p_fdr=res[c].p_fdr[p]) for p in range(128)]
            g, s = res["gangnam"], res["sinchon"]
            rob = (np.log(g.OR) * np.log(s.OR) > 0) & (g.p_fdr < .05) & (s.p_fdr < .05)
            jbest = int(g.p.fillna(1).idxmin())  # Gangnam에서 선택
            mg, ms = data["gangnam"][0], data["sinchon"][0]
            cg = np.column_stack([mg.PatientAge, (mg.PatientSex == "M").astype(float), mg.BMI]).astype(float)
            cs = np.column_stack([ms.PatientAge, (ms.PatientSex == "M").astype(float), ms.BMI]).astype(float)
            summ.append(dict(variant=v, measure=m, disease=d, n_robust_pos=int(rob.sum()), best_pos=Z[jbest], OR_gangnam=g.OR[jbest], OR_sinchon=s.OR[jbest],
                             p_fdr_sinchon=s.p_fdr[jbest],
                             ext_auc_gain=auc_gain(P["gangnam"], mg[d].to_numpy(int), cg, P["sinchon"], ms[d].to_numpy(int), cs, jbest)))
    print(v, "done", flush=True)

pos, sm_ = pd.DataFrame(rows), pd.DataFrame(summ)
with pd.ExcelWriter(f"{D}/landmark_patientwise_preprocessing.xlsx") as xw:
    sm_.round(4).to_excel(xw, sheet_name="summary", index=False)
    sm_.pivot_table(index=["measure", "disease"], columns="variant", values="n_robust_pos")[VARS].to_excel(xw, sheet_name="robust_pos_count")
    sm_.pivot_table(index=["measure", "disease"], columns="variant", values="ext_auc_gain")[VARS].round(4).to_excel(xw, sheet_name="ext_auc_gain")
    pos.round(4).to_excel(xw, sheet_name="pos_curves", index=False)

# 그림: 조직별 1장, 행=전처리, 열=질환, 선=코호트(점 = p<0.05)
for m in MEAS:
    fig, axes = plt.subplots(len(VARS), 3, figsize=(12, 2.2 * len(VARS)), sharex=True)
    for i, v in enumerate(VARS):
        for j, d in enumerate(DIS):
            ax = axes[i, j]
            for c, col in [("gangnam", "tab:blue"), ("sinchon", "tab:red")]:
                g = pos[(pos.variant == v) & (pos.measure == m) & (pos.disease == d) & (pos.cohort == c)]
                ax.plot(g.pos, g.OR, color=col, lw=1.2, label=c)
                ax.scatter(g.pos[g.p < .05], g.OR[g.p < .05], s=4, color=col)
            ax.axhline(1, color="gray", lw=.6)
            ax.set_title(f"{v} | {d}", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{m}: 전처리별 정규화 위치별 +1SD당 OR (나이·성별·BMI 보정), 점 = p<0.05", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{D}/landmark_patientwise_or_curves_{m}.png", dpi=140)
    plt.close(fig)

pd.set_option("display.width", 200)
print(sm_.pivot_table(index=["measure", "disease"], columns="variant", values="n_robust_pos")[VARS])
print(sm_.pivot_table(index=["measure", "disease"], columns="variant", values="ext_auc_gain")[VARS].round(3))
