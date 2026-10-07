from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# patient-wise 전처리에서 나온 AEC/SAT 신호가 스캐너(manufacturer_model) 차이 때문인지 확인.
#  1) 장비 분포와 질환 유병률(교란 여부)  2) 장비 더미를 공변량에 추가한 위치별 OR에서 두 코호트 일관 위치 수 (장비 보정 전/후)
#  3) 환자 수가 가장 많은 장비 1종 안에서만 본 일관 위치 수 (장비 고정)
# 전처리 함수/로더는 landmark_patientwise_preprocessing과 같은 정의(복사), 대상: AEC·SAT·TAMA·VAT x pw_mean/pw_z/pw_L3
# 출력: outputs/data_distribution/landmark_scanner_check.xlsx
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import load_cohort
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
VARS = ["raw", "pw_mean", "pw_z", "pw_L3"]
MIN_N = 100  # 장비 더미로 쓰려면 최소 환자 수, 미만은 other


# 환자별 128점 곡선 + L3 값 + 장비 라벨
def load(c):
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
    vc = lm["manufacturer_model"].value_counts()
    meta["scanner"] = lm["manufacturer_model"].where(lm["manufacturer_model"].map(vc) >= MIN_N, "other").to_numpy()
    return meta, raw, l3v


def prep(x, l3, v):
    if v == "raw":
        return x
    if v == "pw_mean":
        return x / (x.mean(1, keepdims=True) + 1e-6)
    if v == "pw_z":
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return (x + 1) / (l3[:, None] + 1)  # pw_L3


def fit_or(x, y, cov):
    if np.nanstd(x) == 0 or not np.isfinite(x).all():
        return np.nan, np.nan
    X = sm.add_constant(np.column_stack([(x - x.mean()) / x.std(), cov]))
    try:
        f = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception:
        return np.nan, np.nan
    return np.exp(f.params[1]), f.pvalues[1]


def covs(meta, dummies):
    base = np.column_stack([meta.PatientAge, (meta.PatientSex == "M").astype(float), meta.BMI]).astype(float)
    if not dummies:
        return base
    dm = pd.get_dummies(meta.scanner, drop_first=True).to_numpy(float)
    return np.column_stack([base, dm]) if dm.shape[1] else base


data = {c: load(c) for c in ["gangnam", "sinchon"]}

# 1) 장비 분포와 질환 유병률
dist = []
for c, (meta, _, _) in data.items():
    for sc, g in meta.groupby("scanner"):
        dist.append(dict(cohort=c, scanner=sc, n=len(g), female=(g.PatientSex == "F").mean(), **{f"{d}_prev": g[d].mean() for d in DIS}))
dist = pd.DataFrame(dist).round(3)
print(dist.to_string(index=False))

# 2) 장비 보정 전/후 일관 위치 수, 3) 최다 장비 단독
top = {c: data[c][0].scanner.value_counts().index[0] for c in data}
rows = []
for v in VARS:
    for m in MEAS:
        for d in DIS:
            res = {}
            for mode in ("base", "scanner_adj", "top_scanner_only"):
                pv = {}
                for c, (meta, raw, l3v) in data.items():
                    X = prep(raw[m], l3v[m], v)
                    sel = (meta.scanner == top[c]).to_numpy() if mode == "top_scanner_only" else np.ones(len(meta), bool)
                    mm = meta[sel].reset_index(drop=True)
                    cv = covs(mm, mode == "scanner_adj")
                    r = np.array([fit_or(X[sel][:, p], mm[d].to_numpy(int), cv) for p in range(128)])
                    pv[c] = pd.DataFrame({"OR": r[:, 0], "p": r[:, 1]})
                    pv[c]["fdr"] = bh_fdr(pv[c].p.fillna(1).to_numpy())
                g, s = pv["gangnam"], pv["sinchon"]
                res[mode] = int(((np.log(g.OR) * np.log(s.OR) > 0) & (g.fdr < .05) & (s.fdr < .05)).sum())
            rows.append(dict(variant=v, measure=m, disease=d, **res))
    print(v, "done", flush=True)
out = pd.DataFrame(rows)
print(out[out.variant != "raw"].pivot_table(index=["measure", "disease"], columns="variant", values=["base", "scanner_adj", "top_scanner_only"]).to_string())
with pd.ExcelWriter(f"{D}/landmark_scanner_check.xlsx") as xw:
    dist.to_excel(xw, sheet_name="scanner_distribution", index=False)
    out.to_excel(xw, sheet_name="robust_pos_count", index=False)
