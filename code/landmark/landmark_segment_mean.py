from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 곡선 요약을 PC1이 아니라 "구간 평균"으로: 프로젝트의 기존 구간(liver_dome~L1, L1~L3, L3~L5, L5~pubis)별 환자 평균값
#  raw      = 구간 평균 (예: AEC mA, 조직 cm2)
#  pw_mean  = 구간 평균 / 환자 전체(liver_dome~pubis) 평균  (환자 평균 대비 상대값)
# 질환과의 +1SD당 OR: (b) 나이·성별·BMI 보정, (c) +스캐너 더미 보정. 코호트 2개. 구간은 데이터 보고 고르지 않고 기존 SEGMENTS 사용.
# 출력: outputs/data_distribution/landmark_segment_mean.xlsx
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import load_cohort

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
SEG = [("liver_dome", "L1_center"), ("L1_center", "L3_center"), ("L3_center", "L5_center"), ("L5_center", "inferior_pubic_margin")]
MIN_N = 100


def load(c):
    meta = load_cohort(c)[0][["PatientID", "PatientSex", "PatientAge", "BMI"] + DIS].reset_index(drop=True)
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    sl = {k: lm[f"{k}_slice"].to_numpy(int) for k in {a for s in SEG for a in s}}
    feats = {}
    for m in MEAS:
        pre = "aec" if m == "AEC" else m
        a = pd.read_excel(path, sheet_name=f"{pre}_total").set_index("PatientID").filter(regex=f"^{pre}_\\d+$").loc[meta.PatientID].to_numpy(float)
        whole = np.array([np.nanmean(a[i, sl["liver_dome"][i] - 1:sl["inferior_pubic_margin"][i]]) for i in range(len(meta))])
        for (u, v) in SEG:
            seg = np.array([np.nanmean(a[i, sl[u][i] - 1:sl[v][i]]) for i in range(len(meta))])
            name = f"{u.replace('_center', '')}~{v.replace('_center', '')}"
            feats[(m, name, "raw")] = seg
            feats[(m, name, "pw_mean")] = seg / np.where(whole > 0, whole, np.nan)
    vc = lm["manufacturer_model"].value_counts()
    meta["scanner"] = lm["manufacturer_model"].where(lm["manufacturer_model"].map(vc) >= MIN_N, "other").to_numpy()
    return meta, feats


def fit(x, y, cov):
    ok = np.isfinite(x)
    if x[ok].std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(np.column_stack([(x[ok] - x[ok].mean()) / x[ok].std(), cov[ok]]))
    f = sm.Logit(y[ok], X).fit(disp=0, maxiter=100)
    ci = np.exp(f.conf_int()[1])
    return np.exp(f.params[1]), ci[0], ci[1], f.pvalues[1]


rows = []
for c in ["gangnam", "sinchon"]:
    meta, feats = load(c)
    base = np.column_stack([meta.PatientAge, (meta.PatientSex == "M").astype(float), meta.BMI]).astype(float)
    sc = np.column_stack([base, pd.get_dummies(meta.scanner, drop_first=True).to_numpy(float)])
    for (m, seg, v), x in feats.items():
        if m == "VAT" and seg.endswith("inferior_pubic_margin") and seg.startswith("L5"):
            continue  # 치골 쪽 VAT는 거의 0
        for d in DIS:
            y = meta[d].to_numpy(int)
            o1, l1, h1, p1 = fit(x, y, base)
            o2, l2, h2, p2 = fit(x, y, sc)
            rows.append(dict(cohort=c, measure=m, segment=seg, variant=v, disease=d, OR_adj=o1, ci_low=l1, ci_high=h1, p_adj=p1,
                             OR_scanner=o2, p_scanner=p2))
r = pd.DataFrame(rows)
# BH-FDR: 코호트 x 질환 안 (구간 x 조직 x 전처리 전체 검정 수 기준)
from delong_utils import bh_fdr
for col, fc in (("p_adj", "fdr_adj"), ("p_scanner", "fdr_scanner")):
    r[fc] = r.groupby(["cohort", "disease"])[col].transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
w = r.pivot_table(index=["measure", "segment", "variant", "disease"], columns="cohort", values=["OR_adj", "fdr_adj", "OR_scanner", "fdr_scanner"]).reset_index()
w.columns = ["_".join(x).strip("_") for x in w.columns]
w["both_adj"] = (np.log(w.OR_adj_gangnam) * np.log(w.OR_adj_sinchon) > 0) & (w.fdr_adj_gangnam < .05) & (w.fdr_adj_sinchon < .05)
w["both_scanner"] = (np.log(w.OR_scanner_gangnam) * np.log(w.OR_scanner_sinchon) > 0) & (w.fdr_scanner_gangnam < .05) & (w.fdr_scanner_sinchon < .05)
with pd.ExcelWriter(f"{D}/landmark_segment_mean.xlsx") as xw:
    w.round(4).to_excel(xw, sheet_name="both_cohorts", index=False)
    r.round(4).to_excel(xw, sheet_name="detail", index=False)
pd.set_option("display.width", 250)
print("두 코호트 모두 FDR<0.05 & 같은 방향 (보정 / 장비보정) 개수: 조직 x 전처리")
print(w.groupby(["measure", "variant"])[["both_adj", "both_scanner"]].sum())
print(w[w.both_scanner].sort_values(["measure", "variant"])[["measure", "segment", "variant", "disease", "OR_scanner_gangnam", "OR_scanner_sinchon"]].round(2).to_string(index=False))
