from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# patient-wise(환자 평균 대비) 곡선의 코호트별 PC1 점수(곡선 그림 p값과 같은 통계량)를 질환과 비교:
#  (a) 미보정 Mann-Whitney(성별 층화 아님, 전체)  (b) 나이·성별·BMI 보정 로지스틱 OR(+1SD)  (c) (b)+스캐너 더미 보정
# 대상: VAT/SAT/TAMA/AEC, 질환 3개, 코호트 2개. 출력: outputs/data_distribution/landmark_pc1_adjusted.xlsx
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

from clinic4_landmark_vat_auc import load_cohort

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
MIN_N = 100


# 환자별 128점 곡선과 스캐너 라벨 (landmark_scanner_check.load와 같은 정의)
def load(c):
    meta = load_cohort(c)[0][["PatientID", "PatientSex", "PatientAge", "BMI"] + DIS].reset_index(drop=True)
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    s, e = lm["liver_dome_slice"].to_numpy(int), lm["inferior_pubic_margin_slice"].to_numpy(int)
    raw = {}
    for m in MEAS:
        pre = "aec" if m == "AEC" else m
        a = pd.read_excel(path, sheet_name=f"{pre}_total").set_index("PatientID").filter(regex=f"^{pre}_\\d+$").loc[meta.PatientID].to_numpy(float)
        raw[m] = np.stack([np.interp(np.linspace(s[i], e[i], 128), np.arange(1, a.shape[1] + 1), np.nan_to_num(a[i])) for i in range(len(meta))])
    vc = lm["manufacturer_model"].value_counts()
    meta["scanner"] = lm["manufacturer_model"].where(lm["manufacturer_model"].map(vc) >= MIN_N, "other").to_numpy()
    return meta, raw


# 코호트 전체 곡선의 PC1 점수 (비지도, 질환 정보 미사용). SVD 부호는 임의라 코호트 간 비교를 위해
# "치골 쪽(위치 0.9) 곡선 값이 클수록 +"가 되도록 부호를 고정한다
def pc1(X):
    Xc = X - X.mean(0)
    sc = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
    return sc if np.corrcoef(sc, X[:, int(0.9 * 127)])[0, 1] >= 0 else -sc


def fit(score, y, cov):
    X = sm.add_constant(np.column_stack([(score - score.mean()) / score.std(), cov]))
    f = sm.Logit(y, X).fit(disp=0, maxiter=100)
    ci = np.exp(f.conf_int()[1])
    return np.exp(f.params[1]), ci[0], ci[1], f.pvalues[1]


rows = []
for c in ["gangnam", "sinchon"]:
    meta, raw = load(c)
    base = np.column_stack([meta.PatientAge, (meta.PatientSex == "M").astype(float), meta.BMI]).astype(float)
    dm = pd.get_dummies(meta.scanner, drop_first=True).to_numpy(float)
    for m in MEAS:
        X = raw[m] / (raw[m].mean(1, keepdims=True) + 1e-6)  # pw_mean
        sc = pc1(X)
        for d in DIS:
            y = meta[d].to_numpy(int)
            p_raw = mannwhitneyu(sc[y == 1], sc[y == 0]).pvalue
            o1, l1, h1, p1 = fit(sc, y, base)
            o2, l2, h2, p2 = fit(sc, y, np.column_stack([base, dm]))
            rows.append(dict(cohort=c, measure=m, disease=d, p_unadj=p_raw, OR_adj=o1, ci_low=l1, ci_high=h1, p_adj=p1,
                             OR_adj_scanner=o2, ci_low_scanner=l2, ci_high_scanner=h2, p_adj_scanner=p2))
r = pd.DataFrame(rows)
r.round(4).to_excel(f"{D}/landmark_pc1_adjusted.xlsx", index=False)
pd.set_option("display.width", 220)
print(r[r.measure == "AEC"].round(4).to_string(index=False))
print(r[r.measure != "AEC"][["cohort", "measure", "disease", "p_unadj", "OR_adj", "p_adj", "OR_adj_scanner", "p_adj_scanner"]].round(4).to_string(index=False))
