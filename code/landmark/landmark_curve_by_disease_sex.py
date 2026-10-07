from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# outputs/clinic4/figures/aec_curve_by_disease_sex.png 와 같은 형식: liver_dome~pubis 128구간 곡선(VAT/SAT/TAMA/AEC, TAMA/SAT 비율)을
# 질환(HTN/DM/CKD) 유무 x 성별로 평균 ± 95% CI, 패널 제목에 환자별 평균의 Mann-Whitney p. 코호트(gangnam/sinchon)별로 저장.
# 출력: outputs/data_distribution/figures/{cohort}_{measure}_curve_by_disease_sex.png
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from clinic4_landmark_vat_auc import LM, load_cohort

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/data_distribution/figures"
DIS = ["HTN", "DM", "CKD"]
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
COL = {"F": "#c0392b", "M": "#1f3b63"}
NAME = {"F": "여성", "M": "남성"}
YLAB = {"VAT": "VAT 면적 (cm2)", "SAT": "SAT 면적 (cm2)", "TAMA": "TAMA 면적 (cm2)", "AEC": "AEC 튜브 전류 (mA)", "TAMA/SAT": "log((TAMA+1)/(SAT+1))"}
Z = np.linspace(0, 1, 128)


# 환자별 liver_dome~pubis 128점 리샘플 곡선 (load_cohort QC 통과 환자, 조직별 {이름: (n,128)})
def curves(c: str):
    meta = load_cohort(c)[0][["PatientID", "PatientSex"] + DIS]
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    s, e = lm["liver_dome_slice"].to_numpy(int), lm["inferior_pubic_margin_slice"].to_numpy(int)
    out = {}
    for m in MEAS:
        pre = "aec" if m == "AEC" else m
        a = pd.read_excel(path, sheet_name=f"{pre}_total").set_index("PatientID").filter(regex=f"^{pre}_\\d+$").loc[meta.PatientID].to_numpy(float)
        out[m] = np.stack([np.interp(np.linspace(s[i], e[i], 128), np.arange(1, a.shape[1] + 1), np.nan_to_num(a[i])) for i in range(len(meta))])
    out["TAMA/SAT"] = np.log((out["TAMA"] + 1) / (out["SAT"] + 1))
    return meta.reset_index(drop=True), out


# 평균 곡선 + 95% CI 밴드
def band(ax, X, color, label, ls):
    m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
    ax.fill_between(Z, m - 1.96 * se, m + 1.96 * se, color=color, alpha=.12, lw=0)
    ax.plot(Z, m, color=color, ls=ls, lw=2.2, label=label)


# 곡선 하나를 환자별 1개 값으로: raw = 곡선 평균, patient-wise(환자 평균이 항상 1) = 코호트 전체 곡선의 PC1 점수(비지도, 질환 정보 미사용)
def score(X, pw):
    if not pw:
        return X.mean(1)
    Xc = X - X.mean(0)
    return Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]


def pstr(a, b):
    p = mannwhitneyu(a, b).pvalue
    return f"p<0.001 (*)" if p < .001 else f"p={p:.3f} ({'*' if p < .05 else 'ns'})"


# patient-wise 전처리: 환자 자신의 liver~pubis 평균으로 나눈 상대 곡선 (비율형 TAMA/SAT는 제외)
def pw_mean(cv):
    return {m: X / (X.mean(1, keepdims=True) + 1e-6) for m, X in cv.items() if "/" not in m}


for c in ["gangnam", "sinchon"]:
    meta, cv0 = curves(c)
    for tag, cv in (("", cv0), ("_pwmean", pw_mean(cv0))):
      for m, X in cv.items():
          sc = score(X, bool(tag))
          fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
          for ax, d in zip(axes, DIS):
              ps = []
              for sx in ("F", "M"):
                  for st, ls, lab in ((0, "--", "음성"), (1, "-", "양성")):
                      g = X[((meta.PatientSex == sx) & (meta[d] == st)).to_numpy()]
                      band(ax, g, COL[sx], f"{NAME[sx]} {lab}", ls)
                  ps.append(f"{NAME[sx]} {pstr(sc[((meta.PatientSex == sx) & (meta[d] == 0)).to_numpy()], sc[((meta.PatientSex == sx) & (meta[d] == 1)).to_numpy()])}")
              ax.set_title(f"{d}\n" + "  ".join(ps), fontsize=10)
              ax.set_xlabel("z축: 간 → 치골 (128구간)")
              ax.spines[["top", "right"]].set_visible(False)
          axes[0].set_ylabel(YLAB[m] + (" / 환자 평균" if tag else ""))
          axes[0].legend(fontsize=8, frameon=False)
          fig.suptitle(f"질환 유무에 따른 {m}{' patient-wise(환자 평균 대비; p=PC1 점수)' if tag else ''} 곡선 ({c} 코호트, 평균 ± 95% CI; 실선=양성, 점선=음성)", fontsize=12)
          fig.tight_layout()
          fig.savefig(f"{OUT}/{c}_{m.replace('/', '_over_')}{tag}_curve_by_disease_sex.png", dpi=170)
          plt.close(fig)
    print(c, len(meta))
