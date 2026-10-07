from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark slice의 VAT/SAT/TAMA 면적(cm2)과 AEC 값 분포를 성별(All/M/F) x 질환(HTN/DM/CKD Yes/No) 층화로 요약 (QC 제외는 load_cohort 기준)
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clinic4_landmark_vat_auc import LM, TISSUES, load_cohort

sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEASURES = TISSUES + ["AEC"]
UNIT = {"VAT": "cm2", "SAT": "cm2", "TAMA": "cm2", "AEC": "mA"}
LMN = [a.replace("_center", "") for a in LM]


# 해당 환자 landmark slice의 AEC 값을 df에 AEC@landmark 컬럼으로 추가
def add_aec(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    path = f"data/{cohort}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$")
    for a in LM:
        df[f"AEC@{a}"] = [aec.loc[p].to_numpy(float)[int(lm.loc[p, f"{a}_slice"]) - 1] for p in df.PatientID]
    return df


def strata(df: pd.DataFrame):
    out = [("All", "All", pd.Series(True, index=df.index))]
    for d in DIS:
        out += [(d, "Yes", df[d] == 1), (d, "No", df[d] == 0)]
    return out


for c in ["gangnam", "sinchon"]:
    df, _ = load_cohort(c)
    df = add_aec(df, c)
    rows = []
    for sex in ["All", "M", "F"]:
        for d, s, mask in strata(df):
            sub = df[mask & ((df.PatientSex == sex) if sex != "All" else True)]
            for m in MEASURES:
                for a, an in zip(LM, LMN):
                    v = sub[f"{m}@{a}"]
                    rows.append(dict(Measure=m, Sex=sex, Disease=d, Status=s, Landmark=an, n=int(v.count()), mean=v.mean(),
                                     sd=v.std(), median=v.median(), q1=v.quantile(.25), q3=v.quantile(.75), min=v.min(), max=v.max()))
    t = pd.DataFrame(rows)
    t["v"] = t["mean"].map("{:.1f}".format) + "±" + t["sd"].map("{:.1f}".format) + " (" + t["n"].astype(str) + ")"
    t["col"] = t.Sex + " | " + t.Disease + " " + t.Status
    with pd.ExcelWriter(f"{OUT}/{c}_landmark_value_distribution.xlsx") as xw:
        for m in MEASURES:  # 한눈에 비교: 측정값별 시트, 행=landmark, 열=층, 값="mean±sd (n)"
            t[t.Measure == m].pivot(index="Landmark", columns="col", values="v").reindex(LMN).to_excel(xw, sheet_name=f"{m}_compare")
        t.drop(columns=["v", "col"]).round(2).to_excel(xw, sheet_name="detail", index=False)
        dem = [dict(Sex=sex, Disease=d, Status=s, N=len(sub), Age_mean=sub.PatientAge.mean(), BMI_mean=sub.BMI.mean())
               for sex in ["All", "M", "F"] for d, s, mask in strata(df)
               for sub in [df[mask & ((df.PatientSex == sex) if sex != "All" else True)]]]
        pd.DataFrame(dem).round(2).to_excel(xw, sheet_name="demographics", index=False)

    for m in MEASURES:  # 그림: 측정값별 1장, 질환별 패널 (성별 x Yes/No)
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
        for ax, d in zip(axes, DIS):
            for i, (sex, col) in enumerate([("M", "tab:blue"), ("F", "tab:red")]):
                for j, st in enumerate([0, 1]):
                    sub = df[(df.PatientSex == sex) & (df[d] == st)]
                    ax.boxplot([sub[f"{m}@{a}"] for a in LM], positions=[k * 5 + i * 2 + j * .8 for k in range(len(LM))], widths=.7,
                               showfliers=False, patch_artist=True, boxprops=dict(facecolor=col, alpha=.35 if st == 0 else .9))
            ax.set_xticks([k * 5 + 1.4 for k in range(len(LM))]); ax.set_xticklabels(LMN, rotation=60)
            ax.set_title(f"{c} {m}@landmark - {d} (light=No, dark=Yes; blue=M, red=F)", fontsize=9)
        axes[0].set_ylabel(f"{m} ({UNIT[m]})")
        plt.tight_layout(); plt.savefig(f"{OUT}/{c}_{m}_by_landmark.png", dpi=130); plt.close()
    print(c, len(df))
