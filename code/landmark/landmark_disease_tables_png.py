# 코호트·질환별 인원/기본특성 표와 landmark별 체성분 평균(No→Yes, Cohen d) 표를 png로 저장
import sys, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/clinic4/261012"
DIS = ["HTN", "DM", "CKD"]
LMS = ["liver_dome", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "femoral_head", "inferior_pubic_margin"]


# DataFrame을 표 이미지로 저장
def save(df, name, title, w=11):
    fig, ax = plt.subplots(figsize=(w, 0.38 * (len(df) + 2) + .6)); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.4)
    for (r, _), c in t.get_celld().items():
        if r == 0: c.set_facecolor("#dde6f0"); c.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, weight="bold"); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight"); plt.close()


rows = []
for c in ["gangnam", "sinchon"]:
    m = pd.read_excel(f"data/{c}_landmark_filtered.xlsx", sheet_name="patient_metadata")
    for d in DIS:
        y, n = m[m[d] == 1], m[m[d] == 0]
        rows.append([c, d, len(m), f"{len(y)} ({len(y)/len(m):.1%})", f"{n.PatientAge.mean():.1f} → {y.PatientAge.mean():.1f}",
                     f"{n.BMI.mean():.1f} → {y.BMI.mean():.1f}", f"{(n.PatientSex=='M').mean():.2f} → {(y.PatientSex=='M').mean():.2f}"])
save(pd.DataFrame(rows, columns=["cohort", "disease", "N", "Yes n (%)", "Age (No→Yes)", "BMI (No→Yes)", "Male ratio (No→Yes)"]),
     "baseline_by_disease", "Cohort / disease baseline characteristics")

r = pd.read_excel("outputs/data_distribution/landmark_disease_difference.xlsx", sheet_name="detail")
r = r[r.sex == "All"]
for c in ["gangnam", "sinchon"]:
    for ms in ["VAT", "SAT", "TAMA", "AEC"]:
        s = r[(r.cohort == c) & (r.measure == ms)].set_index(["landmark", "disease"])
        out = []
        for lm in LMS:
            row = [lm]
            for d in DIS:
                if (lm, d) in s.index:
                    x = s.loc[(lm, d)]; row.append(f"{x.mean_no:.1f} → {x.mean_yes:.1f} (d={x.cohen_d:.2f}{'*' if x.p_fdr < .05 else ''})")
                else: row.append("-")
            out.append(row)
        save(pd.DataFrame(out, columns=["landmark"] + DIS), f"{c}_{ms}_by_landmark_table",
             f"{c} {ms}: mean No → Yes (Cohen d, * FDR<0.05)", w=10)
