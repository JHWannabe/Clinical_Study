# 4단계: clinic4에 조직 간 비율(VAT/SAT, TAMA/BMI, VAT/TAMA를 12 landmark마다)을 추가했을 때 AUC 변화. clinic4·직전 집합 대비 5-fold CV(풀링/강남/신촌, 3회 반복, Nadeau-Bengio 보정 t)와 강남↔신촌 외부검증(DeLong)
# 질환마다 질환군(동반질환 포함) vs 정상군(HTN·DM·CKD 모두 없음)만 사용하고, 다른 질환이 있는 음성 환자는 제외 (0_data_distribution과 같은 정의). OUT_DIR로 저장 폴더 지정(그 아래 스크립트 이름 폴더에 저장)
import os, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as t_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from _sens import sens_spec, with_ss, train_cutoff

ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / os.environ.get("OUT_DIR", "outputs/261012") / Path(__file__).stem; OUT.mkdir(parents=True, exist_ok=True)
ALL_DIS, REPS = ["HTN", "DM", "CKD"], 3
DIS = ALL_DIS
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center",
      "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 clinic4 + 질환 라벨 + landmark 값({조직}@{landmark}) + liver_dome~pubis 구간평균({조직}_mean) (곡선·landmark가 모두 있는 환자만, sex는 M=1)
def load(cohort):
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    m = pd.read_excel(path, sheet_name="patient_metadata")
    m = m[m.PatientAge >= 20]  # 20세 미만 제외
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {t: pd.read_excel(path, sheet_name=SHEET[t]).set_index("PatientID").filter(regex=rf"^{t.lower() if t == 'AEC' else t}_\d+$") for t in MEAS}
    ids = set(lm.index).intersection(*[set(c.index) for c in cur.values()])
    m = m[m.PatientID.isin(ids)].dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + ALL_DIS)
    m = m.reset_index(drop=True)
    sl = lm.loc[m.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
    for t in MEAS:
        v = np.nan_to_num(cur[t].loc[m.PatientID].to_numpy(float))
        for j, a in enumerate(LM):
            m[f"{t}@{a}"] = v[np.arange(len(v)), sl[:, j] - 1]
        m[f"{t}_mean"] = [v[i, sl[i, 0] - 1:sl[i, -1]].mean() for i in range(len(v))]
    m["sex"] = (m.PatientSex.astype(str).str.upper() == "M").astype(float); m["cohort"] = cohort
    return m


# DeLong 짝지은 AUC 검정 (score_b - score_a의 AUC 차이, 양측 p)
def delong_paired_auc_test(y, sa, sb):
    y = np.asarray(y); pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    S = np.vstack([np.asarray(sa)[np.r_[np.where(pos)[0], np.where(~pos)[0]]], np.asarray(sb)[np.r_[np.where(pos)[0], np.where(~pos)[0]]]])
    tx, ty, tz = (np.vstack([rankdata(r) for r in M]) for M in (S[:, :n1], S[:, n1:], S))
    auc = tz[:, :n1].sum(1) / (n1 * n0) - (n1 + 1) / (2 * n0)
    cov = np.cov((tz[:, :n1] - tx) / n0) / n1 + np.cov(1 - (tz[:, n1:] - ty) / n1) / n0
    diff, var = auc[1] - auc[0], cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return dict(diff=diff, p_value=2 * norm.sf(abs(diff) / np.sqrt(var)) if var > 0 else np.nan)


# LR의 C를 학습 fold 내부 5-fold AUC로 고르는 모델 (평가 fold는 선택에 쓰지 않음)
def mk():
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return GridSearchCV(lr, {"logisticregression__C": np.logspace(-3, 1, 9)}, cv=5, scoring="roc_auc", n_jobs=-1)


# 풀링 5-fold CV를 REPS회 반복한 fold별 AUC (분할은 feature set과 상관없이 같아 fold가 짝지어짐)
def cv(X, y, strat):
    aucs, oofs = [], []
    for k in range(REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], oof[te]))
        oofs.append(oof)
    return with_ss(aucs, y, oofs)


# Nadeau-Bengio 보정 paired t: fold 간 학습셋 중복으로 분산이 작게 나오는 것을 보정 (시험/학습 = 1/4)
def nb_test(diff):
    n, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else 2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / n + .25) * v), n - 1)


# feature set별 포함 근거 ([관례]/[문헌 개념]/[해부학]/[총량 근사]/[탐색]/[상한] 표기, 출처는 일반 개념이며 인용 확인 필요)
RATIONALE = {"clinic4": "기준: 임상 변수 4개",
             "+VAT/SAT (12곳)": "[문헌 개념] 내장/피하 지방 비는 대사 위험 지표로 널리 쓰임",
             "+TAMA/BMI (12곳)": "[문헌 개념] 체격으로 보정한 근육량 (sarcopenia의 ALM/BMI와 같은 개념)",
             "+VAT/TAMA (12곳)": "[문헌 개념] 내장지방 대 근육 비: sarcopenic obesity 평가 개념",
             "+조직 간 비율 전체 (36)": "위 세 비율 전부 (landmark 12곳)"}


def main():
    A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    bmi = A.Weight / (A.Height / 100) ** 2
    rt = {"VAT/SAT": lambda a: A[f"VAT@{a}"] / (A[f"SAT@{a}"] + 1e-6), "TAMA/BMI": lambda a: A[f"TAMA@{a}"] / bmi, "VAT/TAMA": lambda a: A[f"VAT@{a}"] / (A[f"TAMA@{a}"] + 1e-6)}
    sub = lambda ks: {f"{k}@{a}": rt[k](a) for k in ks for a in LM}  # 12 landmark마다 비율
    EXTRA = {"clinic4": {}} | {f"+{k} (12곳)": sub([k]) for k in rt} | {"+조직 간 비율 전체 (36)": sub(rt)}
    base = pd.DataFrame({"sex": A.sex, "PatientAge": A.PatientAge, "Height": A.Height, "Weight": A.Weight})
    SETS, FEATS = {}, {}
    for n, ex in EXTRA.items():  # feature set = clinic4 + 추가 feature(열 이름이 곧 feature 이름)
        df = pd.concat([base, pd.DataFrame(ex, index=A.index)], axis=1); SETS[n], FEATS[n] = df.to_numpy(float), list(df.columns)
    pd.DataFrame([(n, i + 1, f) for n, fs in FEATS.items() for i, f in enumerate(fs)], columns=["feature_set", "no", "feature"]).assign(rationale=lambda d: d.feature_set.map(RATIONALE)).to_excel(OUT / "clinic4_step4_features.xlsx", index=False)
    names = list(SETS)
    strat_c = (A.cohort == "sinchon").to_numpy()
    keep = {d: ((A[d] == 1) | (A[ALL_DIS].sum(axis=1) == 0)).to_numpy() for d in DIS}  # 질환별 사용 환자: 질환군 + 무질환 정상군

    subsets = {"pooled": np.ones(len(A), bool), "gangnam": (A.cohort == "gangnam").to_numpy(), "sinchon": strat_c}
    rows = []
    for ev, m in subsets.items():
        for d in DIS:
            mm = m & keep[d]; y = A[d].to_numpy(int)[mm]
            r = {n: cv(X[mm], y, y * 2 + strat_c[mm]) for n, X in SETS.items()}
            for k, (n, X) in enumerate(SETS.items()):
                diff = r[n] - r["clinic4"]
                rows.append(dict(eval=f"{ev} 5-fold CV", disease=d, feature_set=n, n=len(y), n_pos=y.sum(), n_feat=X.shape[1], AUC=r[n].mean(), Sens=r[n].ss[0], Spec=r[n].ss[1], AUC_sd=r[n].std(ddof=1), dAUC=diff.mean(),
                                 p_cv=np.nan if n == "clinic4" else nb_test(diff),
                                 dAUC_prev=(r[n] - r[names[k - 1]]).mean() if k > 1 else np.nan, p_prev=nb_test(r[n] - r[names[k - 1]]) if k > 1 else np.nan))
    res = pd.DataFrame(rows)
    res.to_excel(OUT / "clinic4_step4.xlsx", index=False)

    ext = []
    for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        tr, te = (A.cohort == a).to_numpy(), (A.cohort == b).to_numpy()
        for d in DIS:
            y, tr_, te_ = A[d].to_numpy(int), tr & keep[d], te & keep[d]
            p = {n: mk().fit(X[tr_], y[tr_]).predict_proba(X[te_])[:, 1] for n, X in SETS.items()}
            thr = {n: train_cutoff(mk, X[tr_], y[tr_]) for n, X in SETS.items()}
            for n in SETS:
                t = delong_paired_auc_test(y[te_], p["clinic4"], p[n]) if n != "clinic4" else dict(diff=0, p_value=np.nan)
                ext.append(dict(train=a, test=b, disease=d, feature_set=n, AUC=roc_auc_score(y[te_], p[n]), **dict(zip(["Sens", "Spec"], sens_spec(y[te_], p[n], thr[n]))), dAUC=t["diff"], p_delong=t["p_value"]))
    ext = pd.DataFrame(ext)
    ext.to_excel(OUT / "clinic4_step4_external.xlsx", index=False)

    print(res.pivot_table(index=["eval", "feature_set"], columns="disease", values="AUC", sort=False)[DIS].round(3))
    print(ext.pivot_table(index=["train", "feature_set"], columns="disease", values="AUC")[DIS].round(3))
    # 평가 방식별 패널: 질환별로 feature set 막대 (CV는 fold SD 에러바)
    evs = list(dict.fromkeys(res["eval"])) + [f"train {a} -> test {b}" for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]]
    fig, axs = plt.subplots(1, len(evs), figsize=(5 * len(evs), 5), sharey=True)
    for ax, ev in zip(axs, evs):
        if ev.endswith("CV"):
            g = res[res["eval"] == ev]; auc = g.pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[names]; sd = g.pivot(index="feature_set", columns="disease", values="AUC_sd")[DIS].loc[names]
            P = g.pivot(index="feature_set", columns="disease", values="p_cv")[DIS].loc[names]
        else:
            a = ev.split()[1]; auc = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[names]; sd = None
            P = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="p_delong")[DIS].loc[names]
        auc.T.plot.bar(ax=ax, yerr=None if sd is None else sd.T, capsize=2, rot=0, legend=False, width=.85)
        for c, n in enumerate(names):  # p<0.05 (vs clinic4) 막대 위에 *
            for i, d in enumerate(DIS):
                if P.at[n, d] < .05:
                    b = ax.patches[c * len(DIS) + i]; ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0 if sd is None else sd.at[n, d]) + .004, "*", ha="center", fontsize=14, fontweight="bold")
        ax.set(ylim=(.5, 1), title=ev, xlabel=""); ax.axhline(.5, color="gray", lw=.8, ls="--")
    axs[0].set_ylabel("AUC")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h + [plt.Line2D([], [], ls="")], l + ["* p<0.05 vs clinic4"], loc="lower center", ncol=len(l) + 1, frameon=False)
    plt.tight_layout(rect=(0, .08, 1, 1)); plt.savefig(OUT / "clinic4_step4.png", dpi=300, bbox_inches="tight", facecolor="white")

if __name__ == "__main__":
    main()
