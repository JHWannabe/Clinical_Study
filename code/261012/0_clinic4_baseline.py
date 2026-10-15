# clinic4(sex/age/height/weight)만으로 HTN/DM/CKD를 분류하는 baseline AUC: 코호트별·풀링 5-fold CV(5회 반복, fold별 AUC 평균±SD) + 코호트 간 외부검증
# 질환마다 질환군(동반질환 포함) vs 정상군(HTN·DM·CKD 모두 없음)만 사용하고, 다른 질환이 있는 음성 환자는 제외 (0_data_distribution과 같은 정의). OUT_DIR로 저장 폴더 지정
import os, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / os.environ.get("OUT_DIR", "outputs/261012/0_baseline"); OUT.mkdir(parents=True, exist_ok=True)
ALL_DIS, REPS = ["HTN", "DM", "CKD"], 3
DIS = ALL_DIS
FEAT = ["sex", "PatientAge", "Height", "Weight"]
# LR의 C를 학습 fold 내부 5-fold AUC로 고르는 모델 (평가 fold는 선택에 쓰지 않음)
mk = lambda: GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": np.logspace(-3, 1, 9)}, cv=5, scoring="roc_auc", n_jobs=-1)


# 질환 d의 사용 환자: 질환군 + 무질환 정상군
use = lambda df, d: df[(df[d] == 1) | (df[ALL_DIS].sum(axis=1) == 0)]


# 코호트의 clinic4 + 질환 라벨 (결측 제외, sex는 M=1)
def load(cohort):
    m = pd.read_excel(ROOT / f"data/{cohort}_landmark_filtered.xlsx", sheet_name="patient_metadata").query("PatientAge >= 20").dropna(subset=FEAT[1:] + ["PatientSex"] + ALL_DIS)
    m = m.reset_index(drop=True)
    m["sex"] = (m.PatientSex.astype(str).str.upper() == "M").astype(float); m["cohort"] = cohort
    return m


# 민감도 90% 이상을 만족하는 가장 큰 cutoff
def sens90(y, p):
    fpr, tpr, th = roc_curve(y, p); return th[np.argmax(tpr >= .9)]


# cutoff t에서의 (민감도, 특이도)
def sens_spec(y, p, t):
    q = p >= t; return q[y == 1].mean(), (~q[y == 0]).mean()


# 5-fold CV를 REPS회 반복: fold별 AUC 평균·SD와 OOF Sens90 cutoff의 민감도·특이도 (층화: 라벨 x 코호트)
def cv_auc(X, y, strat):
    aucs, ss = [], []
    for k in range(REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], oof[te]))
        ss.append(sens_spec(y, oof, sens90(y, oof)))
    return np.mean(aucs), np.std(aucs, ddof=1), *np.mean(ss, axis=0), *np.std(ss, axis=0, ddof=1)  # Sens·Spec의 SD는 반복(REPS회) 간


# 학습 코호트 OOF로 cutoff를 정하고, 학습 코호트 전체로 재학습한 모델을 시험 코호트에 적용
def ext_eval(Xtr, ytr, Xte, yte):
    oof = np.zeros(len(ytr))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
        oof[te] = mk().fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[te])[:, 1]
    p = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), *sens_spec(yte, p, sens90(ytr, oof))


# AUC와 동일한 5-fold CV x REPS의 train fold 적합 계수 평균±SD와 OR (SD당: 표준화 계수, 단위당: 원척도 환산)
def save_coef(data):
    rows = []
    for c, df in data.items():
        for d in DIS:
            dd = use(df, d); X, y = dd[FEAT].to_numpy(float), dd[d].to_numpy(int); B, S = [], []
            for k in range(REPS):
                for tr, _ in StratifiedKFold(5, shuffle=True, random_state=k).split(X, y * 2 + (dd.cohort == "sinchon")):
                    m = mk().fit(X[tr], y[tr]).best_estimator_; B.append(m[-1].coef_[0]); S.append(m[0].scale_)
            B, S = np.array(B), np.array(S); bu = B / S
            for i, f in enumerate(FEAT):
                rows.append(dict(cohort=c, disease=d, feature=f, coef_per_SD=B[:, i].mean(), coef_sd=B[:, i].std(ddof=1), OR_per_SD=np.exp(B[:, i].mean()),
                                 OR_lo=np.exp(np.percentile(B[:, i], 2.5)), OR_hi=np.exp(np.percentile(B[:, i], 97.5)),
                                 coef_per_unit=bu[:, i].mean(), OR_per_unit=np.exp(bu[:, i].mean())))
    co = pd.DataFrame(rows)
    co.to_excel(OUT / "clinic4_baseline_coef_OR.xlsx", index=False)
    fig, axs = plt.subplots(1, len(DIS), figsize=(5 * len(DIS), 4), sharey=False, squeeze=False)
    for ax, d in zip(axs[0], DIS):
        t = co[co.disease == d]
        for j, (c, mk_) in enumerate(zip(["gangnam", "sinchon", "pooled"], "osD")):
            r = t[t.cohort == c].set_index("feature").loc[FEAT]
            ax.errorbar(r.OR_per_SD, np.arange(len(r)) + (j - 1) * .2, xerr=[r.OR_per_SD - r.OR_lo, r.OR_hi - r.OR_per_SD], fmt=mk_, ms=7, capsize=3, label=c)
        t = r
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
        ax.axvline(1, color="gray", ls="--", lw=.8)
        ax.set(xscale="log", title=d, xlabel="Odds ratio (per 1 SD)", ylabel="")
        ax.set_xticks(tk := [.5, .7, 1, 1.5, 2, 3]); ax.set_xticklabels(tk); ax.minorticks_off(); ax.set_xlim(.45, 3.3)
    h, l = axs[0][0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    plt.tight_layout(rect=(0, .07, 1, 1)); plt.savefig(OUT / "clinic4_baseline_coef_OR.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


def main():
    data = {c: load(c) for c in ["gangnam", "sinchon"]}
    data["pooled"] = pd.concat(data.values(), ignore_index=True)
    rows = []
    for c, df in data.items():
        for d in DIS:
            dd = use(df, d); y = dd[d].to_numpy(int)
            auc, sd, sens, spec, sens_sd, spec_sd = cv_auc(dd[FEAT].to_numpy(float), y, y * 2 + (dd.cohort == "sinchon"))
            rows.append(dict(eval=f"{c} 5-fold CV", disease=d, model="LR", n=len(y), n_pos=y.sum(), AUC=auc, AUC_sd=sd, Sens=sens, Spec=spec, Sens_sd=sens_sd, Spec_sd=spec_sd))
    for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        for d in DIS:
            tr, te = use(data[a], d), use(data[b], d)
            auc, sens, spec = ext_eval(tr[FEAT].to_numpy(float), tr[d].to_numpy(int), te[FEAT].to_numpy(float), te[d].to_numpy(int))
            rows.append(dict(eval=f"train {a} -> test {b}", disease=d, model="LR", n=len(te), n_pos=te[d].sum(), AUC=auc, AUC_sd=np.nan, Sens=sens, Spec=spec))
    res = pd.DataFrame(rows)
    res.to_excel(OUT / "clinic4_baseline.xlsx", index=False)
    auc = res.pivot_table(index="eval", columns="disease", values="AUC", sort=False)[DIS]
    sd = res.pivot_table(index="eval", columns="disease", values="AUC_sd", sort=False)[DIS]
    print(auc.round(3))
    ax = auc.T.plot.bar(yerr=sd.T, capsize=3, figsize=(10, 5), rot=0)
    ax.set(ylim=(.5, 1), ylabel="AUC", xlabel="", title="clinic4 baseline AUC")
    ax.axhline(.5, color="gray", lw=.8, ls="--")
    plt.tight_layout(); plt.savefig(OUT / "clinic4_baseline.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    save_coef(data)


if __name__ == "__main__":
    main()
