# 1단계: clinic4에 landmark 12곳 값의 합(조직별)을 하나씩 추가했을 때 AUC 변화. case = +VAT 합(기본), +SAT 합, +TAMA 합, +AEC 합
# 평가: 풀링 5-fold CV(3회 반복, Nadeau-Bengio 보정 t)와 강남↔신촌 외부검증(DeLong). 표/그래프에 baseline(clinic4)을 함께 둠
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
SHEET = {m: "aec_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 clinic4 + 질환 라벨 + 조직별 landmark 12곳 값의 합({m}_sum) (곡선·landmark가 모두 있는 환자만, sex는 M=1)
def load(cohort):
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    m = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {t: pd.read_excel(path, sheet_name=SHEET[t]).set_index("PatientID").filter(regex=rf"^{t.lower() if t == 'AEC' else t}_\d+$") for t in MEAS}
    ids = set(lm.index).intersection(*[set(c.index) for c in cur.values()])
    m = m[m.PatientID.isin(ids)].dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + ALL_DIS)
    m = m.reset_index(drop=True)
    sl = lm.loc[m.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
    for t in MEAS:
        v = np.nan_to_num(cur[t].loc[m.PatientID].to_numpy(float))
        m[f"{t}_sum"] = v[np.arange(len(m))[:, None], sl - 1].sum(axis=1)
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


# 풀링 5-fold CV를 REPS회 반복한 fold별 AUC와 표준화 계수 (분할은 feature set과 상관없이 같아 fold가 짝지어짐)
def cv(X, y, strat):
    aucs, B, oofs = [], [], []
    for k in range(REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            m = mk().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], oof[te]))
            B.append(m.best_estimator_[-1].coef_[0])
        oofs.append(oof)
    return with_ss(aucs, y, oofs), np.array(B)


# Nadeau-Bengio 보정 paired t: fold 간 학습셋 중복으로 분산이 작게 나오는 것을 보정 (시험/학습 = 1/4)
def nb_test(diff):
    n, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else 2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / n + .25) * v), n - 1)


# 질환별 패널에 feature별 OR(SD당)의 CV fold 평균과 2.5~97.5% 범위 (clinic4 변수는 clinic4 모델, 합 변수는 해당 +case 모델)
def save_coef_png(coef):
    feats = ["sex", "PatientAge", "Height", "Weight"] + [f"{t} 합" for t in MEAS]
    labels = feats + [f"{t} 합 (4조직 동시)" for t in MEAS]
    fig, axs = plt.subplots(1, len(DIS), figsize=(5 * len(DIS), 10), sharey=False, squeeze=False)
    for ax, d in zip(axs[0], DIS):
        for j, (c, mk_) in enumerate(zip(["gangnam", "sinchon", "pooled"], "osD")):
            g = coef[(coef.disease == d) & (coef["eval"] == c)]
            r = pd.concat([g[(g.feature_set == ("clinic4" if i < 4 else f"+{f}")) & (g.feature == f)] for i, f in enumerate(feats)] + [g[(g.feature_set == "+4조직 합") & (g.feature == f"{t} 합")] for t in MEAS])
            ax.errorbar(r.OR_per_SD, np.arange(len(r)) + (j - 1) * .2, xerr=[r.OR_per_SD - r.OR_lo, r.OR_hi - r.OR_per_SD], fmt=mk_, ms=7, capsize=3, label=c)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.grid(axis="x", alpha=.3)
        ax.axvline(1, color="gray", ls="--", lw=.8); ax.set(xscale="log", title=d, xlabel="Odds ratio (per 1 SD)")
        ax.set_xticks(tk := [.5, .7, 1, 1.5, 2, 3]); ax.set_xticklabels(tk); ax.minorticks_off(); ax.set_xlim(.45, 3.3)
    h, l = axs[0][0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
    plt.tight_layout(rect=(0, .07, 1, 1)); plt.savefig(OUT / "clinic4_step1_coef_OR.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


# feature set별 포함 근거 ([관례]/[문헌 개념]/[해부학]/[총량 근사]/[탐색]/[상한] 표기, 출처는 일반 개념이며 인용 확인 필요)
RATIONALE = {"clinic4": "기준: 임상 변수 4개 (sex, age, height, weight)",
             "+VAT 합": "[총량 근사] landmark 12곳 값의 합 = 조직 총량의 대리 변수 (내장지방)", "+SAT 합": "[총량 근사] landmark 12곳 값의 합 (피하지방)",
             "+TAMA 합": "[총량 근사] landmark 12곳 값의 합 (근육)", "+AEC 합": "[총량 근사] landmark 12곳 값의 합 (AEC)",
             "+4조직 합": "[총량 근사] 네 조직의 총량을 동시에 반영 (조직 간 상관으로 계수가 불안정할 수 있음)"}


def main():
    A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight])
    SETS = {"clinic4": C4} | {f"+{t} 합": np.column_stack([C4, A[f"{t}_sum"]]) for t in MEAS}
    SETS["+4조직 합"] = np.column_stack([C4] + [A[f"{t}_sum"] for t in MEAS])
    FEATS = {n: ["sex", "PatientAge", "Height", "Weight"] + ([f"{t} 합" for t in MEAS] if n == "+4조직 합" else [] if n == "clinic4" else [n[1:]]) for n in SETS}
    pd.DataFrame([(n, i + 1, f) for n, fs in FEATS.items() for i, f in enumerate(fs)], columns=["feature_set", "no", "feature"]).assign(rationale=lambda d: d.feature_set.map(RATIONALE)).to_excel(OUT / "clinic4_step1_features.xlsx", index=False)
    strat_c = (A.cohort == "sinchon").to_numpy()
    keep = {d: ((A[d] == 1) | (A[ALL_DIS].sum(axis=1) == 0)).to_numpy() for d in DIS}  # 질환별 사용 환자: 질환군 + 무질환 정상군
    subsets = {"pooled": np.ones(len(A), bool), "gangnam": (A.cohort == "gangnam").to_numpy(), "sinchon": strat_c}

    rows, coef = [], []
    for ev, m in subsets.items():
        for d in DIS:
            mm = m & keep[d]; y = A[d].to_numpy(int)[mm]
            cr = {n: cv(X[mm], y, y * 2 + strat_c[mm]) for n, X in SETS.items()}; r = {n: v[0] for n, v in cr.items()}
            for n, X in SETS.items():
                for i, f in enumerate(FEATS[n]):
                    b = cr[n][1][:, i]
                    coef.append(dict(eval=ev, disease=d, feature_set=n, feature=f, coef_per_SD=b.mean(), coef_sd=b.std(ddof=1), OR_per_SD=np.exp(b.mean()),
                                     OR_lo=np.exp(np.percentile(b, 2.5)), OR_hi=np.exp(np.percentile(b, 97.5))))
                diff = r[n] - r["clinic4"]
                rows.append(dict(eval=f"{ev} 5-fold CV", disease=d, feature_set=n, n=len(y), n_pos=y.sum(), n_feat=X.shape[1], AUC=r[n].mean(), Sens=r[n].ss[0], Spec=r[n].ss[1], AUC_sd=r[n].std(ddof=1),
                                 dAUC=diff.mean(), p_cv=np.nan if n == "clinic4" else nb_test(diff)))
    FL = {n: ", ".join(fs) for n, fs in FEATS.items()}  # feature set별 feature 목록 문자열
    res, coef = pd.DataFrame(rows), pd.DataFrame(coef)
    res["features"] = res.feature_set.map(FL)
    coef.to_excel(OUT / "clinic4_step1_coef_OR.xlsx", index=False)
    save_coef_png(coef)

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
    ext["features"] = ext.feature_set.map(FL)

    # baseline(clinic4)과 각 case를 평가 방식별로 나란히: 행 = 평가 방식 x 질환 x case
    cmp_ = []
    for d in DIS:
        for n in list(SETS)[1:]:
            for ev, g in res[res.disease == d].groupby("eval", sort=False):
                g = g.set_index("feature_set")
                cmp_.append(dict(eval=ev, disease=d, case=n, baseline_AUC=g.AUC["clinic4"], baseline_sd=g.AUC_sd["clinic4"], case_AUC=g.AUC[n], case_sd=g.AUC_sd[n], dAUC=g.dAUC[n], p=g.p_cv[n]))
            for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
                e = ext[(ext.disease == d) & (ext.train == a)].set_index("feature_set")
                cmp_.append(dict(eval=f"train {a} -> test {b}", disease=d, case=n, baseline_AUC=e.AUC["clinic4"], baseline_sd=np.nan, case_AUC=e.AUC[n], case_sd=np.nan, dAUC=e.dAUC[n], p=e.p_delong[n]))
    cmp_ = pd.DataFrame(cmp_)
    cmp_["features"] = cmp_.case.map(FL)
    # 행 = case, 열 = 평가 방식 x 질환 (ΔAUC / p)
    order = list(dict.fromkeys(cmp_["eval"]))
    pv = lambda v: cmp_.pivot(index="case", columns=["eval", "disease"], values=v).reindex(index=list(SETS)[1:], columns=pd.MultiIndex.from_product([order, DIS]))
    dA, pP = pv("dAUC"), pv("p")
    with pd.ExcelWriter(OUT / "clinic4_step1.xlsx") as xw:
        dA.to_excel(xw, sheet_name="dAUC")
        pP.to_excel(xw, sheet_name="p")
        cmp_.sort_values("disease", kind="stable").to_excel(xw, sheet_name="compare", index=False)
        res.to_excel(xw, sheet_name="cv", index=False)
        ext.to_excel(xw, sheet_name="external", index=False)
        ws = xw.sheets["dAUC"]  # 0을 흰색, 증가 빨강 / 감소 파랑
        ws.conditional_format(3, 1, 2 + len(dA), dA.shape[1], {"type": "3_color_scale", "min_type": "num", "min_value": -.03, "mid_type": "num", "mid_value": 0,
                                                                "max_type": "num", "max_value": .03, "min_color": "#5B8FD9", "mid_color": "#FFFFFF", "max_color": "#E06666"})
    print(dA.round(3).to_string())

    # 평가 방식별 패널: 질환별로 feature set 막대 (CV는 fold SD 에러바)
    evs = list(dict.fromkeys(res["eval"])) + [f"train {a} -> test {b}" for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]]
    fig, axs = plt.subplots(1, len(evs), figsize=(5 * len(evs), 5), sharey=True)
    for ax, ev in zip(axs, evs):
        if ev.endswith("CV"):
            g = res[res["eval"] == ev]; auc = g.pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[list(SETS)]; sd = g.pivot(index="feature_set", columns="disease", values="AUC_sd")[DIS].loc[list(SETS)]
            P = g.pivot(index="feature_set", columns="disease", values="p_cv")[DIS].loc[list(SETS)]
        else:
            a = ev.split()[1]; auc = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[list(SETS)]; sd = None
            P = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="p_delong")[DIS].loc[list(SETS)]
        auc.T.plot.bar(ax=ax, yerr=None if sd is None else sd.T, capsize=2, rot=0, legend=False, width=.85)
        for c, n in enumerate(list(SETS)):  # p<0.05 (vs clinic4) 막대 위에 *
            for i, d in enumerate(DIS):
                if P.at[n, d] < .05:
                    b = ax.patches[c * len(DIS) + i]; ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0 if sd is None else sd.at[n, d]) + .004, "*", ha="center", fontsize=14, fontweight="bold")
        ax.set(ylim=(.5, 1), title=ev, xlabel=""); ax.axhline(.5, color="gray", lw=.8, ls="--")
    axs[0].set_ylabel("AUC")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h + [plt.Line2D([], [], ls="")], l + ["* p<0.05 vs clinic4"], loc="lower center", ncol=len(l) + 1, frameon=False)
    plt.tight_layout(rect=(0, .08, 1, 1)); plt.savefig(OUT / "clinic4_step1.png", dpi=300, bbox_inches="tight", facecolor="white")

if __name__ == "__main__":
    main()
