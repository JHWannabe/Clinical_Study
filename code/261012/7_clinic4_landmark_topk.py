# 7단계: landmark 12곳 전체 모델의 계수 크기 순으로 landmark를 정렬해 상위 1~6곳만 clinic4에 추가했을 때 AUC 변화. 순위는 학습 fold 안에서만 계산(누수 방지). clinic4·직전 집합 대비 5-fold CV(풀링/강남/신촌, 3회 반복, Nadeau-Bengio 보정 t)와 강남↔신촌 외부검증(DeLong)
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
KS = [1, 2, 3, 4, 5, 6]  # 상위 몇 곳까지 쓸지
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 clinic4 + 질환 라벨 + landmark 값({조직}@{landmark}) (곡선·landmark가 모두 있는 환자만, sex는 M=1)
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


# 열 순서가 [clinic4 4개, landmark별 (VAT, SAT, TAMA, AEC)]인 행렬에서, 학습 데이터로 12곳 전체 LR을 적합해 landmark를 |표준화 계수|(조직 4개 합) 순으로 정렬하고 상위 k곳만으로 LR을 다시 적합
class TopK:
    cache = {}  # 같은 학습 데이터면 k가 달라도 순위를 한 번만 계산

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        key = hash((X.tobytes(), y.tobytes()))
        if key not in TopK.cache:
            b = mk().fit(X, y).best_estimator_[-1].coef_[0][4:].reshape(len(LM), len(MEAS))
            TopK.cache[key] = np.argsort(-np.abs(b).sum(axis=1))
        self.order_ = TopK.cache[key]
        self.cols_ = list(range(4)) + [4 + len(MEAS) * j + i for j in self.order_[:self.k] for i in range(len(MEAS))]
        self.model_ = mk().fit(X[:, self.cols_], y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[:, self.cols_])


# 5-fold CV를 REPS회 반복한 fold별 AUC와 fold별 landmark 순서(TopK일 때) (분할은 feature set과 상관없이 같아 fold가 짝지어짐)
def cv(X, y, strat, make):
    aucs, orders, oofs = [], [], []
    for k in range(REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            m = make().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], oof[te]))
            if hasattr(m, "order_"): orders.append(m.order_)
        oofs.append(oof)
    return with_ss(aucs, y, oofs), np.array(orders)


# Nadeau-Bengio 보정 paired t: fold 간 학습셋 중복으로 분산이 작게 나오는 것을 보정 (시험/학습 = 1/4)
def nb_test(diff):
    n, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else 2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / n + .25) * v), n - 1)


# feature set별 포함 근거
SEL = "[탐색] 12곳 전체 모델의 계수 크기(조직 4개의 |표준화 계수| 합)로 학습 fold 안에서 landmark를 순위화해 상위 {k}곳만 사용: 필요한 landmark 수를 줄여도 성능이 유지되는지 확인 (순위는 학습 fold 안에서만 계산해 누수 방지)"
RATIONALE = {"clinic4": "기준: 임상 변수 4개"} | {f"+상위 {k}곳": SEL.format(k=k) for k in KS} | {"+전체 12곳": "[상한] 12곳 전부를 썼을 때의 정보량 (참고용)"}


def main():
    A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    full = pd.DataFrame({"sex": A.sex, "PatientAge": A.PatientAge, "Height": A.Height, "Weight": A.Weight} | {f"{t}@{a}": A[f"{t}@{a}"] for a in LM for t in MEAS}).to_numpy(float)
    SETS = {"clinic4": full[:, :4]} | {f"+상위 {k}곳": full for k in KS} | {"+전체 12곳": full}  # 상위 k곳 집합은 전체 행렬을 받아 모델 안에서 고른다
    MAKE = {"clinic4": mk} | {f"+상위 {k}곳": (lambda k=k: TopK(k)) for k in KS} | {"+전체 12곳": mk}
    NF = {"clinic4": 4} | {f"+상위 {k}곳": 4 + 4 * k for k in KS} | {"+전체 12곳": 4 + 4 * len(LM)}
    pd.DataFrame([(n, NF[n], RATIONALE[n]) for n in SETS], columns=["feature_set", "n_feat", "rationale"]).to_excel(OUT / "clinic4_step7_features.xlsx", index=False)
    names = list(SETS)
    strat_c = (A.cohort == "sinchon").to_numpy()
    keep = {d: ((A[d] == 1) | (A[ALL_DIS].sum(axis=1) == 0)).to_numpy() for d in DIS}  # 질환별 사용 환자: 질환군 + 무질환 정상군

    subsets = {"pooled": np.ones(len(A), bool), "gangnam": (A.cohort == "gangnam").to_numpy(), "sinchon": strat_c}
    rows, sel = [], []
    for ev, m in subsets.items():
        for d in DIS:
            mm = m & keep[d]; y = A[d].to_numpy(int)[mm]
            cr = {n: cv(X[mm], y, y * 2 + strat_c[mm], MAKE[n]) for n, X in SETS.items()}; r = {n: v[0] for n, v in cr.items()}
            o = cr["+상위 1곳"][1]  # fold별 landmark 순서 -> landmark별 평균 순위·상위 1/3 포함 빈도
            rank = np.argsort(o, axis=1)
            sel += [dict(eval=f"{ev} 5-fold CV", disease=d, landmark=a.replace("_center", ""), mean_rank=rank[:, j].mean() + 1, top1_freq=(rank[:, j] == 0).mean(), top3_freq=(rank[:, j] < 3).mean()) for j, a in enumerate(LM)]
            for k, (n, X) in enumerate(SETS.items()):
                diff = r[n] - r["clinic4"]
                rows.append(dict(eval=f"{ev} 5-fold CV", disease=d, feature_set=n, n=len(y), n_pos=y.sum(), n_feat=NF[n], AUC=r[n].mean(), Sens=r[n].ss[0], Spec=r[n].ss[1], AUC_sd=r[n].std(ddof=1), dAUC=diff.mean(),
                                 p_cv=np.nan if n == "clinic4" else nb_test(diff),
                                 dAUC_prev=(r[n] - r[names[k - 1]]).mean() if k > 1 else np.nan, p_prev=nb_test(r[n] - r[names[k - 1]]) if k > 1 else np.nan))
    res = pd.DataFrame(rows)
    res.to_excel(OUT / "clinic4_step7.xlsx", index=False)
    pd.DataFrame(sel).to_excel(OUT / "clinic4_step7_selection.xlsx", index=False)

    ext = []
    for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        tr, te = (A.cohort == a).to_numpy(), (A.cohort == b).to_numpy()
        for d in DIS:
            y, tr_, te_ = A[d].to_numpy(int), tr & keep[d], te & keep[d]
            p = {n: MAKE[n]().fit(X[tr_], y[tr_]).predict_proba(X[te_])[:, 1] for n, X in SETS.items()}
            thr = {n: train_cutoff(MAKE[n], X[tr_], y[tr_]) for n, X in SETS.items()}
            for n in SETS:
                t = delong_paired_auc_test(y[te_], p["clinic4"], p[n]) if n != "clinic4" else dict(diff=0, p_value=np.nan)
                ext.append(dict(train=a, test=b, disease=d, feature_set=n, AUC=roc_auc_score(y[te_], p[n]), **dict(zip(["Sens", "Spec"], sens_spec(y[te_], p[n], thr[n]))), dAUC=t["diff"], p_delong=t["p_value"]))
    ext = pd.DataFrame(ext)
    ext.to_excel(OUT / "clinic4_step7_external.xlsx", index=False)

    print(res.pivot_table(index=["eval", "feature_set"], columns="disease", values="AUC", sort=False)[DIS].round(3))
    print(ext.pivot_table(index=["train", "feature_set"], columns="disease", values="AUC")[DIS].round(3))
    # 평가 방식별 패널: 질환별로 feature set 막대 (CV는 fold SD 에러바)
    evs = list(dict.fromkeys(res["eval"])) + [f"train {a} -> test {b}" for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]]
    fig, axs = plt.subplots(2, 3, figsize=(24, 12), sharey=True); axs = axs.ravel(); axs[-1].axis("off")  # 마지막 칸은 범례 자리
    n = len(names) - 1  # clinic4는 연회색, 나머지는 turbo 색을 흰색과 섞은 파스텔톤으로,  서로 멀리 떨어진 순서로 배정해 이웃한 막대끼리도 구분되게
    stride = next(k for k in range(max(n // 3, 1), n + 1) if np.gcd(k, n) == 1)
    pastel = lambda c: 0.72 * np.array(c) + 0.28  # 무지개색에 흰색을 약간 섞어 선명함은 유지하고 눈부심만 줄임
    colors = ["#a8a8a8"] + [pastel(plt.cm.turbo(np.linspace(.05, .95, n))[(i * stride) % n]) for i in range(n)]
    for ax, ev in zip(axs, evs):
        if ev.endswith("CV"):
            g = res[res["eval"] == ev]; auc = g.pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[names]; sd = g.pivot(index="feature_set", columns="disease", values="AUC_sd")[DIS].loc[names]
            P = g.pivot(index="feature_set", columns="disease", values="p_cv")[DIS].loc[names]
        else:
            a = ev.split()[1]; auc = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="AUC")[DIS].loc[names]; sd = None
            P = ext[ext.train == a].pivot(index="feature_set", columns="disease", values="p_delong")[DIS].loc[names]
        auc.T.plot.bar(ax=ax, yerr=None if sd is None else sd.T, capsize=2, rot=0, legend=False, width=.85, color=colors, ecolor="#777777")
        for c, n in enumerate(names):  # p<0.05 (vs clinic4) 막대 위에 *
            for i, d in enumerate(DIS):
                if P.at[n, d] < .05:
                    b = ax.patches[c * len(DIS) + i]; ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0 if sd is None else sd.at[n, d]) + .004, "*", ha="center", fontsize=14, fontweight="bold")
        ax.set(ylim=(.5, 1), title=ev, xlabel=""); ax.axhline(.5, color="gray", lw=.8, ls="--")
    for ax in axs[[0, 3]]: ax.set_ylabel("AUC")
    h, l = axs[0].get_legend_handles_labels()
    axs[-1].legend(h + [plt.Line2D([], [], ls="")], l + ["* p<0.05 vs clinic4"], loc="center", ncol=2, frameon=False, fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "clinic4_step7.png", dpi=300, bbox_inches="tight", facecolor="white")

if __name__ == "__main__":
    main()
