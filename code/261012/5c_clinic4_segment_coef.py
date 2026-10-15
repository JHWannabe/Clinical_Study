# best model(clinic4 + 4종 구간 평균·합산, 36개)의 LR 계수: 풀링 5-fold CV(REPS회) train fold마다 학습한 모델의 표준화 계수(OR per SD) 평균과 95% 범위
import importlib.util, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
sp = importlib.util.spec_from_file_location("seg5", HERE / "5_clinic4_segment_stats.py"); m5 = importlib.util.module_from_spec(sp); sp.loader.exec_module(m5)
OUT = HERE.parents[1] / "outputs/261012/5c_clinic4_segment_coef"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    A = pd.concat([m5.load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    cols = ["sex", "PatientAge", "Height", "Weight"] + [f"{t}_{s}{k}" for t in m5.MEAS for s in ("mean", "sum") for k in range(4)]
    X = A[cols].to_numpy(float); strat_c = (A.cohort == "sinchon").to_numpy(); rows = []
    for d in m5.DIS:
        mm = ((A[d] == 1) | (A[m5.ALL_DIS].sum(axis=1) == 0)).to_numpy(); y = A[d].to_numpy(int)[mm]; Xd = X[mm]; coefs = []
        for k in range(m5.REPS):
            for tr, _ in StratifiedKFold(5, shuffle=True, random_state=k).split(Xd, y * 2 + strat_c[mm]):
                coefs.append(m5.mk().fit(Xd[tr], y[tr]).best_estimator_[-1].coef_[0])
        c = np.array(coefs)  # (REPS*5, 특성 수), 로그 OR per SD
        for j, f in enumerate(cols):
            rows.append(dict(disease=d, feature=f, coef=c[:, j].mean(), OR=np.exp(c[:, j].mean()), OR_lo=np.exp(np.percentile(c[:, j], 2.5)), OR_hi=np.exp(np.percentile(c[:, j], 97.5)), sign_consistent=(np.sign(c[:, j]) == np.sign(c[:, j].mean())).mean()))
    res = pd.DataFrame(rows); res.to_excel(OUT / "clinic4_step5c_coef.xlsx", index=False)
    piv = res.pivot(index="feature", columns="disease", values="coef").loc[cols, m5.DIS]
    fig, ax = plt.subplots(figsize=(6.5, 11)); v = np.abs(piv.to_numpy()).max()
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(m5.DIS); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(3): ax.text(j, i, f"{piv.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="표준화 계수 (log OR per SD)"); fig.tight_layout(); fig.savefig(OUT / "clinic4_step5c_coef.png", dpi=200, facecolor="white")
    print(res.pivot(index="feature", columns="disease", values="OR").loc[cols, m5.DIS].round(2).to_string())
    print(res.pivot(index="feature", columns="disease", values="sign_consistent").loc[cols, m5.DIS].round(2).to_string())


if __name__ == "__main__":
    main()
