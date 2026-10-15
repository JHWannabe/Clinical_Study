# 261023 미팅 피드백 분석: (1) clinic4 vs best(+구간 평균·합산) ROC·balanced accuracy·유병률 (2) 가설 기반 비교(clinic4 / +총량 / +총량+분포) (3) 사례별 p의 BH 보정
# 환자군·모델·CV는 261015/5_clinic4_segment_stats.py와 동일 (풀링 5-fold x3). FINAL_ROOT 환경변수 = final_dataset(조영 필터용)이 있는 폴더
import os, sys, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/261015")); sys.stdout.reconfigure(encoding="utf-8")
import _contrast
_contrast.ROOT = Path(os.environ.get("FINAL_ROOT", ROOT))
st = importlib.import_module("5_clinic4_segment_stats")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
OUT = ROOT / "outputs/261023"; OUT.mkdir(parents=True, exist_ok=True)
DIS, MEAS, LM, SEG = st.DIS, st.MEAS, st.LM, st.SEG
CLIN = ["sex", "PatientAge", "Height", "Weight"]


# 구간 곡선의 SD·첨도(분포 feature)를 st.load 결과에 추가
def load_all():
    out = []
    for c in ["gangnam", "sinchon"]:
        m = st.load(c); path = ROOT / f"data/{c}_landmark_filtered.xlsx"
        lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
        sl = lm.loc[m.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
        for t in MEAS:
            v = np.nan_to_num(pd.read_excel(path, sheet_name=st.SHEET[t]).set_index("PatientID").filter(regex=rf"^{t.lower() if t == 'AEC' else t}_\d+$").loc[m.PatientID].to_numpy(float))
            for k, (p, q) in enumerate(SEG):
                segs = [v[i, min(sl[i, p], sl[i, q]) - 1:max(sl[i, p], sl[i, q])] for i in range(len(v))]
                m[f"{t}_sd{k}"] = [s.std() for s in segs]; m[f"{t}_kurt{k}"] = [kurtosis(s) if len(s) > 3 and s.std() > 0 else 0 for s in segs]
        out.append(m)
    return pd.concat(out, ignore_index=True)


# 풀링 5-fold x REPS: fold별 AUC와 rep별 OOF 확률
def cv_oof(X, y, strat):
    aucs, oofs = [], []
    for k in range(st.REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            oof[te] = st.mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; aucs.append(roc_auc_score(y[te], oof[te]))
        oofs.append(oof)
    return np.array(aucs), oofs


# OOF 확률에서 balanced accuracy가 최대인 cutoff의 (BA, sens, spec)
def best_ba(y, p):
    f, t, _ = roc_curve(y, p); ba = (t + 1 - f) / 2; i = ba.argmax(); return ba[i], t[i], 1 - f[i]


def main():
    A = load_all(); sc = (A.cohort == "sinchon").to_numpy()
    keep = {d: ((A[d] == 1) | (A[DIS].sum(axis=1) == 0)).to_numpy() for d in DIS}
    f = lambda cols: A[CLIN + cols].to_numpy(float)
    best = [f"{t}_{s}{k}" for t in MEAS for s in ("mean", "sum") for k in range(4)]
    tot = [f"{t}_sum{k}" for t in MEAS for k in range(4)]
    dist = [f"{t}_{s}{k}" for t in MEAS for s in ("sd", "kurt") for k in range(4)]
    sets = {"clinic4": f([]), "+ 총량 (구간 합산 16)": f(tot), "+ 총량 + 분포 (SD·첨도 포함 48)": f(tot + dist), "best (구간 평균·합산 32)": f(best)}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8)); rows, prev = [], []
    for ax, d in zip(axs, DIS):
        m = keep[d]; y = A[d].to_numpy(int)[m]; s = y * 2 + sc[m]; res = {}
        for n, X in sets.items():
            res[n] = cv_oof(X[m], y, s)
        base = res["clinic4"][0]
        for n, (a, oofs) in res.items():
            ba = np.mean([best_ba(y, o) for o in oofs], axis=0)
            rows.append(dict(disease=d, feature_set=n, AUC=a.mean(), AUC_sd=a.std(ddof=1), dAUC=(a - base).mean(), p_cv=np.nan if n == "clinic4" else st.nb_test(a - base), BA=ba[0], Sens_at_BA=ba[1], Spec_at_BA=ba[2]))
        for n in ("clinic4", "best (구간 평균·합산 32)"):
            fp, tp, _ = roc_curve(y, res[n][1][0]); ax.plot(fp, tp, lw=2, label=f"{n.split(' (')[0]}  AUC {res[n][0].mean():.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(title=f"{d} (n={len(y)}, 질환 {y.sum()}명)", xlabel="1 - Specificity", ylabel="Sensitivity"); ax.legend(loc="lower right")
        prev.append(dict(disease=d, n=len(y), n_pos=int(y.sum()), prevalence=y.mean(), majority_acc=1 - y.mean()))
    plt.tight_layout(); plt.savefig(OUT / "roc_baseline_vs_best.png", dpi=300, facecolor="white", bbox_inches="tight")
    pd.DataFrame(rows).to_excel(OUT / "hypothesis_ba.xlsx", index=False); pd.DataFrame(prev).to_excel(OUT / "prevalence.xlsx", index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(prev).round(3))


if __name__ == "__main__":
    main()
