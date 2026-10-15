from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 학습된 ResNet18({질환}_best.pt)의 Grad-CAM으로 양성/음성 차이를 정량화한다(학습 없음).
# 환자별 CAM을 합=1로 정규화 -> (1) landmark별 CAM 비중(양성 vs 음성, bootstrap CI, Mann-Whitney, BH-FDR)
# (2) 클래스 평균 CAM 지도와 차이(pos-neg) (3) CAM이 놓인 조직(overlay 색 기준 VAT/SAT/근육/기타) 비중
# (4) 이 CAM 특징만으로 pos/neg를 분류하는 5-fold CV AUC(구분 가능성 수치). 대상: gangnam test + sinchon 외부(고정 모델)
# 주의: 조직은 몬타주 overlay 색의 규칙 기반 분류(근사)이고, CAM은 항상 '양성 logit에 기여한 위치'다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import clinic4_landmark_image_cnn as M
from clinic4_landmark_vat_auc import holdout_split, load_cohort, matrix
from clinic4_logistic_regression import DISEASES, SEED, balanced_idx, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
OUT = M.FIG_DIR / "cam_analysis"
TISSUES = ["VAT", "SAT", "muscle", "other"]
RNG = np.random.default_rng(SEED)


# 몸 영역 픽셀을 overlay 색으로 VAT(적)/SAT(황)/근육(청)/기타로 분류. 반환 (12,PX,PX) int(0..3)
def tissue_map(img1: np.ndarray) -> np.ndarray:
    r, g, b = (img1[..., i].astype(int) for i in range(3))
    t = np.full(r.shape, 3)
    t[b > r + 25] = 2
    t[(r > b + 40) & (g > b + 20) & (abs(r - g) < 60)] = 1
    t[(r > g + 30) & (r > b + 20)] = 0
    return t


# 환자 1명 -> landmark 비중(12), 조직 비중(4), 정규화 CAM 지도(12,PX,PX)
def patient_feats(net, img1, x1):
    cam = M.gradcam(net, img1, x1, normalize=False)
    cam = cam / max(cam.sum(), 1e-12)
    tm = tissue_map(img1)
    return cam.sum((1, 2)), np.array([cam[tm == k].sum() for k in range(4)]), cam


def boot_ci(v, n=1000):
    return np.percentile([RNG.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5])


def analyse(disease, net, img, x, y, score, cohort):
    L, T, C = zip(*[patient_feats(net, img[i], x[i]) for i in range(len(y))])
    L, T, C = np.array(L), np.array(T), np.array(C)
    rows = []
    for name, feat, labels in (("landmark", L, M.LM_NAMES), ("tissue", T, TISSUES)):
        for j, lab in enumerate(labels):
            p, n = feat[y == 1, j], feat[y == 0, j]
            (plo, phi), (nlo, nhi) = boot_ci(p), boot_ci(n)
            rows.append({"disease": disease, "cohort": cohort, "level": name, "feature": lab, "pos_mean": p.mean(), "neg_mean": n.mean(),
                         "diff": p.mean() - n.mean(), "pos_lo": plo, "pos_hi": phi, "neg_lo": nlo, "neg_hi": nhi, "p_mwu": mannwhitneyu(p, n).pvalue})
    st = pd.DataFrame(rows)
    st["p_fdr"] = bh_fdr(st["p_mwu"].to_numpy())
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    auc = {k: roc_auc_score(y, cross_val_predict(clf, f, y, cv=cv, method="predict_proba")[:, 1])
           for k, f in (("CAM landmark share", L), ("CAM tissue share", T), ("CAM landmark+tissue", np.hstack([L, T])))}
    auc["model score"] = roc_auc_score(y, score)
    return st, auc, L, T, C


def plot(disease, cohort, st, auc, C, y, img):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [3, 1.2]})
    for a, level, labels in ((ax[0], "landmark", M.LM_NAMES), (ax[1], "tissue", TISSUES)):
        d = st[st.level == level].reset_index(drop=True)
        xs = np.arange(len(labels))
        for off, k, lab in ((-0.2, "pos", "positive"), (0.2, "neg", "negative")):
            a.bar(xs + off, d[f"{k}_mean"], 0.4, yerr=[d[f"{k}_mean"] - d[f"{k}_lo"], d[f"{k}_hi"] - d[f"{k}_mean"]], capsize=2, label=lab)
        for j, pf in enumerate(d["p_fdr"]):
            if pf < 0.05:
                a.text(j, max(d.loc[j, "pos_hi"], d.loc[j, "neg_hi"]) * 1.02, "***" if pf < 0.001 else "**" if pf < 0.01 else "*", ha="center")
        a.set_xticks(xs), a.set_xticklabels(labels, rotation=45 if level == "landmark" else 0, ha="right" if level == "landmark" else "center")
        a.set(title=f"{disease} {cohort}: CAM {level} share", ylabel="share of CAM mass"), a.legend()
    fig.text(0.01, 0.01, "bars: mean, 95% bootstrap CI; * = BH-FDR<0.05 (Mann-Whitney).  CAM-only 5-fold AUC: " + ", ".join(f"{k}={v:.3f}" for k, v in auc.items()), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)), fig.savefig(OUT / f"{disease}_{cohort}_cam_share.png", dpi=150), plt.close(fig)

    mp, mn = C[y == 1].mean(0), C[y == 0].mean(0)
    mx, dm = max(mp.max(), mn.max()), np.abs(mp - mn).max()
    fig, ax = plt.subplots(3, 12, figsize=(24, 6.6))
    for j in range(12):
        gray = img[:, j].mean(0).astype(np.uint8)  # 클래스 무관 평균 해부 배경(정렬 안 된 평균이라 흐림)
        for r, (m, kw) in enumerate(((mp[j], dict(cmap="jet", vmin=0, vmax=mx)), (mn[j], dict(cmap="jet", vmin=0, vmax=mx)),
                                     (mp[j] - mn[j], dict(cmap="bwr", vmin=-dm, vmax=dm)))):
            ax[r, j].imshow(gray, alpha=0.6), ax[r, j].imshow(m, alpha=0.55, **kw), ax[r, j].axis("off")
        ax[0, j].set_title(M.LM_NAMES[j], fontsize=8)
    for r, t in enumerate(("pos mean", "neg mean", "pos - neg\n(red = pos higher)")):
        ax[r, 0].text(-0.05, 0.5, t, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"{disease} {cohort}: class-mean Grad-CAM (patient-normalised, n_pos={int((y == 1).sum())}, n_neg={int((y == 0).sum())})")
    fig.tight_layout(), fig.savefig(OUT / f"{disease}_{cohort}_cam_mean_maps.png", dpi=110), plt.close(fig)


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    df = df[[(M.IMG_DIR["gangnam"] / f"{p}.png").exists() for p in df["PatientID"]]].reset_index(drop=True)
    df_ext = df_ext[[(M.IMG_DIR["sinchon"] / f"{p}.png").exists() for p in df_ext["PatientID"]]].reset_index(drop=True)
    img_full, img_ext_full = M.load_panels("gangnam", df["PatientID"].tolist()), M.load_panels("sinchon", df_ext["PatientID"].tolist())
    x_full, sc = matrix(df, [])
    x_ext_full, _ = matrix(df_ext, [], sc)
    stats, aucs = [], []
    for d in DISEASES:
        rng = np.random.default_rng(SEED)  # CNN 학습과 같은 balanced/split 재현(같은 rng 호출 순서)
        y_full = df[d].to_numpy(int)
        idx = balanced_idx(y_full, rng)
        y_ext_full = df_ext[d].to_numpy(int)
        idx_ext = balanced_idx(y_ext_full, rng)
        _, _, te = holdout_split(y_full[idx])
        net = M.Net(x_full.shape[1]).to(M.DEV)
        net.load_state_dict(torch.load(M.FIG_DIR / f"{d}_best.pt"))
        net.eval()
        for cohort, im, xx, yy in (("gangnam_test", img_full[idx][te], x_full[idx][te], y_full[idx][te]),
                                   ("sinchon_external", img_ext_full[idx_ext], x_ext_full[idx_ext], y_ext_full[idx_ext])):
            st, auc, L, T, C = analyse(d, net, im, xx, yy, M.predict(net, im, xx), cohort)
            plot(d, cohort, st, auc, C, yy, im)
            stats.append(st), aucs.append({"disease": d, "cohort": cohort, **auc})
            print(f"[{d}/{cohort}] n={len(yy)} " + ", ".join(f"{k}={v:.3f}" for k, v in auc.items()))
            print(st[st.p_fdr < 0.05][["level", "feature", "pos_mean", "neg_mean", "p_fdr"]].round(4).to_string(index=False))
    save_sheet(pd.concat(stats), M.OUT_XLSX, "cam_stats")
    save_sheet(pd.DataFrame(aucs), M.OUT_XLSX, "cam_classify_auc")


if __name__ == "__main__":
    main()
