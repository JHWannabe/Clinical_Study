from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 단독 질환 vs 질환 0개 코호트에서 baseline과 VAT+SAT+TAMA+AEC(전체 landmark) 모델의 ROC curve, confusion matrix.
# 분할/학습은 run_disease_holdout과 같은 순서(balanced -> 7/1/2 split -> holdout_fit). cutoff는 valid의 Youden index를 test와 sinchon에 그대로 적용.
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from clinic4_landmark_vat_auc import holdout_fit, holdout_split
from clinic4_logistic_regression import DISEASES, SEED, balanced_idx, youden_threshold
from landmark_all_input_vs_clinic4 import SETS, matrix
from landmark_mlp_sets import load
from landmark_single_disease_auc import subset

sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/clinic4/landmark/single_disease/"
MODELS = list(SETS)
SHORT = {m: m.split(" ")[0] if m != "baseline" else m for m in MODELS}
SHORT[MODELS[-1]] = "ALL(VAT+SAT+TAMA+AEC)"


def evaluate(g, s, d, cols):
    x, sc = matrix(g, cols)
    xe, _ = matrix(s, cols, sc)
    rng = np.random.default_rng(SEED)
    y_full = g[d].to_numpy(int)
    idx = balanced_idx(y_full, rng)
    idx_e = balanced_idx(s[d].to_numpy(int), rng)
    x, y, xe, ye = x[idx], y_full[idx], xe[idx_e], s[d].to_numpy(int)[idx_e]
    sp = holdout_split(y)
    o = holdout_fit(x, y, sp, list(range(x.shape[1])), xe)
    thr = youden_threshold(y[sp[1]], o["valid_score"])
    return {"internal": (y[sp[2]], o["test_score"]), "external": (ye, o["ext_score"])}, thr


def main():
    df, ext = load("gangnam"), load("sinchon")
    fig_r, ax_r = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    figs_c = {c: plt.subplots(3, len(MODELS), figsize=(3.2 * len(MODELS), 10), constrained_layout=True) for c in ("internal", "external")}
    for j, d in enumerate(DISEASES):
        g, s = subset(df, d, "normal"), subset(ext, d, "normal")
        for k, m in enumerate(MODELS):
            res, thr = evaluate(g, s, d, SETS[m])
            for i, coh in enumerate(["internal", "external"]):
                y, p = res[coh]
                fpr, tpr, _ = roc_curve(y, p)
                ax_r[i, j].plot(fpr, tpr, color=f"C{k}", lw=2 if k in (0, len(MODELS) - 1) else 1, label=f"{SHORT[m]} {roc_auc_score(y, p):.3f}")
                ax = figs_c[coh][1][j, k]
                ConfusionMatrixDisplay(confusion_matrix(y, p >= thr), display_labels=["neg", "pos"]).plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(f"{d} {SHORT[m]}\ncutoff={thr:.2f}, n={len(y)}", fontsize=8)
        for i, coh in enumerate(["internal (gangnam test)", "external (sinchon)"]):
            ax_r[i, j].plot([0, 1], [0, 1], "k--", lw=0.7)
            ax_r[i, j].set(title=f"{d} - {coh}", xlabel="1 - specificity", ylabel="sensitivity")
            ax_r[i, j].legend(loc="lower right", fontsize=8, title="AUC")
    fig_r.suptitle("ROC: baseline vs +VAT/SAT/TAMA/AEC (single-disease vs no-disease)")
    fig_r.savefig(OUT + "landmark_single_disease_roc.png", dpi=200)
    for c, (f, _) in figs_c.items():
        f.suptitle(f"Confusion matrix, {c} (cutoff = Youden on validation)")
        f.savefig(OUT + f"landmark_single_disease_confusion_{c}.png", dpi=200)


if __name__ == "__main__":
    main()
