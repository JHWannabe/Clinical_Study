from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# CNN(이미지) 예측과 기존 landmark feature 모델 예측의 ensemble 평가(학습 없음).
# feature 모델 = clinic4_landmark_vat_auc의 baseline / Model1(+VAT_sum) / Model13(+FPCA all3)을 같은 split으로 재산출.
# ensemble = 두 모델 확률의 logit 단순 평균(가중치 튜닝 없음 -> test/external 누수 없음). feature 단독/CNN 단독 대비 paired DeLong.
# CNN 예측은 landmark_image_cnn.xlsx의 predictions 시트(gangnam test + sinchon 외부)를 patient_id로 병합.

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clinic4_landmark_vat_auc import add_fpca, add_ratios, load_cohort, matrix, model_sets, run_disease_holdout
from clinic4_logistic_regression import CLINIC4_DIR, DISEASES, save_sheet
from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
CNN_XLSX = CLINIC4_DIR / "landmark/landmark_image_cnn.xlsx"
OUT_XLSX = "landmark/landmark_image_ensemble.xlsx"
FEATS = ["baseline", "Model1", "Model13 +FPCA(all3)"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def main() -> None:
    (df, curves), (df_ext, curves_ext) = load_cohort("gangnam"), load_cohort("sinchon")
    df, df_ext = add_ratios(df), add_ratios(df_ext)
    df, df_ext, _ = add_fpca(df, curves, df_ext, curves_ext)
    cnn = pd.read_excel(CNN_XLSX, sheet_name="predictions").rename(columns={"score": "cnn"})
    sets = model_sets()
    rows = []
    for name in FEATS:
        cols = [] if name == "baseline" else ["VAT_sum"] + sets[name]
        x, sc = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, sc)
        for d in DISEASES:
            _, pred = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            pred = pred.assign(cohort=pred.cohort.map({"gangnam": "gangnam", "sinchon": "sinchon"}))
            m = pred.merge(cnn[cnn.disease == d][["cohort", "patient_id", "cnn"]], on=["cohort", "patient_id"], how="inner")
            for cohort, g in m.groupby("cohort"):
                y, f, c = g["y"].to_numpy(int), g["score"].to_numpy(), g["cnn"].to_numpy()
                e = 1 / (1 + np.exp(-(logit(f) + logit(c)) / 2))
                vf, vc = delong_paired_auc_test(y, f, e), delong_paired_auc_test(y, c, e)
                rows.append({"disease": d, "cohort": cohort, "feature_model": name, "n": len(y), "auc_feature": vf["auc_a"], "auc_cnn": vc["auc_a"],
                             "auc_ensemble": vf["auc_b"], "p_ens_vs_feature": vf["p_value"], "p_ens_vs_cnn": vc["p_value"]})
    out = pd.DataFrame(rows)
    save_sheet(out, OUT_XLSX, "ensemble")
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5), sharey="row")
    for r, cohort in enumerate(("gangnam", "sinchon")):
        for c, d in enumerate(DISEASES):
            g = out[(out.disease == d) & (out.cohort == cohort)].reset_index(drop=True)
            xs = np.arange(len(g))
            for off, col, lab in ((-0.27, "auc_feature", "feature"), (0, "auc_cnn", "CNN"), (0.27, "auc_ensemble", "ensemble")):
                ax[r, c].bar(xs + off, g[col], 0.27, label=lab)
            for i, p in enumerate(g["p_ens_vs_feature"]):
                ax[r, c].text(i + 0.27, g.loc[i, "auc_ensemble"] + 0.005, "p=%.3f" % p, ha="center", fontsize=7, rotation=90, va="bottom")
            ax[r, c].set_xticks(xs), ax[r, c].set_xticklabels(g["feature_model"], fontsize=8), ax[r, c].set_ylim(0.5, 1.0)
            ax[r, c].set_title(f"{d} {'gangnam test' if cohort == 'gangnam' else 'sinchon external'}"), ax[r, c].legend(fontsize=7)
    fig.suptitle("AUC: feature model vs CNN vs logit-average ensemble (p = ensemble vs feature, paired DeLong)"), fig.tight_layout()
    fig.savefig(CLINIC4_DIR / "landmark/image_cnn/ensemble_auc.png", dpi=140)


if __name__ == "__main__":
    main()
