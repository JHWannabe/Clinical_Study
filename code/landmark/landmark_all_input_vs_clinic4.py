from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# clinic4(성별/나이/신장/체중) baseline 대비, 12개 landmark의 VAT/SAT/TAMA/AEC를 전부 입력한 모델의 AUC 개선 여부.
# 프로토콜은 clinic4_landmark_vat_auc와 동일(run_disease_holdout: 1:1 balanced -> train/valid/test=7/1/2, gangnam 7/1/2 hold-out + sinchon 외부검증), 같은 landmark 코호트에서 비교.
# 출력: outputs/clinic4/landmark/landmark_all_input_vs_clinic4.xlsx
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import delong_vs, run_disease_holdout
from clinic4_logistic_regression import CLINICAL_COLS, DISEASES, f3, format_floats, save_sheet
from landmark_mlp_sets import LMN, load

sys.stdout.reconfigure(encoding="utf-8")
SETS = {"baseline": [], **{t: [f"{t}@{n}" for n in LMN] for t in ("VAT", "SAT", "TAMA", "AEC")}}
SETS["VAT+SAT+TAMA+AEC (all landmarks)"] = [c for v in list(SETS.values())[1:] for c in v]


def matrix(df, cols, scaler=None):
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_COLS + cols].to_numpy(float)
    scaler = scaler or StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


def main():
    df, ext = load("gangnam"), load("sinchon")
    rows, preds = [], []
    for name, cols in SETS.items():
        x, sc = matrix(df, cols)
        xe, _ = matrix(ext, cols, sc)
        for d in DISEASES:
            r, p = run_disease_holdout(d, name, x, df, xe, ext)
            rows.append(r)
            preds.append(p)
    coef_heatmap(df)
    s = pd.DataFrame(rows)
    save_sheet(format_floats(s), "landmark/landmark_all_input_vs_clinic4.xlsx", "summary")
    dl = delong_vs(pd.concat(preds, ignore_index=True), "baseline")
    save_sheet(format_floats(dl), "landmark/landmark_all_input_vs_clinic4.xlsx", "delong_vs_baseline")
    print(s.pivot(index="model", columns="disease", values=["internal_auc", "external_auc"]).reindex(list(SETS)).map(f3).to_string())
    print(s[["disease", "model", "n_train", "n_valid", "n_test", "n_external"]].head(3).to_string())
    print(dl.to_string())


# 전체 입력 모델을 gangnam 전체로 적합(L2, C=0.1 고정)한 표준화 계수(log-OR/1SD)를 조직 x landmark 히트맵으로 저장
def coef_heatmap(df):
    cols = SETS["VAT+SAT+TAMA+AEC (all landmarks)"]
    x, _ = matrix(df, cols)
    names = ["Male"] + CLINICAL_COLS + cols
    tis = list(SETS)[1:5]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    out = {}
    for ax, d in zip(axes, DISEASES):
        coef = pd.Series(LogisticRegression(C=0.1, max_iter=2000).fit(x, df[d]).coef_[0], index=names)
        out[d] = coef
        m = pd.DataFrame([[coef[f"{t}@{n}"] for n in LMN] for t in tis], index=tis, columns=LMN)
        v = m.abs().to_numpy().max()
        im = ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks(range(len(LMN)), LMN, rotation=60, ha="right")
        ax.set_yticks(range(len(tis)), tis)
        for i in range(len(tis)):
            for j in range(len(LMN)):
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{d} (clinic4: " + ", ".join(f"{k}={coef[k]:.2f}" for k in names[:4]) + ")", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Standardized logistic coefficient (log-OR per 1SD), all-landmark input model, gangnam")
    fig.savefig("outputs/clinic4/landmark/landmark_all_input_coef_heatmap.png", dpi=200)
    save_sheet(format_floats(pd.DataFrame(out).rename_axis("feature").reset_index()), "landmark/landmark_all_input_vs_clinic4.xlsx", "coefficients")


if __name__ == "__main__":
    main()
