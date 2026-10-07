from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# clinic4_landmark_vat_auc.py의 Model1~18에 대해 질환별 로지스틱 회귀(언더샘플링 없이 코호트 전체, 표준화 feature)를 적합하고
# 모델이 추가한 feature의 OR(+1SD당, 95% CI, p)을 낸다. clinic4 보정 상태에서의 독립적 연관성을 보는 용도(예측 AUC와 별개).
# 단면 데이터라 HR/RR은 불가 -> OR만 산출. BH-FDR은 코호트x질환 안의 모든 (모델, feature) 행에 대해 보정.

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import add_fpca, add_ratios, load_cohort, matrix, model_sets
from clinic4_logistic_regression import CLINICAL_COLS, DISEASES, format_floats, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")

OUT_XLSX = "landmark/landmark_vat_auc.xlsx"


def or_table(cohort: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, extra in model_sets().items():
        if name == "baseline":
            continue
        cols = ["VAT_sum"] + extra
        x, _ = matrix(df, cols)
        terms = ["intercept", "sex_M"] + CLINICAL_COLS + cols
        if np.linalg.matrix_rank(x - x.mean(0)) < x.shape[1] - 0:  # 완전 공선성이면 OR이 무의미 -> 건너뜀
            print(f"[{cohort}] {name}: rank-deficient (공선성), OR 생략")
            continue
        for d in DISEASES:
            fit = sm.Logit(df[d].to_numpy(int), sm.add_constant(x)).fit(disp=0, maxiter=200)
            ci = np.exp(fit.conf_int())
            for i, t in enumerate(terms):
                if t in cols and (name == "Model1" or t != "VAT_sum"):  # VAT_sum은 Model1에서만 보고(이후 모델은 보정변수)
                    rows.append({"cohort": cohort, "disease": d, "model": name, "term": t, "n": len(df),
                                 "coef": fit.params[i], "OR_per_SD": np.exp(fit.params[i]),
                                 "ci_low": ci[i, 0], "ci_high": ci[i, 1], "p_value": fit.pvalues[i]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("disease")["p_value"].transform(lambda p: bh_fdr(p.to_numpy()))
    return out


def main() -> None:
    (df, curves), (df_ext, curves_ext) = load_cohort("gangnam"), load_cohort("sinchon")
    df, df_ext = add_ratios(df), add_ratios(df_ext)
    df, df_ext, _ = add_fpca(df, curves, df_ext, curves_ext)
    res = pd.concat([or_table("gangnam", df), or_table("sinchon", df_ext)], ignore_index=True)
    save_sheet(format_floats(res), OUT_XLSX, "odds_ratio")

    # 두 코호트 모두 p<0.05이고 OR 방향이 같은 feature(모델 간 중복 제거 위해 term 기준 첫 등장)만 요약 출력
    w = res.pivot_table(index=["disease", "model", "term"], columns="cohort", values=["OR_per_SD", "p_value"])
    ok = w[(w["p_value"] < 0.05).all(axis=1) & (np.sign(np.log(w["OR_per_SD"])).nunique(axis=1) == 1)]
    print("두 코호트 모두 p<0.05 & 같은 방향\n" + ok.round(3).to_string())


if __name__ == "__main__":
    main()
