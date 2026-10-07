from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 단일 landmark OR에서 두 코호트가 일치한 근육 질 신호(NAMA 낮음 / LAMA·IMATA 높음, 하부·상부 구간)를 소수 feature로 묶어
# AUC(5-fold + sinchon 외부)와 OR(모델 내 다변량, 전체 코호트)을 함께 본다. landmark를 하나씩 고르면 선택 편향이 커서
# 구간 평균(하부=femoral_head/S1/pubis, 상부=T10/T12)으로 묶었다. 구간은 OR 결과를 보고 정한 것이라 sinchon이 완전 독립은 아님.

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import run_disease_holdout
from clinic4_landmark_vat_auc import delong_vs, load_cohort, matrix
from clinic4_landmark_vat_auc import MUSCLE
from clinic4_logistic_regression import CLINICAL_COLS, DISEASES, f3, format_floats, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")

OUT_XLSX = "landmark/landmark_muscle_region_auc.xlsx"
REGIONS = {"low": ["femoral_head_center", "S1_center", "inferior_pubic_margin"], "up": ["T10_center", "T12_center"]}
MODELS = {
    "baseline": [], "Model1": [],
    "A +TAMA_low": ["TAMA_low"],
    "B +TAMA_up": ["TAMA_up"],
    "C +TAMA_low,up": ["TAMA_low", "TAMA_up"],
}


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    for r, lms in REGIONS.items():
        for t in MUSCLE:
            df[f"{t}_{r}"] = df[[f"{t}@{a}" for a in lms]].mean(axis=1)
    return df


def or_in_model(cohort: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, extra in MODELS.items():
        if not extra:
            continue
        cols = ["VAT_sum"] + extra
        x, _ = matrix(df, cols)
        terms = ["intercept", "sex_M"] + CLINICAL_COLS + cols
        for d in DISEASES:
            fit = sm.Logit(df[d].to_numpy(int), sm.add_constant(x)).fit(disp=0, maxiter=200)
            ci = np.exp(fit.conf_int())
            for i, t in enumerate(terms):
                if t in cols:
                    rows.append({"cohort": cohort, "disease": d, "model": name, "term": t, "OR_per_SD": np.exp(fit.params[i]),
                                 "ci_low": ci[i, 0], "ci_high": ci[i, 1], "p_value": fit.pvalues[i]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("disease")["p_value"].transform(lambda p: bh_fdr(p.to_numpy()))
    return out


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")
    df, df_ext = add_regions(df), add_regions(df_ext)

    rows, preds = [], []
    for name, extra in MODELS.items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, scaler = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, scaler)
        for d in DISEASES:
            row, pred = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            rows.append(row)
            preds.append(pred)
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    stats = pd.concat([delong_vs(preds, "baseline"), delong_vs(preds, "Model1")], ignore_index=True)
    ors = pd.concat([or_in_model("gangnam", df), or_in_model("sinchon", df_ext)], ignore_index=True)
    for sheet, d in (("predictions", preds), ("summary", format_floats(summary)), ("delong", format_floats(stats)),
                     ("odds_ratio", format_floats(ors))):
        save_sheet(d, OUT_XLSX, sheet)

    cols = ["internal_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    order = [(m, d) for m in cols for d in DISEASES]
    names = list(MODELS)
    print("AUC (internal OOF | external sinchon)\n" + piv.loc[names][order].map(f3).to_string())
    print("\ndelta vs Model1\n" + piv.sub(piv.loc["Model1"], axis=1).loc[names[1:]][order].map(f3).to_string())
    s = stats[(stats.ref == "Model1") & (stats.p_value < 0.05)]
    print("\nDeLong vs Model1 p<0.05\n" + s[["disease", "cohort", "model", "diff", "p_value", "p_fdr"]].to_string())
    o = ors.assign(OR=ors.OR_per_SD.round(2), p=ors.p_value.round(3))
    print("\nOR (모델 내, 두 코호트)\n" + o.pivot_table(index=["disease", "model", "term"], columns="cohort", values=["OR", "p"]).to_string())


if __name__ == "__main__":
    main()
