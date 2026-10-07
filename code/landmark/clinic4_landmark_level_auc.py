from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# FPCA/비율 대신 landmark 단면의 체성분 값(cm2, 해당 landmark slice 1장)을 그대로 feature로 쓴다.
# (1) AUC: Model1(clinic4 + liver~pubis VAT 합)에 landmark 집합 x 조직 집합의 단면 값을 더한 5-fold CV + sinchon 외부검증
# (2) OR: landmark x 조직을 하나씩 Model1에 넣어 +1SD당 OR -> 위치별 연관성 프로파일(BH-FDR은 코호트x질환 안에서 보정)
# 데이터 로딩/행렬/5-fold 로직은 clinic4_landmark_vat_auc, DeLong 비교도 그쪽 delong_vs를 재사용.

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from clinic4_landmark_vat_auc import run_disease_holdout
from clinic4_landmark_vat_auc import LM, MUSCLE, TISSUES, add_ratios, delong_vs, load_cohort, matrix
from clinic4_logistic_regression import CLINICAL_COLS, DISEASES, f3, format_floats, save_sheet
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")

OUT_XLSX = "landmark/landmark_level_auc.xlsx"
LM_SETS = {"L3": ["L3_center"], "L1,L3,L5": ["L1_center", "L3_center", "L5_center"], "all12": LM}
DEAD = ("VAT", "inferior_pubic_margin")  # 치골 하단엔 복강 내 지방이 없어 항상 0 -> 제외
TISSUE_SETS = {**{t: [t] for t in TISSUES}, "VAT+SAT": ["VAT", "SAT"], "all3": TISSUES}


def model_sets() -> dict[str, list[str]]:
    sets = {"baseline": [], "Model1": []}
    for l, lms in LM_SETS.items():
        for t, ts in TISSUE_SETS.items():
            sets[f"+{t}@{l}"] = [f"{x}@{a}" for a in lms for x in ts if (x, a) != DEAD]
    return sets


def or_single(cohort: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in LM:
        for t in TISSUES:
            if (t, a) == DEAD:
                continue
            term = f"{t}@{a}"
            cols = ["VAT_sum"] + ([term] if term != "VAT_sum" else [])
            x, _ = matrix(df, cols)
            for d in DISEASES:
                fit = sm.Logit(df[d].to_numpy(int), sm.add_constant(x)).fit(disp=0, maxiter=200)
                ci = np.exp(fit.conf_int())[-1]
                rows.append({"cohort": cohort, "disease": d, "landmark": a, "tissue": t, "n": len(df), "coef": fit.params[-1],
                             "OR_per_SD": np.exp(fit.params[-1]), "ci_low": ci[0], "ci_high": ci[1], "p_value": fit.pvalues[-1]})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("disease")["p_value"].transform(lambda p: bh_fdr(p.to_numpy()))
    return out


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")

    rows, preds = [], []
    for name, extra in model_sets().items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, scaler = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, scaler)
        for d in DISEASES:
            row, pred = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            rows.append(row)
            preds.append(pred)
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    stats = pd.concat([delong_vs(preds, "baseline"), delong_vs(preds, "Model1")], ignore_index=True)
    ors = pd.concat([or_single("gangnam", df), or_single("sinchon", df_ext)], ignore_index=True)
    for sheet, d in (("predictions", preds), ("summary", format_floats(summary)), ("delong", format_floats(stats)),
                     ("odds_ratio_single", format_floats(ors))):
        save_sheet(d, OUT_XLSX, sheet)

    cols = ["internal_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    order = [(m, d) for m in cols for d in DISEASES]
    names = list(model_sets())
    print("AUC (internal OOF | external sinchon)\n" + piv.loc[names][order].map(f3).to_string())
    print("\ndelta vs Model1\n" + piv.sub(piv.loc["Model1"], axis=1).loc[names[1:]][order].map(f3).to_string())

    w = ors.pivot_table(index=["disease", "landmark", "tissue"], columns="cohort", values=["OR_per_SD", "p_value"])
    ok = w[(w["p_value"] < 0.05).all(axis=1) & (np.sign(np.log(w["OR_per_SD"])).nunique(axis=1) == 1)]
    print("\n단일 landmark 값: 두 코호트 모두 p<0.05 & 같은 방향\n" + ok.round(3).to_string())


if __name__ == "__main__":
    main()
