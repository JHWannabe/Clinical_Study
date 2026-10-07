from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 단독 질환(HTN/DM/CKD 중 정확히 하나) positive vs 질환 0개 negative 코호트에서, clinic4 baseline 대비 12개 landmark의 AEC/VAT/SAT/TAMA 추가 효과(AUC, DeLong).
# feature/hold-out/외부검증은 landmark_all_input_vs_clinic4 + clinic4_landmark_vat_auc를 재사용. 출력: outputs/clinic4/landmark/single_disease/landmark_single_disease_aec_bodycomp.xlsx
import sys

import pandas as pd

from clinic4_landmark_vat_auc import delong_vs, run_disease_holdout
from clinic4_logistic_regression import DISEASES, f3, format_floats, save_sheet
from landmark_all_input_vs_clinic4 import SETS, matrix
from landmark_mlp_sets import load
from landmark_single_disease_auc import subset

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    df, ext = load("gangnam"), load("sinchon")
    rows, preds = [], []
    for d in DISEASES:
        g, s = subset(df, d, "normal"), subset(ext, d, "normal")
        for name, cols in SETS.items():
            x, sc = matrix(g, cols)
            xe, _ = matrix(s, cols, sc)
            r, p = run_disease_holdout(d, name, x, g, xe, s)
            rows.append(r)
            preds.append(p)
    summ = pd.DataFrame(rows)
    dl = delong_vs(pd.concat(preds, ignore_index=True), "baseline")
    save_sheet(format_floats(summ), "landmark/single_disease/landmark_single_disease_aec_bodycomp.xlsx", "summary")
    save_sheet(format_floats(dl), "landmark/single_disease/landmark_single_disease_aec_bodycomp.xlsx", "delong_vs_baseline")
    print(summ.pivot(index="model", columns="disease", values=["internal_auc", "external_auc"]).reindex(list(SETS)).map(f3).to_string())
    print(dl[["disease", "cohort", "model", "p_value", "p_fdr"]].map(lambda v: f3(v) if isinstance(v, float) else v).to_string())


if __name__ == "__main__":
    main()
