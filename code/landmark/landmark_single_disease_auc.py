from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# HTN/DM/CKD 중 정확히 하나만 1인(단독 질환) 환자를 positive로 쓰고, negative를 (normal) 질환 0개 환자 / (other) 다른 단독 질환 환자로 두 방식 비교.
# feature·hold-out/외부검증 로직은 clinic4_landmark_vat_auc를 그대로 재사용.
import sys

import pandas as pd

from clinic4_landmark_vat_auc import DISEASES, add_fpca, add_ratios, f3, format_floats, load_cohort, matrix, model_sets, run_disease_holdout, save_sheet

sys.stdout.reconfigure(encoding="utf-8")


# negative 정의별로 환자를 거르고 label을 df[disease]에 덮어쓴다
def subset(df: pd.DataFrame, disease: str, neg: str) -> pd.DataFrame:
    n = df[DISEASES].sum(axis=1)
    pos = (n == 1) & (df[disease] == 1)
    negm = (n == 0) if neg == "normal" else (n == 1) & (df[disease] == 0)
    out = df[pos | negm].copy()
    out[disease] = pos[out.index].astype(int)
    return out.reset_index(drop=True)


def main() -> None:
    (df, curves), (df_ext, curves_ext) = load_cohort("gangnam"), load_cohort("sinchon")
    df, df_ext = add_ratios(df), add_ratios(df_ext)
    df, df_ext, _ = add_fpca(df, curves, df_ext, curves_ext)
    rows = []
    for neg in ("normal", "other"):
        for disease in DISEASES:
            g, s = subset(df, disease, neg), subset(df_ext, disease, neg)
            for name, extra in model_sets().items():
                cols = [] if name == "baseline" else ["VAT_sum"] + extra
                x, scaler = matrix(g, cols)
                x_ext, _ = matrix(s, cols, scaler)
                row, _ = run_disease_holdout(disease, name, x, g, x_ext, s)
                rows.append({"negative": neg, "n_pos_gangnam": int(g[disease].sum()), "n_neg_gangnam": int((g[disease] == 0).sum()),
                             "n_pos_sinchon": int(s[disease].sum()), "n_neg_sinchon": int((s[disease] == 0).sum()), **row})
    summary = pd.DataFrame(rows)
    save_sheet(format_floats(summary), "landmark/single_disease/landmark_single_disease_auc.xlsx", "summary")
    for neg, g in summary.groupby("negative"):
        print(f"\n== negative={neg} ==  n(pos/neg) gangnam/sinchon:",
              {d: tuple(int(g[g.disease == d].iloc[0][c]) for c in ("n_pos_gangnam", "n_neg_gangnam", "n_pos_sinchon", "n_neg_sinchon")) for d in DISEASES})
        piv = g.pivot(index="model", columns="disease", values=["internal_auc", "external_auc"])
        print(piv.loc[list(model_sets())].map(f3).to_string())


if __name__ == "__main__":
    main()
