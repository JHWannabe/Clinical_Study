from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 12개 anchor를 모두 쓰지 않고 일부만 선택했을 때의 AUC (Model1 = clinic4 + VAT 합 위에 anchor 체성분 값 추가).
# (A) 위치 구간으로 미리 정한 부분집합 x 조직 집합 - 결과를 보고 고르지 않으므로 선택 편향 없음
# (B) nested 선택: trainval(7/1 split의 train+valid) 안에서만 35개(anchor x 조직) 중 상위 k개를 univariate F-score로 고르고 valid 단일 fold로 튜닝.
#     외부(sinchon)는 gangnam 전체로 같은 절차를 수행해 frozen 적용 -> 선택이 평가 데이터를 보지 않는다.
# 데이터 로딩/행렬/hold-out(A)/DeLong은 clinic4_landmark_vat_auc, clinic4_landmark_level_auc 재사용.

import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from clinic4_landmark_vat_auc import holdout_fit, holdout_split, run_disease_holdout
from clinic4_landmark_level_auc import DEAD
from clinic4_landmark_vat_auc import LM, MUSCLE, TISSUES, delong_vs, load_cohort, matrix
from clinic4_logistic_regression import DISEASES, PARAM_GRID, SEED, balanced_idx, f3, format_floats, save_sheet

sys.stdout.reconfigure(encoding="utf-8")

OUT_XLSX = "landmark/landmark_anchor_subset_auc.xlsx"
ANCHOR_SUBSETS = {  # 위치 구간(해부학적으로 미리 정함)
    "upper(liver,T10-12)": ["liver_dome", "T10_center", "T11_center", "T12_center"],
    "mid(L1-3)": ["L1_center", "L2_center", "L3_center"],
    "lowspine(L4,L5,S1)": ["L4_center", "L5_center", "S1_center"],
    "pelvis(femoral,pubis)": ["femoral_head_center", "inferior_pubic_margin"],
    "rep4(T12,L3,S1,femoral)": ["T12_center", "L3_center", "S1_center", "femoral_head_center"],
}
TISSUE_SETS = {"VAT+SAT": ["VAT", "SAT"], "TAMA": ["TAMA"], "all3": TISSUES}
CANDIDATES = [f"{t}@{a}" for a in LM for t in TISSUES if (t, a) != DEAD]  # 35개(12 anchor x 3조직 - VAT@pubis)
KS = [3, 5, 10]


def fixed_sets() -> dict[str, list[str]]:
    sets = {"baseline": [], "Model1": []}
    for a, lms in ANCHOR_SUBSETS.items():
        for t, ts in TISSUE_SETS.items():
            sets[f"{t}@{a}"] = [f"{x}@{l}" for l in lms for x in ts if (x, l) != DEAD]
    return sets


# disease 하나에 대한 nested 선택 5-fold OOF + 외부 AUC. 선택 대상은 anchor 컬럼(CANDIDATES)뿐, clinic4/VAT 합은 항상 포함
def run_nested(disease: str, k: int, df: pd.DataFrame, df_ext: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x_all, scaler = matrix(df, ["VAT_sum"] + CANDIDATES)
    x_ext_all, _ = matrix(df_ext, ["VAT_sum"] + CANDIDATES, scaler)
    n_fixed = 5  # sex + age/height/weight + VAT_sum
    rng = np.random.default_rng(SEED)
    y_full, y_ext_full = df[disease].to_numpy(int), df_ext[disease].to_numpy(int)
    idx = balanced_idx(y_full, rng)
    idx_ext = balanced_idx(y_ext_full, rng)
    x, y, pid = x_all[idx], y_full[idx], df["PatientID"].to_numpy()[idx]
    x_ext, y_ext, pid_ext = x_ext_all[idx_ext], y_ext_full[idx_ext], df_ext["PatientID"].to_numpy()[idx_ext]

    split = holdout_split(y)
    idx_tv = np.concatenate([split[0], split[1]])  # 선택은 trainval 안에서만(test 누수 없음), 튜닝은 valid 단일 fold GridSearchCV
    sel = SelectKBest(f_classif, k=k).fit(x[idx_tv][:, n_fixed:], y[idx_tv])
    picks = [n_fixed + i for i in np.flatnonzero(sel.get_support())]
    out = holdout_fit(x, y, split, list(range(n_fixed)) + picks, x_ext)
    name = f"nested top{k}"
    row = {"disease": disease, "model": name, "internal_auc": roc_auc_score(y[split[2]], out["test_score"]), "external_auc": roc_auc_score(y_ext, out["ext_score"]),
           "n_features": 5 + k, "selected_all_data": ",".join(CANDIDATES[c - n_fixed] for c in picks)}
    pred = pd.concat([pd.DataFrame({"disease": disease, "model": name, "cohort": "gangnam", "patient_id": pid[split[2]], "y": y[split[2]],
                                    "score": out["test_score"], "fold": -1}),
                      pd.DataFrame({"disease": disease, "model": name, "cohort": "sinchon", "patient_id": pid_ext, "y": y_ext,
                                    "score": out["ext_score"], "fold": -1})])
    return row, pred


def main() -> None:
    (df, _), (df_ext, _) = load_cohort("gangnam"), load_cohort("sinchon")

    rows, preds = [], []
    for name, extra in fixed_sets().items():
        cols = [] if name == "baseline" else ["VAT_sum"] + extra
        x, scaler = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, scaler)
        for d in DISEASES:
            row, pred = run_disease_holdout(d, name, x, df, x_ext, df_ext)
            row["n_features"] = 4 if name == "baseline" else 5 + len(extra)
            rows.append(row)
            preds.append(pred)
    for k in KS:
        for d in DISEASES:
            row, pred = run_nested(d, k, df, df_ext)
            rows.append(row)
            preds.append(pred)
            print(f"[{d}] nested top{k}: int={f3(row['internal_auc'])} ext={f3(row['external_auc'])} all-data picks={row['selected_all_data']}")
    summary, preds = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    stats = pd.concat([delong_vs(preds, "baseline"), delong_vs(preds, "Model1")], ignore_index=True)
    for sheet, d in (("predictions", preds), ("summary", format_floats(summary.drop(columns=["selected_all_data"], errors="ignore"))),
                     ("nested_selected_all_data", summary.dropna(subset=["selected_all_data"])[["disease", "model", "selected_all_data"]]),
                     ("delong", format_floats(stats))):
        save_sheet(d, OUT_XLSX, sheet)

    cols = ["internal_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    order = [(m, d) for m in cols for d in DISEASES]
    names = list(dict.fromkeys(summary.model))
    nf = summary.drop_duplicates("model").set_index("model")["n_features"]
    show = piv.loc[names][order].map(f3)
    show.insert(0, "#feat", nf.loc[names].astype(int))
    print("\nAUC (internal hold-out test | external sinchon)\n" + show.to_string())
    print("\ndelta vs Model1\n" + piv.sub(piv.loc["Model1"], axis=1).loc[names[2:]][order].map(f3).to_string())


if __name__ == "__main__":
    main()
