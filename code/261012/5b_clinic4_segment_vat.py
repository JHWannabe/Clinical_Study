# step 5(구간 평균·합산)에서 VAT만 넣었을 때: clinic4 + VAT 구간 feature의 풀링 5-fold CV (step 5와 같은 분할·모델·검정)
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
sp = importlib.util.spec_from_file_location("seg5", HERE / "5_clinic4_segment_stats.py"); m5 = importlib.util.module_from_spec(sp); sp.loader.exec_module(m5)
OUT = HERE.parents[1] / "outputs/261012/5b_clinic4_segment_vat"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    A = pd.concat([m5.load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    base = A[["sex", "PatientAge", "Height", "Weight"]]
    vat = lambda stat: A[[f"VAT_{stat}{k}" for k in range(4)]]
    SETS = {"clinic4": base, "+VAT 구간 평균 (4)": pd.concat([base, vat("mean")], axis=1), "+VAT 구간 합산 (4)": pd.concat([base, vat("sum")], axis=1),
            "+VAT 구간 평균·합산 (8)": pd.concat([base, vat("mean"), vat("sum")], axis=1)}
    strat_c = (A.cohort == "sinchon").to_numpy(); rows = []
    for d in m5.DIS:
        mm = ((A[d] == 1) | (A[m5.ALL_DIS].sum(axis=1) == 0)).to_numpy(); y = A[d].to_numpy(int)[mm]
        r = {n: m5.cv(X.to_numpy(float)[mm], y, y * 2 + strat_c[mm]) for n, X in SETS.items()}
        for n, X in SETS.items():
            diff = r[n] - r["clinic4"]
            rows.append(dict(disease=d, feature_set=n, n_feat=X.shape[1], AUC=r[n].mean(), AUC_sd=r[n].std(ddof=1), dAUC=diff.mean(), p_cv=np.nan if n == "clinic4" else m5.nb_test(diff),
                             Sens=r[n].ss[0], Spec=r[n].ss[1], Sens_sd=r[n].ss_sd[0], Spec_sd=r[n].ss_sd[1]))
    res = pd.DataFrame(rows); res.to_excel(OUT / "clinic4_step5b_vat.xlsx", index=False); print(res.round(3).to_string())


if __name__ == "__main__":
    main()
