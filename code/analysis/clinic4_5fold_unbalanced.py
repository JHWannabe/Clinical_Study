from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 덱 Table 3(질환별 성능)과 DeLong p-value의 원천 파이프라인. 강남 전체(언더샘플링 없음, n=1,260)를 환자 단위 stratified
# 5-fold CV로 튜닝+OOF 평가하고, 강남 전체로 재학습한 고정 모델을 신촌 전체(n=1,123)에 적용한다.
#  - baseline: 성별 + 표준화한 나이/키/몸무게
#  - best: baseline + AEC/체성분 FPCA 점수(PCA·scaler를 fold의 학습 데이터로만 fit해 누수 없음, 외부검증은 강남 전체로 fit)
# 하이퍼파라미터는 덱 Table 3의 fold AUC/External AUC를 재현하는 조합을 FIXED로 고정한다(최초 탐색 스크립트가 repo에 없고 평균 AUC가
# 거의 동률인 조합이 여럿이라 argmax만으로는 같은 조합이 선택되지 않음). `--search`를 주면 5-fold 평균 AUC 최대 조합을 새로 탐색한다.
# 예측값은 predictions_5fold_unbalanced.xlsx에 저장하고, 덱의 숫자(fold AUC, External AUC, DeLong p)를 이 파일에서 다시 만들 수 있다.

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import clinic4_5fold_cv as cv
from clinic4_aec_bodycomp_logistic import BODYCOMP_CURVES, CLINICAL_BASE_COLS, AEC_FPCA_N, AEC_PREFIX, curve_cols, load_data_with_curves
from clinic4_logistic_regression import (CLINIC4_DIR, DATA_XLSX, DISEASES, EXTERNAL_COHORTS, PARAM_GRID, SEED,
                                          load_data, save_sheet)
from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
SEARCH = "--search" in sys.argv
OUT_XLSX = "predictions_5fold_unbalanced.xlsx"
FIXED = {("HTN", "baseline"): (10, "l2", None), ("HTN", "best"): (1, "l1", None), ("DM", "baseline"): (0.01, "l2", "balanced"),
         ("DM", "best"): (0.1, "l2", "balanced"), ("CKD", "baseline"): (10, "l2", None), ("CKD", "best"): (0.1, "l1", None)}
# 덱 Table 3-1~3-3의 fold AUC(Fold 1~5)와 External AUC: 재현 확인용
DECK = {("HTN", "baseline"): ([.767, .797, .814, .814, .787], .718), ("HTN", "best"): ([.761, .832, .847, .828, .785], .741),
        ("DM", "baseline"): ([.729, .760, .729, .617, .726], .665), ("DM", "best"): ([.694, .781, .760, .684, .762], .706),
        ("CKD", "baseline"): ([.754, .770, .813, .799, .716], .615), ("CKD", "best"): ([.782, .781, .846, .749, .760], .681)}
CURVES = [("aec", AEC_PREFIX, AEC_FPCA_N)] + [(n, pre, k) for n, (_, pre, k) in BODYCOMP_CURVES.items()]


# best 모델용 fold 입력: PCA와 scaler를 학습 fold로만 fit하고 학습/검증 행렬을 반환
def best_fold_features(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sex = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    base = df[CLINICAL_BASE_COLS].to_numpy(float); a, b = [base[tr]], [base[te]]
    for _, pre, k in CURVES:
        raw = df[curve_cols(pre)].to_numpy(float); pca = PCA(k, random_state=SEED).fit(raw[tr])
        a.append(pca.transform(raw[tr])); b.append(pca.transform(raw[te]))
    a, b = np.hstack(a), np.hstack(b); sc = StandardScaler().fit(a)
    return np.column_stack([sex[tr], sc.transform(a)]), np.column_stack([sex[te], sc.transform(b)])


def run(model: str, disease: str, df, x_full, xe_full, ye) -> tuple[dict, pd.DataFrame]:
    y = df[disease].to_numpy(int); skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = []
    for tr, te in skf.split(x_full, y):
        a, b = (x_full[tr], x_full[te]) if model == "baseline" else best_fold_features(df, tr, te)
        folds.append((tr, te, a, b))
    best = None
    if not SEARCH:
        C, pen, cw = FIXED[(disease, model)]; params = {"C": C, "penalty": pen, "class_weight": cw, "solver": "liblinear"}
        aucs = [roc_auc_score(y[te], LogisticRegression(max_iter=2000, **params).fit(a, y[tr]).predict_proba(b)[:, 1]) for tr, te, a, b in folds]
        best = (np.mean(aucs), params, aucs)
    for params in (ParameterGrid(PARAM_GRID) if SEARCH else []):
        aucs = [roc_auc_score(y[te], LogisticRegression(max_iter=2000, **params).fit(a, y[tr]).predict_proba(b)[:, 1]) for tr, te, a, b in folds]
        if best is None or np.mean(aucs) > best[0] + 1e-12: best = (np.mean(aucs), params, aucs)
    _, params, aucs = best
    oof, fold_id = np.empty(len(y)), np.empty(len(y), dtype=int)
    for k, (tr, te, a, b) in enumerate(folds):
        oof[te] = LogisticRegression(max_iter=2000, **params).fit(a, y[tr]).predict_proba(b)[:, 1]; fold_id[te] = k
    ext = LogisticRegression(max_iter=2000, **params).fit(x_full, y).predict_proba(xe_full)[:, 1]
    row = {"disease": disease, "model": model, "fold_aucs": ",".join(f"{v:.3f}" for v in aucs), "internal_mean": float(np.mean(aucs)),
           "internal_sd": float(np.std(aucs, ddof=1)), "oof_auc": float(roc_auc_score(y, oof)), "external_auc": float(roc_auc_score(ye, ext)),
           "brier_gangnam_oof": float(brier_score_loss(y, oof)), "brier_sinchon_frozen": float(brier_score_loss(ye, ext)),
           "best_C": params["C"], "best_penalty": params["penalty"], "best_class_weight": str(params["class_weight"])}
    pred = pd.concat([pd.DataFrame({"disease": disease, "model": model, "cohort": "gangnam", "patient_id": df["PatientID"].to_numpy(), "y": y, "score": oof, "fold": fold_id}),
                      pd.DataFrame({"disease": disease, "model": model, "cohort": "sinchon", "patient_id": np.arange(len(ye)), "y": ye, "score": ext, "fold": -1})])
    return row, pred


def main() -> None:
    xb, dfb, xeb, dfeb = cv.build_baseline(); xt, dft, xet, dfet = cv.build_best()
    rows, preds = [], []
    for model, (x, df, xe, dfe) in (("baseline", (xb, dfb, xeb, dfeb)), ("best", (xt, dft, xet, dfet))):
        for d in DISEASES:
            r, p = run(model, d, df, x, xe, dfe[d].to_numpy(int)); rows.append(r); preds.append(p)
            tf, te_ = DECK[(d, model)]; dev = max(max(abs(float(v) - t) for v, t in zip(r["fold_aucs"].split(","), tf)), abs(r["external_auc"] - te_))
            print(f"[{d}/{model}] folds={r['fold_aucs']} ext={r['external_auc']:.3f} params=({r['best_C']},{r['best_penalty']},{r['best_class_weight']}) 덱 대비 최대 편차 {dev:.4f}")
    summary, pred = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    tests = []
    for d in DISEASES:
        for cohort in ("gangnam", "sinchon"):
            a = pred[(pred.disease == d) & (pred.model == "baseline") & (pred.cohort == cohort)]; b = pred[(pred.disease == d) & (pred.model == "best") & (pred.cohort == cohort)]
            r = delong_paired_auc_test(a["y"].to_numpy(), a["score"].to_numpy(), b["score"].to_numpy())
            tests.append({"disease": d, "cohort": cohort, "auc_baseline": r["auc_a"], "auc_best": r["auc_b"], "delta": r["diff"], "delong_p": r["p_value"]})
            print(f"DeLong {d}/{cohort}: {r['auc_a']:.3f} -> {r['auc_b']:.3f} (Δ{r['diff']:+.3f}) p={r['p_value']:.4g}")
    save_sheet(pred, OUT_XLSX, "predictions"); save_sheet(summary, OUT_XLSX, "summary"); save_sheet(pd.DataFrame(tests), OUT_XLSX, "delong")


if __name__ == "__main__":
    main()
