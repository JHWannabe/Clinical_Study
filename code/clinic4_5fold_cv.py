from __future__ import annotations

# 덱(docs/*.pptx)의 Methods 슬라이드가 이미 설명하고 있는 새 방법론(hold-out 없이 gangnam 전체를
# patient-level stratified 5-fold CV로 튜닝+평가, sinchon은 그 best_params로 gangnam 전체를 재학습한
# frozen 모델로 외부검증)을 실제로 구현한다. clinic4_logistic_regression.py의 7:1:2 hold-out 방식
# (run_disease/run_and_save)은 건드리지 않고 그대로 둔 채, 이 스크립트가 별도로 5-fold 버전을 산출한다.
# baseline(clinic4만) / best(clinic4 + AEC FPCA + 체성분 FPCA) 두 feature set을 모두 돌린다.

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from clinic4_logistic_regression import (
    CLINIC4_DIR, DATA_XLSX, DISEASES, EXTERNAL_COHORTS, PARAM_GRID, SEED,
    balanced_idx, f3, format_floats, load_data, save_sheet, sens_spec, youden_threshold,
)
from clinic4_logistic_regression import clinical_matrix as baseline_clinical_matrix
from clinic4_aec_bodycomp_logistic import add_curve_features, load_data_with_curves
from clinic4_aec_bodycomp_logistic import clinical_matrix as bodycomp_clinical_matrix

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

N_SPLITS = 5
OUT_XLSX = "predictions_5fold.xlsx"  # 기존 predictions.xlsx(hold-out)와 별개 파일


# disease 하나 + feature set(x/df: gangnam 전체(미언더샘플링) 행렬, x_ext/df_ext: sinchon 전체)에 대해
# 1:1 balanced undersampling -> 5-fold CV grid search(단일 best_params) -> 5-fold OOF -> pooled Youden
# cutoff -> gangnam 전체(balanced)로 frozen 재학습 -> sinchon(balanced) 외부검증까지 수행한다.
# rng는 disease마다 새로 시드해, gangnam balanced_idx를 먼저 뽑고 그다음 sinchon balanced_idx를 뽑는다
# (clinic4_logistic_regression.run_disease와 동일한 호출 순서 - rng state를 맞추기 위함).
def run_disease_5fold(disease: str, model_name: str, x_full: np.ndarray, df: pd.DataFrame,
                       x_ext_full: np.ndarray, df_ext: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    y_full = df[disease].to_numpy(dtype=int)
    pid_full = df["PatientID"].to_numpy()
    idx = balanced_idx(y_full, rng)
    x, y, pid = x_full[idx], y_full[idx], pid_full[idx]

    y_ext_full = df_ext[disease].to_numpy(dtype=int)
    pid_ext_full = df_ext["PatientID"].to_numpy()
    idx_ext = balanced_idx(y_ext_full, rng)
    x_ext, y_ext, pid_ext = x_ext_full[idx_ext], y_ext_full[idx_ext], pid_ext_full[idx_ext]

    # (b) 하이퍼파라미터 탐색과 (c) outer OOF loop에 동일한 단일 StratifiedKFold 객체를 재사용한다
    # (fold 정의가 두 단계에서 다르면 안 됨 - Methods 문구가 말하는 "5-fold CV"는 하나의 fold 세트)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    grid = GridSearchCV(LogisticRegression(max_iter=2000), PARAM_GRID, scoring="roc_auc", cv=skf).fit(x, y)
    best_params = grid.best_params_

    # (c) patient-level stratified 5-fold OOF (grid search와 동일한 skf.split을 그대로 재사용)
    oof_score = np.empty(len(y))
    fold_id = np.empty(len(y), dtype=int)
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        model = LogisticRegression(max_iter=2000, **best_params).fit(x[train_idx], y[train_idx])
        score = model.predict_proba(x[test_idx])[:, 1]
        oof_score[test_idx] = score
        fold_id[test_idx] = fold_i
        fold_aucs.append(roc_auc_score(y[test_idx], score))
    auc_mean, auc_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs, ddof=1))

    # (d) pooled OOF에서 Youden cutoff
    threshold = youden_threshold(y, oof_score)
    sens, spec = sens_spec(y, oof_score, threshold)
    brier_gangnam = brier_score_loss(y, oof_score)

    # (e) best_params로 gangnam 전체(balanced) 재학습한 frozen 모델 -> sinchon(balanced) 외부검증,
    # cutoff는 gangnam OOF에서 고정한 threshold 그대로 적용
    frozen = LogisticRegression(max_iter=2000, **best_params).fit(x, y)
    ext_score = frozen.predict_proba(x_ext)[:, 1]
    ext_auc = roc_auc_score(y_ext, ext_score)
    ext_sens, ext_spec = sens_spec(y_ext, ext_score, threshold)
    brier_ext = brier_score_loss(y_ext, ext_score)

    print(f"[{disease}/{model_name}] n_gangnam={len(y)} best_params={best_params} "
          f"fold_AUCs={[f3(a) for a in fold_aucs]} internal_AUC={f3(auc_mean)}+-{f3(auc_std)} "
          f"thr={f3(threshold)} sens={f3(sens)} spec={f3(spec)} brier_gangnam={f3(brier_gangnam)} | "
          f"n_sinchon={len(y_ext)} external_AUC={f3(ext_auc)} ext_sens={f3(ext_sens)} ext_spec={f3(ext_spec)} "
          f"brier_sinchon={f3(brier_ext)}")

    summary_row = {
        "disease": disease, "model": model_name,
        "n_gangnam": len(y), "n_sinchon": len(y_ext),
        "internal_auc_mean": auc_mean, "internal_auc_std": auc_std,
        "fold_aucs": ",".join(f3(a) for a in fold_aucs),
        "internal_pooled_auc": float(roc_auc_score(y, oof_score)),
        "external_auc": float(ext_auc),
        "threshold": float(threshold),
        "internal_sensitivity": float(sens), "internal_specificity": float(spec),
        "external_sensitivity": float(ext_sens), "external_specificity": float(ext_spec),
        "brier_gangnam_oof": float(brier_gangnam), "brier_sinchon_frozen": float(brier_ext),
        "best_C": best_params["C"], "best_penalty": best_params["penalty"],
        "best_class_weight": best_params["class_weight"],
    }
    pred_df = pd.concat([
        pd.DataFrame({"disease": disease, "model": model_name, "cohort": "gangnam", "patient_id": pid,
                      "y": y, "score": oof_score, "fold": fold_id}),
        pd.DataFrame({"disease": disease, "model": model_name, "cohort": "sinchon", "patient_id": pid_ext,
                      "y": y_ext, "score": ext_score, "fold": -1}),
    ], ignore_index=True)
    return summary_row, pred_df


def build_baseline() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    df = load_data(DATA_XLSX)
    x, scaler = baseline_clinical_matrix(df)
    df_ext = load_data(EXTERNAL_COHORTS["sinchon"])
    x_ext, _ = baseline_clinical_matrix(df_ext, scaler=scaler)
    return x, df, x_ext, df_ext


def build_best() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    df, pcas = add_curve_features(load_data_with_curves(DATA_XLSX))
    x, scaler = bodycomp_clinical_matrix(df)
    df_ext, _ = add_curve_features(load_data_with_curves(EXTERNAL_COHORTS["sinchon"]), pcas=pcas)
    x_ext, _ = bodycomp_clinical_matrix(df_ext, scaler=scaler)
    return x, df, x_ext, df_ext


def main() -> None:
    datasets = {"baseline": build_baseline(), "best": build_best()}

    summary_rows = []
    pred_dfs = []
    for model_name, (x_full, df, x_ext_full, df_ext) in datasets.items():
        for disease in DISEASES:
            row, pred_df = run_disease_5fold(disease, model_name, x_full, df, x_ext_full, df_ext)
            summary_rows.append(row)
            pred_dfs.append(pred_df)

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(pred_dfs, ignore_index=True)

    save_sheet(predictions, OUT_XLSX, "predictions")
    save_sheet(format_floats(summary), OUT_XLSX, "summary")
    print(f"Saved sheets 'predictions'/'summary' to {CLINIC4_DIR / OUT_XLSX}")


if __name__ == "__main__":
    main()
