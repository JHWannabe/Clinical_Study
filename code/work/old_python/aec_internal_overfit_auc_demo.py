from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, clinical_scores, load_dataset  # noqa: E402
from aec_region_constrained_cnn_gate import d1, d2, row_z  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_internal_overfit_auc_demo"
SEED = 20260701


def auc_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """정답 라벨 y와 점수 score로 ROC AUC와 Mann-Whitney U 검정 p-value를 계산해 함께 반환한다."""
    auc = float(roc_auc_score(y, score))
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p


def score_model(model, x: np.ndarray) -> np.ndarray:
    """모델이 지원하는 방식(predict_proba > decision_function > predict 순)으로 입력 x에 대한 점수를 산출한다."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def make_feature_matrix(d: dict, clinical_score: np.ndarray, clinical_z: np.ndarray) -> np.ndarray:
    """AEC 곡선(norm/raw)의 값·1차/2차 미분과 임상 점수(및 z, 제곱, 세제곱)를 결합해 특징 행렬을 만든다."""
    norm = d["norm"]
    raw = d["smooth_raw"]
    z = row_z(norm)
    slope = row_z(d1(norm))
    curv = row_z(d2(norm))
    raw_z = row_z(raw)
    smi_like = np.column_stack([clinical_score, clinical_z, clinical_z**2, clinical_z**3])
    return np.column_stack([norm, z, slope, curv, raw_z, smi_like]).astype(float)


def oof_external(model_factory, xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """내부 데이터(xg, yg)를 5-fold로 나눠 fold별 모델을 학습, out-of-fold 예측(oof)과 외부 데이터(xs)에 대한 fold별 예측 평균을 반환한다."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(yg), dtype=float)
    ext_scores = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(yg)), yg)):
        model = model_factory(SEED + fold)
        model.fit(xg[tr], yg[tr])
        oof[va] = score_model(model, xg[va])
        ext_scores.append(score_model(model, xs))
    return oof, np.mean(ext_scores, axis=0)


def main() -> None:
    """여러 모델(과적합되기 쉬운 ExtraTrees/RandomForest/RBF SVM/1-NN 등)에 대해 내부 재대입(resubstitution) AUC, 내부 OOF AUC,
    외부(sdata) 재학습 AUC, 외부 fold 앙상블 AUC를 비교 계산하여 과적합 정도를 보여주는 요약 CSV와 점수 CSV를 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    xg = make_feature_matrix(g, clinical_oof, c_g)
    xs = make_feature_matrix(s, clinical_ext, c_s)
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)

    factories = {
        "ExtraTrees_memorizer": lambda seed: ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=None,
            bootstrap=False,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest_memorizer": lambda seed: RandomForestClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=None,
            bootstrap=False,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "RBF_SVM_highC": lambda seed: make_pipeline(
            StandardScaler(),
            SVC(C=200.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed),
        ),
        "KNN_1_neighbor": lambda seed: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    }

    rows = []
    score_cols = {
        "dataset": ["g1090_internal"] * len(yg) + ["sdata_external"] * len(ys),
        "row_index": list(range(len(yg))) + list(range(len(ys))),
        "y_low_smi": np.r_[yg, ys],
    }

    for name, factory in factories.items():
        print(f"fitting {name}", flush=True)
        model = factory(SEED + 999)
        model.fit(xg, yg)
        train_score = score_model(model, xg)
        external_refit_score = score_model(model, xs)
        oof_score, external_fold_score = oof_external(factory, xg, yg, xs)

        train_auc, train_p = auc_p(yg, train_score)
        oof_auc, oof_p = auc_p(yg, oof_score)
        ext_refit_auc, ext_refit_p = auc_p(ys, external_refit_score)
        ext_fold_auc, ext_fold_p = auc_p(ys, external_fold_score)
        rows.append(
            {
                "model": name,
                "internal_resubstitution_auc": train_auc,
                "internal_resubstitution_p": train_p,
                "internal_oof_auc": oof_auc,
                "internal_oof_p": oof_p,
                "external_refit_auc": ext_refit_auc,
                "external_refit_p": ext_refit_p,
                "external_fold_ensemble_auc": ext_fold_auc,
                "external_fold_ensemble_p": ext_fold_p,
            }
        )
        score_cols[f"{name}_internal_resubstitution_or_external_refit"] = np.r_[train_score, external_refit_score]
        score_cols[f"{name}_oof_or_external_fold_ensemble"] = np.r_[oof_score, external_fold_score]

    summary = pd.DataFrame(rows).sort_values("internal_resubstitution_auc", ascending=False)
    summary.to_csv(OUT_DIR / "internal_overfit_auc_summary.csv", index=False)
    pd.DataFrame(score_cols).to_csv(OUT_DIR / "internal_overfit_auc_scores.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"Saved to {OUT_DIR}")


# g1090/sdata 데이터를 불러와 여러 과적합 성향 모델의 내부/외부 AUC를 비교 산출하고 결과를 outputs 폴더에 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
