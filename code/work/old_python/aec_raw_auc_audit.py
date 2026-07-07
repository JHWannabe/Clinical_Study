from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402
from aec_vendor_neutral_preprocessing_audit import company_from_manufacturer  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_raw_auc_audit"
WORK_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SEED = 20260701
SIGMA = 1.0


def load_dataset(path: Path) -> dict:
    """엑셀에서 원시/정규화 AEC_128 곡선, 라벨, 제조사 범주를 함께 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    return {"meta": meta, "raw": raw, "norm": row_norm(raw), "y": y, "company": company}


def smoothed_raw(raw: np.ndarray) -> np.ndarray:
    """원시 곡선에 가우시안 평활화(폭 SIGMA)를 적용."""
    return ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")


def feature_sets(raw: np.ndarray) -> dict[str, np.ndarray]:
    """평활화된 원시곡선 자체, 로그버전, 8개 요약통계(평균/표준편차/최소/최대/사분위/양끝값), 평균값만
    쓰는 1차원 버전까지 4가지 "특징 세트"를 만듦 (모양 특징 대신 가장 단순한 원시 표현부터 검증)."""
    x = smoothed_raw(raw)
    lx = np.log(np.clip(x, 1e-6, None))
    summary = np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.percentile(x, 25, axis=1),
            np.percentile(x, 75, axis=1),
            x[:, 0],
            x[:, -1],
        ]
    )
    return {
        "raw_smoothed_128": x,
        "raw_log_smoothed_128": lx,
        "raw_smoothed_summary8": summary,
        "raw_smoothed_mean_only": x.mean(axis=1, keepdims=True),
    }


def make_folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """클래스 비율을 유지하는 5-fold 학습/검증 인덱스 쌍을 생성."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def model_for_dim(n_features: int):
    """특징 차원이 8 이하면 표준화+로지스틱, 그보다 크면 PCA(12차원)까지 추가한 파이프라인을 생성 (차원이 클 때 과적합 방지)."""
    if n_features <= 8:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.25),
        )
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(12, n_features), random_state=SEED),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", C=0.25),
    )


def score_model(model, x: np.ndarray) -> np.ndarray:
    """모델에 decision_function이 있으면 그것을, 없으면 양성 클래스 확률을 점수로 사용."""
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def oof_external(xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5-fold로 학습해 train의 out-of-fold 점수를, 전체 train으로 재학습한 모델로 외부 점수를 계산."""
    oof = np.zeros(len(yg), dtype=float)
    for tr, va in make_folds(yg):
        model = model_for_dim(xg.shape[1])
        model.fit(xg[tr], yg[tr])
        oof[va] = score_model(model, xg[va])
    final = model_for_dim(xg.shape[1])
    final.fit(xg, yg)
    return oof, score_model(final, xs)


def auc_p(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U 검정으로 AUC의 유의성(p값)을 계산."""
    if len(np.unique(y)) < 2:
        return np.nan
    return float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """주어진 임계값에서 정확도·민감도·특이도·균형정확도와 혼동행렬을 계산."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def model_metrics(dataset: str, model: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """데이터셋/모델 이름별로 AUC·AP·Brier와 혼동행렬 지표를 한 행으로 정리."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "dataset": dataset,
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "auc_p_mannwhitney": auc_p(y, score),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6))),
        "threshold": float(threshold),
        **binary_metrics(y, score, threshold),
    }


def bootstrap_auc_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, seed: int, n_boot: int = 3000) -> dict:
    """두 점수(base vs candidate)의 AUC 차이를 부트스트랩으로 신뢰구간과 양측 p값과 함께 추정."""
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y, candidate) - roc_auc_score(y, base))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], base[idx])))
    arr = np.asarray(vals)
    p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)) if len(arr) else np.nan
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "delta_auc_p_boot": float(min(1.0, p)) if np.isfinite(p) else np.nan,
    }


def company_cv_auc(x: np.ndarray, company: np.ndarray) -> dict:
    """AEC 특징만으로 CT 제조사를 판별하는 교차검증 모델의 정확도·균형정확도·매크로AUC를 계산 — 이 특징
    세트에 장비 흔적이 얼마나 남아있는지 재는 지표."""
    keep = company != "Other"
    x = x[keep]
    y = company[keep]
    labels = np.array(sorted(np.unique(y)))
    y_idx = np.array([np.where(labels == label)[0][0] for label in y])
    counts = pd.Series(y).value_counts()
    skf = StratifiedKFold(n_splits=int(min(5, counts.min())), shuffle=True, random_state=SEED)
    proba = np.zeros((len(y_idx), len(labels)), dtype=float)
    pred = np.zeros_like(y_idx)
    for tr, va in skf.split(x, y_idx):
        model = model_for_dim(x.shape[1])
        model.fit(x[tr], y_idx[tr])
        proba[va] = model.predict_proba(x[va])
        pred[va] = model.predict(x[va])
    aucs = []
    for i in range(len(labels)):
        yy = (y_idx == i).astype(int)
        aucs.append(roc_auc_score(yy, proba[:, i]))
    return {
        "company_cv_accuracy": float(np.mean(pred == y_idx)),
        "company_cv_balanced_accuracy": float(np.mean([np.mean(pred[y_idx == i] == i) for i in range(len(labels))])),
        "company_macro_auc": float(np.mean(aucs)),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 정교한 모양 특징이나 CNN 없이, 그냥 "평활화한 원시 곡선"
    자체나 그 요약통계만 써도 임상변수 대비 추가 정보가 있는가? — 가장 단순한 베이스라인 점검):

    1. g1090/sdata를 로드하고 임상점수·임상 Youden 임계값을 준비한다.
    2. feature_sets로 4가지 단순 특징 세트(평활화 원시 128차원, 로그 128차원, 8개 요약통계,
       평균값 1차원)를 만든다.
    3. 각 특징 세트마다: oof_external로 (차원에 따라 PCA 유무가 다른) 로지스틱 모델을 5-fold 학습해
       train OOF/외부 점수를 구하고, 임상점수와 스택한 결합모델도 만들어 AUC 등 성능을 비교.
       부트스트랩으로 결합모델의 delta AUC 신뢰구간도 계산하고, company_cv_auc로 이 특징이 CT
       제조사를 얼마나 잘 맞히는지(장비 흔적)도 함께 기록.
    4. 임상 단독 기준선과 4개 특징세트 x (AEC단독/결합) 성능을 모두 모아 CSV로 저장하고, 환자별 점수도 CSV로 저장.
    5. 핵심 지표(AUC, 유의성, 민감도/특이도, delta AUC, 회사 판별력)를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    g = load_dataset(g_path)
    s = load_dataset(s_path)
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)
    cg, cs, _ = clinical_scores(g, s)
    clinical_th = threshold_youden(yg, cg)

    fg = feature_sets(g["raw"])
    fs = feature_sets(s["raw"])
    rows = [
        model_metrics("Gangnam internal OOF", "clinical", yg, cg, clinical_th),
        model_metrics("Sinchon external", "clinical", ys, cs, clinical_th),
    ]
    scores_out = []
    company_all = np.concatenate([g["company"], s["company"]])

    for name in fg:
        ag, ase = oof_external(fg[name], yg, fs[name])
        aec_th = threshold_youden(yg, ag)
        rows.append(model_metrics("Gangnam internal OOF", f"{name}_aec_only", yg, ag, aec_th))
        rows.append(model_metrics("Sinchon external", f"{name}_aec_only", ys, ase, aec_th))

        stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)
        stack.fit(np.column_stack([cg, ag]), yg)
        fusion_g = stack.decision_function(np.column_stack([cg, ag]))
        fusion_s = stack.decision_function(np.column_stack([cs, ase]))
        fusion_th = threshold_youden(yg, fusion_g)
        rows.append(
            {
                **model_metrics("Gangnam internal OOF", f"clinical_plus_{name}", yg, fusion_g, fusion_th),
                **bootstrap_auc_delta(yg, cg, fusion_g, seed=SEED + 1),
            }
        )
        rows.append(
            {
                **model_metrics("Sinchon external", f"clinical_plus_{name}", ys, fusion_s, fusion_th),
                **bootstrap_auc_delta(ys, cs, fusion_s, seed=SEED + 2),
            }
        )

        company = company_cv_auc(np.vstack([fg[name], fs[name]]), company_all)
        rows[-1].update(company)
        rows[-2].update(company)

        scores_out.append(
            pd.DataFrame(
                {
                    "feature_set": name,
                    "cohort": "Gangnam",
                    "y": yg,
                    "clinical": cg,
                    "aec_raw_score": ag,
                    "clinical_plus_aec_raw_score": fusion_g,
                    "company": g["company"],
                }
            )
        )
        scores_out.append(
            pd.DataFrame(
                {
                    "feature_set": name,
                    "cohort": "Sinchon",
                    "y": ys,
                    "clinical": cs,
                    "aec_raw_score": ase,
                    "clinical_plus_aec_raw_score": fusion_s,
                    "company": s["company"],
                }
            )
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "raw_aec_auc_metrics.csv", index=False)
    pd.concat(scores_out, ignore_index=True).to_csv(OUT_DIR / "raw_aec_scores.csv", index=False)

    print("\nRAW AEC AUC METRICS")
    print(
        metrics[
            [
                "dataset",
                "model",
                "auc",
                "auc_p_mannwhitney",
                "sensitivity",
                "specificity",
                "accuracy",
                "delta_auc",
                "delta_auc_p_boot",
                "company_cv_balanced_accuracy",
                "company_macro_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 4가지 단순 원시 특징 세트에 대해 임상 단독/AEC 단독/결합 모델의 AUC와
    # 장비 판별력을 비교하는 베이스라인 감사 파이프라인이 수행된다.
    main()
