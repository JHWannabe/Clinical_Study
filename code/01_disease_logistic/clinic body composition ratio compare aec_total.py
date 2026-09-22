from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from shap_utils import run_shap_analysis

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# 다른 0907 스크립트들과 output dir을 공유하면 같은 폴더에 다른 모델 구성을 덮어써 충돌하므로, 이 스크립트
# 전용 폴더를 따로 둔다
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0907" / "clinic6_body_composition_ratio_compare_aec_total"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
AGE_CUTOFF = 20
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
AEC_UPLOW_RATIO_COL = "aec_uplow_ratio"

# aec_total 시트(사용자 요청 2026-09-07: "aec_total 시트 데이터로도 성능비교"): liver~pubis로 크롭한
# aec_128/aec_cropped와 달리 크롭 없이 스캔 전체 구간을 원본 해상도로 담고 있어 환자마다 길이(n_slices)가
# 다르다(gangnam 47~372, sinchon 38~560). Upper/Lower ratio는 리샘플링 없이 환자별 실제 n_slices 절반
# 기준으로 바로 계산 가능하지만, FPCA는 PCA 공분산 계산에 공통 차원이 필요해 aec_128과 동일한 원리(0~1
# 구간을 128등분 후 선형보간)로 리샘플링해야 한다
N_TOTAL_RESAMPLE = 128
TOTAL_RESAMPLED_COLS = [f"aec_total_resampled_{i}" for i in range(1, N_TOTAL_RESAMPLE + 1)]
AEC_UPLOW_RATIO_TOTAL_COL = "aec_uplow_ratio_total"

VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
MEAN_MAS_COL = "mean_mAs"
VATSAT_RATIO_COL = "vatsat_ratio"
SMI_COL = "SMI"
# gangnam/sinchon_final_dataset.xlsx에는 근육 데이터가 없어 CT_AEC_process/scripts/body_composition_sum.py
# 산출물(TotalSegmentator tissue_4_types, pubis~liver 구간 합산)에서 NAMA/LAMA를 가져와야 함
# (aec_muscle_composition_compare.py와 동일 소스). PatientID로 병합해 SMI(=TAMA/Height_m^2)를 계산
MUSCLE_XLSX = {
    INTERNAL_XLSX: Path(r"C:\Users\jhjun\OneDrive\Desktop\CT_AEC_process\data\Gangnam\강남_body_composition.xlsx"),
    EXTERNAL_XLSX: Path(r"C:\Users\jhjun\OneDrive\Desktop\CT_AEC_process\data\Sinchon\신촌_body_composition.xlsx"),
}
# 사용자 확인(2026-09-07): clinic body composition compare.py는 VAT/SAT를 절대값 2개로 따로 넣어 clinic7
# (SMI 포함)이었으나, 이 파일은 VAT/SAT를 비율(VSR=VAT/SAT) 1개로 합쳐 baseline에 추가 -> clinic4+VSR+SMI
# =clinic6. 비율/SMI도 실제 크기가 있는 연속값이라 age/height/weight와 동일하게 StandardScaler 대상에 포함
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight", VATSAT_RATIO_COL, SMI_COL]
EXCLUDED_PATIENT_IDS: set[int] = set()

FPCA_COMPONENT_CANDIDATES_MAX = 20
# 사용자 확인(2026-09-07): "clinic4+aec fpca 3개" - FPCA는 elbow로 새로 탐색하지 않고 3개 컴포넌트로 고정
FPCA_N_FIXED = 3
MIN_POSITIVES = 2

# 사용자 요청(2026-09-07): "하이퍼파라미터를 변경해서 optimization 진행" - internal 5-fold CV로 C x penalty
# (l1/l2/elasticnet) 그리드서치를 1회 수행해 질환/모델별 최적 조합을 확정한 뒤([[feedback_internal_external_
# validation_discipline]] 준수, external은 탐색에 전혀 쓰지 않음), 그 최적값을 고정해 재사용(사용자 확인
# 2026-09-07: "best model의 hyperparameter로 설정하고 나머지 조건들은 제거"). 그리드서치 코드 자체는 더 이상
# 실행하지 않고 outputs/0907/clinic6_body_composition_ratio_compare/logistic_regression_summary.csv의
# tuned_C/tuned_penalty 결과값을 그대로 옮겨왔다
BEST_LOGREG_PARAMS: dict[str, dict[str, dict]] = {
    "HTN": {
        "clinic6": {"C": 1.0, "penalty": "l2", "solver": "saga"},
        "clinic6_fpca3": {"C": 1.0, "penalty": "l1", "solver": "saga"},
        "clinic6_uplow_ratio": {"C": 1.0, "penalty": "l2", "solver": "saga"},
    },
    "DM": {
        "clinic6": {"C": 1.0, "penalty": "l1", "solver": "saga"},
        "clinic6_fpca3": {"C": 1.0, "penalty": "l1", "solver": "saga"},
        "clinic6_uplow_ratio": {"C": 0.1, "penalty": "l2", "solver": "saga"},
    },
    "CKD": {
        "clinic6": {"C": 100.0, "penalty": "l1", "solver": "saga"},
        "clinic6_fpca3": {"C": 100.0, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
        "clinic6_uplow_ratio": {"C": 10.0, "penalty": "l1", "solver": "saga"},
    },
}

# clinic6_fpca3_total/clinic6_uplow_ratio_total은 aec_total 신규 추가(사용자 요청 2026-09-07)라 아직
# 그리드서치를 거치지 않음 - 기존 모델들이 튜닝 전에 쓰던 것과 동일한 기본값(C=1.0, L2, lbfgs)을 임시로 사용
_DEFAULT_LOGREG_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
for _feat_params in BEST_LOGREG_PARAMS.values():
    _feat_params["clinic6_fpca3_total"] = dict(_DEFAULT_LOGREG_PARAMS)
    _feat_params["clinic6_uplow_ratio_total"] = dict(_DEFAULT_LOGREG_PARAMS)

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

# clinic4(age/sex/height/weight)에 VSR+SMI 2개를 더 baseline에 추가했다는 의미로 "clinic6"로 명명(사용자
# 확인 2026-09-07: "body composition 코드에 SMI도 추가" -> 4+VSR+SMI=6)
MODEL_ORDER = ["clinic6", "clinic6_fpca3", "clinic6_uplow_ratio", "clinic6_fpca3_total", "clinic6_uplow_ratio_total"]
MODEL_EXTRA_COLS = {
    "clinic6": [],
    "clinic6_fpca3": [],
    "clinic6_uplow_ratio": [AEC_UPLOW_RATIO_COL],
    "clinic6_fpca3_total": [],
    "clinic6_uplow_ratio_total": [AEC_UPLOW_RATIO_TOTAL_COL],
}
# clinic6_fpca3만 fold별 PCA refit이 필요한 fpca_oof_proba 경로를 탄다(AEC-128 원본에서 직접 파생되는
# 유일한 모델). uplow_ratio는 환자별로 이미 확정된 값이라 다른 clinic extra_cols와 동일하게 취급
MODEL_USES_AEC = {"clinic6": False, "clinic6_fpca3": True, "clinic6_uplow_ratio": False,
                   "clinic6_fpca3_total": True, "clinic6_uplow_ratio_total": False}
# fpca3 모델은 aec_128(크롭+128리샘플) 원본을, fpca3_total 모델은 aec_total을 128포인트로 리샘플한 배열을
# 쓴다. uplow_ratio류는 extra_cols로 이미 처리되어 여기서는 None
MODEL_AEC_SOURCE: dict[str, str | None] = {
    "clinic6": None, "clinic6_fpca3": "128", "clinic6_uplow_ratio": None,
    "clinic6_fpca3_total": "total", "clinic6_uplow_ratio_total": None,
}
DELONG_PAIRS = [
    ("clinic6", "clinic6_fpca3"),
    ("clinic6", "clinic6_uplow_ratio"),
    ("clinic6", "clinic6_fpca3_total"),
    ("clinic6", "clinic6_uplow_ratio_total"),
    ("clinic6_fpca3", "clinic6_fpca3_total"),
    ("clinic6_uplow_ratio", "clinic6_uplow_ratio_total"),
]
MODEL_LABELS = {
    "clinic6": "clinic6",
    "clinic6_fpca3": "clinic6 + AEC(128) FPCA(3)",
    "clinic6_uplow_ratio": "clinic6 + AEC(128) Upper/Lower ratio",
    "clinic6_fpca3_total": "clinic6 + AEC(total) FPCA(3)",
    "clinic6_uplow_ratio_total": "clinic6 + AEC(total) Upper/Lower ratio",
}
# 사용자 요청(2026-09-08): AEC(128)/AEC(total) 변형이 한 이미지에 섞여 있으면 비교가 어려워 family를
# 분리해 각각 별도 이미지(logistic_regression_auc_summary_aec128_compare.png 등)로 저장한다
FAMILIES = {
    "aec128_compare": [MODEL_ORDER[0], MODEL_ORDER[1], MODEL_ORDER[2]],
    "aectotal_compare": [MODEL_ORDER[0], MODEL_ORDER[3], MODEL_ORDER[4]],
}

_REF_AVG_DIM = (16.0 + 11.0) / 2
_LABEL_FS_RATIO = 32.5 / _REF_AVG_DIM
_TICK_FS_RATIO = 30.0 / _REF_AVG_DIM
_LEGEND_FS_RATIO = 27.5 / _REF_AVG_DIM


def scaled_fontsizes(width: float, height: float) -> tuple[float, float, float]:
    avg_dim = (width + height) / 2
    return _LABEL_FS_RATIO * avg_dim, _TICK_FS_RATIO * avg_dim, _LEGEND_FS_RATIO * avg_dim

# body_composition_sum.py 산출물에서 seg_status=="ok"만 취하고 TAMA(=NAMA_sum_cm2+LAMA_sum_cm2, 정상+
# 저감쇠 근육 총면적)를 계산해 [[project_smi_label_bug_and_residual_evidence]] 정의(SMI=TAMA/Height_m^2)로
# SMI를 산출. NAMA/LAMA_sum_cm2는 pubis~liver 구간 전체(n_slices_range, 환자당 평균 146슬라이스)를 합산한
# 값이라 그대로 Height^2로 나누면 SMI가 6900대로 나와 임상 범위(20~60대)를 크게 벗어난다 - n_slices_range로
# 나눠 슬라이스당 평균 근육 면적으로 정규화해야 memory 컷오프(M<45.4/F<34.4)와 부합하는 값이 나온다(직접
# 검증: 정규화 후 평균 47.4, M 15.0%/F 2.7%가 저컷오프 - 실제 발생 가능한 분포). 근육 데이터 커버리지가
# 100%가 아니므로(gangnam/sinchon 약 90%, aec_muscle_composition_compare.py 확인) how="left"로 병합하고,
# SMI가 없는 환자는 main()의 required_cols NaN 필터에서 제외된다
def merge_smi(merged: pd.DataFrame, xlsx_path: Path) -> pd.DataFrame:
    bc = pd.read_excel(MUSCLE_XLSX[xlsx_path], sheet_name="body_composition", engine="openpyxl")
    bc = bc[bc["seg_status"] == "ok"].copy()
    bc["TAMA_mean_cm2"] = (bc["NAMA_sum_cm2"].astype(float) + bc["LAMA_sum_cm2"].astype(float)) / bc["n_slices_range"]
    merged = merged.merge(bc[["PatientID", "TAMA_mean_cm2"]], on="PatientID", how="left")
    height_m = merged["Height"].astype(float) / 100
    merged[SMI_COL] = merged["TAMA_mean_cm2"] / (height_m ** 2)
    return merged.drop(columns="TAMA_mean_cm2")



# 환자별 실제 길이(vals의 length)를 0~1 구간으로 보고 target_n개 그리드로 선형보간(aec_128이 aec_cropped를
# 128개로 리샘플한 것과 동일한 원리를 aec_total에도 적용)
def _resample_curve(vals: np.ndarray, target_n: int) -> np.ndarray:
    n = len(vals)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_n)
    return np.interp(x_new, x_old, vals)


# aec_total(크롭 없는 스캔 전체 구간, 환자마다 길이 n_slices가 다름)을 병합. Upper/Lower ratio는 리샘플링
# 없이 환자별 실제 n_slices 절반 기준으로 바로 계산하고, FPCA용으로는 N_TOTAL_RESAMPLE(128)개로 선형보간한
# 배열을 별도 컬럼(TOTAL_RESAMPLED_COLS)에 저장해 기존 aec_128 파이프라인을 그대로 재사용할 수 있게 한다
def merge_aec_total(merged: pd.DataFrame, xlsx_path: Path) -> pd.DataFrame:
    total = pd.read_excel(xlsx_path, sheet_name="aec_total", engine="openpyxl")
    total_cols = sorted((c for c in total.columns if c.startswith("aec_")), key=lambda c: int(c.split("_")[1]))

    ratios, resampled_rows = [], []
    for _, row in total.iterrows():
        n = int(row["n_slices"])
        vals = row[total_cols[:n]].to_numpy(dtype=float)
        half = n // 2
        ratios.append(vals[:half].mean() / vals[half:].mean())
        resampled_rows.append(_resample_curve(vals, N_TOTAL_RESAMPLE))

    total_out = pd.DataFrame(resampled_rows, columns=TOTAL_RESAMPLED_COLS)
    total_out.insert(0, "PatientID", total["PatientID"].to_numpy())
    total_out[AEC_UPLOW_RATIO_TOTAL_COL] = ratios

    before = len(merged)
    merged = merged.merge(total_out, on="PatientID", how="inner")
    assert len(merged) == before, f"{xlsx_path.name}: metadata/aec_total merge dropped rows"
    return merged


def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[~meta["PatientID"].isin(EXCLUDED_PATIENT_IDS)].reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    merged[MEAN_MAS_COL] = merged[AEC_COLS].astype(float).mean(axis=1)
    # AEC-128은 liver->pubis 순서로 채취된 곡선이므로 앞쪽 절반(aec_1~64)이 Upper(간 쪽), 뒤쪽 절반
    # (aec_65~128)이 Lower(치골 쪽). 환자별 Upper/Lower 평균 비율을 별도 예측 변수로 사용
    half = N_SLICES // 2
    upper_mean = merged[AEC_COLS[:half]].astype(float).mean(axis=1)
    lower_mean = merged[AEC_COLS[half:]].astype(float).mean(axis=1)
    merged[AEC_UPLOW_RATIO_COL] = upper_mean / lower_mean
    # VAT/SAT 절대값 2개 대신 비율(VSR=VAT/SAT) 1개로 baseline에 반영(사용자 확인 2026-09-07)
    merged[VATSAT_RATIO_COL] = merged[VAT_COL].astype(float) / merged[SAT_COL].astype(float)
    merged = merge_smi(merged, xlsx_path)
    merged = merge_aec_total(merged, xlsx_path)
    return merged


def select_fpca_n_by_elbow(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))

    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    elbow_n = int(np.argmax(dist)) + 1

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] elbow(Kneedle) n_components = {elbow_n} (누적분산비율={cum_var[elbow_n]:.4f}) — 참고용 진단값. "
          f"실제 모델에 쓰는 컴포넌트 수는 run()에서 FPCA_N_FIXED로 고정")
    return elbow_n, cum_var


# clinic6(age/height/weight/vatsat_ratio/SMI+sex) + extra_cols(clinic6_uplow_ratio 모델의 AEC Upper/Lower
# 비율처럼 fold와 무관하게 이미 확정된 값) + (aec_extra가 있으면 clinic6_fpca3 모델의 fold별 FPCA 점수)를
# 결합. sex만 0/1 명목형이라 스케일링하지 않고 그대로 붙이고, 실제 크기가 있는 연속형(age/height/weight/
# vatsat_ratio/SMI)은 StandardScaler(clinic용/AEC용)로 표준화. scaler는 internal에서 fit해 external에
# frozen 적용
def build_matrix(meta: pd.DataFrame, extra_cols: list[str], aec_extra: np.ndarray | None = None,
                  scaler: StandardScaler | None = None, aec_scaler: StandardScaler | None = None
                  ) -> tuple[np.ndarray, StandardScaler, StandardScaler | None]:
    cols = CLINICAL_BASE_COLS + extra_cols
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    clinic = np.column_stack([sex_m, scaled])
    if aec_extra is None:
        return clinic, scaler, None
    if aec_scaler is None:
        aec_scaler = StandardScaler().fit(aec_extra)
    x = np.column_stack([clinic, aec_scaler.transform(aec_extra)])
    return x, scaler, aec_scaler


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))




# AEC를 포함하는 모델(clinic6_fpca3)의 internal OOF 확률. PCA(고유함수 추정)를 검증 fold를 제외한 학습
# fold에서만 fit해 곡선 정보 누수를 막는다(uplow_ratio 등 다른 extra_cols는 fold와 무관하게 환자별로
# 이미 확정된 값이라 누수가 아님)
def fpca_oof_proba(meta: pd.DataFrame, aec_raw: np.ndarray, extra_cols: list[str], y: np.ndarray, n_fpca: int,
                    cv: StratifiedKFold, logreg_params: dict) -> np.ndarray:
    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(aec_raw, y):
        pca = PCA(n_components=n_fpca, random_state=SEED).fit(aec_raw[train_idx])
        fpca_train, fpca_test = pca.transform(aec_raw[train_idx]), pca.transform(aec_raw[test_idx])

        x_train, scaler, aec_scaler = build_matrix(meta.iloc[train_idx], extra_cols, fpca_train)
        x_test, _, _ = build_matrix(meta.iloc[test_idx], extra_cols, fpca_test, scaler, aec_scaler)

        model = LogisticRegression(**logreg_params).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


def _delong_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_covariance(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_neg = scores.shape[1] - n_pos
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    k = scores.shape[0]
    tx = np.vstack([_delong_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_delong_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_delong_midrank(scores[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"),
                "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


# AUC의 bootstrap 95% CI 산출(external frozen 평가에만 적용, internal은 5-fold CV OOF 점추정만 사용).
# 양성 비율이 낮아 일부 resample은 한쪽 클래스가 비므로 그런 반복은 제외
def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 3000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for bi in rng.integers(0, n, size=(n_boot, n)):
        y_bi = y[bi]
        if len(np.unique(y_bi)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_bi, score[bi]))
    if len(boot_aucs) < n_boot * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# Table 1: 내부/외부 코호트 기저 특성 비교. 레퍼런스 논문(Chang & Yoon et al., Radiology 2024) Table 1
# 스타일 반영(사용자 확인 2026-08-25: 서식/스타일만 반영, 내부-외부 코호트 비교 구조는 유지) —
# (1) Demographics/Anthropometry/CT-derived measures/Comorbidities 4개 section 헤더로 구역화,
# (2) 연속형 변수를 정규분포(*, mean±SD, Welch's t-test) vs 비정규분포(†, median[IQR], Mann-Whitney U
#     test)로 구분. VAT/SAT는 skewness 0.6-1.1로 뚜렷한 우측왜도(신체 지방 분포의 통상적 특성)라 비정규,
#     Age/Height/Weight/mean_mAs는 |skew|<=0.5라 정규로 분류(D'Agostino-Pearson 검정은 표본수가 커
#     민감도가 과도해 판정 기준으로 쓰지 않고 skewness 크기로 판단). 사용자 확인(2026-09-07): csv/xlsx
# 중복 저장하지 않고 csv 한 파일만 저장
def build_table1(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    def section(label: str) -> None:
        rows.append({"section": True, "variable": label, "internal": "", "external": "",
                     "test": "", "statistic": "", "p_value": ""})

    def add_continuous(col: str, label: str, normal: bool) -> None:
        a = pd.to_numeric(meta_int[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(meta_ext[col], errors="coerce").dropna().to_numpy()
        if normal:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            internal = f"{a.mean():.1f} ± {a.std(ddof=1):.1f}"
            external = f"{b.mean():.1f} ± {b.std(ddof=1):.1f}"
            test, marker = "Welch t-test", "*"
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            q1a, q3a = np.percentile(a, [25, 75])
            q1b, q3b = np.percentile(b, [25, 75])
            internal = f"{np.median(a):.1f} ({q1a:.1f}–{q3a:.1f})"
            external = f"{np.median(b):.1f} ({q1b:.1f}–{q3b:.1f})"
            test, marker = "Mann-Whitney U test", "†"
        rows.append({
            "section": False, "variable": f"{label}{marker}",
            "internal": internal, "external": external,
            "test": test, "statistic": round(float(stat), 3), "p_value": round(float(p), 4),
        })

    def add_categorical(is_pos_int: pd.Series, is_pos_ext: pd.Series, label: str) -> None:
        n_int, n_ext = int(is_pos_int.sum()), int(is_pos_ext.sum())
        big_n_int, big_n_ext = len(is_pos_int), len(is_pos_ext)
        table = np.array([[n_int, big_n_int - n_int], [n_ext, big_n_ext - n_ext]])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        rows.append({
            "section": False, "variable": label,
            "internal": f"{n_int} ({n_int / big_n_int:.1%})",
            "external": f"{n_ext} ({n_ext / big_n_ext:.1%})",
            "test": "chi-square", "statistic": round(float(chi2), 3), "p_value": round(float(p), 4),
        })

    section("Demographics")
    add_continuous("PatientAge", "Age (years)", normal=True)
    add_categorical(meta_int["PatientSex"].astype(str).str.upper().eq("M"),
                     meta_ext["PatientSex"].astype(str).str.upper().eq("M"), "Male sex, n (%)")

    section("Anthropometry")
    add_continuous("Height", "Height (cm)", normal=True)
    add_continuous("Weight", "Weight (kg)", normal=True)

    section("CT-derived measures")
    add_continuous(VAT_COL, "VAT (cm2)", normal=False)
    add_continuous(SAT_COL, "SAT (cm2)", normal=False)
    add_continuous(MEAN_MAS_COL, "Mean tube current, mean_mAs (mA)", normal=True)

    section("Comorbidities, n (%)")
    for feat in FEATURES:
        add_categorical(pd.to_numeric(meta_int[feat], errors="coerce").fillna(0).astype(int).eq(1),
                         pd.to_numeric(meta_ext[feat], errors="coerce").fillna(0).astype(int).eq(1), feat)

    table1 = pd.DataFrame(rows)
    table1.to_csv(output_dir / "table1_patient_characteristics.csv", index=False)
    print(f"Saved Table 1 to {output_dir / 'table1_patient_characteristics.csv'}")
    return table1


def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# OR(승산비) 점추정치를 모델별 forest plot으로 시각화(계수표는 xlsx로만 저장되고 그래프가 없었음, 사용자
# 요청 2026-09-07: "OR에 대해서도 시각화 저장해"). CI는 계산하지 않고 점추정치만 표시한다(L1/elasticnet 등
# 패널티가 적용된 계수라 일반적인 Wald 표준오차 기반 CI가 통계적으로 부정확해 오해를 줄 수 있음)
def plot_or_forest(feat: str, coef_sheets: dict[str, pd.DataFrame], model_list: list[str], out_path: Path) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    panel_w, panel_h = 6.5, 5.5
    label_fs, tick_fs, _ = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs = label_fs * 1.5, tick_fs * 1.5

    fig, axes = plt.subplots(1, len(model_list), figsize=(panel_w * len(model_list), panel_h))
    axes = np.atleast_1d(axes)
    for ax, model_name in zip(axes, model_list):
        df = coef_sheets[model_name]
        df = df[df["term"] != "intercept"].iloc[::-1].reset_index(drop=True)
        y_pos = np.arange(len(df))
        ax.scatter(df["odds_ratio"], y_pos, color=colors[model_name], s=70, zorder=3)
        ax.hlines(y_pos, 1.0, df["odds_ratio"], color=colors[model_name], linewidth=1.2, alpha=0.6, zorder=2)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        # 로그축 기본 포맷은 좁은 범위에서 "1.2x10^0" 식 과학적 표기가 잔뜩 겹쳐 지저분해지므로 일반 숫자
        # 표기로 바꾸고 minor tick 라벨은 끈다(사용자 요청 2026-09-07 OR 시각화 후 발견)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["term"], fontsize=tick_fs)
        ax.set_xlabel("Odds Ratio (log scale)", fontsize=label_fs)
        ax.set_title(MODEL_LABELS[model_name], fontsize=label_fs, fontweight="bold", color="#161616")
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"{feat} Odds Ratio (internal full-fit)", fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved OR forest plot to {out_path}")


def plot_roc(feat: str, curves: dict[str, dict[str, np.ndarray]], stats_by_model: dict[str, dict[str, dict]],
             out_path: Path, model_list: list[str]) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    # 사용자 요청(2026-09-07): 그래프(축 박스)가 정방형으로 보이도록 높이를 늘림
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    # 사용자 확인(2026-08-27): plot_auc_summary와 동일하게 비율 스케일값의 1.5배 적용
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    # 사용자 요청(2026-09-07): 패널 사이/범례 위 여백을 줄여 그래프가 캔버스에서 차지하는 비중을 키움. 이후
    # internal/external 두 패널 사이 간격은 다시 벌려달라는 요청(2026-09-07)으로 wspace만 재조정
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        for model_name in model_list:
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=1.8,
                     label=f"{MODEL_LABELS[model_name]} AUC={s['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("1 - Specificity", fontsize=label_fs)
        ax.set_ylabel("Sensitivity", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)  # figsize 비율과 무관하게 축 박스 자체를 정방형으로 고정
        ax.tick_params(labelsize=tick_fs)
        # 원점에서 x축/y축 "0.0"이 겹쳐 두 번 보이므로 x축 쪽 0.0만 지운다(사용자 요청 2026-09-07)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        # clinic compare.py(outputs/0904/clinic_compare)와 동일한 폰트 크기/배치로 통일(사용자 확인 2026-09-07).
        # 패널 간 여백(wspace)을 줄인 만큼 범례 폭도 줄여야 옆 패널과 다시 겹치지 않는다
        ax.legend(fontsize=legend_fs * 0.75, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")



# Precision-Recall curve(사용자 요청 2026-09-07: "논문 figure로 쓸 수 있는 quality output 추가"). CKD처럼
# 유병률이 8~17%로 낮은 질환은 ROC-AUC가 낙관적으로 보일 수 있어 PR-AUC(Average Precision)로 보완한다.
# 기준선은 유병률(무작위 분류기의 기대 정밀도)
def plot_pr_curve(feat: str, curves: dict[str, dict[str, np.ndarray]], out_path: Path,
                   model_list: list[str]) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        prevalence = float(y.mean())
        for model_name in model_list:
            score = curves[cohort][model_name]
            precision, recall, _ = precision_recall_curve(y, score)
            ap = average_precision_score(y, score)
            ax.plot(recall, precision, color=colors[model_name], linewidth=1.8,
                     label=f"{MODEL_LABELS[model_name]} AP={ap:.3f}")
        ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1, label=f"Prevalence={prevalence:.3f}")
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=label_fs)
        ax.set_ylabel("Precision (PPV)", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.legend(fontsize=legend_fs * 0.75, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR curve plot to {out_path}")


# Calibration plot(예측확률 vs 관찰빈도, 사용자 요청 2026-09-07). AUC는 판별력만 보고 확률이 실제로 맞는지는
# 보지 않아 TRIPOD 체크리스트가 요구하는 calibration 검증이 지금까지 빠져 있었다. quantile bin 방식이라
# 이벤트 수가 적으면(예: CKD) bin 수를 줄여 각 bin에 최소한의 표본이 들어가게 한다
def plot_calibration(feat: str, curves: dict[str, dict[str, np.ndarray]], out_path: Path,
                      model_list: list[str]) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        n_bins = int(np.clip(min(int(y.sum()), int(len(y) - y.sum())), 2, 10))
        for model_name in model_list:
            score = curves[cohort][model_name]
            frac_pos, mean_pred = calibration_curve(y, score, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", markersize=6, color=colors[model_name], linewidth=1.8,
                     label=MODEL_LABELS[model_name])
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Mean predicted probability", fontsize=label_fs)
        ax.set_ylabel("Observed frequency", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.legend(fontsize=legend_fs * 0.75, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot to {out_path}")


# Decision Curve Analysis(순이익 net benefit, 사용자 요청 2026-09-07). "AEC가 clinic baseline 대비 실제
# 임상적으로 쓸모 있는가"라는 질문에 AUC delta만으로는 답이 약할 때 리뷰어가 흔히 요구하는 분석(Vickers &
# Elkin 2006). net benefit(pt) = TP/n - FP/n * pt/(1-pt). Treat All/Treat None 기준선과 함께 표시
def _net_benefit(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    net_benefits = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred_pos = score >= pt
        tp = float(np.sum(pred_pos & (y == 1)))
        fp = float(np.sum(pred_pos & (y == 0)))
        net_benefits[i] = tp / n - fp / n * (pt / (1 - pt))
    return net_benefits


def plot_dca(feat: str, curves: dict[str, dict[str, np.ndarray]], out_path: Path,
             model_list: list[str]) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    thresholds = np.linspace(0.01, 0.8, 80)
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        prevalence = float(y.mean())
        treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        ax.plot(thresholds, treat_all, color="black", linestyle=":", linewidth=1.3, label="Treat All")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Treat None")
        for model_name in model_list:
            score = curves[cohort][model_name]
            nb = _net_benefit(y, score, thresholds)
            ax.plot(thresholds, nb, color=colors[model_name], linewidth=1.8, label=MODEL_LABELS[model_name])
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Threshold probability", fontsize=label_fs)
        ax.set_ylabel("Net benefit", fontsize=label_fs)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.05, max(prevalence * 1.3, 0.1))
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.legend(fontsize=legend_fs * 0.75, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DCA plot to {out_path}")


# 모델별 AUC(internal/external)/Se/Sp/DeLong p를 논문 Table 2 형태의 이미지로 렌더링(사용자 요청
# 2026-09-07). 지금까지 csv/xlsx로만 있어 논문에 바로 넣을 수 없었다. DeLong p는 model_list[0](baseline)
# 대비 값
def plot_summary_table(feat: str, stats_by_model: dict[str, dict[str, dict]], delong_lookup: dict[tuple, dict],
                        model_list: list[str], out_path: Path) -> None:
    baseline = model_list[0]
    rows = []
    for model_name in model_list:
        s_int, s_ext = stats_by_model[model_name]["internal"], stats_by_model[model_name]["external"]
        auc_int = f"{s_int['auc']:.3f}"
        auc_ext = f"{s_ext['auc']:.3f} ({s_ext['auc_ci_lower']:.3f}-{s_ext['auc_ci_upper']:.3f})"
        if model_name == baseline:
            p_str = "Reference"
        else:
            p = delong_lookup.get((baseline, model_name), {}).get("p_value")
            p_str = "-" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")
        rows.append([MODEL_LABELS[model_name], auc_int, auc_ext, f"{s_ext['sensitivity']:.3f}",
                     f"{s_ext['specificity']:.3f}", p_str])
    col_labels = ["Model", "AUC (internal)", "AUC (external, 95% CI)", "Sensitivity\n(external)",
                  "Specificity\n(external)", f"DeLong p\nvs {MODEL_LABELS[baseline]}"]

    # Model 컬럼(예: "clinic4 + AEC Upper/Lower ratio")이 다른 컬럼보다 훨씬 길어 등폭 배분이면 잘리므로
    # 컬럼별 너비를 직접 지정(사용자 확인 2026-09-07: 표 이미지에서 첫 컬럼 텍스트 잘림 발견)
    col_widths = [0.30, 0.13, 0.20, 0.13, 0.13, 0.14]
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.7 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.2)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")
    ax.set_title(f"{feat} Summary", fontsize=17, fontweight="bold", color="#161616", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary table image to {out_path}")


def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    # 참고용으로 elbow 진단은 그대로 계산해 저장하되, 모델에는 사용자 지정 고정값(FPCA_N_FIXED=3)을 쓴다
    elbow_n, _ = select_fpca_n_by_elbow(aec_int_raw)
    n_fpca = FPCA_N_FIXED
    if elbow_n != n_fpca:
        print(f"[FPCA] 참고: elbow={elbow_n}이지만 사용자 지정 고정값 n_components={n_fpca}을(를) 사용")
    pca_full = PCA(n_components=n_fpca, random_state=SEED).fit(aec_int_raw)
    fpca_int_full, fpca_ext_full = pca_full.transform(aec_int_raw), pca_full.transform(aec_ext_raw)
    print(f"[FPCA] explained variance ratio (PC1-{n_fpca}): {pca_full.explained_variance_ratio_.round(4)}")

    # aec_total을 128포인트로 리샘플한 배열에도 동일하게 FPCA 적용(사용자 요청 2026-09-07: "aec_total 시트
    # 데이터로도 성능비교"). 컴포넌트 수는 aec_128과 동일하게 FPCA_N_FIXED=3으로 고정해 공정 비교
    aec_total_int_raw = meta_int[TOTAL_RESAMPLED_COLS].astype(float).to_numpy()
    aec_total_ext_raw = meta_ext[TOTAL_RESAMPLED_COLS].astype(float).to_numpy()
    elbow_n_total, _ = select_fpca_n_by_elbow(aec_total_int_raw)
    if elbow_n_total != n_fpca:
        print(f"[FPCA-total] 참고: elbow={elbow_n_total}이지만 사용자 지정 고정값 n_components={n_fpca}을(를) 사용")
    pca_total_full = PCA(n_components=n_fpca, random_state=SEED).fit(aec_total_int_raw)
    fpca_total_int_full = pca_total_full.transform(aec_total_int_raw)
    fpca_total_ext_full = pca_total_full.transform(aec_total_ext_raw)
    print(f"[FPCA-total] explained variance ratio (PC1-{n_fpca}): "
          f"{pca_total_full.explained_variance_ratio_.round(4)}")

    valid_features: list[str] = []
    skipped = []
    for feat in FEATURES:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int, mask_val_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        n_pos_int, n_neg_int = int(y_int_all[mask_val_int].sum()), int(mask_val_int.sum() - y_int_all[mask_val_int].sum())
        n_pos_ext, n_neg_ext = int(y_ext_all[mask_val_ext].sum()), int(mask_val_ext.sum() - y_ext_all[mask_val_ext].sum())
        print(f"[{feat}] internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만"
            print(msg)
            skipped.append({"feature": feat, "reason": msg})
            continue
        valid_features.append(feat)

    summary_rows, delong_rows = [], []
    predictions_sheets: dict[str, pd.DataFrame] = {}

    for feat in valid_features:
        slug = FEATURES[feat]
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_int, mask_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int].astype(int), y_ext_all[mask_ext].astype(int)

        meta_int_m, meta_ext_m = meta_int.loc[mask_int].reset_index(drop=True), meta_ext.loc[mask_ext].reset_index(drop=True)
        aec_int_m, aec_ext_m = aec_int_raw[mask_int], aec_ext_raw[mask_ext]
        fpca_int_m, fpca_ext_m = fpca_int_full[mask_int], fpca_ext_full[mask_ext]
        aec_total_int_m, aec_total_ext_m = aec_total_int_raw[mask_int], aec_total_ext_raw[mask_ext]
        fpca_total_int_m, fpca_total_ext_m = fpca_total_int_full[mask_int], fpca_total_ext_full[mask_ext]

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        stats_by_model: dict[str, dict[str, dict]] = {m: {} for m in MODEL_ORDER}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODEL_ORDER}
        threshold_by_model: dict[str, float] = {}
        coef_sheets = {}
        feat_predictions_rows = []

        for model_name in MODEL_ORDER:
            extra_cols = MODEL_EXTRA_COLS[model_name]
            uses_aec = MODEL_USES_AEC[model_name]
            aec_source = MODEL_AEC_SOURCE[model_name]

            logreg_params = {**BEST_LOGREG_PARAMS[feat][model_name], "max_iter": 5000}
            if uses_aec:
                if aec_source == "128":
                    aec_src_m, fpca_src_int_m, fpca_src_ext_m = aec_int_m, fpca_int_m, fpca_ext_m
                else:
                    aec_src_m, fpca_src_int_m, fpca_src_ext_m = aec_total_int_m, fpca_total_int_m, fpca_total_ext_m
                x_int_full, scaler, aec_scaler = build_matrix(meta_int_m, extra_cols, fpca_src_int_m)
                x_ext_full, _, _ = build_matrix(meta_ext_m, extra_cols, fpca_src_ext_m, scaler, aec_scaler)
                oof_proba = fpca_oof_proba(meta_int_m, aec_src_m, extra_cols, y_int, n_fpca, cv, logreg_params)
            else:
                x_int_full, scaler, _ = build_matrix(meta_int_m, extra_cols)
                x_ext_full, _, _ = build_matrix(meta_ext_m, extra_cols, scaler=scaler)
                oof_proba = cross_val_predict(LogisticRegression(**logreg_params), x_int_full, y_int, cv=cv,
                                               method="predict_proba")[:, 1]
            print(f"[{feat} / {model_name}] fixed(best) LogisticRegression: "
                  f"{ {k: v for k, v in logreg_params.items() if k != 'max_iter'} }")

            model = LogisticRegression(**logreg_params).fit(x_int_full, y_int)
            ext_proba = model.predict_proba(x_ext_full)[:, 1]

            threshold = youden_threshold(y_int, oof_proba)
            threshold_by_model[model_name] = threshold

            for cohort, meta_c, y, score in (("internal", meta_int_m, y_int, oof_proba),
                                              ("external", meta_ext_m, y_ext, ext_proba)):
                auc = float(roc_auc_score(y, score))
                # internal은 5-fold CV OOF 점추정만 쓰고, external frozen 평가에만 bootstrap CI를 붙인다
                ci_lo, ci_hi = bootstrap_auc_ci(y, score) if cohort == "external" else (float("nan"), float("nan"))
                cls_stats = classification_stats(y, score, threshold)
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "auc": auc,
                     "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi, "threshold": threshold,
                     "tuned_C": logreg_params["C"], "tuned_penalty": logreg_params["penalty"], **cls_stats}
                ci_str = f" 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]" if cohort == "external" else ""
                print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
                      f"AUC={s['auc']:.3f}{ci_str} Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")
                summary_rows.append({"feature": feat, "model": model_name, "cohort": cohort, **s})
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

                feat_predictions_rows.append(pd.DataFrame({
                    "model": model_name, "cohort": cohort,
                    "patient_id": meta_c["PatientID"].to_numpy(), "manufacturer": meta_c["Manufacturer"].astype(str).to_numpy(),
                    "y": y, "score": score, "threshold": threshold,
                }))

            n_extra_aec = n_fpca if uses_aec else 0
            fpca_prefix = "fpca_pc" if aec_source == "128" else "fpcatotal_pc"
            input_cols = (["sex_M", "age", "height", "weight", "vatsat_ratio", "smi"] + extra_cols
                          + [f"{fpca_prefix}{i}" for i in range(1, n_extra_aec + 1)])
            coef_df = pd.DataFrame({"term": input_cols + ["intercept"],
                                     "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)])})
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

            shap_label_fs, shap_tick_fs, _ = scaled_fontsizes(6.5, 5.5)
            shap_label_fs, shap_tick_fs = shap_label_fs * 1.5, shap_tick_fs * 1.5
            run_shap_analysis(model, x_int_full, x_ext_full, input_cols,
                               meta_int_m["PatientID"].to_numpy(), meta_ext_m["PatientID"].to_numpy(),
                               y_int, y_ext, feat_dir, slug, model_name, feat, MODEL_LABELS[model_name],
                               shap_label_fs, shap_tick_fs)

        predictions_sheets[slug] = pd.concat(feat_predictions_rows, ignore_index=True)

        for baseline, extended in DELONG_PAIRS:
            for cohort, y in (("internal", y_int), ("external", y_ext)):
                d = delong_paired_auc_test(y, scores_by_model[baseline][cohort], scores_by_model[extended][cohort])
                print(f"[{feat} / {cohort}] DeLong {baseline} vs {extended}: AUC diff={d['diff']:+.4f} "
                      f"z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "baseline_model": baseline,
                                     "extended_model": extended, "auc_baseline": d["auc_a"],
                                     "auc_extended": d["auc_b"], "auc_diff": d["diff"], "z": d["z"],
                                     "p_value": d["p_value"]})

        write_sheets(feat_dir / f"{slug}_logistic_coefficients.xlsx", coef_sheets)
        plot_or_forest(feat, coef_sheets, MODEL_ORDER, feat_dir / f"{slug}_or_forest.png")

        curves = {
            "internal": {"y": y_int, **{m: scores_by_model[m]["internal"] for m in MODEL_ORDER}},
            "external": {"y": y_ext, **{m: scores_by_model[m]["external"] for m in MODEL_ORDER}},
        }
        for family_name, model_list in FAMILIES.items():
            plot_roc(feat, curves, stats_by_model, feat_dir / f"{slug}_roc_curve_{family_name}.png", model_list)
            plot_pr_curve(feat, curves, feat_dir / f"{slug}_pr_curve_{family_name}.png", model_list)
            plot_calibration(feat, curves, feat_dir / f"{slug}_calibration_{family_name}.png", model_list)
            plot_dca(feat, curves, feat_dir / f"{slug}_dca_{family_name}.png", model_list)

        # DeLong p-value는 external cohort 기준, model_list[0](baseline) 대비 값만 요약 테이블에 반영
        delong_lookup_feat = {
            (row["baseline_model"], row["extended_model"]): row
            for row in delong_rows if row["feature"] == feat and row["cohort"] == "external"
        }
        plot_summary_table(feat, stats_by_model, delong_lookup_feat, MODEL_ORDER, feat_dir / f"{slug}_summary_table.png")

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    write_sheets(output_dir / "predictions.xlsx", predictions_sheets)

    for family_name, model_list in FAMILIES.items():
        plot_auc_summary(summary, output_dir / f"logistic_regression_auc_summary_{family_name}.png", model_list)
    build_and_save_delta_table(summary, delong_df, output_dir)


def plot_auc_summary(summary: pd.DataFrame, out_path: Path, model_list: list[str]) -> None:
    colors = {"clinic6": "#898781", "clinic6_fpca3": "#1baf7a", "clinic6_uplow_ratio": "#e2622e",
              "clinic6_fpca3_total": "#1baf7a", "clinic6_uplow_ratio_total": "#e2622e"}
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_list)

    panel_w, panel_h = 2 * len(features) + 2, 6
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    # 사용자 확인(2026-08-27): logistic_regression_auc_summary_*.png는 비율 스케일값 대비 폰트가 2배 더 커야
    # 하되(1차 확인), 2배 적용 결과에서 다시 0.75배로 축소 요청(2026-08-27) -> 최종 배율 2*0.75=1.5
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_list):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            offset = (i - (len(model_list) - 1) / 2) * width
            # external은 bootstrap 95% CI(auc_ci_lower/upper)를 오차막대로 표시, internal은 OOF 점추정치라
            # CI가 없어(NaN) 오차막대 생략([[project_pvalue_test_glossary]] bootstrap CI와 동일 정의)
            if cohort == "external" and rows["auc_ci_lower"].notna().all():
                yerr = np.vstack([rows["auc"] - rows["auc_ci_lower"], rows["auc_ci_upper"] - rows["auc"]])
                ax.bar(x + offset, rows["auc"], width, yerr=yerr, capsize=3,
                       error_kw={"elinewidth": 1.3, "ecolor": "#161616"},
                       label=MODEL_LABELS[model_name], color=colors[model_name])
            else:
                ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[model_name], color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=tick_fs)
        ax.set_ylim(0.5, 1.0)
        title = f"{cohort} (95% CI)" if cohort == "external" else cohort
        ax.set_title(title, fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(alpha=0.3, axis="y")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15), fontsize=legend_fs,
               frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


def build_and_save_delta_table(summary: pd.DataFrame, delong: pd.DataFrame, output_dir: Path) -> None:
    pivot = summary.pivot(index=["feature", "cohort"], columns="model", values="auc")
    pivot.columns = [f"auc_{m}" for m in pivot.columns]
    pivot = pivot.reset_index()

    merged = pivot
    for baseline, extended in DELONG_PAIRS:
        d = delong[(delong["baseline_model"] == baseline) & (delong["extended_model"] == extended)][
            ["feature", "cohort", "auc_diff", "z", "p_value"]].copy()
        colname = f"{extended}_minus_{baseline}"
        d = d.rename(columns={"auc_diff": f"delta_auc_{colname}", "z": f"delong_z_{colname}",
                               "p_value": f"delong_p_{colname}"})
        merged = merged.merge(d, on=["feature", "cohort"], how="left")

    feature_order = list(FEATURES.keys())
    merged["feature"] = pd.Categorical(merged["feature"], categories=feature_order, ordered=True)
    merged["cohort"] = pd.Categorical(merged["cohort"], categories=["internal", "external"], ordered=True)
    merged = merged.sort_values(["feature", "cohort"]).reset_index(drop=True)

    merged.round(4).to_csv(output_dir / "auc_delta_summary.csv", index=False)
    print(f"Saved AUC/delta summary table to {output_dir / 'auc_delta_summary.csv'}")


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    required_cols = CLINICAL_BASE_COLS + [MEAN_MAS_COL]  # VAT/SAT는 이미 CLINICAL_BASE_COLS에 포함

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_int).sum()}/{len(mask_int)}, "
          f"external {(~mask_ext).sum()}/{len(mask_ext)}")
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Final cohort: internal n={len(meta_int)}, external n={len(meta_ext)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_table1(meta_int, meta_ext, OUTPUT_DIR)

    run(meta_int, meta_ext, OUTPUT_DIR)


if __name__ == "__main__":
    main()
