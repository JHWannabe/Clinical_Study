from __future__ import annotations

# code/0807/step4_clinic_aec_logistic.py(임상변수 4개 vs clinic4+AEC 8구간평균)를 베이스로 세 가지를 추가한다.
# (1) step2_clinic_aec_ratio.py에서 linear regression으로 비교한 AEC 형태 feature 후보(SD/Skewness/
#     상하위50%비율/FPCA/전체결합) 중 internal OOF R^2가 가장 높은 조합을 "clinic4_aec_best"라는 두 번째
#     모델로 연결 — "linear regression(step2)에서 선택된 best 조합을 logistic에도 동일 적용" 서술을 위해
#     step2와 같은 후보 집합만 비교하고(8구간평균은 step2에 없어 후보에서 제외), 선택은 실행 시점에 내부
#     CV로만 수행하고 external은 전혀 쓰지 않는다.
# (2) 스캐너(Manufacturer)별 서브그룹 AUC: clinic4 vs clinic4_aec_best 모델을 스캐너별로 나눠 재계산(항목4).
# (3) clinic4처럼 AEC 파생 feature도 내부 코호트 기준으로 표준화해 결합(항목7) — OR/계수 해석 일관성.
# (4) clinic4 vs clinic4_aec_best ΔAUC + DeLong p-value 요약표(scope × cohort, csv/xlsx/png) —
#     이 스크립트 자신의 output(logistic_regression_summary.csv/delong_auc_comparison.csv)만 읽으므로
#     별도 스크립트(구 step4_auc_delta_table.py)로 두지 않고 여기 통합했다(같은 output dir=같은 py 원칙).
# (5) Se/Sp는 threshold 선택법 2종(Youden's J / fixed-sensitivity SE_TARGET=90%)을 모두 internal OOF에서
#     구해 external에 freeze 적용하고, logistic_regression_summary.csv에 *_youden/*_se90 컬럼으로 병기한다
#     (AUC는 threshold-independent라 선택법과 무관하게 동일하므로 한 번만 계산).
# (6) FPCA n_components는 고정값(3) 대신 internal 코호트 AEC 곡선의 scree curve에서 elbow(Kneedle 방식:
#     축을 0~1로 정규화한 뒤 첫점-끝점을 잇는 직선까지 수직거리가 최대인 지점)로 매 run()마다 계산한다
#     (사용자 확인: "R제곱값 말고 누적분산비율로 확인해" 이후 "elbow로 교체해서 재확인" -
#     [[project_fpca_n3_vs_n12_comparison]]의 고정값 n=3 대신 적용).

import textwrap
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "past_step3_logistic"
AUC_DELTA_SCOPES = ["total", "male", "female"]
AUC_DELTA_COHORT_ORDER = ["internal", "external"]

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COMPONENT_CANDIDATES_MAX = 20  # elbow 탐색에 쓸 n_components 상한(사용자 확인: 고정값 n=3 대신 elbow로 계산)
MIN_POSITIVES = 2  # 이 미만이면 ROC/logistic 자체가 정의되지 않아 해당 feature/cohort를 skip
SCANNER_MIN_N = 30  # 스캐너별 서브그룹 분석에서 이 인원 미만인 스캐너는 제외
SE_TARGET = 0.90  # fixed-sensitivity threshold 목표치(Youden과 병기 비교용)

# 이분화 대상 체성분 feature -> (파일명 slug, cutoff 방향). TAMA/NAMA/LAMA/IMATA는 라벨변수(TAMA)와
# 상관이 높아 순환논리가 되므로 제외하고, 독립적으로 해석 가능한 VAT/SAT/Total Fat 절대값과 그 비율만 사용
# ([[feedback_no_circular_label_feature]]). "high" = mean+1SD 초과면 이상(지방/내장지방 비중 과다).
FEATURES: dict[str, tuple[str, str]] = {
    "SAT(피하지방)_SUM": ("sat", "high"),
    "VAT(내장지방)_SUM": ("vat", "high"),
    "Total Fat_SUM": ("total_fat", "high"),
    "VAT_TotalFat_ratio": ("vat_totalfat_ratio", "high"),
    "VAT_SAT_ratio": ("vat_sat_ratio", "high"),
    "VAT_TAMA_ratio": ("vat_tama_ratio", "high"),
    "TotalFat_TAMA_ratio": ("totalfat_tama_ratio", "high"),
    "SAT_TotalFat_ratio": ("sat_totalfat_ratio", "high"),
    "SAT_TAMA_ratio": ("sat_tama_ratio", "high"),
}


# VAT/Total Fat, VAT/SAT, VAT/TAMA, Total Fat/TAMA, SAT/Total Fat, SAT/TAMA 6개 비율 컬럼을 원본 SUM 컬럼으로부터 산출해 추가
# (step2_clinic_aec_ratio.py/step4_aec_diagnostics.py와 동일)
def add_ratio_features(meta: pd.DataFrame) -> pd.DataFrame:
    vat = pd.to_numeric(meta["VAT(내장지방)_SUM"], errors="coerce")
    sat = pd.to_numeric(meta["SAT(피하지방)_SUM"], errors="coerce")
    tama = pd.to_numeric(meta["TAMA_SUM"], errors="coerce")
    total_fat = pd.to_numeric(meta["Total Fat_SUM"], errors="coerce")

    meta = meta.copy()
    meta["VAT_TotalFat_ratio"] = vat / total_fat
    meta["VAT_SAT_ratio"] = vat / sat
    meta["VAT_TAMA_ratio"] = vat / tama
    meta["TotalFat_TAMA_ratio"] = total_fat / tama
    meta["SAT_TotalFat_ratio"] = sat / total_fat
    meta["SAT_TAMA_ratio"] = sat / tama
    return meta


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합, 체성분 비율 컬럼을 추가
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return add_ratio_features(merged)


# raw AEC-128 행렬(n x 128)에서 환자별 SD, Skewness, 슬라이스 위치 기준 상위/하위 50% 평균 비율을 산출
# (step3_clinic_aec_shape.py의 shape_features()와 동일)
def shape_features(aec_matrix: np.ndarray) -> dict[str, np.ndarray]:
    sd = aec_matrix.std(axis=1, ddof=1)
    skewness = stats.skew(aec_matrix, axis=1)

    half = N_SLICES // 2
    upper_mean = aec_matrix[:, :half].mean(axis=1)
    lower_mean = aec_matrix[:, half:].mean(axis=1)
    upper_lower_ratio = upper_mean / lower_mean

    return {"sd": sd, "skew": skewness, "upper_lower_ratio": upper_lower_ratio}


# internal 코호트 raw AEC-128 곡선에 PCA를 fit해 top-n_components score를 산출, external에는 frozen 적용
def fpca_scores(aec_int: np.ndarray, aec_ext: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=SEED).fit(aec_int)
    return pca.transform(aec_int), pca.transform(aec_ext)


# scree curve(컴포넌트별 개별 explained variance ratio)를 구하고, 축을 0~1로 정규화한 뒤 첫점-끝점을
# 잇는 직선(chord)까지의 수직거리를 계산한다. 거리가 최대인 지점이 elbow(Satopaa et al. 2011 Kneedle
# 알고리즘). 축 정규화 없이 원래 스케일로 거리를 재면 축 단위 차이 때문에 결과가 왜곡되므로 정규화가 필수다
def _scree_and_elbow_distance(cum_var: pd.Series) -> tuple[pd.Series, pd.Series]:
    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    return scree, pd.Series(dist, index=scree.index)


# internal 코호트 AEC-128 곡선의 scree curve에서 elbow(첫점-끝점 직선까지 수직거리가 최대인 지점)를
# n_components로 선택한다. 다운스트림 예측성능(AUC)으로 n을 고르면 그 성능 자체가 선택 기준에 쓰인 데이터로
# 최적화되어 낙관적으로 부풀려질 위험이 있어, FPCA/PCA 표준 scree test 관행대로 분산 감소 패턴만으로 n을 정한다
def select_best_fpca_n(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))
    _, dist = _scree_and_elbow_distance(cum_var)
    best_n = int(dist.idxmax())

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] 선택된 elbow n_components = {best_n} (누적분산비율={cum_var[best_n]:.4f}, "
          f"chord-거리={dist[best_n]:.4f})")
    return best_n, cum_var


# n_components별 누적/개별 explained variance ratio, elbow 판단에 쓴 chord-거리, 선택된 best_n을 엑셀로
# 저장(그래프의 원자료)
def save_cum_var_excel(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values,
                        "individual_variance_ratio": scree.values, "elbow_chord_distance": dist.values})
    df["selected_best_n"] = df["n_components"] == best_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + (있으면) AEC feature도 별도 표준화해 결합.
# scaler/aec_scaler 생략 시 새로 학습(내부 코호트용) — AEC feature(구간평균/SD/FPCA score 등)는 clinic4와
# 스케일이 크게 달라 raw로 섞으면 계수 해석이 왜곡되므로 clinic4와 동일하게 내부 코호트 기준으로 표준화한다(항목7).
# 성별을 고정한 남/여 개별 실행에서는 sex가 상수가 되어 계수가 무의미해지므로 include_sex=False로 제외
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     aec_scaler: StandardScaler | None = None,
                     include_sex: bool = True) -> tuple[np.ndarray, StandardScaler, StandardScaler | None]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    if aec_extra is None:
        return clinic, scaler, None
    if aec_scaler is None:
        aec_scaler = StandardScaler().fit(aec_extra)
    x = np.column_stack([clinic, aec_scaler.transform(aec_extra)])
    return x, scaler, aec_scaler


# FPCA score가 들어간 회귀 후보(aec_fpca/aec_shape_all)의 internal OOF 예측을 산출. PCA(고유함수 추정)를
# 전체 데이터로 먼저 fit하고 나서 CV를 돌리면 검증 fold의 곡선 정보가 고유함수 추정에 이미 들어가 OOF R^2가
# 낙관적으로 부풀려지는 data leakage가 된다 - 그래서 PCA.fit을 fold의 train 구간에서만 수행하고 검증 fold는
# transform만 한다(shape_extra_full은 fold와 무관하게 이미 각 샘플별로 독립 계산된 값 - leak 아님, None이면 미포함)
def _fpca_oof_predict(meta: pd.DataFrame, aec_raw: np.ndarray, shape_extra_full: np.ndarray | None,
                       y_full: np.ndarray, mask: np.ndarray, n_fpca: int, include_sex: bool,
                       cv: KFold) -> np.ndarray:
    aec_masked = aec_raw[mask]
    meta_masked = meta.loc[mask].reset_index(drop=True)
    shape_masked = shape_extra_full[mask] if shape_extra_full is not None else None
    y = y_full[mask]

    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(aec_masked):
        pca = PCA(n_components=n_fpca, random_state=SEED).fit(aec_masked[train_idx])
        fpca_train = pca.transform(aec_masked[train_idx])
        fpca_test = pca.transform(aec_masked[test_idx])

        if shape_masked is not None:
            extra_train = np.column_stack([shape_masked[train_idx], fpca_train])
            extra_test = np.column_stack([shape_masked[test_idx], fpca_test])
        else:
            extra_train, extra_test = fpca_train, fpca_test

        x_train, scaler, aec_scaler = clinical_matrix(meta_masked.iloc[train_idx], extra_train,
                                                        include_sex=include_sex)
        x_test, _, _ = clinical_matrix(meta_masked.iloc[test_idx], extra_test, scaler, aec_scaler,
                                        include_sex=include_sex)

        model = LinearRegression().fit(x_train, y[train_idx])
        oof[test_idx] = model.predict(x_test)
    return oof


# FPCA score가 들어간 clinic4_aec_best 모델(logistic)의 internal OOF 확률을 산출. _fpca_oof_predict와
# 동일한 이유로 PCA를 fold의 train 구간에서만 fit한다(분류이므로 LogisticRegression+predict_proba, cv는
# StratifiedKFold를 그대로 받아 cv.split(X, y)로 층화)
def _fpca_oof_proba_predict(meta: pd.DataFrame, aec_raw: np.ndarray, shape_extra_full: np.ndarray | None,
                             y_full: np.ndarray, mask: np.ndarray, n_fpca: int, include_sex: bool,
                             cv: StratifiedKFold) -> np.ndarray:
    aec_masked = aec_raw[mask]
    meta_masked = meta.loc[mask].reset_index(drop=True)
    shape_masked = shape_extra_full[mask] if shape_extra_full is not None else None
    y = y_full[mask]

    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(aec_masked, y):
        pca = PCA(n_components=n_fpca, random_state=SEED).fit(aec_masked[train_idx])
        fpca_train = pca.transform(aec_masked[train_idx])
        fpca_test = pca.transform(aec_masked[test_idx])

        if shape_masked is not None:
            extra_train = np.column_stack([shape_masked[train_idx], fpca_train])
            extra_test = np.column_stack([shape_masked[test_idx], fpca_test])
        else:
            extra_train, extra_test = fpca_train, fpca_test

        x_train, scaler, aec_scaler = clinical_matrix(meta_masked.iloc[train_idx], extra_train,
                                                        include_sex=include_sex)
        x_test, _, _ = clinical_matrix(meta_masked.iloc[test_idx], extra_test, scaler, aec_scaler,
                                        include_sex=include_sex)

        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


# 후보명 -> FPCA 포함 방식("fpca_only"=FPCA score만, "shape_all"=SD/Skew/상하위비율+FPCA 결합).
# 이 두 후보만 OOF 계산 시 fold별 PCA refit(_fpca_oof_predict/_fpca_oof_proba_predict)을 거쳐야 함
FPCA_CANDIDATE_KINDS = {"aec_fpca": "fpca_only", "aec_shape_all": "shape_all"}


# step2_clinic_aec_ratio.py와 동일한 5개 AEC 후보(SD/Skewness/상하위50%비율/FPCA/전체결합) 중 internal
# OOF R^2가 target_features(9개 연속형 SUM/비율) 평균으로 가장 높은 조합을 선택 — external은 전혀 쓰지
# 않는다(내부 CV로만 모델 선택, [[feedback_internal_external_validation_discipline]]). linear regression
# (step2)에서 선택된 best 조합을 logistic(step3)에도 동일 적용한다는 서술을 위한 핵심 로직.
# fpca_int/fpca_ext는 전체 internal 코호트로 fit한 뒤 external에는 frozen 적용하는 최종 후보 계수 산출용
# ([[feedback_internal_external_validation_discipline]] - 이건 leakage 아님). 이 함수가 반환하는 조합
# "선택"에 쓰이는 internal OOF R^2 자체는 fpca_int를 직접 쓰지 않고 fold별로 PCA를 다시 fit해 산출한다
def select_best_shape_model(meta_int: pd.DataFrame, aec_int_raw: np.ndarray, aec_ext_raw: np.ndarray,
                             include_sex: bool, target_features: list[str],
                             cv: KFold, n_fpca: int) -> tuple[str, np.ndarray, np.ndarray]:
    shape_int, shape_ext = shape_features(aec_int_raw), shape_features(aec_ext_raw)
    fpca_int, fpca_ext = fpca_scores(aec_int_raw, aec_ext_raw, n_fpca)
    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])

    candidates = {
        "aec_sd": (shape_int["sd"].reshape(-1, 1), shape_ext["sd"].reshape(-1, 1)),
        "aec_skew": (shape_int["skew"].reshape(-1, 1), shape_ext["skew"].reshape(-1, 1)),
        "aec_uplow_ratio": (shape_int["upper_lower_ratio"].reshape(-1, 1), shape_ext["upper_lower_ratio"].reshape(-1, 1)),
        "aec_fpca": (fpca_int, fpca_ext),
        "aec_shape_all": (
            np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"], fpca_int]),
            np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"], fpca_ext]),
        ),
    }

    mean_r2 = {}
    for name, (aec_int, _) in candidates.items():
        fpca_kind = FPCA_CANDIDATE_KINDS.get(name)
        x_int_all = None
        if fpca_kind is None:
            x_int_all, _, _ = clinical_matrix(meta_int, aec_int, include_sex=include_sex)
        r2s = []
        for feat in target_features:
            y_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y_all)
            y = y_all[mask]
            if fpca_kind is None:
                oof = cross_val_predict(LinearRegression(), x_int_all[mask], y, cv=cv)
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof = _fpca_oof_predict(meta_int, aec_int_raw, shape_extra, y_all, mask, n_fpca,
                                         include_sex, cv)
            r2s.append(r2_score(y, oof))
        mean_r2[name] = float(np.mean(r2s))

    best_name = max(mean_r2, key=lambda n: mean_r2[n])
    print(f"[Step2 조합 선택] 후보별 internal OOF 평균 R^2: {({k: round(v, 4) for k, v in mean_r2.items()})}")
    print(f"[Step2 조합 선택] 선택된 조합 = {best_name} (internal OOF 평균 R^2={mean_r2[best_name]:.4f})")
    aec_int_best, aec_ext_best = candidates[best_name]
    return best_name, aec_int_best, aec_ext_best


# feature별 라벨 산출에 쓸 연속값: 모든 feature에 대해 raw SUM 값을 그대로 사용
def label_source_values(meta: pd.DataFrame, feat: str) -> np.ndarray:
    return pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)


# internal 코호트의 성별(M/F) 그룹별로 mean±1SD cutoff을 산출 ("low"->mean-1SD, "high"->mean+1SD).
# 남/여 개별 실행에서는 데이터에 한쪽 성별만 존재하므로 실제 존재하는 성별에 대해서만 산출
def sex_specific_cutoffs(values: np.ndarray, sex: np.ndarray, direction: str) -> dict[str, float]:
    cutoffs = {}
    for s in ("M", "F"):
        v = values[(sex == s) & np.isfinite(values)]
        if len(v) == 0:
            continue
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        cutoffs[s] = mean - 1 * sd if direction == "low" else mean + 1 * sd
    return cutoffs


# 성별 cutoff으로 이분형 라벨(이상=1)을 산출. cutoff은 항상 internal에서 고정한 절대값을 그대로 적용(재계산 금지)
def apply_cutoff_label(values: np.ndarray, sex: np.ndarray, cutoffs: dict[str, float], direction: str) -> np.ndarray:
    th = np.full(len(sex), np.nan)
    for s, c in cutoffs.items():
        th[sex == s] = c
    return ((values < th) if direction == "low" else (values > th)).astype(int)


# DeLong midrank (동순위는 평균 순위로 처리)
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


# scores: (n_scores, n_samples), 앞 n_pos개 열이 양성. DeLong et al.(1988)/Sun & Xu(2014) 알고리즘으로
# 각 score의 AUC와 그 공분산 행렬을 산출한다 (같은 표본으로 계산된 두 AUC를 비교하려면 이 공분산이 필요)
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


# 같은 환자 집합(같은 y)에서 나온 두 예측 점수(score_a vs score_b)의 AUC가 서로 다른지 검정하는 paired DeLong test.
# clinic4와 clinic4+AEC는 동일 환자에 대해 평가되므로 독립 two-sample test가 아니라 이 paired test를 써야 함
def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


# AUC의 bootstrap 95% CI 산출. 양성 비율이 낮아 일부 resample은 한쪽 클래스가 비므로 그런 반복은 제외
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


# internal OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold를 선택 (external에는 이 값을 고정 적용)
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


# internal OOF ROC에서 sensitivity>=target_sens을 만족하는 지점 중 threshold가 가장 높은(=specificity가
# 가장 높은) 지점을 선택 (external에는 이 값을 고정 적용). roc_curve는 threshold 내림차순으로 정렬되고
# tpr은 threshold가 낮아질수록 단조증가하므로, tpr>=target_sens을 처음 만족하는 인덱스가 그 조건을
# 만족하는 가장 높은 threshold다
def fixed_sensitivity_threshold(y: np.ndarray, score: np.ndarray, target_sens: float = SE_TARGET) -> float:
    _, tpr, thresholds = roc_curve(y, score)
    idx = int(np.argmax(tpr >= target_sens))
    return float(thresholds[idx])


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# feat/model/cohort별 핵심 통계를 한 줄로 출력. Se/Sp는 Youden threshold와 fixed-Se(SE_TARGET) threshold
# 두 세트를 병기해 이후 비교할 수 있게 한다
def _log(feat: str, model_name: str, cohort: str, s: dict) -> None:
    ci_lo, ci_hi = s["auc_ci_lower"], s["auc_ci_upper"]
    print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.3f} 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}] | "
          f"Youden: Se={s['sensitivity_youden']:.3f} Sp={s['specificity_youden']:.3f} Acc={s['accuracy_youden']:.3f} | "
          f"Se>={SE_TARGET:.0%}: Se={s['sensitivity_se90']:.3f} Sp={s['specificity_se90']:.3f} Acc={s['accuracy_se90']:.3f}")


# 이미 계산된 확률 점수(oof_proba 또는 ext_proba)를 스캐너(Manufacturer)별로 나눠 AUC를 재계산.
# 재학습 없이 예측값만 재슬라이싱하므로 비용이 거의 없음. SCANNER_MIN_N 미만이거나 한쪽 클래스만
# 있는 스캐너는 AUC가 정의되지 않아 제외한다
def scanner_subgroup_auc(meta: pd.DataFrame, mask: np.ndarray, y: np.ndarray, score: np.ndarray,
                          feat: str, model_name: str, cohort: str) -> list[dict]:
    scanners = meta["Manufacturer"].astype(str).to_numpy()[mask]
    rows = []
    for scanner in pd.unique(scanners):
        sel = scanners == scanner
        n = int(sel.sum())
        if n < SCANNER_MIN_N or len(np.unique(y[sel])) < 2:
            continue
        rows.append({"feature": feat, "model": model_name, "cohort": cohort, "scanner": scanner, "n": n,
                      "n_pos": int(y[sel].sum()), "auc": float(roc_auc_score(y[sel], score[sel]))})
    return rows


# feature별로 clinic4 vs (internal 기준) 최고 AEC 모델의 스캐너별 AUC를 나란히 막대그래프로 비교
# overall_df(스캐너로 나누지 않은 전체 표본 기준 feature/model/cohort/auc 요약)를 주면 모델별
# 색상에 맞춘 점선으로 "스캐너 통합 시 AUC" 기준선을 함께 그린다
def plot_scanner_subgroup_auc(scanner_df: pd.DataFrame, out_path: Path,
                               overall_df: pd.DataFrame | None = None,
                               model_labels: dict[str, str] | None = None) -> None:
    model_labels = model_labels or {}
    if scanner_df.empty:
        print(f"[스캐너 서브그룹] 표본이 {SCANNER_MIN_N}명 이상이고 두 클래스가 모두 있는 스캐너가 없어 그래프를 생략합니다.")
        return

    features = list(scanner_df["feature"].unique())
    cohorts = ["internal", "external"]
    # step4_aec_diagnostics.py의 scanner subgroup 그래프와 동일한 폭 배정: 코호트별 스캐너 개수에
    # 비례해 열 너비를 정하고, 긴 스캐너명은 textwrap으로 접어 회전 없이도 겹치지 않게 한다
    per_category_width = 3.2
    label_wrap_width = 11
    n_scanners_by_col = [
        max((scanner_df[(scanner_df["feature"] == f) & (scanner_df["cohort"] == c)]["scanner"].nunique()
             for f in features), default=1)
        for c in cohorts
    ]
    col_widths = [max(n, 1) * per_category_width for n in n_scanners_by_col]
    fig, axes = plt.subplots(len(features), 2, figsize=(sum(col_widths), 9 * len(features)), squeeze=False,
                              gridspec_kw={"width_ratios": col_widths})
    for row, feat in enumerate(features):
        for col, cohort in enumerate(cohorts):
            ax = axes[row][col]
            sub = scanner_df[(scanner_df["feature"] == feat) & (scanner_df["cohort"] == cohort)]
            if sub.empty:
                ax.set_visible(False)
                continue
            scanners = sorted(sub["scanner"].unique())
            models_here = list(sub["model"].unique())
            x = np.arange(len(scanners)) * 2.0
            width = 0.8 / max(len(models_here), 1)
            colors = ["#6b6a66", "#2a78d6", "#e2622e"]

            scanner_arr = sub["scanner"].to_numpy()
            model_arr = sub["model"].to_numpy()
            auc_arr = sub["auc"].to_numpy(dtype=float)
            for i, model_name in enumerate(models_here):
                display_name = model_labels.get(model_name, model_name)
                vals = [float(np.mean(auc_arr[(scanner_arr == s) & (model_arr == model_name)])) for s in scanners]
                offset = (i - (len(models_here) - 1) / 2) * width
                ax.bar(x + offset, vals, width, label=display_name, color=colors[i % len(colors)])
                if overall_df is not None:
                    orow = overall_df[(overall_df["feature"] == feat) & (overall_df["model"] == model_name)
                                       & (overall_df["cohort"] == cohort)]
                    if not orow.empty:
                        ax.axhline(float(orow["auc"].iloc[0]), color=colors[i % len(colors)], linestyle="--",
                                   linewidth=2.5, alpha=0.9, zorder=3, label=f"{display_name} (스캐너 통합)")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance level (AUC=0.5)")
            ax.set_xticks(x)
            wrapped = ["\n".join(textwrap.wrap(s, label_wrap_width)) for s in scanners]
            ax.set_xticklabels(wrapped, fontsize=30, rotation=0)
            ax.set_ylim(0.6, 1.0)
            ax.set_title(f"{feat} — {cohort} 스캐너별 AUC", fontsize=30, fontweight="bold", color="#161616",
                         pad=24)
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=30)

    # 패널마다 반복되던 범례를 없애고, 그림 전체 하단에 공통 범례 하나로 통합(모든 패널이 같은
    # feature 목록 내에서는 동일한 모델 조합을 쓰므로 라벨이 패널마다 달라질 위험이 없음)
    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            if ax.get_visible():
                handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            break
    # x축 눈금이 2줄로 접힐 수 있어(label_wrap_width) 범례와 겹치지 않도록, feature 수와 무관하게
    # 항상 일정한 절대 여백(인치)을 하단에 확보한다
    fig_height_in = fig.get_size_inches()[1]
    bottom_margin_in = 2.2
    bottom_frac = bottom_margin_in / fig_height_in
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=30,
               bbox_to_anchor=(0.5, bottom_frac * 0.15), frameon=True)

    fig.tight_layout(rect=(0, bottom_frac, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup AUC plot to {out_path}")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# feature 하나에 대해 internal(OOF)/external(frozen) ROC curve를 clinic4/clinic4_aec_best 2개 모델
# 겹쳐 그림. 범례에 AUC(95%CI)를 표시하고, 제목에 DeLong test(clinic4 vs clinic4_aec_best) p-value와
# 유의/비유의 판정(p<0.05)을 함께 표기
def plot_roc_dual(feat: str, model_order: list[str], curves: dict[str, dict[str, np.ndarray]],
                   stats_by_model: dict[str, dict[str, dict]],
                   delong_by_model: dict[str, dict[str, dict]], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#2a78d6", "clinic4_aec_best": "#e2622e"}
    labels = {"clinic4": "clinic4", "clinic4_aec_best": "clinic4 + AEC(step2 best)"}
    cohorts = ["internal", "external"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cohort in zip(axes, cohorts):
        for model_name in model_order:
            y = curves[cohort]["y"]
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=1.8,
                    label=f"{labels[model_name]} AUC={s['auc']:.3f} "
                          f"[{s['auc_ci_lower']:.3f}, {s['auc_ci_upper']:.3f}]")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        p = delong_by_model["clinic4_aec_best"][cohort]["p_value"]
        p_label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        sig_label = "유의" if p < 0.05 else "비유의"
        ax.set_title(f"{feat} ({cohort})\n{p_label}", fontsize=20,
                     fontweight="bold", color=INK_PRIMARY)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# select_best_shape_model()이 고른 후보 이름 -> 범례에 쓸 한글 설명(어떤 AEC 형태 feature인지 명시)
AEC_CANDIDATE_LABELS = {
    "aec_sd": "SD(표준편차)",
    "aec_skew": "Skewness(비대칭도)",
    "aec_uplow_ratio": "상하위50%비율",
    "aec_fpca": "FPCA",
    "aec_shape_all": "전체결합: SD·Skew·상하위비율·FPCA",
}


# 전체 feature에 걸쳐 AUC(95%CI)를 2개 모델 x internal/external로 비교하는 막대그래프
def plot_auc_summary(summary: pd.DataFrame, model_order: list[str], out_path: Path, best_name: str) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#6b6a66", "clinic4_aec_best": "#e2622e"}
    labels = {"clinic4": "clinic4",
              "clinic4_aec_best": f"+AEC({AEC_CANDIDATE_LABELS.get(best_name, best_name)})"}
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f][0] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(6 * len(features) / 2 + 3, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_order):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            err = np.abs(np.vstack([rows["auc"] - rows["auc_ci_lower"], rows["auc_ci_upper"] - rows["auc"]]))
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, yerr=err, capsize=3, label=labels[model_name], color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        wrapped_slugs = [s.replace("_", "\n") for s in slugs]
        ax.set_xticklabels(wrapped_slugs, fontsize=20, rotation=0)
        ax.set_ylim(0.6, 1.0)
        ax.set_title(cohort, fontsize=20, fontweight="bold", color=INK_PRIMARY)
        ax.set_ylabel("AUC", fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.grid(alpha=0.3, axis="y")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(model_order),
               bbox_to_anchor=(0.5, -0.3), fontsize=20, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


# clinic4(include_sex=False면 clinic3) baseline과, clinic4+AEC(step2에서 linear regression으로 비교한
# 5개 후보 중 internal-best 조합) 2개 모델로 체성분 feature 9종(VAT/SAT/Total Fat 절대값 + 비율 6종)의
# 이상 여부를 logistic regression 예측. cutoff은 internal 성별 mean±1SD로
# 산출/고정 후 external에 그대로 적용, internal OOF로 모델을 학습/평가하고 external에는 고정 모델을 1회만
# 적용(freeze), AUC는 DeLong test(clinic4 vs clinic4_aec_best)로 비교, 스캐너별 서브그룹 AUC도 함께 산출
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper().to_numpy()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper().to_numpy()

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    n_fpca, cum_var = select_best_fpca_n(aec_int_raw)
    save_cum_var_excel(cum_var, n_fpca, output_dir / "fpca_cumulative_variance.xlsx")

    cv_reg = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_name, aec_int_best, aec_ext_best = select_best_shape_model(
        meta_int, aec_int_raw, aec_ext_raw, include_sex, list(FEATURES.keys()), cv_reg, n_fpca)
    # clinic4_aec_best의 internal OOF 확률(fold별 PCA refit)에 필요 - best_name이 aec_shape_all이면
    # SD/Skew/상하위비율도 함께 결합해야 하므로 여기서 한 번 구해 둠
    shape_int_for_best = shape_features(aec_int_raw)
    shape_int_stack = np.column_stack(
        [shape_int_for_best["sd"], shape_int_for_best["skew"], shape_int_for_best["upper_lower_ratio"]])
    best_fpca_kind = FPCA_CANDIDATE_KINDS.get(best_name)

    model_order = ["clinic4", "clinic4_aec_best"]
    aec_by_model = {
        "clinic4": {"aec_int": None, "aec_ext": None},
        "clinic4_aec_best": {"aec_int": aec_int_best, "aec_ext": aec_ext_best},
    }
    x_int_by_model: dict[str, np.ndarray] = {}
    x_ext_by_model: dict[str, np.ndarray] = {}
    clinic_scaler = None
    for model_name in model_order:
        spec = aec_by_model[model_name]
        x_int, clinic_scaler, aec_scaler = clinical_matrix(meta_int, spec["aec_int"], include_sex=include_sex)
        x_ext, _, _ = clinical_matrix(meta_ext, spec["aec_ext"], clinic_scaler, aec_scaler, include_sex=include_sex)
        x_int_by_model[model_name], x_ext_by_model[model_name] = x_int, x_ext

    cutoff_rows = []
    summary_rows = []
    delong_rows = []
    scanner_rows = []
    skipped = []

    for feat, (slug, direction) in FEATURES.items():
        values_int = label_source_values(meta_int, feat)
        values_ext = label_source_values(meta_ext, feat)

        mask_val_int = np.isfinite(values_int)
        cutoffs = sex_specific_cutoffs(values_int[mask_val_int], sex_int[mask_val_int], direction)

        mask_val_ext = np.isfinite(values_ext)
        y_int_all = apply_cutoff_label(values_int, sex_int, cutoffs, direction)
        y_ext_all = apply_cutoff_label(values_ext, sex_ext, cutoffs, direction)

        for s, cutoff_val in cutoffs.items():
            v = values_int[mask_val_int & (sex_int == s)]
            cutoff_rows.append({"feature": feat, "sex": s, "direction": direction,
                                 "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "cutoff": cutoff_val, "n_internal": int(len(v))})

        n_pos_int = int(y_int_all[mask_val_int].sum())
        n_neg_int = int(mask_val_int.sum() - n_pos_int)
        n_pos_ext = int(y_ext_all[mask_val_ext].sum())
        n_neg_ext = int(mask_val_ext.sum() - n_pos_ext)
        cutoff_str = ", ".join(f"{s}={c:.2f}" for s, c in cutoffs.items())
        print(f"[{feat}] cutoff {cutoff_str} | "
              f"internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")

        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = (f"[{feat}] SKIP: mean±1SD cutoff이 이 코호트에서 한쪽 클래스를 {MIN_POSITIVES}명 미만으로 "
                   f"만들어 logistic regression/ROC 산출이 불가함 "
                   f"(internal pos={n_pos_int} neg={n_neg_int}, external pos={n_pos_ext} neg={n_neg_ext})")
            print(msg)
            skipped.append({"feature": feat, "reason": msg, "n_pos_internal": n_pos_int,
                             "n_neg_internal": n_neg_int, "n_pos_external": n_pos_ext, "n_neg_external": n_neg_ext})
            continue

        mask_int = mask_val_int
        mask_ext = mask_val_ext
        y_int = y_int_all[mask_int]
        y_ext = y_ext_all[mask_ext]

        n_splits = max(2, min(N_FOLDS, n_pos_int, n_neg_int))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        stats_by_model: dict[str, dict[str, dict]] = {m: {} for m in model_order}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in model_order}
        coef_sheets = {}

        for model_name in model_order:
            x_int = x_int_by_model[model_name][mask_int]
            x_ext = x_ext_by_model[model_name][mask_ext]

            fpca_kind = best_fpca_kind if model_name == "clinic4_aec_best" else None
            if fpca_kind is None:
                oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x_int, y_int,
                                               cv=cv, method="predict_proba")[:, 1]
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof_proba = _fpca_oof_proba_predict(meta_int, aec_int_raw, shape_extra, y_int_all, mask_int,
                                                     n_fpca, include_sex, cv)
            model = LogisticRegression(max_iter=2000).fit(x_int, y_int)
            ext_proba = model.predict_proba(x_ext)[:, 1]

            # 두 threshold 선택법을 internal OOF에서 각각 고정한 뒤 external에는 그대로 적용(freeze) —
            # 이후 Youden vs fixed-Se(SE_TARGET) 비교를 위해 둘 다 저장한다
            threshold_youden = youden_threshold(y_int, oof_proba)
            threshold_se90 = fixed_sensitivity_threshold(y_int, oof_proba, SE_TARGET)

            for cohort, y, score in (("internal", y_int, oof_proba), ("external", y_ext, ext_proba)):
                auc = float(roc_auc_score(y, score))
                ci_lo, ci_hi = bootstrap_auc_ci(y, score)
                cls_youden = classification_stats(y, score, threshold_youden)
                cls_se90 = classification_stats(y, score, threshold_se90)
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                     "auc": auc, "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi,
                     "threshold_youden": threshold_youden,
                     **{f"{k}_youden": v for k, v in cls_youden.items()},
                     "threshold_se90": threshold_se90,
                     **{f"{k}_se90": v for k, v in cls_se90.items()}}
                _log(feat, model_name, cohort, s)
                summary_rows.append({"feature": feat, "model": model_name, "cohort": cohort, **s})
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

            if model_name == "clinic4_aec_best":
                input_cols_extra = [f"aec_best_{best_name}_{i}"
                                     for i in range(x_int.shape[1] - (4 if include_sex else 3))]
            else:
                input_cols_extra = []
            input_cols = (["sex_M"] if include_sex else []) + ["age", "height", "weight"] + input_cols_extra
            coef_df = pd.DataFrame({
                "term": input_cols + ["intercept"],
                "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
            })
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

        for model_name in ("clinic4", "clinic4_aec_best"):
            scanner_rows += scanner_subgroup_auc(meta_int, mask_int, y_int, scores_by_model[model_name]["internal"],
                                                  feat, model_name, "internal")
            scanner_rows += scanner_subgroup_auc(meta_ext, mask_ext, y_ext, scores_by_model[model_name]["external"],
                                                  feat, model_name, "external")

        delong_by_model = {}
        for model_name in ("clinic4_aec_best",):
            delong_int = delong_paired_auc_test(y_int, scores_by_model["clinic4"]["internal"],
                                                 scores_by_model[model_name]["internal"])
            delong_ext = delong_paired_auc_test(y_ext, scores_by_model["clinic4"]["external"],
                                                 scores_by_model[model_name]["external"])
            delong_by_model[model_name] = {"internal": delong_int, "external": delong_ext}
            for cohort, d in (("internal", delong_int), ("external", delong_ext)):
                print(f"[{feat} / {cohort}] DeLong clinic4 vs {model_name}: "
                      f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "comparison_model": model_name,
                                     "auc_clinic4": d["auc_a"], "auc_aec_model": d["auc_b"],
                                     "auc_diff": d["diff"], "z": d["z"], "p_value": d["p_value"]})

        write_sheets(feat_dir / f"{slug}_logistic_coefficients.xlsx", coef_sheets)

        curves = {
            "internal": {"y": y_int, **{m: scores_by_model[m]["internal"] for m in model_order}},
            "external": {"y": y_ext, **{m: scores_by_model[m]["external"] for m in model_order}},
        }
        plot_roc_dual(feat, model_order, curves, stats_by_model, delong_by_model,
                      feat_dir / f"{slug}_roc_curve.png")

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)
        print(f"Saved skipped feature log to {output_dir / 'skipped_features.csv'}")

    pd.DataFrame(cutoff_rows).to_csv(output_dir / "label_cutoffs.csv", index=False)
    print(f"Saved label cutoffs to {output_dir / 'label_cutoffs.csv'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    scanner_df = pd.DataFrame(scanner_rows)
    scanner_df.to_csv(output_dir / "scanner_subgroup_auc.csv", index=False)
    print(f"Saved scanner subgroup AUC to {output_dir / 'scanner_subgroup_auc.csv'}")
    model_labels = {"clinic4": "clinic4",
                     "clinic4_aec_best": f"+AEC({AEC_CANDIDATE_LABELS.get(best_name, best_name)})"}

    # PPT 슬라이드당 3개 feature(9개 = 절대값 3 + 비율 6)만 들어가므로, 발표 자료용으로 동일한
    # feature_dict 순서를 3개씩 끊어 별도 파일로도 저장(step4_aec_diagnostics.py의 top3/bottom3와 동일 패턴)
    feature_order = list(FEATURES.keys())
    for group_name, feats in (("abs", feature_order[0:3]), ("ratio_top3", feature_order[3:6]),
                               ("ratio_bottom3", feature_order[6:9])):
        group_scanner_df = scanner_df[scanner_df["feature"].isin(feats)]
        plot_scanner_subgroup_auc(group_scanner_df, output_dir / f"scanner_subgroup_auc_{group_name}.png",
                                   overall_df=summary, model_labels=model_labels)

    if not summary.empty:
        plot_auc_summary(summary, model_order, output_dir / "logistic_regression_auc_summary.png", best_name)


# scope(total/male/female)별로 이 스크립트가 저장한 logistic_regression_summary.csv(AUC/CI)와
# delong_auc_comparison.csv(clinic4 vs clinic4_aec_best paired DeLong test)를 하나의 표로 합쳐
# feature x cohort(internal/external)별 AUC_clinic4, AUC_aec_best, delta(AUC), DeLong p-value를
# 한눈에 보도록 정리한다. step3 자신의 output만 읽으므로 이 스크립트 안에 둔다(같은 output dir=같은 py 원칙)
def build_auc_delta_scope_table(scope_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(scope_dir / "logistic_regression_summary.csv")
    delong = pd.read_csv(scope_dir / "delong_auc_comparison.csv")

    pivot = summary.pivot(index=["feature", "cohort"], columns="model",
                           values=["auc", "auc_ci_lower", "auc_ci_upper"])
    pivot.columns = [f"{stat}_{model}" for stat, model in pivot.columns]
    pivot = pivot.reset_index()

    # delong_paired_auc_test(y, score_a=clinic4, score_b=aec_best)의 diff/z는 auc_a-auc_b = clinic4-aec_best
    # 부호이므로, "aec_best - clinic4"로 보여주려면 부호를 뒤집어야 함(안 뒤집으면 개선인데 음수로 표시되는 버그)
    delong_slim = delong[["feature", "cohort", "auc_diff", "z", "p_value"]].rename(
        columns={"p_value": "delong_p_value"})
    delong_slim["delta_auc_aec_best_minus_clinic4"] = -delong_slim["auc_diff"]
    delong_slim["delong_z"] = -delong_slim["z"]
    delong_slim = delong_slim.drop(columns=["auc_diff", "z"])

    merged = pivot.merge(delong_slim, on=["feature", "cohort"], how="left")
    cols = ["feature", "cohort",
            "auc_clinic4", "auc_ci_lower_clinic4", "auc_ci_upper_clinic4",
            "auc_clinic4_aec_best", "auc_ci_lower_clinic4_aec_best", "auc_ci_upper_clinic4_aec_best",
            "delta_auc_aec_best_minus_clinic4", "delong_z", "delong_p_value"]

    result = merged[cols].round(4)
    feature_order = list(FEATURES.keys())
    result["feature"] = pd.Categorical(result["feature"], categories=feature_order, ordered=True)
    result["cohort"] = pd.Categorical(result["cohort"], categories=AUC_DELTA_COHORT_ORDER, ordered=True)
    return result.sort_values(["feature", "cohort"]).reset_index(drop=True)


# scope+cohort 하나(내부 또는 외부만)의 표를 이미지(png)로 저장. delta 컬럼은 개선(양수)=빨강/악화(음수)=파랑으로
# 색칠하고(국내 관행상 상승=빨강/하락=파랑), DeLong p<0.05(유의)인 행은 전체에 굵은 테두리를 둘러 강조함
def plot_auc_delta_cohort_table(table: pd.DataFrame, scope: str, cohort: str, out_path: Path) -> None:
    feature_order = list(FEATURES.keys())
    slugs = {f: FEATURES[f][0] for f in feature_order}

    rows = []
    for _, r in table.iterrows():
        p = r["delong_p_value"]
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append([
            slugs[r["feature"]],
            f"{r['auc_clinic4']:.3f} [{r['auc_ci_lower_clinic4']:.3f}, {r['auc_ci_upper_clinic4']:.3f}]",
            f"{r['auc_clinic4_aec_best']:.3f} [{r['auc_ci_lower_clinic4_aec_best']:.3f}, "
            f"{r['auc_ci_upper_clinic4_aec_best']:.3f}]",
            f"{r['delta_auc_aec_best_minus_clinic4']:+.3f}",
            p_str,
        ])
    col_labels = ["Feature", "AUC clinic4 [95% CI]", "AUC clinic4+AEC(step2 best) [95% CI]",
                  "ΔAUC (AEC-clinic4)", "DeLong p"]
    col_widths = [0.14, 0.28, 0.35, 0.13, 0.10]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(26, 1.2 + 0.9 * n_rows))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(24)
    tbl.scale(1, 3.2)

    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_edgecolor("#cfcdc7")
            cell.set_text_props(weight="bold", color="white", fontsize=26)
            cell.set_facecolor("#161616")
            continue
        sig = table.iloc[row_i - 1]["delong_p_value"] < 0.05
        cell.set_edgecolor("#c23b22" if sig else "#cfcdc7")
        cell.set_linewidth(2.6 if sig else 1.0)
        cell.set_facecolor("#f2f1ee" if row_i % 2 == 0 else "white")
        if col_i == 3:
            delta = table.iloc[row_i - 1]["delta_auc_aec_best_minus_clinic4"]
            cell.set_text_props(color="#0055bd" if delta < 0 else "#d30909", weight="bold" if sig else "normal")

    ax.set_title(f"clinic4 vs clinic4+AEC(step2 best) AUC 비교 — {scope} ({cohort})", fontsize=30,
                 fontweight="bold", color="#161616", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC/delta table image to {out_path}")


# scope 3개(total/male/female)를 각각 읽어 표 이미지·xlsx·csv를 해당 scope 폴더(outputs/step3/{scope}/)에
# 바로 저장하고(공용 auc_delta 폴더는 쓰지 않음), 3개 scope를 합친 all 표는 scope에 속하지 않으므로
# outputs/step3 루트에 저장한다. main()에서 3개 scope의 run()이 모두 끝난 뒤(csv가 다 저장된 뒤) 마지막에 호출한다
def run_auc_delta_table() -> None:
    tables = {}
    for scope in AUC_DELTA_SCOPES:
        scope_dir = OUTPUT_DIR / scope
        if not (scope_dir / "logistic_regression_summary.csv").exists():
            print(f"[스킵] {scope_dir}에 summary csv가 없습니다.")
            continue
        table = build_auc_delta_scope_table(scope_dir)
        for cohort in AUC_DELTA_COHORT_ORDER:
            plot_auc_delta_cohort_table(table[table["cohort"] == cohort], scope, cohort,
                                         scope_dir / f"auc_delta_summary_{cohort}.png")
        table.to_csv(scope_dir / "auc_delta_summary.csv", index=False)
        print(f"Saved AUC/delta summary table to {scope_dir / 'auc_delta_summary.csv'}")
        tables[scope] = table.assign(scope=scope)

    if not tables:
        return
    all_table = pd.concat(tables.values(), ignore_index=True)
    all_cols = ["scope"] + [c for c in all_table.columns if c != "scope"]
    all_table = all_table[all_cols]

    all_table.to_csv(OUTPUT_DIR / "auc_delta_summary.csv", index=False)
    print(f"Saved AUC/delta summary table to {OUTPUT_DIR / 'auc_delta_summary.csv'}")


# internal/external 코호트를 로드/전처리 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[clinical_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_clinical_int = valid_rows(meta_int)
    mask_clinical_ext = valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper()

    run(meta_int, meta_ext, OUTPUT_DIR / "total", include_sex=True)
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True),
            meta_ext[sex_ext == sex_label].reset_index(drop=True), sub_dir, include_sex=False)

    run_auc_delta_table()


if __name__ == "__main__":
    main()
