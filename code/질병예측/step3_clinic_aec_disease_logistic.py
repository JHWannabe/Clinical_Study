from __future__ import annotations

# code/0814/step3_clinic_aec_logistic.py를 베이스로, 예측 대상을 체성분 비율/절대값(연속값 -> mean±1SD
# cutoff으로 이분화)에서 HTN/DM/CKD 진단 이진값(metadata의 HTN/DM/CKD 컬럼, 이미 0/1)으로 교체한 버전.
# HTN/DM/CKD는 이미 진단된 이진 라벨이라 label_source_values/sex_specific_cutoffs/apply_cutoff_label
# 단계(연속값 -> cutoff -> 이분화) 자체가 필요 없어 전부 제거하고, metadata 컬럼값을 그대로 라벨로 쓴다.
# 원본 step3의 select_best_shape_model은 "연속값에 linear regression, R^2로 조합 선택"이었으나 HTN/DM/CKD엔
# 연속값 proxy가 없으므로(사용자 확인: "linear는 제외해") logistic regression + internal OOF AUC 기준으로
# 5개 AEC 형태 feature 조합(SD/Skewness/상하위50%비율/FPCA/전체결합) 중 질환(HTN/DM/CKD)마다 독립적으로
# internal OOF AUC가 가장 높은 조합을 선택한다(select_best_shape_model_per_feature, 사용자 확인: "3질환
# 평균 기준 공통 조합 하나만 채택하지 말고 질환별 개별 최적으로 변경해" - 질환마다 AEC 곡선과의 연관 형태가
# 다를 수 있어 공통 조합을 강제하면 일부 질환에서 최적이 아닌 조합이 쓰이게 됨). FPCA n_components도 AUC 기반이 아니라
# internal 코호트 scree curve의 elbow(Kneedle 방식: 축 정규화 후 첫점-끝점 직선까지 수직거리 최대 지점)를
# 참고로 계산하되(사용자 확인: "R제곱값 말고 누적분산비율로 확인해" 이후 "elbow로 교체해서 재확인" -
# project_fpca_n3_vs_n12_comparison의 고정값 n=3 대신 적용), 실제 채택값은 사용자 확인 2026-08-14:
# "컴포넌트 수를 6으로 해줘"에 따라 6으로 고정한다(FPCA_N_FIXED, select_best_fpca_n 참고).
# 그 외 DeLong test/ROC 비교/AUC delta 요약표 로직은 원본 step3와 동일하게 유지한다.
# 스캐너 서브그룹 AUC/Se/Sp/Acc 분석은 code/step4_clinic_aec_disease_scanner.py로 분리했다(사용자 확인: "step4 py파일을
# 생성해서 scanner별 auc, se/sp/acc를 확인해" - [[feedback_output_dir_single_producer]] 원칙에 따라 이 스크립트는
# 모델 학습·전체 코호트 성능(AUC/Se/Sp/Acc/DeLong)까지만 담당하고, 스캐너별 재슬라이싱은 step4가 전담). 이 스크립트는
# step4가 재학습 없이 스캐너별로 재슬라이싱할 수 있도록 환자별 예측확률(oof_proba/ext_proba)과 고정 threshold,
# Manufacturer를 predictions.csv로 함께 저장한다.

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step3_disease_logistic"
AUC_DELTA_COHORT_ORDER = ["internal", "external"]

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
# 사용자 확인 2026-08-18: "input feature로 vat/sat 값도 추가해줘" 이후 "VAT_COL, SAT_COL을 따로 넣지 말고
# VAT/SAT 비율로 넣어", "구 clinic4 vs clinic5+aec feature로 비교해" - baseline은 원래 clinic4(sex/age/
# height/weight)를 그대로 유지하고, AEC를 더하는 쪽 모델에만 VAT/SAT 비율 1개를 추가로 얹어 clinic5+AEC로
# 비교한다(clinical_matrix의 include_vat_sat_ratio 플래그로 baseline/AEC-best 모델이 서로 다른 clinic
# 컬럼셋을 쓰도록 분기, load_cohort에서 비율을 미리 계산)
VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
VAT_SAT_RATIO_COL = "VAT_SAT_ratio"
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]  # clinic4/clinic3 baseline(변경 없음)
ALL_CLINICAL_COLS = CLINICAL_BASE_COLS + [VAT_SAT_RATIO_COL]  # clinic5/clinic4(no-sex)까지 포함한 결측 체크용
FPCA_COMPONENT_CANDIDATES_MAX = 20  # 사용자 확인: FPCA n_components는 R^2/AUC가 아니라 scree curve elbow로
                                     # 결정([[project_fpca_n3_vs_n12_comparison]]의 고정값 n=3 대신 계산,
                                     # step2_clinic_aec_disease_select.py와 elbow 로직 동일하게 맞춤)
FPCA_N_FIXED = 3  # 사용자 확인 2026-08-14: n=3/6/12 비교 검토 후 elbow 참고값과 동일한 3으로 재확정(step2와 동일)
MIN_POSITIVES = 2  # 이 미만이면 ROC/logistic 자체가 정의되지 않아 해당 feature/cohort를 skip

# 예측 대상(진단 이진값, 이미 0/1) -> 파일명에 쓸 slug. HTN/DM/CKD는 metadata에 이미 진단 여부로 존재하므로
# 원본 step3의 cutoff 방향(low/high) 개념이 없다
FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합, VAT/SAT SUM 절대값으로
# VAT/SAT 비율(clinic5+AEC 모델의 input feature)을 함께 계산해 둔다
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    merged[VAT_SAT_RATIO_COL] = (pd.to_numeric(merged[VAT_COL], errors="coerce")
                                  / pd.to_numeric(merged[SAT_COL], errors="coerce"))
    return merged


# raw AEC-128 행렬(n x 128)에서 환자별 SD, Skewness, 슬라이스 위치 기준 상위/하위 50% 평균 비율을 산출
# (step3_clinic_aec_logistic.py의 shape_features()와 동일)
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
# (step2_clinic_aec_disease_select.py의 _scree_and_elbow_distance와 동일 로직)
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
# 참고용으로 계산한다(step2_clinic_aec_disease_select.py의 select_best_fpca_n과 동일 로직). 다운스트림
# 예측성능(AUC/R^2)으로 n을 고르면 그 성능 자체가 선택 기준에 쓰인 데이터로 최적화되어 낙관적으로
# 부풀려질 위험이 있어(사용자 확인: "R제곱값 말고 누적분산비율로 확인해" 이후 "elbow로 교체해서 재확인"),
# FPCA/PCA 표준 scree test 관행대로 분산 감소 패턴만으로 n을 정했으나, 최종 n_components는 사용자 확인
# 2026-08-14: "컴포넌트 수를 6으로 해줘"에 따라 FPCA_N_FIXED(6)로 고정한다(elbow 값은 비교 참고용으로 유지)
def select_best_fpca_n(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))
    _, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    best_n = FPCA_N_FIXED

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] elbow 참고값 n_components = {elbow_n} (누적분산비율={cum_var[elbow_n]:.4f}, "
          f"chord-거리={dist[elbow_n]:.4f}) / 실제 채택 n_components = {best_n}(고정)")
    return best_n, cum_var


# n_components별 누적/개별 explained variance ratio, elbow 판단에 쓴 chord-거리, 선택된 best_n을 엑셀로
# 저장(그래프의 원자료)
def save_cum_var_excel(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values,
                        "individual_variance_ratio": scree.values, "elbow_chord_distance": dist.values})
    df["is_elbow_reference"] = df["n_components"] == elbow_n
    df["selected_best_n"] = df["n_components"] == best_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# internal 전체로 fit한 FPCA(n=FPCA_N_FIXED)를 external에 frozen 적용한 환자별 컴포넌트 값을 PatientID와
# 함께 엑셀로 저장(추후 확인용). select_best_shape_model_per_feature 내부에서도 동일한 fpca_scores()
# 호출로 같은 값이 재계산되지만(질환/arm에 무관하게 aec_int_raw/aec_ext_raw/n_fpca만으로 결정), 그 값은
# candidate 선택용으로만 쓰이고 별도 저장되지 않아 이 함수를 추가한다
def save_fpca_scores_excel(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, fpca_int: np.ndarray,
                            fpca_ext: np.ndarray, out_path: Path) -> None:
    cols = [f"FPCA{i}" for i in range(1, fpca_int.shape[1] + 1)]
    df_int = pd.DataFrame(fpca_int, columns=cols)
    df_int.insert(0, "PatientID", meta_int["PatientID"].to_numpy())
    df_ext = pd.DataFrame(fpca_ext, columns=cols)
    df_ext.insert(0, "PatientID", meta_ext["PatientID"].to_numpy())
    write_sheets(out_path, {"internal": df_int, "external": df_ext})


# age/height/weight(+include_vat_sat_ratio시 VAT/SAT 비율) 행렬 구성 + 표준화 + (include_sex시) sex 열 +
# (있으면) AEC feature도 별도 표준화해 결합. clinic4 baseline은 include_vat_sat_ratio=False로 기존 3개
# 임상변수만 쓰고, clinic5_aec_best 모델만 True로 VAT/SAT 비율을 추가한다(사용자 확인 2026-08-18: "구 clinic4
# vs clinic5+aec feature로 비교해")
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     aec_scaler: StandardScaler | None = None, include_sex: bool = True,
                     include_vat_sat_ratio: bool = False) -> tuple[np.ndarray, StandardScaler, StandardScaler | None]:
    cols = CLINICAL_BASE_COLS + ([VAT_SAT_RATIO_COL] if include_vat_sat_ratio else [])
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
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


# 진단 이진값(HTN/DM/CKD) 클래스 균형에 맞춘 StratifiedKFold fold 수(최소 2, 최대 N_FOLDS)
def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


# FPCA score가 들어간 AEC-best 모델(logistic)의 internal OOF 확률을 산출. PCA(고유함수 추정)를
# 전체 데이터로 먼저 fit하고 나서 CV를 돌리면 검증 fold의 곡선 정보가 고유함수 추정에 이미 들어가 OOF AUC가
# 낙관적으로 부풀려지는 data leakage가 된다 - 그래서 PCA.fit을 fold의 train 구간에서만 수행한다
# (shape_extra_full은 fold와 무관하게 이미 각 샘플별로 독립 계산된 값 - leak 아님, None이면 미포함).
# include_vat_sat_ratio는 호출하는 arm이 clinic4_aec_best(False)인지 clinic5_aec_best(True)인지에 따라 다르다
def _fpca_oof_proba_predict(meta: pd.DataFrame, aec_raw: np.ndarray, shape_extra_full: np.ndarray | None,
                             y_full: np.ndarray, mask: np.ndarray, n_fpca: int, include_sex: bool,
                             cv: StratifiedKFold, include_vat_sat_ratio: bool) -> np.ndarray:
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
                                                        include_sex=include_sex,
                                                        include_vat_sat_ratio=include_vat_sat_ratio)
        x_test, _, _ = clinical_matrix(meta_masked.iloc[test_idx], extra_test, scaler, aec_scaler,
                                        include_sex=include_sex, include_vat_sat_ratio=include_vat_sat_ratio)

        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


# 후보명 -> FPCA 포함 방식("fpca_only"=FPCA score만, "shape_all"=SD/Skew/상하위비율+FPCA 결합).
# 이 두 후보만 OOF 계산 시 fold별 PCA refit(_fpca_oof_proba_predict)을 거쳐야 함
FPCA_CANDIDATE_KINDS = {"aec_fpca": "fpca_only", "aec_shape_all": "shape_all"}


# 5개 AEC 형태 feature 후보(SD/Skewness/상하위50%비율/FPCA/전체결합) 중 각 질환(HTN/DM/CKD)별로 internal
# OOF AUC가 가장 높은 조합을 질환마다 독립적으로 선택한다(질환 간 평균을 내지 않음 - 사용자 확인: "3질환
# 평균 기준 공통 조합 하나만 채택하지 말고 질환별 개별 최적으로 변경해"). external은 전혀 쓰지 않는다
# (내부 CV로만 모델 선택, [[feedback_internal_external_validation_discipline]]). 원본 step3의
# select_best_shape_model은 연속값에 linear regression, R^2로 조합을 선택했으나 HTN/DM/CKD는 이미
# 이진값이라 연속값 proxy가 없으므로 logistic regression + AUC로 바꿨다(사용자 확인: "linear는 제외해").
# fpca_int/fpca_ext는 전체 internal 코호트로 fit한 뒤 external에는 frozen 적용하는 최종 후보 계수 산출용.
# 사용자 확인 2026-08-18: "clinic4 vs clinic4+AEC(best) vs clinic5+AEC(best)로 비교해" - clinic4_aec_best와
# clinic5_aec_best는 각각 자기 baseline(VAT/SAT 비율 유무) 위에서 독립적으로 최적 AEC 조합을 고르므로,
# 이 함수를 include_vat_sat_ratio=False/True로 두 번 호출해 서로 다른 best 조합을 얻는다
def select_best_shape_model_per_feature(meta_int: pd.DataFrame, aec_int_raw: np.ndarray, aec_ext_raw: np.ndarray,
                                         include_sex: bool, target_features: list[str],
                                         n_fpca: int, include_vat_sat_ratio: bool
                                         ) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
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

    auc_by_feature: dict[str, dict[str, float]] = {feat: {} for feat in target_features}
    for name, (aec_int, _) in candidates.items():
        fpca_kind = FPCA_CANDIDATE_KINDS.get(name)
        x_int_all = None
        if fpca_kind is None:
            x_int_all, _, _ = clinical_matrix(meta_int, aec_int, include_sex=include_sex,
                                               include_vat_sat_ratio=include_vat_sat_ratio)
        for feat in target_features:
            y_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y_all)
            y = y_all[mask].astype(int)
            if min(y.sum(), len(y) - y.sum()) < MIN_POSITIVES:
                continue
            cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)
            if fpca_kind is None:
                oof = cross_val_predict(LogisticRegression(max_iter=2000), x_int_all[mask], y, cv=cv,
                                         method="predict_proba")[:, 1]
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof = _fpca_oof_proba_predict(meta_int, aec_int_raw, shape_extra, y_all, mask, n_fpca,
                                               include_sex, cv, include_vat_sat_ratio)
            auc_by_feature[feat][name] = float(roc_auc_score(y, oof))

    best_by_feature: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for feat in target_features:
        aucs = auc_by_feature[feat]
        if not aucs:
            continue
        best_name = max(aucs, key=lambda n: aucs[n])
        print(f"[{feat} 조합 선택] 후보별 internal OOF AUC: {({k: round(v, 4) for k, v in aucs.items()})}")
        print(f"[{feat} 조합 선택] 선택된 조합 = {best_name} (internal OOF AUC={aucs[best_name]:.4f})")
        best_by_feature[feat] = (best_name, *candidates[best_name])
    return best_by_feature


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
# clinic4와 clinic5+AEC는 동일 환자에 대해 평가되므로 독립 two-sample test가 아니라 이 paired test를 써야 함
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


# internal OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold를 선택 (external에는 이 값을 고정 적용)
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# feat/model/cohort별 핵심 통계를 한 줄로 출력
def _log(feat: str, model_name: str, cohort: str, s: dict) -> None:
    print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.3f} "
          f"Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# feature(질환) 하나에 대해 internal(OOF)/external(frozen) ROC curve를 clinic4/clinic4_aec_best/
# clinic5_aec_best 3개 모델 겹쳐 그림(사용자 확인 2026-08-18: "clinic4 vs clinic4+AEC(best) vs
# clinic5+AEC(best)로 비교해"). 범례에 AUC(95%CI)와 그 질환/arm에서 선택된 AEC 조합명을 표시하고, 제목에
# DeLong test(clinic4 vs 각 AEC-best 모델) p-value 2개를 함께 표기해 "AEC 단독 효과"와 "VAT/SAT비+AEC
# 결합 효과"를 한눈에 구분한다
def plot_roc_dual(feat: str, model_order: list[str], curves: dict[str, dict[str, np.ndarray]],
                   stats_by_model: dict[str, dict[str, dict]],
                   delong_by_model: dict[str, dict[str, dict]], out_path: Path,
                   aec_label_by_model: dict[str, str]) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#2a78d6", "clinic4_aec_best": "#1baf7a", "clinic5_aec_best": "#e2622e"}
    labels = {
        "clinic4": "clinic4",
        "clinic4_aec_best": f"clinic4 + AEC({aec_label_by_model.get('clinic4_aec_best', '')})",
        "clinic5_aec_best": f"clinic5(+VAT/SAT비) + AEC({aec_label_by_model.get('clinic5_aec_best', '')})",
    }
    cohorts = ["internal", "external"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cohort in zip(axes, cohorts):
        for model_name in model_order:
            y = curves[cohort]["y"]
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=1.8,
                    label=f"{labels[model_name]} AUC={s['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)

        def _p_label(model_name: str) -> str:
            p = delong_by_model[model_name][cohort]["p_value"]
            return "p<0.001" if p < 0.001 else f"p={p:.3f}"

        title = (f"{feat} ({cohort})\n"
                 f"AEC: {_p_label('clinic4_aec_best')} / VAT/SAT비+AEC: {_p_label('clinic5_aec_best')}")
        ax.set_title(title, fontsize=18, fontweight="bold", color=INK_PRIMARY)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# select_best_shape_model_per_feature()가 고른 후보 이름 -> 범례에 쓸 한글 설명(어떤 AEC 형태 feature인지 명시)
AEC_CANDIDATE_LABELS = {
    "aec_sd": "SD(표준편차)",
    "aec_skew": "Skewness(비대칭도)",
    "aec_uplow_ratio": "상하위50%비율",
    "aec_fpca": "FPCA",
    "aec_shape_all": "전체결합: SD·Skew·상하위비율·FPCA",
}


# 전체 feature(질환)에 걸쳐 AUC(95%CI)를 3개 모델 x internal/external로 비교하는 막대그래프.
# AEC-best 모델은 질환/arm마다 선택된 AEC 조합이 달라 범례에 특정 조합명을 못 박지 않고 일반화된 라벨을
# 쓴다 - 질환별 실제 조합명은 auc_delta_summary 표의 "AEC 조합" 열과 ROC plot 범례에서 확인 가능
def plot_auc_summary(summary: pd.DataFrame, model_order: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    colors = {"clinic4": "#6b6a66", "clinic4_aec_best": "#1baf7a", "clinic5_aec_best": "#e2622e"}
    labels = {"clinic4": "clinic4", "clinic4_aec_best": "+AEC(질환별 최적조합)",
              "clinic5_aec_best": "+VAT/SAT비+AEC(질환별 최적조합)"}
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(6 * len(features) / 2 + 3, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_order):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, label=labels[model_name], color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=20, rotation=0)
        ax.set_ylim(0.5, 1.0)
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


# clinic4(sex/age/height/weight, include_sex=False면 clinic3) baseline과, 여기에 AEC만 추가한
# clinic4_aec_best, VAT/SAT 비율+AEC를 함께 추가한 clinic5_aec_best까지 3개 모델로 HTN/DM/CKD 3개 진단을
# logistic regression 예측한다(사용자 확인 2026-08-18: "clinic4 vs clinic4+AEC(best) vs clinic5+AEC(best)로
# 비교해" - clinic4_aec_best와 clinic5_aec_best를 나란히 두어 AEC 단독 효과와 VAT/SAT+AEC 결합 효과를
# 분리해서 본다). metadata의 HTN/DM/CKD 값을 그대로 라벨로 쓰고(cutoff 도출 없음), internal OOF로 모델을
# 학습/평가하고 external에는 고정 모델을 1회만 적용(freeze), AUC는 DeLong test(clinic4 vs 각 AEC-best
# 모델)로 비교, 스캐너별 서브그룹 AUC도 함께 산출. AEC 조합은 질환마다, 그리고 arm(clinic4_aec_best/
# clinic5_aec_best)마다 독립적으로 고른 최적 조합을 쓴다(select_best_shape_model_per_feature)
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    n_fpca, cum_var = select_best_fpca_n(aec_int_raw)
    save_cum_var_excel(cum_var, n_fpca, output_dir / "fpca_cumulative_variance.xlsx")

    fpca_int_full, fpca_ext_full = fpca_scores(aec_int_raw, aec_ext_raw, n_fpca)
    save_fpca_scores_excel(meta_int, meta_ext, fpca_int_full, fpca_ext_full, output_dir / "fpca_scores.xlsx")

    # internal/external 양쪽에서 최소 표본을 만족하는 feature만 골라내고, 조합 선택도 이 대상에만 수행한다
    valid_features: list[str] = []
    skipped = []
    for feat in FEATURES:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int = np.isfinite(y_int_all)
        mask_val_ext = np.isfinite(y_ext_all)
        n_pos_int = int(y_int_all[mask_val_int].sum())
        n_neg_int = int(mask_val_int.sum() - n_pos_int)
        n_pos_ext = int(y_ext_all[mask_val_ext].sum())
        n_neg_ext = int(mask_val_ext.sum() - n_pos_ext)
        print(f"[{feat}] internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = (f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만이라 logistic regression/ROC 산출이 불가함 "
                   f"(internal pos={n_pos_int} neg={n_neg_int}, external pos={n_pos_ext} neg={n_neg_ext})")
            print(msg)
            skipped.append({"feature": feat, "reason": msg, "n_pos_internal": n_pos_int,
                             "n_neg_internal": n_neg_int, "n_pos_external": n_pos_ext, "n_neg_external": n_neg_ext})
            continue
        valid_features.append(feat)

    # clinic4_aec_best/clinic5_aec_best는 각자의 baseline(VAT/SAT 비율 유무) 위에서 독립적으로 최적 AEC
    # 조합을 고른다 - 같은 AEC 조합이 두 baseline 모두에 최적이라는 보장이 없으므로 따로 계산한다
    best_by_feature_c4aec = select_best_shape_model_per_feature(
        meta_int, aec_int_raw, aec_ext_raw, include_sex, valid_features, n_fpca, include_vat_sat_ratio=False)
    best_by_feature_c5aec = select_best_shape_model_per_feature(
        meta_int, aec_int_raw, aec_ext_raw, include_sex, valid_features, n_fpca, include_vat_sat_ratio=True)

    # AEC-best 모델의 internal OOF 확률(fold별 PCA refit)에 필요 - 조합이 aec_shape_all이면
    # SD/Skew/상하위비율도 함께 결합해야 하므로 여기서 한 번 구해 둠(모든 질환에 공통, feature 무관)
    shape_int_for_best = shape_features(aec_int_raw)
    shape_int_stack = np.column_stack(
        [shape_int_for_best["sd"], shape_int_for_best["skew"], shape_int_for_best["upper_lower_ratio"]])

    model_order = ["clinic4", "clinic4_aec_best", "clinic5_aec_best"]
    summary_rows = []
    delong_rows = []
    predictions_rows = []  # step4_clinic_aec_disease_scanner.py가 스캐너별 재슬라이싱에 쓸 환자별 예측확률

    for feat in valid_features:
        slug = FEATURES[feat]
        best_name_c4, aec_int_best_c4, aec_ext_best_c4 = best_by_feature_c4aec[feat]
        best_name_c5, aec_int_best_c5, aec_ext_best_c5 = best_by_feature_c5aec[feat]
        best_name_by_model = {"clinic4_aec_best": best_name_c4, "clinic5_aec_best": best_name_c5}
        best_fpca_kind_by_model = {m: FPCA_CANDIDATE_KINDS.get(n) for m, n in best_name_by_model.items()}

        aec_by_model = {
            "clinic4": {"aec_int": None, "aec_ext": None, "include_vat_sat_ratio": False},
            "clinic4_aec_best": {"aec_int": aec_int_best_c4, "aec_ext": aec_ext_best_c4,
                                  "include_vat_sat_ratio": False},
            "clinic5_aec_best": {"aec_int": aec_int_best_c5, "aec_ext": aec_ext_best_c5,
                                  "include_vat_sat_ratio": True},
        }
        x_int_by_model: dict[str, np.ndarray] = {}
        x_ext_by_model: dict[str, np.ndarray] = {}
        for model_name in model_order:
            spec = aec_by_model[model_name]
            x_int, clinic_scaler, aec_scaler = clinical_matrix(
                meta_int, spec["aec_int"], include_sex=include_sex,
                include_vat_sat_ratio=spec["include_vat_sat_ratio"])
            x_ext, _, _ = clinical_matrix(
                meta_ext, spec["aec_ext"], clinic_scaler, aec_scaler, include_sex=include_sex,
                include_vat_sat_ratio=spec["include_vat_sat_ratio"])
            x_int_by_model[model_name], x_ext_by_model[model_name] = x_int, x_ext

        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int = y_int_all[mask_int].astype(int)
        y_ext = y_ext_all[mask_ext].astype(int)

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        stats_by_model: dict[str, dict[str, dict]] = {m: {} for m in model_order}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in model_order}
        threshold_by_model: dict[str, float] = {}
        coef_sheets = {}

        for model_name in model_order:
            x_int = x_int_by_model[model_name][mask_int]
            x_ext = x_ext_by_model[model_name][mask_ext]

            fpca_kind = best_fpca_kind_by_model.get(model_name)
            if fpca_kind is None:
                oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x_int, y_int,
                                               cv=cv, method="predict_proba")[:, 1]
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof_proba = _fpca_oof_proba_predict(meta_int, aec_int_raw, shape_extra, y_int_all, mask_int,
                                                     n_fpca, include_sex, cv,
                                                     aec_by_model[model_name]["include_vat_sat_ratio"])
            model = LogisticRegression(max_iter=2000).fit(x_int, y_int)
            ext_proba = model.predict_proba(x_ext)[:, 1]

            threshold = youden_threshold(y_int, oof_proba)
            threshold_by_model[model_name] = threshold

            for cohort, meta_c, mask_c, y, score in (
                ("internal", meta_int, mask_int, y_int, oof_proba),
                ("external", meta_ext, mask_ext, y_ext, ext_proba),
            ):
                auc = float(roc_auc_score(y, score))
                cls_stats = classification_stats(y, score, threshold)
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                     "auc": auc, "threshold": threshold, **cls_stats}
                _log(feat, model_name, cohort, s)
                row = {"feature": feat, "model": model_name, "cohort": cohort, **s}
                if model_name in best_name_by_model:
                    row["aec_combo"] = best_name_by_model[model_name]
                summary_rows.append(row)
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

                patient_ids = meta_c["PatientID"].to_numpy()[mask_c]
                manufacturers = meta_c["Manufacturer"].astype(str).to_numpy()[mask_c]
                predictions_rows.append(pd.DataFrame({
                    "feature": feat, "model": model_name, "cohort": cohort,
                    "patient_id": patient_ids, "manufacturer": manufacturers,
                    "y": y, "score": score, "threshold": threshold,
                }))

            include_ratio = aec_by_model[model_name]["include_vat_sat_ratio"]
            n_clinical = (1 if include_sex else 0) + len(CLINICAL_BASE_COLS) + (1 if include_ratio else 0)
            if model_name in best_name_by_model:
                input_cols_extra = [f"aec_best_{best_name_by_model[model_name]}_{i}"
                                     for i in range(x_int.shape[1] - n_clinical)]
            else:
                input_cols_extra = []
            clinical_terms = ["age", "height", "weight"] + (["vat_sat_ratio"] if include_ratio else [])
            input_cols = (["sex_M"] if include_sex else []) + clinical_terms + input_cols_extra
            coef_df = pd.DataFrame({
                "term": input_cols + ["intercept"],
                "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
            })
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

        delong_by_model = {}
        for model_name in ("clinic4_aec_best", "clinic5_aec_best"):
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
        aec_label_by_model = {m: AEC_CANDIDATE_LABELS.get(n, n) for m, n in best_name_by_model.items()}
        plot_roc_dual(feat, model_order, curves, stats_by_model, delong_by_model,
                      feat_dir / f"{slug}_roc_curve.png", aec_label_by_model)

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)
        print(f"Saved skipped feature log to {output_dir / 'skipped_features.csv'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    predictions = pd.concat(predictions_rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved per-patient predictions to {output_dir / 'predictions.csv'}")

    if not summary.empty:
        plot_auc_summary(summary, model_order, output_dir / "logistic_regression_auc_summary.png")


# AEC-best 두 arm(clinic4_aec_best/clinic5_aec_best) 각각의 delong_auc_comparison.csv 행(comparison_model로
# 구분)에서 clinic4 대비 delta/AUC/p-value를 뽑아 "{colname}_{model_name}" 열로 만든다
def _delong_slim_for_model(delong: pd.DataFrame, model_name: str, colname: str) -> pd.DataFrame:
    d = delong[delong["comparison_model"] == model_name][["feature", "cohort", "auc_diff", "z", "p_value"]].copy()
    # delong_paired_auc_test(y, score_a=clinic4, score_b=aec_best)의 diff/z는 auc_a-auc_b = clinic4-aec_best
    # 부호이므로, "aec_best - clinic4"로 보여주려면 부호를 뒤집어야 함(안 뒤집으면 개선인데 음수로 표시되는 버그)
    d[f"delta_auc_{colname}_minus_clinic4"] = -d["auc_diff"]
    d[f"delong_z_{colname}"] = -d["z"]
    d = d.rename(columns={"p_value": f"delong_p_{colname}"}).drop(columns=["auc_diff", "z"])
    return d


# 이 스크립트가 저장한 logistic_regression_summary.csv(AUC/CI)와 delong_auc_comparison.csv(clinic4 vs
# clinic4_aec_best/clinic5_aec_best paired DeLong test)를 하나의 표로 합쳐 feature x cohort(internal/
# external)별 AUC_clinic4/AUC_clinic4_aec_best/AUC_clinic5_aec_best, 두 arm 각각의 delta(AUC), DeLong
# p-value를 한눈에 보도록 정리한다(사용자 확인 2026-08-18: "clinic4 vs clinic4+AEC(best) vs clinic5+AEC(best)
# 로 비교해"). 질환/arm마다 선택된 AEC 조합이 다르므로(select_best_shape_model_per_feature) aec_combo 열도
# arm별로 붙인다. step3 자신의 output만 읽으므로 이 스크립트 안에 둔다(같은 output dir=같은 py 원칙)
def build_auc_delta_scope_table(scope_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(scope_dir / "logistic_regression_summary.csv")
    delong = pd.read_csv(scope_dir / "delong_auc_comparison.csv")

    pivot = summary.pivot(index=["feature", "cohort"], columns="model", values="auc")
    pivot.columns = [f"auc_{model}" for model in pivot.columns]
    pivot = pivot.reset_index()

    for model_name in ("clinic4_aec_best", "clinic5_aec_best"):
        combo_map = (summary[summary["model"] == model_name]
                     .drop_duplicates(subset=["feature"]).set_index("feature")["aec_combo"])
        pivot[f"aec_combo_{model_name}"] = pivot["feature"].map(combo_map).map(
            lambda n: AEC_CANDIDATE_LABELS.get(n, n))

    merged = pivot
    for model_name in ("clinic4_aec_best", "clinic5_aec_best"):
        merged = merged.merge(_delong_slim_for_model(delong, model_name, model_name),
                               on=["feature", "cohort"], how="left")

    cols = ["feature", "cohort", "aec_combo_clinic4_aec_best", "aec_combo_clinic5_aec_best",
            "auc_clinic4", "auc_clinic4_aec_best", "auc_clinic5_aec_best",
            "delta_auc_clinic4_aec_best_minus_clinic4", "delong_p_clinic4_aec_best",
            "delta_auc_clinic5_aec_best_minus_clinic4", "delong_p_clinic5_aec_best"]

    result = merged[cols].round(4)
    feature_order = list(FEATURES.keys())
    result["feature"] = pd.Categorical(result["feature"], categories=feature_order, ordered=True)
    result["cohort"] = pd.Categorical(result["cohort"], categories=AUC_DELTA_COHORT_ORDER, ordered=True)
    return result.sort_values(["feature", "cohort"]).reset_index(drop=True)


# internal+external 전체 코호트를 한 표(이미지 하나)로 저장(기존엔 cohort별로 파일을 따로 만들었으나
# 사용자 확인: "두 파일을 통합해" - 표 앞쪽에 Cohort 열을 추가하고 internal 6행 -> external 6행 순으로
# 이어 붙인 뒤, 두 블록 경계(external 첫 행)에 굵은 구분선을 둔다). delta 컬럼은 개선(양수)=빨강/악화(음수)=
# 파랑으로 색칠하고(국내 관행상 상승=빨강/하락=파랑), DeLong p<0.05(유의)인 행은 전체에 굵은 테두리를 둘러 강조함
def plot_auc_delta_combined_table(table: pd.DataFrame, out_path: Path) -> None:
    feature_order = list(FEATURES.keys())
    slugs = {f: FEATURES[f] for f in feature_order}

    table = table.sort_values(["cohort", "feature"]).reset_index(drop=True)
    rows = []
    for _, r in table.iterrows():
        p_c4 = r["delong_p_clinic4_aec_best"]
        p_c5 = r["delong_p_clinic5_aec_best"]
        p_c4_str = "<0.001" if p_c4 < 0.001 else f"{p_c4:.3f}"
        p_c5_str = "<0.001" if p_c5 < 0.001 else f"{p_c5:.3f}"
        rows.append([
            r["cohort"],
            slugs[r["feature"]],
            r["aec_combo_clinic4_aec_best"],
            r["aec_combo_clinic5_aec_best"],
            f"{r['auc_clinic4']:.3f}",
            f"{r['auc_clinic4_aec_best']:.3f}",
            f"{r['auc_clinic5_aec_best']:.3f}",
            f"{r['delta_auc_clinic4_aec_best_minus_clinic4']:+.3f}",
            p_c4_str,
            f"{r['delta_auc_clinic5_aec_best_minus_clinic4']:+.3f}",
            p_c5_str,
        ])
    col_labels = ["Cohort", "Feature", "AEC 조합\n(clinic4+AEC)", "AEC 조합\n(clinic5+AEC)",
                  "AUC\nclinic4", "AUC\nclinic4+AEC", "AUC\nclinic5+AEC",
                  "ΔAUC\n(clinic4+AEC-clinic4)", "DeLong p\n(clinic4+AEC)",
                  "ΔAUC\n(clinic5+AEC-clinic4)", "DeLong p\n(clinic5+AEC)"]
    col_widths = [0.07, 0.07, 0.12, 0.12, 0.08, 0.09, 0.09, 0.11, 0.09, 0.11, 0.09]
    delta_col_c4aec, delta_col_c5aec = 7, 9

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(36, 1.4 + 0.9 * n_rows))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    tbl.scale(1, 3.6)

    # 코호트가 바뀌는 첫 행(row_i)을 기록해두고, 그 행의 "위쪽"에만 굵은 구분선을 별도로 그린다.
    # 유의성 강조는 delta 컬럼 텍스트 색상(빨강/파랑)만으로 표시하고, 테두리·배경은 전부
    # 동일한 연한 회색 테두리 + 흰 배경으로 통일한다(사용자 요청: 빨간 테두리·회색 음영 제거)
    block_start = table["cohort"].ne(table["cohort"].shift()).to_numpy()
    divider_row_i = None
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_edgecolor("#cfcdc7")
            cell.set_text_props(weight="bold", color="white", fontsize=18)
            cell.set_facecolor("#161616")
            continue
        r = table.iloc[row_i - 1]
        cell.set_edgecolor("#cfcdc7")
        cell.set_linewidth(1.0)
        cell.set_facecolor("white")
        if block_start[row_i - 1] and row_i > 1:
            divider_row_i = row_i
        if col_i == delta_col_c4aec:
            delta = r["delta_auc_clinic4_aec_best_minus_clinic4"]
            sig = r["delong_p_clinic4_aec_best"] < 0.05
            cell.set_text_props(color="#0055bd" if delta < 0 else "#d30909", weight="bold" if sig else "normal")
        elif col_i == delta_col_c5aec:
            delta = r["delta_auc_clinic5_aec_best_minus_clinic4"]
            sig = r["delong_p_clinic5_aec_best"] < 0.05
            cell.set_text_props(color="#0055bd" if delta < 0 else "#d30909", weight="bold" if sig else "normal")

    ax.set_title("clinic4 vs clinic4+AEC(best) vs clinic5+AEC(best) AUC 비교 (internal vs external)", fontsize=26,
                 fontweight="bold", color="#161616", pad=10)
    fig.tight_layout()

    # 셀 좌표는 fig.tight_layout()으로 축 위치가 확정된 뒤에야 최종값이 되므로, 구분선은
    # tight_layout() 이후 실제 렌더 픽셀 window extent를 읽어 figure 좌표로 변환해 그린다
    # (tight_layout보다 먼저 그리면 이후 레이아웃 변경으로 위치가 어긋남)
    if divider_row_i is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left_cell = tbl[(divider_row_i, 0)]
        right_cell = tbl[(divider_row_i, len(col_labels) - 1)]
        bbox_left = left_cell.get_window_extent(renderer)
        bbox_right = right_cell.get_window_extent(renderer)
        inv = fig.transFigure.inverted()
        x0, y_top = inv.transform((bbox_left.x0, bbox_left.y1))
        x1, _ = inv.transform((bbox_right.x1, bbox_right.y1))
        line = plt.Line2D([x0, x1], [y_top, y_top], transform=fig.transFigure,
                           color="#161616", linewidth=4.0, solid_capstyle="butt", zorder=10)
        fig.add_artist(line)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC/delta table image to {out_path}")


# 이 스크립트가 저장한 logistic_regression_summary.csv/delong_auc_comparison.csv를 읽어 표 이미지·xlsx·csv를
# OUTPUT_DIR 바로 아래에 저장한다(성별 층화분석 제거로 scope 개념이 없어짐, 사용자 확인: "female, male 성별
# 층화분석은 제거하고 total만 진행"). main()에서 run()이 끝난 뒤(csv가 저장된 뒤) 마지막에 호출한다
def run_auc_delta_table() -> None:
    if not (OUTPUT_DIR / "logistic_regression_summary.csv").exists():
        print(f"[스킵] {OUTPUT_DIR}에 summary csv가 없습니다.")
        return
    table = build_auc_delta_scope_table(OUTPUT_DIR)
    plot_auc_delta_combined_table(table, OUTPUT_DIR / "auc_delta_summary.png")
    table.to_csv(OUTPUT_DIR / "auc_delta_summary.csv", index=False)
    print(f"Saved AUC/delta summary table to {OUTPUT_DIR / 'auc_delta_summary.csv'}")
    table.to_excel(OUTPUT_DIR / "auc_delta_summary.xlsx", sheet_name="total", index=False)
    print(f"Saved AUC/delta summary table to {OUTPUT_DIR / 'auc_delta_summary.xlsx'}")


# internal/external 코호트를 로드/전처리 후 전체 코호트(sex 포함)로 run()을 실행(성별 층화분석 제거)
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[ALL_CLINICAL_COLS].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_clinical_int = valid_rows(meta_int)
    mask_clinical_ext = valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    run(meta_int, meta_ext, OUTPUT_DIR, include_sex=True)
    run_auc_delta_table()


if __name__ == "__main__":
    main()
