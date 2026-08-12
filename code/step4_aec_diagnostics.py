from __future__ import annotations

# step2_clinic_aec_ratio.py와 동일한 clinic4+AEC 형태 feature 모델 구조(비율 4종 + VAT/SAT/Total Fat
# 절대값)를 그대로 재사용해 두 가지 진단을 추가한다.
# (1) 스캐너(Manufacturer)별 서브그룹 R^2: feature별 internal-best AEC 모델과 clinic4 baseline의
#     예측값을 스캐너별로 나눠 재계산(재학습 없음) -> 특정 스캐너에만 성능이 쏠려있는지 확인.
# (2) AEC 파생 feature(SD/Skewness/FPCA score 등) 표준화 검토: clinic4처럼 내부 코호트 기준으로
#     표준화해 clinic4와 스케일을 맞추고, OLS는 스케일 불변이라는 이론을 R^2 비교로 1회 검증한다.
# (3) 스캐너별 R^2(위 (1))와 step3_clinic_aec_logistic.py의 스캐너별 AUC를 feature 평균으로 묶은 요약표.
# outputs/step4 하위에만 쓰는 output(같은 output dir=같은 py 원칙)이라 이 파일에 통합했다.
# (step3 결과만으로 만드는 ΔAUC/DeLong p-value 요약표는 step3 자신의 output이라 step3_clinic_aec_logistic.py로 옮겼다)

import textwrap
from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4_scanner_subgroup"
STEP3_DIR = PROJECT_ROOT / "outputs" / "step3_logistic"  # step3_clinic_aec_logistic.py가 쓴 scanner_subgroup_auc.csv 위치

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COMPONENT_CANDIDATES = list(range(1, 21))
SCANNER_MIN_N = 30  # 스캐너별 서브그룹 분석에서 이 인원 미만인 스캐너는 제외

FEATURES = {
    "VAT_TotalFat_ratio": "VAT/TotalFat",
    "VAT_SAT_ratio": "VAT/SAT",
    "VAT_TAMA_ratio": "VAT/TAMA",
    "TotalFat_TAMA_ratio": "TotalFat/TAMA",
    "SAT_TotalFat_ratio": "SAT/TotalFat",
    "SAT_TAMA_ratio": "SAT/TAMA",
}
ABS_FEATURES = {
    "VAT(내장지방)_SUM": "VAT",
    "SAT(피하지방)_SUM": "SAT",
    "Total Fat_SUM": "Total Fat",
}


# VAT/Total Fat, VAT/SAT, VAT/TAMA, Total Fat/TAMA, SAT/Total Fat, SAT/TAMA 6개 비율 컬럼을 원본 SUM 컬럼으로부터 산출해 추가
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
def shape_features(aec_matrix: np.ndarray) -> dict[str, np.ndarray]:
    sd = aec_matrix.std(axis=1, ddof=1)
    skewness = stats.skew(aec_matrix, axis=1)

    half = N_SLICES // 2
    upper_mean = aec_matrix[:, :half].mean(axis=1)
    lower_mean = aec_matrix[:, half:].mean(axis=1)
    upper_lower_ratio = upper_mean / lower_mean

    return {"sd": sd, "skew": skewness, "upper_lower_ratio": upper_lower_ratio}


# internal 코호트 raw AEC-128 곡선에 PCA를 fit해 top-n_components score를 산출, external에는 frozen 적용
def fpca_scores(aec_int: np.ndarray, aec_ext: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=SEED).fit(aec_int)
    return pca.transform(aec_int), pca.transform(aec_ext), pca


# n_components 후보별 internal OOF R^2를 target_features 평균으로 산출해 best n_components를 결정.
# (n_components 자체의 표/시각화는 step1_aec_fpca.py의 몫이라 여기서는 모델 구성에 필요한 값만 계산)
# PCA(고유함수 추정)는 반드시 각 fold의 학습 데이터에서만 fit하고 검증 fold는 그 고유함수에 transform만 해야
# 함 - 전체 데이터로 먼저 PCA를 fit하면 검증 fold의 곡선 정보가 고유함수 추정에 이미 들어가 OOF R^2가
# 낙관적으로 부풀려지는 data leakage가 됨. 그래서 PCA.fit을 fold 루프 안(train_idx에만)으로 옮긴다
def select_best_fpca_n(meta_int: pd.DataFrame, aec_int_raw: np.ndarray, include_sex: bool,
                        target_features: list[str], cv: KFold) -> int:
    mean_by_n = {}
    r2_by_n_feat: dict[int, list[float]] = {n: [] for n in FPCA_COMPONENT_CANDIDATES}
    for feat in target_features:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        mask_int = np.isfinite(y_int_all)
        y_int = y_int_all[mask_int]
        aec_feat = aec_int_raw[mask_int]
        meta_feat = meta_int.loc[mask_int].reset_index(drop=True)

        for n in FPCA_COMPONENT_CANDIDATES:
            oof = np.empty(len(y_int))
            for train_idx, test_idx in cv.split(aec_feat):
                pca = PCA(n_components=n, random_state=SEED).fit(aec_feat[train_idx])
                fpca_train = pca.transform(aec_feat[train_idx])
                fpca_test = pca.transform(aec_feat[test_idx])

                x_train, scaler, aec_scaler = clinical_matrix(meta_feat.iloc[train_idx], fpca_train,
                                                                include_sex=include_sex)
                x_test, _, _ = clinical_matrix(meta_feat.iloc[test_idx], fpca_test, scaler, aec_scaler,
                                                include_sex=include_sex)

                model = LinearRegression().fit(x_train, y_int[train_idx])
                oof[test_idx] = model.predict(x_test)
            r2_by_n_feat[n].append(r2_score(y_int, oof))

    for n in FPCA_COMPONENT_CANDIDATES:
        mean_by_n[n] = float(np.mean(r2_by_n_feat[n]))

    best_n = max(mean_by_n, key=lambda n: mean_by_n[n])
    print(f"[FPCA] 선택된 best n_components = {best_n} (internal OOF 평균 R^2={mean_by_n[best_n]:.4f})")
    return best_n


# FPCA score가 들어간 모델(clinic4_aec_fpca/clinic4_aec_shape_all)의 internal OOF 예측을 산출.
# select_best_fpca_n과 동일한 이유로 PCA를 fold의 train 구간에서만 fit하고 검증 fold는 transform만 한다
# (shape_extra_full은 fold와 무관하게 이미 각 샘플별로 독립 계산된 SD/Skew/상하위비율 등 - leak 아님, None이면 미포함)
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


# 모델명 -> FPCA 포함 방식. "fpca_only"는 FPCA score만, "shape_all"은 SD/Skew/상하위비율+FPCA 결합.
# 이 두 모델만 internal OOF 계산 시 _fpca_oof_predict(fold별 PCA refit)를 거쳐야 함
FPCA_MODEL_KINDS = {"clinic4_aec_fpca": "fpca_only", "clinic4_aec_shape_all": "shape_all"}


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + (있으면) AEC 형태 feature도 별도 표준화해 결합.
# scaler/aec_scaler 생략 시 새로 학습(내부 코호트용) — AEC feature(SD/FPCA score 등)는 clinic4와 스케일이
# 크게 달라 raw로 섞으면 계수 해석이 왜곡되므로 clinic4와 동일하게 내부 코호트 기준으로 표준화한다(항목7)
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     aec_scaler: StandardScaler | None = None, include_sex: bool = True):
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


# clinic4+AEC 결합 모델에서 AEC feature를 표준화해도(OLS는 스케일 불변) R²가 그대로인지 1회 검증 —
# 항목7(FPCA/형태 feature 표준화 방법 검토)의 결과를 코드로 증빙하기 위한 sanity check
def verify_aec_scaling_invariance(meta_int: pd.DataFrame, aec_extra: np.ndarray, y_all: np.ndarray,
                                   include_sex: bool) -> None:
    mask = np.isfinite(y_all)
    y = y_all[mask]

    rest = meta_int[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scaler = StandardScaler().fit(rest)
    clinic = scaler.transform(rest)
    if include_sex:
        clinic = np.column_stack(
            [(meta_int["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), clinic])
    x_raw = np.column_stack([clinic, aec_extra])[mask]
    x_scaled, _, _ = clinical_matrix(meta_int, aec_extra, include_sex=include_sex)
    x_scaled = x_scaled[mask]

    r2_raw = r2_score(y, LinearRegression().fit(x_raw, y).predict(x_raw))
    r2_scaled = r2_score(y, LinearRegression().fit(x_scaled, y).predict(x_scaled))
    print(f"[표준화 검증] AEC feature raw R²={r2_raw:.6f} vs 표준화 후 R²={r2_scaled:.6f} "
          f"(diff={abs(r2_raw - r2_scaled):.2e}, OLS는 이론상 스케일 불변)")


# R^2/RMSE/MAE/Pearson r(유의성)과 R^2의 bootstrap 95% CI를 산출
def regression_significance_stats(y: np.ndarray, pred: np.ndarray, n_boot: int = 3000, seed: int = SEED) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(np.mean(np.abs(y - pred)))
    r, p_value = stats.pearsonr(y, pred)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot_r2 = np.array([r2_score(y[bi], pred[bi]) for bi in rng.integers(0, n, size=(n_boot, n))])
    ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])

    return {"n": int(n), "r2": float(r2_score(y, pred)), "r2_ci_lower": float(ci_lo), "r2_ci_upper": float(ci_hi),
            "rmse": rmse, "mae": mae, "pearson_r": r, "p_value": p_value}


# feat/model/cohort별 핵심 통계를 한 줄로 출력
def _log(feat: str, model_name: str, cohort: str, s: dict) -> None:
    print(f"[{feat} / {model_name} / {cohort}] R2={s['r2']:.4f} 95%CI=[{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}] "
          f"RMSE={s['rmse']:.3f} MAE={s['mae']:.3f} Pearson p={s['p_value']:.3e}")


# 이미 계산된 예측값(oof 또는 pred_ext)을 스캐너(Manufacturer)별로 나눠 R^2 등을 재계산.
# 재학습 없이 예측값만 재슬라이싱하므로 비용이 거의 없음. SCANNER_MIN_N 미만인 스캐너는 통계가
# 불안정해 제외한다. mask는 해당 feature의 결측 제외 mask로, y/pred와 순서가 일치해야 함
def scanner_subgroup_r2(meta: pd.DataFrame, mask: np.ndarray, y: np.ndarray, pred: np.ndarray,
                         feat: str, model_name: str, cohort: str) -> list[dict]:
    scanners = meta["Manufacturer"].astype(str).to_numpy()[mask]
    rows = []
    for scanner in pd.unique(scanners):
        sel = scanners == scanner
        n = int(sel.sum())
        if n < SCANNER_MIN_N:
            continue
        s = regression_significance_stats(y[sel], pred[sel])
        rows.append({"feature": feat, "model": model_name, "cohort": cohort, "scanner": scanner, "n": n, **s})
    return rows


# feature별로 clinic4 vs (internal 기준) 최고 AEC 모델의 스캐너별 R^2를 나란히 막대그래프로 비교.
# overall_df(스캐너로 나누지 않고 전체 표본을 합쳐 계산한 요약, feature/model/cohort/r2 컬럼)를 주면
# 모델별 색상과 맞춘 점선으로 "스캐너 통합 시 성능" 기준선을 함께 그린다
def plot_scanner_subgroup_r2(scanner_df: pd.DataFrame, features_dict: dict[str, str], out_path: Path,
                              overall_df: pd.DataFrame | None = None) -> None:
    if scanner_df.empty:
        print(f"[스캐너 서브그룹] 표본이 {SCANNER_MIN_N}명 이상인 스캐너가 없어 그래프를 생략합니다.")
        return

    features = [f for f in features_dict if f in scanner_df["feature"].unique()]
    cohorts = ["internal", "external"]
    # 코호트(열)별로 등장하는 스캐너 수가 다르므로(internal 4개 vs external 7개 등), 라벨이 겹치지
    # 않도록 열 너비를 스캐너 개수에 비례시킨다(회전 대신 카테고리당 고정 폭 확보). 긴 스캐너명은
    # textwrap으로 여러 줄로 접어 카테고리당 필요 폭을 줄임 -> 전체 캔버스가 과도하게 넓어져
    # 축소 보기에서 글자가 오히려 작아 보이는 문제를 방지
    per_category_width = 3.2
    label_wrap_width = 11
    n_scanners_by_col = [
        max((scanner_df[(scanner_df["feature"] == f) & (scanner_df["cohort"] == c)]["scanner"].nunique()
             for f in features), default=1)
        for c in cohorts
    ]
    col_widths = [max(n, 1) * per_category_width for n in n_scanners_by_col]
    fig, axes = plt.subplots(len(features), 2, figsize=(sum(col_widths), 11 * len(features)), squeeze=False,
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
            r2_arr = sub["r2"].to_numpy(dtype=float)
            for i, model_name in enumerate(models_here):
                vals = [float(np.mean(r2_arr[(scanner_arr == s) & (model_arr == model_name)])) for s in scanners]
                offset = (i - (len(models_here) - 1) / 2) * width
                ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])
                if overall_df is not None:
                    orow = overall_df[(overall_df["feature"] == feat) & (overall_df["model"] == model_name)
                                       & (overall_df["cohort"] == cohort)]
                    if not orow.empty:
                        ax.axhline(float(orow["r2"].iloc[0]), color=colors[i % len(colors)], linestyle="--",
                                   linewidth=2.5, alpha=0.9, zorder=3, label=f"{model_name} (스캐너 통합)")
            ax.set_xticks(x)
            wrapped = ["\n".join(textwrap.wrap(s, label_wrap_width)) for s in scanners]
            ax.set_xticklabels(wrapped, fontsize=24, rotation=0)
            ax.set_title(f"{features_dict[feat]} — {cohort} 스캐너별 R²", fontsize=30,
                         fontweight="bold", color="#161616", pad=40)
            ax.set_ylim(0, 0.8)
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=30)

    # 패널마다 반복되던 범례를 없애고, 그림 전체 하단에 공통 범례 하나로 통합(모든 패널이 같은
    # feature_dict 내에서는 동일한 모델 조합을 쓰므로 라벨이 패널마다 달라질 위험이 없음)
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
    bottom_margin_in = 1.6
    bottom_frac = bottom_margin_in / fig_height_in
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=30,
               bbox_to_anchor=(0.5, bottom_frac * 0.15), frameon=True)

    fig.tight_layout(rect=(0, bottom_frac, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup R2 plot to {out_path}")


# features_dict에 속한 각 예측 대상에 대해 model_order의 모든 모델을 internal(OOF)/external(frozen)로 평가.
# feature별로 internal R^2가 가장 높은 AEC 모델을 골라 clinic4와 함께 스캐너별 서브그룹 R^2도 산출한다(항목4).
# aec_int_raw/shape_int_stack/best_n/include_sex는 FPCA_MODEL_KINDS에 속한 모델의 internal OOF를 fold별
# PCA refit(_fpca_oof_predict)으로 계산하기 위해 필요 - 그 외 모델은 기존과 동일하게 cross_val_predict 사용
def evaluate_features(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, models: dict, model_order: list[str],
                       features_dict: dict[str, str], cv: KFold, aec_int_raw: np.ndarray,
                       shape_int_stack: np.ndarray, best_n: int, include_sex: bool
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    scanner_rows = []
    for feat in features_dict:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]

        model_stats = {}
        scores_by_model = {}
        for model_name, spec in models.items():
            x_int, x_ext = spec["x_int_all"][mask_int], spec["x_ext_all"][mask_ext]

            fpca_kind = FPCA_MODEL_KINDS.get(model_name)
            if fpca_kind is None:
                oof = cross_val_predict(LinearRegression(), x_int, y_int, cv=cv)
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof = _fpca_oof_predict(meta_int, aec_int_raw, shape_extra, y_int_all, mask_int, best_n,
                                         include_sex, cv)
            model = LinearRegression().fit(x_int, y_int)
            pred_ext = cast(np.ndarray, model.predict(x_ext))

            stats_int = regression_significance_stats(y_int, oof)
            stats_ext = regression_significance_stats(y_ext, pred_ext)
            _log(feat, model_name, "internal OOF", stats_int)
            _log(feat, model_name, "external frozen internal model", stats_ext)
            summary_rows += [{"feature": feat, "model": model_name, "cohort": "internal", **stats_int},
                              {"feature": feat, "model": model_name, "cohort": "external", **stats_ext}]
            model_stats[model_name] = {"internal": stats_int, "external": stats_ext}
            scores_by_model[model_name] = {"internal": oof, "external": pred_ext}

        best_aec_model = max(model_order[1:], key=lambda m: model_stats[m]["internal"]["r2"])
        for model_name in ("clinic4", best_aec_model):
            scanner_rows += scanner_subgroup_r2(meta_int, mask_int, y_int, scores_by_model[model_name]["internal"],
                                                 feat, model_name, "internal")
            scanner_rows += scanner_subgroup_r2(meta_ext, mask_ext, y_ext, scores_by_model[model_name]["external"],
                                                 feat, model_name, "external")

    return pd.DataFrame(summary_rows), pd.DataFrame(scanner_rows)


# 스캐너별 서브그룹 R^2 결과를 csv로 저장하고 그래프도 함께 저장(overall_summary는 스캐너 통합 기준선용)
def save_scanner_subgroup(scanner_df: pd.DataFrame, features_dict: dict[str, str], output_dir: Path,
                           name: str, overall_summary: pd.DataFrame | None = None) -> None:
    scanner_df.to_csv(output_dir / f"clinic_aec_{name}_scanner_subgroup.csv", index=False)
    print(f"Saved scanner subgroup summary to {output_dir / f'clinic_aec_{name}_scanner_subgroup.csv'}")
    plot_scanner_subgroup_r2(scanner_df, features_dict, output_dir / f"clinic_aec_{name}_scanner_subgroup_r2.png",
                              overall_df=overall_summary)


# step3_clinic_aec_logistic.py(AUC)와 이 스크립트(R²)가 각각 산출한 스캐너별 서브그룹 결과(9개 feature x
# 스캐너 x cohort)를, feature 9종 평균으로 묶어 스캐너 1행짜리 요약표
# (cohort/scanner/n/R²(clinic4)/R²(+AEC)/AUC(clinic4)/AUC(+AEC))로 정리한다. PPT 본문에 넣을 표용 — 부록의
# feature별 세부 차트와 별개로, "스캐너 전체에서 방향이 일관되는가"를 한 장으로 보여주기 위한 집계
def build_scanner_summary(step3_dir: Path, step4_dir: Path) -> pd.DataFrame:
    r2_abs = pd.read_csv(step4_dir / "clinic_aec_abs_scanner_subgroup.csv")
    r2_ratio = pd.read_csv(step4_dir / "clinic_aec_ratio_scanner_subgroup.csv")
    auc = pd.read_csv(step3_dir / "scanner_subgroup_auc.csv")

    r2_all = pd.concat([r2_abs, r2_ratio], ignore_index=True)
    r2_all["model_grp"] = r2_all["model"].where(r2_all["model"] == "clinic4", "aec")
    auc = auc.copy()
    auc["model_grp"] = auc["model"].where(auc["model"] == "clinic4", "aec")

    r2_agg = r2_all.groupby(["cohort", "scanner", "model_grp"]).agg(
        r2=("r2", "mean"), n=("n", "first")).reset_index()
    auc_agg = auc.groupby(["cohort", "scanner", "model_grp"]).agg(
        auc=("auc", "mean")).reset_index()

    r2_piv = r2_agg.pivot_table(index=["cohort", "scanner"], columns="model_grp", values="r2").reset_index()
    r2_piv = r2_piv.rename(columns={"clinic4": "r2_clinic4", "aec": "r2_aec"})
    n_df = r2_agg.groupby(["cohort", "scanner"])["n"].first().reset_index()
    auc_piv = auc_agg.pivot_table(index=["cohort", "scanner"], columns="model_grp", values="auc").reset_index()
    auc_piv = auc_piv.rename(columns={"clinic4": "auc_clinic4", "aec": "auc_aec"})

    out = r2_piv.merge(n_df, on=["cohort", "scanner"]).merge(auc_piv, on=["cohort", "scanner"])
    out = out[["cohort", "scanner", "n", "r2_clinic4", "r2_aec", "auc_clinic4", "auc_aec"]]
    out["r2_delta"] = out["r2_aec"] - out["r2_clinic4"]
    out["auc_delta"] = out["auc_aec"] - out["auc_clinic4"]
    # internal(강남)이 위, external(신촌)이 아래 오도록 고정(알파벳순이면 external이 먼저 나옴)
    cohort_order = pd.CategoricalDtype(categories=["internal", "external"], ordered=True)
    out["cohort"] = out["cohort"].astype(cohort_order)
    return out.sort_values(["cohort", "scanner"]).reset_index(drop=True)


# scope 하나(output_dir)에 대해 step3의 AUC와 방금 이 스크립트가 저장한 R²를 묶어 요약 csv로 저장.
# step3가 아직 안 돌았으면(파일 없음) 건너뜀
def save_scanner_summary(output_dir: Path, scope: str) -> None:
    step3_dir = STEP3_DIR / scope
    if not (step3_dir / "scanner_subgroup_auc.csv").exists():
        print(f"[{scope}] scanner_subgroup_auc.csv 없음 — step3_clinic_aec_logistic.py 먼저 실행 필요, 건너뜀")
        return
    summary = build_scanner_summary(step3_dir, output_dir)
    out_path = output_dir / "scanner_performance_summary.csv"
    summary.round(4).to_csv(out_path, index=False)
    print(f"Saved scanner performance summary to {out_path}")
    print(summary.round(3).to_string())


# clinic4(include_sex=False면 clinic3) baseline과 +AEC 형태 feature(SD/Skewness/상하위50%비율/FPCA/전체결합)를
# 표준화해 추가한 5개 모델을 internal(OOF)/external(frozen)로 비교하며, 스캐너별 서브그룹 R^2도 함께 산출한다
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    shape_int, shape_ext = shape_features(aec_int_raw), shape_features(aec_ext_raw)

    fpca_target_features = list(FEATURES.keys()) + list(ABS_FEATURES.keys())
    best_n = select_best_fpca_n(meta_int, aec_int_raw, include_sex, fpca_target_features, cv)
    fpca_int, fpca_ext, _ = fpca_scores(aec_int_raw, aec_ext_raw, best_n)

    model_order = ["clinic4", "clinic4_aec_sd", "clinic4_aec_skew", "clinic4_aec_uplow_ratio",
                    "clinic4_aec_fpca", "clinic4_aec_shape_all"]
    models = {
        "clinic4": {"aec_int": None, "aec_ext": None},
        "clinic4_aec_sd": {"aec_int": shape_int["sd"].reshape(-1, 1), "aec_ext": shape_ext["sd"].reshape(-1, 1)},
        "clinic4_aec_skew": {"aec_int": shape_int["skew"].reshape(-1, 1), "aec_ext": shape_ext["skew"].reshape(-1, 1)},
        "clinic4_aec_uplow_ratio": {"aec_int": shape_int["upper_lower_ratio"].reshape(-1, 1),
                                     "aec_ext": shape_ext["upper_lower_ratio"].reshape(-1, 1)},
        "clinic4_aec_fpca": {"aec_int": fpca_int, "aec_ext": fpca_ext},
        "clinic4_aec_shape_all": {
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"], fpca_int]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"], fpca_ext]),
        },
    }
    for model_name, spec in models.items():
        x_int_all, scaler, aec_scaler = clinical_matrix(meta_int, spec["aec_int"], include_sex=include_sex)
        x_ext_all, _, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler, aec_scaler, include_sex=include_sex)
        spec["x_int_all"], spec["x_ext_all"] = x_int_all, x_ext_all

    verify_aec_scaling_invariance(meta_int, models["clinic4_aec_shape_all"]["aec_int"],
                                   pd.to_numeric(meta_int[next(iter(FEATURES))], errors="coerce").to_numpy(dtype=float),
                                   include_sex)

    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])

    ratio_summary, ratio_scanner = evaluate_features(meta_int, meta_ext, models, model_order, FEATURES, cv,
                                                       aec_int_raw, shape_int_stack, best_n, include_sex)
    ratio_summary.to_csv(output_dir / "clinic_aec_ratio_summary.csv", index=False)
    save_scanner_subgroup(ratio_scanner, FEATURES, output_dir, "ratio", overall_summary=ratio_summary)

    # PPT 슬라이드당 3개 feature만 들어가므로, 6개 비율 feature를 3개씩 top3/bottom3로도 별도 저장
    ratio_items = list(FEATURES.items())
    top3, bottom3 = dict(ratio_items[:3]), dict(ratio_items[3:])
    plot_scanner_subgroup_r2(ratio_scanner, top3, output_dir / "clinic_aec_ratio_scanner_subgroup_r2_top3.png",
                              overall_df=ratio_summary)
    plot_scanner_subgroup_r2(ratio_scanner, bottom3, output_dir / "clinic_aec_ratio_scanner_subgroup_r2_bottom3.png",
                              overall_df=ratio_summary)

    abs_summary, abs_scanner = evaluate_features(meta_int, meta_ext, models, model_order, ABS_FEATURES, cv,
                                                  aec_int_raw, shape_int_stack, best_n, include_sex)
    abs_summary.to_csv(output_dir / "clinic_aec_abs_summary.csv", index=False)
    save_scanner_subgroup(abs_scanner, ABS_FEATURES, output_dir, "abs", overall_summary=abs_summary)


# internal/external 코호트를 로드/전처리 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]

    def valid_clinical_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[clinical_cols].apply(pd.to_numeric, errors="coerce")
        return vals.notna().all(axis=1).to_numpy()

    mask_clinical_int = valid_clinical_rows(meta_int)
    mask_clinical_ext = valid_clinical_rows(meta_ext)
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

    for scope in ("total", "male", "female"):
        save_scanner_summary(OUTPUT_DIR / scope, scope)


if __name__ == "__main__":
    main()
