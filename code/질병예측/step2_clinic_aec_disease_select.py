from __future__ import annotations

# code/0814/step2_clinic_aec_linear.py를 베이스로, 예측 대상을 체성분 비율/절대값에서 HTN/DM/CKD
# 진단 이진값(metadata의 HTN/DM/CKD 컬럼, 이미 0/1)으로 교체한 버전. HTN/DM/CKD는 cutoff으로 새로
# 이분화할 연속값이 아니라 이미 진단된 이진 라벨이므로, linear regression/R^2 기반 비교는 제외하고
# (사용자 확인: "linear는 제외해") clinic4 vs clinic5(+VAT/SAT비)+AEC 형태 feature(SD/Skewness/상하위50%
# 비율/FPCA/전체결합) 4개 모델을 전부 logistic regression + internal OOF AUC 기준으로 비교한다(사용자 확인
# 2026-08-18: "input feature로 vat/sat 값도 추가해줘" 이후 "VAT/SAT 비율로 넣어", "구 clinic4 vs
# clinic5+aec feature로 비교해" - VAT/SAT 비율은 baseline엔 넣지 않고 AEC 추가 모델에만 함께 얹는다).
# FPCA n_components는 internal 코호트 AEC 곡선의 scree curve에서 elbow(Kneedle 방식: 축을 0~1로 정규화한
# 뒤 첫점-끝점을 잇는 직선까지 수직거리가 최대인 지점)를 참고로 계산하되, 실제 채택값은 사용자 확인
# 2026-08-14: "컴포넌트 수를 6으로 해줘"에 따라 6으로 고정한다(FPCA_N_FIXED, select_best_fpca_n 참고).
# 이 스크립트는 step3_clinic_aec_disease_logistic.py의 조합 선택(select_best_shape_model_logistic)과
# 별개로 동작하는 진단/비교용 스크립트다 - step3도 기존 step3_clinic_aec_logistic.py와 동일하게 조합
# 선택을 자체적으로 재계산하므로(스크립트간 결합 없음), 이 스크립트의 산출물은 참고용 비교 자료다.

import textwrap
from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step2_disease_select"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
# baseline(clinic4)은 그대로 두고, AEC를 더하는 모델에만 VAT/SAT 비율 1개를 추가로 얹어 clinic5+AEC로
# 비교한다(step3_clinic_aec_disease_logistic.py와 동일하게 맞춤, clinical_matrix의 include_vat_sat_ratio 참고)
VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
VAT_SAT_RATIO_COL = "VAT_SAT_ratio"
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]  # clinic4/clinic3 baseline(변경 없음)
ALL_CLINICAL_COLS = CLINICAL_BASE_COLS + [VAT_SAT_RATIO_COL]  # clinic5까지 포함한 결측 체크용
FPCA_COMPONENT_CANDIDATES = list(range(1, 21))
FPCA_N_FIXED = 3  # 사용자 확인 2026-08-14: n=3/6/12 비교 검토 후 elbow 참고값과 동일한 3으로 재확정
MIN_POSITIVES = 2  # 이 미만이면 AUC 자체가 정의되지 않아 해당 scope/target을 skip

# 예측 대상(진단 이진값) -> 파일명/그래프에 쓸 slug
FEATURES = {"HTN": "HTN", "DM": "DM", "CKD": "CKD"}


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
# (step2_clinic_aec_linear.py의 shape_features()와 동일)
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


# age/height/weight(+include_vat_sat_ratio시 VAT/SAT 비율) 행렬 구성 + 표준화 + (include_sex시) sex 열 +
# (있으면) AEC 형태 feature 결합. AEC 파생 feature(SD/Skew/상하위비율/FPCA score)는 원 단위가 서로 달라
# clinic feature와 함께 L2 정규화 로지스틱에 넣으면 스케일이 큰 feature가 결과를 좌우한다 -
# step3_clinic_aec_disease_logistic.py의 clinical_matrix()와 동일하게 aec_extra 전용 StandardScaler로
# 별도 표준화한다(step2/step3 결과 불일치 수정, 사용자 확인 2026-08-14: CKD internal에서 shape_all/FPCA
# 순위가 스케일링 유무로 뒤집히는 것을 발견). clinic4 baseline은 include_vat_sat_ratio=False로 기존 3개
# 임상변수만 쓰고, AEC를 더하는 모델만 True로 VAT/SAT 비율을 추가한다(clinic5+AEC, 사용자 확인 2026-08-18)
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     aec_scaler: StandardScaler | None = None, include_sex: bool = True,
                     include_vat_sat_ratio: bool = False):
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


# scree curve(컴포넌트별 개별 explained variance ratio)를 구하고, 축을 0~1로 정규화한 뒤 첫점-끝점을
# 잇는 직선(chord)까지의 수직거리를 계산한다. 거리가 최대인 지점이 elbow(Satopaa et al. 2011 Kneedle
# 알고리즘). 축 정규화 없이 원래 스케일(n_components 1~20 vs 분산비율 0~1)로 거리를 재면 축 단위 차이
# 때문에 결과가 왜곡되므로 정규화가 필수다
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
# 참고용으로 계산한다. 다운스트림 예측성능(AUC/R^2)으로 n을 고르면 그 성능 자체가 선택 기준에 쓰인
# 데이터로 최적화되어 낙관적으로 부풀려질 위험이 있어(사용자 확인: "R제곱값 말고 누적분산비율로 확인해"
# 이후 "elbow로 교체해서 재확인"), FPCA/PCA 표준 scree test 관행대로 분산 감소 패턴만으로 n을 정했으나,
# 최종 n_components는 사용자 확인 2026-08-14: "컴포넌트 수를 6으로 해줘"에 따라 FPCA_N_FIXED(6)로 고정한다
# (elbow 값은 fpca_component_search.png/fpca_cumulative_variance.xlsx에 비교 참고용으로 계속 남긴다)
def select_best_fpca_n(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES[-1], aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))
    _, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    best_n = FPCA_N_FIXED

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] elbow 참고값 n_components = {elbow_n} (누적분산비율={cum_var[elbow_n]:.4f}, "
          f"chord-거리={dist[elbow_n]:.4f}) / 실제 채택 n_components = {best_n}(고정)")
    return best_n, cum_var


# n_components별 누적/개별 explained variance ratio, elbow 판단에 쓴 chord-거리, elbow 참고값과 실제
# 채택된 best_n(고정값)을 엑셀로 저장(그래프의 원자료). elbow_n과 selected_best_n이 다를 수 있으므로
# (2026-08-14부터 채택값은 FPCA_N_FIXED=6, elbow는 참고용) 두 열을 분리해서 남긴다
def save_cum_var_excel(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values,
                        "individual_variance_ratio": scree.values, "elbow_chord_distance": dist.values})
    df["is_elbow_reference"] = df["n_components"] == elbow_n
    df["selected_best_n"] = df["n_components"] == best_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# FPCA score가 들어간 모델(clinic5_aec_fpca/clinic5_aec_shape_all)의 internal OOF 확률을 산출.
# PCA(고유함수 추정)를 전체 데이터로 먼저 fit하면 검증 fold의 곡선 정보가 고유함수 추정에 이미 들어가
# OOF AUC가 낙관적으로 부풀려지는 data leakage가 되므로, PCA를 fold의 train 구간에서만 fit한다.
# 이 함수는 항상 AEC를 더하는 모델에만 쓰이므로 clinical_matrix에 VAT/SAT 비율을 포함시킨다
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
                                                       include_sex=include_sex, include_vat_sat_ratio=True)
        x_test, _, _ = clinical_matrix(meta_masked.iloc[test_idx], extra_test, scaler=scaler,
                                        aec_scaler=aec_scaler, include_sex=include_sex, include_vat_sat_ratio=True)

        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


# 모델명 -> FPCA 포함 방식. "fpca_only"는 FPCA score만, "shape_all"은 SD/Skew/상하위비율+FPCA 결합
FPCA_MODEL_KINDS = {"clinic5_aec_fpca": "fpca_only", "clinic5_aec_shape_all": "shape_all"}


# 좌: n_components별 누적 explained variance ratio, 우: scree curve+chord+elbow 판단 근거를 나란히 표시
def plot_fpca_component_search(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, dist = _scree_and_elbow_distance(cum_var)
    elbow_n = int(dist.idxmax())
    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    ax = axes[0]
    ax.plot(cum_var.index, cum_var.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="누적 explained variance ratio")
    if elbow_n != best_n:
        ax.axvline(elbow_n, color="#898781", linestyle=":", linewidth=3, label=f"elbow 참고값 n={elbow_n}")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"채택된 n={best_n}")
    ax.set_xticks(list(cum_var.index))
    ax.set_xlabel("FPCA n_components", fontsize=42)
    ax.set_ylabel("누적 explained variance ratio", fontsize=42)
    ax.set_title("누적분산비율", fontsize=40, fontweight="bold", color="#161616", pad=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(loc="lower right", fontsize=26, frameon=False)

    ax = axes[1]
    ax.plot(scree.index, scree.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="개별 explained variance ratio (scree)")
    ax.plot([scree.index[0], scree.index[-1]], [scree.values[0], scree.values[-1]],
            color="#898781", linestyle=":", linewidth=3, label="첫점-끝점 직선(chord)")
    if elbow_n != best_n:
        ax.axvline(elbow_n, color="#898781", linestyle=":", linewidth=3, label=f"elbow 참고값 n={elbow_n}")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"채택된 n={best_n}")
    ax.set_xticks(list(scree.index))
    ax.set_xlabel("FPCA n_components", fontsize=42)
    ax.set_ylabel("개별 explained variance ratio", fontsize=42)
    ax.set_title("Scree curve (elbow 판단 근거)", fontsize=40, fontweight="bold", color="#161616", pad=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(loc="upper right", fontsize=26, frameon=False)

    fig.suptitle("FPCA 컴포넌트 수 선택 (internal 코호트, elbow 참고 + 고정 n 채택)", fontsize=44, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA component search plot to {out_path}")


# AUC와 n/n_pos/prevalence를 산출 (bootstrap CI는 사용자 확인으로 제거)
def classification_significance_stats(y: np.ndarray, proba: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, proba))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "auc": auc}


# feat/model/cohort별 핵심 통계를 한 줄로 출력
def _log(feat: str, model_name: str, cohort: str, s: dict) -> None:
    print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.4f}")


# clinic4/전체형태(FPCA제외)/FPCA/전체결합 4개 모델의 internal·external AUC를 feature(질환)별 막대그래프로 비교
# (뒤 3개 모델은 clinic5, 즉 VAT/SAT 비율이 추가로 포함된 상태에서 AEC 조합만 다름)
def plot_auc_comparison(summary: pd.DataFrame, model_order: list[str], features_dict: dict[str, str],
                         fpca_n: int, out_path: Path) -> None:
    features = list(features_dict.keys())
    slugs = [features_dict[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)
    colors = {
        "clinic4": "#898781",
        "clinic5_aec_shape_no_fpca": "#2a78d6",
        "clinic5_aec_fpca": "#eb6834",
        "clinic5_aec_shape_all": "#1baf7a",
    }
    labels = {
        "clinic4": "clinic4",
        "clinic5_aec_shape_no_fpca": "+VAT/SAT비+AEC (SD+Skewness+상/하위50%비율, FPCA제외)",
        "clinic5_aec_fpca": f"+VAT/SAT비+AEC FPCA(PC1-{fpca_n})",
        "clinic5_aec_shape_all": "+VAT/SAT비+AEC (SD+Skewness+상/하위50%비율+FPCA)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(30, 16.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows = summary[summary["cohort"] == cohort].set_index(["feature", "model"])
        for i, model_name in enumerate(model_order):
            auc_vals = [cast(float, rows.loc[(f, model_name), "auc"]) if (f, model_name) in rows.index else np.nan
                        for f in features]
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, auc_vals, width, label=labels[model_name], color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=42)
        if cohort == "internal":
            ax.set_ylabel("AUC", fontsize=48)
        ax.set_title(f"AUC ({cohort})", fontsize=40, fontweight="bold", color="#161616", pad=30)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="both", labelsize=36)
        ax.set_ylim(0.5, 1.0)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(model_order),
               bbox_to_anchor=(0.5, -0.1), fontsize=36, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC comparison plot to {out_path}")


# plot_auc_comparison()과 동일한 4개 모델 x internal/external AUC를 표 이미지로도 저장(그래프의 수치 확인용).
# feature별 최고 AUC 모델은 굵게 표시하고, 마지막 열에 best AUC - clinic4 AUC delta를 추가한다(개선=빨강/악화=파랑).
# 모델 라벨이 길어 컬럼이 잘리지 않도록 헤더를 textwrap으로 줄바꿈하고, Feature/delta 열보다 모델 열을 넓게 고정폭 지정
def plot_auc_comparison_table(summary: pd.DataFrame, model_order: list[str], features_dict: dict[str, str],
                               fpca_n: int, out_path: Path) -> None:
    features = list(features_dict.keys())
    slugs = {f: features_dict[f] for f in features}
    labels = {
        "clinic4": "clinic4",
        "clinic5_aec_shape_no_fpca": "+VAT/SAT비+AEC (SD+Skew+상하위비율)",
        "clinic5_aec_fpca": f"+VAT/SAT비+AEC FPCA(PC1-{fpca_n})",
        "clinic5_aec_shape_all": "+VAT/SAT비+AEC (SD+Skew+상하위비율+FPCA)",
    }
    header_wrap_width = 16
    col_labels = (["Feature"] + ["\n".join(textwrap.wrap(labels[m], header_wrap_width)) for m in model_order]
                  + ["ΔAUC\n(best-clinic4)"])
    n_model_cols = len(model_order)
    col_widths = [0.13] + [0.75 / n_model_cols] * n_model_cols + [0.12]
    delta_col = n_model_cols + 1

    fig, axes = plt.subplots(2, 1, figsize=(3.5 * n_model_cols, 2 * (1.8 + 0.9 * len(features))))
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows_df = summary[summary["cohort"] == cohort].set_index(["feature", "model"])
        rows = []
        best_col_by_row = []
        delta_by_row = []
        for f in features:
            aucs = [cast(float, rows_df.loc[(f, m), "auc"]) if (f, m) in rows_df.index else float("nan")
                    for m in model_order]
            clinic4_auc = aucs[model_order.index("clinic4")]
            best_auc = float(np.nanmax(aucs)) if np.any(np.isfinite(aucs)) else float("nan")
            delta = best_auc - clinic4_auc if np.isfinite(best_auc) and np.isfinite(clinic4_auc) else float("nan")
            row = ([slugs[f]] + [f"{v:.4f}" if np.isfinite(v) else "-" for v in aucs]
                   + [f"{delta:+.4f}" if np.isfinite(delta) else "-"])
            rows.append(row)
            best_col_by_row.append(int(np.nanargmax(aucs)) + 1 if np.any(np.isfinite(aucs)) else -1)
            delta_by_row.append(delta)

        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(15)
        tbl.scale(1, 3.6)
        for (row_i, col_i), cell in tbl.get_celld().items():
            if row_i == 0:
                cell.set_text_props(weight="bold", color="white", fontsize=15)
                cell.set_facecolor("#161616")
            else:
                cell.set_facecolor("#f2f1ee" if row_i % 2 == 0 else "white")
                if col_i == best_col_by_row[row_i - 1]:
                    cell.set_text_props(weight="bold")
                elif col_i == delta_col:
                    delta = delta_by_row[row_i - 1]
                    if np.isfinite(delta):
                        cell.set_text_props(color="#d30909" if delta > 0 else "#0055bd", weight="bold")
        ax.set_title(f"AUC ({cohort})", fontsize=22, fontweight="bold", color="#161616", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC comparison table image to {out_path}")


# features_dict(HTN/DM/CKD)에 속한 각 진단 대상에 대해 model_order의 모든 모델을 internal(OOF)/
# external(frozen)로 평가. 양성 표본이 MIN_POSITIVES 미만인 경우는 skip
def evaluate_features(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, models: dict, model_order: list[str],
                       features_dict: dict[str, str], aec_int_raw: np.ndarray, shape_int_stack: np.ndarray,
                       best_n: int, include_sex: bool) -> pd.DataFrame:
    summary_rows = []
    for feat in features_dict:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int].astype(int), y_ext_all[mask_ext].astype(int)

        if min(y_int.sum(), len(y_int) - y_int.sum(), y_ext.sum(), len(y_ext) - y_ext.sum()) < MIN_POSITIVES:
            print(f"[{feat}] SKIP: 양성/음성 표본 부족 (internal pos={y_int.sum()}, external pos={y_ext.sum()})")
            continue

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        model_stats = {}
        for model_name, spec in models.items():
            x_int, x_ext = spec["x_int_all"][mask_int], spec["x_ext_all"][mask_ext]

            fpca_kind = FPCA_MODEL_KINDS.get(model_name)
            if fpca_kind is None:
                oof = cross_val_predict(LogisticRegression(max_iter=2000), x_int, y_int, cv=cv,
                                         method="predict_proba")[:, 1]
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof = _fpca_oof_proba_predict(meta_int, aec_int_raw, shape_extra, y_int_all,
                                               mask_int, best_n, include_sex, cv)
            model = LogisticRegression(max_iter=2000).fit(x_int, y_int)
            pred_ext = cast(np.ndarray, model.predict_proba(x_ext)[:, 1])

            stats_int = classification_significance_stats(y_int, oof)
            stats_ext = classification_significance_stats(y_ext, pred_ext)
            _log(feat, model_name, "internal OOF", stats_int)
            _log(feat, model_name, "external frozen internal model", stats_ext)
            summary_rows += [{"feature": feat, "model": model_name, "cohort": "internal", **stats_int},
                              {"feature": feat, "model": model_name, "cohort": "external", **stats_ext}]
            model_stats[model_name] = {"internal": stats_int, "external": stats_ext}

        for model_name in model_order[1:]:
            d_int = model_stats[model_name]["internal"]["auc"] - model_stats["clinic4"]["internal"]["auc"]
            d_ext = model_stats[model_name]["external"]["auc"] - model_stats["clinic4"]["external"]["auc"]
            print(f"[{feat}] AUC delta vs clinic4 ({model_name}): internal={d_int:+.4f} external={d_ext:+.4f}")

    return pd.DataFrame(summary_rows)


# 평가 결과 summary를 csv로 저장하고 internal/external AUC 피벗을 출력한 뒤 비교 그래프를 저장
def save_summary_and_plot(summary: pd.DataFrame, model_order: list[str], features_dict: dict[str, str],
                           fpca_n: int, output_dir: Path) -> None:
    summary.to_csv(output_dir / "clinic_aec_disease_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'clinic_aec_disease_summary.csv'}")

    if summary.empty:
        print("[경고] summary가 비어 있어 피벗/그래프를 생략합니다.")
        return

    pivot_int = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="auc")
    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="auc")
    print("\n=== internal OOF AUC ===")
    print(pivot_int.reindex(columns=model_order).round(4))
    print("\n=== external (frozen) AUC ===")
    print(pivot_ext.reindex(columns=model_order).round(4))

    plot_auc_comparison(summary, model_order, features_dict, fpca_n, output_dir / "clinic_aec_disease_auc_comparison.png")
    plot_auc_comparison_table(summary, model_order, features_dict, fpca_n,
                               output_dir / "clinic_aec_disease_auc_comparison_table.png")


# clinic4(sex/age/height/weight, include_sex=False면 clinic3) baseline과, 여기에 VAT/SAT 비율과 +AEC
# 형태 feature(SD/Skewness/상하위50%비율/FPCA/전체결합)를 함께 추가한 clinic5+AEC 3개 모델을 HTN/DM/CKD
# 각각에 대해 internal(OOF)/external(frozen) logistic AUC로 비교(사용자 확인 2026-08-18: "구 clinic4 vs
# clinic5+aec feature로 비교해")
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    shape_int, shape_ext = shape_features(aec_int_raw), shape_features(aec_ext_raw)

    best_n, cum_var = select_best_fpca_n(aec_int_raw)
    plot_fpca_component_search(cum_var, best_n, output_dir / "fpca_component_search.png")
    save_cum_var_excel(cum_var, best_n, output_dir / "fpca_cumulative_variance.xlsx")

    fpca_int, fpca_ext, pca = fpca_scores(aec_int_raw, aec_ext_raw, best_n)
    print(f"[FPCA] explained variance ratio (PC1-{best_n}): {pca.explained_variance_ratio_.round(4)}")

    model_order = ["clinic4", "clinic5_aec_shape_no_fpca", "clinic5_aec_fpca", "clinic5_aec_shape_all"]
    models = {
        "clinic4": {"aec_int": None, "aec_ext": None, "include_vat_sat_ratio": False},
        "clinic5_aec_shape_no_fpca": {
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"]]),
            "include_vat_sat_ratio": True,
        },
        "clinic5_aec_fpca": {"aec_int": fpca_int, "aec_ext": fpca_ext, "include_vat_sat_ratio": True},
        "clinic5_aec_shape_all": {
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"], fpca_int]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"], fpca_ext]),
            "include_vat_sat_ratio": True,
        },
    }
    for model_name, spec in models.items():
        x_int_all, scaler, aec_scaler = clinical_matrix(meta_int, spec["aec_int"], include_sex=include_sex,
                                                          include_vat_sat_ratio=spec["include_vat_sat_ratio"])
        x_ext_all, _, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler, aec_scaler, include_sex=include_sex,
                                           include_vat_sat_ratio=spec["include_vat_sat_ratio"])
        spec["x_int_all"], spec["x_ext_all"] = x_int_all, x_ext_all

    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])

    summary = evaluate_features(meta_int, meta_ext, models, model_order, FEATURES, aec_int_raw,
                                 shape_int_stack, best_n, include_sex)
    save_summary_and_plot(summary, model_order, FEATURES, best_n, output_dir)


# internal/external 코호트를 로드/전처리 후 전체 코호트(sex 포함)로 run()을 실행(성별 층화분석 제거,
# 사용자 확인: "female, male 성별 층화분석은 제거하고 total만 진행"). output도 OUTPUT_DIR 바로 아래에 저장한다
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    def valid_clinical_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[ALL_CLINICAL_COLS].apply(pd.to_numeric, errors="coerce")
        return vals.notna().all(axis=1).to_numpy()

    mask_clinical_int = valid_clinical_rows(meta_int)
    mask_clinical_ext = valid_clinical_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    run(meta_int, meta_ext, OUTPUT_DIR, include_sex=True)


if __name__ == "__main__":
    main()
