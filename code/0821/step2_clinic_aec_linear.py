from __future__ import annotations

# 임상변수 4개(sex/age/height/weight) + AEC-128 곡선의 통계적 형태 feature(SD, Skewness, 슬라이스
# 위치 기준 상위/하위 50% 평균 비율, FPCA top-n score)를 개별/결합 추가 input으로 써서 체성분
# 비율(VAT/Total Fat, VAT/SAT, VAT/TAMA, Total Fat/TAMA)과 VAT/SAT/Total Fat 절대값 예측 선형회귀 성능을
# clinic4 baseline과 비교한다. FPCA의 n_components는 internal 코호트 AEC 곡선의 scree curve에서
# elbow(Kneedle 방식: 축을 0~1로 정규화한 뒤 첫점-끝점을 잇는 직선까지 수직거리가 최대인 지점)로 매 run()마다
# 결정한다(사용자 확인: "R제곱값 말고 누적분산비율로 확인해" 이후 "elbow로 교체해서 재확인" - 다운스트림
# 예측성능이 아니라 FPCA/PCA 표준 scree test 관행대로 분산 감소 패턴만으로 n을 정함).
# step3_clinic_aec_shape.py를 복사해 예측 대상만 개별 체성분 절대량 -> 체성분 비율로 바꾼 버전.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0821" / "step2_linear"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COMPONENT_CANDIDATES_MAX = 20  # elbow 탐색에 쓸 n_components 상한

# 예측 대상 체성분 비율(파생 컬럼) -> 파일명에 쓸 영문 slug
# SAT_TotalFat_ratio/SAT_TAMA_ratio는 연구미팅 slide2 항목3("SAT/Total_Fat등의 여러 비율")에 따라 추가
FEATURES = {
    "VAT_TotalFat_ratio": "VAT/TotalFat",
    "VAT_SAT_ratio": "VAT/SAT",
    "VAT_TAMA_ratio": "VAT/TAMA",
    "TotalFat_TAMA_ratio": "TotalFat/TAMA",
    "SAT_TotalFat_ratio": "SAT/TotalFat",
    "SAT_TAMA_ratio": "SAT/TAMA",
}

# VAT/SAT/Total Fat 절대값(SUM)만 따로 비교하기 위한 예측 대상 -> 표시 라벨
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


# raw AEC-128 행렬(n x 128)에서 환자별 SD, Skewness, 슬라이스 위치 기준 상위/하위 50% 평균 비율을 산출.
# 상위 50%=앞쪽 절반 슬라이스(aec_1~64), 하위 50%=뒤쪽 절반 슬라이스(aec_65~128) —
# step3_clinic_aec_back_half.py의 FRONT_COLS/BACK_COLS와 동일한 위치 기준 분할이며, 그 두 구간을 별도
# feature로 각각 넣어보는 back_half.py와 달리 여기서는 두 구간 평균의 비율(단일 값)을 형태 feature로 쓴다.
def shape_features(aec_matrix: np.ndarray) -> dict[str, np.ndarray]:
    sd = aec_matrix.std(axis=1, ddof=1)
    skewness = stats.skew(aec_matrix, axis=1)

    half = N_SLICES // 2
    upper_mean = aec_matrix[:, :half].mean(axis=1)
    lower_mean = aec_matrix[:, half:].mean(axis=1)
    upper_lower_ratio = upper_mean / lower_mean

    return {"sd": sd, "skew": skewness, "upper_lower_ratio": upper_lower_ratio}


# internal 코호트 raw AEC-128 곡선에 PCA를 fit해 top-n_components score를 산출(등간격 128포인트이므로
# 표준 PCA가 FPCA의 이산 근사가 됨), external에는 동일 변환을 frozen 적용
def fpca_scores(aec_int: np.ndarray, aec_ext: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=SEED).fit(aec_int)
    return pca.transform(aec_int), pca.transform(aec_ext), pca


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + (있으면) AEC 형태 feature 결합. scaler 생략 시 새로 학습(내부 코호트용).
# 성별을 고정한 남/여 개별 실행에서는 sex가 상수가 되어 계수가 무의미해지므로 include_sex=False로 제외
def clinical_matrix(meta: pd.DataFrame, aec_extra: np.ndarray | None = None, scaler: StandardScaler | None = None,
                     include_sex: bool = True):
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    x = clinic if aec_extra is None else np.column_stack([clinic, aec_extra])
    return x, scaler


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
# n_components로 선택한다. 다운스트림 예측성능(R^2)으로 n을 고르면 그 성능 자체가 선택 기준에 쓰인 데이터로
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

        x_train, scaler = clinical_matrix(meta_masked.iloc[train_idx], extra_train, include_sex=include_sex)
        x_test, _ = clinical_matrix(meta_masked.iloc[test_idx], extra_test, scaler=scaler, include_sex=include_sex)

        model = LinearRegression().fit(x_train, y[train_idx])
        oof[test_idx] = model.predict(x_test)
    return oof


# 모델명 -> FPCA 포함 방식. "fpca_only"는 FPCA score만, "shape_all"은 SD/Skew/상하위비율+FPCA 결합.
# 이 두 모델만 internal OOF 계산 시 _fpca_oof_predict(fold별 PCA refit)를 거쳐야 함
FPCA_MODEL_KINDS = {"clinic4_aec_fpca": "fpca_only", "clinic4_aec_shape_all": "shape_all"}


# 좌: n_components별 누적 explained variance ratio, 우: scree curve+chord+elbow 판단 근거를 나란히 표시
def plot_fpca_component_search(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    scree, _ = _scree_and_elbow_distance(cum_var)
    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    ax = axes[0]
    ax.plot(cum_var.index, cum_var.values, marker="o", markersize=14, linewidth=4, color="#161616",
            label="누적 explained variance ratio")
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"선택된 elbow n={best_n}")
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
    ax.axvline(best_n, color="#e2622e", linestyle="--", linewidth=3, label=f"선택된 elbow n={best_n}")
    ax.set_xticks(list(scree.index))
    ax.set_xlabel("FPCA n_components", fontsize=42)
    ax.set_ylabel("개별 explained variance ratio", fontsize=42)
    ax.set_title("Scree curve (elbow 판단 근거)", fontsize=40, fontweight="bold", color="#161616", pad=30)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=30)
    ax.legend(loc="upper right", fontsize=26, frameon=False)

    fig.suptitle("FPCA 컴포넌트 수 선택 (internal 코호트, elbow 방식)", fontsize=44, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA component search plot to {out_path}")


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


# clinic4/전체형태(FPCA제외)/FPCA/전체결합 4개 모델(SD/Skewness 단독은 그래프에서 제외)의 internal·external R^2를 feature별 막대그래프로 비교
def plot_r2_comparison(summary: pd.DataFrame, model_order: list[str], features_dict: dict[str, str],
                        fpca_n: int, out_path: Path) -> None:
    features = list(features_dict.keys())
    slugs = [features_dict[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)
    # baseline은 중립 회색, +AEC 계열 3종은 dataviz 스킬 팔레트의 categorical slot 1-3(blue/orange/aqua,
    # all-pairs 사전검증된 순서)을 고정 순서로 배정해 계열간 비교가 쉽도록 함
    colors = {
        "clinic4": "#898781",
        "clinic4_aec_shape_no_fpca": "#2a78d6",
        "clinic4_aec_fpca": "#eb6834",
        "clinic4_aec_shape_all": "#1baf7a",
    }
    labels = {
        "clinic4": "clinic4",
        "clinic4_aec_shape_no_fpca": "+AEC (SD+Skewness+상/하위50%비율, FPCA제외)",
        "clinic4_aec_fpca": f"+AEC FPCA(PC1-{fpca_n})",
        "clinic4_aec_shape_all": "+AEC (SD+Skewness+상/하위50%비율+FPCA)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(45, 16.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows = summary[summary["cohort"] == cohort].set_index(["feature", "model"])
        for i, model_name in enumerate(model_order):
            r2_vals = [cast(float, rows.loc[(f, model_name), "r2"]) for f in features]
            vals = r2_vals
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=labels[model_name], color=colors[model_name])
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=42)
        if cohort == "internal":
            ax.set_ylabel("R²", fontsize=48)
        ax.set_title(f"R² ({cohort})", fontsize=40, fontweight="bold", color="#161616", pad=30)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="both", labelsize=36)

    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(model_order),
               bbox_to_anchor=(0.5, -0.1), fontsize=42, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R2 comparison plot to {out_path}")


# features_dict에 속한 각 예측 대상에 대해 model_order의 모든 모델을 internal(OOF)/external(frozen)로 평가.
# aec_int_raw/shape_int_stack/best_n/include_sex는 FPCA_MODEL_KINDS에 속한 모델의 internal OOF를 fold별
# PCA refit(_fpca_oof_predict)으로 계산하기 위해 필요 - 그 외 모델은 기존과 동일하게 cross_val_predict 사용
def evaluate_features(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, models: dict, model_order: list[str],
                       features_dict: dict[str, str], cv: KFold, aec_int_raw: np.ndarray,
                       shape_int_stack: np.ndarray, best_n: int, include_sex: bool) -> pd.DataFrame:
    summary_rows = []
    for feat in features_dict:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        # 컬럼 전체 결측 등으로 유효 샘플이 없으면(회귀/부트스트랩 불가) 이 feature는 건너뜀
        if mask_int.sum() < 2 or mask_ext.sum() < 2:
            print(f"[{feat}] 유효 샘플 부족(internal={mask_int.sum()}, external={mask_ext.sum()}) - 평가 건너뜀")
            continue
        y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]

        model_stats = {}
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

        for model_name in model_order[1:]:
            d_int = model_stats[model_name]["internal"]["r2"] - model_stats["clinic4"]["internal"]["r2"]
            d_ext = model_stats[model_name]["external"]["r2"] - model_stats["clinic4"]["external"]["r2"]
            print(f"[{feat}] R^2 delta vs clinic4 ({model_name}): internal={d_int:+.4f} external={d_ext:+.4f}")

    return pd.DataFrame(summary_rows)


# 평가 결과 summary를 csv로 저장하고 internal/external R^2 피벗을 출력한 뒤 비교 그래프를 저장
def save_summary_and_plot(summary: pd.DataFrame, model_order: list[str], features_dict: dict[str, str],
                           fpca_n: int, output_dir: Path, name: str) -> None:
    summary.to_csv(output_dir / f"clinic_aec_{name}_summary.csv", index=False)
    print(f"Saved summary to {output_dir / f'clinic_aec_{name}_summary.csv'}")

    pivot_int = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="r2")
    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="r2")
    print(f"\n=== internal OOF R^2 ({name}) ===")
    print(pivot_int[model_order].round(4))
    print(f"\n=== external (frozen) R^2 ({name}) ===")
    print(pivot_ext[model_order].round(4))

    # evaluate_features에서 유효 샘플 부족으로 건너뛴 feature는 그래프 대상에서도 제외(KeyError 방지)
    available_features = {k: v for k, v in features_dict.items() if k in summary["feature"].unique()}
    if len(available_features) < len(features_dict):
        skipped = sorted(set(features_dict) - set(available_features))
        print(f"[{name}] 유효 샘플 부족으로 그래프에서 제외된 feature: {skipped}")

    plot_model_order = [m for m in model_order if m not in ("clinic4_aec_sd", "clinic4_aec_skew")]
    plot_r2_comparison(summary, plot_model_order, available_features, fpca_n, output_dir / f"clinic_aec_{name}_r2_comparison.png")


# clinic4(include_sex=False면 clinic3) baseline과, +AEC 형태 feature(SD/Skewness/상하위50%비율/전체결합)를 추가한
# 4개 모델을 internal(OOF)/external(frozen)로 비교. 체성분 비율(FEATURES)과 VAT/SAT/Total Fat 절대값(ABS_FEATURES)을
# 각각 별도의 summary/그래프로 저장한다.
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    shape_int, shape_ext = shape_features(aec_int_raw), shape_features(aec_ext_raw)

    best_n, cum_var = select_best_fpca_n(aec_int_raw)
    save_cum_var_excel(cum_var, best_n, output_dir / "fpca_cumulative_variance.xlsx")
    plot_fpca_component_search(cum_var, best_n, output_dir / "fpca_component_search.png")

    fpca_int, fpca_ext, pca = fpca_scores(aec_int_raw, aec_ext_raw, best_n)
    print(f"[FPCA] explained variance ratio (PC1-{best_n}): {pca.explained_variance_ratio_.round(4)}")

    model_order = ["clinic4", "clinic4_aec_sd", "clinic4_aec_skew", "clinic4_aec_shape_no_fpca",
                    "clinic4_aec_fpca", "clinic4_aec_shape_all"]
    models = {
        "clinic4": {"aec_int": None, "aec_ext": None},
        "clinic4_aec_sd": {"aec_int": shape_int["sd"].reshape(-1, 1), "aec_ext": shape_ext["sd"].reshape(-1, 1)},
        "clinic4_aec_skew": {"aec_int": shape_int["skew"].reshape(-1, 1), "aec_ext": shape_ext["skew"].reshape(-1, 1)},
        "clinic4_aec_shape_no_fpca": {
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"]]),
        },
        "clinic4_aec_fpca": {"aec_int": fpca_int, "aec_ext": fpca_ext},
        "clinic4_aec_shape_all": {
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"], fpca_int]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"], fpca_ext]),
        },
    }
    for model_name, spec in models.items():
        x_int_all, scaler = clinical_matrix(meta_int, spec["aec_int"], include_sex=include_sex)
        x_ext_all, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler, include_sex=include_sex)
        spec["x_int_all"], spec["x_ext_all"] = x_int_all, x_ext_all

    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])

    ratio_summary = evaluate_features(meta_int, meta_ext, models, model_order, FEATURES, cv, aec_int_raw,
                                       shape_int_stack, best_n, include_sex)
    save_summary_and_plot(ratio_summary, model_order, FEATURES, best_n, output_dir, "ratio")

    abs_summary = evaluate_features(meta_int, meta_ext, models, model_order, ABS_FEATURES, cv, aec_int_raw,
                                     shape_int_stack, best_n, include_sex)
    save_summary_and_plot(abs_summary, model_order, ABS_FEATURES, best_n, output_dir, "abs")


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


if __name__ == "__main__":
    main()
