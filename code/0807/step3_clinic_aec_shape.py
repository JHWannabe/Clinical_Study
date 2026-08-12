from __future__ import annotations

# 임상변수 4개(sex/age/height/weight) + AEC-128 곡선의 통계적 형태 feature(SD, Skewness, 슬라이스
# 위치 기준 상위/하위 50% 평균 비율, FPCA top-3 score)를 개별/결합 추가 input으로 써서 체성분
# feature(연속형)를 예측하는 선형회귀 성능을 clinic4 baseline과 비교한다.

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step3"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
N_FPCA_COMPONENTS = 3

# 예측 대상 연속형 체성분 feature -> 파일명에 쓸 영문 slug
FEATURES = {
    "IMATA_SUM": "imata",
    "NAMA_SUM": "nama",
    "LAMA_SUM": "lama",
    "TAMA_SUM": "tama",
    "SAT(피하지방)_SUM": "sat",
    "VAT(내장지방)_SUM": "vat",
    "Total Fat_SUM": "total_fat",
}


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")

    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


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
def fpca_scores(aec_int: np.ndarray, aec_ext: np.ndarray,
                 n_components: int = N_FPCA_COMPONENTS) -> tuple[np.ndarray, np.ndarray, PCA]:
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


# clinic4/SD/Skewness/Upper-Lower비율/전체결합 5개 모델의 internal·external R^2를 feature별 막대그래프로 비교
def plot_r2_comparison(summary: pd.DataFrame, model_order: list[str], out_path: Path) -> None:
    features = list(FEATURES.keys())
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)
    colors = {
        "clinic4": "#6b6a66",
        "clinic4_aec_sd": "#2a78d6",
        "clinic4_aec_skew": "#e2622e",
        "clinic4_aec_uplow_ratio": "#4caf50",
        "clinic4_aec_fpca": "#16a085",
        "clinic4_aec_shape_all": "#9b59b6",
    }
    labels = {
        "clinic4": "clinic4",
        "clinic4_aec_sd": "+AEC SD",
        "clinic4_aec_skew": "+AEC Skewness",
        "clinic4_aec_uplow_ratio": "+AEC 상/하위50% 비율",
        "clinic4_aec_fpca": f"+AEC FPCA(PC1-{N_FPCA_COMPONENTS})",
        "clinic4_aec_shape_all": "+AEC 형태 전체",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows = summary[summary["cohort"] == cohort].set_index(["feature", "model"])
        for i, model_name in enumerate(model_order):
            r2_vals = [cast(float, rows.loc[(f, model_name), "r2"]) for f in features]
            vals = r2_vals
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=labels[model_name], color=colors[model_name])
        ax.set_xticks(x)
        ax.set_xticklabels(slugs)
        ax.set_title(f"R² ({cohort})", fontsize=36, fontweight="bold", color="#161616")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=21)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R2 comparison plot to {out_path}")


# clinic4(include_sex=False면 clinic3) baseline과, +AEC 형태 feature(SD/Skewness/상하위50%비율/전체결합)를 추가한
# 4개 모델을 internal(OOF)/external(frozen)로 비교
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    shape_int, shape_ext = shape_features(aec_int_raw), shape_features(aec_ext_raw)
    fpca_int, fpca_ext, pca = fpca_scores(aec_int_raw, aec_ext_raw)
    print(f"[FPCA] explained variance ratio (PC1-{N_FPCA_COMPONENTS}): {pca.explained_variance_ratio_.round(4)}")

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
            "aec_int": np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]]),
            "aec_ext": np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"]]),
        },
    }
    for model_name, spec in models.items():
        x_int_all, scaler = clinical_matrix(meta_int, spec["aec_int"], include_sex=include_sex)
        x_ext_all, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler, include_sex=include_sex)
        spec["x_int_all"], spec["x_ext_all"] = x_int_all, x_ext_all

    summary_rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]

        model_stats = {}
        for model_name, spec in models.items():
            x_int, x_ext = spec["x_int_all"][mask_int], spec["x_ext_all"][mask_ext]

            oof = cross_val_predict(LinearRegression(), x_int, y_int, cv=cv)
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

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "clinic_aec_shape_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'clinic_aec_shape_summary.csv'}")

    pivot_int = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="r2")
    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="r2")
    print("\n=== internal OOF R^2 ===")
    print(pivot_int[model_order].round(4))
    print("\n=== external (frozen) R^2 ===")
    print(pivot_ext[model_order].round(4))

    plot_r2_comparison(summary, model_order, output_dir / "clinic_aec_shape_r2_comparison.png")


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
