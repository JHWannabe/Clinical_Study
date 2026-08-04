from __future__ import annotations

# 임상변수 4개(sex/age/height/weight) + AEC-128의 앞/뒤 50%(64슬라이스) 평균 각각을 추가 input으로 써서
# 체성분 feature(연속형)를 예측하는 선형회귀 성능을 clinic4 baseline과 비교한다.
# n_seg=2 상관분석에서 뒷구간이 대부분 feature에서 더 강한 단순상관을 보였는데, 실제 모델 R^2로도
# 그 우위가 재현되는지(혹은 조인트모델 다중공선성 아티팩트였는지) 확인하기 위한 스크립트.

from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step3" / "aec_back_half"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FRONT_COLS = AEC_COLS[: N_SLICES // 2]   # 앞 50% (슬라이스 1~64)
BACK_COLS = AEC_COLS[N_SLICES // 2:]     # 뒤 50% (슬라이스 65~128)

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


# sex/age/height/weight 행렬 구성 + 표준화(sex 제외) + (있으면) AEC 반구간 평균 결합. scaler 생략 시 새로 학습(내부 코호트용)
def clinical_matrix(meta: pd.DataFrame, aec_mean: np.ndarray | None = None, scaler: StandardScaler | None = None):
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    clinic4 = np.column_stack([sex_m, scaler.transform(rest)])
    x = clinic4 if aec_mean is None else np.column_stack([clinic4, aec_mean])
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


# clinic4/front50/back50 3개 모델의 internal·external R^2를 feature별 막대그래프로 비교
def plot_r2_comparison(summary: pd.DataFrame, model_order: list[str], out_path: Path) -> None:
    features = list(FEATURES.keys())
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.25
    colors = {"clinic4": "#6b6a66", "clinic4_aec_front50": "#2a78d6", "clinic4_aec_back50": "#e2622e"}
    labels = {"clinic4": "clinic4", "clinic4_aec_front50": "+AEC 앞50%", "clinic4_aec_back50": "+AEC 뒤50%"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows = summary[summary["cohort"] == cohort].set_index(["feature", "model"])
        for i, model_name in enumerate(model_order):
            vals = [rows.loc[(f, model_name), "r2"] for f in features]
            err_lo = [rows.loc[(f, model_name), "r2"] - rows.loc[(f, model_name), "r2_ci_lower"] for f in features]
            err_hi = [rows.loc[(f, model_name), "r2_ci_upper"] - rows.loc[(f, model_name), "r2"] for f in features]
            ax.bar(x + (i - 1) * width, vals, width, yerr=[err_lo, err_hi], capsize=2,
                   label=labels[model_name], color=colors[model_name])
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=30, ha="right")
        ax.set_title(f"R² ({cohort})", fontsize=12, fontweight="bold", color="#161616")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R2 comparison plot to {out_path}")


# clinic4 baseline과, clinic4+AEC 앞50%/뒤50% 평균을 추가한 두 모델을 internal(OOF)/external(frozen)로 비교
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    front_mean_int = meta_int[FRONT_COLS].astype(float).to_numpy().mean(axis=1, keepdims=True)
    front_mean_ext = meta_ext[FRONT_COLS].astype(float).to_numpy().mean(axis=1, keepdims=True)
    back_mean_int = meta_int[BACK_COLS].astype(float).to_numpy().mean(axis=1, keepdims=True)
    back_mean_ext = meta_ext[BACK_COLS].astype(float).to_numpy().mean(axis=1, keepdims=True)

    model_order = ["clinic4", "clinic4_aec_front50", "clinic4_aec_back50"]
    models = {
        "clinic4": {"aec_int": None, "aec_ext": None},
        "clinic4_aec_front50": {"aec_int": front_mean_int, "aec_ext": front_mean_ext},
        "clinic4_aec_back50": {"aec_int": back_mean_int, "aec_ext": back_mean_ext},
    }
    for model_name, spec in models.items():
        x_int_all, scaler = clinical_matrix(meta_int, spec["aec_int"])
        x_ext_all, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler)
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
    summary.to_csv(OUTPUT_DIR / "clinic_aec_back_half_summary.csv", index=False)
    print(f"Saved summary to {OUTPUT_DIR / 'clinic_aec_back_half_summary.csv'}")

    pivot_int = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="r2")
    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="r2")
    print("\n=== internal OOF R^2 ===")
    print(pivot_int[model_order].round(4))
    print("\n=== external (frozen) R^2 ===")
    print(pivot_ext[model_order].round(4))

    plot_r2_comparison(summary, model_order, OUTPUT_DIR / "clinic_aec_back_half_r2_comparison.png")


if __name__ == "__main__":
    main()
