from __future__ import annotations

# 임상변수만으로 체성분 feature(연속형)를 예측하는 baseline 선형회귀 파이프라인

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step0"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709

# 예측 대상 연속형 체성분 feature -> 파일명에 쓸 영문 slug
FEATURES = {
    "IMATA_SUM": "imata",
    "NAMA_SUM": "nama",
    "LAMA_SUM": "lama",
    "VAT(내장지방)_SUM": "vat",
    "SAT(피하지방)_SUM": "sat",
    "TAMA_SUM": "tama",
    "Total Fat_SUM": "total_fat",
}


# 엑셀 metadata 시트를 로드
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)


# sex/age/height/weight 행렬 구성 + 표준화(sex 제외). scaler 생략 시 새로 학습(내부 코호트용)
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None = None):
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


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


# feat/cohort별 핵심 통계를 한 줄로 출력
def _log(feat: str, cohort: str, s: dict) -> None:
    print(f"[{feat} / {cohort}] R2={s['r2']:.4f} 95%CI=[{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}] "
          f"RMSE={s['rmse']:.3f} MAE={s['mae']:.3f} Pearson p={s['p_value']:.3e}")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트("clinic4")만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# R^2를 핵심 수치로 크게 강조 표시하고 RMSE/Pearson r/p-value/n은 보조 텍스트로 배치
def _draw_scatter(ax: Axes, y: np.ndarray, pred: np.ndarray, s: dict, title: str, lims: tuple[float, float]) -> None:
    INK_PRIMARY, INK_MUTED, POINT_COLOR, FIT_COLOR = "#161616", "#6b6a66", "#2a78d6", "#e2622e"
    p_str = "p<1e-300" if s["p_value"] == 0 else f"p={s['p_value']:.2e}"

    ax.scatter(y, pred, s=14, alpha=0.45, color=POINT_COLOR, edgecolors="none")
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)

    fit_slope, fit_intercept = np.polyfit(y, pred, 1)
    fit_x = np.array(lims)
    ax.plot(fit_x, fit_slope * fit_x + fit_intercept, color=FIT_COLOR, linestyle="-", linewidth=1.5)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.text(0.05, 0.93, f"R² = {s['r2']:.3f}", ha="left", va="top",
             fontsize=22, fontweight="bold", color=INK_PRIMARY, transform=ax.transAxes)
    ax.text(0.05, 0.85, f"95% CI [{s['r2_ci_lower']:.3f}, {s['r2_ci_upper']:.3f}]\n"
             f"RMSE={s['rmse']:.2f}  MAE={s['mae']:.2f}\n"
             f"Pearson r={s['pearson_r']:.3f}, {p_str} (n={s['n']})",
             ha="left", va="top", fontsize=9, color=INK_MUTED, transform=ax.transAxes)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# 코호트별(internal/external) 실측-예측 산점도를 한 figure에 나란히 배치
def plot_scatter_dual(rows: list[tuple[np.ndarray, np.ndarray, dict, str]], out_path: Path) -> None:
    lims = (float(min(min(y.min(), pred.min()) for y, pred, _, _ in rows)),
            float(max(max(y.max(), pred.max()) for y, pred, _, _ in rows)))
    fig, axes = plt.subplots(1, len(rows), figsize=(6.5 * len(rows), 6.5))
    for ax, (y, pred, s, title) in zip(np.atleast_1d(axes), rows):
        _draw_scatter(ax, y, pred, s, title, lims)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {out_path}")


# 잔차 vs 예측값 / Normal Q-Q / 잔차 히스토그램 / Scale-Location 4종을 한 행에 배치
def _draw_residual_row(axes: np.ndarray, y: np.ndarray, pred: np.ndarray, row_title: str) -> None:
    INK_PRIMARY, POINT_COLOR, FIT_COLOR = "#161616", "#2a78d6", "#e2622e"
    residuals = y - pred
    std_resid = (residuals - residuals.mean()) / residuals.std(ddof=1)

    ax = axes[0]
    ax.scatter(pred, residuals, s=12, alpha=0.45, color=POINT_COLOR, edgecolors="none")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{row_title}\nResiduals vs Predicted", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)

    ax = axes[1]
    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    osm, osr = np.asarray(osm, dtype=float), np.asarray(osr, dtype=float)
    slope, intercept, r, _, _ = stats.linregress(osm, osr)
    ax.scatter(osm, osr, s=12, alpha=0.45, color=POINT_COLOR, edgecolors="none")
    ax.plot(osm, slope * osm + intercept, color=FIT_COLOR, linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Standardized Residual")
    ax.set_title(f"Normal Q-Q (r={r:.3f})", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(residuals, bins=30, color=POINT_COLOR, alpha=0.6, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    shapiro_n = min(len(residuals), 5000)
    _, shapiro_p = stats.shapiro(residuals[:shapiro_n])
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Histogram (Shapiro p={shapiro_p:.2e})", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.scatter(pred, np.sqrt(np.abs(std_resid)), s=12, alpha=0.45, color=POINT_COLOR, edgecolors="none")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("sqrt(|Standardized Residual|)")
    ax.set_title("Scale-Location", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)


# feature 하나에 대해 internal/external 잔차 진단 4종을 2행(코호트)x4열로 저장
def plot_residual_diagnostics(y_int: np.ndarray, pred_int: np.ndarray, y_ext: np.ndarray, pred_ext: np.ndarray,
                               feat: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    _draw_residual_row(axes[0], y_int, pred_int, f"{feat} (internal, OOF)")
    _draw_residual_row(axes[1], y_ext, pred_ext, f"{feat} (external, frozen internal model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual diagnostics plot to {out_path}")


# 전체 feature에 걸쳐 R^2(95%CI)/RMSE/MAE/Pearson r을 internal vs external 막대그래프로 비교
def plot_metrics_summary(summary: pd.DataFrame, out_path: Path) -> None:
    metric_labels = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "pearson_r": "Pearson r"}
    features = list(FEATURES.keys())
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, metric in zip(axes.ravel(), metric_labels):
        int_rows = summary[summary["cohort"] == "internal"].set_index("feature").loc[features]
        ext_rows = summary[summary["cohort"] == "external"].set_index("feature").loc[features]

        int_err = ext_err = None
        if metric == "r2":
            int_err = np.abs(np.vstack([int_rows["r2"] - int_rows["r2_ci_lower"],
                                         int_rows["r2_ci_upper"] - int_rows["r2"]]))
            ext_err = np.abs(np.vstack([ext_rows["r2"] - ext_rows["r2_ci_lower"],
                                         ext_rows["r2_ci_upper"] - ext_rows["r2"]]))

        ax.bar(x - width / 2, int_rows[metric], width, yerr=int_err, capsize=3,
               label="internal (OOF)", color="#2a78d6")
        ax.bar(x + width / 2, ext_rows[metric], width, yerr=ext_err, capsize=3,
               label="external (frozen)", color="#e2622e")
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, rotation=30, ha="right")
        ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold", color="#161616")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics summary plot to {out_path}")


CORRELATION_CSV = OUTPUT_DIR / "clinical_only_feature_correlations.csv"


# 통합 상관계수 CSV에서 이 스크립트가 소유한 predictor_group만 교체 저장, 다른 스크립트가 쓴 predictor_group(aec_segment 등)은 보존
def write_correlation_rows(rows: list[dict], predictor_group: str) -> None:
    new_df = pd.DataFrame(rows)
    new_df["predictor_group"] = predictor_group
    for col in ("n_seg", "segment"):
        if col not in new_df.columns:
            new_df[col] = np.nan
    if CORRELATION_CSV.exists():
        old_df = pd.read_csv(CORRELATION_CSV)
        old_df = old_df[old_df["predictor_group"] != predictor_group]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    cols = ["feature", "cohort", "predictor_group", "predictor", "n_seg", "segment", "r", "p_value", "n"]
    combined[cols].to_csv(CORRELATION_CSV, index=False)
    print(f"Saved correlation rows (predictor_group={predictor_group}) to {CORRELATION_CSV}")


# output feature 각각과 clinic4(sex/age/height/weight) 입력변수 각각의 단순 Pearson r/p를 산출
def feature_clinic4_correlations(x_all: np.ndarray, meta: pd.DataFrame, input_cols: list[str],
                                  cohort: str) -> list[dict]:
    rows = []
    for feat in FEATURES:
        y_all = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        y, x = y_all[mask], x_all[mask]
        for j, col in enumerate(input_cols):
            r, p = stats.pearsonr(x[:, j], y)
            rows.append({"feature": FEATURES[feat], "cohort": cohort, "predictor": col,
                         "r": float(r), "p_value": float(p), "n": int(mask.sum())})
    return rows


# feature x clinic4 상관계수를 internal/external 나란히 히트맵으로 시각화(칸에 r)
def plot_feature_clinic4_corr_heatmap(corr_df: pd.DataFrame, input_cols: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    predictor_labels = {"sex_M": "Sex(M)", "age_z": "Age", "height_z": "Height", "weight_z": "Weight"}
    features = list(FEATURES.values())
    cohorts = ["internal", "external"]

    fig, axes = plt.subplots(1, 2, figsize=(4.2 * len(input_cols) / 2 + 3, 0.6 * len(features) + 2))
    for ax, cohort in zip(axes, cohorts):
        mat = corr_df[corr_df["cohort"] == cohort].pivot(index="feature", columns="predictor", values="r")
        mat = mat.loc[features, input_cols]

        im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                r_val = mat.iat[i, j]
                ax.text(j, i, f"{r_val:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(r_val) > 0.5 else INK_PRIMARY)

        ax.set_xticks(range(len(input_cols)))
        ax.set_xticklabels([predictor_labels[c] for c in input_cols], rotation=30, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold", color=INK_PRIMARY)

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Pearson r")
    fig.suptitle("Output feature vs clinic4 Pearson r", fontsize=13, fontweight="bold", color=INK_PRIMARY)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-clinic4 correlation heatmap to {out_path}")


# internal로 학습/평가 후 external에 고정 모델 적용, feature별 R^2/산점도/계수를 산출
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]

    def valid_clinical_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[clinical_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_clinical_int = valid_clinical_rows(meta_int)
    mask_clinical_ext = valid_clinical_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    x_int_all, scaler = clinical_matrix(meta_int)
    x_ext_all, _ = clinical_matrix(meta_ext, scaler)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    input_cols = ["sex_M", "age_z", "height_z", "weight_z"]
    io_int_df = pd.DataFrame(x_int_all, columns=input_cols)
    io_int_df.insert(0, "PatientID", meta_int["PatientID"].to_numpy())
    io_ext_df = pd.DataFrame(x_ext_all, columns=input_cols)
    io_ext_df.insert(0, "PatientID", meta_ext["PatientID"].to_numpy())

    summary_rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int, x_int = y_int_all[mask_int], x_int_all[mask_int]
        y_ext, x_ext = y_ext_all[mask_ext], x_ext_all[mask_ext]

        oof = cross_val_predict(LinearRegression(), x_int, y_int, cv=cv)
        model = LinearRegression().fit(x_int, y_int)
        pred_ext = model.predict(x_ext)

        stats_int = regression_significance_stats(y_int, oof)
        stats_ext = regression_significance_stats(y_ext, pred_ext)
        _log(feat, "internal OOF", stats_int)
        _log(feat, "external frozen internal model", stats_ext)
        summary_rows += [{"feature": feat, "cohort": "internal", **stats_int},
                          {"feature": feat, "cohort": "external", **stats_ext}]

        coef_df = pd.DataFrame({
            "term": ["sex_M", "age", "height", "weight", "intercept"],
            "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
        }).round(4)
        feat_dir = OUTPUT_DIR / slug
        feat_dir.mkdir(parents=True, exist_ok=True)
        write_sheets(feat_dir / f"{slug}_coefficients.xlsx", {"clinic4": coef_df})

        int_actual_full = np.full(len(x_int_all), np.nan)
        int_pred_full = np.full(len(x_int_all), np.nan)
        int_actual_full[mask_int], int_pred_full[mask_int] = y_int, oof
        io_int_df[f"{slug}_actual"] = int_actual_full
        io_int_df[f"{slug}_predicted"] = int_pred_full

        ext_actual_full = np.full(len(x_ext_all), np.nan)
        ext_pred_full = np.full(len(x_ext_all), np.nan)
        ext_actual_full[mask_ext], ext_pred_full[mask_ext] = y_ext, pred_ext
        io_ext_df[f"{slug}_actual"] = ext_actual_full
        io_ext_df[f"{slug}_predicted"] = ext_pred_full

        plot_scatter_dual([
            (y_int, oof, stats_int, f"{feat}: Clinical-only Linear Regression (internal, OOF)"),
            (y_ext, pred_ext, stats_ext, f"{feat}: Clinical-only Linear Regression (external, frozen internal model)"),
        ], feat_dir / f"{slug}_linear_regression_scatter.png")

        plot_residual_diagnostics(y_int, oof, y_ext, pred_ext, feat,
                                   feat_dir / f"{slug}_linear_regression_diagnostics.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "clinical_only_linear_regression_summary.csv", index=False)
    print(f"Saved summary to {OUTPUT_DIR / 'clinical_only_linear_regression_summary.csv'}")

    plot_metrics_summary(summary, OUTPUT_DIR / "clinical_only_linear_regression_metrics_summary.png")

    corr_rows = (feature_clinic4_correlations(x_int_all, meta_int, input_cols, "internal")
                 + feature_clinic4_correlations(x_ext_all, meta_ext, input_cols, "external"))
    write_correlation_rows(corr_rows, "clinic4")
    corr_df = pd.DataFrame(corr_rows)
    plot_feature_clinic4_corr_heatmap(corr_df, input_cols, OUTPUT_DIR / "clinical_only_feature_clinic4_correlation_heatmap.png")

    write_sheets(OUTPUT_DIR / "gangnam_io.xlsx", {"clinic4": io_int_df})
    write_sheets(OUTPUT_DIR / "sinchon_io.xlsx", {"clinic4": io_ext_df})


if __name__ == "__main__":
    main()
