from __future__ import annotations

# AEC-128을 N_SEG개 구간으로 나눈 뒤, 그 구간 평균 중 target(체성분 feature)과의 연관성이 높은
# 상위 k개 구간만 선택해 clinic4에 추가했을 때와, N_SEG구간을 전부 넣었을 때의 성능(R^2)을 비교한다.
# 구간 선택(SelectKBest)은 CV 각 fold의 학습 데이터에서만 수행해 leakage를 막는다.

from pathlib import Path
from typing import Any, cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step2" / "segment_select"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

N_SEG = 8  # AEC-128을 몇 구간으로 나눌지 (구간평균 기준 후보 pool 크기)
K_VALUES = list(range(1, N_SEG + 1))  # k=N_SEG는 선택 없이 구간 전부 사용(참조선)

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


# n_seg개 구간 컬럼명 생성 (예: n_seg=8 -> aec_seg8_1..aec_seg8_8)
def segment_col_names(n_seg: int) -> list[str]:
    return [f"aec_seg{n_seg}_{i}" for i in range(1, n_seg + 1)]


# raw AEC-128 행렬(n x 128)을 n_seg개 구간으로 나눠 구간별 평균 행렬(n x n_seg)을 산출
def segment_means(aec_matrix: np.ndarray, n_seg: int) -> np.ndarray:
    chunks = np.array_split(aec_matrix, n_seg, axis=1)
    return np.column_stack([c.mean(axis=1) for c in chunks])


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")

    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 + AEC 구간평균 결합. scaler 생략 시 새로 학습(내부 코호트용).
# 성별을 고정한 남/여 개별 실행에서는 sex가 상수가 되어 계수가 무의미해지므로 include_sex=False로 제외
def clinical_matrix(meta: pd.DataFrame, aec_seg: np.ndarray, scaler: StandardScaler | None = None,
                     include_sex: bool = True):
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    x = np.column_stack([clinic, aec_seg])
    return x, scaler


# clinic4(앞 4열)는 그대로 통과, AEC 구간열(뒤 n_aec열)만 SelectKBest(f_regression)로 top-k 선택하는 파이프라인.
# k>=n_aec면 선택 없이 전부 통과시켜 "전체 구간 사용" 모델과 동일하게 만든다
def build_pipeline(n_clinic: int, n_aec: int, k: int) -> Pipeline:
    clinic_idx = list(range(n_clinic))
    aec_idx = list(range(n_clinic, n_clinic + n_aec))
    aec_step = "passthrough" if k >= n_aec else SelectKBest(score_func=f_regression, k=k)
    ct = ColumnTransformer([("clinic", "passthrough", clinic_idx), ("aec", aec_step, aec_idx)])
    return Pipeline([("select", ct), ("lr", LinearRegression())])


# 내부 전체 데이터로 fit한 파이프라인에서 실제 선택된 AEC 구간 번호(1-based)를 추출
def selected_segment_indices(pipeline: Pipeline, n_seg: int, k: int) -> list[int]:
    if k >= n_seg:
        return list(range(1, n_seg + 1))
    selector: SelectKBest = pipeline.named_steps["select"].named_transformers_["aec"]
    return [int(i) + 1 for i in selector.get_support(indices=True)]


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


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트(top1..topN)만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# k(선택 구간 수)를 x축으로 R^2 추이를 feature별 subplot에 그림.
# internal(OOF)과 external(frozen) 곡선을 한 subplot에 겹쳐 표시하고,
# 각 cohort의 clinic4 baseline을 같은 색 점선으로 함께 표시
def plot_k_trend(pivot_int: pd.DataFrame, pivot_ext: pd.DataFrame, out_path: Path) -> None:
    feat_order = [
        "IMATA_SUM", "LAMA_SUM", "NAMA_SUM", "TAMA_SUM",
        "SAT(피하지방)_SUM", "VAT(내장지방)_SUM", "Total Fat_SUM",
    ]
    feats = [f for f in feat_order if f in pivot_int.index]
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    color_int, color_ext = "#2a78d6", "#e07b1a"

    for ax, feat in zip(axes_flat, feats):
        y_int = pivot_int.loc[feat, [f"top{k}" for k in K_VALUES]].to_numpy(dtype=float)
        y_ext = pivot_ext.loc[feat, [f"top{k}" for k in K_VALUES]].to_numpy(dtype=float)
        ax.plot(K_VALUES, y_int, marker="o", color=color_int, linewidth=1.5, label="internal OOF")
        ax.plot(K_VALUES, y_ext, marker="s", color=color_ext, linewidth=1.5, label="external frozen")
        ax.axhline(pivot_int.loc[feat, "clinic4"], color=color_int, linestyle="--", linewidth=1,
                    label="clinic4 baseline (internal)")
        ax.axhline(pivot_ext.loc[feat, "clinic4"], color=color_ext, linestyle="--", linewidth=1,
                    label="clinic4 baseline (external)")
        ax.set_xticks(K_VALUES)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("선택 구간 수 (k)")
        ax.set_ylabel("R²")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(feats):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved k-trend plot to {out_path}")


# clinic4(include_sex=False면 clinic3) baseline과, +AEC N_SEG구간 중 상위 k개만 선택한 모델들(k=1..N_SEG)을 나란히 학습/평가.
# internal로 학습/평가 후 external에 고정 모델 적용, feature별 R^2와 선택된 구간 번호를 비교
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    aec_seg_int = segment_means(aec_int_raw, N_SEG)
    aec_seg_ext = segment_means(aec_ext_raw, N_SEG)
    seg_names = segment_col_names(N_SEG)

    x_int_all, scaler = clinical_matrix(meta_int, aec_seg_int, include_sex=include_sex)
    x_ext_all, _ = clinical_matrix(meta_ext, aec_seg_ext, scaler, include_sex=include_sex)
    n_clinic = 4 if include_sex else 3

    model_order = ["clinic4"] + [f"top{k}" for k in K_VALUES]

    input_cols = (["sex_M"] if include_sex else []) + ["age_z", "height_z", "weight_z"] + seg_names

    def make_io_df(x_all: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(x_all, columns=input_cols)
        df.insert(0, "PatientID", meta["PatientID"].to_numpy())
        return df

    io_int_sheets = {f"top{k}": make_io_df(x_int_all, meta_int) for k in K_VALUES}
    io_ext_sheets = {f"top{k}": make_io_df(x_ext_all, meta_ext) for k in K_VALUES}

    summary_rows = []
    selected_rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]
        x_int, x_ext = x_int_all[mask_int], x_ext_all[mask_ext]

        # clinic4 baseline (AEC 구간 없이 clinic4 4열만 사용)
        clinic_only_pipeline = Pipeline([
            ("select", ColumnTransformer([("clinic", "passthrough", list(range(n_clinic))),
                                           ("aec", "drop", list(range(n_clinic, n_clinic + N_SEG)))])),
            ("lr", LinearRegression()),
        ])
        oof = cross_val_predict(clinic_only_pipeline, x_int, y_int, cv=cv)
        model = clinic_only_pipeline.fit(x_int, y_int)
        pred_ext = cast(np.ndarray, model.predict(x_ext))
        stats_int = regression_significance_stats(y_int, oof)
        stats_ext = regression_significance_stats(y_ext, pred_ext)
        _log(feat, "clinic4", "internal OOF", stats_int)
        summary_rows += [{"feature": feat, "model": "clinic4", "cohort": "internal", **stats_int},
                          {"feature": feat, "model": "clinic4", "cohort": "external", **stats_ext}]

        for k in K_VALUES:
            model_name = f"top{k}"
            pipeline = build_pipeline(n_clinic, N_SEG, k)
            oof = cross_val_predict(pipeline, x_int, y_int, cv=cv)
            fitted = pipeline.fit(x_int, y_int)
            pred_ext = cast(np.ndarray, fitted.predict(x_ext))

            stats_int = regression_significance_stats(y_int, oof)
            stats_ext = regression_significance_stats(y_ext, pred_ext)
            _log(feat, model_name, "internal OOF", stats_int)
            _log(feat, model_name, "external frozen internal model", stats_ext)
            summary_rows += [{"feature": feat, "model": model_name, "cohort": "internal", **stats_int},
                              {"feature": feat, "model": model_name, "cohort": "external", **stats_ext}]

            seg_idx = selected_segment_indices(fitted, N_SEG, k)
            selected_rows.append({"feature": feat, "k": k,
                                   "selected_segments": ",".join(seg_names[i - 1] for i in seg_idx)})

            int_actual_full = np.full(len(x_int_all), np.nan)
            int_pred_full = np.full(len(x_int_all), np.nan)
            int_actual_full[mask_int], int_pred_full[mask_int] = y_int, oof
            io_int_sheets[model_name][f"{slug}_actual"] = int_actual_full
            io_int_sheets[model_name][f"{slug}_predicted"] = int_pred_full

            ext_actual_full = np.full(len(x_ext_all), np.nan)
            ext_pred_full = np.full(len(x_ext_all), np.nan)
            ext_actual_full[mask_ext], ext_pred_full[mask_ext] = y_ext, pred_ext
            io_ext_sheets[model_name][f"{slug}_actual"] = ext_actual_full
            io_ext_sheets[model_name][f"{slug}_predicted"] = ext_pred_full

    write_sheets(output_dir / "gangnam_io.xlsx", io_int_sheets)
    write_sheets(output_dir / "sinchon_io.xlsx", io_ext_sheets)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "clinic_aec_segment_select_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'clinic_aec_segment_select_summary.csv'}")

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(output_dir / "selected_segments_by_k.csv", index=False)
    print(f"Saved selected-segment log to {output_dir / 'selected_segments_by_k.csv'}")

    # feature x model R^2 비교표 (internal OOF 기준). 컬럼은 k 오름차순으로 고정
    pivot = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="r2")
    pivot = pivot.reindex(columns=model_order)
    for model_name in model_order[1:]:
        pivot[f"delta_r2_vs_clinic4_{model_name}"] = pivot[model_name] - pivot["clinic4"]
    for k in K_VALUES[:-1]:
        pivot[f"delta_r2_vs_top{N_SEG}_top{k}"] = pivot[f"top{k}"] - pivot[f"top{N_SEG}"]
    pivot = pivot.round(4)
    pivot.to_csv(output_dir / "clinic4_vs_topk_r2_comparison.csv")
    print(f"Saved comparison table to {output_dir / 'clinic4_vs_topk_r2_comparison.csv'}")
    print(pivot)

    # feature x model R^2 비교표 (external frozen 기준) — internal에서 좋아 보이는 k가 실제로 일반화되는지 확인용
    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="r2")
    pivot_ext = pivot_ext.reindex(columns=model_order)
    for model_name in model_order[1:]:
        pivot_ext[f"delta_r2_vs_clinic4_{model_name}"] = pivot_ext[model_name] - pivot_ext["clinic4"]
    pivot_ext = pivot_ext.round(4)
    pivot_ext.to_csv(output_dir / "clinic4_vs_topk_r2_comparison_external.csv")
    print(f"Saved external comparison table to {output_dir / 'clinic4_vs_topk_r2_comparison_external.csv'}")
    print(pivot_ext)

    plot_k_trend(pivot, pivot_ext, output_dir / "clinic4_aec_segment_select_k_trend.png")


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
