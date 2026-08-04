from __future__ import annotations

# 임상변수 4개(sex/age/height/weight) + AEC-128을 1~128개 구간(2배씩 증가)으로 나눈 구간별 평균값으로
# 체성분 feature(연속형)를 예측하는 선형회귀 파이프라인. clinic 4개만 쓴 baseline과 구간 수별 R^2 추이를 비교한다.

from pathlib import Path
from typing import Any, cast
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step1"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]  # 128슬라이스를 몇 구간으로 나눠 구간별 평균을 낼지

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


# 구간 수 n_seg별 컬럼명 생성 (예: n_seg=4 -> aec_seg4_1..aec_seg4_4), coef 라벨링용
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


# sex/age/height/weight 행렬 구성 + 표준화(sex 제외) + (있으면) AEC 구간평균 결합. scaler 생략 시 새로 학습(내부 코호트용)
def clinical_matrix(meta: pd.DataFrame, aec: np.ndarray | None = None, scaler: StandardScaler | None = None):
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    clinic4 = np.column_stack([sex_m, scaler.transform(rest)])
    x = clinic4 if aec is None else np.column_stack([clinic4, aec])
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


# 모델명(clinic4_aec_mean{N})을 gangnam_io/sinchon_io.xlsx, {slug}_coefficients.xlsx의 시트명(mean{N})으로 변환
def sheet_name_for(model_name: str) -> str:
    return model_name.replace("clinic4_aec_", "")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
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


CORRELATION_CSV = OUTPUT_DIR / "clinical_only_feature_correlations.csv"


# 통합 상관계수 CSV에서 이 스크립트가 소유한 predictor_group만 교체 저장, 다른 스크립트가 쓴 predictor_group(clinic4 등)은 보존
def write_correlation_rows(rows: list[dict], predictor_group: str) -> None:
    new_df = pd.DataFrame(rows)
    new_df["predictor_group"] = predictor_group
    if "predictor" not in new_df.columns:
        new_df["predictor"] = new_df.apply(lambda row: f"seg{int(row['n_seg'])}_{int(row['segment'])}", axis=1)
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


# output feature 각각과 구간 수 n_seg의 AEC 구간평균 각 구간 간 단순 Pearson r/p를 산출
def feature_aec_correlations(aec_seg: np.ndarray, meta: pd.DataFrame, n_seg: int, cohort: str) -> list[dict]:
    rows = []
    for feat in FEATURES:
        y_all = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        y, x = y_all[mask], aec_seg[mask]
        for seg_i in range(aec_seg.shape[1]):
            r, p = stats.pearsonr(x[:, seg_i], y)
            rows.append({"feature": FEATURES[feat], "cohort": cohort, "n_seg": n_seg,
                         "segment": seg_i + 1, "r": float(r), "p_value": float(p), "n": int(mask.sum())})
    return rows


# output feature 각각과 "clinic4(sex/age/height/weight) + AEC 전체평균(n_seg=1)" 입력변수 각각의 단순 Pearson r/p를 산출
def feature_clinic4_aec_mean_correlations(x_all: np.ndarray, meta: pd.DataFrame, input_cols: list[str],
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


# feature x (clinic4 + AEC 전체평균) 상관계수를 internal/external 나란히 히트맵으로 시각화(칸에 r)
def plot_feature_clinic4_aec_mean_corr_heatmap(corr_df: pd.DataFrame, input_cols: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    predictor_labels = {"sex_M": "Sex(M)", "age_z": "Age", "height_z": "Height", "weight_z": "Weight",
                         "aec_seg1_1": "AEC mean"}
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
    fig.suptitle("Output feature vs clinic4 + AEC mean Pearson r", fontsize=13, fontweight="bold",
                 color=INK_PRIMARY)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-clinic4+AEC mean correlation heatmap to {out_path}")


# feature x AEC 전체평균(aec_seg1_1) 단독 상관계수를 히트맵으로 시각화. internal 위, external 아래로 세로 배치
def plot_feature_aec_mean_only_corr_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    features = list(FEATURES.values())
    cohorts = ["internal", "external"]
    sub_all = corr_df[corr_df["predictor"] == "aec_seg1_1"]

    fig, axes = plt.subplots(len(cohorts), 1, figsize=(6, 0.55 * len(features) + 1.2), squeeze=False)
    im = None
    for row, cohort in enumerate(cohorts):
        ax = axes[row, 0]
        sub = sub_all[sub_all["cohort"] == cohort].set_index("feature").loc[features]
        mat = sub[["r"]].to_numpy()

        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for i in range(mat.shape[0]):
            r_val = mat[i, 0]
            ax.text(0, i, f"{r_val:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(r_val) > 0.5 else INK_PRIMARY)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xticks([])
        ax.set_title(cohort, fontsize=12, fontweight="bold", color=INK_PRIMARY)

    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03, label="Pearson r")
    fig.suptitle("Output feature vs AEC mean Pearson r", fontsize=13, fontweight="bold",
                 color=INK_PRIMARY)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-AEC mean only correlation heatmap to {out_path}")


# clinic4(sex/age/height/weight) 각각과 AEC 전체평균(aec_seg1_1) 간 단순 Pearson r/p를 산출(다중공선성 확인용)
def clinic4_vs_aec_mean_correlations(x_all: np.ndarray, input_cols: list[str], cohort: str) -> list[dict]:
    aec_idx = input_cols.index("aec_seg1_1")
    rows = []
    for j, col in enumerate(input_cols):
        if col == "aec_seg1_1":
            continue
        r, p = stats.pearsonr(x_all[:, j], x_all[:, aec_idx])
        rows.append({"variable": col, "cohort": cohort, "r": float(r), "p_value": float(p), "n": int(x_all.shape[0])})
    return rows


# clinic4 변수 x AEC 전체평균 단독 상관계수를 히트맵으로 시각화(AEC와 가장 겹치는 clinic4 변수 확인용)
def plot_clinic4_vs_aec_mean_corr_heatmap(corr_df: pd.DataFrame, input_cols: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    predictor_labels = {"sex_M": "Sex(M)", "age_z": "Age", "height_z": "Height", "weight_z": "Weight"}
    variables = [c for c in input_cols if c != "aec_seg1_1"]
    cohorts = ["internal", "external"]

    fig, axes = plt.subplots(len(cohorts), 1, figsize=(6, 0.55 * len(variables) + 1.2), squeeze=False)
    im = None
    for row, cohort in enumerate(cohorts):
        ax = axes[row, 0]
        sub = corr_df[corr_df["cohort"] == cohort].set_index("variable").loc[variables]
        mat = sub[["r"]].to_numpy()

        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for i in range(mat.shape[0]):
            r_val = mat[i, 0]
            ax.text(0, i, f"{r_val:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(r_val) > 0.5 else INK_PRIMARY)

        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels([predictor_labels[c] for c in variables])
        ax.set_xticks([])
        ax.set_title(cohort, fontsize=12, fontweight="bold", color=INK_PRIMARY)

    fig.subplots_adjust(top=0.86, hspace=0.6)
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03, label="Pearson r")
    fig.suptitle("clinic4 vs AEC mean", fontsize=13, fontweight="bold", color=INK_PRIMARY, y=1.04)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clinic4 vs AEC mean only correlation heatmap to {out_path}")


# 구간 수(행)별 output feature x AEC 구간 상관 히트맵을 internal/external 나란히 그림.
# 구간이 8개 이하면 셀에 r 표시, 많으면 색만 표시
def plot_feature_aec_correlation_heatmap(corr_df: pd.DataFrame, segment_counts: list[int], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    features = list(FEATURES.values())
    cohorts = ["internal", "external"]
    nrows = len(segment_counts)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 2.0 * nrows), squeeze=False)

    im = None
    for row, n_seg in enumerate(segment_counts):
        for col, cohort in enumerate(cohorts):
            ax = axes[row, col]
            sub = corr_df[(corr_df["n_seg"] == n_seg) & (corr_df["cohort"] == cohort)]
            mat = sub.pivot(index="feature", columns="segment", values="r").loc[features]
            im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            if n_seg <= 8:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        r_val = mat.iat[i, j]
                        ax.text(j, i, f"{r_val:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(r_val) > 0.5 else INK_PRIMARY)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features if col == 0 else [], fontsize=8)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(f"n_seg={n_seg}", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(cohort, fontsize=11, fontweight="bold", color=INK_PRIMARY)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Pearson r")
    fig.suptitle("Output feature vs AEC segment-mean Pearson r (row=n_seg)",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-AEC segment correlation heatmap to {out_path}")


# 구간 수(1->128, 2배씩 증가)를 x축으로 internal/external R^2 추이를 feature별 subplot에 함께 그림.
# clinic4 baseline은 internal/external 각각 점선 기준선으로 표시
def plot_segment_trend(pivot_int: pd.DataFrame, pivot_ext: pd.DataFrame, model_order: list[str], out_path: Path) -> None:
    seg_models = [m for m in model_order if m != "clinic4"]
    seg_counts = [1 if m == "clinic4_aec_mean" else int(m.replace("clinic4_aec_mean", "")) for m in seg_models]

    # 윗줄: IMATA/LAMA/NAMA/TAMA(근육계열), 아랫줄: SAT/VAT/Total Fat(지방계열) 순서로 고정 배치
    feat_order = [
        "IMATA_SUM", "LAMA_SUM", "NAMA_SUM", "TAMA_SUM",
        "SAT(피하지방)_SUM", "VAT(내장지방)_SUM", "Total Fat_SUM",
    ]
    feats = [f for f in feat_order if f in pivot_int.index]
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    order = np.argsort(seg_counts)
    x = np.array(seg_counts)[order]

    for ax, feat in zip(axes_flat, feats):
        y_int = pivot_int.loc[feat, seg_models].to_numpy(dtype=float)[order]
        y_ext = pivot_ext.loc[feat, seg_models].to_numpy(dtype=float)[order]
        ax.plot(x, y_int, marker="o", color="#2a78d6", linewidth=1.5, label="internal")
        ax.plot(x, y_ext, marker="o", color="#d6722a", linewidth=1.5, label="external")
        ax.axhline(pivot_int.loc[feat, "clinic4"], color="#2a78d6", linestyle="--", linewidth=1, label="clinic4 baseline (internal)")
        ax.axhline(pivot_ext.loc[feat, "clinic4"], color="#d6722a", linestyle="--", linewidth=1, label="clinic4 baseline (external)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(seg_counts)
        ax.set_xticklabels(seg_counts)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("AEC 구간 수")
        ax.set_ylabel("R²")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(feats):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved segment trend plot to {out_path}")


# clinic 4개(sex/age/height/weight)만 쓴 baseline과, clinic 4개+AEC 구간평균(SEGMENT_COUNTS구간)을 쓴 모델들을 나란히 학습/평가.
# internal로 학습/평가 후 external에 고정 모델 적용, feature별 R^2/산점도/계수를 산출하고 baseline 대비 delta를 비교
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

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    corr_rows = []
    for n_seg in SEGMENT_COUNTS:
        corr_rows += feature_aec_correlations(segment_means(aec_int_raw, n_seg), meta_int, n_seg, "internal")
        corr_rows += feature_aec_correlations(segment_means(aec_ext_raw, n_seg), meta_ext, n_seg, "external")
    write_correlation_rows(corr_rows, "aec_segment")
    corr_df = pd.DataFrame(corr_rows)
    plot_feature_aec_correlation_heatmap(corr_df, SEGMENT_COUNTS,
                                          OUTPUT_DIR / "clinical_only_feature_aec_segment_correlation_heatmap.png")

    model_order = ["clinic4"]
    models = {"clinic4": {"aec_int": None, "aec_ext": None,
                           "input_cols": ["sex_M", "age_z", "height_z", "weight_z"],
                           "coef_names": ["sex_M", "age", "height", "weight", "intercept"]}}
    for n_seg in SEGMENT_COUNTS:
        aec_cols = segment_col_names(n_seg)
        model_name = "clinic4_aec_mean" if n_seg == 1 else f"clinic4_aec_mean{n_seg}"
        model_order.append(model_name)
        models[model_name] = {
            "aec_int": segment_means(aec_int_raw, n_seg),
            "aec_ext": segment_means(aec_ext_raw, n_seg),
            "input_cols": ["sex_M", "age_z", "height_z", "weight_z"] + aec_cols,
            "coef_names": ["sex_M", "age", "height", "weight"] + aec_cols + ["intercept"],
        }

    for model_name, spec in models.items():
        x_int_all, scaler = clinical_matrix(meta_int, spec["aec_int"])
        x_ext_all, _ = clinical_matrix(meta_ext, spec["aec_ext"], scaler)
        spec["x_int_all"], spec["x_ext_all"] = x_int_all, x_ext_all

        spec["io_int_df"] = pd.DataFrame(x_int_all, columns=spec["input_cols"])
        spec["io_int_df"].insert(0, "PatientID", meta_int["PatientID"].to_numpy())
        spec["io_ext_df"] = pd.DataFrame(x_ext_all, columns=spec["input_cols"])
        spec["io_ext_df"].insert(0, "PatientID", meta_ext["PatientID"].to_numpy())

    clinic4_mean_spec = models["clinic4_aec_mean"]
    clinic4_mean_corr_rows = (
        feature_clinic4_aec_mean_correlations(clinic4_mean_spec["x_int_all"], meta_int,
                                               clinic4_mean_spec["input_cols"], "internal")
        + feature_clinic4_aec_mean_correlations(clinic4_mean_spec["x_ext_all"], meta_ext,
                                                 clinic4_mean_spec["input_cols"], "external")
    )
    write_correlation_rows(clinic4_mean_corr_rows, "clinic4_aec_mean")
    plot_feature_clinic4_aec_mean_corr_heatmap(
        pd.DataFrame(clinic4_mean_corr_rows), clinic4_mean_spec["input_cols"],
        OUTPUT_DIR / "clinical_only_feature_clinic4_aec_mean_correlation_heatmap.png")
    plot_feature_aec_mean_only_corr_heatmap(
        pd.DataFrame(clinic4_mean_corr_rows),
        OUTPUT_DIR / "clinical_only_feature_aec_mean_only_correlation_heatmap.png")

    clinic4_aec_corr_rows = (
        clinic4_vs_aec_mean_correlations(clinic4_mean_spec["x_int_all"], clinic4_mean_spec["input_cols"], "internal")
        + clinic4_vs_aec_mean_correlations(clinic4_mean_spec["x_ext_all"], clinic4_mean_spec["input_cols"], "external")
    )
    plot_clinic4_vs_aec_mean_corr_heatmap(
        pd.DataFrame(clinic4_aec_corr_rows), clinic4_mean_spec["input_cols"],
        OUTPUT_DIR / "clinic4_vs_aec_mean_correlation_heatmap.png")

    summary_rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)

        mask_int = np.isfinite(y_int_all)
        mask_ext = np.isfinite(y_ext_all)
        y_int = y_int_all[mask_int]
        y_ext = y_ext_all[mask_ext]

        feat_dir = OUTPUT_DIR / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        model_stats = {}
        coef_sheets = {}
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
            model_stats[model_name] = stats_int

            if model_name != "clinic4":  # clinic4 계수는 step0_clinic-only_baseline.py가 소유
                coef_df = pd.DataFrame({
                    "term": spec["coef_names"],
                    "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
                }).round(4)
                coef_sheets[sheet_name_for(model_name)] = coef_df

            int_actual_full = np.full(len(spec["x_int_all"]), np.nan)
            int_pred_full = np.full(len(spec["x_int_all"]), np.nan)
            int_actual_full[mask_int], int_pred_full[mask_int] = y_int, oof
            spec["io_int_df"][f"{slug}_actual"] = int_actual_full
            spec["io_int_df"][f"{slug}_predicted"] = int_pred_full

            ext_actual_full = np.full(len(spec["x_ext_all"]), np.nan)
            ext_pred_full = np.full(len(spec["x_ext_all"]), np.nan)
            ext_actual_full[mask_ext], ext_pred_full[mask_ext] = y_ext, pred_ext
            spec["io_ext_df"][f"{slug}_actual"] = ext_actual_full
            spec["io_ext_df"][f"{slug}_predicted"] = ext_pred_full

            plot_scatter_dual([
                (y_int, oof, stats_int, f"{feat} ({model_name}): internal, OOF"),
                (y_ext, pred_ext, stats_ext, f"{feat} ({model_name}): external, frozen internal model"),
            ], feat_dir / f"{slug}_{model_name}_linear_regression_scatter.png")

        for model_name in model_order[1:]:
            delta_r2 = model_stats[model_name]["r2"] - model_stats["clinic4"]["r2"]
            print(f"[{feat}] internal OOF R^2 delta ({model_name} - clinic4) = {delta_r2:+.4f}")

        write_sheets(feat_dir / f"{slug}_coefficients.xlsx", coef_sheets)

    int_sheets = {sheet_name_for(m): spec["io_int_df"] for m, spec in models.items() if m != "clinic4"}
    ext_sheets = {sheet_name_for(m): spec["io_ext_df"] for m, spec in models.items() if m != "clinic4"}
    write_sheets(OUTPUT_DIR / "gangnam_io.xlsx", int_sheets)
    write_sheets(OUTPUT_DIR / "sinchon_io.xlsx", ext_sheets)

    summary = pd.DataFrame(summary_rows)

    # clinic4 vs 구간 수별 clinic4_aec_mean{N} R^2 비교표 (internal OOF 기준). 컬럼은 구간 수 오름차순으로 고정
    # (pandas pivot은 컬럼을 알파벳순 정렬하므로 mean16이 mean2보다 앞에 오는 문제를 막기 위해 명시적으로 재정렬)
    pivot = summary[summary["cohort"] == "internal"].pivot(index="feature", columns="model", values="r2")
    pivot = pivot.reindex(columns=model_order)
    aec_model_names = model_order[1:]
    for model_name in aec_model_names:
        pivot[f"delta_r2_{model_name}"] = pivot[model_name] - pivot["clinic4"]
    pivot = pivot.round(4)
    print(pivot)

    pivot_ext = summary[summary["cohort"] == "external"].pivot(index="feature", columns="model", values="r2")
    pivot_ext = pivot_ext.reindex(columns=model_order).round(4)

    # 이 스크립트가 소유한 시트(regression_summary/r2_comparison)만 교체 저장.
    # 다른 파생 분석(generalization_gap 등)이 쓰는 시트는 clinic4_aec_mean_summary.xlsx에 그대로 보존됨
    summary_path = OUTPUT_DIR / "clinic4_aec_mean_summary.xlsx"
    write_sheets(summary_path, {
        "regression_summary": summary,
        "r2_comparison": pivot.reset_index(),
    })

    plot_segment_trend(pivot, pivot_ext, model_order, OUTPUT_DIR / "clinic4_aec_mean_segment_r2_trend.png")


if __name__ == "__main__":
    main()
