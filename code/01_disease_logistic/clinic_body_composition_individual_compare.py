from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0910"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
AGE_CUTOFF = 20
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
FPCA_N_FIXED = 3  # [[project_fpca_n_fixed_to_6]]: 최종 고정값 3(elbow와 동일), 기존 AEC 비교 스크립트와 동일 기준
FPCA_COMPONENT_CANDIDATES_MAX = 20
MIN_POSITIVES = 2

CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

# 사용자 요청(2026-09-10): gangnam/sinchon_final_dataset.xlsx의 metadata 시트에 AEC 말고도 체성분
# 스칼라 변수(BMI/SMI/VAT/SAT/IMATA/NAMA/LAMA/TAMA)가 이미 병합되어 있음(구버전처럼 CT_AEC_process에서
# 별도 병합할 필요 없음). 이를 AEC와 동일한 방식(clinic4에 변수 1개씩 개별 추가)으로 성능 비교
# 사용자 확인(2026-09-10): (1) 8개 변수 전부 재비교 (2) 변수 1개씩 개별 추가 (3) baseline은 clinic4 단독
BODY_COMP_COLS = ["BMI", "SMI", "VAT", "SAT", "IMATA", "NAMA", "LAMA", "TAMA"]
# VIF 점검 전용 목록(실제 모델 비교 MODEL_ORDER에는 포함 안 함) - PI 요청(2026-09-16): "BMI/SMI/TAMA/LAMA/
# NAMA/IMATA/VAT/SAT/TAT 넣을 때 VIF 확인". TAT=VAT+SAT를 스칼라 9종 동시투입 체크에만 추가로 사용
VIF_SCALAR_COLS = BODY_COMP_COLS + ["TAT"]

# 사용자 요청(2026-09-10): "aec에서 upper/lower비율을 사용한 것처럼 체성분도 진행" + "체성분에 대해서도
# FPCA도 비교 추가". metadata의 스칼라값과 별도로 SFA_128/VFA_128/TAMA_128/LAMA_128/NAMA_128/
# IMATA_128 시트가 aec_128과 동일 구조(PatientID당 liver->pubis 128구간 곡선)로 존재하므로, AEC와 동일한
# FPCA(3)/Upper-Lower ratio 두 표현을 그대로 적용. SFA=SAT, VFA=VAT의 원본 약어. BMI/SMI는 슬라이스 단위
# 측정값이 아니라 곡선 시트가 없어 FPCA/ratio를 만들 수 없으므로 제외
CURVE_SOURCES: dict[str, tuple[str, str]] = {
    "AEC": ("aec_128", "aec"),
    "VAT": ("VFA_128", "VFA"),
    "SAT": ("SFA_128", "SFA"),
    "IMATA": ("IMATA_128", "IMATA"),
    "NAMA": ("NAMA_128", "NAMA"),
    "LAMA": ("LAMA_128", "LAMA"),
    "TAMA": ("TAMA_128", "TAMA"),
}

# 사용자 요청(2026-09-10): "fpca할때 고정하지말고 elbow n 값으로 조절해서 ... 비교". 직전 elbow(Kneedle)
# 검증 결과(outputs/0910/fpca_loading/elbow_diagnostics.csv, 이번 실행값) AEC/IMATA만 elbow=3(고정값과
# 일치)이고 VAT/SAT/NAMA/LAMA/TAMA는 전부 elbow=2 - 이 값을 그대로 고정 모델 파라미터로 채택(0907
# BEST_LOGREG_PARAMS와 동일하게, 매 실행마다 elbow를 다시 계산해 모델 구조까지 바꾸지 않고 검증된 값을
# 고정 사용). run()에서 select_fpca_n_by_elbow로 재계산한 elbow와 다르면 경고만 출력(비치명적)
CURVE_FPCA_N_ELBOW: dict[str, int] = {"AEC": 3, "VAT": 2, "SAT": 2, "IMATA": 3, "NAMA": 2, "LAMA": 2, "TAMA": 2}


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


# 사용자 요청(2026-09-10): "VAT/SAT ratio를 사용한 것처럼 다른 체성분도 비율을 할 수 있는게 있으면 진행".
# metadata 스칼라값끼리의 비율 중 문헌상 의미가 확립된 지표만 선정: VAT/SAT(내장/피하지방 비율, 중심비만
# 지표), LAMA/TAMA(myosteatosis 비율 - 저감쇠 근육이 전체 근육에서 차지하는 비중,
# [[feedback_lama_direction_is_fat_like]]와 동일하게 LAMA=지방침윤 근육으로 취급), IMATA/TAMA(근육 내
# 지방침윤 비율). NAMA/TAMA는 1-LAMA/TAMA와 사실상 동일 정보라 중복 제외
INTER_VAR_RATIOS: dict[str, tuple[str, str]] = {
    "VAT_SAT_ratio": ("VAT", "SAT"),
    "LAMA_TAMA_ratio": ("LAMA", "TAMA"),
    "IMATA_TAMA_ratio": ("IMATA", "TAMA"),
}

MODEL_ORDER = ["clinic4"] + [f"clinic4_{c.lower()}" for c in BODY_COMP_COLS]
MODEL_ORDER += [f"clinic4_{r.lower()}" for r in INTER_VAR_RATIOS]
for _key in CURVE_SOURCES:
    MODEL_ORDER += [f"clinic4_{_key.lower()}_fpca_elbow", f"clinic4_{_key.lower()}_uplow_ratio"]
# 사용자 요청(2026-09-10): "fpca랑 upper/lower ratio 말고 다른 방법 있나" -> slope(liver→pubis 선형추세)
# 제안 채택. "시도해보고 결과가 좋지 않으면 바로 제거해"라는 조건부 실험이므로 별도 section으로 격리
for _key in CURVE_SOURCES:
    MODEL_ORDER += [f"clinic4_{_key.lower()}_slope"]

MODEL_EXTRA_COLS: dict[str, list[str]] = {"clinic4": []}
for _c in BODY_COMP_COLS:
    MODEL_EXTRA_COLS[f"clinic4_{_c.lower()}"] = [_c]
for _r in INTER_VAR_RATIOS:
    MODEL_EXTRA_COLS[f"clinic4_{_r.lower()}"] = [_r]
for _key in CURVE_SOURCES:
    MODEL_EXTRA_COLS[f"clinic4_{_key.lower()}_fpca_elbow"] = []
    MODEL_EXTRA_COLS[f"clinic4_{_key.lower()}_uplow_ratio"] = [f"{_key}_uplow_ratio"]
    MODEL_EXTRA_COLS[f"clinic4_{_key.lower()}_slope"] = [f"{_key}_slope"]

# 사용자 요청(2026-09-10): "결과값을 하나의 png로 하지말고 여러개의 section으로 나눠서 저장". 모델이
# 26개(clinic4 포함)로 늘어나 표/그래프 1장에 다 담으면 가독성이 떨어지므로, 도입 순서대로 4개 section
# (스칼라 8종 / 변수간 비율 3종 / 곡선 FPCA(elbow n) 7종 / 곡선 Upper-Lower ratio 7종)으로 나눠 각각 별도
# 파일로 저장. 각 section에는 비교 기준을 보여주기 위해 clinic4를 항상 첫 행으로 포함
MODEL_SECTIONS: dict[str, list[str]] = {
    "scalar": ["clinic4"] + [f"clinic4_{c.lower()}" for c in BODY_COMP_COLS],
    "inter_var_ratio": ["clinic4"] + [f"clinic4_{r.lower()}" for r in INTER_VAR_RATIOS],
    "curve_fpca_elbow": ["clinic4"] + [f"clinic4_{k.lower()}_fpca_elbow" for k in CURVE_SOURCES],
    "curve_uplow_ratio": ["clinic4"] + [f"clinic4_{k.lower()}_uplow_ratio" for k in CURVE_SOURCES],
    "curve_slope": ["clinic4"] + [f"clinic4_{k.lower()}_slope" for k in CURVE_SOURCES],
}
SECTION_TITLES = {
    "scalar": "Body composition scalar variables",
    "inter_var_ratio": "Inter-variable ratios (VAT/SAT, LAMA/TAMA, IMATA/TAMA)",
    "curve_fpca_elbow": "Curve FPCA(elbow-selected n, per curve)",
    "curve_uplow_ratio": "Curve Upper/Lower ratio",
    "curve_slope": "Curve slope (liver->pubis linear trend)",
}
MODEL_TO_SECTION = {m: sec for sec, models in MODEL_SECTIONS.items() for m in models if m != "clinic4"}

# FPCA 기반 모델만 fold별 PCA refit이 필요한 fpca_oof_proba 경로를 탄다(곡선 원본에서 직접 파생). 값이
# 있으면 해당 모델이 사용할 CURVE_SOURCES 키(어느 곡선에서 FPCA를 뽑을지), 없으면 일반 extra_cols 경로.
# n_components는 고정값이 아니라 CURVE_FPCA_N_ELBOW[key]를 그대로 사용(사용자 요청: "고정하지말고 elbow n
# 값으로 조절")
MODEL_CURVE_SOURCE: dict[str, str | None] = {m: None for m in MODEL_ORDER}
for _key in CURVE_SOURCES:
    MODEL_CURVE_SOURCE[f"clinic4_{_key.lower()}_fpca_elbow"] = _key

MODEL_LABELS = {"clinic4": "clinic4"}
MODEL_LABELS.update({f"clinic4_{c.lower()}": f"clinic4 + {c}" for c in BODY_COMP_COLS})
MODEL_LABELS.update({f"clinic4_{r.lower()}": f"clinic4 + {r.replace('_', '/', 1).replace('_ratio', '')}"
                      for r in INTER_VAR_RATIOS})
MODEL_LABELS.update({f"clinic4_{k.lower()}_fpca_elbow": f"clinic4 + {k} FPCA(n={CURVE_FPCA_N_ELBOW[k]})"
                      for k in CURVE_SOURCES})
MODEL_LABELS.update({f"clinic4_{k.lower()}_uplow_ratio": f"clinic4 + {k} Upper/Lower ratio" for k in CURVE_SOURCES})
MODEL_LABELS.update({f"clinic4_{k.lower()}_slope": f"clinic4 + {k} slope" for k in CURVE_SOURCES})
MODEL_LABELS["bodycomp_only"] = "clinic4 + 체성분(질환별 최고, 고정)"

# 사용자 요청(2026-09-10): "체성분 데이터까지를 고정으로하고 aec의 유무에 대한 비교도 진행". 지금까지는
# clinic4 baseline에 변수 1개씩만 추가해 AEC와 체성분을 "나란히" 비교했다면, 이번엔 질환별로 가장 성능이
# 좋았던 체성분 표현(스칼라/비율/uplow_ratio/slope 중 - FPCA는 추상적 투영이라 "체성분 데이터를 고정"한다는
# 취지에 안 맞아 제외)을 baseline에 고정으로 얹은 뒤, 그 위에 AEC(3가지 표현)를 추가했을 때 추가 개선이
# 있는지를 본다 - "이미 최선의 체성분 신호가 들어간 모델에 AEC가 더 보탤 게 있는가"라는 질문
AEC_ARM_MODELS = ["clinic4_aec_fpca_elbow", "clinic4_aec_uplow_ratio", "clinic4_aec_slope"]
AEC_ARM_LABELS = {"clinic4_aec_fpca_elbow": "AEC FPCA", "clinic4_aec_uplow_ratio": "AEC Upper/Lower ratio",
                   "clinic4_aec_slope": "AEC slope"}
# 고정 체성분 baseline 후보 = FPCA가 아닌 모델(및 AEC 자신은 제외)
NONAEC_NONFPCA_MODELS = [m for m in MODEL_ORDER if m != "clinic4" and "aec" not in m
                          and MODEL_CURVE_SOURCE[m] is None]

# baseline(clinic4) 대비 각 확장 모델의 DeLong 검정. AEC를 함께 넣어 체성분 변수들의 개선폭을 AEC와
# 나란히 비교할 수 있게 함(사용자 요청: "aec처럼 사용해서 성능 비교")
DELONG_PAIRS = [("clinic4", m) for m in MODEL_ORDER if m != "clinic4"]

# 하이퍼파라미터는 변수별/질환별 그리드서치를 새로 돌리지 않고 0907 스크립트의 미탐색 모델과 동일한
# 기본값(C=1.0, L2, lbfgs)을 전부에 동일 적용(사용자에게 튜닝 범위 관련 요청 없었음)
LOGREG_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 5000}

_REF_AVG_DIM = (16.0 + 11.0) / 2
_LABEL_FS_RATIO = 32.5 / _REF_AVG_DIM
_TICK_FS_RATIO = 30.0 / _REF_AVG_DIM
_LEGEND_FS_RATIO = 27.5 / _REF_AVG_DIM


def scaled_fontsizes(width: float, height: float) -> tuple[float, float, float]:
    avg_dim = (width + height) / 2
    return _LABEL_FS_RATIO * avg_dim, _TICK_FS_RATIO * avg_dim, _LEGEND_FS_RATIO * avg_dim


# metadata(체성분 스칼라값 포함) + AEC/체성분 6종의 128구간 곡선 시트를 모두 병합하고, 각 곡선의
# Upper(앞쪽 64구간)/Lower(뒤쪽 64구간) 평균 비율을 계산
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    for ratio_col, (num_col, den_col) in INTER_VAR_RATIOS.items():
        meta[ratio_col] = meta[num_col].astype(float) / meta[den_col].astype(float)
    # VIF 점검 전용(실제 모델 비교 대상 아님) - PI 요청(2026-09-16): TAT(총 지방면적)=VAT+SAT 넣었을 때도 확인
    meta["TAT"] = meta["VAT"].astype(float) + meta["SAT"].astype(float)
    n0 = len(meta)
    for key, (sheet, prefix) in CURVE_SOURCES.items():
        curve = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
        meta = meta.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
        assert len(meta) == n0, f"{xlsx_path.name}: metadata/{sheet} merge dropped rows"

    half = N_SLICES // 2
    for key, (_sheet, prefix) in CURVE_SOURCES.items():
        cols = curve_cols(prefix)
        upper_mean = meta[cols[:half]].astype(float).mean(axis=1)
        lower_mean = meta[cols[half:]].astype(float).mean(axis=1)
        meta[f"{key}_uplow_ratio"] = upper_mean / lower_mean

    # 사용자 요청(2026-09-10): "fpca랑 upper/lower ratio로 input활용하고 있는데 다른 방법이 더 있나" ->
    # slope(liver→pubis 선형추세) 채택. 환자별 곡선을 slice index(1..128)에 대해 최소제곱 직선으로 맞췄을
    # 때의 기울기. 벡터화된 최소제곱: slope = sum((x-x̄)*y) / sum((x-x̄)^2), y_mean 항은 x가 중심화돼 있어
    # 자동 상쇄되므로 행렬곱 한 번으로 전체 환자 처리 가능
    x = np.arange(1, N_SLICES + 1, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    for key, (_sheet, prefix) in CURVE_SOURCES.items():
        vals = meta[curve_cols(prefix)].astype(float).to_numpy()
        meta[f"{key}_slope"] = (vals @ x_centered) / denom
    return meta


# clinic4(age/height/weight+sex) + extra_cols(체성분 스칼라/ratio 변수 1개, fold와 무관하게 이미 확정된
# 값) + (curve_extra가 있으면 FPCA 모델의 fold별 FPCA 점수)를 결합. sex만 0/1 명목형이라 스케일링하지
# 않고 그대로 붙이고, 연속형(age/height/weight/체성분 변수)은 StandardScaler로 표준화. scaler는
# internal에서 fit해 external에 frozen 적용([[feedback_internal_external_validation_discipline]])
def build_matrix(meta: pd.DataFrame, extra_cols: list[str], curve_extra: np.ndarray | None = None,
                  scaler: StandardScaler | None = None, curve_scaler: StandardScaler | None = None
                  ) -> tuple[np.ndarray, StandardScaler, StandardScaler | None]:
    cols = CLINICAL_BASE_COLS + extra_cols
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    clinic = np.column_stack([sex_m, scaled])
    if curve_extra is None:
        return clinic, scaler, None
    if curve_scaler is None:
        curve_scaler = StandardScaler().fit(curve_extra)
    x = np.column_stack([clinic, curve_scaler.transform(curve_extra)])
    return x, scaler, curve_scaler


def input_cols_for(model_name: str) -> list[str]:
    curve_key = MODEL_CURVE_SOURCE[model_name]
    cols = ["sex_M", "age", "height", "weight"] + MODEL_EXTRA_COLS[model_name]
    if curve_key is not None:
        cols += [f"{curve_key.lower()}_fpca_pc{i}" for i in range(1, CURVE_FPCA_N_ELBOW[curve_key] + 1)]
    return cols


# 사용자 요청(2026-09-16): "실행할때 모델별로 VIF도 확인할 수 있게". 이미 fit에 쓰는 x_int_full(clinic4 +
# 그 모델의 추가 변수, 표준화됨)에 그대로 VIF를 계산 - VIF는 스케일에 불변이라 표준화 여부와 무관
def compute_vif(x_int_full: np.ndarray, input_cols: list[str], added_cols: list[str]) -> pd.DataFrame:
    x = add_constant(x_int_full)
    vif = pd.Series([variance_inflation_factor(x, i) for i in range(1, x.shape[1])], index=input_cols, name="vif")
    df = vif.reset_index().rename(columns={"index": "variable"})
    df["is_added"] = df["variable"].isin(added_cols)
    return df


VIF_EXACT_THRESHOLD = 1000  # 항등식(TAT=VAT+SAT, TAMA=NAMA+LAMA)이 slope/FPCA처럼 선형연산을 거치면
# 부동소수점 오차로 정확히 inf는 아니어도 VIF가 수억 단위로 튀는 경우가 있어 "사실상 완전공선"으로 같이 취급
SECTION_COLORS = {
    "scalar": "#2c7fb8", "inter_var_ratio": "#41ab5d", "curve_uplow_ratio": "#e67e22",
    "curve_slope": "#8856a7", "curve_fpca_elbow": "#c0392b",
}


def _vif_bar_color(v: float) -> str:
    if not np.isfinite(v) or v >= 10:
        return "#c0392b"
    return "#e67e22" if v >= 5 else "#2c7fb8"


def _annotate_vif_bars(ax, bars, raw_values: pd.Series) -> None:
    for bar, raw in zip(bars, raw_values):
        is_exact = (not np.isfinite(raw)) or raw >= VIF_EXACT_THRESHOLD
        label = " VIF=inf(exact)" if is_exact else f" {raw:.1f}"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)


# 사용자 요청(2026-09-16): "PI가 BMI/SMI/TAMA/LAMA/NAMA/IMATA/VAT/SAT/TAT 넣을 때 VIF 확인해봐야 할 것
# 같다" - 스칼라 9종(TAT 포함, VIF 점검 전용)을 clinic4와 함께 전부 한 모델에 넣었을 때의 VIF
def plot_vif_scalar_combined(feat: str, meta_int_m: pd.DataFrame, meta_ext_m: pd.DataFrame, out_path: Path) -> None:
    x_int, scaler, _ = build_matrix(meta_int_m, VIF_SCALAR_COLS)
    x_ext, _, _ = build_matrix(meta_ext_m, VIF_SCALAR_COLS, scaler=scaler)
    cols = ["sex_M", "age", "height", "weight"] + VIF_SCALAR_COLS

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, cohort, x_full in zip(axes, ["internal", "external"], [x_int, x_ext]):
        vif = compute_vif(x_full, cols, VIF_SCALAR_COLS)
        display_vif = vif["vif"].clip(upper=50)
        colors = [_vif_bar_color(v) for v in vif["vif"]]
        bars = ax.barh(vif["variable"], display_vif, color=colors)
        _annotate_vif_bars(ax, bars, vif["vif"])
        ax.axvline(5, color="#e67e22", linestyle="--", linewidth=1, label="VIF=5")
        ax.axvline(10, color="#c0392b", linestyle="--", linewidth=1, label="VIF=10")
        ax.set_title(cohort, fontweight="bold")
        ax.set_xlabel("VIF")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"{feat}: VIF — clinic4 + body composition scalar 9종 동시 투입", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved VIF plot to {out_path}")


# 사용자 요청(2026-09-16): "clinic4+TAMA, clinic4+IMATA, clinic4+LAMA 등으로 다 나눠서 비교" - 실제
# fit되는 "clinic4 + 변수 1개"짜리 개별모델의 VIF를 섹션(또는 전체)별로 모아서 표시
def plot_vif_per_model(vif_df_feat: pd.DataFrame, model_list: list[str], title: str, out_path: Path,
                        color_by_section: bool = False) -> None:
    models = [m for m in model_list if m != "clinic4"]
    fig, axes = plt.subplots(1, 2, figsize=(9, max(3.0, 0.35 * len(models) + 1.5)), sharey=True)
    for ax, cohort in zip(axes, ["internal", "external"]):
        rows = []
        for m in models:
            sub = vif_df_feat[(vif_df_feat["model"] == m) & (vif_df_feat["cohort"] == cohort)
                               & vif_df_feat["is_added"]]
            rows.append({"model": MODEL_LABELS[m], "vif": sub["vif"].max(),
                         "section": MODEL_TO_SECTION.get(m, "scalar")})
        df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
        display_vif = df["vif"].clip(upper=50)
        colors = ([SECTION_COLORS[s] for s in df["section"]] if color_by_section
                  else [_vif_bar_color(v) for v in df["vif"]])
        bars = ax.barh(df["model"], display_vif, color=colors)
        _annotate_vif_bars(ax, bars, df["vif"])
        ax.axvline(5, color="#e67e22", linestyle="--", linewidth=1)
        ax.axvline(10, color="#c0392b", linestyle="--", linewidth=1)
        ax.set_title(cohort, fontweight="bold")
        ax.set_xlabel("VIF of added variable")
        ax.grid(alpha=0.3, axis="x")
    if color_by_section:
        handles = [plt.Line2D([0], [0], color=c, lw=6) for c in SECTION_COLORS.values()]
        fig.legend(handles, SECTION_COLORS.keys(), loc="lower center", ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1) if color_by_section else (0, 0, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VIF plot to {out_path}")


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


# 곡선 원본에서 fold별 PCA를 검증 fold를 제외한 학습 fold에서만 fit해 곡선 정보 누수를 막는다
def fpca_oof_proba(meta: pd.DataFrame, curve_raw: np.ndarray, y: np.ndarray, n_fpca: int,
                    cv: StratifiedKFold, extra_cols: list[str] | None = None) -> np.ndarray:
    extra_cols = extra_cols or []
    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(curve_raw, y):
        pca = PCA(n_components=n_fpca, random_state=SEED).fit(curve_raw[train_idx])
        fpca_train, fpca_test = pca.transform(curve_raw[train_idx]), pca.transform(curve_raw[test_idx])
        x_train, scaler, curve_scaler = build_matrix(meta.iloc[train_idx], extra_cols, fpca_train)
        x_test, _, _ = build_matrix(meta.iloc[test_idx], extra_cols, fpca_test, scaler, curve_scaler)
        model = LogisticRegression(**LOGREG_PARAMS).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


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


def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"),
                "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# AUC의 bootstrap 95% CI 산출(external frozen 평가에만 적용, internal은 5-fold CV OOF 점추정만 사용)
def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float]:
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


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# 사용자 요청(2026-09-18): "internal 성능 확인 할 때에는 OOF보다는 Fold별 성능으로 mean, std로 표현해" -
# OOF(pooled) 예측 1개 값 대신, 같은 fold 분할(y와 SEED만으로 결정되므로 모델과 무관하게 재현 가능)로 나눠
# fold별 지표를 구하고 mean/std로 요약. internal_fold_id는 cross_val_predict/fpca_oof_proba가 내부적으로
# 쓰는 것과 동일한 StratifiedKFold(shuffle=True, random_state=SEED) 분할을 밖에서 재현한다
def internal_fold_id(y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)
    fold_id = np.empty(len(y), dtype=int)
    for fi, (_, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        fold_id[test_idx] = fi
    return fold_id


def fold_metric_stats(y: np.ndarray, score: np.ndarray, metric_fn) -> tuple[float, float]:
    fold_id = internal_fold_id(y)
    vals = [metric_fn(y[fold_id == fi], score[fold_id == fi]) for fi in np.unique(fold_id)]
    return float(np.mean(vals)), float(np.std(vals))


def fold_classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    fold_id = internal_fold_id(y)
    rows = [classification_stats(y[fold_id == fi], score[fold_id == fi], threshold) for fi in np.unique(fold_id)]
    df = pd.DataFrame(rows)
    return {**df.mean().to_dict(), **{f"{c}_std": v for c, v in df.std(ddof=0).to_dict().items()}}


def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


_PC_COLORS = {1: "#1f5fa8", 2: "#d9791e", 3: "#2ea86b"}


# 사용자 요청(2026-09-10): "혹시 모르니까 체성분에 대해서 elbow needle 구해서 n수 검증". [[project_fpca_n_
# fixed_to_6]]에서 AEC는 elbow=고정값(3)임을 이미 확인했으나 체성분 6종은 검증한 적이 없어, 0907
# select_fpca_n_by_elbow와 동일한 Kneedle(누적분산 scree의 첫점-끝점 직선에서 최대거리) 로직을 그대로
# 적용해 FPCA_N_FIXED=3이 각 곡선에도 타당한지 참고용으로 확인만 한다(모델에는 영향 없음)
def select_fpca_n_by_elbow(curve_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, curve_raw.shape[0], curve_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(curve_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))

    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    elbow_n = int(np.argmax(dist)) + 1
    return elbow_n, cum_var


def plot_elbow_diagnostics(cum_var_by_key: dict[str, pd.Series], elbow_by_key: dict[str, int],
                            out_path: Path) -> None:
    keys = list(cum_var_by_key)
    ncols = 4
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    flat_axes = np.atleast_2d(axes).flatten()
    for ax, key in zip(flat_axes, keys):
        cum_var = cum_var_by_key[key]
        elbow_n = elbow_by_key[key]
        ax.plot(cum_var.index, cum_var.to_numpy(), color="#1f5fa8", linewidth=1.8, marker="o", markersize=3)
        ax.axvline(elbow_n, color="#d9791e", linestyle="--", linewidth=1.3)
        ax.axvline(FPCA_N_FIXED, color="#2ea86b", linestyle=":", linewidth=1.3)
        ax.set_title(f"{key} (elbow n={elbow_n}, 고정 n={FPCA_N_FIXED})", fontsize=11, fontweight="bold")
        ax.set_xlabel("n_components", fontsize=9)
        ax.set_ylabel("누적 explained variance", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in flat_axes[len(keys):]:
        ax.axis("off")
    fig.suptitle("FPCA n_components elbow(Kneedle) 검증 — 주황선=elbow, 초록선=고정값(3)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved elbow diagnostics plot to {out_path}")


# 사용자 요청(2026-09-10): "aec의 fpca 그래프를 확인했듯이 체성분의 fpca도 확인 가능하게".
# code/03_aec_deep_learning/compare/aec_fpca_loading_curves.py와 동일한 2패널 스타일(평균 곡선 +
# PC1-3 loading curve/eigenfunction)을 AEC뿐 아니라 나머지 6개 체성분 곡선(VAT/SAT/IMATA/NAMA/LAMA/TAMA)
# 에도 그대로 적용 - run()에서 이미 fit한 PCA 객체를 그대로 재사용(재계산하지 않음)
def plot_fpca_loading_curves(key: str, pca: PCA, out_path: Path) -> None:
    evr = pca.explained_variance_ratio_
    mean_curve = pca.mean_
    slices = list(range(1, N_SLICES + 1))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.plot(slices, mean_curve, color="#161616", linewidth=2.5, label="internal 평균 곡선")
    ax.set_title(f"internal 코호트 평균 {key} 곡선(PCA 기준선)", fontsize=19, fontweight="bold")
    ax.set_xlabel(f"{key} slice index (liver→pubis)", fontsize=16)
    ax.set_ylabel(f"{key} (raw)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.3)

    n_components = pca.n_components_
    ax = axes[1]
    for i in range(n_components):
        pc = i + 1
        ax.plot(slices, pca.components_[i], color=_PC_COLORS[pc], linewidth=2.5,
                 label=f"PC{pc} (explained var={evr[i] * 100:.1f}%)")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title(f"{key} PC1-{n_components} loading curve(누적 explained variance={evr.sum() * 100:.1f}%)",
                 fontsize=19, fontweight="bold")
    ax.set_xlabel(f"{key} slice index (liver→pubis)", fontsize=16)
    ax.set_ylabel("Loading (eigenfunction)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FPCA loading curve plot to {out_path}")


# 사용자 요청(2026-09-10): "그래프에서 구별이 편하도록 색상 재설정". MODEL_ORDER 전체(26개) 순서로 hsv
# 연속 팔레트를 나누면 실제 그래프 1장에는 그 중 한 section(4~9개)만 나오는데, 전체 26개 기준 좁은 구간의
# 색만 배정돼(예: scalar section이 전부 빨강~노랑 계열) 인접 색이 잘 구별되지 않았다. clinic4는 모든
# section에 공통 baseline으로 나오므로 고정 회색으로 통일하고, section별 확장모델은 그 section
# 안에서만 tab10(정성적 팔레트, 색상 간 대비가 큼)을 처음부터 다시 배정 - 각 section의 확장모델 수가
# 3~8개로 tab10 10색 이내라 겹치지 않는다
_CLINIC4_COLOR = np.array([0.35, 0.35, 0.35, 1.0])
# tab10의 8번째 색(회색, index 7)은 clinic4 baseline의 회색과 육안 구별이 거의 안 돼 제외하고 나머지
# 9색만 순환(가장 많은 section인 scalar가 확장모델 8개라 9색이면 안 겹침)
_palette = np.array([c for i, c in enumerate(plt.cm.tab10.colors) if i != 7])
_MODEL_COLORS: dict[str, np.ndarray] = {"clinic4": _CLINIC4_COLOR}
for _section_models in MODEL_SECTIONS.values():
    _others = [m for m in _section_models if m != "clinic4"]
    for _i, _m in enumerate(_others):
        _MODEL_COLORS[_m] = _palette[_i % len(_palette)]
_MODEL_COLORS["bodycomp_only"] = np.array([0.15, 0.55, 0.25, 1.0])


# OR(승산비) 점추정치를 모델별 forest plot으로 시각화. CI는 계산하지 않고 점추정치만 표시(L1/elasticnet 등
# 패널티가 적용된 계수라 일반적인 Wald 표준오차 기반 CI가 통계적으로 부정확해 오해를 줄 수 있음)
def _draw_or_panel(ax, coef_sheets: dict[str, pd.DataFrame], model_name: str, label_fs: float,
                    tick_fs: float, auc_ext: float | None = None) -> None:
    df = coef_sheets[model_name]
    df = df[df["term"] != "intercept"].iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df))
    color = _MODEL_COLORS[model_name]
    ax.scatter(df["odds_ratio"], y_pos, color=color, s=70, zorder=3)
    ax.hlines(y_pos, 1.0, df["odds_ratio"], color=color, linewidth=1.2, alpha=0.6, zorder=2)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["term"], fontsize=tick_fs)
    ax.set_xlabel("Odds Ratio (log)", fontsize=label_fs * 0.85)
    title = MODEL_LABELS[model_name]
    if auc_ext is not None:
        title += f"\nAUC(ext)={auc_ext:.3f}"
    ax.set_title(title, fontsize=label_fs * 0.8, fontweight="bold", color="#161616")
    ax.tick_params(axis="x", labelsize=tick_fs * 0.85)
    ax.grid(alpha=0.3, axis="x")


# 사용자 요청(2026-09-10): "or_forest 그래프 배치를 2줄로". 패널이 한 줄로 길게 이어지면 넓은 화면에서만
# 읽기 편해 2행 grid로 배치(패널 수가 홀수면 남는 칸은 숨김)
def plot_or_forest(feat: str, coef_sheets: dict[str, pd.DataFrame], model_list: list[str], section_title: str,
                    out_path: Path, stats_by_model: dict[str, dict[str, dict]] | None = None) -> None:
    panel_w, panel_h = 6.0, 5.5
    label_fs, tick_fs, _ = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs = label_fs * 1.5, tick_fs * 1.5

    n = len(model_list)
    ncols = -(-n // 2)  # ceil(n/2)
    fig, axes = plt.subplots(2, ncols, figsize=(panel_w * ncols, panel_h * 2))
    flat_axes = np.atleast_2d(axes).reshape(2, ncols).flatten()
    for ax, model_name in zip(flat_axes, model_list):
        auc_ext = stats_by_model[model_name]["external"]["auc"] if stats_by_model else None
        _draw_or_panel(ax, coef_sheets, model_name, label_fs, tick_fs, auc_ext)
    for ax in flat_axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{feat} Odds Ratio (internal full-fit) - {section_title}", fontsize=label_fs, fontweight="bold",
                 color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved OR forest plot to {out_path}")


# 사용자 요청(2026-09-10): "질환별로 모두 합쳐서 top 5를 비교할 수 있게도 해줘". 26개 모델(section 구분
# 없이 전체) 중 clinic4 대비 external ΔAUC가 가장 큰 5개를 질환별로 뽑아 한 그림에 행(질환)x열(top5)로
# 나란히 배치 - 어느 체성분 표현이 어느 질환에서 가장 잘 듣는지 한눈에 비교
def plot_or_forest_top5_by_disease(coef_sheets_by_feat: dict[str, dict[str, pd.DataFrame]],
                                    top5_by_feat: dict[str, list[str]], features: list[str], out_path: Path,
                                    stats_by_model_by_feat: dict[str, dict[str, dict[str, dict]]] | None = None
                                    ) -> None:
    ncols = 1 + max(len(v) for v in top5_by_feat.values())
    nrows = len(features)
    panel_w, panel_h = 5.0, 4.6
    label_fs, tick_fs, _ = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs = label_fs * 1.4, tick_fs * 1.3

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    for row, feat in enumerate(features):
        model_list = ["clinic4"] + top5_by_feat[feat]
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(model_list):
                ax.axis("off")
                continue
            model_name = model_list[col]
            auc_ext = stats_by_model_by_feat[feat][model_name]["external"]["auc"] if stats_by_model_by_feat else None
            _draw_or_panel(ax, coef_sheets_by_feat[feat], model_name, label_fs * 0.75, tick_fs * 0.75, auc_ext)
            if col == 0:
                ax.set_ylabel(feat, fontsize=label_fs, fontweight="bold", color="#161616")
    fig.suptitle("Top 5 body composition representations by external ΔAUC vs clinic4 (odds ratio, internal full-fit)",
                 fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top5 cross-disease OR forest plot to {out_path}")


def plot_roc(feat: str, curves: dict[str, dict[str, np.ndarray]], stats_by_model: dict[str, dict[str, dict]],
             model_list: list[str], section_title: str, out_path: Path) -> None:
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.2

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        n_total, n_pos = len(y), int(np.sum(y))
        for model_name in model_list:
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            auc_str = f"{s['auc']:.3f}±{s['auc_std']:.3f}" if cohort == "internal" else f"{s['auc']:.3f}"
            ax.plot(fpr, tpr, color=_MODEL_COLORS[model_name], linewidth=1.8,
                     label=f"{MODEL_LABELS[model_name]} AUC={auc_str} [{n_total}/{n_pos}]")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("1 - Specificity", fontsize=label_fs)
        ax.set_ylabel("Sensitivity", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.legend(fontsize=legend_fs * 0.7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(section_title, fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# CKD처럼 유병률이 낮은 질환은 ROC-AUC가 낙관적으로 보일 수 있어 PR-AUC(Average Precision)로 보완
def plot_pr_curve(feat: str, curves: dict[str, dict[str, np.ndarray]], model_list: list[str], section_title: str,
                   out_path: Path) -> None:
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.2

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        prevalence = float(y.mean())
        for model_name in model_list:
            score = curves[cohort][model_name]
            precision, recall, _ = precision_recall_curve(y, score)
            if cohort == "internal":
                ap_mean, ap_std = fold_metric_stats(y, score, average_precision_score)
                ap_str = f"{ap_mean:.3f}±{ap_std:.3f}"
            else:
                ap_str = f"{average_precision_score(y, score):.3f}"
            ax.plot(recall, precision, color=_MODEL_COLORS[model_name], linewidth=1.8,
                     label=f"{MODEL_LABELS[model_name]} AP={ap_str}")
        ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1, label=f"Prevalence={prevalence:.3f}")
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=label_fs)
        ax.set_ylabel("Precision (PPV)", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.legend(fontsize=legend_fs * 0.7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(section_title, fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR curve plot to {out_path}")


# 예측확률 vs 관찰빈도(calibration). AUC는 판별력만 보고 확률이 실제로 맞는지는 보지 않아 TRIPOD 체크리스트가
# 요구하는 calibration 검증. quantile bin 방식이라 이벤트 수가 적으면(CKD) bin 수를 줄여 최소 표본을 확보
def plot_calibration(feat: str, curves: dict[str, dict[str, np.ndarray]], model_list: list[str], section_title: str,
                      out_path: Path) -> None:
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.2

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        n_bins = int(np.clip(min(int(y.sum()), int(len(y) - y.sum())), 2, 10))
        for model_name in model_list:
            score = curves[cohort][model_name]
            frac_pos, mean_pred = calibration_curve(y, score, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", markersize=6, color=_MODEL_COLORS[model_name], linewidth=1.8,
                     label=MODEL_LABELS[model_name])
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Mean predicted probability", fontsize=label_fs)
        ax.set_ylabel("Observed frequency", fontsize=label_fs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.legend(fontsize=legend_fs * 0.7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(section_title, fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot to {out_path}")


# Decision Curve Analysis(순이익 net benefit, Vickers & Elkin 2006). AUC delta만으로는 "실제 임상적으로
# 쓸모 있는가"에 약한 답이 될 때 보완하는 분석. net benefit(pt) = TP/n - FP/n * pt/(1-pt)
def _net_benefit(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    net_benefits = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred_pos = score >= pt
        tp = float(np.sum(pred_pos & (y == 1)))
        fp = float(np.sum(pred_pos & (y == 0)))
        net_benefits[i] = tp / n - fp / n * (pt / (1 - pt))
    return net_benefits


def plot_dca(feat: str, curves: dict[str, dict[str, np.ndarray]], model_list: list[str], section_title: str,
             out_path: Path) -> None:
    panel_w, panel_h = 7.5, 7.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.2

    thresholds = np.linspace(0.01, 0.8, 80)
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), gridspec_kw={"wspace": 0.3})
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        prevalence = float(y.mean())
        treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        ax.plot(thresholds, treat_all, color="black", linestyle=":", linewidth=1.3, label="Treat All")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Treat None")
        for model_name in model_list:
            score = curves[cohort][model_name]
            nb = _net_benefit(y, score, thresholds)
            ax.plot(thresholds, nb, color=_MODEL_COLORS[model_name], linewidth=1.8, label=MODEL_LABELS[model_name])
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel("Threshold probability", fontsize=label_fs)
        ax.set_ylabel("Net benefit", fontsize=label_fs)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.05, max(prevalence * 1.3, 0.1))
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=tick_fs)
        ax.legend(fontsize=legend_fs * 0.7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(section_title, fontsize=label_fs, fontweight="bold", color="#161616")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DCA plot to {out_path}")


def plot_summary_table(feat: str, stats_by_model: dict[str, dict[str, dict]], delong_lookup: dict[tuple, dict],
                        model_list: list[str], section_title: str, out_path: Path) -> None:
    rows = []
    for model_name in model_list:
        s_int, s_ext = stats_by_model[model_name]["internal"], stats_by_model[model_name]["external"]
        auc_int = f"{s_int['auc']:.3f}±{s_int['auc_std']:.3f}"
        auc_ext = f"{s_ext['auc']:.3f} ({s_ext['auc_ci_lower']:.3f}-{s_ext['auc_ci_upper']:.3f})"
        if model_name == "clinic4":
            p_str, q_str = "Reference", "Reference"
        else:
            d = delong_lookup.get(("clinic4", model_name), {})
            p, q = d.get("p_value"), d.get("q_value_bh")
            p_str = "-" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
            q_str = "-" if q is None or np.isnan(q) else f"{q:.3f}"
        rows.append([MODEL_LABELS[model_name], auc_int, auc_ext, f"{s_ext['sensitivity']:.3f}",
                     f"{s_ext['specificity']:.3f}", p_str, q_str])
    col_labels = ["Model", "AUC (internal,\nfold mean±std)", "AUC (external, 95% CI)", "Sensitivity\n(external)",
                  "Specificity\n(external)", "DeLong p\nvs clinic4", "BH-FDR q\n(external family)"]
    col_widths = [0.26, 0.11, 0.19, 0.11, 0.11, 0.11, 0.13]
    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.55 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.9)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")
    ax.set_title(f"{feat} Summary - {section_title}", fontsize=17, fontweight="bold", color="#161616", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary table image to {out_path}")


def plot_auc_summary(summary: pd.DataFrame, model_list: list[str], section_title: str, out_path: Path) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.85 / len(model_list)

    panel_w, panel_h = 2 * len(features) + 4, 7
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)
    label_fs, tick_fs, legend_fs = label_fs * 1.5, tick_fs * 1.5, legend_fs * 1.4

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_list):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            offset = (i - (len(model_list) - 1) / 2) * width
            if cohort == "external" and rows["auc_ci_lower"].notna().all():
                yerr = np.vstack([rows["auc"] - rows["auc_ci_lower"], rows["auc_ci_upper"] - rows["auc"]])
                ax.bar(x + offset, rows["auc"], width, yerr=yerr, capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "#161616"},
                       label=MODEL_LABELS[model_name], color=_MODEL_COLORS[model_name])
            elif cohort == "internal" and rows["auc_std"].notna().all():
                ax.bar(x + offset, rows["auc"], width, yerr=rows["auc_std"], capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "#161616"},
                       label=MODEL_LABELS[model_name], color=_MODEL_COLORS[model_name])
            else:
                ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[model_name],
                       color=_MODEL_COLORS[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=tick_fs)
        ax.set_ylim(0.5, 1.0)
        title = f"{cohort} (95% CI)" if cohort == "external" else f"{cohort} (fold mean±std)"
        ax.set_title(title, fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(section_title, fontsize=label_fs, fontweight="bold", color="#161616")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22), fontsize=legend_fs,
               frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


def build_and_save_delta_table(summary: pd.DataFrame, delong: pd.DataFrame, output_dir: Path) -> None:
    pivot = summary.pivot(index=["feature", "cohort"], columns="model", values="auc")
    pivot.columns = [f"auc_{m}" for m in pivot.columns]
    pivot = pivot.reset_index()

    merged = pivot
    for baseline, extended in DELONG_PAIRS:
        d = delong[(delong["baseline_model"] == baseline) & (delong["extended_model"] == extended)][
            ["feature", "cohort", "auc_diff", "z", "p_value", "q_value_bh"]].copy()
        d = d.rename(columns={"auc_diff": f"delta_auc_{extended}", "z": f"delong_z_{extended}",
                               "p_value": f"delong_p_{extended}", "q_value_bh": f"delong_q_bh_{extended}"})
        merged = merged.merge(d, on=["feature", "cohort"], how="left")

    feature_order = list(FEATURES.keys())
    merged["feature"] = pd.Categorical(merged["feature"], categories=feature_order, ordered=True)
    merged["cohort"] = pd.Categorical(merged["cohort"], categories=["internal", "external"], ordered=True)
    merged = merged.sort_values(["feature", "cohort"]).reset_index(drop=True)

    merged.round(4).to_csv(output_dir / "auc_delta_summary.csv", index=False)
    print(f"Saved AUC/delta summary table to {output_dir / 'auc_delta_summary.csv'}")


def plot_incremental_summary_table(feat: str, bodycomp_label: str, stats_by_model: dict[str, dict[str, dict]],
                                    delong_lookup: dict[str, dict], out_path: Path) -> None:
    rows = []
    s_int, s_ext = stats_by_model["bodycomp_only"]["internal"], stats_by_model["bodycomp_only"]["external"]
    rows.append([f"{bodycomp_label} (AEC 없음)", f"{s_int['auc']:.3f}±{s_int['auc_std']:.3f}",
                 f"{s_ext['auc']:.3f} ({s_ext['auc_ci_lower']:.3f}-{s_ext['auc_ci_upper']:.3f})",
                 f"{s_ext['sensitivity']:.3f}", f"{s_ext['specificity']:.3f}", "Reference", "Reference"])
    for aec_arm in AEC_ARM_MODELS:
        s_int, s_ext = stats_by_model[aec_arm]["internal"], stats_by_model[aec_arm]["external"]
        d = delong_lookup.get(aec_arm, {})
        p, q = d.get("p_value"), d.get("q_value_bh")
        p_str = "-" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
        q_str = "-" if q is None or np.isnan(q) else f"{q:.3f}"
        rows.append([f"+ {AEC_ARM_LABELS[aec_arm]}", f"{s_int['auc']:.3f}±{s_int['auc_std']:.3f}",
                     f"{s_ext['auc']:.3f} ({s_ext['auc_ci_lower']:.3f}-{s_ext['auc_ci_upper']:.3f})",
                     f"{s_ext['sensitivity']:.3f}", f"{s_ext['specificity']:.3f}", p_str, q_str])
    col_labels = ["Model", "AUC (internal,\nfold mean±std)", "AUC (external, 95% CI)", "Sensitivity\n(external)",
                  "Specificity\n(external)", "DeLong p\nvs 체성분만", "BH-FDR q\n(9건 family)"]
    col_widths = [0.28, 0.11, 0.19, 0.11, 0.11, 0.1, 0.1]
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.6 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")
    ax.set_title(f"{feat}: 체성분 고정 후 AEC 추가효과", fontsize=17, fontweight="bold", color="#161616", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved incremental summary table image to {out_path}")


def save_aec_incremental_value(summary_rows: list[dict], delong_rows: list[dict], baseline_log: list[dict],
                                valid_features: list[str], output_dir: Path) -> None:
    incr_dir = output_dir / "aec_incremental_value"
    incr_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    delong = pd.DataFrame(delong_rows)
    for cohort in ["internal", "external"]:
        mask = delong["cohort"] == cohort
        delong.loc[mask, "q_value_bh"] = bh_fdr(delong.loc[mask, "p_value"].to_numpy())

    summary.to_csv(incr_dir / "summary.csv", index=False)
    delong.to_csv(incr_dir / "delong.csv", index=False)
    pd.DataFrame(baseline_log).to_csv(incr_dir / "fixed_bodycomp_baseline.csv", index=False)
    print(f"Saved AEC incremental-value results to {incr_dir}")

    baseline_label_by_feat = {row["feature"]: row["fixed_bodycomp_label"] for row in baseline_log}
    for feat in valid_features:
        stats_by_model = {
            m: {c: summary[(summary.feature == feat) & (summary.model == m) & (summary.cohort == c)].iloc[0].to_dict()
                for c in ["internal", "external"]}
            for m in ["bodycomp_only"] + AEC_ARM_MODELS
        }
        delong_lookup = {
            row["aec_arm"]: row
            for row in delong[(delong.feature == feat) & (delong.cohort == "external")].to_dict("records")
        }
        plot_incremental_summary_table(feat, baseline_label_by_feat[feat], stats_by_model, delong_lookup,
                                        incr_dir / f"{FEATURES[feat]}_summary_table.png")

    plot_auc_summary(summary, ["bodycomp_only"] + AEC_ARM_MODELS,
                      "AEC added on top of fixed best body-composition feature (per disease)",
                      incr_dir / "auc_summary.png")


def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_int_raw: dict[str, np.ndarray] = {}
    curve_ext_raw: dict[str, np.ndarray] = {}
    fpca_int_full: dict[str, np.ndarray] = {}
    fpca_ext_full: dict[str, np.ndarray] = {}
    # 사용자 요청(2026-09-10): "aec의 fpca 그래프를 확인했듯이 체성분의 fpca도 확인 가능하게" - AEC 포함
    # 7개 곡선 전부의 PC1-3 loading curve(평균곡선+eigenfunction)를 저장해 FPCA(3) 모델이 각 변수의 곡선
    # 형태에서 실제로 무엇을 포착하는지 시각적으로 검증 가능하게 함
    fpca_loading_dir = output_dir / "fpca_loading"
    fpca_loading_dir.mkdir(parents=True, exist_ok=True)
    elbow_rows, cum_var_by_key, elbow_by_key = [], {}, {}
    for key, (_sheet, prefix) in CURVE_SOURCES.items():
        cols = curve_cols(prefix)
        curve_int_raw[key] = meta_int[cols].astype(float).to_numpy()
        curve_ext_raw[key] = meta_ext[cols].astype(float).to_numpy()

        # 사용자 요청(2026-09-10): "fpca할때 고정하지말고 elbow n 값으로 조절". 모델에 실제로 쓰는
        # n_components는 CURVE_FPCA_N_ELBOW(직전 elbow 검증 결과를 고정 채택한 값)이고, FPCA_N_FIXED(3)는
        # 더 이상 모델에 쓰이지 않는다(과거 고정값과의 비교 참고선으로만 elbow_diagnostics에 남김)
        n_fpca = CURVE_FPCA_N_ELBOW[key]
        pca_full = PCA(n_components=n_fpca, random_state=SEED).fit(curve_int_raw[key])
        fpca_int_full[key] = pca_full.transform(curve_int_raw[key])
        fpca_ext_full[key] = pca_full.transform(curve_ext_raw[key])
        print(f"[FPCA/{key}] n_components={n_fpca}(elbow) explained variance ratio: "
              f"{pca_full.explained_variance_ratio_.round(4)}")
        plot_fpca_loading_curves(key, pca_full, fpca_loading_dir / f"{key.lower()}.png")

        elbow_n, cum_var = select_fpca_n_by_elbow(curve_int_raw[key])
        cum_var_by_key[key], elbow_by_key[key] = cum_var, elbow_n
        if elbow_n != n_fpca:
            print(f"[FPCA/{key}] 경고: 재계산된 elbow={elbow_n}이 고정 채택값 CURVE_FPCA_N_ELBOW={n_fpca}과 "
                  f"다름 - 데이터가 바뀌었을 수 있으니 상수 갱신 검토 필요")
        match = "일치" if elbow_n == FPCA_N_FIXED else "불일치"
        print(f"[FPCA/{key}] elbow(Kneedle) n_components={elbow_n} (구 고정값 {FPCA_N_FIXED}과 {match}), "
              f"고정값 기준 누적분산={cum_var[FPCA_N_FIXED]:.4f}, elbow 기준 누적분산={cum_var[elbow_n]:.4f}")
        elbow_rows.append({"curve": key, "elbow_n": elbow_n, "fixed_n": FPCA_N_FIXED,
                            "cum_var_at_fixed_n": round(float(cum_var[FPCA_N_FIXED]), 4),
                            "cum_var_at_elbow_n": round(float(cum_var[elbow_n]), 4),
                            "matches_fixed": elbow_n == FPCA_N_FIXED})
    pd.DataFrame(elbow_rows).to_csv(fpca_loading_dir / "elbow_diagnostics.csv", index=False)
    print(f"Saved elbow diagnostics table to {fpca_loading_dir / 'elbow_diagnostics.csv'}")
    plot_elbow_diagnostics(cum_var_by_key, elbow_by_key, fpca_loading_dir / "elbow_diagnostics.png")

    valid_features: list[str] = []
    skipped = []
    for feat in FEATURES:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int, mask_val_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        n_pos_int = int(y_int_all[mask_val_int].sum())
        n_neg_int = int(mask_val_int.sum() - n_pos_int)
        n_pos_ext = int(y_ext_all[mask_val_ext].sum())
        n_neg_ext = int(mask_val_ext.sum() - n_pos_ext)
        print(f"[{feat}] internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만"
            print(msg)
            skipped.append({"feature": feat, "reason": msg})
            continue
        valid_features.append(feat)

    summary_rows, delong_rows, vif_rows = [], [], []
    coef_sheets_by_feat: dict[str, dict[str, pd.DataFrame]] = {}
    curves_by_feat: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    # 사용자 요청(2026-09-10): "체성분 데이터까지를 고정으로하고 aec의 유무에 대한 비교" - 별도 family로
    # 취급(메인 96건 family와 목적이 다름: "AEC vs 체성분"이 아니라 "체성분 고정 후 AEC 추가효과")
    aec_incr_summary_rows, aec_incr_delong_rows, aec_incr_baseline_log = [], [], []

    for feat in valid_features:
        slug = FEATURES[feat]
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_int, mask_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int].astype(int), y_ext_all[mask_ext].astype(int)

        meta_int_m = meta_int.loc[mask_int].reset_index(drop=True)
        meta_ext_m = meta_ext.loc[mask_ext].reset_index(drop=True)
        curve_int_m = {key: arr[mask_int] for key, arr in curve_int_raw.items()}
        fpca_int_m = {key: arr[mask_int] for key, arr in fpca_int_full.items()}
        fpca_ext_m = {key: arr[mask_ext] for key, arr in fpca_ext_full.items()}

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        stats_by_model: dict[str, dict[str, dict]] = {m: {} for m in MODEL_ORDER}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODEL_ORDER}
        coef_sheets: dict[str, pd.DataFrame] = {}

        for model_name in MODEL_ORDER:
            extra_cols = MODEL_EXTRA_COLS[model_name]
            curve_key = MODEL_CURVE_SOURCE[model_name]

            if curve_key is not None:
                x_int_full, scaler, curve_scaler = build_matrix(meta_int_m, [], fpca_int_m[curve_key])
                x_ext_full, _, _ = build_matrix(meta_ext_m, [], fpca_ext_m[curve_key], scaler, curve_scaler)
                oof_proba = fpca_oof_proba(meta_int_m, curve_int_m[curve_key], y_int, CURVE_FPCA_N_ELBOW[curve_key],
                                            cv)
            else:
                x_int_full, scaler, _ = build_matrix(meta_int_m, extra_cols)
                x_ext_full, _, _ = build_matrix(meta_ext_m, extra_cols, scaler=scaler)
                oof_proba = cross_val_predict(LogisticRegression(**LOGREG_PARAMS), x_int_full, y_int, cv=cv,
                                               method="predict_proba")[:, 1]

            input_cols = input_cols_for(model_name)
            added_cols = input_cols[4:]  # sex_M/age/height/weight 다음이 이 모델의 추가 변수(들)
            for cohort, x_full in (("internal", x_int_full), ("external", x_ext_full)):
                vif_model_df = compute_vif(x_full, input_cols, added_cols)
                vif_model_df.insert(0, "cohort", cohort)
                vif_model_df.insert(0, "model", model_name)
                vif_model_df.insert(0, "feature", feat)
                vif_rows.extend(vif_model_df.to_dict("records"))

            model = LogisticRegression(**LOGREG_PARAMS).fit(x_int_full, y_int)
            ext_proba = model.predict_proba(x_ext_full)[:, 1]

            # 사용자 요청(2026-09-18): "baseline과 다른 모델의 성능을 비교하려면 threshold를 고정시키는게
            # 맞지 않나" -> 모델마다 자기 OOF에서 따로 Youden threshold를 구하면 confusion matrix 차이가
            # "모델이 더 잘 예측해서"인지 "그냥 다른 지점에서 잘랐기 때문"인지 구분이 안 됨. MODEL_ORDER의
            # 첫 모델(clinic4)에서 구한 threshold를 그 feat의 모든 모델에 고정 적용(code/0916과 동일 원칙)
            if model_name == "clinic4":
                baseline_threshold = youden_threshold(y_int, oof_proba)
            threshold = baseline_threshold

            for cohort, y, score in (("internal", y_int, oof_proba), ("external", y_ext, ext_proba)):
                ci_lo, ci_hi = bootstrap_auc_ci(y, score) if cohort == "external" else (float("nan"), float("nan"))
                if cohort == "internal":
                    auc, auc_std = fold_metric_stats(y, score, roc_auc_score)
                    cls_stats = fold_classification_stats(y, score, threshold)
                else:
                    auc, auc_std = float(roc_auc_score(y, score)), float("nan")
                    cls_stats = {**classification_stats(y, score, threshold),
                                 "sensitivity_std": float("nan"), "specificity_std": float("nan"),
                                 "accuracy_std": float("nan")}
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "auc": auc,
                     "auc_std": auc_std, "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi, "threshold": threshold,
                     **cls_stats}
                ci_str = (f" 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]" if cohort == "external"
                          else f" fold_std={auc_std:.3f}")
                print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
                      f"AUC={s['auc']:.3f}{ci_str} Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} "
                      f"Acc={s['accuracy']:.3f}")
                summary_rows.append({"feature": feat, "model": model_name, "cohort": cohort, **s})
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

            coef_df = pd.DataFrame({"term": input_cols + ["intercept"],
                                     "coefficient": np.concatenate([model.coef_.ravel(),
                                                                     np.atleast_1d(model.intercept_)])})
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

        for baseline, extended in DELONG_PAIRS:
            for cohort, y in (("internal", y_int), ("external", y_ext)):
                d = delong_paired_auc_test(y, scores_by_model[baseline][cohort], scores_by_model[extended][cohort])
                print(f"[{feat} / {cohort}] DeLong {baseline} vs {extended}: AUC diff={d['diff']:+.4f} "
                      f"z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "baseline_model": baseline,
                                     "extended_model": extended, "auc_baseline": d["auc_a"],
                                     "auc_extended": d["auc_b"], "auc_diff": d["diff"], "z": d["z"],
                                     "p_value": d["p_value"]})

        # 사용자 요청(2026-09-10): "체성분 데이터까지를 고정으로하고 aec의 유무에 대한 비교도 진행". 이
        # feat에서 이미 계산된 delong_rows(clinic4 대비 external ΔAUC)로부터 FPCA가 아닌 모델 중 가장 개선폭이
        # 큰 것을 "고정 체성분 baseline"으로 채택하고, 그 위에 AEC 3표현을 각각 추가했을 때 추가 개선이
        # 있는지를 본다
        feat_ext_rows = [r for r in delong_rows if r["feature"] == feat and r["cohort"] == "external"
                          and r["extended_model"] in NONAEC_NONFPCA_MODELS]
        bodycomp_model = max(feat_ext_rows, key=lambda r: r["auc_diff"])["extended_model"]
        bodycomp_extra_cols = MODEL_EXTRA_COLS[bodycomp_model]
        print(f"[{feat}] AEC 추가효과 분석의 고정 체성분 baseline = {MODEL_LABELS[bodycomp_model]}")
        aec_incr_baseline_log.append({"feature": feat, "fixed_bodycomp_model": bodycomp_model,
                                       "fixed_bodycomp_label": MODEL_LABELS[bodycomp_model]})

        incr_scores: dict[str, dict[str, np.ndarray]] = {}
        incr_stats: dict[str, dict[str, dict]] = {}
        # "bodycomp_only" = clinic4 + 고정 체성분(FPCA 아님, 항상 plain extra_cols)만 - cross_val_predict로 충분
        x_int_full, scaler, _ = build_matrix(meta_int_m, bodycomp_extra_cols)
        x_ext_full, _, _ = build_matrix(meta_ext_m, bodycomp_extra_cols, scaler=scaler)
        oof_proba = cross_val_predict(LogisticRegression(**LOGREG_PARAMS), x_int_full, y_int, cv=cv,
                                       method="predict_proba")[:, 1]
        model = LogisticRegression(**LOGREG_PARAMS).fit(x_int_full, y_int)
        ext_proba = model.predict_proba(x_ext_full)[:, 1]
        incr_scores["bodycomp_only"] = {"internal": oof_proba, "external": ext_proba}

        for aec_arm in AEC_ARM_MODELS:
            arm_extra_cols = bodycomp_extra_cols + MODEL_EXTRA_COLS[aec_arm]
            arm_curve_key = MODEL_CURVE_SOURCE[aec_arm]  # "AEC"면 FPCA, 아니면 None(uplow_ratio/slope는 plain)
            if arm_curve_key is not None:
                n_fpca = CURVE_FPCA_N_ELBOW[arm_curve_key]
                x_int_full, scaler, curve_scaler = build_matrix(meta_int_m, bodycomp_extra_cols,
                                                                  fpca_int_m[arm_curve_key])
                x_ext_full, _, _ = build_matrix(meta_ext_m, bodycomp_extra_cols, fpca_ext_m[arm_curve_key],
                                                 scaler, curve_scaler)
                oof_proba = fpca_oof_proba(meta_int_m, curve_int_m[arm_curve_key], y_int, n_fpca, cv,
                                            extra_cols=bodycomp_extra_cols)
            else:
                x_int_full, scaler, _ = build_matrix(meta_int_m, arm_extra_cols)
                x_ext_full, _, _ = build_matrix(meta_ext_m, arm_extra_cols, scaler=scaler)
                oof_proba = cross_val_predict(LogisticRegression(**LOGREG_PARAMS), x_int_full, y_int, cv=cv,
                                               method="predict_proba")[:, 1]
            model = LogisticRegression(**LOGREG_PARAMS).fit(x_int_full, y_int)
            ext_proba = model.predict_proba(x_ext_full)[:, 1]
            incr_scores[aec_arm] = {"internal": oof_proba, "external": ext_proba}

        for arm_name, arm_scores in incr_scores.items():
            threshold = youden_threshold(y_int, arm_scores["internal"])
            incr_stats[arm_name] = {}
            for cohort, y, score in (("internal", y_int, arm_scores["internal"]),
                                      ("external", y_ext, arm_scores["external"])):
                ci_lo, ci_hi = bootstrap_auc_ci(y, score) if cohort == "external" else (float("nan"), float("nan"))
                if cohort == "internal":
                    auc, auc_std = fold_metric_stats(y, score, roc_auc_score)
                    cls_stats = fold_classification_stats(y, score, threshold)
                else:
                    auc, auc_std = float(roc_auc_score(y, score)), float("nan")
                    cls_stats = {**classification_stats(y, score, threshold),
                                 "sensitivity_std": float("nan"), "specificity_std": float("nan"),
                                 "accuracy_std": float("nan")}
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "auc": auc,
                     "auc_std": auc_std, "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi, "threshold": threshold,
                     **cls_stats}
                incr_stats[arm_name][cohort] = s
                aec_incr_summary_rows.append({"feature": feat, "model": arm_name, "cohort": cohort, **s})

        for aec_arm in AEC_ARM_MODELS:
            for cohort, y in (("internal", y_int), ("external", y_ext)):
                d = delong_paired_auc_test(y, incr_scores["bodycomp_only"][cohort], incr_scores[aec_arm][cohort])
                print(f"[{feat} / {cohort}] AEC 추가효과 DeLong bodycomp_only vs {aec_arm}: "
                      f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
                aec_incr_delong_rows.append({"feature": feat, "cohort": cohort, "fixed_bodycomp_model": bodycomp_model,
                                              "aec_arm": aec_arm, "auc_bodycomp_only": d["auc_a"],
                                              "auc_bodycomp_plus_aec": d["auc_b"], "auc_diff": d["diff"],
                                              "z": d["z"], "p_value": d["p_value"]})

        # 사용자 요청(2026-09-10): "output 세부 폴더로 다 나눠서 저장". 질환별로 output_dir/{slug}/ 폴더를
        # 만들고, 그 안에 다시 section별 하위 폴더(scalar/inter_var_ratio/curve_fpca_elbow/curve_uplow_ratio)를
        # 둬서 파일을 분산 저장(3질환 x 4섹션 x 6종 그래프 = 72개 이미지가 한 폴더에 뒤섞이는 것을 방지)
        coef_sheets_by_feat[feat] = coef_sheets

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)
        # BH-FDR q값이 필요 없는 산출물(계수/OR forest/ROC/PR/calibration/DCA)은 q값 계산(전체 feature 완료
        # 후)을 기다릴 필요가 없어 이 시점에 바로 section별로 저장
        write_sheets(feat_dir / "logistic_coefficients.xlsx", coef_sheets)
        curves = {
            "internal": {"y": y_int, **{m: scores_by_model[m]["internal"] for m in MODEL_ORDER}},
            "external": {"y": y_ext, **{m: scores_by_model[m]["external"] for m in MODEL_ORDER}},
        }
        curves_by_feat[feat] = curves
        for section_name, model_list in MODEL_SECTIONS.items():
            section_title = SECTION_TITLES[section_name]
            section_dir = feat_dir / section_name
            section_dir.mkdir(parents=True, exist_ok=True)
            plot_or_forest(feat, coef_sheets, model_list, section_title, section_dir / "or_forest.png")
            plot_roc(feat, curves, stats_by_model, model_list, section_title, section_dir / "roc_curve.png")
            plot_pr_curve(feat, curves, model_list, section_title, section_dir / "pr_curve.png")
            plot_calibration(feat, curves, model_list, section_title, section_dir / "calibration.png")
            plot_dca(feat, curves, model_list, section_title, section_dir / "dca.png")

        # 사용자 요청(2026-09-16): "이미지 저장도 해줘. 그러면 code/0916/vif_check.py가 필요없어지겠네" -
        # VIF 수치(vif_by_model.csv)뿐 아니라 그래프도 이 스크립트 실행만으로 바로 생성
        vif_dir = feat_dir / "vif"
        vif_dir.mkdir(parents=True, exist_ok=True)
        plot_vif_scalar_combined(feat, meta_int_m, meta_ext_m, vif_dir / "scalar_combined.png")
        vif_df_feat = pd.DataFrame([r for r in vif_rows if r["feature"] == feat])
        for section_name, model_list in MODEL_SECTIONS.items():
            plot_vif_per_model(vif_df_feat, model_list, f"{feat}: VIF — {SECTION_TITLES[section_name]}",
                                vif_dir / f"per_model_{section_name}.png")
        plot_vif_per_model(vif_df_feat, MODEL_ORDER, f"{feat}: VIF — 32개 확장모델 전체",
                            vif_dir / "per_model_all.png", color_by_section=True)

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    delong_df = pd.DataFrame(delong_rows)
    # BH-FDR: cohort(internal/external)별로 "체성분 스칼라 8종 + 변수간 비율 3종 + 곡선 7종 x
    # (FPCA3/uplow_ratio)"를 하나의 검정군으로 묶어 보정. AEC 포함 여부와 무관하게 clinic4 대비 모든 확장모델
    # 비교가 동일 목적(단일 변수/표현 추가 효과 탐색)이라 같은 family로 취급
    for cohort in ["internal", "external"]:
        mask = delong_df["cohort"] == cohort
        delong_df.loc[mask, "q_value_bh"] = bh_fdr(delong_df.loc[mask, "p_value"].to_numpy())

    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    # 사용자 요청(2026-09-16): "실행할때 모델별로 VIF도 확인할 수 있게" - 모델(clinic4 + 변수 1개)별
    # 다중공선성을 확인. is_added=True 행이 그 모델에서 새로 추가한 변수(들)의 VIF
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(output_dir / "vif_by_model.csv", index=False)
    print(f"Saved VIF by model to {output_dir / 'vif_by_model.csv'}")
    high_vif = vif_df[vif_df["is_added"] & (vif_df["vif"] >= 5)]
    if len(high_vif):
        print(f"[VIF 경고] 추가 변수의 VIF>=5인 모델 {len(high_vif)}건:")
        print(high_vif[["feature", "cohort", "model", "variable", "vif"]].to_string(index=False))

    # 사용자 요청(2026-09-10): "질환별로 모두 합쳐서 top 5를 비교". section 구분 없이 26개 모델 전체에서
    # clinic4 대비 external ΔAUC가 가장 큰 5개를 질환별로 선정
    top5_by_feat: dict[str, list[str]] = {}
    for feat in valid_features:
        ext = delong_df[(delong_df.feature == feat) & (delong_df.cohort == "external")]
        top5_by_feat[feat] = ext.sort_values("auc_diff", ascending=False)["extended_model"].head(5).tolist()
    stats_by_model_by_feat: dict[str, dict[str, dict[str, dict]]] = {}

    # q값이 확정된 뒤에야 그릴 수 있는 요약표(및 이를 포함한 top5 전체 산출물)를 2차 루프에서 section별로
    # 저장. 사용자 요청(2026-09-10): "top5는 or_forest 뿐만아니라 auc등 모든 것으로 저장" - top5도 다른
    # section과 동일하게 summary_table/roc/pr/calibration/dca를 전부 생성
    for feat in valid_features:
        slug = FEATURES[feat]
        stats_by_model = {
            m: {c: summary[(summary.feature == feat) & (summary.model == m) & (summary.cohort == c)].iloc[0].to_dict()
                for c in ["internal", "external"]}
            for m in MODEL_ORDER
        }
        stats_by_model_by_feat[feat] = stats_by_model
        delong_lookup_feat = {
            (row["baseline_model"], row["extended_model"]): row
            for row in delong_df[(delong_df.feature == feat) & (delong_df.cohort == "external")].to_dict("records")
        }
        feat_dir = output_dir / slug
        for section_name, model_list in MODEL_SECTIONS.items():
            section_dir = feat_dir / section_name
            section_dir.mkdir(parents=True, exist_ok=True)
            plot_summary_table(feat, stats_by_model, delong_lookup_feat, model_list, SECTION_TITLES[section_name],
                                section_dir / "summary_table.png")

        top5_title = "Top 5 by external ΔAUC vs clinic4"
        top5_model_list = ["clinic4"] + top5_by_feat[feat]
        top5_dir = feat_dir / "top5"
        top5_dir.mkdir(parents=True, exist_ok=True)
        coef_sheets = coef_sheets_by_feat[feat]
        curves = curves_by_feat[feat]
        plot_or_forest(feat, coef_sheets, top5_model_list, top5_title, top5_dir / "or_forest.png", stats_by_model)
        plot_roc(feat, curves, stats_by_model, top5_model_list, top5_title, top5_dir / "roc_curve.png")
        plot_pr_curve(feat, curves, top5_model_list, top5_title, top5_dir / "pr_curve.png")
        plot_calibration(feat, curves, top5_model_list, top5_title, top5_dir / "calibration.png")
        plot_dca(feat, curves, top5_model_list, top5_title, top5_dir / "dca.png")
        plot_summary_table(feat, stats_by_model, delong_lookup_feat, top5_model_list, top5_title,
                            top5_dir / "summary_table.png")

    plot_or_forest_top5_by_disease(coef_sheets_by_feat, top5_by_feat, valid_features,
                                    output_dir / "top5_comparison_or_forest.png", stats_by_model_by_feat)

    # top5_comparison_or_forest.png에 그려지는 질환별 clinic4+top5 모델의 변수별 coefficient/OR을 함께 저장
    top5_coef_rows = []
    for feat in valid_features:
        for model_name in ["clinic4"] + top5_by_feat[feat]:
            df = coef_sheets_by_feat[feat][model_name].copy()
            df.insert(0, "model", MODEL_LABELS[model_name])
            df.insert(0, "feature", feat)
            top5_coef_rows.append(df)
    top5_coef_path = output_dir / "top5_comparison_coefficients.xlsx"
    pd.concat(top5_coef_rows, ignore_index=True).to_excel(top5_coef_path, index=False)
    print(f"Saved top5 comparison coefficients to {top5_coef_path}")

    auc_summary_dir = output_dir / "auc_summary"
    auc_summary_dir.mkdir(parents=True, exist_ok=True)
    for section_name, model_list in MODEL_SECTIONS.items():
        plot_auc_summary(summary, model_list, SECTION_TITLES[section_name],
                          auc_summary_dir / f"{section_name}.png")
    build_and_save_delta_table(summary, delong_df, output_dir)

    save_aec_incremental_value(aec_incr_summary_rows, aec_incr_delong_rows, aec_incr_baseline_log,
                                valid_features, output_dir)


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    required_cols = CLINICAL_BASE_COLS + BODY_COMP_COLS

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_int).sum()}/{len(mask_int)}, "
          f"external {(~mask_ext).sum()}/{len(mask_ext)}")
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Final cohort: internal n={len(meta_int)}, external n={len(meta_ext)}")

    run(meta_int, meta_ext, OUTPUT_DIR)


if __name__ == "__main__":
    main()
