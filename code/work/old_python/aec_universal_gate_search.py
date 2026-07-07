from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_residual_phenotype_discordance import add_shape_features  # noqa: E402
from aec_conditional_value import DATA_DIR, matrix_from_sheet, resample_rows, row_norm  # noqa: E402
from aec_offset_score import clinical_raw  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_universal_gate_search"
SEED = 20260629


def sigmoid(x: np.ndarray) -> np.ndarray:
    """로짓 값을 0~1 확률로 변환 (오버플로 방지를 위해 -40~40으로 클리핑)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fill_raw_aec128(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """엑셀에서 원시 aec_128 행렬과 그 행 정규화 버전, 저근감소증 라벨·성별을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    aec128 = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")
    raw = matrix_from_sheet(aec128)
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / (height_m**2)
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return meta, raw, norm, y, sex


def load_legacy_aec256(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """레거시 방식대로 aec_128과 aec_cropped를 각각 리샘플·정규화해 256차원으로 이어붙인 결합 곡선을 만듦."""
    a128 = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")
    crop = pd.read_excel(path, sheet_name="aec_cropped", engine="openpyxl")
    a128_mat = resample_rows(matrix_from_sheet(a128), 128)
    crop_mat = resample_rows(matrix_from_sheet(crop), 128)
    x = np.column_stack([row_norm(a128_mat) - 1.0, row_norm(crop_mat) - 1.0])
    return a128, x


def clinical_model() -> Pipeline:
    """임상 변수 전용 로지스틱 회귀 파이프라인(정규화 거의 없음)을 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def aec_svm_model(n_features: int, c: float = 0.2, k: int = 128) -> Pipeline:
    """결측대체→표준화→상위 k개 특징 선택→선형 SVM으로 이어지는 AEC 전용 분류 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(k, n_features))),
            ("svm", LinearSVC(C=c, class_weight="balanced", max_iter=20000, random_state=SEED)),
        ]
    )


def score_model(model, x: np.ndarray) -> np.ndarray:
    """모델에 decision_function이 있으면 그 값을, 없으면 양성 클래스 확률을 점수로 반환."""
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def crossfit_external_estimator(model_factory, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5-fold 교차검증으로 train 전체의 out-of-fold 점수를 만들고, 전체 train으로 재학습한 모델로 외부 점수도 함께 계산."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr), dtype=float)
    for fold, (tr, va) in enumerate(skf.split(xtr, ytr), start=1):
        model = model_factory(fold)
        model.fit(xtr[tr], ytr[tr])
        oof[va] = score_model(model, xtr[va])
    final = model_factory(99)
    final.fit(xtr, ytr)
    return oof, score_model(final, xte)


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """목표 민감도(target) 이상을 유지하면서 특이도가 가장 높은 임계값을 찾음 (해당하는 값이 없으면 분위수로 근사)."""
    best = None
    for th in np.unique(score):
        pred = score >= th
        tp = np.sum(pred & (y == 1))
        fn = np.sum(~pred & (y == 1))
        tn = np.sum(~pred & (y == 0))
        fp = np.sum(pred & (y == 0))
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens >= target and (best is None or spec > best[1]):
            best = (float(th), spec)
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return float(best[0])


def threshold_youden(y: np.ndarray, score: np.ndarray) -> float:
    """Youden index(민감도+특이도-1)를 최대화하는 임계값을 후보 점수들 중에서 탐색."""
    best_th = float(np.min(score))
    best_j = -np.inf
    for th in np.unique(score):
        pred = score >= th
        tp = np.sum(pred & (y == 1))
        fn = np.sum(~pred & (y == 1))
        tn = np.sum(~pred & (y == 0))
        fp = np.sum(pred & (y == 0))
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_th = float(th)
    return best_th


def binary_counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측(pred)과 실제 라벨로 TP/FP/FN/TN과 민감도·특이도·PPV·NPV를 계산."""
    pred = pred.astype(bool)
    yb = y.astype(bool)
    tp = int(np.sum(pred & yb))
    fp = int(np.sum(pred & ~yb))
    fn = int(np.sum(~pred & yb))
    tn = int(np.sum(~pred & ~yb))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def enrich_stats(y: np.ndarray, clinical_pos: np.ndarray, aec_keep: np.ndarray) -> dict:
    """임상 양성군을 AEC로 유지/하향조정한 두 그룹의 표본수·이벤트·유병률과, Fisher 정확검정으로 오즈비·p값을 계산."""
    cp = clinical_pos.astype(bool)
    keep = aec_keep.astype(bool)
    cp_keep = cp & keep
    cp_deesc = cp & ~keep
    a = int(np.sum(y[cp_keep] == 1))
    b = int(np.sum(y[cp_keep] == 0))
    c = int(np.sum(y[cp_deesc] == 1))
    d = int(np.sum(y[cp_deesc] == 0))
    if (a + b) == 0 or (c + d) == 0:
        odds_ratio = np.nan
        p = np.nan
    else:
        odds_ratio, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return {
        "clinical_pos_aec_keep_n": int(np.sum(cp_keep)),
        "clinical_pos_aec_keep_events": a,
        "clinical_pos_aec_keep_prevalence": a / (a + b) if a + b else np.nan,
        "clinical_pos_aec_deesc_n": int(np.sum(cp_deesc)),
        "clinical_pos_aec_deesc_events": c,
        "clinical_pos_aec_deesc_prevalence": c / (c + d) if c + d else np.nan,
        "within_clinical_pos_or_keep_vs_deesc": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan,
        "within_clinical_pos_fisher_p": float(p) if np.isfinite(p) else np.nan,
    }


def rule_metrics(y: np.ndarray, clinical_score: np.ndarray, clinical_th: float, aec_risk_score: np.ndarray, aec_th: float) -> dict:
    """임상 임계값과 AEC 임계값으로 정의된 "게이트 규칙"을 적용해, 임상 단독 대비 민감도손실/특이도이득/PPV이득과 하향조정군 통계까지 모두 계산."""
    clinical_pos = clinical_score >= clinical_th
    aec_keep = aec_risk_score >= aec_th
    final_pos = clinical_pos & aec_keep
    base = binary_counts(y, clinical_pos)
    rule = binary_counts(y, final_pos)
    enrich = enrich_stats(y, clinical_pos, aec_keep)
    deesc = clinical_pos & ~aec_keep
    return {
        "clinical_positive_n": int(np.sum(clinical_pos)),
        "clinical_positive_events": int(np.sum(y[clinical_pos] == 1)),
        "clinical_positive_prevalence": float(np.mean(y[clinical_pos])) if np.sum(clinical_pos) else np.nan,
        **{f"clinical_{k}": v for k, v in base.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "specificity_gain": rule["specificity"] - base["specificity"],
        "ppv_gain": rule["ppv"] - base["ppv"],
        "false_positives_removed": int(np.sum(deesc & (y == 0))),
        "true_positives_lost": int(np.sum(deesc & (y == 1))),
        **enrich,
    }


def auc_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """점수로부터 AUC/AP/로그손실/Brier를 계산."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def build_aec_scores(meta_g, raw_g, norm_g, y_g, meta_s, raw_s, norm_s) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """정규화/원시 곡선의 모양 특징 여러 개, 레거시 256차원 AEC SVM 점수, 그리고 핵심 특징들의 평균 z점수까지
    총 10개의 "AEC 단독 위험 점수" 후보를 만들고, 각각의 방향을 g1090에서만 고정한 뒤 메타데이터와 함께 반환."""
    norm_feat_g = add_shape_features(norm_g)
    norm_feat_s = add_shape_features(norm_s)
    raw_feat_g = add_shape_features(raw_g)
    raw_feat_s = add_shape_features(raw_s)
    raw_mean_g = raw_g.mean(axis=1)
    raw_mean_s = raw_s.mean(axis=1)

    g = pd.DataFrame(index=np.arange(len(y_g)))
    s = pd.DataFrame(index=np.arange(len(raw_s)))

    # Higher score should mean higher low-SMI risk. Direction is fixed on g1090 only.
    candidates = {
        "norm_tail_rebound": (
            norm_feat_g["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float),
            norm_feat_s["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float),
        ),
        "norm_late_max_minus_trough": (
            norm_feat_g["late_max_minus_trough_min"].to_numpy(dtype=float),
            norm_feat_s["late_max_minus_trough_min"].to_numpy(dtype=float),
        ),
        "norm_recovery_slope": (
            norm_feat_g["recovery_slope_81_113"].to_numpy(dtype=float),
            norm_feat_s["recovery_slope_81_113"].to_numpy(dtype=float),
        ),
        "norm_tail_area_above1": (
            norm_feat_g["tail_area_above_1_114_128"].to_numpy(dtype=float),
            norm_feat_s["tail_area_above_1_114_128"].to_numpy(dtype=float),
        ),
        "norm_trough_area_below1": (
            norm_feat_g["trough_area_below_1_60_95"].to_numpy(dtype=float),
            norm_feat_s["trough_area_below_1_60_95"].to_numpy(dtype=float),
        ),
        "norm_tail_minus_trough_mean": (
            norm_feat_g["tail_minus_trough_mean"].to_numpy(dtype=float),
            norm_feat_s["tail_minus_trough_mean"].to_numpy(dtype=float),
        ),
        "raw_tail_rebound": (
            raw_feat_g["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float),
            raw_feat_s["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float),
        ),
        "raw_tail_minus_trough_mean": (
            raw_feat_g["tail_minus_trough_mean"].to_numpy(dtype=float),
            raw_feat_s["tail_minus_trough_mean"].to_numpy(dtype=float),
        ),
        "raw_tail_rebound_over_mean": (
            raw_feat_g["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float) / raw_mean_g,
            raw_feat_s["tail_rebound_114_128_max_minus_60_95_min"].to_numpy(dtype=float) / raw_mean_s,
        ),
    }
    score_meta = {}
    for name, (vg, vs) in candidates.items():
        r = np.corrcoef(vg, y_g)[0, 1]
        sign = 1.0 if r >= 0 else -1.0
        g[name] = sign * vg
        s[name] = sign * vs
        score_meta[name] = {
            "source": "interpretable_aec128_raw_or_normalized",
            "direction_fixed_by_g1090_low_smi_correlation": sign,
            "g1090_auc": float(roc_auc_score(y_g, sign * vg)),
        }

    x256_g = np.column_stack([norm_g - 1.0, row_norm(raw_g)[:, :0]])  # placeholder-free shape guard
    # Use legacy aec_128 + aec_cropped normalized score when available.
    _, xlegacy_g = load_legacy_aec256(DATA_DIR / "g1090.xlsx")
    _, xlegacy_s = load_legacy_aec256(DATA_DIR / "sdata.xlsx")
    aec_oof, aec_ext = crossfit_external_estimator(
        lambda seed: aec_svm_model(xlegacy_g.shape[1], c=0.2, k=128),
        xlegacy_g,
        y_g,
        xlegacy_s,
    )
    if roc_auc_score(y_g, aec_oof) < 0.5:
        aec_oof = -aec_oof
        aec_ext = -aec_ext
        sign = -1.0
    else:
        sign = 1.0
    g["legacy_aec256_svm_norm_only"] = aec_oof
    s["legacy_aec256_svm_norm_only"] = aec_ext
    score_meta["legacy_aec256_svm_norm_only"] = {
        "source": "AEC-only LinearSVM on normalized aec_128 and aec_cropped, no sex, no clinical boundary",
        "direction_fixed_by_g1090_oof_auc": sign,
        "g1090_auc": float(roc_auc_score(y_g, aec_oof)),
    }

    # Small universal score: average z of the strongest interpretable morphology candidates, direction fixed by g1090.
    core = ["norm_tail_rebound", "norm_late_max_minus_trough", "norm_tail_area_above1"]
    z_g = []
    z_s = []
    for name in core:
        mu = float(g[name].mean())
        sd = float(g[name].std(ddof=1)) or 1.0
        z_g.append((g[name].to_numpy(dtype=float) - mu) / sd)
        z_s.append((s[name].to_numpy(dtype=float) - mu) / sd)
    g["universal_norm_tail_core_mean_z"] = np.mean(np.column_stack(z_g), axis=1)
    s["universal_norm_tail_core_mean_z"] = np.mean(np.column_stack(z_s), axis=1)
    if roc_auc_score(y_g, g["universal_norm_tail_core_mean_z"]) < 0.5:
        g["universal_norm_tail_core_mean_z"] *= -1
        s["universal_norm_tail_core_mean_z"] *= -1
        sign = -1.0
    else:
        sign = 1.0
    score_meta["universal_norm_tail_core_mean_z"] = {
        "source": "Mean z-score of normalized tail rebound, late max-minus-trough, tail area above 1; no model fitting",
        "direction_fixed_by_g1090_auc": sign,
        "g1090_auc": float(roc_auc_score(y_g, g["universal_norm_tail_core_mean_z"])),
    }
    return g, s, score_meta


def bootstrap_external_delta(
    y: np.ndarray,
    clinical_score: np.ndarray,
    clinical_th: float,
    aec_score: np.ndarray,
    aec_th: float,
    n_boot: int = 3000,
) -> dict:
    """게이트 규칙의 민감도손실/특이도이득/PPV이득/하향조정군 유병률/오즈비를 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 123)
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        m = rule_metrics(yy, clinical_score[idx], clinical_th, aec_score[idx], aec_th)
        rows.append(
            [
                m["sensitivity_loss"],
                m["specificity_gain"],
                m["ppv_gain"],
                m["false_positives_removed"],
                m["true_positives_lost"],
                m["clinical_pos_aec_deesc_prevalence"],
                m["within_clinical_pos_or_keep_vs_deesc"],
            ]
        )
    arr = np.asarray(rows)
    names = [
        "sensitivity_loss",
        "specificity_gain",
        "ppv_gain",
        "false_positives_removed",
        "true_positives_lost",
        "deesc_prevalence",
        "or_keep_vs_deesc",
    ]
    out = {}
    for i, name in enumerate(names):
        vals = arr[:, i]
        vals = vals[np.isfinite(vals)]
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_ci2.5"] = float(np.quantile(vals, 0.025))
        out[f"{name}_ci97.5"] = float(np.quantile(vals, 0.975))
    return out


def plot_universal_gate(rows: pd.DataFrame) -> None:
    """외부 데이터에서 상위 AEC 점수 후보들의 특이도이득 vs 민감도손실 트레이드오프를, 임상규칙(youden/sens90) 두 패널로 나눠 산점도로 그려 PNG로 저장."""
    plot = rows[
        rows["dataset"].eq("sdata_external")
        & rows["clinical_rule"].isin(["youden", "sens90"])
        & rows["deesc_fraction_train"].isin([0.05, 0.10, 0.125, 0.15, 0.20])
    ].copy()
    plot["label"] = plot["aec_score"] + "\nq=" + plot["deesc_fraction_train"].astype(str)
    top_scores = (
        plot.sort_values(["clinical_pos_aec_deesc_prevalence", "specificity_gain"], ascending=[True, False])
        ["aec_score"]
        .drop_duplicates()
        .head(6)
        .tolist()
    )
    plot = plot[plot["aec_score"].isin(top_scores)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharex=False)
    for ax, clin in zip(axes, ["youden", "sens90"]):
        sub = plot[plot["clinical_rule"].eq(clin)].copy()
        sub = sub.sort_values(["aec_score", "deesc_fraction_train"])
        x = np.arange(len(sub))
        ax.scatter(
            sub["specificity_gain"],
            sub["sensitivity_loss"],
            s=60 + 3 * sub["clinical_pos_aec_deesc_n"],
            c=sub["clinical_pos_aec_deesc_prevalence"],
            cmap="viridis_r",
            edgecolor="#333333",
            linewidth=0.5,
        )
        for _, r in sub.iterrows():
            if r["deesc_fraction_train"] in {0.10, 0.125, 0.15}:
                ax.text(
                    r["specificity_gain"],
                    r["sensitivity_loss"],
                    f"{r['aec_score'].replace('_', ' ')[:18]}\nq={r['deesc_fraction_train']}",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )
        ax.axhline(0, color="#555555", lw=1, ls="--")
        ax.axvline(0, color="#555555", lw=1, ls="--")
        ax.set_title(f"sdata external: clinical {clin}", loc="left", fontweight="bold")
        ax.set_xlabel("Specificity gain")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sensitivity loss")
    fig.suptitle("Universal AEC-negative de-escalation gate", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "universal_gate_external_tradeoff.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (6/30 첫 스크립트 — 질문: 성별이나 임상경계 가중치 같은 "특별
    취급" 없이, 그냥 "임상 양성군 중 AEC 위험점수가 낮은 사람"을 하향조정하는 가장 단순한 보편적
    규칙만으로도 안전하게 위양성을 줄일 수 있는가?):

    1. g1090/sdata를 로드하고, 임상 단독 모델의 OOF/외부 점수와 임계값(Youden/민감도85·90·95%)을 준비.
    2. build_aec_scores로 10개의 서로 다른 AEC 단독 위험점수 후보(모양특징 여러 개, 레거시 SVM,
       핵심특징 평균 z점수)를 만든다.
    3. 4개 임상규칙 x 10개 AEC 점수 x 9개 하향조정 비율(3~30%) = 360개 조합 각각에 대해
       rule_metrics로 "임상 양성 중 AEC 하위 분위수 이하는 하향조정" 규칙의 성능을 train/외부 양쪽에서 계산.
    4. 외부 데이터에서 하향조정 인원 25명 이상, 하향조정군 사건율 5% 이하, 민감도손실 3% 이하인
       "강한 후보"만 골라 shortlist로 저장.
    5. train/외부 결과를 나란히 비교하는 표를 만들고, shortlist 1위·레거시 SVM 폴백·해석가능한
       핵심특징 폴백 세 가지를 골라 부트스트랩으로 신뢰구간을 재확인.
    6. 트레이드오프 산점도를 그리고, 방법론·임상 기준선·shortlist·전체 결과를 JSON으로 저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_g, raw_g, norm_g, y_g, sex_g = fill_raw_aec128(DATA_DIR / "g1090.xlsx")
    meta_s, raw_s, norm_s, y_s, sex_s = fill_raw_aec128(DATA_DIR / "sdata.xlsx")

    xclin_g = clinical_raw(meta_g)
    xclin_s = clinical_raw(meta_s)
    clin_oof, clin_ext = crossfit_external_estimator(lambda seed: clinical_model(), xclin_g, y_g, xclin_s)
    aec_g, aec_s, score_meta = build_aec_scores(meta_g, raw_g, norm_g, y_g, meta_s, raw_s, norm_s)

    clinical_thresholds = {
        "youden": threshold_youden(y_g, clin_oof),
        "sens85": threshold_for_min_sensitivity(y_g, clin_oof, 0.85),
        "sens90": threshold_for_min_sensitivity(y_g, clin_oof, 0.90),
        "sens95": threshold_for_min_sensitivity(y_g, clin_oof, 0.95),
    }
    deesc_fracs = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for clinical_rule, clinical_th in clinical_thresholds.items():
        cp_g = clin_oof >= clinical_th
        for score_name in aec_g.columns:
            train_score = aec_g[score_name].to_numpy(dtype=float)
            ext_score = aec_s[score_name].to_numpy(dtype=float)
            for frac in deesc_fracs:
                if np.sum(cp_g) < 5:
                    continue
                aec_th = float(np.quantile(train_score[cp_g], frac))
                for dataset, y, cscore, ascore in [
                    ("g1090_oof", y_g, clin_oof, train_score),
                    ("sdata_external", y_s, clin_ext, ext_score),
                ]:
                    row = {
                        "dataset": dataset,
                        "clinical_rule": clinical_rule,
                        "clinical_threshold_from_g1090": clinical_th,
                        "aec_score": score_name,
                        "aec_score_source": score_meta[score_name]["source"],
                        "aec_score_g1090_auc": score_meta[score_name]["g1090_auc"],
                        "deesc_fraction_train": frac,
                        "aec_threshold_from_g1090_clinical_positive_quantile": aec_th,
                    }
                    row.update(rule_metrics(y, cscore, clinical_th, ascore, aec_th))
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "universal_gate_all_tradeoffs.csv", index=False)

    # Candidate shortlist: external de-escalated event prevalence <= 5%, at least 25 de-escalated, and sensitivity loss <= 3%.
    ext = out[out["dataset"].eq("sdata_external")].copy()
    shortlist = ext[
        (ext["clinical_pos_aec_deesc_n"] >= 25)
        & (ext["clinical_pos_aec_deesc_prevalence"] <= 0.05)
        & (ext["sensitivity_loss"] <= 0.03)
    ].sort_values(["clinical_pos_aec_deesc_prevalence", "specificity_gain", "clinical_pos_aec_deesc_n"], ascending=[True, False, False])
    shortlist.to_csv(OUT_DIR / "universal_gate_external_shortlist_strong.csv", index=False)

    paired = out.pivot_table(
        index=["clinical_rule", "aec_score", "deesc_fraction_train"],
        columns="dataset",
        values=[
            "clinical_pos_aec_deesc_n",
            "clinical_pos_aec_deesc_events",
            "clinical_pos_aec_deesc_prevalence",
            "sensitivity_loss",
            "specificity_gain",
            "ppv_gain",
            "false_positives_removed",
            "true_positives_lost",
            "within_clinical_pos_or_keep_vs_deesc",
            "within_clinical_pos_fisher_p",
        ],
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{dataset}" for metric, dataset in paired.columns]
    paired = paired.reset_index()
    paired = paired.sort_values(
        [
            "clinical_pos_aec_deesc_prevalence_sdata_external",
            "specificity_gain_sdata_external",
            "clinical_pos_aec_deesc_n_sdata_external",
        ],
        ascending=[True, False, False],
    )
    paired.to_csv(OUT_DIR / "universal_gate_paired_train_external.csv", index=False)

    # Bootstrap the most useful universal candidate and a fully interpretable fallback.
    boot_rows = []
    selected = []
    if not shortlist.empty:
        selected.append(shortlist.iloc[0])
    fallback = ext[
        (ext["clinical_rule"].eq("youden"))
        & (ext["aec_score"].eq("legacy_aec256_svm_norm_only"))
        & (ext["deesc_fraction_train"].isin([0.10, 0.125, 0.15]))
    ].sort_values(["clinical_pos_aec_deesc_prevalence", "specificity_gain"])
    if not fallback.empty:
        selected.append(fallback.iloc[0])
    interp = ext[
        (ext["clinical_rule"].eq("youden"))
        & (ext["aec_score"].eq("universal_norm_tail_core_mean_z"))
        & (ext["deesc_fraction_train"].isin([0.10, 0.125, 0.15]))
    ].sort_values(["clinical_pos_aec_deesc_prevalence", "specificity_gain"])
    if not interp.empty:
        selected.append(interp.iloc[0])
    seen = set()
    for r in selected:
        key = (r["clinical_rule"], r["aec_score"], float(r["deesc_fraction_train"]))
        if key in seen:
            continue
        seen.add(key)
        boot = bootstrap_external_delta(
            y_s,
            clin_ext,
            float(r["clinical_threshold_from_g1090"]),
            aec_s[str(r["aec_score"])].to_numpy(dtype=float),
            float(r["aec_threshold_from_g1090_clinical_positive_quantile"]),
        )
        boot_rows.append(
            {
                "clinical_rule": r["clinical_rule"],
                "aec_score": r["aec_score"],
                "deesc_fraction_train": float(r["deesc_fraction_train"]),
                **boot,
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "universal_gate_selected_bootstrap_external.csv", index=False)

    plot_universal_gate(out)

    clinical_ref = {
        "g1090_oof": auc_metrics(y_g, clin_oof),
        "sdata_external": auc_metrics(y_s, clin_ext),
        "clinical_thresholds_from_g1090": clinical_thresholds,
    }
    summary = {
        "definition": {
            "universal_gate": "Clinical-positive patients are de-escalated if their AEC-only risk score falls below a fixed g1090 clinical-positive quantile threshold. No sex term and no clinical-boundary weighting are used.",
            "aec_negative_rule": "AEC risk score < g1090 quantile among clinical-positive patients",
            "deescalation_fractions_tested": deesc_fracs,
            "clinical_rules": list(clinical_thresholds.keys()),
        },
        "clinical_reference": clinical_ref,
        "aec_scores": score_meta,
        "external_shortlist": shortlist.head(30).to_dict(orient="records"),
        "top_paired": paired.head(30).to_dict(orient="records"),
        "outputs": {
            "all_tradeoffs": str(OUT_DIR / "universal_gate_all_tradeoffs.csv"),
            "external_shortlist": str(OUT_DIR / "universal_gate_external_shortlist_strong.csv"),
            "paired_train_external": str(OUT_DIR / "universal_gate_paired_train_external.csv"),
            "bootstrap": str(OUT_DIR / "universal_gate_selected_bootstrap_external.csv"),
            "plot": str(OUT_DIR / "universal_gate_external_tradeoff.png"),
        },
    }
    (OUT_DIR / "universal_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nClinical reference")
    print(json.dumps(clinical_ref, indent=2))
    print("\nExternal strong universal shortlist")
    cols = [
        "clinical_rule",
        "aec_score",
        "deesc_fraction_train",
        "clinical_positive_n",
        "clinical_positive_events",
        "clinical_pos_aec_deesc_n",
        "clinical_pos_aec_deesc_events",
        "clinical_pos_aec_deesc_prevalence",
        "clinical_pos_aec_keep_prevalence",
        "within_clinical_pos_or_keep_vs_deesc",
        "within_clinical_pos_fisher_p",
        "false_positives_removed",
        "true_positives_lost",
        "specificity_gain",
        "sensitivity_loss",
        "ppv_gain",
    ]
    if shortlist.empty:
        print("None")
    else:
        print(shortlist[cols].head(25).to_string(index=False))
    print("\nPaired top")
    pcols = [
        "clinical_rule",
        "aec_score",
        "deesc_fraction_train",
        "clinical_pos_aec_deesc_n_g1090_oof",
        "clinical_pos_aec_deesc_events_g1090_oof",
        "clinical_pos_aec_deesc_prevalence_g1090_oof",
        "clinical_pos_aec_deesc_n_sdata_external",
        "clinical_pos_aec_deesc_events_sdata_external",
        "clinical_pos_aec_deesc_prevalence_sdata_external",
        "specificity_gain_sdata_external",
        "sensitivity_loss_sdata_external",
    ]
    print(paired[pcols].head(25).to_string(index=False))
    print("\nBootstrap selected")
    print(boot_df.to_string(index=False) if not boot_df.empty else "None")
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
