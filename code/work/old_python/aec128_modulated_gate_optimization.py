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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    zfit_apply,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec128_modulated_gate_optimization"
SEED = 20260630


def load_aec128_norm(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """엑셀에서 aec_128을 읽어 행 정규화(-1 오프셋)한 곡선과 저근감소증 라벨을 반환."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    x = row_norm(raw) - 1.0
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return meta, x, y


def aec128_model(seed: int, c: float = 0.2, k: int = 64) -> Pipeline:
    """결측대체→표준화→상위 64개 특징 선택→선형 SVM으로 이어지는 AEC 전용 분류 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("svm", LinearSVC(C=c, class_weight="balanced", max_iter=20000, random_state=seed)),
        ]
    )


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


def deesc_metrics(y: np.ndarray, clinical_z: np.ndarray, gate_z: np.ndarray, th: float) -> dict:
    """게이트 규칙(임상양성 중 게이트점수가 임계값 미만이면 하향조정)의 유지/하향조정군 통계, 민감도손실/
    특이도이득/PPV이득, Fisher 오즈비·p값을 모두 계산."""
    clinical_pos = clinical_z >= th
    gate_pos = gate_z >= th
    final_pos = clinical_pos & gate_pos
    deesc = clinical_pos & ~gate_pos
    keep = clinical_pos & gate_pos
    base = binary_counts(y, clinical_pos)
    rule = binary_counts(y, final_pos)

    keep_event = int(np.sum(y[keep] == 1))
    keep_nonevent = int(np.sum(y[keep] == 0))
    de_event = int(np.sum(y[deesc] == 1))
    de_nonevent = int(np.sum(y[deesc] == 0))
    if keep_event + keep_nonevent and de_event + de_nonevent:
        orr, fisher_p = stats.fisher_exact([[keep_event, keep_nonevent], [de_event, de_nonevent]])
    else:
        orr, fisher_p = np.nan, np.nan

    return {
        **{f"clinical_{k}": v for k, v in base.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "clinical_positive_n": int(np.sum(clinical_pos)),
        "clinical_positive_events": int(np.sum(y[clinical_pos] == 1)),
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_event,
        "deesc_prevalence": de_event / (de_event + de_nonevent) if de_event + de_nonevent else np.nan,
        "fp_removed": de_nonevent,
        "tp_lost": de_event,
        "specificity_gain": rule["specificity"] - base["specificity"],
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "ppv_gain": rule["ppv"] - base["ppv"],
        "or_keep_vs_deesc": float(orr) if np.isfinite(orr) else np.nan,
        "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    """로짓 값을 0~1 확률로 변환 (오버플로 방지를 위해 -40~40으로 클리핑)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def modulation_weight(c: np.ndarray, th: float, family: str, center: float, width: float, power: float) -> np.ndarray:
    """임상점수와 임계값의 거리(d)를 받아, 5가지 함수형태(gaussian/right_decay/tent/sigmoid_band/
    clinical_uncertainty) 중 하나로 AEC 반영 가중치를 계산하고, power로 날카로움을 추가 조절."""
    d = c - th
    if family == "gaussian":
        w = np.exp(-0.5 * ((d - center) / width) ** 2)
    elif family == "right_decay":
        # Maximal just above threshold, decays as clinical confidence increases.
        w = np.exp(-np.maximum(d - center, 0.0) / width) * (d >= center - width).astype(float)
    elif family == "tent":
        w = np.maximum(0.0, 1.0 - np.abs(d - center) / width)
    elif family == "sigmoid_band":
        left = sigmoid((d - (center - width)) / max(width / 4, 1e-6))
        right = sigmoid(((center + width) - d) / max(width / 4, 1e-6))
        w = left * right
    elif family == "clinical_uncertainty":
        # Threshold-centered analogue of p*(1-p), expressed on standardized score.
        w = np.exp(-np.abs(d - center) / width)
    else:
        raise ValueError(f"unknown family {family}")
    if power != 1.0:
        w = np.power(np.clip(w, 0.0, 1.0), power)
    return np.clip(w, 0.0, 1.0)


def transform_aec(a: np.ndarray, transform: str) -> np.ndarray:
    """AEC 점수에 5가지 변형(linear/clip2/neg_only/neg_clip2/softsign) 중 하나를 적용해 반환."""
    if transform == "linear":
        return a
    if transform == "clip2":
        return np.clip(a, -2.0, 2.0)
    if transform == "neg_only":
        return np.minimum(a, 0.0)
    if transform == "neg_clip2":
        return np.clip(np.minimum(a, 0.0), -2.0, 0.0)
    if transform == "softsign":
        return a / (1.0 + np.abs(a))
    raise ValueError(transform)


def build_gate(c: np.ndarray, a: np.ndarray, th: float, params: dict) -> np.ndarray:
    """params에 지정된 가중함수·변형을 적용해 "임상점수 + 람다 x 가중치 x 변형된 AEC" 형태의 게이트 점수를 계산."""
    w = modulation_weight(c, th, params["family"], params["center"], params["width"], params["power"])
    aa = transform_aec(a, params["aec_transform"])
    return c + params["lambda"] * w * aa


def bootstrap_external(y: np.ndarray, c: np.ndarray, g: np.ndarray, th: float, n_boot: int = 3000) -> pd.DataFrame:
    """게이트 규칙의 하향조정군 통계·민감도손실·특이도이득·PPV이득을 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 111)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        m = deesc_metrics(yy, c[idx], g[idx], th)
        vals.append(
            [
                m["deesc_n"],
                m["deesc_events"],
                m["deesc_prevalence"],
                m["fp_removed"],
                m["tp_lost"],
                m["specificity_gain"],
                m["sensitivity_loss"],
                m["ppv_gain"],
            ]
        )
    arr = np.asarray(vals)
    rows = []
    for i, metric in enumerate(
        ["deesc_n", "deesc_events", "deesc_prevalence", "fp_removed", "tp_lost", "specificity_gain", "sensitivity_loss", "ppv_gain"]
    ):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(x)),
                "ci2.5": float(np.quantile(x, 0.025)),
                "ci97.5": float(np.quantile(x, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def selection_score(row: pd.Series) -> float:
    """하향조정 인원 20명 미만/민감도손실 2.5% 초과/하향조정군 유병률 8% 초과 중 하나라도 해당하면
    탈락(-1e9)시키고, 나머지는 특이도이득·FP제거수·TP손실·유병률을 가중합해 train 선택 점수를 계산."""
    # Priority: very low de-escalated event rate, many removed FPs, limited TP loss.
    if row["deesc_n"] < 20:
        return -1e9
    if row["sensitivity_loss"] > 0.025:
        return -1e9
    if row["deesc_prevalence"] > 0.08:
        return -1e9
    return (
        2.5 * row["specificity_gain"]
        + 0.003 * row["fp_removed"]
        - 0.10 * row["tp_lost"]
        - 0.35 * row["deesc_prevalence"]
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 게이트 가중함수의 "형태"를 5가지, 중심·폭·강도·날카로움·
    AEC변형까지 대규모 그리드로 바꿔가며 찾으면, 이전보다 더 나은 하향조정 규칙을 찾을 수 있는가?
    — train에서만 고르고 외부에서 검증하는 대규모 파라미터 탐색):

    1. g1090/sdata를 로드하고, 임상 단독 모델과 AEC128 단독(SVM) 모델의 OOF/외부 점수를 표준화해 준비.
    2. 민감도 목표 4종(90/92.5/95/97.5%)에서 임상 임계값을 고정.
    3. 임계값 4 x 가중함수형태 5 x 중심 6 x 폭 7 x 람다 8 x 날카로움(power) 4 x AEC변형 5 =
       총 6720개 파라미터 조합 각각에 대해 build_gate로 게이트 점수를 만들고 deesc_metrics로
       train/외부 성능을 모두 계산 (엄청난 규모의 그리드서치).
    4. train 결과만으로 selection_score를 계산해 "하향조정 20명 이상, 민감도손실 2.5% 이하,
       하향조정군 유병률 8% 이하"인 후보 중 점수가 가장 높은 상위 100개를 뽑는다 (외부 데이터는
       선택에 전혀 관여하지 않음 — train-only 선택 규칙).
    5. 비교용으로 "외부 데이터를 봤다면 얼마나 좋았을까"를 보여주는 external_oracle 상위 100개도
       별도로 계산해두되, 이건 주 증거가 아니라 상한선 참고용이라고 명시.
    6. train 선택 1위, train 선택 중 외부에서도 조건을 통과하는 후보, 외부 오라클 1위 세 가지를
       뽑아 부트스트랩으로 외부 신뢰구간을 재확인.
    7. 외부 특이도이득-민감도손실 산점도에 선택된 후보들을 강조 표시해 저장하고, 방법론·상위결과·
       전체 산출물 경로를 JSON으로 저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_g, xg, yg = load_aec128_norm(DATA_DIR / "g1090.xlsx")
    meta_s, xs, ys = load_aec128_norm(DATA_DIR / "sdata.xlsx")
    xclin_g, xclin_s, _ = clinical_matrix(meta_g, meta_s)
    folds = make_folds(yg, 5)

    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_g, yg, xclin_s, folds)
    aec_oof, aec_ext = oof_and_external(lambda seed: aec128_model(seed), xg, yg, xs, folds)
    c_g, c_s, _, _ = zfit_apply(clinical_oof, clinical_ext)
    a_g, a_s, _, _ = zfit_apply(aec_oof, aec_ext)
    if np.corrcoef(a_g, yg)[0, 1] < 0:
        a_g = -a_g
        a_s = -a_s

    threshold_targets = {
        "sens90": 0.90,
        "sens925": 0.925,
        "sens95": 0.95,
        "sens975": 0.975,
    }
    c_mu = float(np.mean(clinical_oof))
    c_sd = float(np.std(clinical_oof)) or 1.0
    thresholds = {
        name: (threshold_for_min_sensitivity(yg, clinical_oof, target) - c_mu) / c_sd
        for name, target in threshold_targets.items()
    }

    grid = []
    for threshold_name, th in thresholds.items():
        for family in ["gaussian", "right_decay", "tent", "sigmoid_band", "clinical_uncertainty"]:
            for center in [-0.30, -0.15, 0.0, 0.15, 0.30, 0.50]:
                for width in [0.20, 0.30, 0.40, 0.55, 0.75, 1.00, 1.25]:
                    for lam in [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70, 1.00]:
                        for power in [0.7, 1.0, 1.5, 2.0]:
                            for aec_transform in ["linear", "clip2", "neg_only", "neg_clip2", "softsign"]:
                                params = {
                                    "threshold_name": threshold_name,
                                    "threshold_z": float(th),
                                    "family": family,
                                    "center": center,
                                    "width": width,
                                    "lambda": lam,
                                    "power": power,
                                    "aec_transform": aec_transform,
                                }
                                gg = build_gate(c_g, a_g, th, params)
                                gs = build_gate(c_s, a_s, th, params)
                                row_g = {
                                    "dataset": "g1090_oof",
                                    **params,
                                    **deesc_metrics(yg, c_g, gg, th),
                                }
                                row_s = {
                                    "dataset": "sdata_external",
                                    **params,
                                    **deesc_metrics(ys, c_s, gs, th),
                                }
                                grid.extend([row_g, row_s])
    all_df = pd.DataFrame(grid)
    all_df.to_csv(OUT_DIR / "modulated_gate_all_grid_long.csv", index=False)

    paired = all_df.pivot_table(
        index=["threshold_name", "threshold_z", "family", "center", "width", "lambda", "power", "aec_transform"],
        columns="dataset",
        values=[
            "deesc_n",
            "deesc_events",
            "deesc_prevalence",
            "fp_removed",
            "tp_lost",
            "specificity_gain",
            "sensitivity_loss",
            "ppv_gain",
            "or_keep_vs_deesc",
            "fisher_p",
        ],
        aggfunc="first",
    )
    paired.columns = [f"{m}_{d}" for m, d in paired.columns]
    paired = paired.reset_index()
    paired["train_selection_score"] = paired.apply(lambda r: selection_score(r.rename(lambda x: x.replace("_g1090_oof", ""))), axis=1)
    paired = paired.sort_values("train_selection_score", ascending=False)
    paired.to_csv(OUT_DIR / "modulated_gate_paired_grid.csv", index=False)

    train_selected = paired[paired["train_selection_score"] > -1e8].head(100).copy()
    train_selected.to_csv(OUT_DIR / "train_selected_modulated_candidates_top100.csv", index=False)

    external_oracle = paired[
        (paired["deesc_n_sdata_external"] >= 20)
        & (paired["sensitivity_loss_sdata_external"] <= 0.025)
        & (paired["deesc_prevalence_sdata_external"] <= 0.08)
    ].copy()
    external_oracle["external_oracle_score"] = (
        2.5 * external_oracle["specificity_gain_sdata_external"]
        + 0.003 * external_oracle["fp_removed_sdata_external"]
        - 0.10 * external_oracle["tp_lost_sdata_external"]
        - 0.35 * external_oracle["deesc_prevalence_sdata_external"]
    )
    external_oracle = external_oracle.sort_values("external_oracle_score", ascending=False)
    external_oracle.head(100).to_csv(OUT_DIR / "external_oracle_modulated_candidates_top100.csv", index=False)

    selected = []
    if not train_selected.empty:
        selected.append(("train_selected", train_selected.iloc[0]))
    robust = train_selected[
        (train_selected["deesc_n_sdata_external"] >= 20)
        & (train_selected["deesc_prevalence_sdata_external"] <= 0.08)
        & (train_selected["sensitivity_loss_sdata_external"] <= 0.025)
    ]
    if not robust.empty:
        selected.append(("train_selected_external_pass", robust.iloc[0]))
    if not external_oracle.empty:
        selected.append(("external_oracle_reference", external_oracle.iloc[0]))

    boot_tables = []
    seen = set()
    for label, row in selected:
        key = tuple(row[x] for x in ["threshold_name", "family", "center", "width", "lambda", "power", "aec_transform"])
        if key in seen:
            continue
        seen.add(key)
        params = {
            "family": row["family"],
            "center": float(row["center"]),
            "width": float(row["width"]),
            "lambda": float(row["lambda"]),
            "power": float(row["power"]),
            "aec_transform": row["aec_transform"],
        }
        th = float(row["threshold_z"])
        gs = build_gate(c_s, a_s, th, params)
        boot = bootstrap_external(ys, c_s, gs, th)
        for k in ["threshold_name", "family", "center", "width", "lambda", "power", "aec_transform"]:
            boot[k] = row[k]
        boot["selection_label"] = label
        boot_tables.append(boot)
    boot_df = pd.concat(boot_tables, ignore_index=True) if boot_tables else pd.DataFrame()
    boot_df.to_csv(OUT_DIR / "selected_modulated_external_bootstrap.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sample = paired[
        (paired["deesc_n_sdata_external"] >= 15)
        & (paired["sensitivity_loss_sdata_external"] <= 0.05)
    ].copy()
    ax.scatter(
        sample["specificity_gain_sdata_external"],
        sample["sensitivity_loss_sdata_external"],
        s=np.clip(sample["deesc_n_sdata_external"], 15, 90),
        c=sample["deesc_prevalence_sdata_external"],
        cmap="viridis_r",
        alpha=0.55,
        edgecolor="none",
    )
    for label, row in selected:
        ax.scatter(
            row["specificity_gain_sdata_external"],
            row["sensitivity_loss_sdata_external"],
            s=140,
            facecolor="none",
            edgecolor="#C84630",
            linewidth=2.0,
        )
        ax.text(
            row["specificity_gain_sdata_external"],
            row["sensitivity_loss_sdata_external"],
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlabel("sdata specificity gain")
    ax.set_ylabel("sdata sensitivity loss")
    ax.set_title("Expanded AEC128 modulation search", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "expanded_modulation_external_tradeoff.png", dpi=220)
    plt.close(fig)

    summary = {
        "method": {
            "aec_score": "OOF/external LinearSVM score from normalized AEC_128 only, direction fixed on g1090 OOF.",
            "selection": "Parameters are selected on g1090 OOF by a utility score favoring FP removal, low de-escalated event rate, and low sensitivity loss; sdata is locked external evaluation.",
            "external_oracle": "Reported separately only to show the ceiling if external data were tuned on; not primary evidence.",
        },
        "train_selected_top10": train_selected.head(10).to_dict(orient="records"),
        "external_oracle_top10": external_oracle.head(10).to_dict(orient="records"),
        "outputs": {
            "all_grid": str(OUT_DIR / "modulated_gate_all_grid_long.csv"),
            "paired_grid": str(OUT_DIR / "modulated_gate_paired_grid.csv"),
            "train_selected": str(OUT_DIR / "train_selected_modulated_candidates_top100.csv"),
            "external_oracle": str(OUT_DIR / "external_oracle_modulated_candidates_top100.csv"),
            "bootstrap": str(OUT_DIR / "selected_modulated_external_bootstrap.csv"),
            "plot": str(OUT_DIR / "expanded_modulation_external_tradeoff.png"),
        },
    }
    (OUT_DIR / "modulated_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    show_cols = [
        "threshold_name",
        "family",
        "center",
        "width",
        "lambda",
        "power",
        "aec_transform",
        "deesc_n_g1090_oof",
        "deesc_events_g1090_oof",
        "deesc_prevalence_g1090_oof",
        "fp_removed_g1090_oof",
        "tp_lost_g1090_oof",
        "specificity_gain_g1090_oof",
        "sensitivity_loss_g1090_oof",
        "deesc_n_sdata_external",
        "deesc_events_sdata_external",
        "deesc_prevalence_sdata_external",
        "fp_removed_sdata_external",
        "tp_lost_sdata_external",
        "specificity_gain_sdata_external",
        "sensitivity_loss_sdata_external",
        "fisher_p_sdata_external",
        "train_selection_score",
    ]
    print("\nTrain-selected top")
    print(train_selected[show_cols].head(20).to_string(index=False) if not train_selected.empty else "None")
    print("\nTrain-selected and external-pass top")
    print(robust[show_cols].head(20).to_string(index=False) if not robust.empty else "None")
    print("\nExternal oracle top")
    print(external_oracle[show_cols + ["external_oracle_score"]].head(20).to_string(index=False) if not external_oracle.empty else "None")
    print("\nBootstrap selected")
    print(boot_df.to_string(index=False) if not boot_df.empty else "None")
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
