from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_five_strategy_audit import FEATURE_SPECS, clinical_pipeline, clinical_scores, load_all  # noqa: E402
from aec_offset_score import fit_offset_ridge, sigmoid  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_soft_gate_models"
SEED = 20260629
OUTER_SPLITS = 5
INNER_SPLITS = 3
LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


FEATURE_GROUPS: dict[str, list[str]] = {
    "tail_rebound": ["tail_rebound_114_128_max_minus_60_95_min"],
    "late_rebound": ["late_rebound_91_128_max_minus_60_95_min"],
    "tail_level": ["tail_level_114_128"],
    "trough_range": ["trough_range_60_95"],
    "roughness": ["roughness_75_90"],
    "recovery_slope": ["recovery_slope_81_113"],
    "tail_plus_trough": ["tail_rebound_114_128_max_minus_60_95_min", "trough_range_60_95"],
    "robust_shape3": [
        "tail_rebound_114_128_max_minus_60_95_min",
        "trough_range_60_95",
        "roughness_75_90",
    ],
    "all8_shape": list(FEATURE_SPECS.keys()),
}


GATE_DESCRIPTIONS = {
    "no_gate": "w(p)=1; ordinary clinical-offset AEC term.",
    "uncertainty_mid": "w(p)=4*p*(1-p); maximal at p=0.5, shrinks near confident low/high.",
    "soft_high_uncertainty": "w(p)=6.75*p^2*(1-p); maximal near p=2/3, favors high-but-not-certain clinical risk.",
    "soft_high_risk": "w(p)=p; AEC influence increases continuously with clinical risk.",
}


def gate_weight(prob: np.ndarray, gate: str) -> np.ndarray:
    """임상 확률 p로부터 AEC 항에 곱할 "부드러운(threshold-free)" 가중치 w(p)를 계산 (게이트 종류에 따라 형태가 다름 — 딱딱한 임계값 대신 연속함수로 게이팅)."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if gate == "no_gate":
        return np.ones_like(p)
    if gate == "uncertainty_mid":
        return 4.0 * p * (1.0 - p)
    if gate == "soft_high_uncertainty":
        return 6.75 * (p**2) * (1.0 - p)
    if gate == "soft_high_risk":
        return p
    raise ValueError(f"Unknown gate: {gate}")


def standardize_train_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train의 평균/표준편차(결측 제외)로 train·test를 함께 표준화하고, 결측은 train 평균으로 대체."""
    xtr = np.asarray(xtr, dtype=float)
    xte = np.asarray(xte, dtype=float)
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    xtr_f = np.where(np.isfinite(xtr), xtr, mu)
    xte_f = np.where(np.isfinite(xte), xte, mu)
    return (xtr_f - mu) / sd, (xte_f - mu) / sd, mu, sd


def make_terms_train_apply(
    f_tr: np.ndarray,
    f_te: np.ndarray,
    clinical_score_tr: np.ndarray,
    clinical_score_te: np.ndarray,
    female_tr: np.ndarray,
    female_te: np.ndarray,
    gate: str,
    female_interaction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """AEC 특징을 표준화한 뒤 게이트 가중치 w(임상확률)를 곱해 "게이트된 AEC 항"을 만들고,
    female_interaction이 True면 여성 여부와의 상호작용항까지 추가한 뒤 다시 표준화."""
    fz_tr, fz_te, _, _ = standardize_train_apply(f_tr, f_te)
    w_tr = gate_weight(sigmoid(clinical_score_tr), gate)[:, None]
    w_te = gate_weight(sigmoid(clinical_score_te), gate)[:, None]
    terms_tr = fz_tr * w_tr
    terms_te = fz_te * w_te
    if female_interaction:
        terms_tr = np.column_stack([terms_tr, terms_tr * female_tr[:, None]])
        terms_te = np.column_stack([terms_te, terms_te * female_te[:, None]])
    terms_tr, terms_te, _, _ = standardize_train_apply(terms_tr, terms_te)
    return terms_tr, terms_te


def metric_dict(y: np.ndarray, score: np.ndarray) -> dict:
    """점수로부터 AUC/AP/로그손실/Brier를 계산."""
    prob = np.clip(sigmoid(score), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def threshold_free_reclassification(y: np.ndarray, base_score: np.ndarray, new_score: np.ndarray) -> dict:
    """임계값 없이, 확률 변화 기반의 IDI(통합판별개선도)와 연속형 NRI(순재분류개선도)를 계산해 새 점수가 기존 점수보다 얼마나 잘 재분류하는지 평가."""
    y = np.asarray(y).astype(bool)
    p0 = sigmoid(base_score)
    p1 = sigmoid(new_score)
    event_delta = float(np.mean(p1[y] - p0[y]))
    nonevent_delta = float(np.mean(p1[~y] - p0[~y]))
    idi = event_delta - nonevent_delta

    up = p1 > p0
    down = p1 < p0
    nri_event = float(np.mean(up[y]) - np.mean(down[y]))
    nri_nonevent = float(np.mean(down[~y]) - np.mean(up[~y]))
    return {
        "event_mean_probability_delta": event_delta,
        "nonevent_mean_probability_delta": nonevent_delta,
        "idi_discrimination_slope_delta": float(idi),
        "continuous_nri": float(nri_event + nri_nonevent),
        "continuous_nri_event_component": nri_event,
        "continuous_nri_nonevent_component": nri_nonevent,
    }


def choose_lambda_inner(
    f: np.ndarray,
    y: np.ndarray,
    clinical_x: np.ndarray,
    female: np.ndarray,
    gate: str,
    female_interaction: bool,
    lambdas: list[float],
    seed: int,
) -> tuple[float, pd.DataFrame]:
    """내부 교차검증으로 여러 lambda 후보에 대해 (임상 오프셋 고정 후) 게이트된 AEC 릿지 로지스틱의 로그손실을 비교해 최적값을 선택."""
    skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    rows = []
    for lam in lambdas:
        scores = np.zeros(len(y), dtype=float)
        for tr, va in skf.split(f, y):
            cm = clinical_pipeline()
            cm.fit(clinical_x[tr], y[tr])
            c_tr = cm.decision_function(clinical_x[tr])
            c_va = cm.decision_function(clinical_x[va])
            terms_tr, terms_va = make_terms_train_apply(
                f[tr], f[va], c_tr, c_va, female[tr], female[va], gate, female_interaction
            )
            alpha, beta = fit_offset_ridge(terms_tr, y[tr], c_tr, lam)
            scores[va] = c_va + alpha + terms_va @ beta
        rows.append(
            {
                "lambda": lam,
                "inner_auc": float(roc_auc_score(y, scores)),
                "inner_average_precision": float(average_precision_score(y, scores)),
                "inner_log_loss": float(log_loss(y, np.clip(sigmoid(scores), 1e-6, 1 - 1e-6))),
                "inner_brier": float(brier_score_loss(y, sigmoid(scores))),
            }
        )
    df = pd.DataFrame(rows)
    best = df.sort_values(["inner_log_loss", "inner_brier", "inner_auc"], ascending=[True, True, False]).iloc[0]
    return float(best["lambda"]), df


def crossfit_soft_gate(
    g: dict,
    feature_names: list[str],
    gate: str,
    female_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """외부 폴드마다 내부 CV로 lambda를 다시 고르는 중첩교차검증으로, 지정된 특징그룹·게이트·여성상호작용
    조합에 대해 train 전체의 out-of-fold 임상/결합 점수를 만듦."""
    f = g["features"][feature_names].to_numpy(dtype=float)
    y = g["y"].astype(int)
    female = g["female"].astype(float)
    clinical_x = g["clinical_x"]

    clinical_oof = np.zeros(len(y), dtype=float)
    combined_oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    skf = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(f, y), start=1):
        best_lam, inner_df = choose_lambda_inner(
            f[tr],
            y[tr],
            clinical_x[tr],
            female[tr],
            gate,
            female_interaction,
            LAMBDAS,
            seed=SEED + 100 * fold,
        )
        cm = clinical_pipeline()
        cm.fit(clinical_x[tr], y[tr])
        c_tr = cm.decision_function(clinical_x[tr])
        c_va = cm.decision_function(clinical_x[va])
        terms_tr, terms_va = make_terms_train_apply(
            f[tr], f[va], c_tr, c_va, female[tr], female[va], gate, female_interaction
        )
        alpha, beta = fit_offset_ridge(terms_tr, y[tr], c_tr, best_lam)
        clinical_oof[va] = c_va
        combined_oof[va] = c_va + alpha + terms_va @ beta
        fold_rows.append(
            {
                "fold": fold,
                "selected_lambda": best_lam,
                "inner_best_log_loss": float(inner_df.loc[inner_df["lambda"].eq(best_lam), "inner_log_loss"].iloc[0]),
                "alpha": float(alpha),
                "n_terms": int(terms_tr.shape[1]),
            }
        )
    return clinical_oof, combined_oof, pd.DataFrame(fold_rows)


def fit_external_soft_gate(
    g: dict,
    s: dict,
    feature_names: list[str],
    gate: str,
    female_interaction: bool,
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    """train(g1090) 전체로 lambda를 고르고 게이트된 오프셋 릿지 모델을 학습해, 외부(sdata) 데이터의 임상/결합 점수를 계산."""
    f_g = g["features"][feature_names].to_numpy(dtype=float)
    f_s = s["features"][feature_names].to_numpy(dtype=float)
    y = g["y"].astype(int)
    best_lam, inner_df = choose_lambda_inner(
        f_g,
        y,
        g["clinical_x"],
        g["female"].astype(float),
        gate,
        female_interaction,
        LAMBDAS,
        seed=SEED + 700,
    )
    cm = clinical_pipeline()
    cm.fit(g["clinical_x"], y)
    c_g = cm.decision_function(g["clinical_x"])
    c_s = cm.decision_function(s["clinical_x"])
    terms_g, terms_s = make_terms_train_apply(
        f_g,
        f_s,
        c_g,
        c_s,
        g["female"].astype(float),
        s["female"].astype(float),
        gate,
        female_interaction,
    )
    alpha, beta = fit_offset_ridge(terms_g, y, c_g, best_lam)
    return c_s, c_s + alpha + terms_s @ beta, best_lam, inner_df


def evaluate_model(
    model_name: str,
    y: np.ndarray,
    clinical_score: np.ndarray,
    combined_score: np.ndarray,
    dataset: str,
    feature_group: str,
    gate: str,
    female_interaction: bool,
    selected_lambda: float,
    n_terms: int,
) -> dict:
    """임상 단독과 결합모델의 지표를 비교해 delta AUC/AP/로그손실/Brier 감소량과 재분류 지표(IDI/NRI)까지 한 행으로 정리."""
    base = metric_dict(y, clinical_score)
    new = metric_dict(y, combined_score)
    recl = threshold_free_reclassification(y, clinical_score, combined_score)
    return {
        "dataset": dataset,
        "model": model_name,
        "feature_group": feature_group,
        "gate": gate,
        "gate_formula": GATE_DESCRIPTIONS[gate],
        "female_interaction": bool(female_interaction),
        "selected_lambda": selected_lambda,
        "n_terms": n_terms,
        **new,
        "delta_auc_vs_clinical": new["auc"] - base["auc"],
        "delta_average_precision_vs_clinical": new["average_precision"] - base["average_precision"],
        "log_loss_reduction_vs_clinical": base["log_loss"] - new["log_loss"],
        "brier_reduction_vs_clinical": base["brier"] - new["brier"],
        **recl,
    }


def bootstrap_delta_ci(y: np.ndarray, base_score: np.ndarray, new_score: np.ndarray, n_boot: int = 2000) -> dict:
    """임상 단독 대비 결합모델의 AUC/AP/로그손실/Brier/IDI/NRI 개선폭을 부트스트랩 재표본추출로 신뢰구간과 p값과 함께 추정."""
    rng = np.random.default_rng(SEED + 900)
    arr = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        base = metric_dict(yy, base_score[idx])
        new = metric_dict(yy, new_score[idx])
        recl = threshold_free_reclassification(yy, base_score[idx], new_score[idx])
        arr.append(
            [
                new["auc"] - base["auc"],
                new["average_precision"] - base["average_precision"],
                base["log_loss"] - new["log_loss"],
                base["brier"] - new["brier"],
                recl["idi_discrimination_slope_delta"],
                recl["continuous_nri"],
            ]
        )
    vals = np.asarray(arr)
    out = {}
    for i, name in enumerate(
        [
            "delta_auc_vs_clinical",
            "delta_average_precision_vs_clinical",
            "log_loss_reduction_vs_clinical",
            "brier_reduction_vs_clinical",
            "idi_discrimination_slope_delta",
            "continuous_nri",
        ]
    ):
        x = vals[:, i]
        out[f"{name}_boot_mean"] = float(np.mean(x))
        out[f"{name}_ci2.5"] = float(np.quantile(x, 0.025))
        out[f"{name}_ci97.5"] = float(np.quantile(x, 0.975))
        out[f"{name}_p_le_0"] = float(np.mean(x <= 0))
    return out


def plot_gate_weights() -> None:
    """4가지 게이트 함수 w(p)의 모양을 임상확률 p에 대해 그려 PNG로 저장."""
    p = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {
        "no_gate": "#777777",
        "uncertainty_mid": "#4C78A8",
        "soft_high_uncertainty": "#F58518",
        "soft_high_risk": "#54A24B",
    }
    for gate in GATE_DESCRIPTIONS:
        ax.plot(p, gate_weight(p, gate), lw=2.2, color=colors[gate], label=gate)
    ax.set_xlabel("Clinical probability p")
    ax.set_ylabel("AEC multiplier w(p)")
    ax.set_title("Threshold-free soft gates", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soft_gate_weight_functions.png", dpi=220)
    plt.close(fig)


def plot_delta_scatter(paired: pd.DataFrame) -> None:
    """모든 모델에 대해 g1090 OOF delta AUC(x축) vs sdata 외부 delta AUC(y축) 산점도를 게이트별 색으로 그려 PNG로 저장 (내부 개선이 외부에서도 재현되는지 확인)."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    color_map = {
        "no_gate": "#777777",
        "uncertainty_mid": "#4C78A8",
        "soft_high_uncertainty": "#F58518",
        "soft_high_risk": "#54A24B",
    }
    for gate, sub in paired.groupby("gate"):
        ax.scatter(
            sub["delta_auc_vs_clinical_g1090_oof"],
            sub["delta_auc_vs_clinical_sdata_external"],
            s=np.where(sub["female_interaction"], 55, 28),
            alpha=0.78,
            color=color_map[gate],
            label=gate,
        )
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel("g1090 OOF delta AUC vs clinical")
    ax.set_ylabel("sdata external delta AUC vs clinical")
    ax.set_title("Soft-gated AEC models: internal vs external AUC gain", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "soft_gate_auc_delta_scatter.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 딱딱한 "임상 임계값 넘으면 AEC 반영" 규칙 대신, 임상확률에
    따라 부드럽게(threshold-free) AEC 영향력을 조절하는 게이트를 쓰면 어떤 조합이 가장 안정적으로
    이득을 주는가? — 9개 특징그룹 x 4개 게이트 x 2개(여성상호작용 유무) = 72개 모델 총점검):

    1. g1090/sdata를 로드하고 임상점수를 준비한다.
    2. 9개 특징그룹(단일 특징 6개 + 조합 2개 + 8개 전부) x 4개 게이트 함수(no_gate/불확실성중간/
       고위험쪽불확실성/고위험단조증가) x 여성상호작용 유무(2) = 72개 모델 각각에 대해:
       - crossfit_soft_gate로 g1090 5-fold(내부 3-fold로 lambda 선택) OOF 임상/결합 점수를 만들고,
       - fit_external_soft_gate로 g1090 전체 학습 모델의 sdata 외부 예측을 구한다.
       - evaluate_model로 임상 단독 대비 AUC/로그손실/IDI/NRI 개선 정도를 계산.
    3. 72개 모델의 성능을 긴 표(long)와, train/외부를 나란히 비교하는 넓은 표(paired)로 각각 CSV로 저장.
    4. train과 외부 모두에서 개선폭이 양수인 모델들, 그리고 상위 10개 모델을 추려 부트스트랩으로
       개선폭 신뢰구간을 재확인.
    5. 게이트 함수 모양과, 모든 모델의 train-vs-외부 delta AUC 산점도를 그려 저장 (내부 개선이
       외부에서 재현되는 모델을 찾기 위함).
    6. 방법론(수식, 게이트 정의, lambda 선택 방식), 임상 기준선, 상위 모델들, "양쪽 모두 양의
       개선"을 보인 모델 목록을 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    clinical_scores(data)
    g = data["g1090"]
    s = data["sdata"]

    clinical_rows = []
    clinical_rows.append({"dataset": "g1090_oof", "model": "clinical_only", **metric_dict(g["y"], g["clinical_score_oof"])})
    clinical_rows.append(
        {"dataset": "sdata_external", "model": "clinical_only", **metric_dict(s["y"], s["clinical_score_external"])}
    )

    rows = []
    score_store: dict[str, dict[str, np.ndarray]] = {}
    fold_store = []
    inner_store = []
    model_count = 0

    for feature_group, feature_names in FEATURE_GROUPS.items():
        for gate in GATE_DESCRIPTIONS:
            for female_interaction in [False, True]:
                model_count += 1
                model_name = f"{feature_group}__{gate}" + ("__female_interaction" if female_interaction else "")
                clinical_oof, combined_oof, fold_df = crossfit_soft_gate(g, feature_names, gate, female_interaction)
                c_ext, combined_ext, external_lam, inner_df = fit_external_soft_gate(
                    g, s, feature_names, gate, female_interaction
                )
                n_terms = len(feature_names) * (2 if female_interaction else 1)
                fold_df.insert(0, "model", model_name)
                fold_df.insert(1, "feature_group", feature_group)
                fold_df.insert(2, "gate", gate)
                fold_df.insert(3, "female_interaction", female_interaction)
                fold_store.append(fold_df)
                inner_df.insert(0, "model", model_name)
                inner_df.insert(1, "feature_group", feature_group)
                inner_df.insert(2, "gate", gate)
                inner_df.insert(3, "female_interaction", female_interaction)
                inner_store.append(inner_df)

                rows.append(
                    evaluate_model(
                        model_name,
                        g["y"],
                        clinical_oof,
                        combined_oof,
                        "g1090_oof",
                        feature_group,
                        gate,
                        female_interaction,
                        selected_lambda=float(fold_df["selected_lambda"].median()),
                        n_terms=n_terms,
                    )
                )
                rows.append(
                    evaluate_model(
                        model_name,
                        s["y"],
                        c_ext,
                        combined_ext,
                        "sdata_external",
                        feature_group,
                        gate,
                        female_interaction,
                        selected_lambda=external_lam,
                        n_terms=n_terms,
                    )
                )
                score_store[model_name] = {
                    "g_clinical": clinical_oof,
                    "g_combined": combined_oof,
                    "s_clinical": c_ext,
                    "s_combined": combined_ext,
                }
                print(f"[{model_count:02d}] {model_name}", flush=True)

    perf = pd.DataFrame(clinical_rows + rows)
    perf.to_csv(OUT_DIR / "soft_gate_model_performance_long.csv", index=False)
    pd.concat(fold_store, ignore_index=True).to_csv(OUT_DIR / "soft_gate_outer_fold_lambdas.csv", index=False)
    pd.concat(inner_store, ignore_index=True).to_csv(OUT_DIR / "soft_gate_external_lambda_cv.csv", index=False)

    model_perf = perf[perf["model"].ne("clinical_only")].copy()
    paired = model_perf.pivot_table(
        index=["model", "feature_group", "gate", "female_interaction", "n_terms"],
        columns="dataset",
        values=[
            "auc",
            "average_precision",
            "log_loss",
            "brier",
            "delta_auc_vs_clinical",
            "delta_average_precision_vs_clinical",
            "log_loss_reduction_vs_clinical",
            "brier_reduction_vs_clinical",
            "idi_discrimination_slope_delta",
            "continuous_nri",
        ],
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{dataset}" for metric, dataset in paired.columns]
    paired = paired.reset_index()
    paired = paired.sort_values(
        ["delta_auc_vs_clinical_g1090_oof", "delta_auc_vs_clinical_sdata_external"],
        ascending=[False, False],
    )
    paired.to_csv(OUT_DIR / "soft_gate_model_performance_paired.csv", index=False)

    # Bootstrap the most relevant models: best by internal OOF and best with positive internal+external gains.
    selected_names = set()
    selected_names.update(paired.head(10)["model"].tolist())
    both_positive = paired[
        (paired["delta_auc_vs_clinical_g1090_oof"] > 0) & (paired["delta_auc_vs_clinical_sdata_external"] > 0)
    ].head(10)
    selected_names.update(both_positive["model"].tolist())
    boot_rows = []
    for model_name in sorted(selected_names):
        st = score_store[model_name]
        meta = paired[paired["model"].eq(model_name)].iloc[0].to_dict()
        boot_rows.append(
            {
                "dataset": "g1090_oof",
                "model": model_name,
                **{k: meta[k] for k in ["feature_group", "gate", "female_interaction", "n_terms"]},
                **bootstrap_delta_ci(g["y"], st["g_clinical"], st["g_combined"]),
            }
        )
        boot_rows.append(
            {
                "dataset": "sdata_external",
                "model": model_name,
                **{k: meta[k] for k in ["feature_group", "gate", "female_interaction", "n_terms"]},
                **bootstrap_delta_ci(s["y"], st["s_clinical"], st["s_combined"]),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT_DIR / "soft_gate_selected_bootstrap_delta_ci.csv", index=False)

    plot_gate_weights()
    plot_delta_scatter(paired)

    top_oof = paired.head(15)
    top_external = paired.sort_values("delta_auc_vs_clinical_sdata_external", ascending=False).head(15)
    both = paired[
        (paired["delta_auc_vs_clinical_g1090_oof"] > 0) & (paired["delta_auc_vs_clinical_sdata_external"] > 0)
    ].copy()

    summary = {
        "method": {
            "primary_equation": "logit(P(low SMI)) = clinical_logit + alpha + beta * standardized[w(p_clinical) * standardized(AEC_feature)]",
            "female_interaction": "When enabled, an additional female-specific gated AEC term is included.",
            "lambda_selection": f"Nested {INNER_SPLITS}-fold CV inside each g1090 outer fold; lambda grid={LAMBDAS}; selected by inner log-loss.",
            "gates": GATE_DESCRIPTIONS,
            "feature_groups": FEATURE_GROUPS,
        },
        "clinical_reference": clinical_rows,
        "top_by_g1090_oof_delta_auc": top_oof.to_dict(orient="records"),
        "top_by_sdata_external_delta_auc": top_external.to_dict(orient="records"),
        "models_positive_in_both_g1090_oof_and_sdata_external": both.to_dict(orient="records"),
        "outputs": {
            "long_performance": str(OUT_DIR / "soft_gate_model_performance_long.csv"),
            "paired_performance": str(OUT_DIR / "soft_gate_model_performance_paired.csv"),
            "bootstrap_ci": str(OUT_DIR / "soft_gate_selected_bootstrap_delta_ci.csv"),
            "gate_plot": str(OUT_DIR / "soft_gate_weight_functions.png"),
            "auc_delta_scatter": str(OUT_DIR / "soft_gate_auc_delta_scatter.png"),
        },
    }
    (OUT_DIR / "soft_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nClinical reference")
    print(pd.DataFrame(clinical_rows).to_string(index=False))
    print("\nTop by g1090 OOF delta AUC")
    print(
        top_oof[
            [
                "model",
                "delta_auc_vs_clinical_g1090_oof",
                "delta_auc_vs_clinical_sdata_external",
                "idi_discrimination_slope_delta_g1090_oof",
                "idi_discrimination_slope_delta_sdata_external",
                "log_loss_reduction_vs_clinical_g1090_oof",
                "log_loss_reduction_vs_clinical_sdata_external",
            ]
        ].to_string(index=False)
    )
    print("\nPositive in both g1090 OOF and sdata external")
    if both.empty:
        print("None")
    else:
        print(
            both[
                [
                    "model",
                    "delta_auc_vs_clinical_g1090_oof",
                    "delta_auc_vs_clinical_sdata_external",
                    "idi_discrimination_slope_delta_g1090_oof",
                    "idi_discrimination_slope_delta_sdata_external",
                    "continuous_nri_g1090_oof",
                    "continuous_nri_sdata_external",
                ]
            ].to_string(index=False)
        )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
