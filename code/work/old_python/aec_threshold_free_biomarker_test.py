from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    clinical_estimator,
    clinical_matrix,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    zfit_apply,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_threshold_free_biomarker_test"
SEED = 20260630
TARGETS = np.round(np.arange(0.75, 0.951, 0.01), 2)


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 행정규화 곡선과 저근감소증 라벨·성별을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "norm": norm, "y": y, "sex": sex}


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
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


def deesc_metrics(y: np.ndarray, clinical: np.ndarray, gate: np.ndarray, threshold: float) -> dict:
    """게이트 규칙의 유지/하향조정군 통계, 민감도손실/특이도이득/순이득(net_rate_gain), Fisher 오즈비·p값을 계산."""
    clinical_pos = clinical >= threshold
    final_pos = clinical_pos & (gate >= threshold)
    deesc = clinical_pos & ~final_pos
    keep = final_pos
    base = counts(y, clinical_pos)
    rule = counts(y, final_pos)
    keep_events = int(np.sum(y[keep] == 1))
    keep_nonevents = int(np.sum(y[keep] == 0))
    de_events = int(np.sum(y[deesc] == 1))
    de_nonevents = int(np.sum(y[deesc] == 0))
    if np.sum(keep) and np.sum(deesc):
        orr, fisher_p = stats.fisher_exact([[keep_events, keep_nonevents], [de_events, de_nonevents]])
    else:
        orr, fisher_p = np.nan, np.nan
    return {
        "clinical_sensitivity": base["sensitivity"],
        "clinical_specificity": base["specificity"],
        "rule_sensitivity": rule["sensitivity"],
        "rule_specificity": rule["specificity"],
        "clinical_positive_n": int(np.sum(clinical_pos)),
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_events,
        "deesc_prevalence": de_events / int(np.sum(deesc)) if np.sum(deesc) else np.nan,
        "fp_removed": de_nonevents,
        "tp_lost": de_events,
        "specificity_gain": rule["specificity"] - base["specificity"],
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "net_rate_gain": (rule["specificity"] - base["specificity"]) - (base["sensitivity"] - rule["sensitivity"]),
        "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
        "or_keep_vs_deesc": float(orr) if np.isfinite(orr) else np.nan,
    }


def visual_score(x: np.ndarray, mid: tuple[int, int], tail: tuple[int, int]) -> np.ndarray:
    """지정된 후반 구간 평균에서 중간 구간 평균을 빼 "후반 반등 강도" 점수를 계산."""
    m0, m1 = mid
    t0, t1 = tail
    return x[:, t0 - 1 : t1].mean(axis=1) - x[:, m0 - 1 : m1].mean(axis=1)


def zfit(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test를 함께 z-표준화."""
    mu = float(np.mean(train))
    sd = float(np.std(train)) or 1.0
    return (train - mu) / sd, (test - mu) / sd


def boundary_weight(clinical: np.ndarray, threshold: float, width: float, center: float) -> np.ndarray:
    """임상점수와 임계값의 거리(중심 이동 가능)에 가우시안 커널을 적용해 게이트 가중치를 계산."""
    return np.exp(-0.5 * (((clinical - threshold) - center) / width) ** 2)


def gate_score(clinical: np.ndarray, visual: np.ndarray, threshold: float, width: float, center: float, lam: float) -> np.ndarray:
    """임상점수에 boundary_weight로 가중된 시각적 점수를 람다 배율로 더해 게이트 점수를 만듦."""
    return clinical + lam * boundary_weight(clinical, threshold, width, center) * visual


def eval_curve(y: np.ndarray, clinical: np.ndarray, visual: np.ndarray, thresholds: dict[float, float], cfg: dict) -> pd.DataFrame:
    """민감도 목표(0.75~0.95, 0.01 간격)마다 대응 임계값에서 게이트 성능을 계산해, 임계값에 따른 성능 곡선을 표로 만듦."""
    rows = []
    for target, threshold in thresholds.items():
        gate = gate_score(clinical, visual, threshold, cfg["width"], cfg["center"], cfg["lambda"])
        rows.append({"target_sensitivity": target, "threshold_z": threshold, **deesc_metrics(y, clinical, gate, threshold)})
    return pd.DataFrame(rows)


def integrated_summary(curve: pd.DataFrame) -> dict:
    """여러 임계값에 걸친 성능 곡선을 하나의 요약(평균/최소/최대 지표)으로 적분해 임계값 선택에 의존하지 않는 종합 지표를 만듦."""
    return {
        "mean_specificity_gain": float(curve["specificity_gain"].mean()),
        "mean_sensitivity_loss": float(curve["sensitivity_loss"].mean()),
        "mean_net_rate_gain": float(curve["net_rate_gain"].mean()),
        "mean_deesc_n": float(curve["deesc_n"].mean()),
        "mean_deesc_events": float(curve["deesc_events"].mean()),
        "mean_deesc_prevalence": float(curve["deesc_prevalence"].mean()),
        "mean_fp_removed": float(curve["fp_removed"].mean()),
        "mean_tp_lost": float(curve["tp_lost"].mean()),
        "min_specificity_gain": float(curve["specificity_gain"].min()),
        "max_sensitivity_loss": float(curve["sensitivity_loss"].max()),
    }


def bootstrap_integrated(y: np.ndarray, clinical: np.ndarray, visual: np.ndarray, thresholds: dict[float, float], cfg: dict, n_boot: int = 2000) -> pd.DataFrame:
    """임계값-무관 종합 지표(integrated_summary)를 부트스트랩 재표본추출로 반복 계산해 각 지표의 신뢰구간과 단측 p값(해당하는 지표만)을 추정."""
    rng = np.random.default_rng(SEED + 1)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        curve = eval_curve(yy, clinical[idx], visual[idx], thresholds, cfg)
        vals.append(integrated_summary(curve))
    boot = pd.DataFrame(vals)
    rows = []
    for metric in boot.columns:
        x = boot[metric].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(x)),
                "ci2.5": float(np.quantile(x, 0.025)),
                "ci97.5": float(np.quantile(x, 0.975)),
                "p_le_0": float((np.sum(x <= 0) + 1) / (len(x) + 1)) if metric in {"mean_specificity_gain", "mean_net_rate_gain"} else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stratified_permutation_p(
    y: np.ndarray,
    clinical: np.ndarray,
    visual: np.ndarray,
    thresholds: dict[float, float],
    cfg: dict,
    observed: dict,
    n_perm: int = 3000,
) -> dict:
    """시각적 점수를 임상점수 10분위 안에서만 무작위로 섞어(임상점수와의 대략적 관계는 유지) 순열분포를
    만들고, 관측된 종합 특이도이득/순이득이 우연히 나올 확률을 순열검정으로 계산."""
    rng = np.random.default_rng(SEED + 2)
    # Preserve rough clinical-score relation by permuting AEC feature within clinical deciles.
    bins = pd.qcut(pd.Series(clinical), q=10, labels=False, duplicates="drop").to_numpy()
    groups = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    vals = {k: [] for k in ["mean_specificity_gain", "mean_net_rate_gain"]}
    for _ in range(n_perm):
        vp = visual.copy()
        for idx in groups:
            vp[idx] = rng.permutation(vp[idx])
        summ = integrated_summary(eval_curve(y, clinical, vp, thresholds, cfg))
        for k in vals:
            vals[k].append(summ[k])
    out = {}
    for k, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[f"perm_p_{k}_ge_observed"] = float((np.sum(a >= observed[k]) + 1) / (len(a) + 1))
        out[f"perm_null_{k}_mean"] = float(np.mean(a))
        out[f"perm_null_{k}_ci97.5"] = float(np.quantile(a, 0.975))
    return out


def logistic_lrt(y: np.ndarray, clinical: np.ndarray, visual: np.ndarray) -> dict:
    """임상점수만 넣은 모델과 시각적 점수까지 넣은 모델의 우도비검정(LRT)으로, 시각적 점수의 계수·오즈비·Wald p값과 카이제곱·p값을 계산."""
    import statsmodels.api as sm

    x0 = sm.add_constant(pd.DataFrame({"clinical": clinical}), has_constant="add")
    x1 = sm.add_constant(pd.DataFrame({"clinical": clinical, "visual": visual}), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    lrt = 2 * (m1.llf - m0.llf)
    return {
        "visual_coef": float(m1.params["visual"]),
        "visual_or_per_1sd": float(np.exp(m1.params["visual"])),
        "visual_wald_p": float(m1.pvalues["visual"]),
        "lrt_chi2_add_visual": float(lrt),
        "lrt_p_add_visual": float(stats.chi2.sf(lrt, 1)),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 특정 하나의 임계값에 의존하지 않고, 75~95% 민감도 구간
    전체에 걸쳐 "평균적으로" 게이트가 이득을 주는지 확인하면 — 즉 임계값 선택 자체를 없애버리면
    — 여전히 유의미한 신호인가?):

    1. g1090/sdata를 로드하고 임상 단독 모델의 OOF/외부 점수를 표준화, 75~95%(1%씩) 21개 민감도
       목표에 대응하는 임상 임계값들을 미리 계산해둔다.
    2. 두 가지 시각적 특징 설정(사전지정판 visual_strict_a_priori, 탐색적 상위판
       visual_upper_exploratory)을 각각 계산해 표준화·부호 고정.
    3. 각 설정마다 eval_curve로 21개 임계값 전체에서의 성능 곡선을 구하고, integrated_summary로
       "임계값에 걸친 평균 특이도이득/민감도손실/순이득" 등 하나의 종합 지표로 압축.
    4. 외부 데이터에 대해서는 stratified_permutation_p로 (임상점수 10분위 내에서 시각점수를
       섞는) 순열검정 p값과, bootstrap_integrated로 종합지표의 신뢰구간을 추가로 계산.
    5. logistic_lrt로 임상점수 대비 시각적 점수의 조건부 연관성(LRT)도 함께 검정.
    6. 두 특징의 임계값별 곡선·종합요약·외부 부트스트랩·조건부 로지스틱 결과를 각각 CSV로 저장.
    7. 외부 데이터에서 두 특징의 "민감도손실 vs 특이도이득" 궤적을 임계값 라벨과 함께 그려 PNG로 저장.
    8. 방법론(종합지표 정의, 순열검정 방식)을 JSON으로 저장하고 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    xclin_g, xclin_s, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"], 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_g, g["y"], xclin_s, folds)
    c_g, c_s, c_mu, c_sd = zfit_apply(clinical_oof, clinical_ext)
    thresholds = {
        float(target): float((threshold_for_min_sensitivity(g["y"], clinical_oof, float(target)) - c_mu) / c_sd)
        for target in TARGETS
    }

    configs = {
        "visual_strict_a_priori": {
            "mid": (63, 92),
            "tail": (112, 128),
            "width": 0.40,
            "center": -0.20,
            "lambda": 0.70,
            "role": "a_priori_interpretable_from_curve_contrast",
        },
        "visual_upper_exploratory": {
            "mid": (55, 84),
            "tail": (88, 116),
            "width": 0.40,
            "center": 0.20,
            "lambda": 0.70,
            "role": "exploratory_high_specificity_frontier_candidate",
        },
    }

    curve_rows = []
    summary_rows = []
    boot_rows = []
    logit_rows = []
    for name, cfg in configs.items():
        vg_raw = visual_score(g["norm"], cfg["mid"], cfg["tail"])
        vs_raw = visual_score(s["norm"], cfg["mid"], cfg["tail"])
        vg, vs = zfit(vg_raw, vs_raw)
        if np.corrcoef(vg, g["y"])[0, 1] < 0:
            vg = -vg
            vs = -vs

        for dataset, y, clinical, visual in [
            ("g1090_oof", g["y"], c_g, vg),
            ("sdata_external", s["y"], c_s, vs),
        ]:
            curve = eval_curve(y, clinical, visual, thresholds, cfg)
            curve["feature"] = name
            curve["dataset"] = dataset
            curve_rows.append(curve)
            summ = {"feature": name, "dataset": dataset, **integrated_summary(curve)}
            if dataset == "sdata_external":
                summ.update(stratified_permutation_p(y, clinical, visual, thresholds, cfg, summ))
                boot = bootstrap_integrated(y, clinical, visual, thresholds, cfg)
                boot["feature"] = name
                boot_rows.append(boot)
            summary_rows.append(summ)
            logit_rows.append({"feature": name, "dataset": dataset, **logistic_lrt(y, clinical, visual)})

    curve_df = pd.concat(curve_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    boot_df = pd.concat(boot_rows, ignore_index=True)
    logit_df = pd.DataFrame(logit_rows)
    curve_df.to_csv(OUT_DIR / "threshold_free_curve_long.csv", index=False)
    summary_df.to_csv(OUT_DIR / "threshold_free_integrated_summary.csv", index=False)
    boot_df.to_csv(OUT_DIR / "threshold_free_external_bootstrap.csv", index=False)
    logit_df.to_csv(OUT_DIR / "conditional_logistic_feature_pvalues.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for (feature, dataset), sub in curve_df.groupby(["feature", "dataset"]):
        if dataset != "sdata_external":
            continue
        label = feature.replace("visual_", "").replace("_", " ")
        ax.plot(sub["sensitivity_loss"], sub["specificity_gain"], marker="o", lw=2, label=label)
        for _, r in sub[sub["target_sensitivity"].isin([0.75, 0.80, 0.85, 0.90, 0.95])].iterrows():
            ax.annotate(f'{int(r["target_sensitivity"] * 100)}', (r["sensitivity_loss"], r["specificity_gain"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#666666", ls="--", lw=1)
    ax.axvline(0, color="#666666", ls="--", lw=1)
    ax.set_xlabel("sdata sensitivity loss")
    ax.set_ylabel("sdata specificity gain")
    ax.set_title("Threshold-free de-escalation curve across clinical-positive thresholds", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "threshold_free_sdata_tradeoff_curve.png", dpi=220)
    plt.close(fig)

    meta = {
        "targets": TARGETS.tolist(),
        "primary_endpoint": "mean_specificity_gain across clinical sensitivity targets 75%-95%",
        "secondary_endpoint": "mean_net_rate_gain = mean(specificity_gain - sensitivity_loss)",
        "permutation": "sdata visual score permuted within clinical-score deciles; p is one-sided P(null >= observed).",
        "outputs": {
            "curve": str(OUT_DIR / "threshold_free_curve_long.csv"),
            "summary": str(OUT_DIR / "threshold_free_integrated_summary.csv"),
            "bootstrap": str(OUT_DIR / "threshold_free_external_bootstrap.csv"),
            "conditional_logistic": str(OUT_DIR / "conditional_logistic_feature_pvalues.csv"),
            "plot": str(OUT_DIR / "threshold_free_sdata_tradeoff_curve.png"),
        },
    }
    (OUT_DIR / "threshold_free_biomarker_test_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nIntegrated summary")
    print(summary_df.to_string(index=False))
    print("\nConditional logistic p-values")
    print(logit_df.to_string(index=False))
    print("\nExternal bootstrap")
    print(boot_df.to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
