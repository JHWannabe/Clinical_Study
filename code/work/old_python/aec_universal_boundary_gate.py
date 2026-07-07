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
    SEED,
    aec_estimator,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    make_folds,
    oof_and_external,
    threshold_youden,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_universal_boundary_gate"


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """목표 민감도(target) 이상을 유지하면서 특이도가 가장 높은 임계값을 찾음 (해당하는 값이 없으면 분위수로 근사)."""
    best = None
    for th in np.unique(score):
        m = binary_metrics(y, score, float(th))
        if m["sensitivity"] >= target and (best is None or m["specificity"] > best[1]):
            best = (float(th), m["specificity"])
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return best[0]


def reclassification(y: np.ndarray, clinical_pos: np.ndarray, gate_pos: np.ndarray) -> dict:
    """임상양성/음성 x 게이트양성/음성 4개 그룹의 표본수·이벤트·유병률을 계산하고, 임상양성군 내
    게이트양성 vs 음성의 Fisher 오즈비·p값, 그리고 게이트 적용 시 순증감한 FP/TP 개수까지 정리."""
    cp = clinical_pos.astype(bool)
    gp = gate_pos.astype(bool)
    rows = {}
    for name, mask in {
        "clinical_pos_gate_pos": cp & gp,
        "clinical_pos_gate_neg": cp & ~gp,
        "clinical_neg_gate_pos": ~cp & gp,
        "clinical_neg_gate_neg": ~cp & ~gp,
    }.items():
        n = int(np.sum(mask))
        e = int(np.sum(y[mask] == 1))
        rows[f"{name}_n"] = n
        rows[f"{name}_events"] = e
        rows[f"{name}_prevalence"] = e / n if n else np.nan
    a = rows["clinical_pos_gate_pos_events"]
    b = rows["clinical_pos_gate_pos_n"] - a
    c = rows["clinical_pos_gate_neg_events"]
    d = rows["clinical_pos_gate_neg_n"] - c
    if rows["clinical_pos_gate_pos_n"] and rows["clinical_pos_gate_neg_n"]:
        orr, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    else:
        orr, p = np.nan, np.nan
    rows["within_clinical_pos_or_gate_pos_vs_neg"] = float(orr) if np.isfinite(orr) else np.nan
    rows["within_clinical_pos_fisher_p"] = float(p) if np.isfinite(p) else np.nan
    rows["false_positives_removed"] = int(np.sum(cp & ~gp & (y == 0)))
    rows["true_positives_lost"] = int(np.sum(cp & ~gp & (y == 1)))
    rows["false_positives_added"] = int(np.sum(~cp & gp & (y == 0)))
    rows["true_positives_gained"] = int(np.sum(~cp & gp & (y == 1)))
    return rows


def paired_rule_metrics(y: np.ndarray, clinical_score: np.ndarray, gate_score: np.ndarray, threshold: float) -> dict:
    """같은 임계값에서 임상 단독 vs 게이트 점수의 지표를 비교해 민감도손실/특이도이득/PPV·NPV이득과 재분류 통계까지 한 번에 계산."""
    clinical_pos = clinical_score >= threshold
    gate_pos = gate_score >= threshold
    clinical = binary_metrics(y, clinical_score, threshold)
    gated = binary_metrics(y, gate_score, threshold)
    out = {
        **{f"clinical_{k}": v for k, v in clinical.items()},
        **{f"gated_{k}": v for k, v in gated.items()},
        "sensitivity_loss": clinical["sensitivity"] - gated["sensitivity"],
        "specificity_gain": gated["specificity"] - clinical["specificity"],
        "ppv_gain": gated["ppv"] - clinical["ppv"],
        "npv_gain": gated["npv"] - clinical["npv"],
    }
    out.update(reclassification(y, clinical_pos, gate_pos))
    return out


def bootstrap_external(y: np.ndarray, clinical_score: np.ndarray, gate_score: np.ndarray, threshold: float, n_boot: int = 3000) -> dict:
    """게이트 규칙의 민감도손실/특이도이득/PPV이득/순FP·TP변화/하향조정군 유병률/오즈비를 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 77)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        m = paired_rule_metrics(yy, clinical_score[idx], gate_score[idx], threshold)
        vals.append(
            [
                m["sensitivity_loss"],
                m["specificity_gain"],
                m["ppv_gain"],
                m["false_positives_removed"] - m["false_positives_added"],
                m["true_positives_lost"] - m["true_positives_gained"],
                m["clinical_pos_gate_neg_prevalence"],
                m["within_clinical_pos_or_gate_pos_vs_neg"],
            ]
        )
    arr = np.asarray(vals)
    names = [
        "sensitivity_loss",
        "specificity_gain",
        "ppv_gain",
        "net_false_positives_removed",
        "net_true_positives_lost",
        "clinical_pos_gate_neg_prevalence",
        "within_clinical_pos_or",
    ]
    out = {}
    for i, name in enumerate(names):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        if len(x) == 0:
            out[f"{name}_mean"] = np.nan
            out[f"{name}_ci2.5"] = np.nan
            out[f"{name}_ci97.5"] = np.nan
        else:
            out[f"{name}_mean"] = float(np.mean(x))
            out[f"{name}_ci2.5"] = float(np.quantile(x, 0.025))
            out[f"{name}_ci97.5"] = float(np.quantile(x, 0.975))
    return out


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 6/29에 만들었던 "여성에게만 적용되는 임상경계 가우시안
    게이트"에서 성별 조건을 떼어내고 전체 환자에게 적용해도, 성별 제한판만큼의 효과가 나오는가?
    — "성별 중립" 버전과 "여성 한정" 기준판을 나란히 비교):

    1. g1090/sdata를 로드해 임상 단독 모델과 AEC 단독(SVM) 모델의 OOF/외부 점수를 표준화해 준비.
    2. 임상 임계값 4종(Youden/민감도85·90·95%) x 게이트 폭 5종 x 람다(AEC 반영강도) 7종 x
       적용대상(전체환자/여성만-기준/남성만-확인) 3종 x 게이트형태(가우시안/하드) 2종 = 총 2100개
       조합에 대해 paired_rule_metrics로 임상 단독 대비 성능을 train/외부 양쪽에서 계산.
    3. "전체환자 적용" 결과 중, train·외부 모두 하향조정 인원 10명 이상·유병률 15% 이하·민감도손실
       4% 이하인 "성별 중립 강한 후보"를 뽑아 정렬.
    4. 외부 데이터 기준 상위 80개 조합도 별도로 저장.
    5. 강한 후보 1위, 더 엄격한 조건(유병률 6%·민감도손실 1.5% 이하)의 고포집 후보, 그리고 기존
       "여성 한정" 기준판(폭0.75, 람다0.25) 세 가지를 뽑아 부트스트랩으로 외부 신뢰구간을 재확인.
    6. "전체환자" vs "여성만" 게이트의 외부 특이도이득-민감도손실 산점도를 그려 비교.
    7. 방법론, 강한 후보 목록, 외부 상위 결과, 부트스트랩 결과를 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(ytr, 5)
    clinical_oof, clinical_test = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    aec_oof, aec_test = oof_and_external(
        lambda seed: aec_estimator(train["aec"].shape[1], seed),
        train["aec"],
        ytr,
        test["aec"],
        folds,
    )
    c_z, c_te_z, _, _ = zfit_apply(clinical_oof, clinical_test)
    a_z, a_te_z, _, _ = zfit_apply(aec_oof, aec_test)
    if np.corrcoef(a_z, ytr)[0, 1] < 0:
        a_z = -a_z
        a_te_z = -a_te_z

    female_tr = train["sex"] == "F"
    female_te = test["sex"] == "F"

    clinical_thresholds_raw = {
        "youden": threshold_youden(ytr, clinical_oof),
        "sens85": threshold_for_min_sensitivity(ytr, clinical_oof, 0.85),
        "sens90": threshold_for_min_sensitivity(ytr, clinical_oof, 0.90),
        "sens95": threshold_for_min_sensitivity(ytr, clinical_oof, 0.95),
    }
    c_mu = float(np.mean(clinical_oof))
    c_sd = float(np.std(clinical_oof)) or 1.0
    clinical_thresholds = {k: (v - c_mu) / c_sd for k, v in clinical_thresholds_raw.items()}

    rows = []
    lambdas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
    widths = [0.40, 0.50, 0.75, 1.00, 1.25]
    gate_masks = {
        "all_patients": (np.ones_like(a_z), np.ones_like(a_te_z)),
        "female_only_reference": (female_tr.astype(float), female_te.astype(float)),
        "male_only_check": ((~female_tr).astype(float), (~female_te).astype(float)),
    }
    for clinical_rule, t in clinical_thresholds.items():
        for width in widths:
            boundary_tr = np.exp(-0.5 * ((c_z - t) / width) ** 2)
            boundary_te = np.exp(-0.5 * ((c_te_z - t) / width) ** 2)
            hard_tr = (np.abs(c_z - t) <= width).astype(float)
            hard_te = (np.abs(c_te_z - t) <= width).astype(float)
            for lam in lambdas:
                for mask_name, (mask_tr, mask_te) in gate_masks.items():
                    for gate_shape, wtr, wte in [
                        ("gaussian_boundary", boundary_tr, boundary_te),
                        ("hard_boundary", hard_tr, hard_te),
                    ]:
                        score_tr = c_z + lam * mask_tr * wtr * a_z
                        score_te = c_te_z + lam * mask_te * wte * a_te_z
                        for dataset, y, cscore, gscore in [
                            ("g1090_oof", ytr, c_z, score_tr),
                            ("sdata_external", yte, c_te_z, score_te),
                        ]:
                            row = {
                                "dataset": dataset,
                                "clinical_rule": clinical_rule,
                                "threshold_z": float(t),
                                "gate_population": mask_name,
                                "gate_shape": gate_shape,
                                "width": width,
                                "lambda": lam,
                            }
                            row.update(paired_rule_metrics(y, cscore, gscore, float(t)))
                            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "boundary_gate_grid_long.csv", index=False)

    paired = out.pivot_table(
        index=["clinical_rule", "gate_population", "gate_shape", "width", "lambda"],
        columns="dataset",
        values=[
            "clinical_pos_gate_neg_n",
            "clinical_pos_gate_neg_events",
            "clinical_pos_gate_neg_prevalence",
            "false_positives_removed",
            "true_positives_lost",
            "false_positives_added",
            "true_positives_gained",
            "specificity_gain",
            "sensitivity_loss",
            "ppv_gain",
            "within_clinical_pos_or_gate_pos_vs_neg",
            "within_clinical_pos_fisher_p",
        ],
        aggfunc="first",
    )
    paired.columns = [f"{m}_{d}" for m, d in paired.columns]
    paired = paired.reset_index()
    paired.to_csv(OUT_DIR / "boundary_gate_grid_paired.csv", index=False)

    universal = paired[paired["gate_population"].eq("all_patients")].copy()
    universal_strong = universal[
        (universal["clinical_pos_gate_neg_n_g1090_oof"] >= 10)
        & (universal["clinical_pos_gate_neg_n_sdata_external"] >= 10)
        & (universal["clinical_pos_gate_neg_prevalence_g1090_oof"] <= 0.15)
        & (universal["clinical_pos_gate_neg_prevalence_sdata_external"] <= 0.15)
        & (universal["sensitivity_loss_g1090_oof"] <= 0.04)
        & (universal["sensitivity_loss_sdata_external"] <= 0.04)
    ].sort_values(
        [
            "clinical_pos_gate_neg_prevalence_sdata_external",
            "clinical_pos_gate_neg_prevalence_g1090_oof",
            "specificity_gain_sdata_external",
        ],
        ascending=[True, True, False],
    )
    universal_strong.to_csv(OUT_DIR / "sex_neutral_boundary_strong_candidates.csv", index=False)

    external_top = paired[paired["dataset" if "dataset" in paired.columns else "clinical_rule"].notna()].copy()
    external_top = paired.sort_values(
        ["clinical_pos_gate_neg_prevalence_sdata_external", "specificity_gain_sdata_external"],
        ascending=[True, False],
    )
    external_top.head(80).to_csv(OUT_DIR / "boundary_gate_external_top80.csv", index=False)

    selected_rows = []
    if not universal_strong.empty:
        selected_rows.append(universal_strong.iloc[0])
        high_coverage = universal_strong[
            (universal_strong["clinical_pos_gate_neg_prevalence_sdata_external"] <= 0.06)
            & (universal_strong["sensitivity_loss_sdata_external"] <= 0.015)
        ].sort_values(["false_positives_removed_sdata_external", "specificity_gain_sdata_external"], ascending=False)
        if not high_coverage.empty:
            selected_rows.append(high_coverage.iloc[0])
    ref = paired[
        (paired["gate_population"].eq("female_only_reference"))
        & (paired["clinical_rule"].eq("youden"))
        & (paired["gate_shape"].eq("gaussian_boundary"))
        & (paired["width"].eq(0.75))
        & (paired["lambda"].eq(0.25))
    ]
    if not ref.empty:
        selected_rows.append(ref.iloc[0])
    boot_rows = []
    long_ext = out[out["dataset"].eq("sdata_external")]
    for r in selected_rows:
        match = long_ext[
            (long_ext["clinical_rule"].eq(r["clinical_rule"]))
            & (long_ext["gate_population"].eq(r["gate_population"]))
            & (long_ext["gate_shape"].eq(r["gate_shape"]))
            & (long_ext["width"].eq(r["width"]))
            & (long_ext["lambda"].eq(r["lambda"]))
        ].iloc[0]
        t = float(match["threshold_z"])
        width = float(match["width"])
        lam = float(match["lambda"])
        if match["gate_shape"] == "gaussian_boundary":
            wte = np.exp(-0.5 * ((c_te_z - t) / width) ** 2)
        else:
            wte = (np.abs(c_te_z - t) <= width).astype(float)
        if match["gate_population"] == "all_patients":
            mask = np.ones_like(a_te_z)
        elif match["gate_population"] == "female_only_reference":
            mask = female_te.astype(float)
        else:
            mask = (~female_te).astype(float)
        gate_score = c_te_z + lam * mask * wte * a_te_z
        boot_rows.append(
            {
                "clinical_rule": match["clinical_rule"],
                "gate_population": match["gate_population"],
                "gate_shape": match["gate_shape"],
                "width": width,
                "lambda": lam,
                **bootstrap_external(yte, c_te_z, gate_score, t),
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_DIR / "selected_boundary_bootstrap_external.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    plot = paired[paired["gate_population"].isin(["all_patients", "female_only_reference"])].copy()
    colors = {"all_patients": "#4C78A8", "female_only_reference": "#F58518"}
    for pop, sub in plot.groupby("gate_population"):
        ax.scatter(
            sub["specificity_gain_sdata_external"],
            sub["sensitivity_loss_sdata_external"],
            s=28,
            alpha=0.55,
            label=pop,
            color=colors[pop],
        )
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlabel("sdata specificity gain")
    ax.set_ylabel("sdata sensitivity loss")
    ax.set_title("Sex-neutral vs female-boundary AEC gates", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sex_neutral_vs_female_boundary_tradeoff.png", dpi=220)
    plt.close(fig)

    summary = {
        "method": "Legacy AEC-only normalized LinearSVM score gated by clinical boundary weight. Universal candidate removes sex term but keeps clinical-boundary weighting.",
        "clinical_thresholds_z": clinical_thresholds,
        "universal_strong_candidates": universal_strong.head(30).to_dict(orient="records"),
        "external_top": external_top.head(30).to_dict(orient="records"),
        "bootstrap_selected": boot_rows,
        "outputs": {
            "long_grid": str(OUT_DIR / "boundary_gate_grid_long.csv"),
            "paired_grid": str(OUT_DIR / "boundary_gate_grid_paired.csv"),
            "universal_strong": str(OUT_DIR / "sex_neutral_boundary_strong_candidates.csv"),
            "bootstrap": str(OUT_DIR / "selected_boundary_bootstrap_external.csv"),
            "plot": str(OUT_DIR / "sex_neutral_vs_female_boundary_tradeoff.png"),
        },
    }
    (OUT_DIR / "boundary_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "clinical_rule",
        "gate_population",
        "gate_shape",
        "width",
        "lambda",
        "clinical_pos_gate_neg_n_g1090_oof",
        "clinical_pos_gate_neg_events_g1090_oof",
        "clinical_pos_gate_neg_prevalence_g1090_oof",
        "clinical_pos_gate_neg_n_sdata_external",
        "clinical_pos_gate_neg_events_sdata_external",
        "clinical_pos_gate_neg_prevalence_sdata_external",
        "false_positives_removed_sdata_external",
        "true_positives_lost_sdata_external",
        "specificity_gain_sdata_external",
        "sensitivity_loss_sdata_external",
        "within_clinical_pos_or_gate_pos_vs_neg_sdata_external",
        "within_clinical_pos_fisher_p_sdata_external",
    ]
    print("\nSex-neutral strong candidates")
    print(universal_strong[cols].head(25).to_string(index=False) if not universal_strong.empty else "None")
    print("\nExternal top, including female reference")
    print(external_top[cols].head(25).to_string(index=False))
    print("\nBootstrap selected")
    print(boot_df.to_string(index=False) if not boot_df.empty else "None")
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
