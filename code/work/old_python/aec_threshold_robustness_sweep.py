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
    aec_estimator,
    clinical_estimator,
    clinical_matrix,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    zfit_apply,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_threshold_robustness_sweep"
SENS_TARGETS = [0.95, 0.90, 0.85, 0.80, 0.75]
SEED = 20260630


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 행정규화 곡선과 저근감소증 라벨을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "norm": norm, "y": y}


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
    """게이트 규칙(임상양성 중 게이트점수가 임계값 미만이면 하향조정)의 유지/하향조정군 통계, 민감도손실/
    특이도이득/PPV이득, Fisher 오즈비·p값을 모두 계산."""
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
        orr, p = stats.fisher_exact([[keep_events, keep_nonevents], [de_events, de_nonevents]])
    else:
        orr, p = np.nan, np.nan

    return {
        **{f"clinical_{k}": v for k, v in base.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "clinical_positive_n": int(np.sum(clinical_pos)),
        "clinical_positive_events": int(np.sum(y[clinical_pos] == 1)),
        "clinical_positive_event_rate": float(np.mean(y[clinical_pos])) if np.sum(clinical_pos) else np.nan,
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_events,
        "deesc_prevalence": de_events / (de_events + de_nonevents) if de_events + de_nonevents else np.nan,
        "fp_removed": de_nonevents,
        "tp_lost": de_events,
        "specificity_gain": rule["specificity"] - base["specificity"],
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "ppv_gain": rule["ppv"] - base["ppv"],
        "or_keep_vs_deesc": float(orr) if np.isfinite(orr) else np.nan,
        "fisher_p": float(p) if np.isfinite(p) else np.nan,
    }


def visual_score(x: np.ndarray, mid: tuple[int, int], tail: tuple[int, int]) -> np.ndarray:
    """지정된 후반 구간 평균에서 중간 구간 평균을 빼 "후반 반등 강도" 점수를 계산."""
    m0, m1 = mid
    t0, t1 = tail
    return x[:, t0 - 1 : t1].mean(axis=1) - x[:, m0 - 1 : m1].mean(axis=1)


def zfit(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test를 함께 z-표준화."""
    mu = float(np.mean(xtr))
    sd = float(np.std(xtr)) or 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def boundary_weight(clinical: np.ndarray, threshold: float, width: float, center: float) -> np.ndarray:
    """임상점수와 임계값의 거리(중심 이동 가능)에 가우시안 커널을 적용해 게이트 가중치를 계산."""
    return np.exp(-0.5 * (((clinical - threshold) - center) / width) ** 2)


def make_gate_score(clinical: np.ndarray, aec_z: np.ndarray, threshold: float, width: float, center: float, lam: float) -> np.ndarray:
    """임상점수에 boundary_weight로 가중된 AEC점수를 람다 배율로 더해 게이트 점수를 만듦."""
    return clinical + lam * boundary_weight(clinical, threshold, width, center) * aec_z


def bootstrap_external(y: np.ndarray, clinical: np.ndarray, gate: np.ndarray, threshold: float, n_boot: int = 2000) -> pd.DataFrame:
    """게이트 규칙의 하향조정군 통계·민감도손실·특이도이득·PPV이득을 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        m = deesc_metrics(yy, clinical[idx], gate[idx], threshold)
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


def plot_external(df: pd.DataFrame) -> None:
    """민감도 목표(75~95%)를 바꿔가며 3개 모델(시각적 엄격/커버리지, AEC128 SVM)의 외부 특이도이득·민감도손실·하향조정 인원·유병률이 어떻게 변하는지 4패널 그래프로 저장."""
    ext = df[df["dataset"].eq("sdata_external")].copy()
    ext["target_pct"] = (ext["target_sensitivity"] * 100).round().astype(int)
    labels = {
        "visual_strict": "visual strict",
        "visual_coverage": "visual coverage",
        "aec128_svm_practical": "AEC128 SVM",
    }
    colors = {
        "visual_strict": "#4C78A8",
        "visual_coverage": "#F58518",
        "aec128_svm_practical": "#54A24B",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    panels = [
        ("specificity_gain", "Specificity gain", axes[0, 0], "{:.1%}"),
        ("sensitivity_loss", "Sensitivity loss", axes[0, 1], "{:.1%}"),
        ("deesc_n", "De-escalated patients", axes[1, 0], "{:.0f}"),
        ("deesc_prevalence", "Low-SMI rate among de-escalated", axes[1, 1], "{:.1%}"),
    ]
    for metric, title, ax, _fmt in panels:
        for model, sub in ext.groupby("model"):
            sub = sub.sort_values("target_sensitivity", ascending=False)
            ax.plot(sub["target_pct"], sub[metric], marker="o", lw=2, label=labels.get(model, model), color=colors.get(model))
            if metric == "deesc_prevalence":
                for _, r in sub.iterrows():
                    ax.annotate(
                        f'{int(r["deesc_events"])}/{int(r["deesc_n"])}',
                        (r["target_pct"], r[metric]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=8,
                    )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()
    axes[0, 0].axhline(0, color="#666666", ls="--", lw=1)
    axes[0, 1].axhline(0, color="#666666", ls="--", lw=1)
    axes[1, 0].set_xlabel("Clinical-positive threshold chosen for training sensitivity (%)")
    axes[1, 1].set_xlabel("Clinical-positive threshold chosen for training sensitivity (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("External sdata robustness to clinical-positive threshold", x=0.02, y=0.995, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "sdata_threshold_sweep_external.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 앞서 찾은 3가지 게이트 방식(시각적 엄격판/커버리지판/
    AEC128 SVM)이, 임상 민감도 목표를 75%부터 95%까지 다양하게 바꿔도 안정적으로 작동하는가?
    — 파라미터는 고정하고 임계값만 바꿔보는 강건성 점검):

    1. g1090/sdata를 로드하고 임상 단독 모델과 AEC128 단독(SVM) 모델의 OOF/외부 점수를 준비.
    2. 이전 스크립트들에서 찾은 3개 게이트(visual_strict, visual_coverage, aec128_svm_practical)의
       고정 파라미터(구간/폭/중심/람다)로 각각의 z-표준화 점수를 만든다.
    3. 5개 민감도 목표(95/90/85/80/75%)마다 g1090에서 임상 임계값을 새로 계산하고, 그 임계값을
       고정한 채(모델 파라미터는 절대 재조정하지 않음) 3개 게이트의 하향조정 성능을 train/외부에서 계산.
    4. 모델별로 5개 임계값에 걸친 최악의 경우(최대 민감도손실, 최대 하향조정군 유병률, 최소
       특이도이득)를 요약해 "이 모델이 임계값이 바뀌어도 안전한가"를 표로 정리.
    5. 각 (모델, 임계값) 조합의 외부 부트스트랩 신뢰구간도 함께 계산.
    6. 민감도 목표에 따른 4가지 지표(특이도이득/민감도손실/하향조정인원/하향조정군유병률) 변화를
       모델별로 겹쳐 그린 그래프를 저장.
    7. 임계값 선택 정책, 게이트별 파라미터, 강건성 요약을 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    xclin_g, xclin_s, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"], 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_g, g["y"], xclin_s, folds)
    c_g, c_s, c_mu, c_sd = zfit_apply(clinical_oof, clinical_ext)

    aec_oof, aec_ext = oof_and_external(lambda seed: aec_estimator(g["norm"].shape[1], seed), g["norm"], g["y"], s["norm"], folds)
    a_g, a_s, _, _ = zfit_apply(aec_oof, aec_ext)
    if np.corrcoef(a_g, g["y"])[0, 1] < 0:
        a_g = -a_g
        a_s = -a_s

    visual_candidates = {
        "visual_strict": {
            "mid": (63, 92),
            "tail": (112, 128),
            "width": 0.40,
            "center": -0.20,
            "lambda": 0.70,
            "note": "lower coverage, lower observed TP loss at 95% external",
        },
        "visual_coverage": {
            "mid": (63, 92),
            "tail": (112, 128),
            "width": 0.25,
            "center": 0.00,
            "lambda": 0.70,
            "note": "train-selected top coverage candidate",
        },
    }
    score_bank = {
        "aec128_svm_practical": {
            "train": a_g,
            "external": a_s,
            "width": 0.40,
            "center": 0.00,
            "lambda": 0.25,
            "note": "AEC128 normalized LinearSVM practical gate",
        }
    }
    for name, cfg in visual_candidates.items():
        vg = visual_score(g["norm"], cfg["mid"], cfg["tail"])
        vs = visual_score(s["norm"], cfg["mid"], cfg["tail"])
        zg, zs = zfit(vg, vs)
        if np.corrcoef(zg, g["y"])[0, 1] < 0:
            zg = -zg
            zs = -zs
        score_bank[name] = {
            "train": zg,
            "external": zs,
            "width": cfg["width"],
            "center": cfg["center"],
            "lambda": cfg["lambda"],
            "mid_start": cfg["mid"][0],
            "mid_end": cfg["mid"][1],
            "tail_start": cfg["tail"][0],
            "tail_end": cfg["tail"][1],
            "note": cfg["note"],
        }

    thresholds = []
    for target in SENS_TARGETS:
        raw_th = threshold_for_min_sensitivity(g["y"], clinical_oof, target)
        thresholds.append(
            {
                "target_sensitivity": target,
                "threshold_raw": raw_th,
                "threshold_z": (raw_th - c_mu) / c_sd,
            }
        )
    pd.DataFrame(thresholds).to_csv(OUT_DIR / "clinical_thresholds_from_g1090.csv", index=False)

    rows = []
    boot_rows = []
    for th_row in thresholds:
        target = float(th_row["target_sensitivity"])
        t = float(th_row["threshold_z"])
        for model, cfg in score_bank.items():
            gate_g = make_gate_score(c_g, cfg["train"], t, cfg["width"], cfg["center"], cfg["lambda"])
            gate_s = make_gate_score(c_s, cfg["external"], t, cfg["width"], cfg["center"], cfg["lambda"])
            b = bootstrap_external(s["y"], c_s, gate_s, t)
            b["model"] = model
            b["target_sensitivity"] = target
            boot_rows.append(b)
            for dataset, y, clinical, gate in [
                ("g1090_oof", g["y"], c_g, gate_g),
                ("sdata_external", s["y"], c_s, gate_s),
            ]:
                row = {
                    "model": model,
                    "dataset": dataset,
                    "target_sensitivity": target,
                    "threshold_z": t,
                    "width": cfg["width"],
                    "center": cfg["center"],
                    "lambda": cfg["lambda"],
                    "model_note": cfg["note"],
                }
                for key in ["mid_start", "mid_end", "tail_start", "tail_end"]:
                    if key in cfg:
                        row[key] = cfg[key]
                row.update(deesc_metrics(y, clinical, gate, t))
                rows.append(row)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT_DIR / "threshold_robustness_long.csv", index=False)
    boot_df = pd.concat(boot_rows, ignore_index=True)
    boot_df.to_csv(OUT_DIR / "threshold_robustness_external_bootstrap.csv", index=False)

    pivot_cols = [
        "clinical_sensitivity",
        "clinical_specificity",
        "clinical_positive_n",
        "clinical_positive_events",
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
    ]
    paired = long_df.pivot_table(
        index=["model", "target_sensitivity", "width", "center", "lambda"],
        columns="dataset",
        values=pivot_cols,
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{dataset}" for metric, dataset in paired.columns]
    paired = paired.reset_index().sort_values(["model", "target_sensitivity"], ascending=[True, False])
    paired.to_csv(OUT_DIR / "threshold_robustness_paired.csv", index=False)

    ext = long_df[long_df["dataset"].eq("sdata_external")].copy()
    robust_flags = []
    for model, sub in ext.groupby("model"):
        robust_flags.append(
            {
                "model": model,
                "min_external_deesc_n": int(sub["deesc_n"].min()),
                "max_external_deesc_prevalence": float(sub["deesc_prevalence"].max()),
                "max_external_sensitivity_loss": float(sub["sensitivity_loss"].max()),
                "min_external_specificity_gain": float(sub["specificity_gain"].min()),
                "total_external_fp_removed_across_thresholds": int(sub["fp_removed"].sum()),
                "total_external_tp_lost_across_thresholds": int(sub["tp_lost"].sum()),
            }
        )
    summary_df = pd.DataFrame(robust_flags).sort_values(["max_external_sensitivity_loss", "max_external_deesc_prevalence"])
    summary_df.to_csv(OUT_DIR / "threshold_robustness_external_summary.csv", index=False)

    plot_external(long_df)

    serializable_score_bank = {
        name: {k: v for k, v in cfg.items() if k not in {"train", "external"}}
        for name, cfg in score_bank.items()
    }
    summary = {
        "threshold_policy": "Clinical-positive thresholds are selected in g1090 OOF to achieve minimum sensitivities 95/90/85/80/75, then applied unchanged to sdata external after training-score z-scaling.",
        "model_policy": "AEC model parameters are fixed; no threshold-specific re-optimization.",
        "score_bank": serializable_score_bank,
        "outputs": {
            "long": str(OUT_DIR / "threshold_robustness_long.csv"),
            "paired": str(OUT_DIR / "threshold_robustness_paired.csv"),
            "external_summary": str(OUT_DIR / "threshold_robustness_external_summary.csv"),
            "external_bootstrap": str(OUT_DIR / "threshold_robustness_external_bootstrap.csv"),
            "plot": str(OUT_DIR / "sdata_threshold_sweep_external.png"),
        },
    }
    (OUT_DIR / "threshold_robustness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    show = ext[
        [
            "model",
            "target_sensitivity",
            "clinical_sensitivity",
            "clinical_specificity",
            "clinical_positive_n",
            "deesc_n",
            "deesc_events",
            "deesc_prevalence",
            "fp_removed",
            "tp_lost",
            "specificity_gain",
            "sensitivity_loss",
            "fisher_p",
        ]
    ].sort_values(["model", "target_sensitivity"], ascending=[True, False])
    print("\nExternal sdata threshold sweep")
    print(show.to_string(index=False))
    print("\nExternal robustness summary")
    print(summary_df.to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
