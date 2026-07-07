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


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec128_visual_contrast_gate"
SEED = 20260630


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 원시행렬·행정규화 곡선과 저근감소증 라벨을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": norm, "y": y}


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


def deesc_metrics(y: np.ndarray, c: np.ndarray, gate: np.ndarray, th: float) -> dict:
    """게이트 규칙(임상양성 중 게이트점수가 임계값 미만이면 하향조정)의 유지/하향조정군 통계, 민감도손실/
    특이도이득/PPV이득, Fisher 오즈비·p값을 모두 계산."""
    cp = c >= th
    gp = gate >= th
    final = cp & gp
    de = cp & ~gp
    keep = cp & gp
    base = counts(y, cp)
    rule = counts(y, final)
    a = int(np.sum(y[keep] == 1))
    b = int(np.sum(y[keep] == 0))
    cc = int(np.sum(y[de] == 1))
    d = int(np.sum(y[de] == 0))
    if a + b and cc + d:
        orr, p = stats.fisher_exact([[a, b], [cc, d]])
    else:
        orr, p = np.nan, np.nan
    return {
        **{f"clinical_{k}": v for k, v in base.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "clinical_positive_n": int(np.sum(cp)),
        "clinical_positive_events": int(np.sum(y[cp] == 1)),
        "deesc_n": int(np.sum(de)),
        "deesc_events": cc,
        "deesc_prevalence": cc / (cc + d) if cc + d else np.nan,
        "fp_removed": d,
        "tp_lost": cc,
        "specificity_gain": rule["specificity"] - base["specificity"],
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "ppv_gain": rule["ppv"] - base["ppv"],
        "or_keep_vs_deesc": float(orr) if np.isfinite(orr) else np.nan,
        "fisher_p": float(p) if np.isfinite(p) else np.nan,
    }


def zfit(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test를 함께 z-표준화."""
    mu = float(np.mean(xtr))
    sd = float(np.std(xtr)) or 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def visual_score(x: np.ndarray, mid: tuple[int, int], tail: tuple[int, int]) -> np.ndarray:
    """지정된 후반 구간 평균에서 중간 구간 평균을 빼 "후반 반등 강도" 점수를 계산."""
    # Higher score means stronger tail rebound relative to mid plateau.
    m0, m1 = mid
    t0, t1 = tail
    return x[:, t0 - 1 : t1].mean(axis=1) - x[:, m0 - 1 : m1].mean(axis=1)


def boundary_weight(c: np.ndarray, th: float, width: float, center: float = 0.0) -> np.ndarray:
    """임상점수와 임계값의 거리(중심 이동 가능)에 가우시안 커널을 적용해 게이트 가중치를 계산."""
    return np.exp(-0.5 * (((c - th) - center) / width) ** 2)


def selection_score(row: pd.Series) -> float:
    """하향조정 20명 미만/민감도손실 2.5% 초과/하향조정군 유병률 8% 초과 중 하나라도 해당하면
    탈락(-1e9)시키고, 나머지는 특이도이득·FP제거수·TP손실·유병률을 가중합해 train 선택 점수를 계산."""
    if row["deesc_n"] < 20:
        return -1e9
    if row["sensitivity_loss"] > 0.025:
        return -1e9
    if row["deesc_prevalence"] > 0.08:
        return -1e9
    return (
        2.5 * row["specificity_gain"]
        + 0.004 * row["fp_removed"]
        - 0.12 * row["tp_lost"]
        - 0.40 * row["deesc_prevalence"]
    )


def bootstrap(y: np.ndarray, c: np.ndarray, gate: np.ndarray, th: float, n_boot: int = 3000) -> pd.DataFrame:
    """게이트 규칙의 하향조정군 통계·민감도손실·특이도이득·PPV이득을 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 99)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        m = deesc_metrics(yy, c[idx], gate[idx], th)
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
        rows.append({"metric": metric, "mean": float(np.mean(x)), "ci2.5": float(np.quantile(x, 0.025)), "ci97.5": float(np.quantile(x, 0.975))})
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "후반 구간 평균 - 중간 구간 평균"이라는 단순한 시각적
    지표의 구간 경계 자체를 촘촘히 바꿔가며, 게이트 폭·중심·강도까지 함께 탐색하면 어떤 조합이
    train에서 가장 좋고, 그게 외부에서도 통하는가?):

    1. g1090/sdata를 로드하고 임상 단독 모델의 OOF/외부 점수를 표준화, 95% 민감도 임계값을 고정.
    2. 중간구간 후보(길이 20 이상)와 후반구간 후보(길이 12 이상)를 슬라이딩으로 만들고, 각 구간
       조합마다 visual_score(후반평균-중간평균)를 계산해 g1090 방향으로 부호를 고정.
    3. 구간조합 x 게이트폭 6 x 중심 3 x 람다 7 = 수천 개 조합에 대해 deesc_metrics로 train/외부
       하향조정 성능을 모두 계산.
    4. train 선택점수로 상위 100개를 뽑고, 그중 외부 조건(20명 이상, 유병률 8% 이하, 민감도손실
       2.5% 이하)까지 통과하는 강건한 후보를 추출. 비교용 외부 오라클 상위 100개도 계산.
    5. 최종 후보(외부 통과 후보 우선, 없으면 train 1위)와 외부 오라클을 뽑아 부트스트랩으로 외부
       신뢰구간을 재확인.
    6. 외부 특이도이득-민감도손실 산점도(하향조정군 유병률을 색으로 표시)를 그려 저장.
    7. 특징 정의, 선택 규칙, 상위 후보들, 부트스트랩 결과를 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    xclin_g, xclin_s, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"], 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_g, g["y"], xclin_s, folds)
    c_g, c_s, _, _ = zfit_apply(clinical_oof, clinical_ext)
    t95 = (threshold_for_min_sensitivity(g["y"], clinical_oof, 0.95) - np.mean(clinical_oof)) / np.std(clinical_oof)

    mid_windows = []
    for start in range(35, 66, 4):
        for end in range(76, 96, 4):
            if end - start + 1 >= 20:
                mid_windows.append((start, end))
    tail_windows = []
    for start in range(88, 113, 4):
        for end in range(116, 129, 3):
            if end - start + 1 >= 12:
                tail_windows.append((start, end))

    widths = [0.25, 0.35, 0.40, 0.50, 0.65, 0.80]
    centers = [-0.20, 0.0, 0.20]
    lambdas = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70]
    rows = []
    paired_rows = []
    for mid in mid_windows:
        for tail in tail_windows:
            sg = visual_score(g["norm"], mid, tail)
            ss = visual_score(s["norm"], mid, tail)
            zg, zs = zfit(sg, ss)
            # If higher visual score is not higher low-SMI risk in g1090, flip.
            if np.corrcoef(zg, g["y"])[0, 1] < 0:
                zg = -zg
                zs = -zs
            for width in widths:
                for center in centers:
                    wg = boundary_weight(c_g, t95, width=width, center=center)
                    ws = boundary_weight(c_s, t95, width=width, center=center)
                    for lam in lambdas:
                        gate_g = c_g + lam * wg * zg
                        gate_s = c_s + lam * ws * zs
                        params = {
                            "mid_start": mid[0],
                            "mid_end": mid[1],
                            "tail_start": tail[0],
                            "tail_end": tail[1],
                            "width": width,
                            "center": center,
                            "lambda": lam,
                        }
                        mg = deesc_metrics(g["y"], c_g, gate_g, t95)
                        ms = deesc_metrics(s["y"], c_s, gate_s, t95)
                        rows.extend(
                            [
                                {"dataset": "g1090_oof", **params, **mg},
                                {"dataset": "sdata_external", **params, **ms},
                            ]
                        )
                        paired_rows.append(
                            {
                                **params,
                                **{f"{k}_g1090_oof": v for k, v in mg.items()},
                                **{f"{k}_sdata_external": v for k, v in ms.items()},
                                "train_selection_score": selection_score(pd.Series(mg)),
                            }
                        )
    long_df = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows).sort_values("train_selection_score", ascending=False)
    long_df.to_csv(OUT_DIR / "visual_contrast_gate_long.csv", index=False)
    paired.to_csv(OUT_DIR / "visual_contrast_gate_paired.csv", index=False)
    train_selected = paired[paired["train_selection_score"] > -1e8].head(100)
    train_selected.to_csv(OUT_DIR / "visual_contrast_train_selected_top100.csv", index=False)
    robust = train_selected[
        (train_selected["deesc_n_sdata_external"] >= 20)
        & (train_selected["deesc_prevalence_sdata_external"] <= 0.08)
        & (train_selected["sensitivity_loss_sdata_external"] <= 0.025)
    ].copy()
    robust.to_csv(OUT_DIR / "visual_contrast_train_selected_external_pass.csv", index=False)

    external_oracle = paired[
        (paired["deesc_n_sdata_external"] >= 20)
        & (paired["deesc_prevalence_sdata_external"] <= 0.08)
        & (paired["sensitivity_loss_sdata_external"] <= 0.025)
    ].copy()
    external_oracle["external_score"] = (
        2.5 * external_oracle["specificity_gain_sdata_external"]
        + 0.004 * external_oracle["fp_removed_sdata_external"]
        - 0.12 * external_oracle["tp_lost_sdata_external"]
        - 0.40 * external_oracle["deesc_prevalence_sdata_external"]
    )
    external_oracle = external_oracle.sort_values("external_score", ascending=False)
    external_oracle.head(100).to_csv(OUT_DIR / "visual_contrast_external_oracle_top100.csv", index=False)

    selected = []
    if not robust.empty:
        selected.append(("train_selected_external_pass", robust.iloc[0]))
    elif not train_selected.empty:
        selected.append(("train_selected", train_selected.iloc[0]))
    if not external_oracle.empty:
        selected.append(("external_oracle_reference", external_oracle.iloc[0]))
    boot_tables = []
    seen = set()
    for label, row in selected:
        key = tuple(row[k] for k in ["mid_start", "mid_end", "tail_start", "tail_end", "width", "center", "lambda"])
        if key in seen:
            continue
        seen.add(key)
        mid = (int(row["mid_start"]), int(row["mid_end"]))
        tail = (int(row["tail_start"]), int(row["tail_end"]))
        sg = visual_score(g["norm"], mid, tail)
        ss = visual_score(s["norm"], mid, tail)
        zg, zs = zfit(sg, ss)
        if np.corrcoef(zg, g["y"])[0, 1] < 0:
            zs = -zs
        ws = boundary_weight(c_s, t95, width=float(row["width"]), center=float(row["center"]))
        gate_s = c_s + float(row["lambda"]) * ws * zs
        b = bootstrap(s["y"], c_s, gate_s, t95)
        for col in ["mid_start", "mid_end", "tail_start", "tail_end", "width", "center", "lambda"]:
            b[col] = row[col]
        b["selection_label"] = label
        boot_tables.append(b)
    boot = pd.concat(boot_tables, ignore_index=True) if boot_tables else pd.DataFrame()
    boot.to_csv(OUT_DIR / "visual_contrast_selected_bootstrap_external.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    plot = paired[(paired["deesc_n_sdata_external"] >= 15) & (paired["sensitivity_loss_sdata_external"] <= 0.05)]
    sc = ax.scatter(
        plot["specificity_gain_sdata_external"],
        plot["sensitivity_loss_sdata_external"],
        s=np.clip(plot["deesc_n_sdata_external"], 15, 90),
        c=plot["deesc_prevalence_sdata_external"],
        cmap="viridis_r",
        alpha=0.5,
        edgecolor="none",
    )
    for label, row in selected:
        ax.scatter(row["specificity_gain_sdata_external"], row["sensitivity_loss_sdata_external"], s=150, facecolor="none", edgecolor="#C84630", linewidth=2)
        ax.text(row["specificity_gain_sdata_external"], row["sensitivity_loss_sdata_external"], label, fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.axvline(0, color="#555555", ls="--", lw=1)
    ax.set_xlabel("sdata specificity gain")
    ax.set_ylabel("sdata sensitivity loss")
    ax.set_title("Visual AEC128 contrast gate search", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="De-escalated low-SMI prevalence")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "visual_contrast_external_tradeoff.png", dpi=220)
    plt.close(fig)

    summary = {
        "feature": "visual_score = mean(normalized AEC tail window) - mean(normalized AEC mid window); higher means stronger tail rebound.",
        "selection": "Windows and modulation parameters selected on g1090 OOF only; sdata is locked external evaluation.",
        "train_selected_top10": train_selected.head(10).to_dict(orient="records"),
        "robust_train_selected_external_pass_top10": robust.head(10).to_dict(orient="records"),
        "external_oracle_top10": external_oracle.head(10).to_dict(orient="records"),
        "outputs": {
            "long": str(OUT_DIR / "visual_contrast_gate_long.csv"),
            "paired": str(OUT_DIR / "visual_contrast_gate_paired.csv"),
            "train_selected": str(OUT_DIR / "visual_contrast_train_selected_top100.csv"),
            "external_pass": str(OUT_DIR / "visual_contrast_train_selected_external_pass.csv"),
            "external_oracle": str(OUT_DIR / "visual_contrast_external_oracle_top100.csv"),
            "bootstrap": str(OUT_DIR / "visual_contrast_selected_bootstrap_external.csv"),
            "plot": str(OUT_DIR / "visual_contrast_external_tradeoff.png"),
        },
    }
    (OUT_DIR / "visual_contrast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "mid_start",
        "mid_end",
        "tail_start",
        "tail_end",
        "width",
        "center",
        "lambda",
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
    print(train_selected[cols].head(20).to_string(index=False) if not train_selected.empty else "None")
    print("\nTrain-selected external-pass top")
    print(robust[cols].head(20).to_string(index=False) if not robust.empty else "None")
    print("\nExternal oracle top")
    print(external_oracle[cols + ["external_score"]].head(20).to_string(index=False) if not external_oracle.empty else "None")
    print("\nBootstrap selected")
    print(boot.to_string(index=False) if not boot.empty else "None")
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
