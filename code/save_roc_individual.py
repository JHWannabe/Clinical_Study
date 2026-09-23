from __future__ import annotations

# predictions.xlsx(모델별 시트)를 읽어 질환×코호트 조합마다 이미지 1장씩 저장한다
# (outputs/clinic4/<roc_individual|roc_individual_bodycomp>/<disease>/<cohort>.png).
# 한 이미지 안에 같은 묶음의 모델을 모두 라인으로 겹쳐 그려 모델간 비교가 되게 함 - AEC가 들어간
# 묶음(roc_individual)과 AEC 없이 체성분만 넣은 묶음(roc_individual_bodycomp)을 따로 저장한다.

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "clinic4"
PRED_XLSX = OUTPUTS_DIR / "predictions.xlsx"  # 모델별 시트(clinic4_logistic_regression.py save_sheet가 씀)

# case명 = predictions.xlsx의 시트명. AEC가 들어간 모델 묶음과, AEC 없이 체성분만 넣은 모델 묶음을
# 따로 겹쳐 그린다(한 이미지에 다 겹치면 범례가 안 읽힘) - baseline은 두 묶음 모두에 기준선으로 포함
MODEL_SETS = {
    "roc_individual": [
        "baseline_unbalanced", "baseline", "aec_fpca_ratio", "aec_fpca_only", "aec_ratio_only",
        "aec_vat", "aec_lama", "aec_nama", "aec_bodycomp",
    ],
    "roc_individual_bodycomp": [
        "baseline_unbalanced", "baseline", "vat", "sat", "lama", "nama", "imata", "vsr",
        "vat_sat", "vat_sat_vsr", "bodycomp",
    ],
}

BASELINE_MODEL = "baseline"  # 이 모델 대비 나머지 모델의 AUC 유의성을 검정(DeLong)


# baseline과 모델 하나를 patient_id로 짝지어(paired) DeLong AUC 차이 검정 -> p-value
def delong_paired_p(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> float:
    return delong_paired_auc_test(y, score_a, score_b)["p_value"]


# 질환×코호트 하나(df_dc: model별 y/score)의 모델별 ROC curve를 한 이미지에 겹쳐 그려 out_path에 저장.
# baseline 대비 유의(p<0.05)한 모델은 범례에 * 표기.
def save_one(disease: str, cohort: str, df_dc: pd.DataFrame, out_path: Path) -> dict[str, float]:
    baseline_g = df_dc[df_dc["model"] == BASELINE_MODEL].set_index("patient_id")

    curves = []
    pvalues: dict[str, float] = {}
    for model, g in df_dc.groupby("model"):
        fpr, tpr, _ = roc_curve(g["y"].to_numpy(), g["score"].to_numpy())
        auc = roc_auc_score(g["y"].to_numpy(), g["score"].to_numpy())

        p = float("nan")
        if model != BASELINE_MODEL and not baseline_g.empty:
            paired = g.set_index("patient_id")[["y", "score"]].join(
                baseline_g[["score"]], how="inner", rsuffix="_baseline"
            )
            if len(paired) >= 2 and paired["y"].nunique() == 2:
                p = delong_paired_p(
                    paired["y"].to_numpy(),
                    paired["score_baseline"].to_numpy(),
                    paired["score"].to_numpy(),
                )
        pvalues[model] = p
        curves.append((auc, model, fpr, tpr, p))
    curves.sort(key=lambda c: c[0], reverse=True)  # 범례를 AUC 내림차순으로 정렬

    fig, ax = plt.subplots(figsize=(6, 6))
    for auc, model, fpr, tpr, p in curves:
        star = " *" if p < 0.05 else ""
        ax.plot(fpr, tpr, linewidth=2, label=f"{model} (AUC={auc:.3f}){star}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{disease} | {cohort}")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.02, 0.02, f"* p<0.05 vs {BASELINE_MODEL} (DeLong)",
        transform=ax.transAxes, fontsize=7, color="gray", va="bottom",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pvalues


# model_names 묶음 하나를 out_dir 밑에 저장(이미지 + comparison_summary/table.csv)
def run_model_set(model_names: list[str], out_dir: Path) -> None:
    frames = []
    for model in model_names:
        try:
            df = pd.read_excel(PRED_XLSX, sheet_name=model)
        except ValueError:
            print(f"[{model}] SKIP: {PRED_XLSX}에 시트 없음")
            continue
        frames.append(df.assign(model=model))
    all_predictions = pd.concat(frames, ignore_index=True)

    n_saved = 0
    summary_rows = []
    for (disease, cohort), g in all_predictions.groupby(["disease", "cohort"]):
        out_path = out_dir / disease / f"{cohort}.png"
        pvalues = save_one(disease, cohort, g, out_path)
        n_saved += 1
        for model, mg in g.groupby("model"):
            auc = roc_auc_score(mg["y"].to_numpy(), mg["score"].to_numpy())
            p = pvalues.get(model, float("nan"))
            summary_rows.append({
                "disease": disease, "cohort": cohort, "model": model,
                "n": len(mg), "n_pos": int(mg["y"].sum()), "auc": auc,
                "p_value_vs_baseline": p, "significant_p<0.05": bool(p < 0.05) if pd.notna(p) else False,
            })
    print(f"Saved {n_saved} ROC images to {out_dir}")

    # self-check: 저장했다고 센 개수만큼 실제 png 파일이 디스크에 있는지 확인
    saved_files = list(out_dir.rglob("*.png"))
    assert len(saved_files) == n_saved, f"expected {n_saved} files, found {len(saved_files)}"
    print(f"OK: {len(saved_files)} PNG files exist on disk")

    # 통합비교결과 테이블: long-format summary + disease/cohort x model 가로 pivot(AUC, 유의 시 * 표기)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    summary["auc_display"] = summary.apply(
        lambda r: f"{r['auc']:.3f}{'*' if r['significant_p<0.05'] else ''}", axis=1
    )
    pivot = summary.pivot_table(
        index=["disease", "cohort"], columns="model", values="auc_display", aggfunc="first"
    ).reindex(columns=model_names)
    pivot_path = out_dir / "comparison_table.csv"
    pivot.to_csv(pivot_path, encoding="utf-8-sig")
    print(f"Saved comparison table to {pivot_path} (long summary: {summary_path})")


def main() -> None:
    for dir_name, model_names in MODEL_SETS.items():
        run_model_set(model_names, OUTPUTS_DIR / dir_name)


if __name__ == "__main__":
    main()
