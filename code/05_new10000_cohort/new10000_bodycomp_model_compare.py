"""사용자 요청(2026-09-18): "이전 연구처럼, clinic4+aec또는 체성분 데이터에 대한 여러 모델에 대한
성능도 비교해" - code/0910/clinic_body_composition_individual_compare.py(이하 bc)의 33개 모델
(clinic4 baseline + 스칼라 8종 + 변수간 비율 3종 + 곡선 FPCA(elbow) 7종 + 곡선 Upper/Lower ratio
7종 + 곡선 slope 7종, AEC는 7개 곡선 중 하나)을 new10000_final_dataset.xlsx 단독 코호트에 그대로
적용. gangnam/sinchon 없이 이 파일 하나만 쓰므로 bc의 internal-fit/external-frozen 구조 대신
new10000_disease_classification.py에서 이미 채택한 5-fold StratifiedKFold OOF 평가를 그대로
확장 사용("internal처럼 5-fold OOF로 해도 된다"는 사용자 확인, 2026-09-18).
체성분 스칼라/곡선 시트는 16개 시트 전체에 값이 있는 환자(PatientID 교집합, n=3,150)에서만
존재하므로(0916/compare_new10000_cohort.py의 common_patient_ids와 동일 로직) 이 서브셋에서
clinic4 baseline도 함께 재계산해 모든 모델을 동일 표본·동일 CV fold로 공정 비교(DeLong 페어링에도
동일 fold가 필요).
후속 요청(2026-09-18): "confusion matrix로도 저장하고, Youden방법으로" - 각 모델 자신의 OOF
확률에서 Youden index 임계값을 구하고(bc.youden_threshold, run()의 internal 방식과 동일), 그
임계값으로 confusion matrix를 계산해 code/0918/build_confusion_matrices.py와 동일한 시각화
스타일(section별로 clinic4 vs 그 section 내 best 모델, 질환을 열로 배치)로 저장한다.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_XLSX = PROJECT_ROOT / "data" / "new10000_final_dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0918" / "new10000_bodycomp_model_compare"
CM_DIR = OUTPUT_DIR / "confusion_matrix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "bc_compare", PROJECT_ROOT / "code" / "0910" / "clinic_body_composition_individual_compare.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

DISEASES = ["DM", "HTN", "DL", "Osteoporosis", "MI", "Stroke"]
SECTION_COLORS = {"baseline": "#555555", **bc.SECTION_COLORS}


# 16개 시트(metadata + 체성분 스칼라/곡선 7종 x cropped/128/total) 전체에 값이 있는 PatientID만
# 사용(0916/compare_new10000_cohort.py common_patient_ids와 동일 로직)
def common_patient_ids() -> set:
    sheets = pd.ExcelFile(NEW_XLSX).sheet_names
    id_sets = [set(pd.read_excel(NEW_XLSX, sheet_name=s, usecols=["PatientID"])["PatientID"]) for s in sheets]
    return set.intersection(*id_sets)


def load_cohort() -> pd.DataFrame:
    meta = pd.read_excel(NEW_XLSX, sheet_name="metadata")
    meta = meta[meta["PatientID"].isin(common_patient_ids())].reset_index(drop=True)
    meta = meta[meta["PatientSex"].astype(str).str.upper().isin(["M", "F"])].reset_index(drop=True)
    for ratio_col, (num_col, den_col) in bc.INTER_VAR_RATIOS.items():
        meta[ratio_col] = meta[num_col].astype(float) / meta[den_col].astype(float)

    curve_cols_all = []
    for key, (sheet, prefix) in bc.CURVE_SOURCES.items():
        curve = pd.read_excel(NEW_XLSX, sheet_name=sheet)
        meta = meta.merge(curve[["PatientID"] + bc.curve_cols(prefix)], on="PatientID", how="inner")
        curve_cols_all += bc.curve_cols(prefix)
    assert not meta[curve_cols_all].isna().any().any(), "커브 시트 병합 후 결측 발생"

    half = bc.N_SLICES // 2
    x = np.arange(1, bc.N_SLICES + 1, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    for key, (_sheet, prefix) in bc.CURVE_SOURCES.items():
        cols = bc.curve_cols(prefix)
        vals = meta[cols].astype(float).to_numpy()
        meta[f"{key}_uplow_ratio"] = meta[cols[:half]].astype(float).mean(axis=1) / \
            meta[cols[half:]].astype(float).mean(axis=1)
        meta[f"{key}_slope"] = (vals @ x_centered) / denom
    return meta


def oof_proba_for(model_name: str, meta: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold,
                   curve_raw: dict[str, np.ndarray]) -> np.ndarray:
    curve_key = bc.MODEL_CURVE_SOURCE[model_name]
    if curve_key is not None:
        return bc.fpca_oof_proba(meta, curve_raw[curve_key], y, bc.CURVE_FPCA_N_ELBOW[curve_key], cv)
    x, _, _ = bc.build_matrix(meta, bc.MODEL_EXTRA_COLS[model_name])
    return cross_val_predict(LogisticRegression(**bc.LOGREG_PARAMS), x, y, cv=cv, method="predict_proba")[:, 1]


def run_disease(disease: str, meta: pd.DataFrame, curve_raw: dict[str, np.ndarray]
                 ) -> tuple[list[dict], list[dict]]:
    y = meta[disease].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=bc.n_splits_for(y), shuffle=True, random_state=bc.SEED)

    oof_by_model: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []
    baseline_threshold = None
    for model_name in bc.MODEL_ORDER:
        oof = oof_proba_for(model_name, meta, y, cv, curve_raw)
        oof_by_model[model_name] = oof

        auc = float(roc_auc_score(y, oof))
        # 사용자 요청(2026-09-18): "baseline과 다른 모델의 성능을 비교하려면 threshold를 고정시키는게
        # 맞지 않나" -> 모델마다 자기 OOF에서 따로 threshold를 구하면 confusion matrix 차이가 모델
        # 성능 차이인지 그냥 다른 지점에서 잘랐기 때문인지 구분 안 됨. clinic4(MODEL_ORDER 첫 모델)의
        # threshold를 그 disease의 모든 모델에 고정 적용(code/0910·0916과 동일 원칙)
        if model_name == "clinic4":
            baseline_threshold = bc.youden_threshold(y, oof)
        thr = baseline_threshold
        cstats = bc.classification_stats(y, oof, thr)
        pred = (oof >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        section = "baseline" if model_name == "clinic4" else bc.MODEL_TO_SECTION[model_name]
        row = {"feature": disease, "model": model_name, "section": section, "n": len(y), "n_pos": int(y.sum()),
               "prevalence": float(y.mean()), "auc": auc, "threshold": thr, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
               **cstats}
        print(f"[{disease}/{model_name}] n={len(y)} n_pos={int(y.sum())} AUC={auc:.3f} "
              f"sens={cstats['sensitivity']:.3f} spec={cstats['specificity']:.3f} acc={cstats['accuracy']:.3f}")
        summary_rows.append(row)

    baseline_oof = oof_by_model["clinic4"]
    delong_rows = []
    for model_name in bc.MODEL_ORDER:
        if model_name == "clinic4":
            continue
        d = bc.delong_paired_auc_test(y, baseline_oof, oof_by_model[model_name])
        delong_rows.append({"feature": disease, "model": model_name,
                             "section": bc.MODEL_TO_SECTION[model_name], "n": len(y),
                             "auc_baseline": d["auc_a"], "auc_extended": d["auc_b"], "auc_diff": d["diff"],
                             "z": d["z"], "p_value": d["p_value"]})
    return summary_rows, delong_rows


def plot_auc_summary(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    for ax, disease in zip(axes.flat, DISEASES):
        df = summary[summary["feature"] == disease].iloc[::-1]
        colors = [SECTION_COLORS.get(s, "#999999") for s in df["section"]]
        ax.barh(df["model"], df["auc"], color=colors)
        for y, auc in enumerate(df["auc"]):
            ax.text(auc, y, f" {auc:.3f}", va="center", fontsize=7)
        clinic4_auc = df.loc[df["model"] == "clinic4", "auc"]
        if len(clinic4_auc):
            ax.axvline(float(clinic4_auc.iloc[0]), color="#555555", linestyle="--", linewidth=1)
        ax.set_xlim(0.4, 1.0)
        n = int(df["n"].iloc[0])
        ax.set_title(f"{disease} (n={n})", fontweight="bold")
        ax.set_xlabel("AUC (5-fold OOF)")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3, axis="x")
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in SECTION_COLORS.values()]
    fig.legend(handles, SECTION_COLORS.keys(), loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("new10000 단독 코호트 — clinic4 + 체성분(스칼라/비율/AEC 등 곡선) 33개 모델, 5-fold OOF",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# 사용자 요청(2026-09-18): 행=Predicted(Positive 위/Negative 아래), 열=Actual(Positive 왼쪽/Negative
# 오른쪽) 배치로 변경 - TP가 좌상단, TN이 우하단에 오도록
def draw_cm(ax, tn: int, fp: int, fn: int, tp: int, title: str) -> None:
    mat = np.array([[tp, fp], [fn, tn]])
    mat_norm = mat / mat.sum()
    ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=mat_norm.max() * 1.15)
    for (i, j), v in np.ndenumerate(mat):
        color = "white" if mat_norm[i, j] > mat_norm.max() * 0.6 else "black"
        ax.text(j, i, f"{v}\n({v / mat.sum():.1%})", ha="center", va="center", fontsize=11, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive (1)", "Negative (0)"], fontsize=9)
    ax.set_yticklabels(["Positive (1)", "Negative (0)"], fontsize=9)
    ax.set_xlabel("Actual Values", fontsize=9)
    ax.set_ylabel("Predicted Values", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")


def best_model_in_section(summary: pd.DataFrame, feat: str, models: list[str]) -> str:
    sub = summary[(summary["feature"] == feat) & (summary["model"].isin(models)) & (summary["model"] != "clinic4")]
    return sub.loc[sub["auc"].idxmax(), "model"]


# code/0918/build_confusion_matrices.py와 동일한 배치(질환을 열, clinic4/best를 행으로) - section별 1장
def plot_confusion_matrices(summary: pd.DataFrame) -> None:
    for section_name, models in bc.MODEL_SECTIONS.items():
        fig, axes = plt.subplots(2, len(DISEASES), figsize=(4.1 * len(DISEASES), 7.2))
        for c, feat in enumerate(DISEASES):
            best_model = best_model_in_section(summary, feat, models)
            for r, model_name in enumerate(["clinic4", best_model]):
                row = summary[(summary["feature"] == feat) & (summary["model"] == model_name)].iloc[0]
                title = (f"{feat}: {bc.MODEL_LABELS[model_name]}\nAUC={row['auc']:.3f} thr={row['threshold']:.3f}")
                draw_cm(axes[r, c], int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"]), title)
        fig.suptitle(f"{bc.SECTION_TITLES[section_name]} — Confusion Matrix (new10000, 5-fold OOF, Youden threshold)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = CM_DIR / f"{section_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    meta = load_cohort()
    # 0910 main()과 동일하게 모든 모델이 쓰는 clinic4 + 체성분 스칼라 8종이 전부 있는 행만 남겨
    # 33개 모델을 동일 표본으로 공정 비교(fold 정렬에도 동일 n이 필요)
    required_cols = bc.CLINICAL_BASE_COLS + bc.BODY_COMP_COLS
    mask = meta[required_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    meta = meta[mask].reset_index(drop=True)
    print(f"new10000 체성분 공통 코호트: n={len(meta)}")
    curve_raw = {key: meta[bc.curve_cols(prefix)].astype(float).to_numpy()
                 for key, (_sheet, prefix) in bc.CURVE_SOURCES.items()}

    all_summary, all_delong = [], []
    for disease in DISEASES:
        summary_rows, delong_rows = run_disease(disease, meta, curve_raw)
        all_summary += summary_rows
        all_delong += delong_rows

    summary = pd.DataFrame(all_summary)
    delong = pd.DataFrame(all_delong)
    delong["q_value_bh"] = bc.bh_fdr(delong["p_value"].to_numpy())

    summary.to_csv(OUTPUT_DIR / "new10000_bodycomp_summary.csv", index=False)
    delong.to_csv(OUTPUT_DIR / "new10000_bodycomp_delong_vs_clinic4.csv", index=False)
    print("\n=== BH-FDR 유의(q<0.05) ===")
    print(delong[delong["q_value_bh"] < 0.05].to_string(index=False))

    plot_auc_summary(summary, OUTPUT_DIR / "new10000_bodycomp_auc_summary.png")
    plot_confusion_matrices(summary)


if __name__ == "__main__":
    main()
