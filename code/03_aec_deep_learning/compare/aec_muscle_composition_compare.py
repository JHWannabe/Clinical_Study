from __future__ import annotations

# clinic4+VAT+SAT baseline에 TAMA(=NAMA_sum_cm2+LAMA_sum_cm2)/IMATA_sum_cm2 근육계열 body composition
# feature를 추가하면 HTN/DM/CKD internal AUC가 개선되는지 스크리닝(2026-09-03, 사용자 요청: "tama/imata등의
# 근육 데이터를 사용하면 개선될까?"). 이 컬럼들은 Clinical_Study/data/gangnam_final_dataset.xlsx에는 없고(현재
# VAT/SAT만 보유), CT_AEC_process/scripts/body_composition_sum.py가 만든
# CT_AEC_process/data/{Gangnam,Sinchon}/{site}_body_composition.xlsx(pubis~liver 구간 합산, TotalSegmentator
# tissue_4_types)에서 PatientID로 가져와 병합한다.
# 근육데이터 커버리지가 100%가 아니라(gangnam 1147/1259=91.1%, sinchon 1015/1123=90.4%, 2026-09-03 확인)
# 병합하면 코호트가 줄어들므로, clinic4/clinic4+VAT+SAT baseline도 같은 축소 코호트로 반드시 재적합해
# 공정 비교한다(step_disease_logistic.py의 predictions.csv를 그대로 재사용하지 않음).
# 모델선택은 internal CV로만 해야 하므로([[feedback_internal_external_validation_discipline]]) 이 스크립트는
# external은 전혀 쓰지 않는 internal-only 스크리닝이다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "muscle"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
MUSCLE_XLSX = Path(r"C:\Users\jhjun\OneDrive\Desktop\CT_AEC_process\data\Gangnam\강남_body_composition.xlsx")

AGE_CUTOFF = 20
N_FOLDS = 5
SEED = 20260709
MIN_POSITIVES = 2

VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
TAMA_COL, IMATA_COL = "TAMA_sum_cm2", "IMATA_sum_cm2"

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

MODEL_EXTRA_COLS: dict[str, list[str]] = {
    "clinic4": [],
    "clinic4_vatsat": [VAT_COL, SAT_COL],
    "clinic4_vatsat_tama": [VAT_COL, SAT_COL, TAMA_COL],
    "clinic4_vatsat_imata": [VAT_COL, SAT_COL, IMATA_COL],
    "clinic4_vatsat_muscle": [VAT_COL, SAT_COL, TAMA_COL, IMATA_COL],
}
MODEL_ORDER = list(MODEL_EXTRA_COLS)
MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_vatsat": "clinic4+VAT+SAT",
    "clinic4_vatsat_tama": "clinic4+VAT+SAT+TAMA",
    "clinic4_vatsat_imata": "clinic4+VAT+SAT+IMATA",
    "clinic4_vatsat_muscle": "clinic4+VAT+SAT+TAMA+IMATA",
}
DELONG_PAIRS = [
    ("clinic4", "clinic4_vatsat"),
    ("clinic4_vatsat", "clinic4_vatsat_tama"),
    ("clinic4_vatsat", "clinic4_vatsat_imata"),
    ("clinic4_vatsat", "clinic4_vatsat_muscle"),
]


def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    return meta


# body_composition_sum.py 산출물에서 seg_status=="ok"만 취하고 TAMA(=NAMA+LAMA, 정상+저감쇠 근육 총면적)를
# 새로 계산해 PatientID/TAMA_sum_cm2/IMATA_sum_cm2만 반환
def load_muscle(xlsx_path: Path) -> pd.DataFrame:
    bc = pd.read_excel(xlsx_path, sheet_name="body_composition", engine="openpyxl")
    bc = bc[bc["seg_status"] == "ok"].copy()
    bc[TAMA_COL] = bc["NAMA_sum_cm2"].astype(float) + bc["LAMA_sum_cm2"].astype(float)
    return bc[["PatientID", TAMA_COL, IMATA_COL]]


def valid_rows(meta: pd.DataFrame) -> np.ndarray:
    required_cols = CLINICAL_BASE_COLS + [VAT_COL, SAT_COL, TAMA_COL, IMATA_COL]
    vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
    mask = vals.notna().all(axis=1).to_numpy()
    valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
    return mask & valid_sex


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


def build_matrix(meta: pd.DataFrame, extra_cols: list[str]) -> np.ndarray:
    cols = CLINICAL_BASE_COLS + extra_cols
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(rest)
    sex = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex, scaled])


# code/질병예측/step_disease_logistic.py, code/aec_cnn_vatsat_disease_compare.py와 동일 구현(중복 허용 - 코드베이스 관례)
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
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


def plot_auc_bar(summary: pd.DataFrame, out_path: Path) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    x = np.arange(len(features))
    width = 0.8 / len(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(6 + 3 * len(features), 8))
    for i, name in enumerate(MODEL_ORDER):
        rows = summary[summary["model"] == name].set_index("feature").reindex(features)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[name])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=24)
    ax.set_ylabel("Internal 5-fold OOF AUC", fontsize=24)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_title("clinic4+VAT+SAT baseline에 TAMA/IMATA 근육 feature 추가 스크리닝 (internal-only)",
                 fontsize=18, fontweight="bold", color="#161616")
    ax.legend(fontsize=13, frameon=False, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC bar plot to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_cohort(INTERNAL_XLSX)
    muscle = load_muscle(MUSCLE_XLSX)
    n_before = len(meta)
    meta = meta.merge(muscle, on="PatientID", how="inner")
    print(f"근육데이터 병합: {n_before}명 -> {len(meta)}명 (커버리지 {len(meta) / n_before:.1%}, "
          f"seg_status=='ok'가 아니거나 CT_AEC_process 코호트에 없는 환자 제외)")

    meta = meta[valid_rows(meta)].reset_index(drop=True)
    print(f"필수 컬럼 결측 제외 후 최종 internal n={len(meta)}")

    summary_rows, delong_rows = [], []
    for feat, slug in FEATURES.items():
        y_all = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        mask_val = np.isfinite(y_all)
        n_pos, n_neg = int(y_all[mask_val].sum()), int(mask_val.sum() - y_all[mask_val].sum())
        if min(n_pos, n_neg) < MIN_POSITIVES:
            print(f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만")
            continue

        meta_m = meta.loc[mask_val].reset_index(drop=True)
        y = y_all[mask_val].astype(int)
        cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)

        oof_by_model = {}
        for name, extra_cols in MODEL_EXTRA_COLS.items():
            x = build_matrix(meta_m, extra_cols)
            oof = cross_val_predict(LogisticRegression(max_iter=2000), x, y, cv=cv, method="predict_proba")[:, 1]
            oof_by_model[name] = oof
            auc = float(roc_auc_score(y, oof))
            summary_rows.append({"feature": feat, "model": name, "n": int(len(y)), "n_pos": int(y.sum()), "auc": auc})
            print(f"[{feat}] {MODEL_LABELS[name]}: internal AUC={auc:.4f}")

        for base, ext in DELONG_PAIRS:
            d = delong_paired_auc_test(y, oof_by_model[base], oof_by_model[ext])
            print(f"[{feat}] DeLong {MODEL_LABELS[ext]} vs {MODEL_LABELS[base]}: "
                  f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
            delong_rows.append({"feature": feat, "baseline_model": base, "extended_model": ext,
                                 "auc_baseline": d["auc_a"], "auc_extended": d["auc_b"],
                                 "auc_diff": d["diff"], "z": d["z"], "p_value": d["p_value"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "muscle_composition_summary.csv", index=False)
    print(f"Saved summary to {OUTPUT_DIR / 'muscle_composition_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(OUTPUT_DIR / "muscle_composition_delong.csv", index=False)
    print(f"Saved DeLong comparison to {OUTPUT_DIR / 'muscle_composition_delong.csv'}")

    if not summary.empty:
        plot_auc_bar(summary, OUTPUT_DIR / "muscle_composition_auc.png")


if __name__ == "__main__":
    main()
