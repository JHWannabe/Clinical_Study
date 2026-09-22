"""사용자 요청(2026-09-18): "데이터가 imbalance하니까 class weight 주면 어때" -> Youden threshold가
이미 불균형을 어느정도 보정하고, calibration/DCA는 실제 유병률을 반영한 확률이 필요해 class_weight로
왜곡하면 해석이 깨지므로(사용자 확인) AUC/DeLong 비교용으로만 class_weight="balanced" 버전을 병행
실행한다. new10000_bodycomp_model_compare.py를 모듈로 그대로 불러와 bc.LOGREG_PARAMS에
class_weight="balanced"만 추가해 재사용(로직 중복 없음) - confusion matrix/calibration은 만들지 않음.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0918" / "new10000_bodycomp_model_compare_balanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "bodycomp_compare", PROJECT_ROOT / "code" / "0918" / "new10000_bodycomp_model_compare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.bc.LOGREG_PARAMS = {**m.bc.LOGREG_PARAMS, "class_weight": "balanced"}


def main() -> None:
    meta = m.load_cohort()
    required_cols = m.bc.CLINICAL_BASE_COLS + m.bc.BODY_COMP_COLS
    mask = meta[required_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    meta = meta[mask].reset_index(drop=True)
    print(f"new10000 체성분 공통 코호트(class_weight=balanced): n={len(meta)}")
    curve_raw = {key: meta[m.bc.curve_cols(prefix)].astype(float).to_numpy()
                 for key, (_sheet, prefix) in m.bc.CURVE_SOURCES.items()}

    all_summary, all_delong = [], []
    for disease in m.DISEASES:
        summary_rows, delong_rows = m.run_disease(disease, meta, curve_raw)
        all_summary += summary_rows
        all_delong += delong_rows

    summary = pd.DataFrame(all_summary)
    delong = pd.DataFrame(all_delong)
    delong["q_value_bh"] = m.bc.bh_fdr(delong["p_value"].to_numpy())

    summary.to_csv(OUTPUT_DIR / "new10000_bodycomp_summary_balanced.csv", index=False)
    delong.to_csv(OUTPUT_DIR / "new10000_bodycomp_delong_vs_clinic4_balanced.csv", index=False)
    print("\n=== BH-FDR 유의(q<0.05) ===")
    print(delong[delong["q_value_bh"] < 0.05].to_string(index=False))

    m.plot_auc_summary(summary, OUTPUT_DIR / "new10000_bodycomp_auc_summary_balanced.png")


if __name__ == "__main__":
    main()
