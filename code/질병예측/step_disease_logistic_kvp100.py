from __future__ import annotations

# step_disease_logistic.py와 동일한 파이프라인(FPCA/CV/threshold/DeLong/NRI-IDI 전부 동일 로직 재사용)을
# kVp=100 환자만으로 제한한 서브셋에 재실행해, 관전압을 100 kVp로 고정했을 때와 원본 그대로(80-140 kVp
# 혼합)일 때 AUC가 어떻게 달라지는지 비교하기 위한 스크립트(사용자 요청 2026-08-26: "kVp=100 한정 서브셋으로
# 실제 재분석을 돌려서 mixed-kVp 결과와 AUC를 직접 비교"). 모델링 로직을 중복 구현하지 않도록
# step_disease_logistic.py의 함수를 그대로 import해서 재사용하고, 코호트 필터링에 kVp==100 조건만 추가한다.
# 결과는 outputs/step_disease_logistic_kvp100/에 별도 저장(같은 output 폴더를 여러 스크립트가 나눠 쓰지
# 않는다는 프로젝트 원칙 유지).

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_disease_logistic import (  # noqa: E402
    CLINICAL_BASE_COLS, EXTERNAL_XLSX, INTERNAL_XLSX, MEAN_MAS_COL, SAT_COL, VAT_COL,
    build_table1, load_cohort, run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_disease_logistic_kvp100"
KVP_FIXED = 100


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    n_int_before, n_ext_before = len(meta_int), len(meta_ext)
    meta_int = meta_int[meta_int["kVp"] == KVP_FIXED].reset_index(drop=True)
    meta_ext = meta_ext[meta_ext["kVp"] == KVP_FIXED].reset_index(drop=True)
    print(f"kVp={KVP_FIXED} 제한: internal {n_int_before}->{len(meta_int)}, external {n_ext_before}->{len(meta_ext)}")

    required_cols = CLINICAL_BASE_COLS + [VAT_COL, SAT_COL, MEAN_MAS_COL]

    def valid_rows(meta: pd.DataFrame) -> pd.Series:
        vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_int).sum()}/{len(mask_int)}, "
          f"external {(~mask_ext).sum()}/{len(mask_ext)}")
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Final cohort (kVp={KVP_FIXED} only): internal n={len(meta_int)}, external n={len(meta_ext)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_table1(meta_int, meta_ext, OUTPUT_DIR)

    run(meta_int, meta_ext, OUTPUT_DIR)


if __name__ == "__main__":
    main()
