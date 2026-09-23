from __future__ import annotations

# 각 모델(case)의 predictions.xlsx 시트에 저장된 patient_id 리스트(질환x코호트별, 순서 포함)를 baseline과
# 직접 비교해 "정말 같은 샘플링 데이터로 평가했는지"를 검증한다. 지난번엔 y(라벨) 시퀀스만 비교했는데,
# 라벨만 같고 환자가 다를 가능성을 배제하기 위해 이번엔 patient_id 자체를 비교한다.
# _unbalanced 시트들은 샘플링 방식 자체가 달라(원본 prevalence, undersampling 없음) 비교 대상에서 제외.
# predictions는 모델별 predictions.csv가 아니라 clinic4 루트의 predictions.xlsx에 모델명=시트명으로
# 저장되는 구조로 바뀌어(clinic4_logistic_regression.py save_sheet) 시트명 리스트로 읽는다.

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "clinic4"
PREDICTIONS_XLSX = OUTPUTS_DIR / "predictions.xlsx"

# predictions.xlsx의 시트명 목록(= 각 모델 스크립트의 OUTPUT_DIR.name/model_name과 동일)
MODELS: list[str] = [
    "aec_fpca_ratio", "aec_fpca_only", "aec_ratio_only",
    "aec_vat", "aec_lama", "aec_nama", "aec_bodycomp",
    "vat", "sat", "lama", "nama", "imata", "bodycomp",
]
BASELINE = "baseline"
OUT_CSV = OUTPUTS_DIR / "sampling_identity_check.csv"


def patient_id_list(df: pd.DataFrame, disease: str, cohort: str) -> list:
    sub = df[(df["disease"] == disease) & (df["cohort"] == cohort)]
    return sub["patient_id"].tolist()


def main() -> None:
    predictions = {name: pd.read_excel(PREDICTIONS_XLSX, sheet_name=name) for name in [BASELINE] + MODELS}
    base_df = predictions[BASELINE]
    diseases_cohorts = base_df[["disease", "cohort"]].drop_duplicates().to_records(index=False).tolist()

    rows = []
    for model, df in predictions.items():
        if model == BASELINE:
            continue
        for disease, cohort in diseases_cohorts:
            base_ids = patient_id_list(base_df, disease, cohort)
            model_ids = patient_id_list(df, disease, cohort)
            identical = base_ids == model_ids  # 순서까지 완전히 동일해야 True
            same_set = set(base_ids) == set(model_ids)
            rows.append({
                "model": model, "disease": disease, "cohort": cohort,
                "n_baseline": len(base_ids), "n_model": len(model_ids),
                "identical_order": identical, "identical_set": same_set,
            })
            mark = "OK" if identical else "MISMATCH"
            print(f"[{mark}] {model} vs {BASELINE} | {disease} {cohort} "
                  f"n_base={len(base_ids)} n_model={len(model_ids)} "
                  f"order_match={identical} set_match={same_set}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    n_mismatch = int((~result["identical_order"]).sum())
    print(f"\n{'ALL MATCH' if n_mismatch == 0 else f'{n_mismatch} MISMATCHES FOUND'} "
          f"({len(result)} model x disease x cohort comparisons)")

    # self-check: baseline과 동일한 balanced-sampling 계열 모델은 patient_id 순서까지 100% 일치해야 함
    assert n_mismatch == 0, f"{n_mismatch}개 조합에서 baseline과 다른 환자/순서로 샘플링됨 - 위 표 확인"
    print("OK: baseline과 모든 balanced 모델의 patient_id가 순서까지 완전히 일치함(진짜 동일 데이터로 비교)")


if __name__ == "__main__":
    main()
