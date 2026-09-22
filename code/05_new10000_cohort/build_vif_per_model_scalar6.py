"""사용자 요청(2026-09-17): 슬라이드7(실험설계③-1, 개별모델 VIF — 체성분 스칼라)에서 BMI, SMI를
제거해 6종으로 축소(3-0 통합 체크가 아니라 3-1 개별모델 체크를 가리킨 것으로 정정). VIF는 이미
outputs/0910/vif_by_model.csv에 전체 계산되어 있으므로 재적합 없이 모델 목록만 필터링해 재사용한다.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "bc_compare", PROJECT_ROOT / "code" / "0910" / "clinic_body_composition_individual_compare.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def main() -> None:
    vif_all = pd.read_csv(PROJECT_ROOT / "outputs" / "0910" / "vif_by_model.csv")
    feat = "HTN"
    vif_df_feat = vif_all[vif_all["feature"] == feat]

    model_list = ["clinic4"] + [m for m in bc.MODEL_SECTIONS["scalar"]
                                 if m not in ("clinic4", "clinic4_bmi", "clinic4_smi")]
    out_path = PROJECT_ROOT / "outputs" / "0910" / "htn" / "vif" / "per_model_scalar.png"
    bc.plot_vif_per_model(vif_df_feat, model_list, f"{feat}: VIF — {bc.SECTION_TITLES['scalar']}", out_path)


if __name__ == "__main__":
    main()
