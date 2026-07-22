from __future__ import annotations

# Stage-1 screen-positive 환자들(TP: 실제 low-SMI, FP: 오탐) 사이에 aec_128 곡선
# 자체가 유의하게 다른지 확인한다. clinic 변수(bmi 등)는 TP/FP를 부분적으로만
# 갈랐다 (error_feature_analysis.md 참고) -- Stage 2가 이 둘을 CT 기반으로 더
# 잘 분리하려면, 애초에 두 그룹의 AEC 곡선 자체에 신호가 있어야 한다.
#
# TP/FP 분할과 patient-normalized AEC-128 로딩은 stage2_dataset.py의 기존
# 파이프라인(build_stage2_inputs / build_stage2_inputs_external)을 그대로
# 재사용하고, 곡선 플롯/순열검정(curve_diff_test)은 aec_curve_comparison.py의
# 유틸을 재사용한다 -- 레거시 파이프라인 재구현 금지 방침에 따라 새 계산 로직을
# 추가하지 않는다.
#
# Run: python code/stage2_aec_tp_fp_comparison.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
curve_mod = import_module("aec_curve_comparison")
stage2 = import_module("stage2_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline" / "aec_curve_comparison"

GROUP_ORDER = ["TP", "FP"]
GROUP_LABELS = ["TP (true low-SMI)", "FP (false positive)"]


def run_cohort(cohort: str, stage1_rows_pos: pd.DataFrame, stage2_input_aec: pd.DataFrame) -> dict:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    df = stage2_input_aec.merge(stage1_rows_pos[["PatientID", "group"]], on="PatientID", how="inner")
    assert len(df) == len(stage2_input_aec), f"{cohort}: TP/FP merge dropped rows"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    curve_mod.plot_curve_comparison(
        ax, df, "group", GROUP_ORDER, GROUP_LABELS, [curve_mod.COL_A, curve_mod.COL_B],
        f"Stage-1 screen-positive ({cohort}): TP vs FP AEC 곡선 비교",
    )
    r = curve_mod.curve_diff_test(df, "group", GROUP_ORDER, GROUP_LABELS)
    ax.text(0.02, 0.02, curve_mod.curve_diff_note(r),
            transform=ax.transAxes, fontsize=8, color=curve_mod.INK_MUTED, va="bottom")
    fig.tight_layout()
    curve_mod.savefig(fig, str(out_dir), "16_aec_curve_tp_vs_fp.png")

    summary_df = pd.DataFrame([{"figure": "16_aec_curve_tp_vs_fp.png", "comparison": "Stage-1 TP vs FP", **r}])
    summary_df["significant_p<0.05"] = summary_df["p_value"] < 0.05
    summary_path = out_dir / "16_tp_vs_fp_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"saved: {summary_path}")
    return r


def main():
    screen = stage2.fit_internal_screen()

    _, rows_pos_int, _, aec_int = stage2.build_stage2_inputs(screen)
    r_int = run_cohort("gangnam", rows_pos_int, aec_int)
    print(f"[gangnam] {curve_mod.curve_diff_note(r_int)}")

    _, rows_pos_ext, _, aec_ext = stage2.build_stage2_inputs_external(screen)
    r_ext = run_cohort("sinchon", rows_pos_ext, aec_ext)
    print(f"[sinchon] {curve_mod.curve_diff_note(r_ext)}")


if __name__ == "__main__":
    main()
