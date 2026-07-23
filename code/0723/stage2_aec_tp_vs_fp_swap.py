from __future__ import annotations

# Stage-2, cohort-swapped: within the Stage-1 screen-positive group (TP/FP only, as
# defined by clinic_only_baseline_swap.py / stage2_dataset_swap.py with
# internal=sinchon.xlsx, external=gangnam.xlsx), test whether the raw AEC-128 curve
# (and its 1st/2nd derivative, dp/d2p) differs between TP (true low-SMI) and FP (false
# positive) -- same whole-curve RMSD permutation test as
# code/stage2_aec_group_comparisons.py's "16" comparison, reused via aec_curve_comparison.py.
# Run: python code/0723/stage2_aec_tp_vs_fp_swap.py

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
curve_mod = import_module("aec_curve_comparison")
stage2_swap = import_module("stage2_dataset_swap")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "outputs" / "0723" / "aec_curve_comparison"

CURVE_TYPE_META = {
    "p": {"suffix": "", "label_suffix": "", "ylabel": "Patient-normalized AEC", "ref_line": 1.0},
    "dp": {"suffix": "_dp", "label_suffix": " (dp)", "ylabel": "d(AEC)/d(slice)", "ref_line": 0.0},
    "d2p": {"suffix": "_d2p", "label_suffix": " (d2p)", "ylabel": "d²(AEC)/d(slice)²", "ref_line": 0.0},
}


def curve_variant(df: pd.DataFrame, curve_type: str) -> pd.DataFrame:
    if curve_type == "p":
        return df
    out = df.copy()
    p = df[curve_mod.AEC_COLS].to_numpy(dtype=float)
    dp = np.gradient(p, axis=1)
    mat = dp if curve_type == "dp" else np.gradient(dp, axis=1)
    out[curve_mod.AEC_COLS] = mat
    return out


def run_tp_vs_fp(cohort: str, stage1_rows_pos: pd.DataFrame, stage2_input_aec: pd.DataFrame) -> list[dict]:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    df = stage2_input_aec.merge(stage1_rows_pos[["PatientID", "group"]], on="PatientID", how="inner")
    assert len(df) == len(stage2_input_aec), f"{cohort}: TP/FP merge dropped rows"

    summary_rows = []
    for curve_type in ["p", "dp", "d2p"]:
        meta = CURVE_TYPE_META[curve_type]
        df_variant = curve_variant(df, curve_type)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        curve_mod.plot_curve_comparison(
            ax, df_variant, "group", ["TP", "FP"], ["TP (true low-SMI)", "FP (false positive)"],
            [curve_mod.COL_A, curve_mod.COL_B],
            f"Stage-1 screen-positive ({cohort}, swapped): TP vs FP AEC 곡선 비교" + meta["label_suffix"],
            ylabel=meta["ylabel"], ref_line=meta["ref_line"],
        )
        r = curve_mod.curve_diff_test(df_variant, "group", ["TP", "FP"], ["TP (true low-SMI)", "FP (false positive)"])
        ax.text(0.02, 0.02, curve_mod.curve_diff_note(r), transform=ax.transAxes, fontsize=8,
                color=curve_mod.INK_MUTED, va="bottom")
        fig.tight_layout()
        fig_name = f"16_aec_curve_tp_vs_fp{meta['suffix']}.png"
        curve_mod.savefig(fig, str(out_dir), fig_name)

        summary_rows.append({"figure": fig_name, "comparison": "Stage-1 TP vs FP" + meta["label_suffix"],
                              "curve_type": curve_type, **r})

    summary_df = pd.DataFrame(summary_rows)
    summary_df["significant_p<0.05"] = summary_df["p_value"] < 0.05
    summary_path = out_dir / "16_stage2_tp_vs_fp_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"saved: {summary_path}")
    for row in summary_rows:
        print(f"[{cohort}] {row['comparison']}: {curve_mod.curve_diff_note(row)}")
    return summary_rows


def main():
    screen = stage2_swap.fit_internal_screen()

    _, rows_pos_int, _, aec_int = stage2_swap.build_stage2_inputs(screen)
    run_tp_vs_fp("sinchon", rows_pos_int, aec_int)

    _, rows_pos_ext, _, aec_ext = stage2_swap.build_stage2_inputs_external(screen)
    run_tp_vs_fp("gangnam", rows_pos_ext, aec_ext)


if __name__ == "__main__":
    main()
