from __future__ import annotations

# Aggregates the stage1_vs_stage2_summary.csv written by 3_aec_cnn_reclassify.py
# (baseline 1D-CNN) and each of its improvement variants (4-10) into one
# comparison table + chart, so the proposed improvements can be read side by side
# against the baseline instead of digging through separate output folders.
#
# This script does NOT retrain anything -- it only reads whatever summary CSVs are
# already sitting in outputs/. Run the scripts you want compared first:
#   python code/3_aec_cnn_reclassify.py       (baseline)
#   python code/4_aec_cnn_pretrain.py         (self-supervised encoder pretraining, full cohort)
#   python code/5_aec_cnn_skip.py             (real residual/skip connection)
#   python code/6_aec_cnn_bagging.py          (bootstrap ensemble diversity)
#   python code/7_aec_cnn_repeatedcv.py       (repeated stratified CV for OOF/threshold)
#   python code/8_aec_cnn_film.py             (FiLM conditioning instead of late concat)
#   python code/9_aec_cnn_fcn.py              (standard FCN 1D-CNN architecture instead of user-defined net)
#   python code/10_aec_cnn_crossattn.py       (cross-attention fusion instead of late concat)
# then:
#   python code/100_compare_cnn_variants.py
#
# Variants that haven't been run yet are skipped with a warning rather than failing
# the whole comparison.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS_DIR / "100_compare_cnn_variants"

# (display label, output-folder name, one-line description of what changed vs. baseline)
VARIANTS = [
    ("baseline",    "3_aec_cnn_reclassify",      "3_aec_cnn_reclassify.py (no change)"),
    ("pretrain",    "4_aec_cnn_pretrain",        "self-supervised encoder pretraining on full internal cohort"),
    ("skip",        "5_aec_cnn_skip",            "real residual/skip connection (block3(h2)+h2)"),
    ("bagging",     "6_aec_cnn_bagging",         "bootstrap-resampled ensemble members"),
    ("repeatedcv",  "7_aec_cnn_repeatedcv",      "5x repeated stratified-kfold OOF averaging"),
    ("film",        "8_aec_cnn_film",            "FiLM side-feature conditioning instead of late concat"),
    ("fcn",         "9_aec_cnn_fcn",             "standard FCN 1D-CNN architecture (Wang et al. 2017) instead of user-defined net"),
    ("crossattn",   "10_aec_cnn_crossattn",      "cross-attention side-feature conditioning instead of late concat"),
]


def load_variant_summary(label: str, folder: str) -> pd.DataFrame | None:
    csv_path = OUTPUTS_DIR / folder / "stage1_vs_stage2_summary.csv"
    if not csv_path.exists():
        print(f"[skip] {label}: {csv_path} not found yet -- run that script first")
        return None
    df = pd.read_csv(csv_path)
    df.insert(0, "variant", label)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    descriptions = {}
    for label, folder, desc in VARIANTS:
        df = load_variant_summary(label, folder)
        if df is not None:
            frames.append(df)
            descriptions[label] = desc

    if not frames:
        print("No variant summaries found -- run at least one of the scripts listed in the header first.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["description"] = combined["variant"].map(descriptions)

    cols = ["variant", "description", "cohort", "sens_delta", "spec_delta", "verdict",
            "ni_verdict", "sens_combined", "spec_combined", "ppv_combined", "acc_combined"]
    table = combined[cols].copy()
    table_path = OUTPUT_DIR / "comparison_summary.csv"
    table.to_csv(table_path, index=False)

    print("\n=== Stage-2 1D-CNN: baseline vs. improvement variants ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(table.to_string(index=False))
    print(f"\nSaved combined table to {table_path}")

    # ---------- grouped bar chart: spec_delta per variant, internal vs external ----------
    labels = [label for label, _, _ in VARIANTS if label in descriptions]
    internal_spec = []
    external_spec = []
    external_pass = []
    for label in labels:
        row_int = combined[(combined["variant"] == label) & (combined["cohort"] == "internal")]
        row_ext = combined[(combined["variant"] == label) & (combined["cohort"] == "external")]
        internal_spec.append(float(row_int["spec_delta"].iloc[0]) if len(row_int) else np.nan)
        external_spec.append(float(row_ext["spec_delta"].iloc[0]) if len(row_ext) else np.nan)
        external_pass.append(str(row_ext["verdict"].iloc[0]) if len(row_ext) else "N/A")

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5.5))
    bars_int = ax.bar(x - width / 2, internal_spec, width, label="internal spec_delta", color="#2a78d6")
    bars_ext = ax.bar(x + width / 2, external_spec, width, label="external spec_delta", color="#eda100")

    for bar, verdict in zip(bars_ext, external_pass):
        color = "#1a7a4c" if verdict == "PASS" else "#c0392b"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                verdict, ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=8, color=color, fontweight="bold")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("specificity delta vs. Stage-1-only")
    ax.set_title("Stage-2 1D-CNN: specificity improvement by variant\n(external label = PASS/FAIL non-inferiority + spec-up verdict)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig_path = OUTPUT_DIR / "spec_delta_comparison.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved comparison chart to {fig_path}")


if __name__ == "__main__":
    main()
