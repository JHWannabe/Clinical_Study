from __future__ import annotations

"""
Final AEC phenotype-enrichment pipeline for low SMI
===================================================

This script is meant to be the clean student handoff version.

Core manuscript idea
--------------------
We do NOT claim that AEC replaces the clinical low-SMI model.
The cleaner claim is:

    Among patients already classified as clinically high-risk,
    an AEC-derived morphology score separates a low-SMI-enriched phenotype
    from a low-SMI-depleted phenotype.

In plain terms:

    1. Use clinical variables first: age, sex, height, weight.
    2. Take the clinically high-risk stratum.
    3. Inside that stratum, use the AEC morphology score.
    4. Compare AEC-high vs AEC-low observed low-SMI prevalence.

Why 20% rather than 25%?
------------------------
The primary threshold is a quintile:

    clinical high-risk = top 20% of clinical score in the internal cohort
    AEC-high phenotype = top 20% of AEC score among clinical-high patients
    AEC-low phenotype  = bottom 20% of AEC score among clinical-high patients

This is more elegant than 25% because "top/bottom quintile" is a common,
pre-specifiable enrichment-stratum definition. It is not a Youden cutoff,
not an external-cohort optimized cutoff, and not a hand-tuned diagnostic
threshold. The script still reports 25% as a sensitivity analysis.

What this script assumes
------------------------
Input raw data:

    work/data_cache/g1090.xlsx
    work/data_cache/sdata.xlsx

These workbooks should have:

    metadata sheet:
        PatientAge, PatientSex, Height, Weight, TAMA, IMATA, BMI, SMI

    aec_128 / aec_cropped sheets:
        aec_1 ... aec_128 or aec_1 ... aec_N

The final AEC morphology score used here is:

    vote_only_logit_l1

It is produced by the direct-vote CNN branch-probability pipeline:

    work/aec_lock_smoothed_deesc_gate.py
    work/aec_region_cnn_pattern_gate.py
    work/aec_direct_vote_auc_boost.py

If their output files do not exist, this master script can run them first.
That makes the workflow reproducible from a clean output folder, while still
keeping the heavy CNN training code in its original module.

Recommended command
-------------------
From the project root:

    python work/aec_final_global_quintile_phenotype_pipeline.py

Outputs
-------
    outputs/aec_final_global_quintile_phenotype/

Main files:

    01_quintile_vs_quartile_enrichment.csv
    02_or_and_diagnostic_metrics.csv
    03_four_cell_characteristics.csv
    04_low_smi_subtype_characteristics.csv
    05_low_smi_subtype_feature_tests.csv
    figure_quintile_enrichment.png
    final_summary.json
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 0. Project paths and final analysis constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "work" / "data_cache"
INTERNAL_XLSX = DATA_DIR / "g1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sdata.xlsx"

OUT_DIR = PROJECT_ROOT / "outputs" / "aec_final_global_quintile_phenotype"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# This file is created by work/aec_direct_vote_auc_boost.py.
# It contains row-level clinical score and AEC morphology scores.
DIRECT_VOTE_SCORE_CSV = PROJECT_ROOT / "outputs" / "aec_direct_vote_auc_boost" / "direct_vote_auc_boost_scores.csv"

# These are the upstream scripts needed to create DIRECT_VOTE_SCORE_CSV
# from raw xlsx.
#
# 1. LOCK_SCRIPT:
#    Searches smoothed, patient-wise normalized AEC features in the internal
#    cohort and locks the interpretable feature set.
#
# 2. PATTERN_GATE_SCRIPT:
#    Trains region-guided CNN branches and saves per-patient branch/vote
#    probabilities to direct_vote_probabilities.npz.
#
# 3. DIRECT_VOTE_AUC_SCRIPT:
#    Converts those CNN branch probabilities into row-level AEC scores,
#    including the final vote_only_logit_l1 score used here.
LOCK_SCRIPT = PROJECT_ROOT / "work" / "aec_lock_smoothed_deesc_gate.py"
PATTERN_GATE_SCRIPT = PROJECT_ROOT / "work" / "aec_region_cnn_pattern_gate.py"
DIRECT_VOTE_AUC_SCRIPT = PROJECT_ROOT / "work" / "aec_direct_vote_auc_boost.py"

# Final selected AEC morphology score.
# It is an AEC-only score: direct-vote CNN branch/vote features are combined
# with sparse L1 logistic regression. Clinical variables are not included here.
AEC_SCORE_COLUMN = "vote_only_logit_l1"

# Primary quantile. Report 0.25 too, but use 0.20 as manuscript primary.
PRIMARY_Q = 0.20
SENSITIVITY_Q = 0.25


# ---------------------------------------------------------------------------
# 1. Optional upstream score generation
# ---------------------------------------------------------------------------

def run_if_needed() -> None:
    """
    Make sure the row-level AEC score file exists.

    If the final score file is already present, we do not rerun CNN training.
    If it is absent, we run the upstream scripts that create it.

    This makes the current script usable in two modes:

    1. Clean run:
       Delete outputs/aec_direct_vote_auc_boost and run this script.
       It will regenerate the needed score file.

    2. Fast rerun:
       Keep the score file and rerun only the final phenotype tables.
    """
    if DIRECT_VOTE_SCORE_CSV.exists():
        print(f"[OK] Found existing AEC score file: {DIRECT_VOTE_SCORE_CSV}")
        return

    print("[INFO] AEC score file is missing. Running upstream direct-vote pipeline.")
    for script in [LOCK_SCRIPT, PATTERN_GATE_SCRIPT, DIRECT_VOTE_AUC_SCRIPT]:
        if not script.exists():
            raise FileNotFoundError(f"Required upstream script is missing: {script}")
        print(f"[RUN] {script.name}")
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(PROJECT_ROOT))

    if not DIRECT_VOTE_SCORE_CSV.exists():
        raise FileNotFoundError(f"Upstream scripts finished, but score file was not created: {DIRECT_VOTE_SCORE_CSV}")


# ---------------------------------------------------------------------------
# 2. Data loading helpers
# ---------------------------------------------------------------------------

def load_metadata(path: Path, cohort: str) -> pd.DataFrame:
    """
    Load only patient-level metadata.

    We do not edit the raw spreadsheet. We create a processed copy in memory.
    """
    meta = pd.read_excel(path, sheet_name="metadata")
    out = pd.DataFrame(index=np.arange(len(meta)))
    out["cohort"] = cohort
    out["row_index"] = np.arange(len(meta))
    out["PatientID"] = meta.get("PatientID", pd.Series(np.arange(len(meta)))).astype(str)
    out["PatientAge"] = pd.to_numeric(meta["PatientAge"], errors="coerce")
    out["PatientSex"] = meta["PatientSex"].astype(str).str.upper()
    out["male"] = (out["PatientSex"] == "M").astype(int)
    out["Height"] = pd.to_numeric(meta["Height"], errors="coerce")
    out["Weight"] = pd.to_numeric(meta["Weight"], errors="coerce")
    out["BMI"] = pd.to_numeric(meta["BMI"], errors="coerce")
    out["TAMA"] = pd.to_numeric(meta["TAMA"], errors="coerce")
    out["IMATA"] = pd.to_numeric(meta["IMATA"], errors="coerce")

    # SMI is included in the provided metadata. If a future file lacks it,
    # compute it from TAMA and height.
    if "SMI" in meta.columns:
        out["SMI"] = pd.to_numeric(meta["SMI"], errors="coerce")
    else:
        out["SMI"] = np.nan
    missing_smi = ~np.isfinite(out["SMI"])
    out.loc[missing_smi, "SMI"] = out.loc[missing_smi, "TAMA"] / ((out.loc[missing_smi, "Height"] / 100.0) ** 2)

    # Body-composition descriptors used only for subtype interpretation.
    out["IMATA_fraction"] = out["IMATA"] / (out["TAMA"] + out["IMATA"])
    out["TAMA_per_weight"] = out["TAMA"] / out["Weight"]
    out["IMATA_per_weight"] = out["IMATA"] / out["Weight"]
    out["log_TAMA_to_IMATA"] = np.log((out["TAMA"] + 1e-3) / (out["IMATA"] + 1e-3))
    out["Manufacturer"] = meta.get("Manufacturer", pd.Series(["unknown"] * len(meta))).astype(str)
    return out


def load_patient_table() -> pd.DataFrame:
    """
    Merge raw metadata with model scores.

    The score file contains:
        - y_low_smi
        - clinical_score
        - AEC score columns, including vote_only_logit_l1
    """
    meta = pd.concat(
        [
            load_metadata(INTERNAL_XLSX, "g1090_internal"),
            load_metadata(EXTERNAL_XLSX, "sdata_external"),
        ],
        ignore_index=True,
    )
    scores = pd.read_csv(DIRECT_VOTE_SCORE_CSV)
    if AEC_SCORE_COLUMN not in scores.columns:
        raise KeyError(f"{AEC_SCORE_COLUMN} not found in {DIRECT_VOTE_SCORE_CSV}")

    scores = scores.rename(columns={"dataset": "cohort", "y_low_smi": "low_smi", AEC_SCORE_COLUMN: "aec_score"})
    keep = ["cohort", "row_index", "low_smi", "clinical_score", "aec_score"]
    merged = meta.merge(scores[keep], on=["cohort", "row_index"], how="inner")

    expected = {"g1090_internal": 1090, "sdata_external": 926}
    observed = merged["cohort"].value_counts().to_dict()
    print(f"[INFO] merged patient counts: {observed}")
    for cohort, n in expected.items():
        if observed.get(cohort, 0) != n:
            print(f"[WARN] Expected {n} rows for {cohort}, observed {observed.get(cohort, 0)}.")
    return merged


# ---------------------------------------------------------------------------
# 3. Metric helpers
# ---------------------------------------------------------------------------

def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def fisher_high_vs_low(y: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict[str, float | int]:
    """
    Compare observed low-SMI prevalence between two strata.

    high = AEC-high phenotype within clinical-high patients
    low  = AEC-low phenotype within clinical-high patients
    """
    y = np.asarray(y, dtype=int)
    high = np.asarray(high, dtype=bool)
    low = np.asarray(low, dtype=bool)

    high_events = int(np.sum(high & (y == 1)))
    high_nonevents = int(np.sum(high & (y == 0)))
    low_events = int(np.sum(low & (y == 1)))
    low_nonevents = int(np.sum(low & (y == 0)))

    high_n = high_events + high_nonevents
    low_n = low_events + low_nonevents
    high_rate = safe_rate(high_events, high_n)
    low_rate = safe_rate(low_events, low_n)

    if high_n > 0 and low_n > 0:
        odds_ratio, p_value = stats.fisher_exact(
            [[high_events, high_nonevents], [low_events, low_nonevents]],
            alternative="greater",
        )
    else:
        odds_ratio, p_value = np.nan, np.nan

    return {
        "aec_high_n": high_n,
        "aec_high_low_smi_n": high_events,
        "aec_high_low_smi_rate": high_rate,
        "aec_low_n": low_n,
        "aec_low_low_smi_n": low_events,
        "aec_low_low_smi_rate": low_rate,
        "absolute_risk_separation": high_rate - low_rate,
        "risk_ratio_high_vs_low": high_rate / low_rate if low_rate and low_rate > 0 else np.inf,
        "odds_ratio_high_vs_low": float(odds_ratio),
        "fisher_p_high_gt_low": float(p_value),
    }


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    """
    Standard diagnostic metrics for C+, A+, C OR A, C AND A.

    These are secondary. The main paper message is phenotype enrichment,
    not binary diagnostic replacement.
    """
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    return {
        "n_positive": int(np.sum(pred)),
        "events_in_positive": tp,
        "positive_rate_ppv": safe_rate(tp, tp + fp),
        "sensitivity_capture": safe_rate(tp, tp + fn),
        "specificity": safe_rate(tn, tn + fp),
        "accuracy": safe_rate(tp + tn, len(y)),
        "positive_workload_fraction": safe_rate(int(np.sum(pred)), len(y)),
    }


def group_characteristics(df: pd.DataFrame, mask: pd.Series, label: str, cohort: str) -> dict[str, object]:
    """
    Summarize clinical/body-composition features for a subgroup.
    """
    sub = df[mask]
    y = df["low_smi"].to_numpy(int)
    m = mask.to_numpy(bool)
    row: dict[str, object] = {
        "cohort": cohort,
        "group": label,
        "n": int(m.sum()),
        "low_smi_n": int(np.sum(y[m] == 1)),
        "low_smi_rate": float(np.mean(y[m])) if m.sum() else np.nan,
        "male_pct": float(sub["male"].mean()) if len(sub) else np.nan,
    }
    for col in [
        "PatientAge",
        "Height",
        "Weight",
        "BMI",
        "TAMA",
        "IMATA",
        "SMI",
        "IMATA_fraction",
        "TAMA_per_weight",
        "IMATA_per_weight",
        "log_TAMA_to_IMATA",
        "clinical_score",
        "aec_score",
    ]:
        row[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else np.nan
        row[f"{col}_sd"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
    return row


# ---------------------------------------------------------------------------
# 4. Main phenotype analysis
# ---------------------------------------------------------------------------

def add_global_flags(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Lock all thresholds using internal/Gangnam data only.

    For each q:
        clinical_high = top q of internal clinical_score
        AEC global + = top q of internal aec_score

    The primary enrichment analysis does not use global AEC+ directly.
    It uses AEC-high and AEC-low among clinical-high patients:
        AEC-high within C+ = top q of AEC score among internal C+
        AEC-low  within C+ = bottom q of AEC score among internal C+
    """
    out = df.copy()
    internal = out[out["cohort"].eq("g1090_internal")]

    clinical_cut = float(np.quantile(internal["clinical_score"], 1.0 - q))
    internal_cpos = internal["clinical_score"] >= clinical_cut
    aec_global_cut = float(np.quantile(internal["aec_score"], 1.0 - q))
    aec_high_cut_in_cpos = float(np.quantile(internal.loc[internal_cpos, "aec_score"], 1.0 - q))
    aec_low_cut_in_cpos = float(np.quantile(internal.loc[internal_cpos, "aec_score"], q))

    out["clinical_pos"] = out["clinical_score"] >= clinical_cut
    out["aec_pos_global"] = out["aec_score"] >= aec_global_cut
    out["aec_high_in_clinical_pos"] = out["clinical_pos"] & (out["aec_score"] >= aec_high_cut_in_cpos)
    out["aec_low_in_clinical_pos"] = out["clinical_pos"] & (out["aec_score"] <= aec_low_cut_in_cpos)
    out["C_or_A"] = out["clinical_pos"] | out["aec_pos_global"]
    out["C_and_A"] = out["clinical_pos"] & out["aec_pos_global"]
    out["cell"] = np.select(
        [
            out["clinical_pos"] & out["aec_pos_global"],
            out["clinical_pos"] & (~out["aec_pos_global"]),
            (~out["clinical_pos"]) & out["aec_pos_global"],
        ],
        ["C+A+", "C+A-", "C-A+"],
        default="C-A-",
    )

    thresholds = {
        "q": q,
        "clinical_top_cut": clinical_cut,
        "aec_global_top_cut": aec_global_cut,
        "aec_high_cut_within_clinical_pos": aec_high_cut_in_cpos,
        "aec_low_cut_within_clinical_pos": aec_low_cut_in_cpos,
    }
    return out, thresholds


def enrichment_table(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Primary table: within clinically high-risk patients, compare AEC-high vs AEC-low.
    """
    flagged, thresholds = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        y = sub["low_smi"].to_numpy(int)
        clinical_pos = sub["clinical_pos"].to_numpy(bool)
        high = sub["aec_high_in_clinical_pos"].to_numpy(bool)
        low = sub["aec_low_in_clinical_pos"].to_numpy(bool)
        clinical_events = int(np.sum(clinical_pos & (y == 1)))
        clinical_n = int(np.sum(clinical_pos))
        row = {
            "q": q,
            "cohort": cohort,
            "clinical_positive_n": clinical_n,
            "clinical_positive_low_smi_n": clinical_events,
            "clinical_positive_low_smi_rate": safe_rate(clinical_events, clinical_n),
            **fisher_high_vs_low(y, high, low),
        }
        row.update(thresholds)
        rows.append(row)
    return pd.DataFrame(rows)


def or_and_table(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Secondary diagnostic table for C+, A+, C OR A, C AND A.

    This is not the primary message, but it answers a natural reviewer question:
    what happens if these are treated as simple binary flags?
    """
    flagged, _ = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        y = sub["low_smi"].to_numpy(int)
        for label, pred in [
            ("Clinical+", sub["clinical_pos"]),
            ("AEC+ global", sub["aec_pos_global"]),
            ("Clinical+ OR AEC+", sub["C_or_A"]),
            ("Clinical+ AND AEC+", sub["C_and_A"]),
        ]:
            rows.append({"q": q, "cohort": cohort, "rule": label, **binary_metrics(y, pred.to_numpy(bool))})
    return pd.DataFrame(rows)


def cell_characteristics(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Four-cell table: C+A+, C+A-, C-A+, C-A-.

    This is useful for explaining that AEC is not a standalone rescue flag.
    It mostly refines clinical-positive patients.
    """
    flagged, _ = add_global_flags(df, q)
    rows: list[dict[str, object]] = []
    for cohort, sub in flagged.groupby("cohort"):
        for cell in ["C+A+", "C+A-", "C-A+", "C-A-"]:
            rows.append(group_characteristics(sub, sub["cell"].eq(cell), cell, cohort))
    return pd.DataFrame(rows)


def low_smi_subtype_tables(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Among actual low-SMI patients, compare AEC+ vs AEC-.

    This tells us what biological/clinical phenotype AEC is capturing.
    In previous results, AEC-positive low-SMI patients looked leaner:
        lower BMI, lower weight, higher TAMA per weight, lower IMATA fraction.
    """
    flagged, _ = add_global_flags(df, q)
    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    features = [
        "PatientAge",
        "male",
        "Height",
        "Weight",
        "BMI",
        "TAMA",
        "IMATA",
        "SMI",
        "IMATA_fraction",
        "TAMA_per_weight",
        "IMATA_per_weight",
        "log_TAMA_to_IMATA",
    ]

    for cohort, sub0 in flagged.groupby("cohort"):
        sub = sub0[sub0["low_smi"].eq(1)].copy()
        groups = {
            "AEC+ lowSMI": sub["aec_pos_global"],
            "AEC- lowSMI": ~sub["aec_pos_global"],
            "C+A+ lowSMI": sub["clinical_pos"] & sub["aec_pos_global"],
            "other lowSMI": ~(sub["clinical_pos"] & sub["aec_pos_global"]),
        }
        for label, mask in groups.items():
            summary_rows.append(group_characteristics(sub, mask, label, cohort))

        # Statistical tests for AEC+ vs AEC- among low-SMI patients.
        mask = sub["aec_pos_global"].to_numpy(bool)
        for col in features:
            a = pd.to_numeric(sub.loc[mask, col], errors="coerce").dropna().to_numpy(float)
            b = pd.to_numeric(sub.loc[~mask, col], errors="coerce").dropna().to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                p_value = np.nan
            elif col == "male":
                p_value = float(
                    stats.fisher_exact(
                        [[int(np.sum(a == 1)), int(np.sum(a == 0))], [int(np.sum(b == 1)), int(np.sum(b == 0))]],
                        alternative="two-sided",
                    ).pvalue
                )
            else:
                p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            test_rows.append(
                {
                    "q": q,
                    "cohort": cohort,
                    "comparison": "AEC+ vs AEC- among lowSMI",
                    "feature": col,
                    "aec_positive_n": int(mask.sum()),
                    "aec_negative_n": int((~mask).sum()),
                    "aec_positive_mean": float(np.mean(a)) if len(a) else np.nan,
                    "aec_negative_mean": float(np.mean(b)) if len(b) else np.nan,
                    "difference": (float(np.mean(a)) - float(np.mean(b))) if len(a) and len(b) else np.nan,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------

def plot_quintile_enrichment(enrich: pd.DataFrame, out_path: Path) -> None:
    """
    Compact figure for the manuscript:
        clinical-high overall vs AEC-low vs AEC-high.
    """
    q20 = enrich[np.isclose(enrich["q"], PRIMARY_Q)].copy()
    cohorts = ["g1090_internal", "sdata_external"]
    labels = ["Gangnam internal", "Sinchon external"]
    colors = ["#6B7280", "#4C78A8", "#D04F5B"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, constrained_layout=True)
    for ax, cohort, label in zip(axes, cohorts, labels):
        row = q20[q20["cohort"].eq(cohort)].iloc[0]
        vals = [
            row["clinical_positive_low_smi_rate"],
            row["aec_low_low_smi_rate"],
            row["aec_high_low_smi_rate"],
        ]
        ns = [
            int(row["clinical_positive_n"]),
            int(row["aec_low_n"]),
            int(row["aec_high_n"]),
        ]
        bars = ax.bar(["Clinical high", "AEC low", "AEC high"], vals, color=colors, width=0.72)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(0, max(0.72, max(vals) + 0.10))
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Observed low-SMI prevalence")
        for bar, val, n in zip(bars, vals, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.018,
                f"{val:.1%}\nn={n}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("AEC morphology enriches low-SMI burden within clinical high-risk patients", fontweight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_if_needed()
    patient = load_patient_table()

    # Report 20% as primary and 25% as sensitivity.
    enrich = pd.concat([enrichment_table(patient, PRIMARY_Q), enrichment_table(patient, SENSITIVITY_Q)], ignore_index=True)
    or_and = pd.concat([or_and_table(patient, PRIMARY_Q), or_and_table(patient, SENSITIVITY_Q)], ignore_index=True)
    cells = pd.concat([cell_characteristics(patient, PRIMARY_Q), cell_characteristics(patient, SENSITIVITY_Q)], ignore_index=True)
    subtype_20, tests_20 = low_smi_subtype_tables(patient, PRIMARY_Q)
    subtype_25, tests_25 = low_smi_subtype_tables(patient, SENSITIVITY_Q)
    subtypes = pd.concat([subtype_20.assign(q=PRIMARY_Q), subtype_25.assign(q=SENSITIVITY_Q)], ignore_index=True)
    tests = pd.concat([tests_20, tests_25], ignore_index=True)

    patient.to_csv(OUT_DIR / "00_patient_level_merged_scores.csv", index=False)
    enrich.to_csv(OUT_DIR / "01_quintile_vs_quartile_enrichment.csv", index=False)
    or_and.to_csv(OUT_DIR / "02_or_and_diagnostic_metrics.csv", index=False)
    cells.to_csv(OUT_DIR / "03_four_cell_characteristics.csv", index=False)
    subtypes.to_csv(OUT_DIR / "04_low_smi_subtype_characteristics.csv", index=False)
    tests.to_csv(OUT_DIR / "05_low_smi_subtype_feature_tests.csv", index=False)
    plot_quintile_enrichment(enrich, OUT_DIR / "figure_quintile_enrichment.png")

    summary = {
        "primary_quantile": PRIMARY_Q,
        "sensitivity_quantile": SENSITIVITY_Q,
        "why_20_percent": "Top/bottom quintile is a conventional pre-specifiable phenotype-enrichment stratum and is less arbitrary than an optimized diagnostic cutoff.",
        "aec_score": AEC_SCORE_COLUMN,
        "primary_claim": "AEC morphology stratifies low-SMI burden among clinically high-risk patients.",
        "important_caution": "This is not an AUC-improvement claim and not a replacement diagnostic test.",
        "outputs": {
            "patient_level": str(OUT_DIR / "00_patient_level_merged_scores.csv"),
            "enrichment": str(OUT_DIR / "01_quintile_vs_quartile_enrichment.csv"),
            "or_and": str(OUT_DIR / "02_or_and_diagnostic_metrics.csv"),
            "four_cells": str(OUT_DIR / "03_four_cell_characteristics.csv"),
            "subtypes": str(OUT_DIR / "04_low_smi_subtype_characteristics.csv"),
            "feature_tests": str(OUT_DIR / "05_low_smi_subtype_feature_tests.csv"),
            "figure": str(OUT_DIR / "figure_quintile_enrichment.png"),
        },
    }
    with (OUT_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPRIMARY ENRICHMENT TABLE")
    show_cols = [
        "q",
        "cohort",
        "clinical_positive_n",
        "clinical_positive_low_smi_rate",
        "aec_low_n",
        "aec_low_low_smi_rate",
        "aec_high_n",
        "aec_high_low_smi_rate",
        "absolute_risk_separation",
        "risk_ratio_high_vs_low",
        "odds_ratio_high_vs_low",
        "fisher_p_high_gt_low",
    ]
    print(enrich[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nOR / AND SECONDARY DIAGNOSTIC TABLE")
    print(or_and.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nLOW-SMI SUBTYPE TESTS, PRIMARY Q=20%, TOP FEATURES")
    top_tests = (
        tests[tests["q"].eq(PRIMARY_Q)]
        .sort_values(["cohort", "p_value"])
        .groupby("cohort")
        .head(8)
    )
    print(top_tests.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
