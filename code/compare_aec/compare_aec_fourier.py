from __future__ import annotations

# TP vs FP AEC-128 curve comparison in the Fourier (spatial-frequency) domain.
# Confusion-matrix membership (TP/FP) is taken from code/baseline/clinic-only_baseline.py's
# clinical-only classifier (same threshold, same OOF/frozen-model scores it already computes).
# Each patient's AEC-128 curve (raw or normalized) is FFT'd along the 128-slice axis; TP vs FP
# are compared as one-sided amplitude spectra. DC (k=0) is dropped -- it's just the curve's mean
# level, already compared in the time domain by compare_aec.py -- so this script isolates
# differences in the curve's *shape* (periodicity), not its overall level.
# Whole-spectrum comparison (RMSD permutation test over all frequency bins at once), not
# bin-by-bin per-frequency testing -- same "whole curve, not point-wise" convention as the
# time-domain script.
# Run: python code/compare_aec/compare_aec_fourier.py

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "fourier"

N_SLICES = 128
N_FREQ = N_SLICES // 2  # Nyquist bin index for 128 real samples (rfft gives bins 0..64)
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

# Same four normalizations as compare_aec.py, computed on the pooled (TP+FP) matrix per cohort
# before splitting back into groups -- each is FFT'd afterward, so "spectrum" always means the
# spectrum of that normalized curve, not of the raw curve.
NORMALIZATIONS = ["raw", "patient_norm", "global_zscore", "standard_scaler"]
NORM_LABELS = {
    "raw": "Raw AEC",
    "patient_norm": "Patient-normalized AEC",
    "global_zscore": "Global z-score AEC",
    "standard_scaler": "StandardScaler AEC",
}

COL_TP = "#2d2ad6"
COL_FP = "#af1b1b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

_spec = importlib.util.spec_from_file_location("clinic_only_baseline", PROJECT_ROOT / "code" / "baseline" / "clinic-only_baseline.py")
assert _spec is not None and _spec.loader is not None
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def load_raw_aec(xlsx_path: Path) -> pd.DataFrame:
    # Straight from the source sheet, cast to float -- no per-patient normalization.
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    out = aec[["PatientID"] + AEC_COLS].copy()
    out[AEC_COLS] = out[AEC_COLS].astype(float)
    return out


def normalize_curves(mat: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return mat
    if method == "patient_norm":
        return mat / mat.mean(axis=1, keepdims=True)
    if method == "global_zscore":
        return (mat - mat.mean()) / mat.std()
    if method == "standard_scaler":
        return StandardScaler().fit_transform(mat)
    raise ValueError(f"Unknown normalization method: {method}")


def to_spectrum(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # One-sided amplitude spectrum of each row (curve) via rfft, DC (k=0) dropped.
    # Standard 2/N scaling for k=1..N/2-1; the Nyquist bin (k=N/2) has no conjugate
    # partner so it keeps a bare 1/N scaling.
    fft_vals = np.fft.rfft(mat, axis=1)[:, 1:]  # drop DC, keep k=1..N_FREQ
    amp = np.abs(fft_vals) / N_SLICES
    amp[:, :-1] *= 2.0  # k=1..N_FREQ-1 get the one-sided factor of 2; Nyquist bin does not
    freq_cycles = np.arange(1, N_FREQ + 1)  # cycles per 128-slice curve
    return freq_cycles, amp


def spectrum_diff_test(mat_a: np.ndarray, mat_b: np.ndarray, n_perm: int = 2000, seed: int = 20260709) -> dict:
    # Whole-spectrum RMSD permutation test between two groups' amplitude spectra --
    # frequency bins are not tested one-by-one, only the full spectrum as one vector.
    mat = np.vstack([mat_a, mat_b])
    labels = np.array([0] * len(mat_a) + [1] * len(mat_b))

    def spectrum_stat(lab: np.ndarray) -> tuple[float, np.ndarray]:
        mean_a = mat[lab == 0].mean(axis=0)
        mean_b = mat[lab == 1].mean(axis=0)
        deviation = mean_a - mean_b
        rmsd = float(np.sqrt(np.mean(deviation ** 2)))
        return rmsd, deviation

    obs_stat, obs_deviation = spectrum_stat(labels)
    peak_idx = int(np.argmax(np.abs(obs_deviation)))

    rng = np.random.default_rng(seed)
    perm_labels = labels.copy()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(perm_labels)
        perm_stats[i] = spectrum_stat(perm_labels)[0]
    p_value = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)

    return {
        "n_tp": int(len(mat_a)), "n_fp": int(len(mat_b)),
        "spectrum_rmsd": obs_stat,
        "peak_freq_cycle": int(peak_idx + 1),
        "peak_deviation": float(obs_deviation[peak_idx]),
        "direction": f"TP {'>' if obs_deviation[peak_idx] > 0 else '<'} FP",
        "n_perm": n_perm, "p_value": float(p_value),
    }


def style_axes(ax) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_tp_vs_fp_spectrum(ax, freq: np.ndarray, mat_tp: np.ndarray, mat_fp: np.ndarray, r: dict, title: str, ylabel: str) -> None:
    for mat, label, n, color in [(mat_tp, "TP", r["n_tp"], COL_TP), (mat_fp, "FP", r["n_fp"], COL_FP)]:
        mean = mat.mean(axis=0)
        ci = 1.96 * stats.sem(mat, axis=0)
        ax.plot(freq, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(freq, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.set_xlim(freq.min(), freq.max())
    ax.set_xlabel("Spatial frequency (cycles per 128-slice scan)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    note = f"RMSD={r['spectrum_rmsd']:.4f}, p={r['p_value']:.3g}, peak Δ={r['peak_deviation']:.4f} @ {r['peak_freq_cycle']} cyc"
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=12, color=INK_MUTED, va="bottom")
    style_axes(ax)


N_SAMPLE_CURVES = 3
SAMPLE_SEED = 20260709

# Every sampled spectrum gets its own distinct color (not just a shared per-group color) so
# all lines in the samples-only figure stay visually distinguishable from each other --
# TP samples stay in cool hues, FP samples in warm hues, but no two lines share a color.
TP_SAMPLE_COLORS = ["#1b4fd8", "#00acc1", "#7a4fd8", "#2e8b57", "#1f77b4", "#4b0082"]
FP_SAMPLE_COLORS = ["#d8271b", "#e0762c", "#c22a68", "#a0522d", "#b8860b", "#8b0000"]


def plot_tp_vs_fp_spectrum_samples(ax, freq: np.ndarray, mat_tp: np.ndarray, ids_tp: np.ndarray, mat_fp: np.ndarray, ids_fp: np.ndarray,
                                    title: str, ylabel: str, n_samples: int = N_SAMPLE_CURVES, seed: int = SAMPLE_SEED) -> list[dict]:
    # Individual raw patient spectra (no mean/CI), each labeled with its PatientID so the
    # exact sampled patients can be traced back -- saved as its own figure, separate from
    # the mean+CI comparison plot.
    rng = np.random.default_rng(seed)
    sampled = []
    for mat, ids, label, colors in [(mat_tp, ids_tp, "TP", TP_SAMPLE_COLORS), (mat_fp, ids_fp, "FP", FP_SAMPLE_COLORS)]:
        sample_idx = rng.choice(len(mat), size=min(n_samples, len(mat)), replace=False)
        for j, i in enumerate(sample_idx):
            pid = ids[i]
            ax.plot(freq, mat[i], color=colors[j % len(colors)], linewidth=1.4, alpha=0.9, label=f"{label} {pid}")
            sampled.append({"group": label, "PatientID": pid})
    ax.set_xlim(freq.min(), freq.max())
    ax.set_xlabel("Spatial frequency (cycles per 128-slice scan)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    style_axes(ax)
    return sampled


def run_cohort(cohort: str, meta: pd.DataFrame, y: np.ndarray, score: np.ndarray, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = baseline.build_group_rows(meta, y, score, th)
    aec = load_raw_aec(xlsx_path)

    df = rows[["PatientID", "group"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(rows), f"{cohort}: TP/FP merge dropped rows"

    full_mat = df[AEC_COLS].to_numpy()
    group_arr = df["group"].to_numpy()
    id_arr = df["PatientID"].to_numpy()

    results = []
    for variant in NORMALIZATIONS:
        # Normalize the pooled TP+FP matrix first, then split -- so global_zscore's
        # scalar mean/std and standard_scaler's per-slice mean/std reflect the whole
        # compared population, not each group fit in isolation. FFT is applied after
        # normalization, per patient curve.
        norm_mat = normalize_curves(full_mat, variant)
        freq, spectrum_mat = to_spectrum(norm_mat)
        group_arr_s = group_arr
        mat_tp = spectrum_mat[group_arr_s == "TP"]
        mat_fp = spectrum_mat[group_arr_s == "FP"]
        ids_tp = id_arr[group_arr_s == "TP"]
        ids_fp = id_arr[group_arr_s == "FP"]
        ylabel = f"{NORM_LABELS[variant]} amplitude spectrum"

        # Whole-spectrum RMSD permutation test between TP and FP amplitude spectra.
        r = spectrum_diff_test(mat_tp, mat_fp)

        out_dir = OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_tp_vs_fp_spectrum(ax, freq, mat_tp, mat_fp, r, f"{cohort} ({title_suffix}): {NORM_LABELS[variant]}-128 FFT, TP vs FP", ylabel)
        fig.tight_layout()
        fig_path = out_dir / f"tp_vs_fp_aec_spectrum_{variant}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[{cohort}/{variant}] n_TP={r['n_tp']} n_FP={r['n_fp']} spectrum_RMSD={r['spectrum_rmsd']:.4f} "
              f"perm_p={r['p_value']:.4g} peak_freq_cycle={r['peak_freq_cycle']} peak_delta={r['peak_deviation']:.4f}")
        print(f"Saved {fig_path}")

        fig_s, ax_s = plt.subplots(figsize=(8, 5.5))
        sampled = plot_tp_vs_fp_spectrum_samples(ax_s, freq, mat_tp, ids_tp, mat_fp, ids_fp,
                                                   f"{cohort} ({title_suffix}): {NORM_LABELS[variant]}-128 FFT samples, TP vs FP (n={N_SAMPLE_CURVES} each)",
                                                   ylabel)
        fig_s.tight_layout()
        samples_path = out_dir / f"tp_vs_fp_aec_spectrum_{variant}_samples.png"
        fig_s.savefig(samples_path, dpi=200, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved {samples_path}")

        sampled_ids_path = out_dir / f"tp_vs_fp_aec_spectrum_{variant}_samples_patient_ids.csv"
        pd.DataFrame(sampled).to_csv(sampled_ids_path, index=False, encoding="utf-8-sig")
        print(f"Saved {sampled_ids_path}")

        summary_path = out_dir / f"tp_vs_fp_aec_spectrum_{variant}_summary.csv"
        pd.DataFrame([{"cohort": cohort, "normalization": variant, **r}]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved {summary_path}")

        results.append({"cohort": cohort, "normalization": variant, **r})
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int, y_int = baseline.load_cohort(baseline.INTERNAL_XLSX)
    x_raw_int = baseline.raw_clinical_matrix(meta_int)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw_int)
    x_int = baseline.apply_clinical_standardizer(x_raw_int, med, mu, sd)
    oof = baseline.oof_scores(x_int, y_int)

    model = baseline.fit_baseline_model(x_int, y_int)
    meta_ext, y_ext = baseline.load_cohort(baseline.EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)
    score_ext = model.decision_function(x_ext)

    th = baseline.threshold_for_sensitivity(y_int, oof, baseline.TARGET_SENSITIVITY)

    results = [
        *run_cohort("gangnam", meta_int, y_int, oof, th, baseline.INTERNAL_XLSX, "internal, OOF"),
        *run_cohort("sinchon", meta_ext, y_ext, score_ext, th, baseline.EXTERNAL_XLSX, "external, frozen internal model"),
    ]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_spectrum_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
