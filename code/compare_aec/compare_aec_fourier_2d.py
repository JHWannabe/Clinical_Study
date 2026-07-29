from __future__ import annotations

# TP vs FP AEC-128 comparison via true 2D DFT.
# Confusion-matrix membership (TP/FP) is taken from code/baseline/clinic-only_baseline.py's
# clinical-only classifier (same threshold, same OOF/frozen-model scores it already computes).
#
# Unlike compare_aec_fourier.py (1D FFT of each patient's own 128-slice curve), this script
# builds one genuine 2D image per group and per normalization variant:
#   x-axis = slice index (1..128)         -- real spatial axis, shared across all patients
#   y-axis = AEC value, binned            -- real measurement axis, shared across all patients
#   pixel  = fraction of that group's patients whose curve passes through that (slice, value) bin
# Both axes carry physical meaning (unlike stacking patients as rows, which would need an
# arbitrary patient ordering). np.fft.fft2 is applied to this density image, one 2D amplitude
# spectrum per group, and TP vs FP spectra are compared as whole images (RMSD permutation test
# over all pixels at once), not pixel-by-pixel -- same "whole curve, not point-wise" convention
# used throughout this project. Each normalization variant is tested twice: once on the raw
# |FFT| magnitude ("linear") and once on log1p(|FFT|) ("log") -- log compresses the few huge
# low-frequency bins so smaller high-frequency differences aren't swamped in the RMSD/permutation
# test, not just in the plot.
# Run: python code/compare_aec/compare_aec_fourier_2d.py

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "fourier_2d"

N_SLICES = 128
N_Y_BINS = 64
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

NORMALIZATIONS = ["raw", "patient_norm", "global_zscore"]
NORM_LABELS = {
    "raw": "Raw AEC",
    "patient_norm": "Patient-normalized AEC",
    "global_zscore": "Global z-score AEC",
}

# "linear" tests the raw |FFT| magnitude (matches the original analysis); "log" applies log1p to
# the magnitude before computing RMSD/peak/permutation, so the test itself is run in log-amplitude
# space (down-weights the few huge low-frequency bins, lets small high-frequency differences count).
FFT_SCALES = ["linear", "log"]
FFT_SCALE_LABELS = {
    "linear": "linear |FFT| magnitude",
    "log": "log1p(|FFT|) magnitude",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

N_PERM = 2000
PERM_SEED = 20260709

_spec = importlib.util.spec_from_file_location("clinic_only_baseline", PROJECT_ROOT / "code" / "baseline" / "clinic-only_baseline.py")
assert _spec is not None and _spec.loader is not None
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def load_raw_aec(xlsx_path: Path) -> pd.DataFrame:
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
    raise ValueError(f"Unknown normalization method: {method}")


def digitize_curves(mat: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # Maps every (patient, slice) value to a y-bin index in [0, N_Y_BINS - 1].
    idx = np.digitize(mat, bin_edges[1:-1])
    return np.clip(idx, 0, len(bin_edges) - 2)


def build_density_image(bin_idx_group: np.ndarray, n_bins: int, n_slices: int) -> np.ndarray:
    # One 2D histogram in a single bincount call: combine (bin, slice) into one flat index
    # instead of looping per slice. Density = count / group size, so every column (slice)
    # sums to 1 regardless of group size -- TP and FP images stay directly comparable.
    n_patients = bin_idx_group.shape[0]
    combined = bin_idx_group + np.arange(n_slices)[None, :] * n_bins
    counts = np.bincount(combined.ravel(), minlength=n_bins * n_slices).reshape(n_slices, n_bins).T
    return counts / n_patients


def spectrum_2d(image: np.ndarray, log: bool = False) -> np.ndarray:
    mag = np.abs(np.fft.fft2(image)) / image.size
    if log:
        mag = np.log1p(mag)
    return np.fft.fftshift(mag)


def image_2d_diff_test(bin_idx: np.ndarray, is_tp: np.ndarray, n_bins: int, n_slices: int,
                        n_perm: int = N_PERM, seed: int = PERM_SEED, log_fft: bool = False) -> dict:
    center = (n_bins // 2, n_slices // 2)

    def stat(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_tp = build_density_image(bin_idx[mask], n_bins, n_slices)
        img_fp = build_density_image(bin_idx[~mask], n_bins, n_slices)
        mag_tp = spectrum_2d(img_tp, log=log_fft)
        mag_fp = spectrum_2d(img_fp, log=log_fft)
        mag_tp[center] = 0.0  # DC = overall density level, identical by construction (columns sum to 1)
        mag_fp[center] = 0.0
        deviation = mag_tp - mag_fp
        rmsd = float(np.sqrt(np.mean(deviation ** 2)))
        return rmsd, deviation, mag_tp, mag_fp, img_tp, img_fp

    obs_stat, obs_deviation, obs_mag_tp, obs_mag_fp, obs_img_tp, obs_img_fp = stat(is_tp)
    peak_flat = int(np.argmax(np.abs(obs_deviation)))
    peak_row, peak_col = np.unravel_index(peak_flat, obs_deviation.shape)
    freq_y = np.fft.fftshift(np.fft.fftfreq(n_bins))[peak_row] * n_bins
    freq_x = np.fft.fftshift(np.fft.fftfreq(n_slices))[peak_col] * n_slices

    rng = np.random.default_rng(seed)
    perm_mask = is_tp.copy()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(perm_mask)
        perm_stats[i] = stat(perm_mask)[0]
    p_value = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)

    return {
        "fft_scale": "log" if log_fft else "linear",
        "n_tp": int(is_tp.sum()), "n_fp": int((~is_tp).sum()),
        "image_rmsd": obs_stat,
        "peak_freq_x_cycle": float(freq_x), "peak_freq_y_cycle": float(freq_y),
        "peak_deviation": float(obs_deviation[peak_row, peak_col]),
        "direction": f"TP {'>' if obs_deviation[peak_row, peak_col] > 0 else '<'} FP",
        "n_perm": n_perm, "p_value": float(p_value),
        "_mag_tp": obs_mag_tp, "_mag_fp": obs_mag_fp, "_img_tp": obs_img_tp, "_img_fp": obs_img_fp,
    }


def plot_image_and_spectrum(cohort: str, variant: str, r: dict, bin_edges: np.ndarray, title_extra: str = "") -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    extent_img = [1, N_SLICES, bin_edges[0], bin_edges[-1]]
    freq_x_edges = np.fft.fftshift(np.fft.fftfreq(N_SLICES)) * N_SLICES
    freq_y_edges = np.fft.fftshift(np.fft.fftfreq(N_Y_BINS)) * N_Y_BINS
    extent_spec = [freq_x_edges[0], freq_x_edges[-1], freq_y_edges[0], freq_y_edges[-1]]

    for ax, img, label in [(axes[0, 0], r["_img_tp"], "TP"), (axes[0, 1], r["_img_fp"], "FP")]:
        ax.imshow(img, origin="lower", extent=extent_img, aspect="auto", cmap="gray")
        ax.set_title(f"{label} density image (n={r['n_tp'] if label == 'TP' else r['n_fp']})",
                    color=INK_PRIMARY, fontsize=10, fontweight="bold")
        ax.set_xlabel("Slice index (1-128)", fontsize=9)
        ax.set_ylabel(NORM_LABELS[variant], fontsize=9)

    is_log_test = r["fft_scale"] == "log"
    for ax, mag, label in [(axes[1, 0], r["_mag_tp"], "TP"), (axes[1, 1], r["_mag_fp"], "FP")]:
        # r["_mag_*"] is already log1p'd when the test itself ran in log space; only apply the
        # display-only log1p here for the linear-space test (huge dynamic range, needs compression
        # to be visible at all).
        display_mag = mag if is_log_test else np.log1p(mag)
        ax.imshow(display_mag, origin="lower", extent=extent_spec, aspect="auto", cmap="gray")
        ax.set_title(f"{label} 2D DFT amplitude ({'log1p, tested in log space' if is_log_test else 'log scale display, tested linear'})",
                    color=INK_PRIMARY, fontsize=10, fontweight="bold")
        ax.set_xlabel("Freq: slice axis (cycles/128 slices)", fontsize=9)
        ax.set_ylabel("Freq: value axis (cycles/64 bins)", fontsize=9)

    fig.suptitle(f"{cohort}{title_extra}: {NORM_LABELS[variant]}-128, TP vs FP -- 2D image + 2D DFT ({FFT_SCALE_LABELS[r['fft_scale']]})",
                 fontsize=12, fontweight="bold")
    note = (f"image RMSD={r['image_rmsd']:.5f}, p={r['p_value']:.3g}, "
            f"peak Δ={r['peak_deviation']:.5f} @ (slice-freq={r['peak_freq_x_cycle']:.0f}, value-freq={r['peak_freq_y_cycle']:.0f})")
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def run_cohort(cohort: str, meta: pd.DataFrame, y: np.ndarray, score: np.ndarray, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = baseline.build_group_rows(meta, y, score, th)
    aec = load_raw_aec(xlsx_path)

    df = rows[["PatientID", "group"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(rows), f"{cohort}: TP/FP merge dropped rows"

    df = df[df["group"].isin(["TP", "FP"])].reset_index(drop=True)
    full_mat = df[AEC_COLS].to_numpy()
    is_tp = (df["group"].to_numpy() == "TP")

    results = []
    for variant in NORMALIZATIONS:
        norm_mat = normalize_curves(full_mat, variant)
        bin_edges = np.linspace(norm_mat.min(), norm_mat.max(), N_Y_BINS + 1)
        bin_idx = digitize_curves(norm_mat, bin_edges)

        out_dir = OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        for fft_scale in FFT_SCALES:
            r = image_2d_diff_test(bin_idx, is_tp, N_Y_BINS, N_SLICES, log_fft=(fft_scale == "log"))

            fig = plot_image_and_spectrum(cohort, variant, r, bin_edges)
            fig_path = out_dir / f"tp_vs_fp_aec_2dfft_{variant}_{fft_scale}fft.png"
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[{cohort}/{variant}/{fft_scale}fft] n_TP={r['n_tp']} n_FP={r['n_fp']} image_RMSD={r['image_rmsd']:.5f} "
                  f"perm_p={r['p_value']:.4g} peak=(x={r['peak_freq_x_cycle']:.0f},y={r['peak_freq_y_cycle']:.0f}) "
                  f"peak_delta={r['peak_deviation']:.5f}")
            print(f"Saved {fig_path}")

            summary_path = out_dir / f"tp_vs_fp_aec_2dfft_{variant}_{fft_scale}fft_summary.csv"
            row_out = {k: v for k, v in r.items() if not k.startswith("_")}
            pd.DataFrame([{"cohort": cohort, "normalization": variant, **row_out}]).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"Saved {summary_path}")

            results.append({"cohort": cohort, "normalization": variant, **row_out})
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
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_2dfft_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
