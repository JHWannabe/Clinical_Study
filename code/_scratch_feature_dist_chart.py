from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "outputs" / "step4" / "feature_distributions_by_sex.png"

INK_PRIMARY = "#161616"
INK_SECONDARY = "#525252"
RED_LIGHT = "#f4b6c2"
BLUE_LIGHT = "#8fb8e8"
CUTOFF_M = "#2a78d6"   # blue -- Male cutoff line
CUTOFF_F = "#dc267f"   # magenta -- Female cutoff line

meta = pd.read_excel(DATA_DIR / "gangnam.xlsx", sheet_name="metadata", engine="openpyxl")
sex = meta["PatientSex"].astype(str).str.upper()

# feature -> (column, direction, cutoff_M, cutoff_F) -- cutoffs are the current mean(+-)1SD values
# fixed in outputs/step4/label_cutoffs.csv (internal/Gangnam, frozen)
FEATURES = [
    ("TAMA_SUM",  "TAMA",  "low",  19026.98, 14113.98),
    ("NAMA_SUM",  "NAMA",  "low",  13837.32, 9607.86),
    ("LAMA_SUM",  "LAMA",  "high", 7639.02,  5168.44),
    ("IMATA_SUM", "IMATA", "high", 1902.54,  1683.31),
    ("SAT(피하지방)_SUM", "SAT", "high", 23421.47, 29798.28),
    ("VAT(내장지방)_SUM", "VAT", "high", 17286.71, 10924.34),
    ("Total Fat_SUM", "Total Fat", "high", 39696.86, 39812.01),
]

fig, axes = plt.subplots(2, 4, figsize=(19, 8))
axes = axes.ravel()

for i, (col, label, direction, cutoff_m, cutoff_f) in enumerate(FEATURES):
    ax = axes[i]
    v_m = pd.to_numeric(meta.loc[sex == "M", col], errors="coerce").dropna().to_numpy()
    v_f = pd.to_numeric(meta.loc[sex == "F", col], errors="coerce").dropna().to_numpy()
    vmax = max(v_m.max(), v_f.max())
    bins = np.linspace(0, vmax * 1.02, 36)

    ax.hist(v_m, bins=bins, color=BLUE_LIGHT, alpha=0.65, label=f"Male (n={len(v_m)})")
    ax.hist(v_f, bins=bins, color=RED_LIGHT, alpha=0.75, label=f"Female (n={len(v_f)})")
    ax.axvline(cutoff_m, color=CUTOFF_M, linestyle="--", linewidth=1.8)
    ax.axvline(cutoff_f, color=CUTOFF_F, linestyle="--", linewidth=1.8)

    dir_label = "mean-1SD" if direction == "low" else "mean+1SD"
    ax.set_title(f"{label} ({dir_label})", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    ax.set_xlabel(col, fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("환자 수", fontsize=8.5, color=INK_PRIMARY)
    ax.tick_params(axis="both", labelsize=7.5, colors=INK_SECONDARY)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8, loc="upper right", frameon=False)

# 8th slot: shared legend / cutoff explainer instead of an empty axes
ax_legend = axes[7]
ax_legend.axis("off")
ax_legend.text(0.02, 0.88, "범례", fontsize=13, fontweight="bold", color=INK_PRIMARY,
                transform=ax_legend.transAxes)
ax_legend.add_patch(Rectangle((0.02, 0.74), 0.08, 0.08, color=BLUE_LIGHT, alpha=0.65,
                               transform=ax_legend.transAxes))
ax_legend.text(0.14, 0.76, "Male (분포)", fontsize=10.5, color=INK_PRIMARY, transform=ax_legend.transAxes)
ax_legend.add_patch(Rectangle((0.02, 0.62), 0.08, 0.08, color=RED_LIGHT, alpha=0.75,
                               transform=ax_legend.transAxes))
ax_legend.text(0.14, 0.64, "Female (분포)", fontsize=10.5, color=INK_PRIMARY, transform=ax_legend.transAxes)
ax_legend.plot([0.02, 0.10], [0.51, 0.51], color=CUTOFF_M, linestyle="--", linewidth=2.0,
               transform=ax_legend.transAxes)
ax_legend.text(0.14, 0.49, "Male cutoff", fontsize=10.5, color=INK_PRIMARY, transform=ax_legend.transAxes)
ax_legend.plot([0.02, 0.10], [0.39, 0.39], color=CUTOFF_F, linestyle="--", linewidth=2.0,
               transform=ax_legend.transAxes)
ax_legend.text(0.14, 0.37, "Female cutoff", fontsize=10.5, color=INK_PRIMARY, transform=ax_legend.transAxes)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_PATH}")
