from __future__ import annotations
# single_disease 결과 xlsx(summary)를 질환 x (내부/외부) AUC 막대그래프 png로 저장
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = "outputs/clinic4/landmark/single_disease/"
s = pd.read_excel(D + "landmark_single_disease_aec_bodycomp.xlsx", sheet_name="summary")
models = list(dict.fromkeys(s.model))
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey=True)
for r, (col, ttl) in enumerate([("internal_auc", "internal (gangnam hold-out test)"), ("external_auc", "external (sinchon)")]):
    for c, d in enumerate(["HTN", "DM", "CKD"]):
        g = s[s.disease == d].set_index("model").loc[models, col]
        ax = axes[r, c]
        ax.bar(range(len(models)), g, color=["gray"] + ["C0"] * (len(models) - 2) + ["C3"])
        ax.axhline(g["baseline"], color="k", ls="--", lw=0.8)
        for i, v in enumerate(g):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0.4, 1.08)
        ax.set_xticks(range(len(models)), [m.replace(" (all landmarks)", "\n(all)") for m in models], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{d} - {ttl}", fontsize=10)
fig.suptitle("Single-disease (only-one) vs no-disease: baseline vs +AEC/body composition (12 landmarks)")
fig.savefig(D + "landmark_single_disease_aec_bodycomp_auc.png", dpi=200)
