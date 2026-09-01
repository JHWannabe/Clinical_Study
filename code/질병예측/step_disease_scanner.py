from __future__ import annotations

# step_disease_logistic.py가 저장한 predictions.csv(5개 모델: clinic4/clinic4_meanmAs/clinic4_meanmAs_aec/
# clinic4_vatsat/clinic4_vatsat_aec, 스캐너 제한 해제 코호트)를 재학습 없이 스캐너(Manufacturer)별로
# 재슬라이싱한다. threshold는 predictions.csv에 이미 고정값으로 들어있어 그대로 재사용(internal OOF에서만
# 탐색 후 동결한 원칙 유지). 2026-08-24 기준 code/질병예측와 code/질병예측_v2 두 폴더를 하나로 통합하며 이
# 파일은 v2(재설계본) 내용으로 교체됐다(사용자 확인: "통합시켜, 최대한 v2의 내용으로 진행").
# 스캐너별 AUC 서브그룹 결과를 표 이미지뿐 아니라 그래프(feature x family 조합별 internal/external 막대그래프,
# step_disease_logistic.py의 plot_auc_summary와 동일한 FAMILIES 분할·색상·y축 0.5-1.0 고정 스타일)로도 저장
# (사용자 요청 2026-08-24: "outputs\step_disease_scanner 그래프로도 저장하게 해").

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP_DIR = PROJECT_ROOT / "outputs" / "step_disease_logistic"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_disease_scanner"

# table1_baseline_characteristics*.py와 동일한 제조사->벤더 매핑. 개별 모델명(manufacturer) 기준으로
# 서브그룹을 나누면 n이 너무 작아 대부분 버려지므로(2026-08-26 사용자 확인: "vendor로 나누는게 낫지 않을까"),
# Vendor(Siemens/GE/Philips/Canon) 단위로 묶어 표본 크기를 확보한다. Canon처럼 벤더 단위로도 n이 매우 작은
# 경우(internal n=3/external n=12)는 처음엔 필터 없이 그대로 노출했으나(2026-08-26: "n<30이라 제외 조건을
# 없애줘"), 이후 소표본 벤더는 "Other"로 묶어서 표현하라는 요청(2026-08-26: "n<30인거는 제외하지말고
# other로 묶어서 표현해")에 따라 OTHER_VENDOR_MIN_N 미만인 벤더는 배제 대신 "Other" 카테고리로 합산한다.
OTHER_VENDOR_MIN_N = 30
VENDOR_PREFIXES = {
    "Siemens": ("SOMATOM", "Sensation", "Definition", "Emotion", "Scope", "Spirit"),
    "GE": ("Revolution", "LightSpeed", "Discovery", "Optima", "Brivo"),
    "Philips": ("Ingenuity", "iCT", "Brilliance", "MX", "IQon"),
    "Canon": ("Aquilion", "Activion", "ECLOS", "Supria"),
}


def classify_vendor(manufacturer: str) -> str:
    for vendor, prefixes in VENDOR_PREFIXES.items():
        if manufacturer.startswith(prefixes):
            return vendor
    raise ValueError(f"매핑되지 않은 CT 스캐너 모델명: {manufacturer!r}")


# outputs/figure/fig_scree_elbow.png(figure_manuscript.py plot_panel_c_scree_elbow, figsize=16x11,
# label/tick/legend fontsize=32.5/30/27.5)와 동일한 "폰트크기:이미지크기" 비율을 유지하는 스케일러
# (사용자 요청 2026-08-27: "이미지 크기와 폰트 크기를 fig_scree_elbow.png 비율로 해줘")
_REF_AVG_DIM = (16.0 + 11.0) / 2
_LABEL_FS_RATIO = 32.5 / _REF_AVG_DIM
_TICK_FS_RATIO = 30.0 / _REF_AVG_DIM
_LEGEND_FS_RATIO = 27.5 / _REF_AVG_DIM


def scaled_fontsizes(width: float, height: float) -> tuple[float, float, float]:
    avg_dim = (width + height) / 2
    return _LABEL_FS_RATIO * avg_dim, _TICK_FS_RATIO * avg_dim, _LEGEND_FS_RATIO * avg_dim


MODEL_ORDER = ["clinic4", "clinic4_meanmAs", "clinic4_meanmAs_aec", "clinic4_vatsat", "clinic4_vatsat_aec"]
MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_meanmAs": "+mean mAs",
    "clinic4_meanmAs_aec": "+mean mAs+AEC",
    "clinic4_vatsat": "+VAT+SAT",
    "clinic4_vatsat_aec": "+VAT+SAT+AEC",
}
MODEL_COLORS = {"clinic4": "#898781", "clinic4_meanmAs": "#2a78d6", "clinic4_meanmAs_aec": "#1baf7a",
                 "clinic4_vatsat": "#a35ad1", "clinic4_vatsat_aec": "#e2622e"}
# step_disease_logistic.py의 FAMILIES와 동일 분할(clinic4 baseline부터 3-way 비교)로 스캐너별 막대그래프도 통일
FAMILIES = {
    "meanmAs": ["clinic4", "clinic4_meanmAs", "clinic4_meanmAs_aec"],
    "vatsat": ["clinic4", "clinic4_vatsat", "clinic4_vatsat_aec"],
}


def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


def scanner_subgroup_stats(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feat, model_name, cohort), grp in predictions.groupby(["feature", "model", "cohort"]):
        threshold = float(grp["threshold"].iloc[0])
        y_all, score_all, vendors = grp["y"].to_numpy(), grp["score"].to_numpy(), grp["vendor"].to_numpy()
        vendor_counts = grp["vendor"].value_counts()
        small_vendors = set(vendor_counts[vendor_counts < OTHER_VENDOR_MIN_N].index)
        grouped_vendors = np.where(np.isin(vendors, list(small_vendors)), "Other", vendors)
        for vendor in pd.unique(grouped_vendors):
            sel = grouped_vendors == vendor
            n = int(sel.sum())
            y, score = y_all[sel], score_all[sel]
            if len(np.unique(y)) < 2:
                continue
            cls_stats = classification_stats(y, score, threshold)
            rows.append({"feature": feat, "model": model_name, "cohort": cohort, "scanner": vendor, "n": n,
                         "n_pos": int(y.sum()), "auc": float(roc_auc_score(y, score)), **cls_stats})
    return pd.DataFrame(rows)


def _sorted_scanner_table(scanner_df: pd.DataFrame) -> pd.DataFrame:
    feature_order = ["HTN", "DM", "CKD"]
    table = scanner_df.copy()
    table["feature"] = pd.Categorical(table["feature"], categories=feature_order, ordered=True)
    table["cohort"] = pd.Categorical(table["cohort"], categories=["internal", "external"], ordered=True)
    table["model"] = pd.Categorical(table["model"], categories=MODEL_ORDER, ordered=True)
    return table.sort_values(["scanner", "cohort", "feature", "model"]).reset_index(drop=True)


def _render_scanner_table_image(table: pd.DataFrame, title: str, out_path: Path) -> None:
    rows = [[r["scanner"], r["cohort"], r["feature"], MODEL_LABELS.get(r["model"], r["model"]), int(r["n"]),
             int(r["n_pos"]), f"{r['auc']:.3f}", f"{r['sensitivity']:.3f}", f"{r['specificity']:.3f}",
             f"{r['accuracy']:.3f}"] for _, r in table.iterrows()]
    col_labels = ["Vendor", "Cohort", "Feature", "Model", "n", "n_pos", "AUC", "Se", "Sp", "Acc"]
    col_widths = [0.19, 0.09, 0.08, 0.15, 0.06, 0.08, 0.09, 0.08, 0.08, 0.08]

    fig_w, fig_h = 28, 1.2 + 0.55 * len(rows)
    title_fs, header_fs, body_fs = scaled_fontsizes(fig_w, fig_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(body_fs)
    tbl.scale(1, 2.4)

    block_start = table["scanner"].ne(table["scanner"].shift()).to_numpy()
    divider_row_idxs = []
    for (row_i, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcdc7")
        if row_i == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=header_fs)
            cell.set_facecolor("#161616")
        else:
            cell.set_facecolor("white")
            if block_start[row_i - 1] and row_i > 1:
                divider_row_idxs.append(row_i)

    ax.set_title(title, fontsize=title_fs, fontweight="bold", color="#161616", pad=10)
    fig.tight_layout()

    if divider_row_idxs:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for divider_row_i in divider_row_idxs:
            left_cell, right_cell = tbl[(divider_row_i, 0)], tbl[(divider_row_i, len(col_labels) - 1)]
            bbox_left, bbox_right = left_cell.get_window_extent(renderer), right_cell.get_window_extent(renderer)
            x0, y_top = inv.transform((bbox_left.x0, bbox_left.y1))
            x1, _ = inv.transform((bbox_right.x1, bbox_right.y1))
            line = plt.Line2D([x0, x1], [y_top, y_top], transform=fig.transFigure, color="#161616", linewidth=3.0,
                               solid_capstyle="butt", zorder=10)
            fig.add_artist(line)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup Se/Sp/Acc table to {out_path}")


def plot_scanner_subgroup_table_paginated(scanner_df: pd.DataFrame, out_dir: Path,
                                           scanners_per_page: int = 2) -> list[Path]:
    if scanner_df.empty:
        return []
    table = _sorted_scanner_table(scanner_df)
    scanners = list(pd.unique(table["scanner"]))
    chunks = [scanners[i:i + scanners_per_page] for i in range(0, len(scanners), scanners_per_page)]

    out_paths = []
    for page_i, chunk in enumerate(chunks, start=1):
        sub = table[table["scanner"].isin(chunk)].reset_index(drop=True)
        out_path = out_dir / f"scanner_subgroup_se_sp_acc_table_{page_i}.png"
        _render_scanner_table_image(sub, f"벤더별 서브그룹 Se/Sp/Acc ({page_i}/{len(chunks)}) — {' · '.join(chunk)}",
                                     out_path)
        out_paths.append(out_path)
    return out_paths


def plot_scanner_auc_bar(scanner_df: pd.DataFrame, feat: str, family_name: str, model_list: list[str],
                          out_path: Path) -> None:
    sub = scanner_df[(scanner_df["feature"] == feat) & (scanner_df["model"].isin(model_list))]
    if sub.empty:
        return
    # "Other"(소표본 합산 카테고리)는 항상 맨 오른쪽에 오도록 정렬(사용자 요청 2026-08-27)
    scanners = sorted(pd.unique(sub["scanner"]), key=lambda v: (v == "Other", v))
    x = np.arange(len(scanners))
    width = 0.8 / len(model_list)

    panel_w, panel_h = max(9, 1.6 * len(scanners) + 3), 6.5
    label_fs, tick_fs, legend_fs = scaled_fontsizes(panel_w, panel_h)

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h))
    for ax, cohort in zip(axes, ["internal", "external"]):
        csub = sub[sub["cohort"] == cohort]
        for i, model_name in enumerate(model_list):
            rows = csub[csub["model"] == model_name].set_index("scanner").reindex(scanners)
            offset = (i - (len(model_list) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[model_name], color=MODEL_COLORS[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(scanners, fontsize=tick_fs, rotation=30, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{feat} ({cohort})", fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.grid(alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(model_list), bbox_to_anchor=(0.5, -0.12),
               fontsize=legend_fs, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup AUC bar plot to {out_path}")


def plot_scanner_auc_bar_all(scanner_df: pd.DataFrame, out_dir: Path) -> None:
    for feat in [f for f in ["HTN", "DM", "CKD"] if f in scanner_df["feature"].unique()]:
        for family_name, model_list in FAMILIES.items():
            out_path = out_dir / f"scanner_subgroup_auc_bar_{feat.lower()}_{family_name}.png"
            plot_scanner_auc_bar(scanner_df, feat, family_name, model_list, out_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(STEP_DIR / "predictions.csv")
    predictions["vendor"] = predictions["manufacturer"].map(classify_vendor)
    scanner_df = scanner_subgroup_stats(predictions)
    for _, r in scanner_df.sort_values(["feature", "cohort", "model", "scanner"]).iterrows():
        print(f"[{r['feature']} / {r['model']} / {r['cohort']} / {r['scanner']}] n={r['n']} n_pos={r['n_pos']} "
              f"AUC={r['auc']:.3f} Se={r['sensitivity']:.3f} Sp={r['specificity']:.3f} Acc={r['accuracy']:.3f}")

    scanner_df.to_csv(OUTPUT_DIR / "scanner_subgroup_auc.csv", index=False)
    print(f"Saved scanner subgroup AUC/Se/Sp/Acc to {OUTPUT_DIR / 'scanner_subgroup_auc.csv'}")
    plot_scanner_subgroup_table_paginated(scanner_df, OUTPUT_DIR, scanners_per_page=2)
    plot_scanner_auc_bar_all(scanner_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
