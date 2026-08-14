from __future__ import annotations

# step3_clinic_aec_disease_logistic.py에서 스캐너(Manufacturer)별 서브그룹 AUC/Se/Sp/Acc 분석을 분리한
# 스크립트(사용자 확인: "step4 py파일을 생성해서 scanner별 auc, se/sp/acc를 확인해"). [[feedback_output_dir_single_producer]]
# 원칙에 따라 outputs/step3_disease_logistic은 모델 학습·전체 코호트 성능만 담당하고, 이 스크립트가 스캐너별
# 서브그룹 분석을 전담한다. step3가 저장한 outputs/step3_disease_logistic/predictions.csv(환자별 예측확률 +
# 고정 threshold + Manufacturer)만 읽어서 재학습 없이 재슬라이싱한다 - Se/Sp/Acc는 전체 코호트에서 구한 고정
# Youden threshold(step3가 internal OOF에서만 탐색 후 동결)를 스캐너별로도 그대로 적용한다. 스캐너별로
# threshold를 다시 구하면 "threshold는 internal에서 한 번만 정하고 고정 적용한다"는 원칙(slide6 방법론,
# [[project_disease_association_pptx_edits]])과 어긋나기 때문이다.

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP3_DIR = PROJECT_ROOT / "outputs" / "step3_disease_logistic"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4_disease_scanner_subgroup"
SCANNER_MIN_N = 30  # 스캐너별 서브그룹 분석에서 이 인원 미만인 스캐너는 제외
MODEL_LABELS = {"clinic4": "clinic4", "clinic4_aec_best": "+AEC(질환별 최적조합)"}


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출(step3의 classification_stats와 동일 로직)
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# step3가 저장한 환자별 예측확률(predictions.csv)을 feature/model/cohort 단위로 스캐너(Manufacturer)별로 나눠
# AUC/Se/Sp/Acc를 재계산. threshold는 predictions.csv에 이미 고정값으로 들어있어 그대로 재사용한다.
# SCANNER_MIN_N 미만이거나 한쪽 클래스만 있는 스캐너는 AUC/Se/Sp가 정의되지 않아 제외한다
def scanner_subgroup_stats(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["feature", "model", "cohort"]
    for (feat, model_name, cohort), grp in predictions.groupby(group_cols):
        threshold = float(grp["threshold"].iloc[0])
        y_all = grp["y"].to_numpy()
        score_all = grp["score"].to_numpy()
        scanners = grp["manufacturer"].to_numpy()
        for scanner in pd.unique(scanners):
            sel = scanners == scanner
            n = int(sel.sum())
            y, score = y_all[sel], score_all[sel]
            if n < SCANNER_MIN_N or len(np.unique(y)) < 2:
                continue
            cls_stats = classification_stats(y, score, threshold)
            rows.append({"feature": feat, "model": model_name, "cohort": cohort, "scanner": scanner, "n": n,
                         "n_pos": int(y.sum()), "auc": float(roc_auc_score(y, score)), **cls_stats})
    return pd.DataFrame(rows)


# feature(질환)별로 clinic4 vs (internal 기준) 최고 AEC 모델의 스캐너별 AUC를 나란히 막대그래프로 비교.
# overall_summary(스캐너로 나누지 않은 전체 표본 기준 feature/model/cohort/auc 요약, step3의
# logistic_regression_summary.csv)를 주면 모델별 색상에 맞춘 점선으로 "스캐너 통합 시 AUC" 기준선을 함께 그린다
def plot_scanner_subgroup_auc(scanner_df: pd.DataFrame, out_path: Path,
                               overall_summary: pd.DataFrame | None = None) -> None:
    if scanner_df.empty:
        print(f"[스캐너 서브그룹] 표본이 {SCANNER_MIN_N}명 이상이고 두 클래스가 모두 있는 스캐너가 없어 그래프를 생략합니다.")
        return

    features = list(scanner_df["feature"].unique())
    cohorts = ["internal", "external"]
    per_category_width = 3.2
    label_wrap_width = 11
    n_scanners_by_col = [
        max((scanner_df[(scanner_df["feature"] == f) & (scanner_df["cohort"] == c)]["scanner"].nunique()
             for f in features), default=1)
        for c in cohorts
    ]
    col_widths = [max(n, 1) * per_category_width for n in n_scanners_by_col]
    fig, axes = plt.subplots(len(features), 2, figsize=(sum(col_widths), 9 * len(features)), squeeze=False,
                              gridspec_kw={"width_ratios": col_widths})
    for row, feat in enumerate(features):
        for col, cohort in enumerate(cohorts):
            ax = axes[row][col]
            sub = scanner_df[(scanner_df["feature"] == feat) & (scanner_df["cohort"] == cohort)]
            if sub.empty:
                ax.set_visible(False)
                continue
            scanners = sorted(sub["scanner"].unique())
            models_here = list(sub["model"].unique())
            x = np.arange(len(scanners)) * 2.0
            width = 0.8 / max(len(models_here), 1)
            colors = ["#6b6a66", "#2a78d6", "#e2622e"]

            scanner_arr = sub["scanner"].to_numpy()
            model_arr = sub["model"].to_numpy()
            auc_arr = sub["auc"].to_numpy(dtype=float)
            for i, model_name in enumerate(models_here):
                display_name = MODEL_LABELS.get(model_name, model_name)
                vals = [float(np.mean(auc_arr[(scanner_arr == s) & (model_arr == model_name)])) for s in scanners]
                offset = (i - (len(models_here) - 1) / 2) * width
                ax.bar(x + offset, vals, width, label=display_name, color=colors[i % len(colors)])
                if overall_summary is not None:
                    orow = overall_summary[(overall_summary["feature"] == feat)
                                            & (overall_summary["model"] == model_name)
                                            & (overall_summary["cohort"] == cohort)]
                    if not orow.empty:
                        ax.axhline(float(orow["auc"].iloc[0]), color=colors[i % len(colors)], linestyle="--",
                                   linewidth=2.5, alpha=0.9, zorder=3, label=f"{display_name} (스캐너 통합)")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x)
            wrapped = ["\n".join(textwrap.wrap(s, label_wrap_width)) for s in scanners]
            ax.set_xticklabels(wrapped, fontsize=30, rotation=0)
            ax.set_ylim(0.5, 1.0)
            ax.set_title(f"{feat} — {cohort} 스캐너별 AUC", fontsize=30, fontweight="bold", color="#161616",
                         pad=24)
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="y", labelsize=30)

    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            if ax.get_visible():
                handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        if handles:
            break
    fig_height_in = fig.get_size_inches()[1]
    bottom_margin_in = 2.2
    bottom_frac = bottom_margin_in / fig_height_in
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=30,
               bbox_to_anchor=(0.5, bottom_frac * 0.15), frameon=True)

    fig.tight_layout(rect=(0, bottom_frac, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup AUC plot to {out_path}")


# scanner_subgroup_stats()의 결과를 Scanner(최상위)→Cohort→Feature→Model 순으로 정렬. 같은 스캐너에서
# 질환(HTN/DM/CKD)별 성능이 인접하게 배치되어야 비교가 쉽다(사용자 확인: "스캐너별로 feature를 다르게 했을 때
# 성능 비교가 편하도록 배치해" — 기존엔 feature가 최상위라 같은 스캐너 행이 표 전체에 흩어져 있었음)
def _sorted_scanner_table(scanner_df: pd.DataFrame) -> pd.DataFrame:
    feature_order = ["HTN", "DM", "CKD"]
    table = scanner_df.copy()
    table["feature"] = pd.Categorical(table["feature"], categories=feature_order, ordered=True)
    table["cohort"] = pd.Categorical(table["cohort"], categories=["internal", "external"], ordered=True)
    return table.sort_values(["scanner", "cohort", "feature", "model"]).reset_index(drop=True)


# 정렬된 표(부분집합이어도 됨)를 표 이미지로 렌더링. 스캐너가 바뀌는 경계마다 굵은 구분선을 그려 블록을
# 한눈에 구분되게 한다(step3의 plot_auc_delta_combined_table과 동일 기법)
def _render_scanner_table_image(table: pd.DataFrame, title: str, out_path: Path) -> None:
    rows = [[r["scanner"], r["cohort"], r["feature"], MODEL_LABELS.get(r["model"], r["model"]), int(r["n"]),
             int(r["n_pos"]), f"{r['auc']:.3f}", f"{r['sensitivity']:.3f}", f"{r['specificity']:.3f}",
             f"{r['accuracy']:.3f}"] for _, r in table.iterrows()]
    col_labels = ["Scanner", "Cohort", "Feature", "Model", "n", "n_pos", "AUC", "Se", "Sp", "Acc"]
    col_widths = [0.20, 0.09, 0.09, 0.16, 0.06, 0.08, 0.09, 0.08, 0.08, 0.08]

    fig, ax = plt.subplots(figsize=(26, 1.2 + 0.55 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(16)
    tbl.scale(1, 2.4)

    block_start = table["scanner"].ne(table["scanner"].shift()).to_numpy()
    divider_row_idxs = []
    for (row_i, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcdc7")
        if row_i == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=17)
            cell.set_facecolor("#161616")
        else:
            cell.set_facecolor("white")
            if block_start[row_i - 1] and row_i > 1:
                divider_row_idxs.append(row_i)

    ax.set_title(title, fontsize=22, fontweight="bold", color="#161616", pad=10)
    fig.tight_layout()

    # 셀 좌표는 tight_layout() 이후 확정되므로, 구분선은 그 뒤에 실제 렌더 window extent를 읽어 그린다
    if divider_row_idxs:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for divider_row_i in divider_row_idxs:
            left_cell = tbl[(divider_row_i, 0)]
            right_cell = tbl[(divider_row_i, len(col_labels) - 1)]
            bbox_left = left_cell.get_window_extent(renderer)
            bbox_right = right_cell.get_window_extent(renderer)
            x0, y_top = inv.transform((bbox_left.x0, bbox_left.y1))
            x1, _ = inv.transform((bbox_right.x1, bbox_right.y1))
            line = plt.Line2D([x0, x1], [y_top, y_top], transform=fig.transFigure,
                               color="#161616", linewidth=3.0, solid_capstyle="butt", zorder=10)
            fig.add_artist(line)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scanner subgroup Se/Sp/Acc table to {out_path}")


# 스캐너별 Se/Sp/Acc 전체를 한 이미지(46+행)로 저장 - 상세 확인·재검토용 원자료
def plot_scanner_subgroup_table(scanner_df: pd.DataFrame, out_path: Path) -> None:
    if scanner_df.empty:
        return
    table = _sorted_scanner_table(scanner_df)
    _render_scanner_table_image(
        table, "스캐너별 서브그룹 Se/Sp/Acc (n≥30, 고정 Youden threshold) — 스캐너 단위로 묶어 질환별 성능 비교",
        out_path)


# pptx 슬라이드 한 장에 넣기엔 전체 표(9개 스캐너, 최대 66행)가 너무 길어 글씨가 읽기 어려워짐
# (사용자 확인: "ppt에 보기 편하도록 구성해줘" -> "스캐너 3~4개씩 묶어 여러 장으로 분할"). 스캐너를
# scanners_per_page개씩 묶어 페이지별 이미지를 따로 저장하고, 저장된 경로 목록을 반환한다
def plot_scanner_subgroup_table_paginated(scanner_df: pd.DataFrame, out_dir: Path,
                                           scanners_per_page: int = 3) -> list[Path]:
    if scanner_df.empty:
        return []
    table = _sorted_scanner_table(scanner_df)
    scanners = list(pd.unique(table["scanner"]))
    chunks = [scanners[i:i + scanners_per_page] for i in range(0, len(scanners), scanners_per_page)]

    out_paths = []
    n_pages = len(chunks)
    for page_i, chunk in enumerate(chunks, start=1):
        sub = table[table["scanner"].isin(chunk)].reset_index(drop=True)
        out_path = out_dir / f"scanner_subgroup_se_sp_acc_table_{page_i}.png"
        title = (f"스캐너별 서브그룹 Se/Sp/Acc ({page_i}/{n_pages}) — {' · '.join(chunk)}")
        _render_scanner_table_image(sub, title, out_path)
        out_paths.append(out_path)
    return out_paths


# predictions.csv를 읽어 스캐너별 AUC/Se/Sp/Acc를 산출하고 csv/그래프/표 이미지로 저장
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(STEP3_DIR / "predictions.csv")
    overall_summary = pd.read_csv(STEP3_DIR / "logistic_regression_summary.csv")

    scanner_df = scanner_subgroup_stats(predictions)
    for _, r in scanner_df.sort_values(["feature", "cohort", "model", "scanner"]).iterrows():
        print(f"[{r['feature']} / {r['model']} / {r['cohort']} / {r['scanner']}] n={r['n']} n_pos={r['n_pos']} "
              f"AUC={r['auc']:.3f} Se={r['sensitivity']:.3f} Sp={r['specificity']:.3f} Acc={r['accuracy']:.3f}")

    scanner_df.to_csv(OUTPUT_DIR / "scanner_subgroup_auc.csv", index=False)
    print(f"Saved scanner subgroup AUC/Se/Sp/Acc to {OUTPUT_DIR / 'scanner_subgroup_auc.csv'}")

    plot_scanner_subgroup_auc(scanner_df, OUTPUT_DIR / "scanner_subgroup_auc.png", overall_summary=overall_summary)
    plot_scanner_subgroup_table(scanner_df, OUTPUT_DIR / "scanner_subgroup_se_sp_acc_table.png")
    plot_scanner_subgroup_table_paginated(scanner_df, OUTPUT_DIR, scanners_per_page=3)


if __name__ == "__main__":
    main()
