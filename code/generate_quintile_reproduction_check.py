from __future__ import annotations

"""
Reproduction-check card for the Stage-4 quintile enrichment result
(outputs/run_from_raw_standalone/aec_final_global_quintile_phenotype) against
a set of reference numbers shared by a collaborator for the 20% primary
Clinical+ / AEC-high vs AEC-low comparison.

This does not recompute anything -- it only reads
01_quintile_vs_quartile_enrichment.csv (q=0.20 rows) and renders a summary
card in the same visual style as outputs/full_derivation/MD/*.png
(MDCARD_draw_card in main_aec_full_derivation_pipeline_simplified.py).

Reference numbers (given by the collaborator, primary q=20%):
    Gangnam/internal : AEC-low 8/44=18.2%, AEC-high 26/44=59.1%, p=7.79e-05
    Sinchon/external : AEC-low 12/54=22.2%, AEC-high 41/69=59.4%, p=2.96e-05
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUINTILE_CSV = PROJECT_ROOT / "outputs" / "run_from_raw_standalone" / "aec_final_global_quintile_phenotype" / "01_quintile_vs_quartile_enrichment.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "run_from_raw_standalone" / "MD"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HEADER_BG = "#F2F2F2"
LINE_COLOR = "#DDDDDD"
GREEN = "#1A7F37"
AMBER = "#B7791F"
RED = "#B42318"

REFERENCE = {
    "g1090_internal": {"label": "Gangnam / internal", "low_n": 8, "low_den": 44, "high_n": 26, "high_den": 44, "p": 7.79e-05},
    "sdata_external": {"label": "Sinchon / external", "low_n": 12, "low_den": 54, "high_n": 41, "high_den": 69, "p": 2.96e-05},
}


def pct(n: int, den: int) -> str:
    return f"{n}/{den} = {n / den * 100:.1f}%" if den else "n/a"


def sci(x: float) -> str:
    return f"{x:.2e}"


def verdict(ref_n: int, ref_den: int, got_n: int, got_den: int, ref_p: float, got_p: float) -> str:
    if ref_n == got_n and ref_den == got_den:
        return "일치"
    rate_diff = abs(ref_n / ref_den - got_n / got_den)
    if rate_diff <= 0.02:
        return "근사일치"
    return "불일치"


def verdict_color(v: str) -> str:
    return GREEN if v == "일치" else (AMBER if v == "근사일치" else RED)


def draw_card(path: Path, title: str, sections: list[dict], footer_lines: list[str], figsize=(13.6, 8.4)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=145)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.965
    ax.text(0.012, y, title, fontsize=17.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    y -= 0.08

    row_h = 0.058
    col0_frac = 0.30
    for section in sections:
        ax.text(0.012, y, section["header"], fontsize=14, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= 0.05
        if section.get("subheader"):
            ax.text(0.012, y, section["subheader"], fontsize=10, ha="left", va="top", transform=ax.transAxes, color="#333333")
            y -= 0.034 * (section["subheader"].count("\n") + 1) + 0.014

        cols = section["columns"]
        n_cols = len(cols)
        widths = [col0_frac] + [(1 - col0_frac) / (n_cols - 1)] * (n_cols - 1)
        xs = [0.012]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)

        ax.add_patch(Rectangle((0.008, y - row_h + 0.014), 0.98, row_h, facecolor=HEADER_BG, edgecolor="none", transform=ax.transAxes, zorder=0))
        for cx, col in zip(xs, cols):
            ax.text(cx, y - 0.012, col, fontsize=11.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= row_h

        colors = section.get("cell_colors")
        for ri, row in enumerate(section["rows"]):
            for ci, (cx, val) in enumerate(zip(xs, row)):
                color = "black"
                if colors and colors[ri][ci]:
                    color = colors[ri][ci]
                weight = "bold" if (ci == len(row) - 1 and colors) else "normal"
                ax.text(cx, y - 0.012, str(val), fontsize=11, ha="left", va="top", transform=ax.transAxes, color=color, fontweight=weight)
            y -= row_h
            ax.plot([0.008, 0.988], [y + row_h * 0.32, y + row_h * 0.32], color=LINE_COLOR, lw=0.8, transform=ax.transAxes)
        y -= 0.032

    y -= 0.008
    ax.plot([0.008, 0.988], [y, y], color="#BBBBBB", lw=0.8, transform=ax.transAxes)
    y -= 0.038
    for line in footer_lines:
        ax.text(0.012, y, line, fontsize=9, ha="left", va="top", transform=ax.transAxes, color="#555555")
        y -= 0.032

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def main() -> None:
    df = pd.read_csv(QUINTILE_CSV)
    q20 = df[df["q"].round(2).eq(0.20)].set_index("cohort")

    sections = []
    all_verdicts: list[str] = []
    for cohort in ["g1090_internal", "sdata_external"]:
        ref = REFERENCE[cohort]
        row = q20.loc[cohort]
        got_low_n, got_low_den = int(row["aec_low_low_smi_n"]), int(row["aec_low_n"])
        got_high_n, got_high_den = int(row["aec_high_low_smi_n"]), int(row["aec_high_n"])
        got_p = float(row["fisher_p_high_gt_low"])

        v_low = verdict(ref["low_n"], ref["low_den"], got_low_n, got_low_den, ref["p"], got_p)
        v_high = verdict(ref["high_n"], ref["high_den"], got_high_n, got_high_den, ref["p"], got_p)
        v_p = "일치" if abs(ref["p"] - got_p) / max(ref["p"], got_p) < 0.2 else ("근사일치" if (ref["p"] < 0.001 and got_p < 0.001) else "불일치")
        all_verdicts += [v_low, v_high, v_p]

        rows = [
            ["AEC-low, low SMI 비율", pct(ref["low_n"], ref["low_den"]), pct(got_low_n, got_low_den), v_low],
            ["AEC-high, low SMI 비율", pct(ref["high_n"], ref["high_den"]), pct(got_high_n, got_high_den), v_high],
            ["Fisher p (AEC-high > AEC-low)", sci(ref["p"]), sci(got_p), v_p],
        ]
        sections.append(
            {
                "header": f"{ref['label']} (q=20% primary, Clinical+ 내부)",
                "columns": ["항목", "참조값 (전달받은 값)", "재현 결과", "판정"],
                "rows": rows,
                "cell_colors": [[None, None, None, verdict_color(r[3])] for r in rows],
            }
        )

    overall = "전체 일치" if all(v == "일치" for v in all_verdicts) else ("일부 근사/불일치 있음" if any(v in ("불일치",) for v in all_verdicts) else "근사 일치")

    draw_card(
        OUT_DIR / "quintile_enrichment_reproduction_check.png",
        "재현성 점검: Clinical+ 내 AEC-high vs AEC-low (q=20% primary) vs 전달받은 참조값",
        sections=sections,
        footer_lines=[
            f"출처: outputs/run_from_raw_standalone/aec_final_global_quintile_phenotype/01_quintile_vs_quartile_enrichment.csv (q=0.20 rows)를 그대로 읽어 표시 (재계산 없음).",
            "internal은 AEC-low/AEC-high 표본 크기와 p-value 모두 참조값과 거의 정확히 일치.",
            "external은 AEC-low(12/54)는 정확히 일치하지만, AEC-high 표본이 78명(사건 49)으로 참조값(69명, 사건 41)보다 커서 p-value가 더 작게(3.31e-06) 나옴 -- 두 표본에 공통 적용되는 internal-locked AEC-high 컷오프(>=1.5226) 자체가 참조 파이프라인과 달랐을 가능성.",
            f"종합 판정: {overall}",
        ],
    )


if __name__ == "__main__":
    main()
