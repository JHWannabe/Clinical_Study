from __future__ import annotations

# code/질병예측/figure_patient_selection_flow.py를 베이스로, 사용자 확인(2026-08-21: "kVp와 scanner 제한 모두
# 풀어 주세요" -> kVp는 데이터상 전량 100kVp로만 추출되어 있어 해제 불가/scanner 제한만 해제로 확인)에 따라
# CT 스캐너 모델/벤더 제한 단계를 완전히 제거한 버전. data/{gangnam,sinchon}_원본.xlsx에서 연령<20 제외만
# 적용하며, 결과 인원(internal 1,088명 / external 925명)이 원본에서 매 실행마다 다시 계산된다(하드코딩 아님).

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure_v2"

INTERNAL_SITE = "Gangnam Severance Hospital"
EXTERNAL_SITE = "Sinchon Severance Hospital"
AGE_CUTOFF = 20

STATE_WHITE = "white"
FINAL_GREEN = "#d9ead3"
BORDER = "#161616"


def load_raw(site_key: str) -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / f"{site_key}_원본.xlsx", sheet_name="metadata", engine="openpyxl")


# 연령<20 제외만 적용한 단일 단계 필터링(스캐너/벤더 제한 없음)
def compute_flow(raw: pd.DataFrame) -> dict:
    n_enroll = len(raw)
    after_age = raw[raw["PatientAge"] >= AGE_CUTOFF]
    return {"n_enroll": n_enroll, "n_age_excluded": n_enroll - len(after_age), "n_final": len(after_age)}


def box(ax, renderer, cx, cy, w, h, text, facecolor, fontsize=13.5, fontweight="normal", min_fontsize=8.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=facecolor, edgecolor=BORDER, linewidth=1.6, zorder=2))
    txt = ax.text(cx, cy, text, ha="center", va="center", fontweight=fontweight, zorder=3, linespacing=1.4)

    (x0, y0) = ax.transData.transform((cx - w / 2, cy - h / 2))
    (x1, y1) = ax.transData.transform((cx + w / 2, cy + h / 2))
    box_w_px, box_h_px = abs(x1 - x0) * 0.92, abs(y1 - y0) * 0.88

    fs = fontsize
    while fs > min_fontsize:
        txt.set_fontsize(fs)
        bbox = txt.get_window_extent(renderer=renderer)
        if bbox.width <= box_w_px and bbox.height <= box_h_px:
            break
        fs -= 0.5
    txt.set_fontsize(fs)


def arrow(ax, xy_from, xy_to, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
                                  color=BORDER, zorder=1))


# 한 코호트 열(Enrollment -> 배제박스(연령<20) -> 최종 Inclusion box) — 스캐너 배제 단계가 없어 원본보다 짧다
def draw_column(ax, renderer, cx_main, cx_excl, flow: dict, cohort_label: str, site: str, main_w=4.4, excl_w=3.6):
    y_enroll, h_enroll = 8.6, 1.7
    y_excl, h_excl = 6.2, 1.15
    y_final, h_final = 3.6, 1.4

    box(ax, renderer, cx_main, y_enroll, main_w, h_enroll,
        f"{flow['n_enroll']:,} patients\nfrom {site}\n({cohort_label})", STATE_WHITE, fontweight="bold")

    arrow(ax, (cx_main, y_enroll - h_enroll / 2), (cx_main, y_final + h_final / 2 + 0.05))
    box(ax, renderer, cx_excl, y_excl, excl_w, h_excl,
        f"{flow['n_age_excluded']} excluded\nAge <{AGE_CUTOFF} years", STATE_WHITE, fontsize=11)
    arrow(ax, (cx_main + 0.08, y_excl), (cx_excl - excl_w / 2 - 0.08, y_excl), lw=1.2)

    box(ax, renderer, cx_main, y_final, main_w, h_final,
        f"{flow['n_final']:,} patients\n({cohort_label})", FINAL_GREEN, fontweight="bold")


def plot_diagram(flow_internal: dict, flow_external: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 8.5))
    xlim, ylim = (-2.7, 18.9), (2.2, 10.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cx_internal_main, cx_internal_excl = 3.0, 7.7
    cx_external_main, cx_external_excl = 12.3, 17.0

    draw_column(ax, renderer, cx_internal_main, cx_internal_excl, flow_internal, "internal cohort", INTERNAL_SITE)
    draw_column(ax, renderer, cx_external_main, cx_external_excl, flow_external, "external cohort", EXTERNAL_SITE)

    ax.text((cx_internal_main + cx_external_main) / 2, 9.85,
             "Both cohorts: CT examinations Jan 2018–Jun 2020 (internal) / 2019 (external); clinical data +\n"
             "abdominal CT at 100 kVp available for the same patient. No CT scanner-model restriction applied\n"
             "(all vendors/models retained); every CT examination in the source dataset was acquired at 100 kVp,\n"
             "so a tube-voltage restriction could not be relaxed within the available data.",
             ha="center", va="center", fontsize=10, style="italic", color="#3a3a3a")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved patient selection flow diagram to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_g, raw_s = load_raw("gangnam"), load_raw("sinchon")
    flow_g, flow_s = compute_flow(raw_g), compute_flow(raw_s)

    print("Internal (Gangnam):", flow_g)
    print("External (Sinchon):", flow_s)

    plot_diagram(flow_g, flow_s, OUTPUT_DIR / "fig1_patient_selection_flow_v2.png")


if __name__ == "__main__":
    main()
