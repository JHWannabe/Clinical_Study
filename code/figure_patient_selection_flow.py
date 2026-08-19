from __future__ import annotations

# Materials 섹션의 코호트 선정 절차(참고 예시 PNG의 Figure 1 스타일)를 실제 데이터로 재현하는 patient-selection
# flow diagram. data/{gangnam,sinchon}_원본.xlsx -> data/{gangnam,sinchon}.xlsx로 가는 두 단계 배제 사유(스캐너/
# 벤더 제한, 연령<20)와 그 인원수는 코드로 직접 재현해 검증했으며(원본에서 순차 필터링한 PatientID 집합이
# 필터링 파일과 완전히 일치), 하드코딩 숫자가 아니라 매 실행마다 원본 엑셀에서 다시 계산한다.
#
# 박스 겹침/텍스트 넘침 방지: 레이아웃은 모든 박스가 고정 xlim/ylim 안에 여유 있게 들어가도록 좌표를 직접
# 계산하고(내부/외부 열, 각 열의 배제 박스가 옆 열과 겹치지 않는 간격 확보), 텍스트는 렌더러로 실제 픽셀
# 크기를 측정해 박스 크기를 넘으면 폰트를 자동으로 줄이는 fit-to-box 로직을 적용한다.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figure"

INTERNAL_SITE = "Gangnam Severance Hospital"
EXTERNAL_SITE = "Sinchon Severance Hospital"
INTERNAL_SCANNERS = ["Sensation 64", "Revolution CT", "Ingenuity Core 128", "SOMATOM Definition AS+"]
EXTERNAL_EXCLUDED_VENDOR_MODELS = ["Aquilion ONE", "Aquilion"]  # Canon scanners
AGE_CUTOFF = 20

LABEL_BLUE = "#b8cfe5"
STATE_WHITE = "white"
FINAL_GREEN = "#d9ead3"
BORDER = "#161616"


# 원본 엑셀(metadata 시트, 이미 clinical+CT 매칭 및 kVp=100 필터 적용된 상태)을 불러온다
def load_raw(site_key: str) -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / f"{site_key}_원본.xlsx")


# 텍스트 서술 순서(스캐너/벤더 제한 -> 연령<20 제외)대로 순차 필터링해 각 단계 인원수를 계산
def compute_flow(raw: pd.DataFrame, keep_scanners: list[str] | None, drop_scanners: list[str] | None) -> dict:
    n_enroll = len(raw)
    if keep_scanners is not None:
        after_scanner = raw[raw["Manufacturer"].isin(keep_scanners)]
    else:
        after_scanner = raw[~raw["Manufacturer"].isin(drop_scanners)]
    n_scanner_excluded = n_enroll - len(after_scanner)
    after_age = after_scanner[after_scanner["PatientAge"] >= AGE_CUTOFF]
    n_age_excluded = len(after_scanner) - len(after_age)
    return {
        "n_enroll": n_enroll,
        "n_scanner_excluded": n_scanner_excluded,
        "n_after_scanner": len(after_scanner),
        "n_age_excluded": n_age_excluded,
        "n_final": len(after_age),
    }


def verify_against_filtered_file(site_key: str, expected_n: int) -> None:
    filtered = pd.read_excel(DATA_DIR / f"{site_key}.xlsx")
    assert len(filtered) == expected_n, f"{site_key}: recomputed n={expected_n} != filtered file n={len(filtered)}"


# 박스를 그리고, 렌더러로 실제 텍스트 픽셀 크기를 측정해 박스 폭/높이를 넘으면 폰트를 자동으로 줄인다
def box(ax, renderer, cx, cy, w, h, text, facecolor, fontsize=12.5, fontweight="normal", min_fontsize=8.0):
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


# 한 코호트 열(Enrollment box -> 배제박스1 -> 중간 box -> 배제박스2 -> 최종 Inclusion box)을 그린다
def draw_column(ax, renderer, cx_main, cx_excl, flow: dict, cohort_label: str, site: str, scanner_reason: str,
                 main_w=4.2, excl_w=3.4):
    y_enroll, h_enroll = 9.3, 1.7
    y_excl1, h_excl1 = 7.55, 1.15
    y_mid, h_mid = 6.15, 0.9
    y_excl2, h_excl2 = 4.55, 1.0
    y_final, h_final = 2.7, 1.25

    box(ax, renderer, cx_main, y_enroll, main_w, h_enroll,
        f"{flow['n_enroll']:,} patients\nfrom {site}\n({cohort_label})", STATE_WHITE, fontweight="bold")

    arrow(ax, (cx_main, y_enroll - h_enroll / 2), (cx_main, y_mid + h_mid / 2 + 0.05))
    box(ax, renderer, cx_excl, y_excl1, excl_w, h_excl1,
        f"{flow['n_scanner_excluded']} excluded\n{scanner_reason}", STATE_WHITE, fontsize=11)
    arrow(ax, (cx_main + 0.08, y_excl1), (cx_excl - excl_w / 2 - 0.08, y_excl1), lw=1.2)

    box(ax, renderer, cx_main, y_mid, main_w, h_mid, f"{flow['n_after_scanner']:,} patients", STATE_WHITE)

    arrow(ax, (cx_main, y_mid - h_mid / 2), (cx_main, y_final + h_final / 2 + 0.05))
    box(ax, renderer, cx_excl, y_excl2, excl_w, h_excl2,
        f"{flow['n_age_excluded']} excluded\nAge <{AGE_CUTOFF} years", STATE_WHITE, fontsize=11)
    arrow(ax, (cx_main + 0.08, y_excl2), (cx_excl - excl_w / 2 - 0.08, y_excl2), lw=1.2)

    box(ax, renderer, cx_main, y_final, main_w, h_final,
        f"{flow['n_final']:,} patients\n({cohort_label})", FINAL_GREEN, fontweight="bold")


def plot_diagram(flow_internal: dict, flow_external: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(19, 10))
    xlim, ylim = (-2.7, 18.9), (1.2, 11.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cx_internal_main, cx_internal_excl = 3.0, 7.6
    cx_external_main, cx_external_excl = 12.3, 16.9

    box(ax, renderer, -1.3, 9.3, 2.3, 0.9, "Enrollment", LABEL_BLUE, fontweight="bold")
    box(ax, renderer, -1.3, 6.15, 2.3, 0.9, "Vendor/scanner\nrestriction", LABEL_BLUE, fontweight="bold",
        fontsize=11)
    box(ax, renderer, -1.3, 2.7, 2.3, 0.9, "Inclusion", LABEL_BLUE, fontweight="bold")

    draw_column(ax, renderer, cx_internal_main, cx_internal_excl, flow_internal, "internal cohort", INTERNAL_SITE,
                "CT scanner not among the\n4 most common models")
    draw_column(ax, renderer, cx_external_main, cx_external_excl, flow_external, "external cohort", EXTERNAL_SITE,
                "Canon CT scanner")

    ax.text((cx_internal_main + cx_external_main) / 2, 10.5,
             "Both cohorts: CT examinations Jan 2018–Jun 2020; clinical data + abdominal CT at\n"
             "100 kVp available for the same patient (matched cohort, kVp filter already applied)",
             ha="center", va="center", fontsize=10, style="italic", color="#3a3a3a")

    #fig.suptitle("Figure. Flow diagram of patient selection", fontsize=17, fontweight="bold", y=0.97)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved patient selection flow diagram to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_g = load_raw("gangnam")
    raw_s = load_raw("sinchon")

    flow_g = compute_flow(raw_g, keep_scanners=INTERNAL_SCANNERS, drop_scanners=None)
    flow_s = compute_flow(raw_s, keep_scanners=None, drop_scanners=EXTERNAL_EXCLUDED_VENDOR_MODELS)

    verify_against_filtered_file("gangnam", flow_g["n_final"])
    verify_against_filtered_file("sinchon", flow_s["n_final"])

    print("Internal (Gangnam):", flow_g)
    print("External (Sinchon):", flow_s)

    plot_diagram(flow_g, flow_s, OUTPUT_DIR / "fig1_patient_selection_flow.png")


if __name__ == "__main__":
    main()
