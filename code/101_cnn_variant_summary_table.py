from __future__ import annotations

# Renders docs/cnn_variant_comparison.md's section-2 summary table ("변형별 요약") as a
# standalone PNG for embedding outside the markdown viewer. Static content -- no data is
# read from outputs/, so this only needs to be re-run when that table's text changes.
#
# Run: python code/101_cnn_variant_summary_table.py

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "docs" / "cnn_variant_comparison_table.png"

HEADER_BG = "#1c1c1c"
HEADER_FG = "#ffffff"
BAND_BG = "#f6f6f4"
TEXT = "#161616"

# (file, what changed, what didn't, cost vs. baseline) -- condensed from
# docs/cnn_variant_comparison.md section 2's table.
ROWS = [
    ("4_aec_cnn_\npretrain.py",
     "conv encoder를 내부 코호트 전체(n=1090, 라벨 없음)에 denoising-autoencoder로 "
     "사전학습한 뒤, 그 가중치로 각 fold/seed 모델을 초기화",
     "아키텍처, 앙상블 절차, 폴드/threshold 로직",
     "사전학습 1회분 추가 (~수십 epoch, 1회만)"),
    ("5_aec_cnn_\nskip.py",
     "ResidualCNN에 실제 residual/skip connection 추가 (relu(res_scale·block3(h2) + h2)), "
     "res_scale_init x block3_kernel(4x2=8 조합) 그리드서치로 튜닝",
     "학습 절차, 앙상블, 폴드/threshold 로직",
     "거의 동일 (파라미터 증가 없음) x 그리드 8배 (grid point마다 baseline 1회 실행 전체를 수행)"),
    ("6_aec_cnn_\nbagging.py",
     "앙상블 멤버마다 fold-train 데이터를 클래스층화 bootstrap 재추출 후 학습, "
     "bootstrap_frac x n_members(3x3=9 조합) 그리드서치로 튜닝",
     "아키텍처, epoch 선택 로직, 폴드/threshold 로직",
     "거의 동일 x 그리드 9배 (grid point마다 baseline 1회 실행 전체를 수행)"),
    ("7_aec_cnn_\nrepeatedcv.py",
     "StratifiedKFold(5-fold)를 서로 다른 시드로 N_REPEATS=5회 반복하고 OOF logit을 평균",
     "아키텍처, 앙상블 절차, threshold 선택 로직 자체",
     "5배 (fold x repeat)"),
    ("8_aec_cnn_\nfilm.py",
     "side feature(clinical/stage1 score)를 마지막 concat 대신 FiLM((1+γ)·z+β)으로 "
     "conv 임베딩에 주입",
     "학습 절차, 앙상블, 폴드/threshold 로직",
     "거의 동일"),
    ("9_aec_cnn_\nfcn.py",
     "ResidualCNN을 시계열 분류 표준 아키텍처인 FCN(128ch/k=8 → 256ch/k=5 → 128ch/k=3, "
     "전 구간 padding=\"same\" + GAP)으로 교체",
     "학습 절차, 앙상블, 폴드/threshold 로직",
     "파라미터 ~150배 증가(~2천 → ~33만), 그리드 없음"),
    ("10_aec_cnn_\ncrossattn.py",
     "side feature가 curve embedding 전체를 균일하게 스케일/이동시키는 FiLM 대신, side "
     "feature로 만든 query가 block3의 32-position 시퀀스에 cross-attention해 위치별로 "
     "다른 context를 뽑아내도록 함 (z = pooled + gate·context, gate는 0-init)",
     "학습 절차, 앙상블, 폴드/threshold 로직",
     "거의 동일 (attention head 파라미터만 소폭 증가)"),
]

# column (left, right) as a fraction of the table width -- 파일 widened, 무엇을 바꿨나
# narrowed to make room, 안 바꾼 것/계산 비용 left roughly as-is.
COL = {"file": (0.000, 0.155), "changed": (0.155, 0.545), "same": (0.545, 0.775), "cost": (0.775, 1.000)}
WRAP = {"file": 13, "changed": 40, "same": 13, "cost": 14}


def wrap_cell(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width, break_long_words=True) or [""])
    return lines


def main() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    line_h = 0.34
    row_pad = 0.34
    header_h = 0.85
    title_h = 0.55

    wrapped_rows = []
    for file_, changed, same, cost in ROWS:
        cells = {
            "file": wrap_cell(file_, WRAP["file"]),
            "changed": wrap_cell(changed, WRAP["changed"]),
            "same": wrap_cell(same, WRAP["same"]),
            "cost": wrap_cell(cost, WRAP["cost"]),
        }
        n_lines = max(len(v) for v in cells.values())
        row_h = n_lines * line_h + row_pad
        wrapped_rows.append((cells, row_h))

    total_h = title_h + header_h + sum(h for _, h in wrapped_rows)
    fig, ax = plt.subplots(figsize=(15.0, total_h * 0.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    fig.suptitle("Stage-2 1D-CNN 개선안 — 변형별 요약 (cnn_variant_comparison.md §2)",
                 x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold", color=TEXT)

    header_top = total_h - title_h
    header_bottom = header_top - header_h
    ax.add_patch(Rectangle((0, header_bottom), 1, header_h, facecolor=HEADER_BG, edgecolor="none", zorder=1))
    for key, lines in [("file", ["파일"]), ("changed", ["무엇을 바꿨나"]), ("same", ["안 바꾼 것"]),
                        ("cost", ["계산 비용", "(baseline 대비)"])]:
        cx = (COL[key][0] + COL[key][1]) / 2
        mid_y = header_bottom + header_h / 2
        top_y = mid_y + (len(lines) - 1) * 0.24 / 2
        for li, line in enumerate(lines):
            ax.text(cx, top_y - li * 0.24, line, ha="center", va="center",
                    color=HEADER_FG, fontsize=13, fontweight="bold")

    y_cursor = header_bottom
    for gi, (cells, row_h) in enumerate(wrapped_rows):
        block_top = y_cursor
        block_bottom = y_cursor - row_h
        if gi % 2 == 0:
            ax.add_patch(Rectangle((0, block_bottom), 1, row_h, facecolor=BAND_BG, edgecolor="none", zorder=0))

        mid_y = (block_top + block_bottom) / 2
        for key, font, weight, size in [("file", "monospace", "bold", 12.5),
                                          ("changed", None, "normal", 12),
                                          ("same", None, "normal", 12),
                                          ("cost", None, "normal", 12)]:
            lines = cells[key]
            cx = (COL[key][0] + COL[key][1]) / 2
            top_y = mid_y + (len(lines) - 1) * line_h / 2
            for li, line in enumerate(lines):
                kwargs = dict(ha="center", va="center", fontsize=size, color=TEXT, fontweight=weight)
                if font:
                    kwargs["fontfamily"] = font
                ax.text(cx, top_y - li * line_h, line, **kwargs)

        y_cursor = block_bottom

    ax.plot([0, 1], [header_bottom, header_bottom], color=HEADER_BG, linewidth=1.2)
    for key in ("file", "changed", "same"):
        x = COL[key][1]
        ax.plot([x, x], [header_bottom, y_cursor], color="#d8d8d4", linewidth=0.8, zorder=2)

    fig.tight_layout(rect=(0.005, 0.005, 0.995, 0.98))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
