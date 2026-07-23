"""AEC 곡선 그룹 비교 p-value 유의성 표 이미지 생성 (PPT 첨부용).

00_group_diff_summary.csv / 16_19_stage2_group_comparison_summary.csv를
유의(p<0.05) / 비유의로 재정렬한 표 PNG를 신촌·강남 각 소스별로 저장한다.
"""
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SIG_BG = "#e3f2ee"
SIG_TEXT = "#0d7d68"
NS_BG = "#f4f3ef"
HEADER_BG = "#0b0b0b"

COLS = ["구분", "비교", "방향", "Curve RMSD", "p-value"]
COL_WIDTHS = [0.09, 0.46, 0.20, 0.13, 0.12]

SINCHON_CLINIC = [
    ("Sig", "01c. Low-SMI 전체, peak 구간(slice 99–119)", "Low > Non-low", "0.0418", "0.0005"),
    ("Sig", "07. BMI ≤ 성별 median", "Low < Non-low", "0.0386", "0.0010"),
    ("Sig", "08. BMI4 · Normal", "Low < Non-low", "0.0429", "0.0010"),
    ("Sig", "07. BMI > 성별 median", "Low > Non-low", "0.0691", "0.0025"),
    ("Sig", "08. BMI4 · Overweight", "Low < Non-low", "0.0645", "0.0025"),
    ("Sig", "01. Low-SMI cutoff (전체)", "Low > Non-low", "0.0266", "0.0115"),
    ("Sig", "04. TAMA ≤ 성별 median", "Low > Non-low", "0.0274", "0.0145"),
    ("Sig", "08. BMI4 · Obese", "Low > Non-low", "0.0915", "0.0145"),
    ("Sig", "01d. Propensity-matched, peak 구간(slice 1–13)", "Low < Non-low", "0.0540", "0.0175"),
    ("Sig", "06. 체중 > median", "Low > Non-low", "0.0354", "0.0240"),
    ("Sig", "01b. Propensity-matched (전체)", "Low < Non-low", "0.0347", "0.0265"),
    ("Sig", "06. 체중 ≤ median", "Low < Non-low", "0.0279", "0.0500"),
    ("NS", "03. 나이 ≤ median", "Low > Non-low", "0.0306", "0.0625"),
    ("NS", "05. 신장 > median", "Low > Non-low", "0.0255", "0.0665"),
    ("NS", "09. Vendor · GE", "Low > Non-low", "0.0344", "0.0685"),
    ("NS", "08. BMI4 · Underweight", "Low < Non-low", "0.0533", "0.1269"),
    ("NS", "03. 나이 > median", "Low > Non-low", "0.0175", "0.2199"),
    ("NS", "02. 성별 · Male", "Low < Non-low", "0.0144", "0.3363"),
    ("NS", "02. 성별 · Female", "Low < Non-low", "0.0195", "0.3713"),
    ("NS", "05. 신장 ≤ median", "Low < Non-low", "0.0166", "0.4833"),
    ("NS", "09. Vendor · Siemens", "Low > Non-low", "0.0138", "0.5107"),
    ("NS", "09. Vendor · Philips", "Low > Non-low", "0.0177", "0.7681"),
]

GANGNAM_CLINIC = [
    ("Sig", "02. 성별 · Female", "Low > Non-low", "0.0530", "0.0005"),
    ("Sig", "02. 성별 · Male", "Low < Non-low", "0.0324", "0.0035"),
    ("Sig", "08. BMI4 · Normal", "Low > Non-low", "0.0385", "0.0035"),
    ("Sig", "01c. Low-SMI 전체, peak 구간(slice 110–128)", "Low > Non-low", "0.0318", "0.0040"),
    ("Sig", "08. BMI4 · Overweight", "Low > Non-low", "0.0649", "0.0065"),
    ("Sig", "09. Vendor · Philips", "Low > Non-low", "0.0563", "0.0095"),
    ("Sig", "07. BMI ≤ 성별 median", "Low > Non-low", "0.0277", "0.0125"),
    ("Sig", "03. 나이 ≤ median", "Low < Non-low", "0.0328", "0.0230"),
    ("Sig", "09. Vendor · Siemens", "Low < Non-low", "0.0273", "0.0275"),
    ("Sig", "04. TAMA ≤ 성별 median", "Low > Non-low", "0.0222", "0.0420"),
    ("Sig", "01. Low-SMI cutoff (전체)", "Low > Non-low", "0.0210", "0.0475"),
    ("NS", "09. Vendor · GE", "Low > Non-low", "0.0420", "0.0570"),
    ("NS", "07. BMI > 성별 median", "Low < Non-low", "0.0425", "0.0670"),
    ("NS", "05. 신장 > median", "Low < Non-low", "0.0236", "0.0695"),
    ("NS", "06. 체중 ≤ median", "Low > Non-low", "0.0222", "0.0700"),
    ("NS", "03. 나이 > median", "Low > Non-low", "0.0201", "0.1139"),
    ("NS", "05. 신장 ≤ median", "Low > Non-low", "0.0290", "0.1164"),
    ("NS", "08. BMI4 · Underweight", "Low < Non-low", "0.0492", "0.1174"),
    ("NS", "06. 체중 > median", "Low > Non-low", "0.0227", "0.1234"),
    ("NS", "08. BMI4 · Obese", "Low < Non-low", "0.0318", "0.2459"),
    ("NS", "01d. Propensity-matched, peak 구간(slice 102–122)", "Low < Non-low", "0.0220", "0.2504"),
    ("NS", "01b. Propensity-matched (전체)", "Low < Non-low", "0.0135", "0.5377"),
]

SINCHON_STAGE2 = [
    ("Sig", "TP vs TN (dp)", "TP < TN", "0.0023", "0.0005"),
    ("Sig", "FP vs TN (dp)", "FP < TN", "0.0017", "0.0005"),
    ("Sig", "FP vs TN (p)", "FP < TN", "0.0226", "0.0045"),
    ("Sig", "TP vs FP (dp)", "TP < FP", "0.0016", "0.0050"),
    ("Sig", "TP vs FP, propensity-matched (dp)", "TP > FP", "0.0023", "0.0095"),
    ("Sig", "TP vs TN (p)", "TP > TN", "0.0289", "0.0135"),
    ("Sig", "TP vs FP, propensity-matched (p)", "TP < FP", "0.0422", "0.0165"),
    ("Sig", "TP vs FP (p)", "TP < FP", "0.0274", "0.0175"),
    ("NS", "TP vs TN (d2p)", "TP > TN", "0.00056", "0.0840"),
    ("NS", "FP vs TN (d2p)", "FP > TN", "0.00042", "0.0935"),
    ("NS", "TP vs FP, propensity-matched (d2p)", "TP > FP", "0.00075", "0.4168"),
    ("NS", "TP vs FP (d2p)", "TP < FP", "0.00054", "0.4253"),
]

GANGNAM_STAGE2 = [
    ("Sig", "TP vs TN (dp)", "TP < TN", "0.0026", "0.0005"),
    ("Sig", "TP vs TN (d2p)", "TP < TN", "0.00059", "0.0005"),
    ("Sig", "FP vs TN (p)", "FP < TN", "0.0287", "0.0005"),
    ("Sig", "FP vs TN (dp)", "FP < TN", "0.0021", "0.0005"),
    ("Sig", "FP vs TN (d2p)", "FP < TN", "0.00037", "0.0005"),
    ("Sig", "TP vs TN (p)", "TP < TN", "0.0325", "0.0020"),
    ("NS", "TP vs FP (dp)", "TP < FP", "0.0012", "0.1554"),
    ("NS", "TP vs FP (d2p)", "TP < FP", "0.00048", "0.1989"),
    ("NS", "TP vs FP (p)", "TP > FP", "0.0155", "0.2274"),
    ("NS", "TP vs FP, propensity-matched (d2p)", "TP > FP", "0.00062", "0.4778"),
    ("NS", "TP vs FP, propensity-matched (dp)", "TP > FP", "0.00093", "0.9815"),
    ("NS", "TP vs FP, propensity-matched (p)", "TP < FP", "0.0057", "0.9830"),
]


ROW_H_IN = 0.30
HEADER_H_IN = 0.34
TITLE_BLOCK_IN = 0.62


def render_table(rows, title, subtitle, out_path):
    n = len(rows)
    table_h_in = HEADER_H_IN + n * ROW_H_IN
    fig_h = TITLE_BLOCK_IN + table_h_in
    fig, ax = plt.subplots(figsize=(10.5, fig_h), facecolor=SURFACE)
    ax.axis("off")

    ax.text(0.0, 1.0, title, transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=INK_PRIMARY, va="top", ha="left")
    ax.text(0.0, 1.0 - 0.34 / fig_h, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=INK_MUTED, va="top", ha="left")

    table_top = 1.0 - TITLE_BLOCK_IN / fig_h
    table = ax.table(cellText=[list(COLS)] + [list(r) for r in rows],
                      colWidths=COL_WIDTHS, loc="upper left",
                      bbox=[0.0, 0.0, 1.0, table_top])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    row_frac = ROW_H_IN / table_h_in
    header_frac = HEADER_H_IN / table_h_in

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.PAD = 0.02
        if r == 0:
            cell.set_height(header_frac)
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="#fcfcfb", fontweight="bold", fontsize=9,
                                 ha="left" if c in (1, 2) else "center" if c == 0 else "right")
            continue
        cell.set_height(row_frac)
        label = rows[r - 1][0]
        is_sig = label == "Sig"
        cell.set_facecolor(SIG_BG if is_sig else NS_BG if r % 2 == 0 else SURFACE)
        if c == 0:
            cell.set_text_props(color=SIG_TEXT if is_sig else INK_MUTED,
                                 fontweight="bold", ha="center")
        elif c == 4:
            cell.set_text_props(color=SIG_TEXT if is_sig else INK_MUTED,
                                 fontweight="bold", ha="right")
        elif c == 3:
            cell.set_text_props(color=INK_SECONDARY, ha="right")
        elif c == 2:
            cell.set_text_props(color=INK_SECONDARY, ha="left")
        else:
            cell.set_text_props(color=INK_PRIMARY, ha="left")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    base = "outputs/0_clinic-only_baseline/aec_curve_comparison"

    render_table(
        SINCHON_CLINIC,
        "신촌 · 임상 변수별 Low-SMI vs Non-low SMI AEC 곡선 비교",
        "00_group_diff_summary · whole-curve RMSD permutation (n_perm=2000) · 유의 12 / 비유의 10, 22건",
        f"{base}/sinchon/00_pvalue_significance_table.png",
    )
    render_table(
        SINCHON_STAGE2,
        "신촌 · Stage-2 오분류(TP/FP/TN) AEC 곡선 비교",
        "16_19_stage2_group_comparison_summary · whole-curve RMSD permutation (n_perm=2000) · 유의 8 / 비유의 4, 12건",
        f"{base}/sinchon/16_19_pvalue_significance_table.png",
    )
    render_table(
        GANGNAM_CLINIC,
        "강남 · 임상 변수별 Low-SMI vs Non-low SMI AEC 곡선 비교",
        "00_group_diff_summary · whole-curve RMSD permutation (n_perm=2000) · 유의 11 / 비유의 11, 22건",
        f"{base}/gangnam/00_pvalue_significance_table.png",
    )
    render_table(
        GANGNAM_STAGE2,
        "강남 · Stage-2 오분류(TP/FP/TN) AEC 곡선 비교",
        "16_19_stage2_group_comparison_summary · whole-curve RMSD permutation (n_perm=2000) · 유의 6 / 비유의 6, 12건",
        f"{base}/gangnam/16_19_pvalue_significance_table.png",
    )
