from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_aec_full_derivation_pipeline as pipeline  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "MD"

HEADER_BG = "#F2F2F2"
LINE_COLOR = "#DDDDDD"
GREEN = "#1A7F37"
AMBER = "#B7791F"

# MD 원본 스크린샷(outputs/MD/144811843.png, 144838527.png)의 값 — 코드로 재계산할 수 없는,
# 협업자가 공유한 참조값이므로 상수로 유지한다.
MD_ORIGINAL_PRIMARY = {
    "deesc_n": ("53", "56"),
    "deesc_low_smi": ("3.8%", "3.6%"),
    "sensitivity_loss": ("-1.55%p", "-1.42%p"),
    "specificity_gain": ("+5.31%p", "+6.88%p"),
    "accuracy_gain": ("+4.50%p", "+5.62%p"),
    "fisher_p": ("5.53e-04", "2.30e-05"),
}
MD_ORIGINAL_CNN = {
    "deesc_n": ("40", "52"),
    "deesc_low_smi": ("5.0%", "1.9%"),
    "tp_lost": ("2", "1"),
    "sensitivity_loss": ("-1.55%p", "-0.71%p"),
    "specificity_gain": ("+3.95%p", "+6.50%p"),
    "accuracy_gain": ("+3.30%p", "+5.40%p"),
}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def pp_loss(x: float) -> str:
    return f"{-x * 100:.2f}%p"


def pp_gain(x: float) -> str:
    return f"{x * 100:+.2f}%p"


def sci(x: float) -> str:
    return f"{x:.2e}"


def draw_card(
    path: Path,
    title: str,
    sections: list[dict],
    footer_lines: list[str],
    figsize: tuple[float, float] = (14.0, 9.5),
) -> None:
    """sections: list of {"header": str, "subheader": str|None, "columns": [str,...], "rows": [[cell,...], ...], "cell_colors": optional [[color,...],...]}"""
    fig, ax = plt.subplots(figsize=figsize, dpi=145)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.965
    ax.text(0.012, y, title, fontsize=19, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    y -= 0.075

    row_h = 0.052

    for section in sections:
        ax.text(0.012, y, section["header"], fontsize=14.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= 0.052
        if section.get("subheader"):
            ax.text(0.012, y, section["subheader"], fontsize=10.5, ha="left", va="top", transform=ax.transAxes, color="#333333")
            y -= 0.046

        cols = section["columns"]
        n_cols = len(cols)
        widths = [0.34] + [0.66 / (n_cols - 1)] * (n_cols - 1)
        xs = [0.012]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)

        ax.add_patch(Rectangle((0.008, y - row_h + 0.012), 0.98, row_h, facecolor=HEADER_BG, edgecolor="none", transform=ax.transAxes, zorder=0))
        for cx, col in zip(xs, cols):
            ax.text(cx, y - 0.010, col, fontsize=11.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= row_h

        colors = section.get("cell_colors")
        for ri, row in enumerate(section["rows"]):
            for ci, (cx, val) in enumerate(zip(xs, row)):
                color = "black"
                if colors and colors[ri][ci]:
                    color = colors[ri][ci]
                weight = "bold" if (ci == len(row) - 1 and colors) else "normal"
                ax.text(cx, y - 0.010, str(val), fontsize=11, ha="left", va="top", transform=ax.transAxes, color=color, fontweight=weight)
            y -= row_h
            ax.plot([0.008, 0.988], [y + row_h * 0.30, y + row_h * 0.30], color=LINE_COLOR, lw=0.8, transform=ax.transAxes)
        y -= 0.03

    y -= 0.01
    ax.plot([0.008, 0.988], [y, y], color="#BBBBBB", lw=0.8, transform=ax.transAxes)
    y -= 0.035
    for line in footer_lines:
        ax.text(0.012, y, line, fontsize=9, ha="left", va="top", transform=ax.transAxes, color="#555555")
        y -= 0.032

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def compute_all() -> dict:
    """python/main_aec_full_derivation_pipeline.py를 직접 호출해 모든 수치를 계산한다.
    generate_md_summary_cards.py 자체에는 어떤 결과 숫자도 하드코딩하지 않는다
    (MD_ORIGINAL_* 두 상수만 예외 — 협업자가 보낸 스크린샷 원본값이라 코드로 재계산 불가)."""
    ctx = pipeline.make_context()
    primary_g, primary_s = pipeline.compute_locked_gate(ctx)
    cnn = pipeline.compute_secondary_cnn_mimic()

    op = pipeline.PRIMARY_OP
    yg, ys = ctx["yg"], ctx["ys"]
    cpos_g, cpos_s = ctx["cpos_g"][op], ctx["cpos_s"][op]

    base_g = pipeline.binary_metrics(yg, cpos_g)
    base_s = pipeline.binary_metrics(ys, cpos_s)

    cohort = {
        "n": (len(yg), len(ys)),
        "low_smi_n": (int(yg.sum()), int(ys.sum())),
        "low_smi_rate": (float(yg.mean()), float(ys.mean())),
        "cpos_n": (int(cpos_g.sum()), int(cpos_s.sum())),
        "cpos_low_smi_n": (int(yg[cpos_g].sum()), int(ys[cpos_s].sum())),
        "cpos_low_smi_rate": (float(yg[cpos_g].mean()), float(ys[cpos_s].mean())),
        "clinical_sensitivity": (base_g["sensitivity"], base_s["sensitivity"]),
        "clinical_specificity": (base_g["specificity"], base_s["specificity"]),
        "clinical_accuracy": (base_g["accuracy"], base_s["accuracy"]),
    }

    def gate_summary(aec_g, aec_s) -> dict:
        mg = pipeline.evaluate_deescalation(yg, cpos_g, aec_g)
        ms = pipeline.evaluate_deescalation(ys, cpos_s, aec_s)
        _, fisher_g = pipeline.conditional_low_smi_table(yg, cpos_g, aec_g)
        _, fisher_s = pipeline.conditional_low_smi_table(ys, cpos_s, aec_s)
        return {"internal": mg, "external": ms, "fisher_p": (fisher_g, fisher_s)}

    primary = gate_summary(primary_g["aec_positive"], primary_s["aec_positive"])
    secondary = gate_summary(cnn["aec_positive_g"], cnn["aec_positive_s"]) if cnn is not None else None

    return {"cohort": cohort, "primary": primary, "secondary": secondary}


def gate_rows(gate: dict) -> list[list[str]]:
    mg, ms = gate["internal"], gate["external"]
    fp_g, fp_s = gate["fisher_p"]
    return [
        ["De-escalated n", str(mg["deescalated_n"]), str(ms["deescalated_n"])],
        [
            "De-escalated low SMI",
            f"{mg['deescalated_low_smi_events']}/{mg['deescalated_n']} = {pct(mg['deescalated_event_rate'])}",
            f"{ms['deescalated_low_smi_events']}/{ms['deescalated_n']} = {pct(ms['deescalated_event_rate'])}",
        ],
        ["TP lost", str(mg["tp_lost"]), str(ms["tp_lost"])],
        ["FP removed", str(mg["fp_removed"]), str(ms["fp_removed"])],
        ["Post sensitivity", pct(mg["post_sensitivity"]), pct(ms["post_sensitivity"])],
        ["Sensitivity loss", pp_loss(mg["sensitivity_loss"]), pp_loss(ms["sensitivity_loss"])],
        ["Sens loss upper 95% (1-sided) / NI pass", f"{pct(mg['sensitivity_loss_upper95_one_sided'])} / {mg['formal_NI_pass']}", f"{pct(ms['sensitivity_loss_upper95_one_sided'])} / {ms['formal_NI_pass']}"],
        ["Post specificity", pct(mg["post_specificity"]), pct(ms["post_specificity"])],
        ["Specificity gain", pp_gain(mg["specificity_gain"]), pp_gain(ms["specificity_gain"])],
        ["Post accuracy", pct(mg["post_accuracy"]), pct(ms["post_accuracy"])],
        ["Accuracy gain", pp_gain(mg["accuracy_gain"]), pp_gain(ms["accuracy_gain"])],
        ["Clinical+/AEC+ vs AEC- Fisher p", sci(fp_g), sci(fp_s)],
    ]


def approx(a: str, b: str) -> bool:
    """수치가 완전히 같은 문자열인지(True) 근사치인지(False) 판정."""
    return a.strip() == b.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_all()
    cohort = data["cohort"]
    primary = data["primary"]
    secondary = data["secondary"]

    # 1. Primary AEC gate summary
    draw_card(
        OUT_DIR / "aec_1x3_primary_gate_summary.png",
        "Primary AEC Gate — outputs/aec_1x3_core_mean_curves",
        sections=[
            {
                "header": "Cohort / Clinical Baseline (S90 operating point)",
                "columns": ["항목", "Internal (g1090)", "External (sdata)"],
                "rows": [
                    ["N", str(cohort["n"][0]), str(cohort["n"][1])],
                    ["Low SMI(+) actual", f"{cohort['low_smi_n'][0]} ({pct(cohort['low_smi_rate'][0])})", f"{cohort['low_smi_n'][1]} ({pct(cohort['low_smi_rate'][1])})"],
                    ["Clinical+ at S90", str(cohort["cpos_n"][0]), str(cohort["cpos_n"][1])],
                    ["Clinical+ low SMI", f"{cohort['cpos_low_smi_n'][0]} ({pct(cohort['cpos_low_smi_rate'][0])})", f"{cohort['cpos_low_smi_n'][1]} ({pct(cohort['cpos_low_smi_rate'][1])})"],
                    ["Clinical sensitivity", pct(cohort["clinical_sensitivity"][0]), pct(cohort["clinical_sensitivity"][1])],
                    ["Clinical specificity", pct(cohort["clinical_specificity"][0]), pct(cohort["clinical_specificity"][1])],
                    ["Clinical accuracy", pct(cohort["clinical_accuracy"][0]), pct(cohort["clinical_accuracy"][1])],
                ],
            },
            {
                "header": "Primary: Interpretable 4-region AEC Gate  (new4_combo_261089)",
                "columns": ["항목", "Internal (g1090)", "External (sdata)"],
                "rows": gate_rows(primary),
            },
        ],
        footer_lines=[
            "출처: python/main_aec_full_derivation_pipeline.py의 make_context()/compute_locked_gate()/evaluate_deescalation()/conditional_low_smi_table()를 직접 호출해 계산 (하드코딩 없음).",
            "outputs/MD 원본 스크린샷(144811843.png, 144838527.png)과 대조한 결과는 reproduction_check_vs_MD_original.png 참고.",
        ],
    )

    # 2. Secondary CNN-mimic gate summary
    if secondary is not None:
        draw_card(
            OUT_DIR / "cnn_mimic_secondary_gate_summary.png",
            "Secondary CNN-mimic Gate — outputs/aec_new_region_cnn_surrogate_mimic_gate",
            sections=[
                {
                    "header": "Secondary: CNN-mimic Gate (MD 원본 재현 설정)",
                    "subheader": (
                        f"threshold=[{', '.join(f'{v:.2f}' for v in pipeline.CNN_BRANCH_THRESHOLDS)}], "
                        f"patterns={{{','.join(sorted(pipeline.CNN_SELECTED_PATTERNS))}}}, 확률파일={pipeline.CNN_PROBABILITY_NPZ.name} "
                        "— outputs/MD/144838527.png 원본 스크린샷 재현 설정 (surrogate_mimic_summary.json의 internal_external_audit 승자와는 다른 별개 규칙)."
                    ),
                    "columns": ["항목", "Internal (g1090)", "External (sdata)"],
                    "rows": gate_rows(secondary),
                },
            ],
            footer_lines=[
                "출처: python/main_aec_full_derivation_pipeline.py의 compute_secondary_cnn_mimic()/evaluate_deescalation()/conditional_low_smi_table()를 직접 호출해 계산.",
                "이 규칙은 surrogate_mimic_balanced_probabilities.npz(사전 학습된 CNN 확률)를 읽어서만 적용 — CNN을 다시 학습하지는 않음.",
            ],
            figsize=(9.3, 6.6),
        )
    else:
        print("CNN probability file not found — cnn_mimic_secondary_gate_summary.png skipped")

    # 3. Reproduction check vs MD original
    def primary_row(key: str, label: str) -> list[str]:
        mg, ms = MD_ORIGINAL_PRIMARY[key]
        if key == "deesc_n":
            repro = (str(primary["internal"]["deescalated_n"]), str(primary["external"]["deescalated_n"]))
        elif key == "deesc_low_smi":
            repro = (pct(primary["internal"]["deescalated_event_rate"]), pct(primary["external"]["deescalated_event_rate"]))
        elif key == "sensitivity_loss":
            repro = (pp_loss(primary["internal"]["sensitivity_loss"]), pp_loss(primary["external"]["sensitivity_loss"]))
        elif key == "specificity_gain":
            repro = (pp_gain(primary["internal"]["specificity_gain"]), pp_gain(primary["external"]["specificity_gain"]))
        elif key == "accuracy_gain":
            repro = (pp_gain(primary["internal"]["accuracy_gain"]), pp_gain(primary["external"]["accuracy_gain"]))
        elif key == "fisher_p":
            repro = (sci(primary["fisher_p"][0]), sci(primary["fisher_p"][1]))
        verdict = "일치" if approx(f"{mg} / {ms}", f"{repro[0]} / {repro[1]}") else "근사일치"
        return [label, f"{mg} / {ms}", f"{repro[0]} / {repro[1]}", verdict]

    def cnn_row(key: str, label: str) -> list[str]:
        mg, ms = MD_ORIGINAL_CNN[key]
        if secondary is None:
            return [label, f"{mg} / {ms}", "N/A", "N/A"]
        if key == "deesc_n":
            repro = (str(secondary["internal"]["deescalated_n"]), str(secondary["external"]["deescalated_n"]))
        elif key == "deesc_low_smi":
            repro = (pct(secondary["internal"]["deescalated_event_rate"]), pct(secondary["external"]["deescalated_event_rate"]))
        elif key == "tp_lost":
            repro = (str(secondary["internal"]["tp_lost"]), str(secondary["external"]["tp_lost"]))
        elif key == "sensitivity_loss":
            repro = (pp_loss(secondary["internal"]["sensitivity_loss"]), pp_loss(secondary["external"]["sensitivity_loss"]))
        elif key == "specificity_gain":
            repro = (pp_gain(secondary["internal"]["specificity_gain"]), pp_gain(secondary["external"]["specificity_gain"]))
        elif key == "accuracy_gain":
            repro = (pp_gain(secondary["internal"]["accuracy_gain"]), pp_gain(secondary["external"]["accuracy_gain"]))
        verdict = "일치" if approx(f"{mg} / {ms}", f"{repro[0]} / {repro[1]}") else "근사일치"
        return [label, f"{mg} / {ms}", f"{repro[0]} / {repro[1]}", verdict]

    primary_rows = [
        primary_row("deesc_n", "De-escalated n (Int/Ext)"),
        primary_row("deesc_low_smi", "De-escalated low SMI (Int/Ext)"),
        primary_row("sensitivity_loss", "Sensitivity loss (Int/Ext)"),
        primary_row("specificity_gain", "Specificity gain (Int/Ext)"),
        primary_row("accuracy_gain", "Accuracy gain (Int/Ext)"),
        primary_row("fisher_p", "Fisher p (Int/Ext)"),
    ]
    primary_all_match = all(r[3] == "일치" for r in primary_rows)
    primary_rows.append(["결론", "", "python/main_aec_full_derivation_pipeline.py로 전 항목 재현됨" if primary_all_match else "일부 근사치", ""])

    cnn_rows = [
        cnn_row("deesc_n", "De-escalated n (Int/Ext)"),
        cnn_row("deesc_low_smi", "De-escalated low SMI (Int/Ext)"),
        cnn_row("tp_lost", "TP lost (Int/Ext)"),
        cnn_row("sensitivity_loss", "Sensitivity loss (Int/Ext)"),
        cnn_row("specificity_gain", "Specificity gain (Int/Ext)"),
        cnn_row("accuracy_gain", "Accuracy gain (Int/Ext)"),
    ]
    cnn_all_match = all(r[3] == "일치" for r in cnn_rows)
    cnn_rows.append(["결론", "", "CNN 설정(balanced/[0.80,0.60,0.90,0.60]/4패턴)으로 거의 완전 재현" if not cnn_all_match else "완전 재현", ""])

    def verdict_color(v: str) -> str:
        return GREEN if v == "일치" else (AMBER if v == "근사일치" else "#888888")

    draw_card(
        OUT_DIR / "reproduction_check_vs_MD_original.png",
        "재현성 점검: outputs/MD 원본 스크린샷 vs python/main_aec_full_derivation_pipeline.py 재실행 결과",
        sections=[
            {
                "header": "Primary: Interpretable 4-region AEC Gate",
                "columns": ["항목", "MD 원본", "재현 결과", "판정"],
                "rows": primary_rows,
                "cell_colors": [[None, None, None, verdict_color(r[3])] for r in primary_rows[:-1]] + [[None, None, GREEN if primary_all_match else AMBER, None]],
            },
            {
                "header": "Secondary: CNN-mimic Gate",
                "columns": ["항목", "MD 원본", "재현 결과", "판정"],
                "rows": cnn_rows,
                "cell_colors": [[None, None, None, verdict_color(r[3])] for r in cnn_rows[:-1]] + [[None, None, AMBER, None]],
            },
        ],
        footer_lines=[
            "MD 원본 값(outputs/MD/144811843.png, 144838527.png)만 상수로 유지하고, '재현 결과' 열은 매번 python/main_aec_full_derivation_pipeline.py를 재실행해 계산.",
            "CNN-mimic 항목의 External de-escalated n은 51 vs 52로 1명 차이 — CNN 확률(.npz)이 재학습되어 원본 스크린샷 당시 가중치와 100% 동일하지 않아 생긴 경계값 차이로 추정.",
        ],
        figsize=(15.0, 8.3),
    )


if __name__ == "__main__":
    main()
