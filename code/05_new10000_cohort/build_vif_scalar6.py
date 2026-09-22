"""사용자 요청(2026-09-17): 슬라이드6(체성분 스칼라 동시투입 VIF 체크)에서 BMI, SMI를 제거해
6종(+TAT)으로 축소. VIF는 clinic4+체성분 설계행렬에만 의존해 HTN/DM/CKD 값이 동일하므로(대표 HTN만
재생성). BODY_COMP_COLS 자체는 슬라이드7 등 개별모델 비교 전체에 쓰이므로 건드리지 않고, 이 차트
전용 상수 VIF_SCALAR_COLS만 모듈 로드 후 재대입해 plot_vif_scalar_combined()를 재사용한다.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "bc_compare", PROJECT_ROOT / "code" / "0910" / "clinic_body_composition_individual_compare.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

SCALAR_COLS = ["VAT", "SAT", "IMATA", "NAMA", "LAMA", "TAMA", "TAT"]  # BMI, SMI 제외(6종+TAT)


# bc.plot_vif_scalar_combined은 suptitle에 "9종"을 하드코딩하고 있어 그대로 재사용하면 제목이 실제
# 종수(6)와 어긋난다 - 동일 로직을 복사하되 제목만 동적으로 맞춘다
def plot_vif_scalar6(feat: str, meta_int_m, meta_ext_m, out_path: Path) -> None:
    x_int, scaler, _ = bc.build_matrix(meta_int_m, SCALAR_COLS)
    x_ext, _, _ = bc.build_matrix(meta_ext_m, SCALAR_COLS, scaler=scaler)
    cols = ["sex_M", "age", "height", "weight"] + SCALAR_COLS

    fig, axes = bc.plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, cohort, x_full in zip(axes, ["internal", "external"], [x_int, x_ext]):
        vif = bc.compute_vif(x_full, cols, SCALAR_COLS)
        display_vif = vif["vif"].clip(upper=50)
        colors = [bc._vif_bar_color(v) for v in vif["vif"]]
        bars = ax.barh(vif["variable"], display_vif, color=colors)
        bc._annotate_vif_bars(ax, bars, vif["vif"])
        ax.axvline(5, color="#e67e22", linestyle="--", linewidth=1, label="VIF=5")
        ax.axvline(10, color="#c0392b", linestyle="--", linewidth=1, label="VIF=10")
        ax.set_title(cohort, fontweight="bold")
        ax.set_xlabel("VIF")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"{feat}: VIF — clinic4 + body composition scalar 6종 동시 투입", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    bc.plt.close(fig)
    print(f"Saved VIF plot to {out_path}")


def main() -> None:
    meta_int = bc.load_cohort(bc.INTERNAL_XLSX)
    meta_ext = bc.load_cohort(bc.EXTERNAL_XLSX)

    feat = "HTN"
    y_int_all = meta_int[feat].astype(float).to_numpy()
    y_ext_all = meta_ext[feat].astype(float).to_numpy()
    meta_int_m = meta_int[bc.np.isfinite(y_int_all)].reset_index(drop=True)
    meta_ext_m = meta_ext[bc.np.isfinite(y_ext_all)].reset_index(drop=True)

    out_path = PROJECT_ROOT / "outputs" / "0910" / "htn" / "vif" / "scalar_combined.png"
    plot_vif_scalar6(feat, meta_int_m, meta_ext_m, out_path)


if __name__ == "__main__":
    main()
