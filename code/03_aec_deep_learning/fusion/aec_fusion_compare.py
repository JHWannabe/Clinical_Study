from __future__ import annotations

# concat/gated/attnpool/crossattn 4개 fusion 스크립트가 각자 outputs/fusion_*/에 저장한
# classification_summary.csv·delong_vs_clinic4.csv를 읽기만 해서(이 스크립트가 쓰는 곳은
# outputs/fusion_compare/뿐 — [[feedback_output_dir_single_producer]]) 하나의 비교 표·그래프로 합친다.
# 4개 스크립트를 먼저 전부 실행해 각자의 outputs/fusion_*/classification_summary.csv가 있어야 동작한다.

from pathlib import Path

import pandas as pd

from aec_fusion_common import PROJECT_ROOT, plot_auc_grouped

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "compare"

FUSION_DIRS = {
    "concat": PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "concat",
    "gated": PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "gated",
    "attnpool": PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "attnpool",
    "crossattn": PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "crossattn",
}
MODEL_ORDER = ["clinic4", "concat", "gated", "attnpool", "crossattn"]
COLORS = {"clinic4": "#6b6a66", "concat": "#2a78d6", "gated": "#2c6b67", "attnpool": "#bd5c1e", "crossattn": "#8a3ea1"}


# 4개 fusion 디렉터리의 classification_summary.csv를 모아 하나의 표로 합침(clinic4 baseline 행은 중복이므로 첫 파일 것만 채택)
def load_summaries() -> pd.DataFrame:
    frames = []
    for i, (fusion_name, out_dir) in enumerate(FUSION_DIRS.items()):
        path = out_dir / "classification_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음 — 먼저 code/aec_fusion_{fusion_name}.py를 실행할 것")
        df = pd.read_csv(path)
        if i > 0:
            df = df[df["model"] != "clinic4"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# 4개 fusion 디렉터리의 delong_vs_clinic4.csv(각 fusion vs clinic4 검정)를 모음
def load_delong() -> pd.DataFrame:
    frames = []
    for fusion_name, out_dir in FUSION_DIRS.items():
        path = out_dir / "delong_vs_clinic4.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"[경고] {path} 없음 — {fusion_name} DeLong 결과 제외")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summaries()
    summary.to_csv(OUTPUT_DIR / "fusion_comparison_summary.csv", index=False)
    print(summary.to_string(index=False))

    delong = load_delong()
    if not delong.empty:
        delong.to_csv(OUTPUT_DIR / "fusion_comparison_delong.csv", index=False)
        print(delong.to_string(index=False))

    plot_auc_grouped(
        summary, OUTPUT_DIR / "fusion_comparison_auc.png",
        model_order=MODEL_ORDER, colors=COLORS, title="AUC 비교 — fusion 방식 4종 vs clinic4",
    )


if __name__ == "__main__":
    main()
