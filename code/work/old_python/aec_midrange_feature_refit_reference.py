from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    OUT_DIR,
    adjusted_deesc_p,
    bootstrap_metrics,
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    plot_selected,
    risk_direction,
    standardize_train_test,
)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_midrange_feature_refit에서 train으로 뽑은 "1위" 특징
    대신, 외부 성능 요약표에서 가장 위에 있는 "참고용(reference)" 특징을 따로 떼어내 더 자세히
    감사하면 어떤가? — 채택된 특징에 대한 추가 확인용 스크립트):

    1. aec_midrange_feature_refit이 저장해둔 외부 성능 요약 CSV와 top80 평가 CSV를 읽어, 그 표의
       맨 위 행(reference)이 가리키는 특징·폭·람다·train_rank를 가져온다.
    2. 그 train_rank에 해당하는 모든 운영점 평가 결과를 뽑아 CSV로 저장하고, 운영점별 성능 그래프로 시각화.
    3. g1090/sdata를 다시 로드해 같은 특징의 표준화·방향고정 값을 재계산.
    4. 스캐너(제조사) 더미변수를 통제했을 때도 하향조정이 유의한지(adjusted_deesc_p, 임상점수
       포함/미포함 두 버전) 확인하고, 부트스트랩으로 외부 신뢰구간도 재계산.
    5. 참고 특징 정보, 외부 운영점 결과, 조정된 p값, 부트스트랩 결과를 콘솔에 출력.
    """
    summary = pd.read_csv(OUT_DIR / "midrange_train_selected_external_primary_summary.csv")
    eval_df = pd.read_csv(OUT_DIR / "midrange_train_selected_top80_external_eval.csv")
    reference = summary.iloc[0]
    rank = int(reference["train_rank"])
    feature = str(reference["feature"])
    width = float(reference["width"])
    lam = float(reference["lambda"])

    ref_eval = eval_df[eval_df["train_rank"].eq(rank)].copy()
    ref_eval.to_csv(OUT_DIR / "external_balanced_reference_all_operating_points.csv", index=False)
    plot_selected(
        ref_eval,
        OUT_DIR / "external_balanced_reference_operating_points.png",
        f"Exploratory external-balanced AEC feature: train rank {rank}",
    )

    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]
    idx = names.index(feature)

    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    adjusted = pd.concat(
        [
            adjusted_deesc_p(s["y"], c_s, xs[:, idx], scanner_s, thresholds, width, lam, include_clinical=False),
            adjusted_deesc_p(s["y"], c_s, xs[:, idx], scanner_s, thresholds, width, lam, include_clinical=True),
        ],
        ignore_index=True,
    )
    adjusted.to_csv(OUT_DIR / "external_balanced_reference_adjusted_pvalues.csv", index=False)
    boot = bootstrap_metrics(s["y"], c_s, xs[:, idx], thresholds, width, lam)
    boot.to_csv(OUT_DIR / "external_balanced_reference_bootstrap.csv", index=False)

    print("reference")
    print(reference.to_string())
    print("\nexternal operating points")
    print(ref_eval[ref_eval["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nadjusted p-values")
    print(adjusted.to_string(index=False))
    print("\nbootstrap")
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
