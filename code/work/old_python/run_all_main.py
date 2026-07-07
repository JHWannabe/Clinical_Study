"""모든 .py 파일을 analysis_file_order.md 에 명시된 순서대로 import 하여
각 파일의 main() 함수를 그 순서대로 실행하는 통합 실행 스크립트.

사용법:
    python run_all_main.py

주의:
    - 실행 순서는 analysis_file_order.md 의 목록(작업 흐름 기준 정렬)을 그대로 따른다.
    - 각 스크립트는 원본 그대로 import 되므로(코드 병합이 아님) 이름 충돌이 없다.
    - 개별 스크립트가 서로를 import 하는 경우(예: aec_lock_smoothed_deesc_gate 가
      aec_conditional_value 를 import) 파이썬의 표준 import 캐싱 덕분에 자연스럽게
      처리되며, main() 은 아래 목록에 명시된 순서대로 정확히 한 번씩만 호출된다.
    - 하나의 main() 이 예외를 던져도 전체 실행을 멈추지 않고 다음 모듈로 진행하며,
      마지막에 성공/실패 요약을 출력한다.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경에서 GUI 창이 뜨지 않도록 고정

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 아래 목록은 analysis_file_order.md 에 정리된 작업 흐름 순서(수정 시각 기준)를 그대로 따른다.
MODULES_IN_ANALYSIS_ORDER = [
    # 6/29 (오전) — 초기 AEC 신호 탐색
    "aec_incremental_value",
    "aec_conditional_value",
    "aec_signal_shape",
    "aec_threshold_tradeoff",
    "rocket_aec_incremental",
    "aec_window_scan",
    "rocket_segment_aec",
    "simple_window_model_bins",
    "offset_aec_score_cyclic",
    "aec_offset_score",
    "clinical90_deescalation",
    # 6/29 (오전~오후) — 128구간 특징 엔지니어링
    "normalized_aec_mean_curves",
    "aec128_common_shape_feature",
    "aec128_visual_shape_features",
    "aec128_cylindrical_features",
    "aec128_deep_feature_mining",
    "aec128_feature_combo_models",
    "aec128_stronger_aec_only",
    "aec128_highdim_aec_only",
    "aec_within_clinical_score",
    "aec128_clinical_residualized_curves",
    "aec128_transformer_offset",
    # 6/29 (오후) — 딥러닝/특징 조합 심화
    "aec128_transformer_lr_sensitivity",
    "aec128_signal_audit",
    "aec128_alternative_shape_features",
    "aec128_mass_feature_combinations",
    "aec128_five_strategy_audit",
    "aec128_soft_gate_models",
    "aec128_residual_phenotype_discordance",
    # # 6/30 (오전) — 유니버설 게이트 탐색
    # "aec_universal_gate_search",
    # "aec_universal_boundary_gate",
    # "aec128_modulated_gate_optimization",
    # "aec128_deescalation_curve_contrast",
    # "aec128_visual_contrast_gate",
    # "aec_threshold_robustness_sweep",
    # "aec_svm_feature_interpretation",
    # "aec_threshold_free_biomarker_test",
    # "aec_conditional_feature_mining_pvalues",
    # "aec_prespecified_morphology_score",
    # "aec_midrange_feature_refit",
    # "aec_midrange_feature_refit_reference",
    # "aec_midrange_reference_curves",
    # "aec_specificity70_target_search",
    # "aec_regression_combo_gate",
    # "aec_targeted_regression_combo",
    # # 6/30 (오후) — 결합 규칙/디에스컬레이션 탐색
    # "aec_deescalation_combo_rule",
    # "aec_pair_combo_deesc_search",
    # "aec_deesc_curve_vs_clinical_auc",
    # "aec_single_operating_points_no_integral",
    # "aec_2of3_s80_90_visualize",
    # "aec_top3000_kofn_fine_search",
    # "aec_top3000_kofn_fine_search_fast",
    # "aec_block_or_consensus_search",
    # "aec_high_gain_both_cohort_search",
    # "aec_high_gain_fast_counts",
    # "plot_clinical_positive_aec_gate_curves",
    # "aec_waviness_feature_test",
    # "aec_waviness_comprehensive_metrics",
    # "aec_sex_subgroup_gate_performance",
    # "aec_sex_specific_feature_search",
    # # 7/1 (오전) — CNN/영상 기반 분석 시작
    # "aec_image_cnn_exploratory",
    # "aec_smoothing_normalization_robustness",
    # "aec_image_cnn_preprocessing_sweep",
    # "aec_cnn_case_extremes",
    # "aec_vendor_neutral_preprocessing_audit",
    # "aec_image_cnn_company_harmonized",
    # "aec_raw_auc_audit",
    # "aec_2dcnn_deep_dive",
    # "aec_2dcnn_deep_dive_extremes",
    # "aec_whole_curve_models",
    # "aec_site_difference_audit",
    # "aec_lock_smoothed_deesc_gate",
    # "aec_simple_morphology_gate_search",
    # "aec_simple_gate_oracle_external_audit",
    # "aec_residualized_simple_gate_audit",
    # # 7/1 (오후) — 영역 기반 CNN 게이트 정교화
    # "aec_semantic_gate_variants",
    # "aec_region_constrained_cnn_gate",
    # "aec_region_cnn_boundary_gate",
    # "aec_region_cnn_teacher_mimic_gate",
    # "aec_region_cnn_direct_vote_gate",
    # "aec_region_cnn_pattern_gate",
    # "aec_cnn_free_pattern_discovery",
    # "aec_region_guided_auc_cnn",
    # "aec_direct_vote_auc_boost",
    # "aec_internal_overfit_auc_demo",
    # "aec_oof_auc_max_search",
    # # 7/1 (저녁) — 재분류/정확도 탐색
    # "aec_all_reclassification",
    # "aec_all_reclassification_threshold_search",
    # "aec_accuracy_max_reclassification_search",
    # "aec_accuracy_rank_all_aec_features",
    # "aec_pick_single_accuracy_diagnosis",
    # "aec_safety_constrained_cnn_tuning",
    # "aec_direct_vote_same_rule_tuning",
    # "aec_region_search_surrogate",
    # "aec_new_region_surrogate_combo_gate",
    # "aec_new_region_guided_cnn_gate",
    # "aec_new_region_cnn_surrogate_mimic_gate",
    # "aec_final_gate_subgroup_audit",
    # # 7/2 — 최종
    # "plot_external_s90_core_1x3_mean_curves",
]


def main() -> None:
    total = len(MODULES_IN_ANALYSIS_ORDER)
    successes: list[tuple[str, float]] = []
    failures: list[tuple[str, BaseException]] = []

    for i, name in enumerate(MODULES_IN_ANALYSIS_ORDER, 1):
        print(f"\n{'=' * 80}\n[{i}/{total}] {name}.main() 실행 시작\n{'=' * 80}", flush=True)
        start = time.time()
        try:
            module = importlib.import_module(name)
            module.main()
        except Exception as exc:  # noqa: BLE001 - 한 모듈의 실패가 전체를 막지 않도록 함
            print(f"[{i}/{total}] {name} 실행 중 오류 발생:", flush=True)
            traceback.print_exc()
            failures.append((name, exc))
        else:
            elapsed = time.time() - start
            print(f"[{i}/{total}] {name}.main() 완료 ({elapsed:.1f}s)", flush=True)
            successes.append((name, elapsed))

    print(f"\n{'=' * 80}\n전체 완료: {total - len(failures)}/{total} 성공\n{'=' * 80}")
    print(f"성공한 모듈 ({len(successes)}개):")
    for name, elapsed in successes:
        print(f"  [OK] {SCRIPT_DIR / f'{name}.py'} ({elapsed:.1f}s)")
    print(f"실패한 파일 목록 ({len(failures)}개):")
    for name, exc in failures:
        print(f"  [FAIL] {SCRIPT_DIR / f'{name}.py'}: {exc!r}")


if __name__ == "__main__":
    main()
