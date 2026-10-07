# 질환 유무 차이 분석 전체 재현: 프로젝트 루트에서 `python code/landmark/run_disease_difference_all.py`
# 순서: 칼럼 정리 -> 분포 요약 -> 미보정 효과크기 -> 보정 OR -> 여러 case feature 탐색 -> PPT (결과는 outputs/data_distribution, docs)
import subprocess
import sys

STEPS = ["drop_hw_implausible", "landmark_value_distribution_strata", "landmark_disease_difference", "landmark_adjusted_or",
         "landmark_feature_cases", "landmark_curve_by_disease_sex", "landmark_patientwise_preprocessing", "landmark_scanner_check", "landmark_pc1_adjusted", "landmark_segment_mean", "landmark_combo_features", "landmark_mlp_sets", "landmark_subgroup_auc", "landmark_subset_search", "landmark_subset_search L", "landmark_forward_selection", "build_pptx_disease_difference"]
for s in STEPS:
    print(f"== {s}", flush=True)
    name, *args = s.split()  # "스크립트 인자" 형식 지원 (예: landmark_subset_search L)
    subprocess.run([sys.executable, f"code/landmark/{name}.py", *args], check=True)
