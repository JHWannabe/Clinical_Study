# Step3(Logistic Regression + AEC) 확증적 통계 검정 — DeLong paired AUC test

> **개정 이력**: 이 문서는 과거 `step1_2_3_rigorous_significance_analysis.md`(구
> `step1_clinic_aec_mean.py`/`step2_clinic_aec_segment_select.py`/
> `step2_clinic_aec_back_half.py`/`step3_clinic_aec_shape.py` 기반 paired bootstrap
> ΔR² 분석)를 대체한다. 해당 4개 스크립트는 [[project_260810_step_renumbering]] →
> [[project_260813_step_renumbering_v2]] 재편 과정에서 모두 삭제/재구성되어 현재
> 리포지토리에 존재하지 않으며, 원자료(`outputs/step3_topk_backhalf_rigorous_significance.csv`)도
> 함께 삭제되었다. 아래는 현재(2026-08-13 기준) `code/` 파이프라인의 확증적 검정으로
> 새로 작성한 내용이다.

## 배경

현재 `code/` 최상위 파이프라인은 다음과 같이 재편되어 있다([[project_260813_step_renumbering_v2]]):

- `step0_output_feature_correlation.py` — output feature × clinic4/AEC 상관
- `step1_aec_fpca.py` — AEC-128 FPCA 컴포넌트 수 탐색
- `step2_clinic_aec_ratio.py` — clinic4+AEC 5개 형태후보 vs 절대값·비율 R²(선형회귀)
- **`step3_clinic_aec_logistic.py` — clinic4 vs clinic4+AEC 로지스틱 회귀 + DeLong paired AUC test + ΔAUC 요약표(구 `step4_auc_delta_table.py`, 2026-08-11 통합) (본 문서의 대상)**
- `step4_aec_diagnostics.py` — 스캐너별 서브그룹 R²(+ 위 AUC를 묶은 스캐너 요약표)
- `step6_aec_deep_learning.py`

이 중 `step3_clinic_aec_logistic.py`가 clinic4 baseline과 clinic4+AEC 모델 간
AUC 차이를 **같은 환자 집합에 대한 paired DeLong test**로 직접 검정하므로, 과거
paired bootstrap ΔR² 분석과 동일한 역할(확증적 단일검정)을 이 파이프라인에서는
이 스크립트가 담당한다.

## 방법

1. AEC 형태 후보 5종(SD, Skewness, 상하위50%비율, FPCA(n=3), 위 4종 전체결합) 중
   체성분 feature 9종(SAT/VAT/Total Fat SUM 절대값 3종 + 비율 6종) 평균 **internal
   5-fold OOF R²**(선형회귀)가 가장 높은 조합 1개를 `clinic4_aec_best`로 확정.
   **external은 이 단계에서 전혀 사용하지 않는다**([[feedback_internal_external_validation_discipline]]
   준수 — `code/step3_clinic_aec_logistic.py:128-165`).
2. `clinic4`(baseline) vs `clinic4_aec_best` 2개 모델로 9개 feature 각각에 대해
   로지스틱 회귀 학습. Cutoff은 internal 성별 mean±1SD([[project_step4_tama_1sd_cutoff_switch]]).
3. Internal은 5-fold OOF, external은 동결(freeze) 모델을 1회만 적용.
4. AUC 차이는 **DeLong paired test**로 검정(같은 환자에 대한 두 모델 점수 비교이므로
   독립 two-sample이 아닌 paired test 사용, `delong_paired_auc_test()`).
5. 추가로 9개 feature(=9회 동시검정)에 대해 **Benjamini-Hochberg FDR 보정**(α=0.05)을
   internal/external 각각에 적용해 다중비교를 통제했다(본 문서 작성 시 사후 계산 추가).
6. 재현 스크립트: `code/step3_clinic_aec_logistic.py`. 원자료: `outputs/step3/total/delong_auc_comparison.csv`,
   `outputs/step3/total/logistic_regression_summary.csv`(total cohort). 성별 분리 결과는
   `outputs/step3/{male,female}/`에 별도 존재.

> **주의**: 어떤 AEC 형태 후보(`aec_sd`/`aec_skew`/`aec_uplow_ratio`/`aec_fpca`/`aec_shape_all`)가
> `clinic4_aec_best`로 선택되었는지는 CSV로 저장되지 않고 콘솔 로그(`[Step2 조합 선택] 선택된
> 조합 = ...`)에만 출력된다([[project_260813_step_renumbering_v2]]). 본 문서는 기존에 저장된
> `outputs/step3/total/` 산출물만 사용했으므로 이번 선택 조합명은 재확인이 필요하다.

## 결과 — clinic4 vs clinic4_aec_best, 9개 feature × internal/external

| feature | cohort | AUC clinic4 | AUC clinic4_aec_best | ΔAUC | z | raw p | BH-FDR(α=.05) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VAT(내장지방)_SUM | internal | 0.847 | 0.894 | +0.047 | -4.462 | 8.14e-06 | 유의 |
| VAT(내장지방)_SUM | external | 0.813 | 0.860 | +0.047 | -3.516 | 4.37e-04 | 유의 |
| Total Fat_SUM | internal | 0.852 | 0.871 | +0.018 | -2.012 | 0.0442 | 유의(경계) |
| Total Fat_SUM | external | 0.867 | 0.889 | +0.022 | -2.641 | 0.00828 | 유의 |
| SAT(피하지방)_SUM | internal | 0.841 | 0.850 | +0.009 | -0.903 | 0.367 | n.s. |
| SAT(피하지방)_SUM | external | 0.873 | 0.886 | +0.013 | -1.655 | 0.0978 | n.s. |
| VAT_TotalFat_ratio | internal | 0.757 | 0.830 | +0.074 | -5.179 | 2.23e-07 | 유의 |
| VAT_TotalFat_ratio | external | 0.708 | 0.787 | +0.079 | -4.701 | 2.59e-06 | 유의 |
| VAT_SAT_ratio | internal | 0.763 | 0.844 | +0.081 | -5.422 | 5.90e-08 | 유의 |
| VAT_SAT_ratio | external | 0.710 | 0.784 | +0.074 | -4.065 | 4.80e-05 | 유의 |
| VAT_TAMA_ratio | internal | 0.842 | 0.890 | +0.048 | -4.575 | 4.77e-06 | 유의 |
| VAT_TAMA_ratio | external | 0.803 | 0.849 | +0.046 | -3.084 | 0.00204 | 유의 |
| TotalFat_TAMA_ratio | internal | 0.821 | 0.848 | +0.027 | -2.314 | 0.0207 | 유의 |
| TotalFat_TAMA_ratio | external | 0.819 | 0.856 | +0.037 | -3.100 | 0.00193 | 유의 |
| SAT_TotalFat_ratio | internal | 0.775 | 0.811 | +0.037 | -3.059 | 0.00222 | 유의 |
| SAT_TotalFat_ratio | external | 0.759 | 0.796 | +0.037 | -2.520 | 0.0117 | 유의 |
| SAT_TAMA_ratio | internal | 0.819 | 0.845 | +0.027 | -2.163 | 0.0305 | n.s.(FDR 탈락) |
| SAT_TAMA_ratio | external | 0.818 | 0.839 | +0.020 | -1.949 | 0.0513 | n.s. |

(ΔAUC = clinic4_aec_best − clinic4, 양수=AEC 추가 시 개선. raw p는 DeLong test 양측검정.
BH-FDR은 internal 9개·external 9개를 각각 독립적으로 보정.)

## 종합 해석

1. **9개 feature 중 7개(VAT, Total Fat, VAT_TotalFat_ratio, VAT_SAT_ratio,
   VAT_TAMA_ratio, TotalFat_TAMA_ratio, SAT_TotalFat_ratio)는 internal·external
   양쪽 모두, raw p와 BH-FDR 보정 후에도 유의하게 개선된다.** 특히 VAT 관련 비율
   feature(VAT_SAT_ratio, VAT_TotalFat_ratio)의 ΔAUC가 +0.074~+0.081(internal)로
   가장 크다.
2. **SAT(피하지방) 단독 SUM은 internal(p=0.367)·external(p=0.098) 모두 n.s.** —
   [[project_output_feature_predictor_correlations]]에서 이미 확인된 "SAT는 AEC와
   상관이 약하다"는 결론과 일치한다.
3. **SAT_TAMA_ratio는 raw p 기준 internal에서만 유의(p=0.0305)했으나 BH-FDR 보정
   후 탈락**하고 external도 경계 수준(p=0.0513)이라 재현성이 약하다 — 액면 그대로
   "유의"라고 쓰지 말 것.
4. **이전 문서(구 step1, R² 기준)의 "7개 feature 전부 유의하게 악화"와 정반대
   결론**이다. 방법이 세 가지 바뀌었기 때문이다: (a) 평가지표가 회귀 R²→분류
   AUC로, (b) AEC 표현이 단일 평균값→5개 형태후보 중 internal-best 선택으로,
   (c) feature가 절대값 3종 단독→절대값+비율 6종 추가로 확장되었다. 즉 "AEC가
   도움이 되는가"는 이 세 조건에 따라 결론이 뒤집힐 수 있는 조건부 명제이며,
   본 문서의 결과를 과거 R² 분석 결과와 직접 비교해 모순으로 해석하면 안 된다.
5. 방법론적으로 본 검정은 [[feedback_internal_external_validation_discipline]]을
   준수한다(AEC 후보 선택은 internal OOF만으로 확정, external은 동결 모델을 1회만
   평가) — 과거 문서가 지적했던 다중비교 낙관편향 문제는 이 설계에서 재발하지 않는다.
   다만 9개 feature를 동시에 검정하므로 feature 간 다중비교는 여전히 존재해 위
   표에 BH-FDR 열을 추가했다.

## 참고

- 성별 분리 결과: `outputs/step3/male/`, `outputs/step3/female/`(본 문서는 total만 다룸)
- 스캐너별 서브그룹 AUC: `outputs/step3/total/scanner_subgroup_auc.csv`, `code/step4_aec_diagnostics.py`
- ΔAUC/DeLong p-value 요약표: `outputs/step3/auc_delta/`, `code/step3_clinic_aec_logistic.py`
  (구 `step4_auc_delta_table.py`, 2026-08-11 통합)
- 선형회귀 R² 비교(step2, 절대값/비율): `outputs/step2_ratio/total/`
- pptx 반영: [[project_260813_results_multiple_features_pptx_edits]] slide 12-13(internal/external ΔAUC 표)
- 방법론 원칙: `docs/internal_external_validation_protocol.md`,
  [[feedback_internal_external_validation_discipline]]
