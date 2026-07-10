# 2단계 Low-SMI 분류 모델 설계

Clinical-only 1차 스크리닝(`code/baseline/clinic-only_baseline.py`)이 만들어내는 대량의 FP를
AEC-128 잔차 정보로 재분류하는 2차 분류기의 설계 원칙. 실제 구현은 두 개의 형제 스크립트로
나뉘어 있으며(각각 `outputs/<파일명>/`에 독립적으로 결과를 저장), 차이는 문서 끝의
"구현 파일별 차이" 절 참고.

## 1. 2차 분류 대상은 "정답 기반 FP/FN"이 아니라 "1차 예측 기반 그룹"

실제 배포 시에는 정답(TAMA/SMI)을 모르므로 어떤 케이스가 FP인지 FN인지 사전에 알 수 없다.
따라서 2차 분류는 **1차에서 Positive로 예측된 그룹(TP+FP)**에만 적용되는 구조여야 한다.

- 이 그룹 내부에서 실제 TN이었던 케이스(=FP)를 Negative로 재분류하면 특이도가 오른다.
- 실제 TP를 잘못 Negative로 뒤집으면 민감도가 깎인다.
- 즉 2차 분류의 목표는 이 그룹의 PPV(=TP/(TP+FP))를 높이는 것과 동치이며, 이는 곧 전체
  코호트 기준 특이도 개선으로 이어진다.

## 2. FN 그룹은 2차 분류 대상에서 제외

내부 코호트 FN은 12명뿐이다. 표본이 너무 작아 이 그룹을 대상으로 한 보정 모델은 통계적으로
불안정하고, TN(515명) 중 일부를 잘못 Positive로 옮기면 특이도만 깎이는 비대칭 리스크가 크다.

Predicted-negative 그룹(TN+FN)은 손대지 않고 1차 결과를 그대로 유지한다. 이는
"Positive 예측군 내부에서 FP→TN 전환"에 집중하는 것이 acceptance criteria(민감도 -5%p 이내,
특이도는 반드시 개선 — `feedback_low_smi_noninferiority_criteria` 메모)를 지키면서 개선을
얻을 수 있는 가장 안전한 레버리지이기 때문이다.

## 3. AEC는 반드시 clinical 변수에 대한 residual로만 사용

Raw/정규화된 aec_128 곡선은 BMI/height/weight 정보를 재인코딩할 뿐이다
(`project_aec_curve_bmi_confound` 메모: Simpson's paradox로 인한 BMI confound).
`error_feature_analysis.md`에서도 FP vs TN 차이가 height(6.8cm 차), weight(3.8kg 차) 등
이미 1차 모델이 쓰는 변수에서 강하게 나타난다(p<0.0001) — raw aec를 그대로 넣으면 1차 모델과
같은 정보만 반복해서 보게 될 위험이 크다.

절차:
1. 표준화된 clinical 피처(age/height/weight/sex)에 대해 각 slice의 patient-normalized AEC 값을
   내부 코호트에서만 선형회귀로 fit한다.
2. 잔차(clinical로 설명되지 않는 AEC 정보)를 취한다.
3. 잔차 곡선을 저차원 요약 피처로 축소한다 — 방식은 여러 가지를 시도했다 (PCA, 등간격
   slice-band 평균, data-driven cluster-band 평균, 이 둘의 조합). 자세한 내용은
   "구현 파일별 차이" 절 참고.
4. 회귀 계수와 curve featurizer는 내부에서 고정(freeze)한 뒤 외부 코호트에 그대로 적용한다.

레거시 파이프라인의 R1-R4 고정 구간 방식은 재사용하지 않는다는 기존 방침
(`feedback_no_legacy_pipeline_reuse`)에 따라, 곡선 표현은 data-driven 방식으로 새로 설계한다 —
고정된 slice 구간의 slope/AUC 같은 locked-region 요약치는 쓰지 않는다. 참고로 raw `aec_cropped`
시트(리샘플링 전)도 curve feature 후보로 검토했으나, 환자마다 스캔 길이(`n_slices_cropped`
110~238)가 달라 slice 인덱스가 환자 간에 정렬돼 있지 않아 제외했다 — column-wise로 그대로
쓰면 scan-length confound를 오히려 재도입하게 된다.

## 4. Simpson's paradox 재확인이 필요한 지점

`16_aec_curve_bmi4_x_lowsmi_facet.png` 결과상 BMI 층에 따라 AEC-LowSMI 관계의 방향이
반대로 뒤집힌다. 2차 분류 대상군(1차 Positive 예측군)은 특정 BMI 구간에 쏠려 있을 가능성이
높으므로, "잔차 AEC가 이 서브그룹 안에서도 실제로 신호를 주는지"를 pooled 상관이 아니라
이 서브그룹 자체에서 재확인해야 한다. Pooled 데이터에서 유의미해 보이는 신호가 이 특정
서브그룹에서는 사라지거나 반대 방향일 수 있다.

## 5. 리키지 방지: 1차/2차 CV 폴드를 일관되게 유지

1차와 2차 모두 동일한 `StratifiedKFold(n_splits, shuffle, seed)` 분할을 재사용한다.
폴드 배정은 y와 seed에만 의존하고 피처(x)에는 의존하지 않으므로, 1차 학습에 쓰인 폴드와
2차 학습/평가에 쓰인 폴드가 항상 일치한다.

이렇게 하지 않으면 "누가 Positive 예측군에 속하는지"(1차 OOF 점수로 결정됨) 자체가
2차 모델의 학습/검증 폴드 구성에 정보 누출을 일으킬 수 있다 — 즉 2차 모델이 자신이
평가받을 폴드의 멤버십 결정에 간접적으로 영향을 준 데이터로 학습되는 상황을 방지한다.

## 임계값 선택 기준

2차 분류기의 임계값(th2)은 다음을 만족하는 후보 중 **screen-positive 그룹의 PPV를
최대화**하도록 선택한다:

- 민감도 **비열등성(Non-Inferiority) 검정을 통과**할 것 (`noninferiority_test_sensitivity`,
  Newcombe(1998) Method 10 — 페어드 비율 차이에 대한 Wilson score 기반 신뢰구간). 민감도
  하락폭의 97.5% 신뢰구간 상한이 margin(0.05) 이하여야 통과 — 단순히 "점추정 하락폭이 5%p
  이내"보다 엄격한 기준으로, 표본 크기에 따른 불확실성을 반영한다. (sentinel 후보 th2=-inf는
  항상 1차 결과를 그대로 재현하므로 하락폭 0, 탐색 공간에 항상 유효한 후보로 포함된다.)

최종 판정은 PPV가 아니라 **전체 코호트의 Sensitivity/Specificity delta**로 내린다
(PPV는 임계값 선택을 위한 내부 목적함수일 뿐). 두 스크립트 모두 임계값 선택(sweep 포함)과
최종 보고 판정에 동일한 비열등성 검정 함수를 공유하므로, 모델 선택에 쓰인 기준과 최종
PASS/FAIL 기준이 어긋나지 않는다. `bandfeat` 버전만, sweep 단계의 설정 선택에는 최종 보고보다
더 엄격한 내부 전용 margin(`SELECTION_MARGIN = 0.03`)을 추가로 사용한다 — band/cluster_band까지
포함한 훨씬 넓은 탐색 공간에서는 margin 경계에 딱 걸쳐 통과하는 설정이 나올 수 있어, 외부
코호트의 표본 노이즈에 대한 여유를 미리 확보하기 위함이다 (외부 결과는 여전히 전혀 보지 않는다).

## 구현 파일별 차이

같은 설계 원칙(1~5)을 공유하는 두 스크립트가 있다. 원칙은 동일하고 **잔차 곡선을 저차원
피처로 요약하는 방식(curve featurizer)만 다르다**:

| 구분 | `code/stage2_aec_residual_reclassify.py` | `code/stage2_aec_residual_reclassify_bandfeat.py` |
| --- | --- | --- |
| Curve feature | 전역 functional PCA만 (분산 90%/95% 목표, k=4~6) | PCA + `band`(등간격 4/8/16/32구간 평균) + `cluster_band`(data-driven 구간 평균) + `combo`(band+PCA 결합) |
| 구간 경계 | 해당 없음 (PCA는 전체 곡선을 혼합) | `band`는 등간격, `cluster_band`는 인접 slice를 내부 코호트 잔차 프로파일 기준 Ward 클러스터링(chain-graph connectivity로 인접성 강제)으로 그룹화 |
| 하이퍼파라미터 튜닝 | 없음 (각 모델 기본값 고정) | 있음 — 9종 curve_feat x {HGB: depth/lr/max_iter 그리드, logreg: C 그리드}를 sweep에 포함, 성능 동률 시 band/cluster_band를 우선하는 해석가능성 타이브레이크 적용 |
| 현재까지 최선 결과 | logreg + PCA(k≈4~6) + stage1 score: internal spec_delta=+4.3%, external +1.8% | HGB(`max_iter=500`으로 튜닝) + cluster_band(8) + stage1 score: internal spec_delta=+4.9%, external +2.5% (현재 최선) |
| 출력 위치 | `outputs/stage2_aec_residual_reclassify/` | `outputs/stage2_aec_residual_reclassify_bandfeat/` |

두 파일 모두 model-selection sweep(모델 종류 x curve feature x stage-1 score 포함 여부, `bandfeat`는
추가로 하이퍼파라미터 그리드)을 내부 OOF로만 수행하고, 외부 코호트는 최종 선택된 설정에 대해
단 한 번만 적용한다 (`feedback_new_scripts_under_code_root` 참고 — 새 스크립트는 `code/` 바로
아래에 위치).

두 스크립트 모두 임계값 선택/최종 판정에 점추정치 대신 CI 기반 비열등성 검정
(`noninferiority_test_sensitivity`, 위 "임계값 선택 기준" 절)을 쓰도록 바뀌면서, 이전 버전
(점추정치 기준, 위 표의 구버전 수치: PCA +8.6%/+5.4%, bandfeat +10.9%/+8.0%) 대비 채택되는
spec_delta 절대값이 전반적으로 낮아졌다 — 더 엄격한 기준으로 임계값을 고르기 때문에 나타나는
자연스러운 결과이며, `bandfeat`가 PCA 단독보다 우수하다는 상대적 순위 자체는 유지된다.

`bandfeat` 버전이 일관되게 더 나은 이유로 추정되는 것: 잔차 신호가 곡선 전체에 고르게 퍼진
global mode가 아니라 특정 구간(예: tail 구간)에 국소적으로 존재해서, PCA 성분에 섞여
희석되는 것보다 구간별 평균이 이 국소 신호를 더 잘 보존하기 때문으로 보인다. `cluster_band`가
`band`보다 나은 것은 등간격 대신 실제 신호가 바뀌는 지점에서 구간을 나누기 때문으로 추정된다.

## 참고 메모

- `feedback_low_smi_noninferiority_criteria` — acceptance criteria (민감도 -5%p, 특이도 gain)
- `feedback_no_legacy_pipeline_reuse` — aec_128 featurization 재설계 방침
- `project_aec_curve_bmi_confound` — AEC-BMI confound (Simpson's paradox) 근거
- `feedback_unified_baseline_model` — clinical-only 베이스라인 재사용 규칙
- `project_stage2_reclassify_results` — 현재까지의 실행 결과 및 한계
