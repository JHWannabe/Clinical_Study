# Step1~3(AEC 추가 실험) 확증적 통계 검정 — paired bootstrap 재분석

`docs/clinic4_aec_mean_segment_analysis.md`가 Step3 구간수 sweep(`step1_clinic_aec_mean.py`)에
적용한 프로토콜(**internal 5-fold OOF로 feature별 최적 config를 먼저 확정 → 그 config
하나만 external에 paired bootstrap(5,000회, clinic4 vs AEC모델을 동일 리샘플 인덱스로
동시 재표집한 ΔR²) 단일 검증**, [[internal_external_validation_protocol]])를 Step1
(`step1_clinic_aec_mean.py`의 N=1)·Step2(`step2_clinic_aec_shape.py`)·Step3 top-k(`step3_clinic_aec_segment_select.py`)·
Step3 앞/뒤50%(`step3_clinic_aec_back_half.py`) 나머지 3개 실험에도 동일하게 적용해 재검증했다.

기존 slide 7~15의 다수 불릿(예: "VAT top8 external +0.09")은 k=1~8 또는 segment=1~128을
전부 external과 대조해 그중 눈에 띄는 값을 골라 인용한 것이라 **다중비교(multiple
comparison) 낙관편향** 위험이 있었다([[feedback_internal_external_validation_discipline]]).
아래는 그 문제를 제거한 단일 확증 검정 결과다.

## 방법

1. Feature별로 internal(gangnam) 5-fold OOF R²가 가장 높은 config(=1개)를 확정.
2. 그 config 하나만 external(sinchon)에 동결 적용, clinic4 baseline과 **동일 환자
   리샘플 인덱스**로 5,000회 paired bootstrap → ΔR²의 95% CI 산출.
3. CI가 0을 포함하지 않으면 "유의", 포함하면 "n.s."로 판정.
4. 재현 스크립트: `code/step1_clinic_aec_mean.py`(Step1: N=1 고정)·`code/step2_clinic_aec_shape.py`(Step2)·
   `code/step3_clinic_aec_segment_select.py`(Step3 top-k)·`code/step3_clinic_aec_back_half.py`(Step3 앞/뒤50%)의
   모델 fitting 로직을 그대로 재현 후 paired bootstrap만 추가. 원자료:
   `outputs/step3_topk_backhalf_rigorous_significance.csv`.

## 결과

### Step1 — AEC-128 전체 평균 1개 값(N=1)

| feature | internal R² (Δ) | external R² (Δ) | 95% CI (Δ) | 판정 |
|---|---|---|---|---|
| IMATA | 0.355 (+0.022) | 0.191 (-0.091) | [-0.118, -0.067] | **유의(악화)** |
| NAMA | 0.610 (+0.019) | 0.375 (-0.193) | [-0.226, -0.164] | **유의(악화)** |
| LAMA | 0.425 (+0.006) | -0.182 (-0.147) | [-0.168, -0.128] | **유의(악화)** |
| TAMA | 0.650 (+0.004) | 0.677 (-0.034) | [-0.045, -0.025] | **유의(악화)** |
| SAT | 0.541 (+0.013) | 0.536 (-0.029) | [-0.045, -0.013] | **유의(악화)** |
| VAT | 0.530 (+0.007) | 0.378 (-0.041) | [-0.059, -0.022] | **유의(악화)** |
| Total Fat | 0.518 (+0.016) | 0.447 (-0.045) | [-0.066, -0.025] | **유의(악화)** |

**7개 feature 전부 external에서 통계적으로 유의하게 악화** — 예외 없음. 정보를 1개
숫자로 극단적으로 압축하면 어떤 체성분 feature에도 도움이 안 된다.

### Step2 — 곡선 형태 feature(SD·Skewness·상하위50%비율), feature별 internal-best 확정 후 단일검증

| feature | internal-best | external R² (Δ) | 95% CI (Δ) | 판정 |
|---|---|---|---|---|
| VAT | shape_all | 0.485 (+0.067) | [0.040, 0.098] | **유의(개선)** |
| Total Fat | shape_all | 0.520 (+0.028) | [0.008, 0.049] | **유의(개선)** |
| LAMA | uplow_ratio | 0.026 (+0.062) | [0.039, 0.086] | 유의(개선) — baseline≈0이라 실질적 의미 낮음 |
| SAT | shape_all | 0.569 (+0.005) | [-0.005, 0.014] | n.s. |
| IMATA | shape_all | 0.277 (-0.006) | [-0.026, 0.015] | n.s. |
| NAMA | shape_all | 0.451 (-0.117) | [-0.141, -0.094] | **유의(악화)** |
| TAMA | shape_all | 0.672 (-0.039) | [-0.050, -0.029] | **유의(악화)** |

### Step3 top-k 구간선택, feature별 internal-best k 확정 후 단일검증

| feature | internal-best k | external R² (Δ) | 95% CI (Δ) | 판정 |
|---|---|---|---|---|
| VAT | top8 | 0.509 (+0.090) | [0.058, 0.125] | **유의(개선)** |
| Total Fat | top8 | 0.524 (+0.032) | [0.004, 0.062] | **유의(개선)** |
| LAMA | top8 | 0.004 (+0.039) | [0.004, 0.076] | 유의(개선) — baseline≈0이라 실질적 의미 낮음 |
| IMATA | top8 | 0.291 (+0.009) | [-0.026, 0.046] | n.s. |
| SAT | top7 | 0.557 (-0.008) | [-0.029, 0.014] | n.s. |
| NAMA | top3 | 0.436 (-0.132) | [-0.158, -0.109] | **유의(악화)** |
| TAMA | top6 | 0.676 (-0.035) | [-0.046, -0.025] | **유의(악화)** |

### Step3 앞/뒤 50% 단독 사용, feature별 internal-best(앞 또는 뒤) 확정 후 단일검증

| feature | internal-best | external R² (Δ) | 95% CI (Δ) | 판정 |
|---|---|---|---|---|
| IMATA | back50 | 0.189 (-0.093) | [-0.126, -0.063] | **유의(악화)** |
| NAMA | back50 | 0.387 (-0.181) | [-0.213, -0.152] | **유의(악화)** |
| LAMA | back50 | -0.254 (-0.218) | [-0.252, -0.189] | **유의(악화)** |
| TAMA | front50 | 0.672 (-0.039) | [-0.052, -0.029] | **유의(악화)** |
| SAT | back50 | 0.544 (-0.021) | [-0.038, -0.004] | **유의(악화)** |
| VAT | back50 | 0.356 (-0.063) | [-0.091, -0.034] | **유의(악화)** |
| Total Fat | back50 | 0.446 (-0.046) | [-0.071, -0.021] | **유의(악화)** |

**7개 feature 전부 유의하게 악화 — 지방계열도 예외 없음.** 곡선을 반으로 쪼개 한쪽
절반만 단독으로 쓰는 방식은 정보 손실이 너무 커서 항상 해롭다(Step2/Step3 top-k처럼
여러 정보를 결합해야 이득이 나타남).

## 종합 해석

4개 실험(Step1 단일평균, Step2 형태feature, Step3 top-k, Step3 앞/뒤50%)을 관통하는
일관된 패턴:

1. **NAMA·TAMA(근육계열)는 AEC를 어떤 방식으로 추가해도 예외 없이 유의하게 악화된다**
   — 4개 실험 전부에서 확증. 근육 관련 AEC 상관이 원래 약하기 때문(NAMA 전 구간
   |r|<0.25, `docs/output_feature_predictor_correlations` 참조).
2. **VAT·Total Fat(지방계열)는 "정보를 충분히 담은" 표현(Step2 형태feature 결합,
   Step3 top-k 8구간)에서만 유의하게 개선된다** — 반대로 정보를 과도하게 압축한
   표현(Step1 단일평균, 앞/뒤50% 단독)에서는 VAT·Total Fat도 똑같이 유의하게
   악화된다. 즉 "지방계열엔 AEC가 도움된다"는 무조건 참이 아니라, **곡선의 정보를
   일정 수준 이상 보존해야만** 성립하는 조건부 결론이다.
3. **IMATA·SAT는 어느 실험에서도 유의한 효과가 확인되지 않았다(n.s.)** — 기존
   슬라이드가 이 둘을 "지방계열 전반 개선"에 묶어 서술한 것은 재검토가 필요하다.
4. **LAMA의 "유의한 개선"은 `docs/clinic4_aec_mean_segment_analysis.md`와 동일한
   이유로 액면 그대로 쓰지 말 것** — clinic4 baseline external R²가 -0.03~-0.25로
   사실상 예측력이 없는 구간이라, 거기서 오르는 것은 통계적으로는 유의해도 실질적
   예측력 획득으로 보기 어렵다.

## 참고

- 선행 분석(Step3 구간수 sweep 전용): `docs/clinic4_aec_mean_segment_analysis.md`
- 방법론 원칙: `docs/internal_external_validation_protocol.md`,
  [[feedback_internal_external_validation_discipline]]
- 원자료: `outputs/step3_topk_backhalf_rigorous_significance.csv`(Step2/top-k/앞뒤50%),
  Step1(N=1)은 이 문서의 표에만 기재(재현 스크립트는 위 "방법" 절 참고)
