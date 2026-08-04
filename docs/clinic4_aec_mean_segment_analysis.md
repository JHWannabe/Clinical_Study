# clinic4 + AEC 구간평균 선형회귀: internal 선택 → external 단일검증 결과

`code/step1_clinic_aec_mean.py` 출력(`outputs/clinic_aec_mean_linear_regression_summary.csv`,
`outputs/clinic4_vs_clinic4_aec_mean_r2_comparison.csv`)에 대한 분석.
[[internal_external_validation_protocol]] 원칙에 따라 재구성함 —
**segment_count(1/2/4/8/16/32/64/128)는 internal(gangnam) 5-fold OOF R²만으로
feature별 최적값을 먼저 확정**하고, 그 확정 모델에 대해서만 external(sinchon)을
**단 한 번** paired bootstrap으로 검증했다. (이전 버전은 8개 segment_count 전부를
external과 대조해 "external이 제일 좋은 것"을 골랐는데, 이는 external을 튜닝에
쓴 것이라 원칙 위반 — 아래 결과는 이를 수정한 재분석이다.)

## 절차

1. Feature별로 `pivot_internal[feature, mean1..128]`에서 `argmax`로 internal 최적
   segment_count 모델을 선택.
2. 그 모델의 external R²를, 같은 환자 index로 clinic4와 페어링한 **paired
   bootstrap(5,000회, 동일 리샘플 인덱스로 clinic4 vs 선택모델 R² 차이)**로
   95% CI 및 2-sided bootstrap p-value 산출. CI가 0을 포함하지 않아야 "유의"로 판정.
3. Internal에서 이겼다고 무조건 external 개선을 주장하지 않고, 이 단일 검증
   결과만을 최종 판단 근거로 삼는다.

## 결과 (`outputs/clinic4_aec_mean_internal_selected_external_check.csv`)

| feature | internal 선택 모델 | clinic4 internal R² | 선택모델 internal R² | clinic4 external R² | 선택모델 external R² | Δexternal R² | 95% CI | 판정 |
|---|---|---|---|---|---|---|---|---|
| VAT(내장지방) | mean16 | 0.5230 | 0.6232 | 0.4183 | 0.4800 | **+0.0617** | [0.024, 0.101] | **유의(개선)** |
| LAMA | mean16 | 0.4193 | 0.4728 | -0.0354 | 0.0020 | +0.0374 | [0.002, 0.075] | 유의(개선) — baseline이 사실상 무신호(R²≈0)라 실질적 의미 낮음 |
| NAMA | mean2 | 0.5912 | 0.6116 | 0.5678 | 0.3903 | **-0.1775** | [-0.211, -0.149] | **유의(악화)** |
| TAMA | mean4 | 0.6460 | 0.6521 | 0.7110 | 0.6798 | **-0.0312** | [-0.043, -0.020] | **유의(악화)** |
| IMATA | mean8 | 0.3329 | 0.3897 | 0.2825 | 0.2913 | +0.0088 | [-0.027, 0.045] | n.s. |
| SAT(피하지방) | mean32 | 0.5276 | 0.5820 | 0.5646 | 0.5600 | -0.0046 | [-0.031, 0.021] | n.s. |
| Total Fat | mean32 | 0.5020 | 0.5895 | 0.4923 | 0.5059 | +0.0136 | [-0.021, 0.050] | n.s. |

## 해석

- **VAT(내장지방)만 진짜로 검증된 개선.** internal 최적 모델(mean16)이 external에서도
  clinic4 대비 유의하게 상회(+0.062, p=0.0004). 7개 feature 중 유일하게 이 절차
  전체를 통과.
- **NAMA·TAMA는 유의하게 악화** — internal에서는 AEC 추가로 소폭 개선되어 보였지만
  (NAMA +0.020, TAMA +0.006) external에서는 각각 -0.178, -0.031로 크게/뚜렷하게
  하락. Internal 개선이 external로 전혀 전이되지 않는 전형적 과적합 패턴.
- **IMATA, SAT, Total Fat은 "유의한 차이 없음"** — 이전(비원칙적) 분석에서 mean8
  근처 특정 segment_count만 짚어 "지방계열은 대체로 개선"이라 정리했던 것과
  달리, feature별 internal-best 모델을 그대로 external에 적용하면 이 세 feature는
  **개선도 악화도 통계적으로 확인되지 않는다.** 즉 AEC 추가가 이 3개 feature엔
  득도 실도 없다는 것이 이번 절차의 결론이며, 이전 결론(Total Fat이 mean8에서
  유의하게 개선)은 여러 segment_count를 external과 비교해 고른 결과였으므로
  **폐기**한다.
- **LAMA의 "유의"는 액면 그대로 쓰지 말 것** — clinic4 baseline external R²가
  -0.0354(사실상 예측력 없음)라서, 거기서 +0.037 오르는 것은 통계적으로는
  유의해도 실질적 예측력 획득은 아님.

## 결론

AEC-128 구간평균을 clinic4 회귀에 추가해 external까지 견고하게 이득을 보는
feature는 **VAT(내장지방) 하나뿐**이며, segment_count는 internal 기준
**mean16**이 최적이다. NAMA·TAMA(근육계열)는 AEC 추가를 권하지 않는다. IMATA,
SAT, Total Fat은 근거 불충분(n.s.)이므로 추가 여부를 결정할 근거가 아직 없다 —
표본을 늘리거나 다른 AEC 표현(예: [[project_aec_mean_regression_generalization_gap]]의
generalization gap 분석)으로 재확인이 필요하다.

## 참고

- 원본 데이터: `outputs/clinic_aec_mean_linear_regression_summary.csv`
- 이 분석 산출물: `outputs/clinic4_aec_mean_internal_selected_external_check.csv`,
  `outputs/clinic4_aec_mean_external_paired_bootstrap.csv`(전체 segment_count별 참고용 원자료)
- 방법론 원칙: `docs/internal_external_validation_protocol.md`,
  [[feedback_internal_external_validation_discipline]]
