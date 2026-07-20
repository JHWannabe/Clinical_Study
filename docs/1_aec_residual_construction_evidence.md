# AEC Residual 구성(Step 2) 검증 근거

`docs/0_aec_bmi_confound_evidence.md`가 "clinical 변수(age/height/weight/sex)가 AEC-128
곡선과 Low-SMI 사이의 confounder다"라는 전제를 실측 데이터로 입증했다면, 이 문서는 그
다음 질문 — `1_aec_residual_reclassify.py` Step 2(`fit_aec_residualizer` /
`apply_aec_residualizer`: 128-slice 정규화 곡선을 표준화된 4변수에 대해 `LinearRegression`으로
회귀하고 실제값-예측값의 잔차만 남기는 절차)가 **실제로 confound를 제거하는지, 얼마나
제거하는지, 그리고 통계적으로 타당하게 적합됐는지**를 검증한다.

산출 스크립트: `code/1_3_aec_residual_construction_evidence.py`
(`code/1_aec_residual_reclassify.py`의 `fit_aec_residualizer`/`apply_aec_residualizer`,
`code/baseline/aec_curve_comparison.py`의 `curve_diff_test`/`plot_curve_comparison`을 그대로
재사용 — 파이프라인 수학을 다시 구현하지 않음). 산출물은
`outputs/1_3_aec_residual_construction_evidence/`.

> Low-SMI 라벨은 `SMI = TAMA / (Height[m])²`(M<45.4, F<34.4 cutoff, internal n=129) —
> `1_aec_residual_reclassify.py`가 실제로 쓰는 정의 — 하나로 통일해 계산했다
> (`docs/0_aec_bmi_confound_evidence.md` 상단 참고).

## 검증 구성

- **Part A. 잔차화 전/후 confound 제거 확인** — doc0의 조건-1(clinical→AEC 연관)과
  조건-3 보강(BMI 4구간 stratification) 검정을 원본 정규화 곡선이 아니라 **잔차 곡선**에
  다시 적용해, 잔차화가 실제로 그 연관을 없애거나 줄이는지 확인. internal(잔차화 회귀가
  적합된 코호트, in-sample)과 external(적합에 전혀 관여하지 않은 frozen 적용,
  out-of-sample) 양쪽에서 수행 — external 결과가 진짜 일반화 검증이다.
- **Part B. 회귀 자체의 통계적 타당성** — slice별 R²(internal in-sample vs external
  out-of-sample), 잔차 분포 진단(정규성/등분산성), 그리고 doc0 1절이 남긴 caveat("BMI의
  비선형 confound가 선형 height·weight 회귀만으로 완전히 제거된다는 보장은 없다")을
  직접 재검정하는 BMI-이차항 재적합.
- **최종 판정**은 Part A/B의 진단 결과를, `1_aec_residual_reclassify.py`가 실제로
  산출한 Stage-1+Stage-2 성능(`stage1_vs_stage2_summary.csv`)과 연결해 "전처리가
  주장대로 작동하는가"와 "그래서 실제로 쓸모가 있는가"를 함께 판정한다.

## Part A1. 조건-1 재검정: 잔차화 전/후 clinical-그룹 간 곡선 차이

`aec_curve_comparison.curve_diff_test()`를 원본 정규화 곡선(before)과 잔차 곡선(after)
양쪽에 그대로 적용 — 그룹 분할 기준(성별/나이·키·체중 median split/BMI 성별 median
split)은 doc0 1-1절과 동일. (`A1_condition1_before_after_summary.csv`,
`A1_*_before_after.png`)

| cohort | 변수 | RMSD (전) | p (전) | RMSD (후) | p (후) | RMSD 감소율 | 후에도 유의(p<0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| internal | 성별 | 0.1178 | 0.0005 | ~0 (1.0e-16) | 1.000 | 100.0% | 아니오 |
| internal | 나이 | 0.0816 | 0.0005 | 0.0067 | 0.220 | 91.7% | 아니오 |
| internal | 신장 | 0.0515 | 0.0005 | 0.0033 | 0.832 | 93.6% | 아니오 |
| internal | 체중 | 0.0911 | 0.0005 | 0.0061 | 0.291 | 93.2% | 아니오 |
| internal | BMI | 0.0751 | 0.0005 | 0.0065 | 0.237 | 91.3% | 아니오 |
| external | 성별 | 0.0855 | 0.0005 | 0.0265 | **0.0005** | 69.0% | **예** |
| external | 나이 | 0.0692 | 0.0005 | 0.0086 | 0.281 | 87.6% | 아니오 |
| external | 신장 | 0.0357 | 0.0005 | 0.0278 | **0.0005** | 22.0% | **예** |
| external | 체중 | 0.0607 | 0.0005 | 0.0308 | **0.0005** | 49.2% | **예** |
| external | BMI | 0.0572 | 0.0005 | 0.0171 | **0.0095** | 70.1% | **예** |

### 해석

- **Internal(in-sample)에서는 5개 변수 전부 잔차화 후 유의성이 사라진다.** 특히
  성별은 RMSD가 사실상 정확히 0(1e-16)이 되는데, 이는 우연이 아니라 **수학적으로
  보장된 결과**다 — `LinearRegression`은 절편을 포함해 적합하므로 정규방정식
  `Xᵀresid=0`이 절편 열(전체 잔차 합=0)과 성별 열(이진 변수) 모두에 대해 성립하고,
  둘을 합치면 "남성 그룹의 잔차 합=0, 여성 그룹의 잔차 합=0"이 대수적으로 강제된다.
  나이/신장/체중은 median-split이라는 범주형 요약이라 이런 대수적 보장은 없지만,
  RMSD가 91~94% 줄어들고 p가 전부 0.05를 넘는 것으로 보아 연속형 회귀 자체는
  median-split 기준의 그룹 차이도 사실상 제거한다.
- **하지만 이 in-sample 결과만으로는 residualizer가 "일반화"됐다고 말할 수 없다** —
  회귀가 바로 이 데이터에 적합됐으니 당연한 결과이기 때문이다. 진짜 검증은
  **external(잔차화 회귀가 전혀 보지 못한 코호트에 frozen 적용)**: 나이만 확실히
  비유의(p=0.281)로 떨어지고, **성별·신장·체중·BMI 네 변수는 RMSD가 22~70% 줄어들되
  여전히 p<0.01로 유의하게 남는다.** 즉 잔차화는 confound를 "완전히 제거"하는 게
  아니라 **상당 부분(22~70%) 줄이는 데 그친다** — 특히 신장은 external에서 RMSD
  감소율이 22%에 불과해 가장 덜 제거된다.
- 이 결과는 `1_1_aec_input_variant_comparison.py`(`residual_reclassify_algorithm.md`
  8.1절)가 이미 보고한 "BMI 잔차화가 patient-normalized보다 나은 방향이지만 폭이 크지
  않다"는 관찰과 정확히 일치하는 메커니즘적 설명이다 — external 잔차 곡선에 여전히
  clinical-연관 신호가 남아있으니, Stage-2가 얻는 이득이 크지 않은 것은 놀랍지 않다.

## Part A2. 조건-3 보강 재검정: 잔차화 전/후 BMI 4구간 내 Low-SMI 비교

doc0 3-1절의 BMI 4구간 stratified 검정(WHO 아시아 기준)을 잔차 곡선에 적용.
(`A2_bmi4_simpsons_paradox_before_after_summary.csv`, `A2_bmi4_x_lowsmi_*.png`)

| cohort | stage | BMI 구간 | n(Low/Non-low) | RMSD | 방향 | p |
| --- | --- | --- | --- | --- | --- | --- |
| internal | 전 | Underweight | 29/28 | 0.049 | Low<Non-low | 0.117 (n.s.) |
| internal | 전 | Normal | 69/360 | 0.039 | Low>Non-low | **0.0035** |
| internal | 전 | Overweight | 17/263 | 0.065 | Low>Non-low | **0.0065** |
| internal | 전 | Obese | 14/310 | 0.032 | Low<Non-low | 0.246 (n.s.) |
| internal | 후 | Underweight | 29/28 | 0.022 | Low>Non-low | 0.642 (n.s.) |
| internal | 후 | Normal | 69/360 | 0.014 | Low>Non-low | 0.241 (n.s.) |
| internal | 후 | Overweight | 17/263 | 0.027 | Low>Non-low | 0.249 (n.s.) |
| internal | 후 | Obese | 14/310 | 0.030 | Low<Non-low | 0.169 (n.s.) |
| external | 전 | Underweight | 23/39 | 0.053 | Low<Non-low | 0.127 (n.s.) |
| external | 전 | Normal | 84/295 | 0.043 | Low<Non-low | **0.0010** |
| external | 전 | Overweight | 24/203 | 0.064 | Low<Non-low | **0.0025** |
| external | 전 | Obese | 10/248 | 0.092 | Low>Non-low | **0.0145** |
| external | 후 | Underweight | 23/39 | 0.041 | Low<Non-low | 0.157 (n.s.) |
| external | 후 | Normal | 84/295 | 0.010 | Low<Non-low | 0.679 (n.s.) |
| external | 후 | Overweight | 24/203 | 0.037 | Low<Non-low | 0.090 (n.s.) |
| external | 후 | Obese | 10/248 | 0.065 | Low>Non-low | 0.071 (n.s.) |

### 해석

- **Internal**: 잔차화 전 Normal/Overweight 두 구간에서 유의했던(p=0.0035, 0.0065)
  Low-SMI vs Non-low-SMI 곡선 차이가, 잔차화 후에는 4개 구간 전부 비유의(p≥0.17)로
  바뀐다 — Part A1의 "성별 잔차는 정확히 0" 같은 대수적 보장은 없는 대상(BMI4×LowSMI
  교차는 애초에 회귀 입력 변수가 아님)인데도 실제로 신호가 사라진다는 점에서, 조건-1
  재검정보다 오히려 더 실질적인 증거다.
- **External(out-of-sample)**: 잔차화 전 유의했던 세 구간(Normal p=0.0010, Overweight
  p=0.0025, Obese p=0.0145)이 잔차화 후 전부 비유의(p=0.679, 0.090, 0.071)로 떨어진다
  — p가 0.05를 정확히 밑돌지 못한 Overweight/Obese도 원래 p 대비 6~36배 상승했다.
  즉 **Part A1의 원시 조건-1 재검정(성별/신장/체중이 external에서 여전히 유의)과
  달리, BMI 4구간 안에서의 Low-SMI 신호 자체는 external에서도 잔차화 후 실질적으로
  사라진다** — Stage-2가 실제로 방어하려는 대상(BMI 계층에 따라 Low-SMI 판정이
  갈리는 것)에 대해서는 잔차화가 out-of-sample에서도 작동한다는 뜻이다. Part A1에서
  남은 신호(성별/신장/체중과 곡선의 잔여 연관)는 Low-SMI 판별과 직접 얽힌 것이라기보다
  코호트 간 촬영 조건 차이(doc0 1-2절의 vendor/scan-length covariate) 같은 다른
  경로일 가능성이 있다 — 이 문서 범위 밖의 후속 확인이 필요하다.

## Part B1. Slice별 R² — 회귀가 실제로 설명력을 갖는지

`reg.predict()`와 실제 곡선값 사이 slice별 R²를 internal(in-sample)·external
(out-of-sample, frozen)에서 각각 계산. (`B1_slice_r2_internal_vs_external.csv/png`)

- **Internal**: mean R²=0.396, max=0.515 (128 slice 전체 양수)
- **External**: mean R²=0.165, max=0.284, **음수 slice 8/128**

### 해석

Internal 평균 R²=0.40은 "임상 4변수가 slice 값 변동의 약 40%를 설명한다"는 뜻으로,
doc0 조건-1의 유의성(p<0.001)이 통계적 유의성을 넘어 실질적 설명력으로도 뒷받침됨을
보여준다. External에서 mean R²=0.165로 줄어드는 것은 예상된 일반화 격차이지만
**0보다는 뚜렷이 크다** — external 128개 slice 중 120개(94%)에서 여전히 양의 R²를
보여, internal에서 학습한 clinical→곡선 관계가 노이즈 적합이 아니라 실제로 다른
코호트에도 어느 정도 이전되는 신호임을 뒷받침한다. 다만 8개 slice(6%)에서는 R²<0
(코호트 평균으로 예측하는 것보다도 못한 성능)로, 일부 slice는 코호트 간 이질성이
회귀로 포착한 관계보다 크다는 것도 함께 보여준다.

## Part B2. 잔차 분포 진단 (정규성·등분산성)

Internal 잔차(1090명×128 slice = 139,520개)를 풀링해 진단.
(`B2_residual_distribution_diagnostics.csv/png`)

- **정규성**: skewness=0.118(거의 대칭), kurtosis(초과)=1.403(약간 두꺼운 꼬리) —
  D'Agostino-Pearson 검정 p≈0(표본이 워낙 커서(n=139,520) 미세한 비정규성도 유의로
  잡힘, 검정력 과잉으로 해석해야 함). skew/kurtosis 절댓값 자체는 크지 않아 OLS
  추정치·잔차 해석에 실무적으로 문제될 수준은 아니다.
- **등분산성**: |잔차|와 예측값의 Spearman ρ=-0.024, p=0.0007 — 역시 대표본 때문에
  통계적으로는 유의하지만 **효과크기(ρ=-0.024)가 사실상 0**이라 이분산성이
  실질적으로 문제되는 수준은 아니다.

### 해석

두 진단 모두 "대표본 때문에 형식적으로는 유의하지만 효과크기는 무시할 만한" 패턴 —
`LinearRegression`(OLS)의 등분산·정규성 가정이 실무적으로 크게 위배되지 않는다는
뜻이며, robust/weighted 회귀 등으로 바꿀 근거는 이 데이터에서 발견되지 않았다.

## Part B3. BMI 이차항 재검정 — 선형 잔차화가 남긴 비선형 confound

doc0 1절의 caveat("residualizer는 height·weight를 선형으로만 회귀시키므로 BMI의
비선형 confound가 완전히 제거된다는 보장은 없다")을 직접 검정. Internal 잔차 곡선을
표준화 BMI + BMI²에 다시 회귀(internal fit, external frozen 적용).
(`B3_bmi_quadratic_residual_recheck.csv/png`)

- **Internal**: mean R²=0.0024, max R²=0.0123 (128 slice 전부 3% 미만)
- **External**: mean R²=-0.0393, max R²=0.0016 (대부분 음수)

### 해석

Internal 기준으로도 BMI+BMI²가 잔차 곡선 변동의 최대 1.2%밖에 설명하지 못하고,
external에서는 평균 R²가 음수(코호트 평균 예측보다도 못함) — **선형 height·weight
회귀만으로 BMI의 비선형 confound가 실질적으로 충분히 제거된다**는 경험적 근거다.
doc0 1절의 caveat은 이론적으로는 여전히 유효하지만(선형 회귀가 비선형 항의 직교성을
대수적으로 보장하지 않으므로), 이 데이터에서 그 잔여분의 크기는 무시할 만한
수준이라는 것이 Part B3의 결론이다.

## 종합

| 검증 | 대상 | 결과 |
| --- | --- | --- |
| A1. 조건-1 재검정 | 성별/나이/신장/체중/BMI 그룹과 잔차 곡선 | internal 전부 비유의; external은 나이만 비유의, 성별·신장·체중·BMI는 RMSD 22~70% 감소했지만 여전히 유의 |
| A2. 조건-3 보강 재검정 | BMI4구간 내 Low-SMI vs Non-low-SMI 잔차 곡선 | internal·external 모두 전 구간 비유의로 전환(잔차화 전 유의했던 구간 포함) |
| B1. slice별 R² | 잔차화 회귀의 설명력 | internal mean 0.396, external mean 0.165 (120/128 slice 양수) — 노이즈 적합이 아닌 실제 전이 가능한 신호 |
| B2. 잔차 분포 진단 | OLS 가정(정규성/등분산성) | 효과크기 기준으로 실무적 위반 없음 |
| B3. BMI 이차항 재검정 | 선형 잔차화가 남긴 비선형 BMI confound | internal 최대 R²=0.012, external은 음수 — 무시할 수준 |

**결론**: `1_aec_residual_reclassify.py` Step 2의 잔차화는 (a) 원래 목적한 BMI/clinical
confound를 — 적어도 Low-SMI 판별과 직접 얽힌 BMI4-stratified 신호(A2) 기준으로는 —
in-sample·out-of-sample 모두에서 실질적으로 제거하고, (b) 회귀 자체도 통계적으로
타당하게 적합됐으며(B1-B3), (c) 선형-전용 설계라는 이론적 한계(doc0 caveat)가 실제
데이터에서 문제를 일으키지 않음을 확인했다. 다만 (d) Part A1이 보여주듯 원시
clinical-그룹 대 곡선 연관은 external에서 22~70%만 줄어들고 완전히 사라지지는
않는다 — residualizer가 "confound를 없앤다"기보다 "상당 부분 줄인다"고 보는 것이
더 정확한 서술이며, 이는 `1_1_aec_input_variant_comparison.py`가 관찰한 Stage-2의
제한적인 개선 폭(8.1절)과 일관된다.

## 최종 판정: 사용 가능한 전처리인가?

**예, 조건부로 사용 가능하다.** 근거를 두 갈래로 나눠서 본다.

1. **전처리 자체가 주장대로 작동하는가 (Part A/B)** — Yes. Step 2가 실제로 방어하려는
   대상(BMI 계층에 따라 Low-SMI 판정이 갈리는 신호, A2)은 internal·external 모두에서
   사라지고, 회귀는 노이즈 적합이 아니라 실전이 가능한 설명력을 갖는다(B1: external
   mean R²=0.165, 120/128 slice 양수). doc0이 남긴 "선형 회귀가 BMI의 비선형 confound를
   다 못 없앨 수 있다"는 caveat도 실측으로는 무시할 수준(B3)임을 확인했다.
2. **그 결과 최종 Stage-1+Stage-2 모델이 실제로 이득을 보는가**
   (`outputs/1_aec_residual_reclassify/stage1_vs_stage2_summary.csv`) — Yes, 하지만
   폭은 작다. Non-inferiority 검정 기준 민감도 손실은 두 코호트 모두 PASS
   (internal 97.5% CI 상한 0.037, external 0.035, margin 0.05 이내)이고, 특이도는
   internal +4.3%p(53.5%→57.8%), external +1.5%p(49.3%→50.8%) 상승한다
   (McNemar spec p=4.2e-10/4.9e-4, 둘 다 유의).

**조건**: (i) 특이도 개선 폭이 external에서 1.5%p로 크지 않다는 점, (ii) Part A1이
보여주듯 잔차 곡선에 clinical-연관 신호가 완전히는 안 빠진다는 점(성별/신장/체중/BMI가
external에서 여전히 유의)을 감안하면, 이 전처리는 "Low-SMI 오분류를 줄이는 통계적으로
유의한 전처리"로는 쓸 수 있지만 "clinical 변수를 대체할 만큼 강력한 독립 신호원"으로
과대해석해서는 안 된다. 참고로 Stage-2가 실제로 학습·평가되는 screen-positive
부분집합(internal n=564, external n=529)은 BMI 4구간 전부 n≥41로 표본 부족 문제는
없다 — 개선 폭이 작은 것은 표본 크기가 아니라 신호 자체의 세기 문제로 봐야 한다.

## 참고 메모

- `0_aec_bmi_confound_evidence` — 이 문서가 이어받는 confound 존재 증명 (이 문서 작업
  중 발견한 SMI 라벨 버그로 함께 정정됨)
- `model_algorithm` — Step 2 residualization 설계 원칙 문서
- `residual_reclassify_algorithm` — 8.1절의 patient-normalized vs BMI-residualized
  비교가 Part A1의 external 결과와 같은 방향의 관찰
- `aec_residual_related_papers` — residualization 방법론 자체의 외부 문헌 근거
