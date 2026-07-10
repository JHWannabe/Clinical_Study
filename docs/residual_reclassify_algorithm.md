# Low-SMI 선별 모델 알고리즘 정리

## 1. 개요

본 파이프라인은 저골격근량(Low-SMI, Skeletal Muscle Index) 환자를 선별하기 위한 **2단계(Stage-1 → Stage-2) 분류 모델**이다.

- **Stage 1** (`code/baseline/clinic-only_baseline.py`): 임상 변수만으로 민감도(Sensitivity) ≥ 90%를 만족하는 1차 선별(screening) 모델.
- **Stage 2** (`code/stage2_aec_residual_reclassify.py`): Stage 1에서 "양성(screen positive)"으로 분류된 환자군만을 대상으로, AEC-128 곡선에서 임상 변수로 설명되지 않는 잔차(residual) 정보를 추가해 특이도(Specificity)를 끌어올리는 재분류 모델.

두 단계 모두 내부 코호트(`g1090.xlsx`, internal)에서 5-fold 교차검증(CV)으로 학습·임계값을 결정하고, 외부 코호트(`sdata.xlsx`, external)에는 내부에서 고정(freeze)된 파라미터만 적용하는 **완전 분리된 검증 구조**를 따른다.

---

## 2. 정답 라벨(Ground Truth) 정의

```
SMI = TAMA / (Height[m])^2
Low-SMI (y=1):
  남성(M): SMI < 45.4
  여성(F): SMI < 34.4
```

- `TAMA`: 전체 복부 근육 면적(Total Abdominal Muscle Area) 등 원본 데이터의 근육량 지표
- 성별에 따라 서로 다른 절단값(cut-off)을 사용하는 표준적인 SMI 기반 저근감 정의

---

## 3. Stage 1 — 임상 변수 기반 1차 선별 모델

### 3.1 입력 특징 (4개)
- `PatientAge`, `Height`, `Weight`, `PatientSex(남=1/여=0)`

### 3.2 전처리
- 결측치는 각 변수의 **중앙값(median)** 으로 대체
- 이후 **표준화(z-score)**: `(x - mean) / std` (평균·표준편차·중앙값은 내부 코호트에서만 산출 후 고정)

### 3.3 모델
- **로지스틱 회귀(Logistic Regression)**, `C=1.0`, `solver="lbfgs"`

### 3.4 학습/임계값 결정 절차
1. 내부 코호트에서 `StratifiedKFold(5-fold)`로 Out-of-Fold(OOF) 예측 점수를 산출 (fold별 재학습, 검증 fold는 학습에 사용되지 않음 → 데이터 누수 방지)
2. OOF 점수 분포에서 **민감도 ≥ 90%를 만족하는 후보 임계값 중 특이도가 가장 높은 값**을 임계값(th1)으로 채택
3. 내부 코호트 전체로 재학습한 **최종 고정 모델**(`fit_baseline_model`)을 외부 코호트에 적용해, th1 기준으로 그대로 평가 (외부 데이터는 학습에 전혀 관여하지 않음)

### 3.5 결과 해석
- Stage 1은 민감도를 최우선으로 확보하는 "그물을 넓게 치는" 모델 → 양성 판정군(screen positive, TP+FP)이 크고 PPV(양성예측도)가 낮은 한계가 있음

---

## 4. Stage 2 — AEC-128 잔차 기반 재분류 모델

### 4.1 목적
Stage 1에서 양성(screen positive, `score1 ≥ th1`)으로 분류된 환자만 대상으로, 위양성(FP)을 줄여 PPV/특이도를 개선한다. Stage 1에서 음성으로 분류된 환자는 건드리지 않는다 (내부 기준 FN이 n=12로 매우 적어, 재보정의 신뢰도가 낮고 잘못 건드릴 경우 민감도·특이도가 오히려 악화될 위험이 크기 때문).

### 4.2 AEC-128 곡선 전처리
1. 환자별 128개 slice 값을 **환자 평균으로 나누어 정규화** (환자 간 스케일 차이 제거, `aec_curve_comparison.py`와 동일한 정규화 방식)
2. **잔차화(Residualization)**: 표준화된 임상 변수(나이·키·몸무게·성별)로 128차원 정규화 곡선을 선형회귀(`LinearRegression`)로 예측하고, 실제값에서 예측값을 뺀 **잔차**만 사용
   - 이유: AEC-128 곡선의 겉보기 저-SMI 신호는 상당 부분이 체질량(BMI)에 의한 심슨의 역설(Simpson's paradox)식 교란이므로, 임상 변수로 설명되는 부분을 제거하고 순수하게 남는 정보만 사용
   - 잔차화 회귀는 **내부 코호트에서만 학습**하고 외부에는 고정 적용
3. **PCA 차원 축소**: 잔차 128차원을 PCA로 축소 (내부 코호트에서만 적합, 누적 설명분산 90%/95% 도달 시점의 성분 수, 최대 10개까지)

### 4.3 Stage-2 특징 벡터
아래를 열 방향으로 결합:
- 표준화된 임상 변수 4개 (Stage 1과 동일)
- AEC 잔차 PCA 점수 (k개, k ≤ 10)
- (설정에 따라 선택적) Stage 1의 로지스틱 회귀 decision score

### 4.4 Stage-2 분류기 후보
- 로지스틱 회귀(`logreg`, C=1.0)
- Gradient Boosting(`HistGradientBoostingClassifier`, max_depth=3, learning_rate=0.06, max_iter=150)

### 4.5 모델 선택(Sweep)
아래 5개 설정 조합을 내부 코호트 OOF에서만 비교하여 최적 설정을 선택 (외부 데이터는 전혀 사용하지 않음):

| model_type | PCA 설명분산 목표 | Stage-1 점수 포함 여부 |
|---|---|---|
| logreg | 0.90 | 미포함 |
| logreg | 0.90 | 포함 |
| logreg | 0.95 | 포함 |
| hgb | 0.90 | 포함 |
| hgb | 0.95 | 포함 |

각 설정에 대해:
1. Stage 1 양성군에 한해 5-fold OOF로 Stage-2 점수 산출 (Stage-1과 **동일한 fold 분할**을 재사용하여 그룹 간 정보 누수 차단)
2. **판정 채택 기준**(acceptance criteria)을 만족하는 임계값(th2) 중 PPV가 최대인 지점 선택:
   - 전체 민감도가 Stage-1-only 대비 **5%p 이상 하락하지 않을 것** (non-inferiority margin)
3. Stage-1 대비 특이도 상승분(spec_delta)이 가장 큰 설정을 최종 채택 (민감도 조건을 만족하는 후보 중에서 우선 선택, 만족하는 후보가 없으면 전체 중 최선)

### 4.6 최종 모델 고정 및 외부 검증
1. 선택된 설정으로 내부 코호트 전체를 이용해 Stage-2 분류기를 재학습 (Stage 1 양성군만 학습에 사용)
2. 외부 코호트에는 내부에서 고정된:
   - 임상 변수 표준화 파라미터
   - Stage-1 로지스틱 회귀 모델 및 th1
   - AEC 잔차화 회귀
   - PCA 변환
   - Stage-2 분류기 및 th2
   
   를 그대로 적용 (재학습 없음, 순수 held-out test)

### 4.7 최종 판정 결합 규칙
- Stage-1 음성 → 최종 음성 (그대로 유지)
- Stage-1 양성 & Stage-2 점수 ≥ th2 → 최종 양성
- Stage-1 양성 & Stage-2 점수 < th2 → 최종 음성 (재분류로 음성 전환)
- Stage-2 점수가 없는 경우(결측 등) → 안전하게 Stage-1 양성 판정 유지

---

## 5. 평가 지표 및 채택/기각 기준

각 코호트(internal/external)별로 Stage-1-only 대비 Stage-1+Stage-2 결합 모델을 비교:
- Accuracy, Sensitivity, Specificity, PPV, NPV, 혼동행렬(Confusion Matrix)

**PASS 조건** (`pass_fail`):
- 민감도 하락폭 ≤ 5%p (`SENS_NONINF_MARGIN = 0.05`)
- 특이도가 Stage-1-only 대비 **반드시 상승**

두 조건을 모두 만족해야 Stage-2 도입이 "성공"으로 판정된다.

---

## 6. 데이터 누수 방지 설계 요약

- 표준화 파라미터, 임계값, 잔차화 회귀, PCA, 모델 선택(sweep) — **전부 내부 코호트에서만 결정**
- 외부 코호트는 어떠한 학습·튜닝 과정에도 관여하지 않고, 마지막에 고정된 파라미터로 1회 평가만 수행
- Stage-1과 Stage-2의 5-fold 분할은 동일한 `StratifiedKFold(seed 고정)`을 재사용하여, 두 단계 사이 정보가 새지 않도록 함
