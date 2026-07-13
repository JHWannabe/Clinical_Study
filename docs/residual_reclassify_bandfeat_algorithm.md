# Low-SMI 선별 모델 알고리즘 정리 (bandfeat 버전)

`code/2_aec_residual_reclassify_bandfeat.py`의 구현 설명. 설계 원칙은
[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md)(`code/1_aec_residual_reclassify.py`)와
동일하며, **Stage-2 잔차 곡선을 저차원 피처로 요약하는 방식(curve featurizer)만 다르다** —
전역 PCA 하나만 쓰는 대신 `band` / `cluster_band` / `combo` / `radiomics1d`를 추가해 내부 OOF
sweep으로 비교한다. 차이점은 4장에 집중돼 있고, 나머지 장은 두 파일이 공유하는 설계를 그대로
서술한다.

## 1. 개요

본 파이프라인은 저골격근량(Low-SMI, Skeletal Muscle Index) 환자를 선별하기 위한 **2단계(Stage-1 → Stage-2) 분류 모델**이다.

- **Stage 1** (`code/baseline/clinic-only_baseline.py`): 임상 변수만으로 민감도(Sensitivity) ≥ 90%를 만족하는 1차 선별(screening) 모델.
- **Stage 2** (`code/2_aec_residual_reclassify_bandfeat.py`): Stage 1에서 "양성(screen positive)"으로 분류된 환자군만을 대상으로, AEC-128 곡선에서 임상 변수로 설명되지 않는 잔차(residual) 정보를 추가해 특이도(Specificity)를 끌어올리는 재분류 모델. 잔차 곡선 요약 방식으로 PCA 외에 band/cluster_band/combo를 추가로 시도한다.

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

## 4. Stage 2 — AEC-128 잔차 기반 재분류 모델 (band/cluster_band/combo 확장)

### 4.1 목적
Stage 1에서 양성(screen positive, `score1 ≥ th1`)으로 분류된 환자만 대상으로, 위양성(FP)을 줄여 PPV/특이도를 개선한다. Stage 1에서 음성으로 분류된 환자는 건드리지 않는다 (내부 기준 FN이 n=12로 매우 적어, 재보정의 신뢰도가 낮고 잘못 건드릴 경우 민감도·특이도가 오히려 악화될 위험이 크기 때문).

### 4.2 AEC-128 곡선 전처리 및 잔차화
1. 환자별 128개 slice 값을 **환자 평균으로 나누어 정규화** (환자 간 스케일 차이 제거, `aec_curve_comparison.py`와 동일한 정규화 방식)
2. **잔차화(Residualization)**: 표준화된 임상 변수(나이·키·몸무게·성별)로 128차원 정규화 곡선을 선형회귀(`LinearRegression`)로 예측하고, 실제값에서 예측값을 뺀 **잔차**만 사용
   - 이유: AEC-128 곡선의 겉보기 저-SMI 신호는 상당 부분이 체질량(BMI)에 의한 심슨의 역설(Simpson's paradox)식 교란이므로, 임상 변수로 설명되는 부분을 제거하고 순수하게 남는 정보만 사용
   - 잔차화 회귀는 **내부 코호트에서만 학습**하고 외부에는 고정 적용
3. 잔차 128차원(`resid`)을 아래 5가지 **curve featurizer** 중 하나로 저차원 요약한다 — `fit_curve_featurizer()` / `transform_curve_featurizer()`.

참고: raw `aec_cropped` 시트(리샘플링 전)도 피처 후보로 검토했으나, 환자마다 `n_slices_cropped`가 110~238로 달라 slice 인덱스가 환자 간 정렬돼 있지 않아 제외했다 — column-wise로 그대로 쓰면 scan-length confound를 재도입하게 된다 (`1_aec_residual_reclassify.py`와 동일한 판단).

### 4.3 Curve featurizer 5종

| kind | 설명 | 파라미터 | 산출 차원(k) |
| --- | --- | --- | --- |
| `pca` | 전역 functional PCA. 잔차 128차원 전체를 혼합해 whole-curve mode로 요약 (내부 코호트에서 fit, 누적 설명분산 목표 도달 시점까지, 최대 10성분) | `pca_var_target` (0.90 / 0.95) | 가변 (설명분산 조건 충족 성분 수) |
| `band` | 128 slice를 `n_bands`개의 **등간격(equal-width)** 인접 구간으로 나눠 구간별 잔차 평균 | `n_bands` ∈ {4, 8, 16, 32} | `n_bands` |
| `cluster_band` | `band`와 동일하되 구간 경계가 **데이터 기반**: 인접 slice만 병합 가능하도록 chain-graph connectivity(위·아래 slice만 연결)로 제약한 Ward 클러스터링(`AgglomerativeClustering(connectivity=..., linkage="ward")`)을 내부 코호트 잔차 프로파일에 적합시켜, 등간격이 아니라 잔차 곡선의 실제 변화 지점에서 구간을 나눔 | `n_bands` ∈ {8, 16} | `n_bands` |
| `combo` | `band`와 `pca` 피처를 열 방향으로 결합 | `(n_bands, pca_var_target)`, 예: `(8, 0.90)` | `n_bands + pca_k` |
| `radiomics1d` | 1D 신호에 맞춰 변형한 고전적 radiomics 피처. (1) first-order 통계량(mean/std/IQR/MAD/energy/RMS/skewness/kurtosis, 8개)은 연속값 잔차 곡선에 직접 계산하고, (2) 1D GLCM(offset=1 co-occurrence 기반 contrast/energy/correlation/homogeneity/entropy, 5개)과 (3) 1D GLRLM(run-length 기반 short/long-run emphasis, gray-level/run-length non-uniformity, run-percentage, 5개)은 잔차를 `n_bins`개 quantile 구간으로 이산화(discretize)한 뒤 계산 | `n_bins` = 8 (이산화 bin 경계는 내부 코호트 잔차 분포의 quantile로 fit 후 고정) | 18 (8+5+5, 고정) |

모든 featurizer는 내부 코호트에서만 fit되고(`fit_aec_residualizer`와 동일한 원칙), 외부 코호트에는 고정된 상태(`feat_state`)를 그대로 적용한다. `radiomics1d`의 이산화 bin 경계도 마찬가지로 내부 코호트에서만 fit된다.

`band`가 전역 PCA보다 나은 경우가 있는 이유로 추정되는 것: 잔차 신호가 곡선 전체에 고르게 퍼진 global mode가 아니라 특정 구간(예: tail 구간)에 국소적으로 존재해서, PCA 성분에 섞여 희석되는 것보다 구간별 평균이 이 국소 신호를 더 잘 보존하기 때문. `cluster_band`가 `band`보다 나은 것은 등간격 대신 실제 신호가 바뀌는 지점에서 구간을 나누기 때문으로 추정된다.

`radiomics1d`는 band/cluster_band와 달리 slice 위치별 국소 평균이 아니라 곡선의 **형태(texture/heterogeneity)** — 얼마나 매끈한지, 값이 얼마나 자주 급변하는지 — 를 요약하므로 원리상 band류와 상호보완적인 정보를 담을 수 있다. 다만 실제 sweep 결과(4.6절)에서는 `cluster_band`/`pca`를 능가하지는 못했다 — 여전히 시도해볼 가치가 있는 후보군으로 남겨두되, 현재까지는 최종 채택으로 이어지지 않았다.

### 4.4 Stage-2 특징 벡터
아래를 열 방향으로 결합:
- 표준화된 임상 변수 4개 (Stage 1과 동일)
- 선택된 curve featurizer의 출력 (k차원, featurizer 종류에 따라 가변)
- (설정에 따라 선택적) Stage 1의 로지스틱 회귀 decision score

### 4.5 Stage-2 분류기 후보
- 로지스틱 회귀(`logreg`, C=1.0)
- Gradient Boosting(`HistGradientBoostingClassifier`, max_depth=3, learning_rate=0.06, max_iter=150)

### 4.6 모델 선택(Sweep)

`SWEEP_CONFIGS`는 두 부분으로 구성된다.

**(a) 기본 19개 설정** (모델 종류 × curve featurizer 종류/파라미터 × stage-1 점수 포함 여부, 모두 각
모델의 기본 하이퍼파라미터 사용):

| model_type | curve_feat (kind, param) | stage-1 점수 포함 |
| --- | --- | --- |
| logreg | pca, 0.90 | 미포함 |
| logreg | pca, 0.90 | 포함 |
| logreg | pca, 0.95 | 포함 |
| hgb | pca, 0.90 | 포함 |
| hgb | pca, 0.95 | 포함 |
| logreg | band, 4 | 포함 |
| logreg | band, 8 | 포함 |
| logreg | band, 16 | 포함 |
| logreg | band, 32 | 포함 |
| hgb | band, 8 | 포함 |
| hgb | band, 16 | 포함 |
| hgb | band, 32 | 포함 |
| logreg | cluster_band, 8 | 포함 |
| logreg | cluster_band, 16 | 포함 |
| hgb | cluster_band, 8 | 포함 |
| logreg | combo, (8, 0.90) | 포함 |
| hgb | combo, (8, 0.90) | 포함 |
| logreg | radiomics1d, 8 | 포함 |
| hgb | radiomics1d, 8 | 포함 |

**(b) 하이퍼파라미터 튜닝 그리드** (`build_tuned_sweep_configs`): 초기 버전은 우연히 기본
하이퍼파라미터에서 1등을 차지한 `cluster_band(8)` 설정 하나에 대해서만 HGB의
`learning_rate`/`max_iter`를 조정했었는데, 이는 편향된 탐색이다 — 기본값에서는 평범해 보이는
curve_feat가 튜닝 후에는 더 나은 internal spec_delta를 낼 수도 있기 때문이다. 그래서 이제
`stage-1 점수 포함`(위 (a)에서 모든 curve_feat에 대해 미포함보다 항상 우세했던 설정)과 결합된
10종 curve_feat(`TUNE_CURVE_FEATS` = pca 0.90/0.95, band 4/8/16/32, cluster_band 8/16, combo
(8, 0.90), radiomics1d 8) 각각에 대해, logreg과 hgb 모두 동일한 그리드로 튜닝한다:

- `HGB_TUNE_GRID`: `max_depth ∈ {2,3,4}` × `learning_rate ∈ {0.03,0.06,0.1,0.15}` × `max_iter ∈ {100,150,300,500}`
- `LOGREG_TUNE_GRID`: `C ∈ {0.1,0.3,1.0,3.0,10.0}`

모델 기본값과 동일한 그리드 포인트는 (a)의 미튜닝 행과 중복되므로 건너뛴다. 이렇게 만들어진
설정들이 (a)에 이어 `SWEEP_CONFIGS`에 추가되어, 총 529개 설정을 내부 OOF로만 비교한다.

각 설정에 대해:
1. Stage 1 양성군에 한해 5-fold OOF로 Stage-2 점수 산출 (Stage-1과 **동일한 fold 분할**을 재사용하여 그룹 간 정보 누수 차단)
2. **판정 채택 기준**(acceptance criteria)을 만족하는 임계값(th2) 중 PPV가 최대인 지점 선택:
   - 민감도 비열등성 검정 통과 ([residual_reclassify_algorithm.md](residual_reclassify_algorithm.md)의
     4.5.1절과 동일한 `noninferiority_test_sensitivity`). 단, sweep 단계의 임계값 선택과
     pass/fail 판정에는 `SELECTION_MARGIN = 0.6 * SENS_NONINF_MARGIN`(=0.03)이라는 **더 엄격한
     내부 전용 margin**을 사용한다 — 최종 보고되는 NI 판정(internal/external 모두)은 항상
     `SENS_NONINF_MARGIN`(0.05)을 그대로 쓴다. band/cluster_band까지 포함한 넓어진 탐색
     공간에서는 내부 CI 경계(0.05)에 딱 걸쳐서 통과하는 설정이 나올 수 있는데, 그런 설정은
     외부 코호트 자체의 표본 노이즈에 대한 여유가 부족할 수 있다. 외부 코호트 결과를 전혀
     보지 않은 채로, 더 빡빡한 내부 기준으로 선발함으로써 이 여유를 확보한다.
3. 민감도 조건을 만족하는 설정들 중 Stage-1 대비 특이도 상승분(spec_delta)이 가장 큰 설정을 최종 채택 (만족하는 후보가 없으면 전체 설정 중 spec_delta 최댓값)
4. **해석가능성 타이브레이크**: spec_delta가 최댓값과 `TIE_BREAK_SPEC_DELTA_TOL`(0.005) 이내로
   근접한 설정들 중, curve_feat 종류가 `band`/`cluster_band`(`BAND_LIKE_FEAT_KINDS`)인 설정이
   있으면 그중에서 최종 선택한다 — 국소 slice 구간 기반 피처가 전체 곡선을 섞는 PCA/combo보다
   임상적으로 해석하기 쉽기 때문에, 성능이 사실상 동률이면 해석가능성을 우선한다.

전체 sweep 결과는 `spec_delta` 내림차순으로 정렬되어 `stage2_sweep_ranking.csv`에 저장된다
(`rank` 컬럼 포함, PASS 여부로 1차 정렬 후 spec_delta로 2차 정렬).

현재까지 최선 결과: **HGB + cluster_band(8) + stage1 score**, `max_iter=500`으로 튜닝
(`max_depth=3`, `learning_rate=0.06`은 기본값 유지) — internal spec_delta=+4.9%, external
spec_delta=+2.5% (같은 sweep에서 PCA만 쓰는 `1_aec_residual_reclassify.py`의 최선인
설정: internal spec_delta=+4.3%, external +1.8%보다 우수). 튜닝/타이브레이크/NI 검정 방식 도입
전 구버전 대비 두 스크립트 모두 spec_delta 절대값이 낮아졌는데, 이는 임계값 선택 기준이
점추정치에서 CI 기반 비열등성 검정으로 엄격해졌기 때문이다
([residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) 4.5.1절 참고).

`radiomics1d` 결과: 529개 설정 중 internal spec_delta 기준 8위(logreg, `C=0.3`,
spec_delta=+3.7%, sens_delta=0.0, NI PASS) — `pca`(0.95)와 비슷한 수준이지만 `cluster_band(8)`
최선 설정(+4.9%)에는 못 미친다. `band`/`cluster_band` 상위권 다수보다는 낫지만 최종 채택으로
이어지지는 않았다 — 곡선의 국소 위치 정보(어느 slice 구간인지)보다 형태(texture) 정보가 이
문제에서는 상대적으로 약한 신호라는 뜻으로 해석할 수 있다. 다만 탐색한 `n_bins=8` 한 가지
설정만으로 나온 결과이므로, bin 수나 GLCM offset을 바꾸면 순위가 달라질 여지는 남아있다.

### 4.7 최종 모델 고정 및 외부 검증
1. 선택된 설정(모델 종류 + 튜닝된 하이퍼파라미터 + curve featurizer + stage1 점수 포함 여부)으로 내부 코호트 전체를 이용해 Stage-2 분류기를 재학습 (Stage 1 양성군만 학습에 사용)
2. 외부 코호트에는 내부에서 고정된:
   - 임상 변수 표준화 파라미터
   - Stage-1 로지스틱 회귀 모델 및 th1
   - AEC 잔차화 회귀
   - 선택된 curve featurizer 상태(`feat_state`) — PCA 성분/band 경계/cluster 라벨 등
   - Stage-2 분류기 및 th2

   를 그대로 적용 (재학습 없음, 순수 held-out test)

### 4.8 최종 판정 결합 규칙
- Stage-1 음성 → 최종 음성 (그대로 유지)
- Stage-1 양성 & Stage-2 점수 ≥ th2 → 최종 양성
- Stage-1 양성 & Stage-2 점수 < th2 → 최종 음성 (재분류로 음성 전환)
- Stage-2 점수가 없는 경우(결측 등) → 안전하게 Stage-1 양성 판정 유지

---

## 5. 평가 지표 및 채택/기각 기준

각 코호트(internal/external)별로 Stage-1-only 대비 Stage-1+Stage-2 결합 모델을 비교:
- Accuracy, Sensitivity, Specificity, PPV, NPV, 혼동행렬(Confusion Matrix), ROC AUC(Stage-1 점수 기준)

**PASS 조건** (`pass_fail`):
- 민감도 **비열등성(Non-Inferiority) 검정 통과** — Newcombe (1998) Method 10, 페어드 비율
  차이에 대한 Wilson score 기반 신뢰구간(`noninferiority_test_sensitivity`). 민감도 하락폭의
  97.5% 신뢰구간 상한이 `SENS_NONINF_MARGIN = 0.05` 이하여야 통과. 최종 보고 판정은 항상 이
  margin(0.05)을 쓰고, sweep 단계의 모델/설정 선택에만 더 엄격한 `SELECTION_MARGIN`(0.03, 4.6절
  참고)을 사용한다. 상세 근거는
  [residual_reclassify_algorithm.md](residual_reclassify_algorithm.md)의 4.5.1절 참고 —
  같은 함수를 공유한다.
- 특이도가 Stage-1-only 대비 **반드시 상승**

두 조건을 모두 만족해야 Stage-2 도입이 "성공"으로 판정된다.

McNemar 검정은 민감도/특이도뿐 아니라 **Accuracy**에도 적용되고, **Net NRI**
(`= 특이도 개선 flip 수 - 민감도 악화 flip 수`)도 함께 계산한다 — 상세 정의는
[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) 5절과 동일.

이 지표들은 `stage1_vs_stage2_summary.csv`, `stage2_sweep_ranking.csv`(전체 sweep 순위, 4.6절)와
두 그림으로 정리된다:
- `stage1_vs_stage2_confusion_matrix.png` — 선택된 설정(모델/curve_feat)의 internal/external
  혼동행렬, McNemar p-value, 비열등성 검정 결과 포함
- `clinical_vs_aec_assisted_table.png` — Clinical-only vs AEC-assisted(선택된 모델·curve_feat
  명시) 요약 표, AUC/Net NRI 포함

---

## 6. 데이터 누수 방지 설계 요약

- 표준화 파라미터, 임계값, 잔차화 회귀, curve featurizer(PCA 성분/band 경계/cluster 라벨), 모델 선택(sweep, 하이퍼파라미터 튜닝 포함) — **전부 내부 코호트에서만 결정**
- 외부 코호트는 어떠한 학습·튜닝 과정에도 관여하지 않고, 마지막에 고정된 파라미터로 1회 평가만 수행
- Stage-1과 Stage-2의 5-fold 분할은 동일한 `StratifiedKFold(seed 고정)`을 재사용하여, 두 단계 사이 정보가 새지 않도록 함
- `cluster_band`의 Ward 클러스터링도 내부 코호트 잔차 프로파일에서만 fit되고, 외부 코호트에는 학습된 라벨(구간 경계)을 그대로 적용한다 — 외부 데이터가 구간 경계 결정에 관여하지 않음
- 하이퍼파라미터 튜닝 그리드(4.6절 (b))도 내부 OOF 결과만으로 평가되며, 외부 코호트 성능을 보고
  최적 설정을 고르지 않는다 — sweep 전체가 내부 전용이라는 원칙은 튜닝 추가 이후에도 동일하다.
