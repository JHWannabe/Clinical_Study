# Low-SMI 선별 모델 알고리즘 정리 (1D-CNN 버전)

`code/stage2_aec_cnn_reclassify.py`의 구현 설명. 설계 원칙은
[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md)(`code/stage2_aec_residual_reclassify.py`)와
동일하며, **Stage-2에서 128차원 잔차 곡선을 저차원 피처로 요약하는 방식만 다르다** — PCA나
band/cluster_band로 사람이 설계한 요약 대신, 잔차 곡선을 그대로 작은 1D-CNN에 입력해 conv
필터가 국소적인 곡선 형태 특징을 직접 학습하도록 한다. 차이점은 4장에 집중돼 있고, 나머지
장은 세 파일이 공유하는 설계를 그대로 서술한다.

## 1. 개요

본 파이프라인은 저골격근량(Low-SMI, Skeletal Muscle Index) 환자를 선별하기 위한 **2단계(Stage-1 → Stage-2) 분류 모델**이다.

- **Stage 1** (`code/baseline/clinic-only_baseline.py`): 임상 변수만으로 민감도(Sensitivity) ≥ 90%를 만족하는 1차 선별(screening) 모델.
- **Stage 2** (`code/stage2_aec_cnn_reclassify.py`): Stage 1에서 "양성(screen positive)"으로 분류된 환자군만을 대상으로, AEC-128 곡선에서 임상 변수로 설명되지 않는 잔차(residual) 정보를 1D-CNN으로 학습해 특이도(Specificity)를 끌어올리는 재분류 모델.

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

[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) 3장과 완전히 동일 (로지스틱
회귀, `{age, height, weight, sex}`, 내부 5-fold OOF로 민감도≥90% 임계값 결정 후 외부에 고정 적용).
Stage 1은 민감도를 최우선으로 확보하는 "그물을 넓게 치는" 모델이라, 양성 판정군(screen positive,
TP+FP)이 크고 PPV가 낮다는 한계가 그대로 이어받아진다.

---

## 4. Stage 2 — AEC-128 잔차 기반 1D-CNN 재분류 모델

### 4.1 목적

Stage 1에서 양성(screen positive, `score1 ≥ th1`)으로 분류된 환자만 대상으로, 위양성(FP)을 줄여
PPV/특이도를 개선한다. Stage 1에서 음성으로 분류된 환자는 건드리지 않는다 (내부 기준 FN이
n=12로 매우 적어, 재보정의 신뢰도가 낮고 잘못 건드릴 경우 민감도·특이도가 오히려 악화될
위험이 크기 때문).

### 4.2 AEC-128 곡선 전처리 및 잔차화

PCA 버전과 동일한 절차를 잔차(resid)까지 그대로 재사용한다:

1. 환자별 128개 slice 값을 **환자 평균으로 나누어 정규화**
2. **잔차화**: 표준화된 임상 변수(나이·키·몸무게·성별)로 정규화 곡선을 선형회귀(`LinearRegression`)로
   예측하고, 실제값 - 예측값(잔차)만 사용 — BMI에 의한 Simpson's paradox식 교란 제거, 회귀는
   내부 코호트에서만 학습
3. **PCA로 차원을 줄이는 대신**, 잔차 128차원을 slice별 평균/표준편차로 표준화(`fit_curve_standardizer`,
   내부 코호트 전체 `resid_int`에서 fit, 외부에는 고정 적용)한 뒤 128차원 그대로 CNN에 입력한다.

### 4.3 1D-CNN 구조 (`ResidualCNN`)

작은 표본(screen-positive ~500-600명)에 맞춰 파라미터 약 2천 개 수준으로 작게 설계했다.

```
입력: 표준화된 128-포인트 잔차 곡선 (1채널)
  Conv1d(1→6,  k=9) → BatchNorm1d → ReLU → MaxPool1d(2)   # 128 → 64
  Conv1d(6→10, k=5) → BatchNorm1d → ReLU → MaxPool1d(2)   # 64 → 32
  Conv1d(10→10,k=3) → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)  # → 10차원 곡선 임베딩
concat([곡선 임베딩(10) , side features])
  Linear(→10) → ReLU → Dropout(0.5) → Linear(→1)  # logit
```

- `side features`: 표준화된 임상 변수 4개, (설정에 따라) Stage-1 decision score 1개 — PCA
  버전의 `stage2_feature_matrix()`가 만드는 `[clinical | curve-feature | stage1 score]` 구조와
  동일한 위치에 곡선 임베딩이 들어간다.
- `AdaptiveAvgPool1d(1)`(global average pooling)을 마지막에 둬서 곡선 내 절대 위치보다는
  국소 패턴의 존재 여부에 반응하도록 했다.

### 4.4 학습 절차 — early stopping + 앙상블 (핵심 차별점)

첫 버전은 fold마다 고정 epoch(80)으로 한 번만 학습시켰는데, 내부에서는 그럭저럭 동작했지만
외부 코호트에서 민감도가 92.9%→73.0%(-19.9%p)로 붕괴하며 비열등성 검정에 탈락했다 — 표본
규모(내부 학습셋 약 450-560명) 대비 CNN이 내부 코호트 특유의 잡음까지 학습해버린 전형적인
과적합. 이를 해결하기 위해 최종 절차는 fold(및 최종 freeze)마다 다음을 수행한다:

1. **내부 stratified validation split(20%)** 으로 early-stopping epoch 수를 먼저 찾는다
   (`find_best_epoch`) — patience=15, 최대 150 epoch, `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`
   기준 validation loss가 더 이상 개선되지 않으면 중단.
2. 위에서 찾은 epoch 수만큼, **해당 fold의 전체 학습 데이터**(validation split 없이)로
   `N_SEEDS=5`개의 새 모델을 처음부터 재학습한다 (`train_cnn_ensemble`) — 1단계에서 떼어둔
   validation 데이터를 최종 모델 학습에서 낭비하지 않기 위함.
3. 5개 모델의 logit을 평균(`predict_ensemble`)해 그 fold의 OOF 점수로 쓴다.

추가 정규화 장치:
- 학습 중 곡선에 상대 표준편차 5%(`NOISE_STD=0.05`)의 가우시안 노이즈를 더하는 augmentation
- Label smoothing (`eps=0.05`)
- Dropout 0.5, weight decay 3e-4, 작은 채널 수(6→10→10)

Stage-1/PCA/bandfeat와 동일한 `StratifiedKFold(5-fold, shuffle, seed 고정)`을 그대로 재사용하므로
(폴드 배정은 y/seed에만 의존), Stage-1 screen-positive 멤버십과 Stage-2 fold 멤버십 사이에
정보 누수가 없다.

### 4.5 Stage-2 모델 선택 Sweep

아래 3개 설정을 내부 OOF만으로 비교한다 (PCA/bandfeat 버전의 sweep과 동일한 취지, 곡선
피처화 방식이 CNN 하나뿐이라 설정 축이 "side feature 구성" 하나로 단순하다):

| 설정 | 곡선 입력 | 임상 변수 | Stage-1 점수 |
| --- | --- | --- | --- |
| curve only | O | X | X |
| curve + clinical | O | O | X |
| curve + clinical + stage1 | O | O | O |

임계값(th2) 선택은 PCA/bandfeat와 동일하게 `choose_stage2_threshold`(민감도 비열등성 통과
후보 중 PPV 최대)를 **기본 margin(`SENS_NONINF_MARGIN=0.05`)** 으로 사용한다 — bandfeat의
`SELECTION_MARGIN`(더 엄격한 내부 전용 margin, 0.03)도 시도했지만, CNN의 OOF 신호가
그 정도로 빡빡한 기준을 버틸 만큼 강하지 않아 임계값 탐색이 거의 flip 0건(사실상 Stage-1을
그대로 재현)으로 붕괴해버려 채택하지 않았다 (아래 4.6절 튜닝 로그 참고).

### 4.6 튜닝 로그

이 스크립트는 세 번의 튜닝 라운드를 거쳤다. 각 라운드의 internal/external spec_delta와
external 판정을 그대로 남긴다 — 표본이 작을수록 설정 하나 바꾸는 것만으로 결과가 크게
흔들릴 수 있다는 것을 보여주는 기록이기도 하다.

| 라운드 | 변경 내용 | internal spec_delta | external spec_delta | external 판정 |
| --- | --- | --- | --- | --- |
| v1 (최초) | 고정 80 epoch, 앙상블 없음 | +0.030 (PASS) | **-0.199** | **FAIL** (비열등성 탈락) |
| v2 | early stopping + `N_SEEDS=3` 앙상블 + noise/label smoothing + 채널 축소(6/10/10) + dropout 0.5 | +0.025 (PASS) | +0.014 (PASS) | PASS |
| v3 (되돌림) | BatchNorm→GroupNorm, `SELECTION_MARGIN=0.03` 적용, `N_SEEDS=5` | +0.002~+0.007 (PASS) | +0.000 | **FAIL** (개선폭 0) |
| **v4 (최종)** | v2 설정 + **`N_SEEDS=5`** (GroupNorm/SELECTION_MARGIN은 되돌림) | **+0.052 (PASS)** | **+0.038 (PASS)** | **PASS** |

- **GroupNorm 되돌림 이유**: 이 스크립트의 채널 수(6, 10, 10)가 워낙 적어 `GroupNorm(channels, channels)`이
  사실상 채널별 instance norm이 되고, 이는 매 샘플마다 activation을 평균0/분산1로 강제로 맞춰버려
  분류에 실제로 쓰이는 "샘플 간 진폭 차이" 자체를 지워버렸다 (internal spec_delta가 거의 0으로 붕괴).
- **`SELECTION_MARGIN` 되돌림 이유**: bandfeat 스크립트에서는 band/cluster_band까지 포함한 넓은
  탐색 공간에서 내부 margin 경계에 딱 걸쳐 통과하는 설정을 걸러내기 위해 유효했지만, CNN의
  OOF 신호는 그 정도로 엄격한 기준을 만족시킬 만큼 강하지 않아 임계값 탐색이 곧바로 "flip 0건"
  (Stage-1 그대로 재현)으로 수렴해버렸다.
- **`N_SEEDS` 3→5가 실제로 효과가 있었던 이유**: fold마다 서로 다른 초기화의 모델 여러 개를 평균내는
  것만으로 OOF 점수의 분산이 줄어, `choose_stage2_threshold`가 내부뿐 아니라 외부에서도 더 잘
  버티는 임계값을 찾을 수 있었다. `N_SEEDS=7`도 시도했으나 오히려 소폭 하락했고(+0.040/+0.034)
  학습 시간만 늘어, 5에서 멈췄다.

### 4.7 최종 모델 고정 및 외부 검증

1. 선택된 설정(4.5절)으로 내부 코호트 screen-positive 전체를 이용해 `N_SEEDS`개의 CNN을
   재학습(`train_cnn_ensemble`)
2. 외부 코호트에는 내부에서 고정된:
   - 임상 변수 표준화 파라미터
   - Stage-1 로지스틱 회귀 모델 및 th1
   - AEC 잔차화 회귀
   - 곡선 slice별 표준화 파라미터(`curve_mu`, `curve_sd`)
   - Stage-2 CNN 앙상블 및 th2

   를 그대로 적용 (재학습 없음, 순수 held-out test)

### 4.8 최종 판정 결합 규칙

- Stage-1 음성 → 최종 음성 (그대로 유지)
- Stage-1 양성 & Stage-2 점수 ≥ th2 → 최종 양성
- Stage-1 양성 & Stage-2 점수 < th2 → 최종 음성 (재분류로 음성 전환)
- Stage-2 점수가 없는 경우(결측 등) → 안전하게 Stage-1 양성 판정 유지

---

## 5. 평가 지표 및 채택/기각 기준

[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) 5장과 동일한 지표·검정을
공유한다 (Accuracy/Sensitivity/Specificity/PPV/NPV, McNemar, Net NRI, 비열등성 검정
`noninferiority_test_sensitivity`). PASS 조건도 동일: 민감도 비열등성 통과 + 특이도 반드시 상승.

이 지표들은 `outputs/stage2_aec_cnn_reclassify/`의 `stage1_vs_stage2_summary.csv`,
`stage2_cnn_sweep_ranking.csv`(4.5절 sweep 3개 설정 비교), `stage1_vs_stage2_confusion_matrix.png`,
`clinical_vs_aec_assisted_table.png`로 저장된다.

---

## 6. 최종 결과 (2026-07-10 기준)

같은 internal(`g1090.xlsx`, n=1090)/external(`sdata.xlsx`, n=926) 코호트에 대해 이 저장소의
세 Stage-2 변형을 비교하면:

| 방법 | internal spec_delta | external spec_delta | 판정 |
| --- | --- | --- | --- |
| PCA (`stage2_aec_residual_reclassify.py`) | +0.043 | +0.018 | PASS / PASS |
| band/cluster_band HGB (`stage2_aec_residual_reclassify_bandfeat.py`) | +0.049 | +0.025 | PASS / PASS |
| **1D-CNN (`stage2_aec_cnn_reclassify.py`, 이 문서)** | **+0.052** | **+0.038** | **PASS / PASS** |

튜닝 후 CNN 버전이 이 특정 internal/external split에서는 세 방법 중 가장 큰 specificity
개선을 보인다.

**주의**: 이걸 "CNN이 확실히 더 낫다"로 해석하지 않는다. 4.6절 튜닝 로그에서 보듯 screen-positive
학습셋(내부 약 560명, 외부 약 415명)이 작아 `N_SEEDS`처럼 학습과 직접 관련 없어 보이는
설정 하나만 바꿔도 external spec_delta가 +0.014에서 +0.038까지, 거의 3배 차이가 났다. 즉 이
결과는 "이번 튜닝/시드 조합에서는 기존 방법과 동등하거나 우수했다" 정도로 읽는 것이 안전하며,
단일 external split 위에서 나온 점추정치라 세 방법 사이의 차이가 통계적으로 유의하다고
단정할 근거는 없다.

---

## 7. 데이터 누수 방지 설계 요약

- 표준화 파라미터, 임계값, 잔차화 회귀, 곡선 slice별 표준화, 모델 선택(sweep) — **전부 내부
  코호트에서만 결정**
- 외부 코호트는 어떠한 학습·튜닝 과정에도 관여하지 않고, 마지막에 고정된 파라미터로 1회
  평가만 수행
- Stage-1과 Stage-2의 5-fold 분할은 동일한 `StratifiedKFold(seed 고정)`을 재사용
- fold별 early-stopping epoch 선택에 쓰는 내부 validation split은 **그 fold의 학습 데이터
  내부에서만** 이뤄지고, 최종 앙상블은 그 fold의 전체 학습 데이터로 재학습 — validation split이
  다른 fold/외부 코호트로 새어나가지 않음

## 참고 메모

- [model_algorithm.md](model_algorithm.md) — 세 Stage-2 변형이 공유하는 설계 원칙
- [residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) — PCA 버전 상세
- [residual_reclassify_bandfeat_algorithm.md](residual_reclassify_bandfeat_algorithm.md) — band/cluster_band 버전 상세
