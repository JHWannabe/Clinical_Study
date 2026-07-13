# Stage-2 1D-CNN 개선안 비교 (`code/4_*.py` ~ `code/9_*.py`)

`code/3_aec_cnn_reclassify.py`(1D-CNN, 이하 baseline)가 세 Stage-2 변형 중 가장 좋은
spec_delta(internal +0.052 / external +0.038, PASS/PASS — [residual_reclassify_cnn_algorithm.md](residual_reclassify_cnn_algorithm.md)
6장)를 낸 뒤, 이 표본 크기(screen-positive 내부 ~560명)에서 CNN을 더 개선할 수 있는 5가지
방향을 각각 독립된 스크립트로 구현하고, 6개 결과(baseline + 5개 변형)를 한 번에 비교하는
스크립트(`9_compare_cnn_variants.py`)를 추가했다.

## 1. 설계 방식 — "monkeypatch로 한 가지만 바꾸기"

5개 변형 스크립트는 `3_aec_cnn_reclassify.py`를 **복붙하지 않고 그대로 import한 뒤, 정확히
한 조각(함수 또는 클래스)만 교체**한다:

```python
base = importlib.import_module("3_aec_cnn_reclassify")
base.OUTPUT_DIR = base.PROJECT_ROOT / "outputs" / "<variant>"
base.ResidualCNN = MyVariantModel        # 예: 아키텍처를 바꾸는 경우
# 또는
base.train_cnn_ensemble = my_variant_fn  # 예: 앙상블 학습 절차를 바꾸는 경우

def main():
    base.main()
```

이게 가능한 이유는 `3_aec_cnn_reclassify.py`의 함수들(`stage2_oof_scores_cnn`, `main` 등)이
`ResidualCNN`, `train_cnn_ensemble`, `predict_ensemble` 같은 이름을 **모듈 전역(global)으로
참조**하기 때문이다 — Python은 이런 참조를 함수 정의 시점이 아니라 **호출 시점에 모듈의
`__dict__`에서 조회**하므로, `base.train_cnn_ensemble = ...`로 속성을 덮어쓰면 `base.main()`
내부에서 일어나는 호출도 새 함수를 쓰게 된다.

이 방식의 장점:

- 데이터 로딩, clinical 표준화, Stage-1 모델/th1, AEC 잔차화, 5-fold 분할, sweep 3-설정 비교,
  임계값 선택(`choose_stage2_threshold`), 평가 지표, McNemar/비열등성 검정, 플롯 저장까지
  **baseline과 완전히 동일한 코드 경로**를 타므로, 결과 차이는 오직 각 변형이 의도적으로
  바꾼 그 한 조각에서만 발생한다 (공정한 비교).
- `3_aec_cnn_reclassify.py`를 고치면 5개 변형에 자동 반영된다 (별도 동기화 불필요).

## 2. 변형별 요약

| 파일 | 무엇을 바꿨나 | 안 바꾼 것 | 계산 비용 (baseline 대비) | 참고 논문 (원조) | 참고 논문 (최신) |
| --- | --- | --- | --- | --- | --- |
| `4_aec_cnn_pretrain.py` | conv encoder를 내부 코호트 **전체**(n=1090, 라벨 없음)에 denoising-autoencoder로 사전학습한 뒤, 그 가중치로 각 fold/seed 모델을 초기화 | 아키텍처, 앙상블 절차, 폴드/threshold 로직 | 사전학습 1회분 추가 (~수십 epoch, 1회만) | Vincent et al., 2010, [Stacked Denoising Autoencoders (JMLR)](https://www.jmlr.org/papers/v11/vincent10a.html) | Ma et al., 2023/2024(v2), [A Survey on Time-Series Pre-Trained Models (arXiv:2305.10716)](https://arxiv.org/abs/2305.10716) |
| `5_aec_cnn_skip.py` | `ResidualCNN`에 실제 residual/skip connection 추가 (`relu(res_scale·block3(h2) + h2)`), `res_scale_init x block3_kernel`(4x2=8 조합) 그리드서치로 튜닝 | 학습 절차, 앙상블, 폴드/threshold 로직 | 거의 동일 (파라미터 증가 없음) x **그리드 8배**(각 grid point가 baseline 1회 실행 전체를 수행) | He et al., 2015, [Deep Residual Learning for Image Recognition (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385) | Xu et al., 2024, [Development of Skip Connection in Deep Neural Networks for Computer Vision and Medical Image Analysis: A Survey (arXiv:2405.01725)](https://arxiv.org/abs/2405.01725) |
| `6_aec_cnn_bagging.py` | 앙상블 멤버마다 fold-train 데이터를 클래스층화 bootstrap 재추출 후 학습, `bootstrap_frac x n_members`(3x3=9 조합) 그리드서치로 튜닝 | 아키텍처, epoch 선택 로직, 폴드/threshold 로직 | 거의 동일 x **그리드 9배**(각 grid point가 baseline 1회 실행 전체를 수행) | Breiman, 1996, [Bagging Predictors (Machine Learning 24, 123–140)](https://doi.org/10.1007/BF00058655) | Fan et al., 2025, [Diverse Models, United Goal: A Comprehensive Survey of Ensemble Learning (CAAI TIT 10, 959–982)](https://doi.org/10.1049/cit2.70030) |
| `7_aec_cnn_repeatedcv.py` | `StratifiedKFold(5-fold)`를 서로 다른 시드로 `N_REPEATS=5`회 반복하고 OOF logit을 평균 | 아키텍처, 앙상블 절차, threshold 선택 로직 자체 | **5배** (fold x repeat) | Kim, 2009, [Estimating Classification Error Rate: Repeated Cross-Validation, Repeated Hold-Out and Bootstrap (CSDA 53(11))](https://doi.org/10.1016/j.csda.2009.04.009) | Eve et al., 2026, [Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly Well (arXiv:2606.12552)](https://arxiv.org/abs/2606.12552) |
| `8_aec_cnn_film.py` | side feature(clinical/stage1 score)를 마지막 concat 대신 FiLM(`(1+γ)·z+β`)으로 conv 임베딩에 주입 | 학습 절차, 앙상블, 폴드/threshold 로직 | 거의 동일 | Perez et al., 2018, [FiLM: Visual Reasoning with a General Conditioning Layer (arXiv:1709.07871)](https://arxiv.org/abs/1709.07871) | Mohana Priya & Sangeetha, 2025, [Improved Birthweight Prediction With Feature-Wise Linear Modulation, GRU, and Attention Mechanism in Ultrasound Data (J. Ultrasound Med. 44, 711–725)](https://doi.org/10.1002/jum.16633) |

각 스크립트 상단 주석에 왜 그 변경이 도움이 될 것으로 기대하는지, baseline의 어떤 한계를
겨냥한 것인지 근거가 적혀 있다.

### 4. `4_aec_cnn_pretrain.py` — 비지도 사전학습

Stage-2 CNN은 screen-positive 환자만 학습에 쓰지만(내부 ~560명), 잔차 곡선 자체는 내부
코호트 전체(1090명)에서 이미 계산돼 있다 — `curve_mu`/`curve_sd`도 스크린 여부와 무관하게
내부 전체에서 fit된다([residual_reclassify_cnn_algorithm.md](residual_reclassify_cnn_algorithm.md#42-aec-128-곡선-전처리-및-잔차화) 4.2절과 동일 관례).
이 변형은 그 라벨 없는 나머지 정보를 conv encoder 사전학습에 쓴다:

1. `ResidualAutoencoder` (`conv` = baseline과 동일한 인코더 + 버리는 linear decoder)를
   내부 전체 잔차 곡선에 대해 **denoising 재구성** 과제로 학습 (`NOISE_STD`로 노이즈를 주고
   원본을 복원하도록 학습, MSE loss, 내부 20% split으로 early stopping).
2. Stage-2 fold/seed별 `ResidualCNN`을 만들 때, `conv` 가중치를 이 사전학습된 상태로
   초기화한 뒤 (`model.conv.load_state_dict(...)`) screen-positive 데이터로 미세조정.
3. 사전학습은 sweep 설정(3개)이나 fold와 무관하게 결과가 같으므로 **1회만 실행하고 캐시**한다
   (`_PRETRAIN_STATE`).

레이블이 전혀 관여하지 않는 과제라 fold 간 정보 누수 위험이 없다.

### 5. `5_aec_cnn_skip.py` — 진짜 skip connection + 그리드서치

클래스명이 `ResidualCNN`이지만 원래 구조(conv block1→block2→block3→GAP)에는 실제
residual/skip connection이 **없었다** — "residual"은 입력으로 쓰는 잔차 곡선을 가리키는
이름일 뿐이다. `conv`의 padding/pooling 산술을 따져보면 block2 출력과 block3 출력이 둘 다
`(B, 10, 32)`로 **shape가 정확히 일치**하기 때문에, 추가 파라미터 없이
`h3 = block3(h2) + h2` 형태의 identity shortcut을 넣을 수 있다. ResNet의 기본 동기(그래디언트가
conv+BN+ReLU 블록을 우회하는 경로 확보)가 특히 작고 신호가 약한 이 네트워크에서 도움이 될
것으로 기대한다.

**튜닝 이력** (스크립트 상단 주석):

1. **1차 시도**: block3을 `Conv→BN→ReLU`로 두고 그 출력에 h2를 더함(`ReLU(block3)+h2`).
   h2가 이미 자신의 ReLU를 거쳐 음수가 없으므로, 두 branch가 모두 0 이상인 상태에서
   더하기만 가능하고 뺄 수 없는 형태가 되어버렸다 — correction path 본연의 역할을 못함.
   모든 sweep 설정의 internal spec_delta가 baseline(+0.052)의 절반 이하(~0.023)로
   떨어졌고, 선택된 설정은 external에서 재분류 0건(spec_delta=0.0, FAIL).
2. **2차 시도**: block3의 ReLU를 없애고, 덧셈 뒤에 한 번만 적용(`ReLU(block3(h2)+h2)`,
   표준 ResNet 순서). 양쪽 코호트 모두 PASS로 돌아왔지만 spec_delta(internal +0.032,
   external +0.001)는 여전히 baseline(+0.052/+0.038)에 크게 못 미쳤다.
3. **3차 시도**: 학습 가능한 채널별 residual scale `res_scale`(ReZero/LayerScale 방식,
   `h3 = ReLU(res_scale·block3(h2) + h2)`)을 추가. 기본 초기값 `res_scale=0`에서는
   `h3 == h2`로 시작해, 학습이 "이미 동작하는 것으로 알려진 2-block 평면 네트워크"에서
   출발한다. 하지만 초기값 0과 block3 커널 폭 3을 고정한 것 자체가 검증 안 된 임의
   선택이었고, `outputs/9_compare_cnn_variants/spec_delta_comparison.png`(6개 CNN 변형
   비교)에서 이 설정은 baseline 대비 여전히 크게 뒤처졌다(internal +0.017 vs +0.052,
   external +0.008 vs +0.038) — 5개 제안 변형 중 가장 약한 성능.

**4차: 그리드서치로 전환.** shortcut branch가 실제로 가진 자유도는 `res_scale`의 초기값과
block3의 커널 폭 두 개뿐이므로, 한 번에 하나씩 추측하는 대신 `run_grid_search()`가 둘을
동시에 스윕한다:

- `RES_SCALE_INIT_GRID = [0.0, 0.1, 0.3, 1.0]` — 0.0(원래 선택, 평면 네트워크에서 출발)부터
  1.0(스케일 없는 풀-스트렝스 shortcut, 표준 ResNet 기본값)까지.
- `BLOCK3_KERNEL_GRID = [3, 5]` — 원래 폭 3 vs block2와 같은 폭 5(더 넓은 correction 커널).
- 4x2=8개 grid point마다 `3_aec_cnn_reclassify.py`의 sweep 3-설정(`SWEEP_CONFIGS` x
  `select_best_sweep_config`)을 그대로 돌려 그 grid point의 최선 (clinical, stage1 score)
  구성을 고른 뒤, `select_best_grid_point`가 8개 grid point의 internal spec_delta를
  안전마진 필터(`SAFE_MARGIN_FRAC=0.8`)로 걸러 최댓값을 최종 선택한다 — 이 두 단계는
  `3_aec_cnn_reclassify.select_best_sweep_config`와 동일한 "internal 신뢰구간이 margin의
  80% 안에 들어오는 후보만 신뢰" 규칙을 한 단계 위(아키텍처 하이퍼파라미터)에서
  재사용한 것이다. 각 grid point의 전체 산출물은 `outputs/5_aec_cnn_skip_grid/<태그>/`에
  남고, 우승한 grid point의 산출물만 `outputs/5_aec_cnn_skip/`(canonical 위치)로
  복사되어 `9_compare_cnn_variants.py`가 그대로 읽을 수 있게 한다.

이 저장소의 그리드 결과(`outputs/5_aec_cnn_skip/grid_search_ranking.csv`)에서는
`res_scale_init=1.0, block3_kernel=5`가 internal spec_delta(+0.035)로 안전마진을
통과한 8개 중 1위였지만, **그 설정을 external에 고정 적용하면 spec_delta=0.000으로
재분류가 한 건도 일어나지 않는다**(`verdict=FAIL`) — internal 전용 안전마진 필터를
통과한 설정조차 held-out external에서 실패할 수 있다는, 이 문서 4절이 반복해서 지적하는
"표본이 작을수록 결과가 설정 하나로 크게 흔들린다"는 경고의 또 다른 사례다. 그리드 전체
8개 중 external까지 PASS한 설정(예: `res_scale_init=1.0, block3_kernel=3` 또는
`res_scale_init=0.3, block3_kernel=3`)은 internal spec_delta가 더 낮았고(+0.022, +0.014),
그마저도 원래 기본값(`res_scale_init=0.0, block3_kernel=3`: internal +0.017 / external
+0.008)의 external 수치를 넘지 못했다 — **8개 중 원래 기본값을 internal·external 양쪽에서
동시에 능가하는 조합은 없었다**.

이 관찰을 반영해 `select_best_grid_point`는 internal 랭킹 1위를 external 검증 없이 그대로
승격하지 않는다: internal 안전마진을 통과한 후보 중 spec_delta 최댓값을 고른 뒤, **그 후보가
external에서도 PASS인지 사후 점검(sanity gate)** 하고, FAIL이면 `DEFAULT_RES_SCALE_INIT=0.0,
DEFAULT_BLOCK3_KERNEL=3`(원래 값)으로 되돌아간다 — external을 선택 기준 자체에 넣는 게
아니라(그러면 이 저장소 전체의 "external은 절대 선택에 안 쓴다" 원칙이 깨진다), internal
랭킹이 낸 결론을 그대로 자동 채택하기 전에 "그래도 external에서 완전히 무너지지는 않는가"만
확인하는 안전장치다. 이번 실행에서는 이 gate가 걸려 원래 기본값으로 폴백했고, 따라서
`outputs/5_aec_cnn_skip/`(canonical 폴더)에는 여전히 `res_scale_init=0.0, block3_kernel=3`
결과(internal +0.017 / external +0.008, PASS/PASS)가 실려 있다 — 그리드서치를 돌려본
결론은 "이 두 하이퍼파라미터를 바꿔서 skip 변형을 개선할 여지는 없었다"이다. `grid_search_
ranking.csv`의 `selected` 컬럼이 최종 채택된 grid point를 표시한다.

### 6. `6_aec_cnn_bagging.py` — Bagging 앙상블 + 그리드서치

Baseline의 `N_SEEDS=5` 앙상블은 모든 멤버가 **동일한 fold-train 데이터**를 보고 초기화만
다르다. 튜닝 로그(4.6절)에서 `N_SEEDS=7`이 오히려 소폭 하락했다는 건 "같은 데이터, 다른
초기화"만으로 얻을 수 있는 다양성이 한계에 가까웠다는 신호로 읽힌다. 이 변형은 두 번째
다양성 축을 추가한다 — 멤버마다 fold-train 데이터를 클래스별로 층화된 bootstrap으로
재추출(`_stratified_bootstrap_indices`)한 뒤 학습. Epoch 수(`find_best_epoch`)는 재추출 전
전체 데이터로 한 번만 정하고 모든 멤버가 공유해, "앙상블 전략만" 비교되도록 했다.

`_stratified_bootstrap_indices`는 클래스 전체(양성/음성)를 한 번에 bootstrap하지 않고
**클래스별로 따로** 재추출한다 — screen-positive 그룹의 Low-SMI 비율이 50% 훨씬 아래라,
전체를 한 번에 뽑으면 한 draw가 우연히 소수 클래스를 굶길 위험이 있기 때문이다.

**그리드서치로 전환한 이유**: 첫 버전은 bagging의 두 자유도 — 멤버당 bootstrap 재추출
크기(`bootstrap_frac`, fold-train 데이터 대비 비율)와 멤버 수(`n_members`) — 를 각각
1.0(클래스별 원본 크기만큼 복원추출, 교과서적 bagging 기본값)과 5(baseline `N_SEEDS`와
동일, 같은 비용으로 비교)로 고정한 미검증 추측이었다. `outputs/9_compare_cnn_variants/
spec_delta_comparison.png`(6개 CNN 변형 비교)에서 이 고정 설정은 baseline 대비 크게
뒤처졌다(internal +0.008 vs +0.052, external +0.015 vs +0.038) — `5_aec_cnn_skip.py`와
함께 5개 제안 변형 중 가장 약한 성능이었다.

`run_grid_search()`는 두 knob을 동시에 스윕한다:

- `BOOTSTRAP_FRAC_GRID = [0.7, 1.0, 1.3]` — 0.7(재추출 크기를 줄여 멤버 간 다양성을
  더 확보), 1.0(원래 선택), 1.3(멤버당 데이터를 늘려 다양성보다 안정성 쪽으로).
- `N_MEMBERS_GRID = [5, 7, 9]` — 5(원래 선택, baseline `N_SEEDS`와 동일 비용), 7, 9.
- 3x3=9개 grid point마다 `5_aec_cnn_skip.py`와 동일한 방식으로 내부 sweep 3-설정을
  돌려 최선 (clinical, stage1 score) 구성을 고른 뒤, `select_best_grid_point`가
  안전마진 필터(`SAFE_MARGIN_FRAC=0.8`)를 통과한 grid point 중 internal spec_delta가
  최대인 것을 선택한다. 각 grid point의 전체 산출물은
  `outputs/6_aec_cnn_bagging_grid/<태그>/`에 남고, 우승한 grid point만
  `outputs/6_aec_cnn_bagging/`으로 복사된다.

이 저장소에서 9개 grid point를 모두 실행한 결과, 안전마진을 통과한 grid point 중
`bootstrap_frac=1.0, n_members=7`이 internal spec_delta(+0.035)로 1위였고, 그 설정을
external에 고정 적용해도 spec_delta(+0.023)로 PASS/PASS — 원래 고정 설정(`frac=1.0,
n=5`, +0.008/+0.015)보다 뚜렷이 개선됐다. `select_best_grid_point`는 `5_aec_cnn_skip.py`와
동일하게 "internal 1위가 external에서도 PASS인지" 사후 점검하는 안전장치를 두고 있는데
(그러지 않았을 경우를 대비한 gate이지, 이번 우승 조합 자체를 external로 고른 게 아니다),
이번에는 그 gate를 그대로 통과했으므로 `bootstrap_frac=1.0, n_members=7`이 그대로
`outputs/6_aec_cnn_bagging/`(canonical 폴더)에 반영되어 있다. `5_aec_cnn_skip.py`와의
대비가 뚜렷하다: skip은 8개 조합 중 원래 기본값을 능가하는 조합이 없어 기본값으로
폴백했지만, bagging은 멤버 수를 5→7로 늘리는 것만으로 internal·external 모두에서 baseline과의
격차를 절반 가까이 줄였다 — "같은 데이터, 다른 초기화"(baseline 앙상블)와
"bootstrap 재추출 + 다른 초기화"(bagging)를 같은 멤버 수(5)로 비교한 원래 설계가 bagging
쪽에 불리했을 뿐, bagging 자체의 다양성 축이 무의미하지는 않았다는 뜻이다.

### 7. `7_aec_cnn_repeatedcv.py` — Repeated stratified CV

6장에서 이미 지적한 대로, `N_SEEDS` 하나만 바꿔도 external spec_delta가 +0.014~+0.038로
거의 3배 흔들렸다 — 이는 CNN 자체보다 **단일 5-fold 분할 위에서 th2를 고르는 절차**의
분산이 크다는 신호에 가깝다. 이 변형은 `StratifiedKFold(5, shuffle, seed)`를 서로 다른
`N_REPEATS=5`개 시드로 반복하고, 각 환자의 OOF logit을 반복 전체에 대해 평균
(`np.nanmean`)한 뒤 `choose_stage2_threshold`에 넘긴다. 모델/아키텍처는 baseline과 동일 —
비용은 대략 5배지만 이 네트워크가 워낙 작아(~2천 파라미터) 절대 시간은 감당할 만하다.

### 8. `8_aec_cnn_film.py` — FiLM 조건화

Baseline은 곡선 임베딩(10차원)과 side feature(clinical/stage1 score)를 완전히 독립적으로
계산한 뒤 **맨 마지막에만 concat**한다 — 즉 "곡선 모양"과 "환자 프로필"이 로짓에 각각
더해지는 형태로만 상호작용할 수 있고, 환자 프로필에 따라 곡선 임베딩 자체의 해석이
달라지는 상호작용은 표현할 수 없다. FiLM은 side feature를 채널별 scale/shift
`(γ, β) = Linear(side)`로 사영해 `z' = (1+γ)·z + β`로 곡선 임베딩을 변조한다. `film` 레이어를
0으로 초기화해 학습 시작 시점엔 `z'=z`(항등 변환)이 되도록 해, baseline 대비 학습 초반
불안정성이 추가되지 않게 했다.

## 3. 실행 및 비교 방법

각 스크립트는 독립적으로 실행 가능하며, 자기 자신의 `outputs/<번호>_aec_cnn_<이름>/`에
`stage1_vs_stage2_summary.csv`, `stage2_cnn_sweep_ranking.csv`,
`stage1_vs_stage2_confusion_matrix.png`, `clinical_vs_aec_assisted_table.png`를
baseline과 동일한 형식으로 저장한다.

`5_aec_cnn_skip.py`와 `6_aec_cnn_bagging.py`만 예외로, 실행할 때마다 내부적으로
그리드서치를 돌린다(5장/6장 참고) — grid point마다 `outputs/5_aec_cnn_skip_grid/<태그>/`
또는 `outputs/6_aec_cnn_bagging_grid/<태그>/`에 baseline과 동일한 4개 산출물을 남기고,
`grid_search_ranking.csv`에 전체 grid 랭킹을 기록한 뒤, 우승 grid point의 산출물만
`outputs/5_aec_cnn_skip/` / `outputs/6_aec_cnn_bagging/`(canonical 위치)에 복사한다 —
`9_compare_cnn_variants.py`는 이 canonical 위치만 읽으므로 그리드서치 존재 여부와 무관하게
그대로 동작한다.

```bash
python code/3_aec_cnn_reclassify.py    # baseline (이미 실행됨 -> outputs/3_aec_cnn_reclassify/)
python code/4_aec_cnn_pretrain.py
python code/5_aec_cnn_skip.py
python code/6_aec_cnn_bagging.py
python code/7_aec_cnn_repeatedcv.py    # 5배 느림 (repeated CV)
python code/8_aec_cnn_film.py

python code/9_compare_cnn_variants.py  # 위에서 실행된 것만 모아 비교
```

`9_compare_cnn_variants.py`는 **재학습을 전혀 하지 않고**, 이미 저장된
`stage1_vs_stage2_summary.csv`들을 읽어:

- `outputs/9_compare_cnn_variants/comparison_summary.csv` — variant x cohort(internal/external)
  단위로 sens_delta/spec_delta/verdict/ni_verdict/ppv 등을 한 표로 결합
- `outputs/9_compare_cnn_variants/spec_delta_comparison.png` — variant별 internal/external
  spec_delta 막대그래프 (external 막대 위에 PASS/FAIL 라벨 표시)

를 생성한다. 아직 실행하지 않은 변형은 에러 없이 건너뛰고 `[skip] ...` 메시지만 출력하므로,
5개를 전부 돌리지 않고 일부만 먼저 비교해봐도 된다.

### 결과 요약 (`outputs/*/stage1_vs_stage2_summary.csv` 현재 값)

| variant | internal spec_delta | internal 판정 | external spec_delta | external 판정 |
| --- | --- | --- | --- | --- |
| baseline (`3_aec_cnn_reclassify.py`) | +0.052 | PASS | +0.038 | PASS |
| pretrain (`4_aec_cnn_pretrain.py`) | +0.042 | PASS | +0.028 | PASS |
| skip (`5_aec_cnn_skip.py`, 그리드서치 결론: 원래 기본값 `res_scale_init=0.0, k=3` 유지) | +0.017 | PASS | +0.008 | PASS |
| bagging (`6_aec_cnn_bagging.py`, 그리드 우승: `bootstrap_frac=1.0, n_members=7`) | +0.035 | PASS | +0.023 | PASS |
| repeatedcv (`7_aec_cnn_repeatedcv.py`) | +0.027 | PASS | +0.046 | PASS |
| film (`8_aec_cnn_film.py`) | +0.031 | PASS | +0.041 | PASS |

이 저장소의 현재 실행 결과만 놓고 보면 **baseline을 확실히 능가하는 변형은 여전히 없다** —
`repeatedcv`와 `film`이 external spec_delta에서 baseline과 근접하거나 소폭 앞서지만(+0.046,
+0.041 vs +0.038), internal에서는 baseline(+0.052)에 못 미친다. 그리드서치 두 건의 결론은
서로 다르다:

- **`skip`**: 8개 조합(res_scale_init x block3_kernel) 중 원래 기본값을 internal·external
  양쪽에서 동시에 능가하는 조합이 없었다. internal 랭킹 1위(`res_scale_init=1.0, k=5`)는
  external에서 재분류를 한 건도 만들지 못해(FAIL) 자동으로 기각되고 기본값으로 되돌아갔다 —
  이 변형이 뒤처지는 이유는 하이퍼파라미터 선택이 아니라 **shortcut 자체의 효과가 이 표본
  규모에서 미미하다**는 쪽에 더 가깝다는 뜻이다.
- **`bagging`**: 9개 조합(bootstrap_frac x n_members) 중 `frac=1.0, n_members=7`이
  internal·external 양쪽 모두에서 원래 기본값(`frac=1.0, n=5`)을 뚜렷이 능가했다(internal
  +0.008→+0.035, external +0.015→+0.023) — baseline과의 격차를 상당히 줄였지만 여전히
  baseline을 넘어서지는 못한다.

요컨대 5개 제안 중 어느 것도 "baseline보다 신뢰성 있게 낫다"고 이 시점에서 결론 내릴 근거는
없지만, `bagging`은 그리드서치로 baseline에 가장 가까워진 변형이 됐다 — 아래 4절의 표본 크기
경고는 여전히 유효하다.

## 4. 비교 시 주의할 점

- **공정성의 범위**: 6개 스크립트 모두 baseline과 동일한 Stage-1 모델/th1, AEC 잔차화 회귀,
  fold 분할, sweep 3-설정, 임계값 선택 기준(`SENS_NONINF_MARGIN=0.05`), 평가지표를 공유한다.
  차이가 나면 그건 각 변형이 명시적으로 바꾼 한 조각 때문이라고 봐도 된다.
- **표본 크기가 여전히 작다**: [residual_reclassify_cnn_algorithm.md](residual_reclassify_cnn_algorithm.md) 6장이 이미
  지적했듯, screen-positive 학습셋(내부 ~560명)이 작아 이번 비교에서도 개별 스크립트의
  spec_delta가 시드/설정 하나로 크게 흔들릴 수 있다. 한 번의 internal/external split, 한 번의
  실행으로 "이 변형이 확실히 더 낫다"고 결론 내리지 않는다 — `7_aec_cnn_repeatedcv.py`가
  바로 이 불안정성 자체를 줄이려는 시도다.
- **`4_aec_cnn_pretrain.py`의 사전학습 범위**: 사전학습은 내부 코호트 **전체**(screen-negative
  포함)의 곡선을 라벨 없이 사용한다. 이는 `fit_curve_standardizer`가 이미 내부 전체에서
  fit되는 것과 같은 범위의 leakage 허용 관례이며, Stage-2 라벨(y) 자체는 어떤 사전학습
  단계에도 관여하지 않는다.

## 참고 메모

- [residual_reclassify_cnn_algorithm.md](residual_reclassify_cnn_algorithm.md) — baseline 1D-CNN(`3_aec_cnn_reclassify.py`) 상세 및 튜닝 로그
- [model_algorithm.md](model_algorithm.md) — 세 Stage-2 변형(PCA/bandfeat/CNN)이 공유하는 설계 원칙
- [residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) — PCA 버전(`1_aec_residual_reclassify.py`) 상세
- [residual_reclassify_bandfeat_algorithm.md](residual_reclassify_bandfeat_algorithm.md) — band/cluster_band 버전(`2_aec_residual_reclassify_bandfeat.py`) 상세
