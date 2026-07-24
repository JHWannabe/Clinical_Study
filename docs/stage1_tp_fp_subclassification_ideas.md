# Stage 1 TP/FP 차이를 이용한 세분화 아이디어 (2026-07-23)

Stage 1 screen-positive 그룹(TP=true low-SMI, FP=false positive) 내에서 AEC-128
기반 세분화를 계속 파고들 가치가 있는지 판단하기 위해, 이번 세션까지 이미 확인된
수치를 먼저 정리하고 그 위에서 다음 시도 우선순위를 잡는다.

## 0. 이미 확인된 것 (재탐색 불필요)

- **Whole-curve RMSD permutation test (TP vs FP)**: 유의하지 않음. BMI
  propensity-matching 후에는 더 약해짐(gangnam p=0.983) —
  `code/stage2_aec_group_comparisons.py` comparison #16/#19,
  `docs/260724_...pptx` Slide 6.
- **AEC curve-shape feature 14개**(std/skew/kurtosis/peak/trough/slope 등):
  raw / propensity-matched / clinic-residualized 세 조건 모두 유의한 feature가
  거의 없음 — `code/stage2_aec_shape_features.py`.
- **AEC branch의 순수 기여도**: Net NRI 개선분(+71 internal / +34 external) 중
  AEC 자체 기여는 +8 / +14뿐. 나머지는 z_clin(Stage-1 LR score) 표준화만으로
  나오는 결과 — `code/stage2_model_no_aec_ablation.py`. 즉 raw AEC-128은 대부분
  clinical confound(체격/BMI)를 재인코딩하는 것이고 독립 정보는 적다.
- **AEC-sum 단일 스칼라**: 그것만으로 spec 53.9%→62.8%(p<1e-21), AUC 0.743
  (`outputs/sinchon_stage2_aec_sum_logreg/clinical_vs_aec_sum_lr_summary.csv`).
  → **AEC의 신호는 "형태(shape)"가 아니라 "레벨(전체 크기)"에 있다**는 게 이번
  세션에서 반복 확인된 패턴.
- **Residualized AEC-128 + CNN**(128차원 그대로 clinic 성분 제거 후 fusion):
  Net NRI는 최고(+77 internal / +52 external)지만 external NI test가 0.32%p
  차이로 **fail** — `code/stage2_model_residualized_aec.py` 2026-07-22 결과 로그.

## 1. 우선순위별 다음 아이디어

### 1순위 — Residualized AEC-sum 스칼라
128차원 residual curve를 CNN에 태우는 대신, `aec_i ~ CLIN_COLS` OLS residual을
슬라이스 단위로 구한 뒤 그 residual을 합/평균해 **스칼라 1개**로 만들어 로지스틱
회귀에 태운다. 128차원+CNN보다 분산이 훨씬 작아, external NI가 0.32%p 부족했던
갭을 스칼라화만으로 메울 가능성이 있다. `code/sinchon_stage2_aec_sum_logreg.py`
(raw sum 버전)를 그대로 복사해 입력만 residual curve의 합으로 바꾸면 된다 —
구현/검증 비용이 가장 낮고, 갭도 가장 가깝다.

### 2순위 — mAs를 독립 변수로 테스트
`metadata` 시트에 kVp/mAs가 이미 있으나 TP/FP 비교에 아직 쓰이지 않았다.
AEC-128은 사실상 슬라이스별 mA 프로파일이라 AEC-sum과 상당 부분 겹칠 수 있지만,
상관계수 확인은 비용이 거의 들지 않는다. 겹치지 않는 잔차가 있다면
`code/stage2_aec_shape_features.py`의 `compare_feature`를 그대로 재사용해
clinic/AEC-shape 패널에 한 줄 추가하는 정도로 충분하다.

### 3순위 — 서브그룹별 interaction 검정
전체 표본 풀링 기준으로는 raw AEC shape가 n.s.였지만,
`code/0723/aec_curve_comparison_global.py`의 `cohort_interaction_test`를
코호트가 아니라 **성별 × BMI quartile** 블록으로 바꿔 돌리면 "정상체중군에서는
AEC가 TP/FP를 가르는데 비만군에서는 안 갈린다" 같은 국소적 유의성이 숨어있는지
확인할 수 있다. 지금까지의 whole-curve/shape 테스트는 전부 풀링 기준이라 이런
조건부 신호를 희석시켰을 수 있다.

### 4순위 — 하드 재분류 대신 3단 gray-zone (비용 큼, 워크플로 변경 필요)
NI 마진이 0.32%p로 아슬아슬한 이유 자체가 "screen-positive를 이분법으로 강제
재분류"하는 설계 때문일 수 있다. Residualized AEC-sum 점수의 상/하위 분위만
confident하게 재분류하고 중간 구간은 그대로 양성 유지(확인검사 회부)하면,
민감도 손실 없이 특이도만 올리는 방향으로 NI 제약을 구조적으로 우회할 수 있다.
다만 임상적 합의가 추가로 필요한 워크플로 변경이라 우선순위를 가장 낮게 둔다.

## 참고
- `stage2_model_session_related_research.md` — 이번 세션 9가지 변경 근거 문헌
- `stage1_sensitivity_threshold_related_research.md` — Stage 1 Se=90% 채택 근거
- `code/0723/stage2_aec_tp_vs_fp_swap.py`, `code/0723/stage2_model_multimodal_swap.py`
  — 코호트 스왑(internal=sinchon) 버전
