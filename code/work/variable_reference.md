# AEC 게이트 변수 사전 (Variable-by-Variable Reference)

이 문서는 [main_aec_full_derivation_pipeline.md](main_aec_full_derivation_pipeline.md), [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md), [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md), [main_plot_internal_s90_core_1x3_mean_curves.md](main_plot_internal_s90_core_1x3_mean_curves.md) 4개 문서에 등장하는 **하드코딩된 상수/변수를 하나씩 꺼내**, 실제 코드 줄 번호와 함께 "왜 이 값인가"를 정리한 참조 문서입니다. `old_python/` 폴더의 원조 탐색 스크립트 2개(`aec_region_search_surrogate.py`, `aec_new_region_guided_cnn_gate.py`)도 근거로 포함했습니다.

**읽는 법**: 표의 "왜 이 값인가" 칸은 코드 주석·비교 실행·통계적 근거 중 실제로 확인 가능한 것만 적었습니다. 근거가 코드/문서 어디에도 없는 경우는 "확인되지 않음"이라고 명시합니다.

---

## 0. 전체 지도 — 이 값들이 어느 파일에서 왔는가

```
old_python/aec_region_search_surrogate.py            (R1~R4를 처음 찾아낸 원조 슬라이딩 윈도우 스캐너)
old_python/aec_new_region_guided_cnn_gate.py          (최종 CNN-mimic 이전의 1세대 CNN 게이트)
        │
        ▼ (파라미터가 다듬어져 이식됨)
python/aec_new_region_surrogate_combo_gate.py         (R1~R4 4구간으로 확정, 4-branch 조합 탐색을 처음 완료)
        │
        ▼ (값이 그대로 복사됨)
python/main_aec_full_derivation_pipeline.py           (전체 탐색 절차를 의존성 없이 재현하는 "족보" 파일)
python/main_aec_new_region_cnn_surrogate_mimic_gate.py (2세대 CNN, TEACHER_BRANCHES를 combo_gate 결과에서 복사)
python/main_plot_external_s90_core_1x3_mean_curves.py  (확정된 BRANCHES/SELECTED_PATTERNS로 그림만 그림)
python/main_plot_internal_s90_core_1x3_mean_curves.py  (위와 동일, internal 데이터만 다름)
```

---

## 1. 데이터 전처리 변수

`load_dataset()` — [main_aec_full_derivation_pipeline.py:230](python/main_aec_full_derivation_pipeline.py#L230), [aec_lock_smoothed_deesc_gate.py:48](python/aec_lock_smoothed_deesc_gate.py#L48) 등 4개 파일에 거의 동일하게 중복 구현.

| 변수 | 값 | 무엇을 하는가 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- | --- |
| `SMOOTHING_SIGMA` / `SIGMA` | 1.0 | AEC-128 곡선에 Gaussian smoothing 적용 시 표준편차 | **경험적 고정.** 스캐너 노이즈로 인한 점 대 점 튀는 값을 눌러주되, R1~R4처럼 10~30포인트 길이의 구간 형태는 보존해야 하므로 약하게만 흐림. σ=1은 1~2포인트 폭의 잡음만 지우는 최소 강도 — 더 큰 σ와 비교한 로그는 파일 안에 없어, 최종 고정값만 확인됨 | full_derivation…:101 |
| 저-SMI 컷오프 (남 45.4 / 여 34.4) | 45.4, 34.4 cm²/m² | SMI = TAMA / 키(m)². 이보다 작으면 근감소증 위험군(y=1) | **외부 기준 채택.** Derstine 기준을 그대로 가져온 값. 인수인계 문서에 "Yoon criteria were considered earlier but discarded by user. Derstine low SMI is fixed"라고 명시 — 다른 기준도 검토했지만 최종적으로 이 값으로 고정한 의사결정 이력이 있음 | GPU_HANDOFF_PROMPT.md:39-52 |
| `patient_wise_mean_normalize` / `row_norm` | 각 행 ÷ 그 행의 평균 | 환자마다 다른 스캐너의 절대 관전류 스케일을 지우고 상대적 모양만 남김 | **0-division 방지.** 평균이 0이거나 NaN인 극단 케이스에서 나눗셈 오류가 나지 않도록 안전판 — 통계적 근거라기보다 방어적 코딩 | full_derivation…:224-227 |
| `matrix_from_aec_sheet`의 결측 대치 | 열별 중앙값(불가 시 전체 중앙값) | 못 읽은/비정상 AEC 값을 같은 신체 지점 다른 환자들의 중앙값으로 채움 | **이상치에 강함.** 중앙값은 소수의 극단값에 덜 흔들림. "왜 평균이 아니라 중앙값인지"의 별도 비교 실험은 코드에 없음 | full_derivation…:212-221 |

---

## 2. 임상 모델 변수

`make_context()` / `clinical_scores()` — 나이·키·몸무게·성별 4개 변수만으로 "임상적 위험군"을 정의하는 1단계 모델.

| 변수 | 값 | 무엇을 하는가 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- | --- |
| `TARGET_OPS` / `OPS` (S80/S85/S90) | 민감도 목표 0.80/0.85/0.90 | 저-SMI 환자를 최소 80/85/90% 잡아내도록 임상점수 임계값을 역산 | **임상적 요구.** 민감도를 낮게 잡으면 위험한 환자를 놓치는 비율이 커져 임상적으로 받아들이기 어려움. 90%(S90)를 주력 운영점으로 삼아 "거의 다 잡아내되 위양성이 얼마나 늘어나는지"를 보여주고, 80/85%는 그 사이의 트레이드오프를 보여주는 대조군 | full_derivation…:102 |
| `LogisticRegression(C=1e6, ...)` | C = 1,000,000 | 임상 로지스틱 회귀의 정규화 강도 (클수록 정규화 없음) | **변수 4개뿐.** age/height/weight/sex_M 단 4개 변수만 쓰는 저차원 모델이라 과적합 위험이 낮으므로, 정규화로 계수를 인위적으로 줄이지 않고 데이터가 말하는 그대로의 계수를 쓰기 위해 정규화를 사실상 껐음 | full_derivation…:297 |
| `stratified_folds(k=5)` | 5-fold | internal 환자를 5등분해 OOF 임상 점수 생성 | **과적합 방지.** 내부 데이터로 만든 점수를 그대로 내부 평가에 다시 쓰면 "답을 이미 본 시험"이 됨. 5-fold는 통계학의 표준 관행값 — 이 값 자체에 대한 별도 민감도 분석은 없음 | full_derivation…:281-288 |
| `threshold_for_min_sensitivity` | 목표 민감도 이상 유지 + 특이도 최대 | 모든 임계값 후보 중 조건을 만족하며 특이도가 가장 좋은 지점 선택 | **두 목표 중 우선순위 고정.** 민감도(위험군을 놓치지 않는 것)를 최우선 제약으로 걸고, 그 다음 특이도를 최적화 — "안전을 먼저 만족한 후 효율을 높인다"는 순서를 코드 구조로 강제 | full_derivation…:332-342 |

---

## 3. Region Scout 탐색 변수 — "왜 하필 그 구간을 봤나"

R1~R4를 사람이 정하기 전, 128개 지점 전체를 기계적으로 훑은 단계. 파라미터 대부분이 `old_python/aec_region_search_surrogate.py`에서 그대로 이식됨.

| 변수 | 값 | 무엇을 하는가 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- | --- |
| `COARSE_STEP` | 8 | 128포인트 전체를 8칸 간격으로 이동하며 윈도우 후보 생성 (1차, 성긴 스캔) | **계산량 절충.** 1칸씩 다 훑으면 후보(길이×위치×서술자×부호×width×lambda)가 수십만 개로 폭증 — 1차는 넓게 훑고, 결과가 좋은 곳만 2차에서 촘촘히 재탐색하는 2단계 전략의 성긴 쪽 간격 | old_python/aec_region_search_surrogate.py:24 |
| `FINE_STEP` | 4 | 1차에서 유망했던 구간 주변만 4칸 간격(coarse의 절반)으로 재탐색 | **해상도 2배.** coarse에서 "대략 어디"를 찾았으니 근방에서는 두 배 촘촘하게 봐서 경계(시작/끝 지점)를 더 정밀하게 좁힘 | old_python/aec_region_search_surrogate.py:25 |
| `COARSE_LENGTHS` | [16, 24, 32] | 1차 스캔에서 시도하는 윈도우 길이 후보 | **해부학적 구간 크기 가정.** 128포인트 전체 대비 너무 짧으면 노이즈에 취약, 너무 길면 "국소적 형태"라는 의미가 사라져 전체의 약 12~25% 길이대에서 대표값만 시도 | old_python/aec_region_search_surrogate.py:26 |
| `FINE_LENGTHS` | [12, 16, 20, 24, 28, 32] | 2차(fine) 스캔에서 시도하는 윈도우 길이 후보 | **촘촘한 재탐색.** coarse가 16/24/32만 봤다면 fine은 그 사이(12,20,28)까지 채워, 실제 확정된 R1(길이12)/R2(길이24)/R4(길이12) 등 최종 길이가 이 목록 안에서 나옴 | old_python/aec_region_search_surrogate.py:27 |
| `lo=33, hi=128` (fine 스캔 범위) | 33~128 구간만 재탐색 (1~32 제외) | 2차 스캔을 처음부터 다시 보지 않고 33번 지점 이후만 촘촘히 봄 | **이전 실행 결과에서 역산.** 임의 고정값이 아니라 `aec_region_search_surrogate.py`의 `fine_windows_from_top(coarse, flank=16)`이 실제로 만든 **적응형** 결과. coarse 스캔에서 internal 기준 통과(또는 점수 상위) 상위 30개 윈도우 각각의 앞뒤로 flank=16만큼 여유를 두고 그 구간만 재탐색했더니, 좋은 후보가 거의 다 33번 이후에 몰려 있었음. `main_aec_full_derivation_pipeline.py`는 이 결론만 `lo=33` 하드코딩으로 단순화해 재현 | old_python/aec_region_search_surrogate.py:243-253 → full_derivation…:717-718 |
| `flank` | 16 | coarse 상위 윈도우의 시작/끝 지점에서 앞뒤로 얼마나 넓혀 fine 재탐색 범위를 잡을지 | **coarse 해상도와 비례.** coarse step(8)의 2배로 잡아 "coarse가 놓쳤을 수 있는 경계 근처"까지 fine 스캔이 커버하도록 여유를 줌 — 정확히 왜 2배인지 별도 근거는 없음 | old_python/aec_region_search_surrogate.py:243 |
| `DESCRIPTORS` (12개 서술자) | level_mean/sd, endpoint_delta, linear_slope, slope_mean/sd, abs_slope_mean/max, curv_mean/sd, abs_curv_mean/max | 구간 하나에서 뽑을 수 있는 레벨/기울기/곡률 3계열의 형태 요약 통계 12가지 | **세 층위 커버.** 레벨(높낮이), 기울기(오르는가/내리는가), 곡률(구부러진 정도)이라는 세 가지 독립적 관점을 각각 평균/변동성/절대값 버전으로 나눠 12개를 만듦 — 특정 서술자 하나에 편향되지 않도록 넓게 시도 | full_derivation…:167-180 |
| `SIGNS` | [-1, +1] | 서술자 값이 클수록 위험(+1)인지 작을수록 위험(-1)인지 둘 다 시도 | **방향을 가정하지 않음.** 방향은 사람 직관이 아니라 데이터가 정하게 함 — 실제로 확정된 4개 branch 중 3개가 sign=-1인 것도 이 전수비교의 결과 | full_derivation…:181 |
| `WIDTHS` | [0.35, 0.50, 0.70] | gate 수식에서 임상 임계값 근처 얼마나 넓은 범위까지 형태 특징 영향을 열어줄지 정하는 가우시안 폭 | **임계값 근처로 한정.** 너무 좁으면 극소수에게만 영향, 너무 넓으면 임상점수가 낮은 사람까지 흔들려 "임계값 근처에서만 재분류"라는 취지가 흐려짐 — 좁은 대역(0.35~0.70)만 시도한 것 자체가 설계 의도 | full_derivation…:182 |
| `LAMBDAS` | [0.25, 0.40, 0.55, 0.70] | 형태 특징이 최종 게이트 점수에 얼마나 강하게 반영되는지(가중치) | **임상 점수를 지배하지 않도록.** 1보다 훨씬 작은 값들로만 구성 — 형태 특징은 임상 점수를 "보정"하는 역할이지 "대체"가 아님. 확정된 4개 branch가 모두 lambda=0.25(최소값)로 수렴한 것은 안전 기준(민감도 손실 제약)을 만족하려면 영향력을 약하게 눌러야 했다는 뜻 | full_derivation…:183 |
| `internal_selection_score` 가중치 (0.45/0.20/0.20) | 평균정확도 + 0.45×최소정확도증가 + 0.20×최소특이도증가 − 0.20×최대민감도손실 | 스카우트 단계에서 어떤 후보 윈도우가 유망한지 하나의 숫자로 줄세우는 점수 공식 | **최악의 경우 기준.** 평균이 아니라 "최소" 정확도/특이도 증가, "최대" 민감도 손실을 써서 특정 운영점에서만 반짝 좋은 후보가 상위에 오르지 않도록 함. 0.45/0.20/0.20이라는 정확한 비율의 근거는 **확인되지 않음** | old_python/aec_region_search_surrogate.py:198-207 |

---

## 4. 확정 구간(LOCKED_REGIONS) & 게이트 수식

| 구간 | 범위 | 왜 이 경계인가 |
| --- | --- | --- |
| `R1_045_056` | 45–56 (길이12) | Scout 단계 fine 스캔(33~128, 길이 12~32) 상위권 결과를 시각적으로 해석해 확정. 해부학적 장기 이름과는 무관(§9 참고) |
| `R2_057_080` | 57–80 (길이24) | 동일. R1 바로 다음 구간, R1과는 다른 서술자(level_mean)가 유리하게 나온 구간 |
| `R3_097_128` | 97–128 (길이32) | 후반부(간 방향)에서 유의미한 기울기 신호가 나온 구간 |
| `R4_117_128` | 117–128 (길이12) | R3의 부분집합(끝쪽 12포인트)을 별도 구간으로 분리 — R3(넓은 후반부 기울기)와 R4(가장 끝부분의 급격한 변화)를 별개 신호로 취급 |

**공통 수식 — `branch_gate_score`**

```python
boundary   = exp(-0.5 * ((clinical_z - threshold) / width)^2)   # 임상점수가 임계값 근처일 때만 1에 가까워지는 가우시안 창
gate_score = clinical_z + lambda * boundary * sign * feature_z   # 형태 특징을 그 창 폭만큼만 더해줌
branch_vote = gate_score < threshold                              # 다시 임계값 아래로 내려가면 "위험강등 투표"
```

**왜 이런 형태인가** — 임상점수가 임계값에서 멀리 떨어진 사람(이미 확실히 저위험/고위험)에게는 `boundary≈0`이 되어 형태 특징이 사실상 영향을 주지 않습니다. 오직 "임상적으로 애매하게 양성 판정을 받은" 사람들만 형태 신호로 다시 흔들리도록 설계된 것 — 코드 구조 자체가 "AEC는 1차 검사가 아니라 2차 재분류"라는 임상적 프레이밍을 수식으로 구현한 것입니다.

근거: full_derivation…:469-475 (`branch_gate_score`/`branch_vote`)

---

## 5. 1차 규칙(LOCKED_PRIMARY_BRANCHES) — region별 4개 변수의 이유

각 region마다 **feature(어떤 서술자)**, **sign(부호)**, **width(폭)**, **lambda(가중치)** 4개 변수가 확정되어 있습니다. lambda는 4개 모두 0.25로 동일합니다.

### R1 · endpoint_delta · sign −1 · width 0.50
투표 조건: `clinical_z - 0.25·boundary·endpoint_delta < threshold`

`endpoint_delta` = 구간의 마지막 지점 값 − 첫 지점 값 (구간 안에서 순수하게 얼마나 올라갔거나 내려갔는지). sign=-1은 "R1 구간에서 곡선이 뚜렷하게 내려가는 사람일수록 위험강등 후보"라는 뜻. width=0.50은 R2(0.70)보다 좁고 R3(0.35)보다 넓은, 세 후보 중 중간값이 스크리닝에서 선택됨.

### R2 · level_mean · sign −1 · width 0.70
투표 조건: `clinical_z - 0.25·boundary·level_mean < threshold`

`level_mean` = 구간 전체의 평균 높이. R1(변화량)과 달리 R2는 "이 구간이 전반적으로 얼마나 낮은가"라는 절대적 레벨을 봄 — 같은 후반부라도 서로 다른 종류의 신호(변화량 vs 절대수준)를 쓰도록 자연스럽게 나뉜 것. width=0.70(가장 넓은 값)이 선택된 것은, R2 신호가 임계값에서 조금 더 멀리 떨어진 환자군까지 확장해도 민감도 손실 제약을 넘지 않았다는 뜻으로 해석됨.

### R3 · linear_slope · sign +1 · width 0.35
투표 조건: `clinical_z + 0.25·boundary·linear_slope < threshold`

`linear_slope` = 구간에 최소자승 직선을 적합했을 때의 기울기. **유일하게 sign=+1**인 branch — R3에서는 "기울기가 클수록(가파르게 오를수록)" 위험강등 방향. 다른 3개가 전부 sign=-1인 것과 반대인데, R3(97-128, 간 방향으로 올라가는 회복구간)에서는 "가파르게 회복하는 모양"이 저위험 형태라는, R1/R2(전반부)와는 반대되는 방향성이 실제로 존재함을 보여줌. width=0.35(가장 좁은 값)는 임계값 아주 근처의 환자에게만 적용하도록 보수적으로 제한된 것.

### R4 · endpoint_delta · sign −1 · width 0.50
투표 조건: `clinical_z - 0.25·boundary·endpoint_delta < threshold`

R1과 같은 서술자·같은 부호·같은 width를 R4(R3의 마지막 12포인트 부분집합)에서도 다시 사용. R3가 "전체 후반부의 완만한 기울기"를 본다면 R4는 "가장 끝부분에서의 급격한 끝점 변화"를 별도로 포착 — 같은 서술자를 두 번 쓴 것이 중복이 아니라, R3와 R4가 각각 다른 시간축 스케일(넓은 추세 vs 국소적 끝단)을 담당하도록 의도된 설계.

> **공통적으로 lambda=0.25인 이유**: 4개 후보(0.25/0.40/0.55/0.70) 중 가장 작은 값이 4개 region 모두에서 채택되었습니다. 이는 §7의 비열등성(noninferiority) 검정 — "민감도 손실의 95% 상한이 5%p를 넘으면 안 된다" — 를 통과하기 위해 형태 특징의 영향력을 최대한 약하게 눌러야 했다는 뜻입니다.

---

## 6. 패턴 & 안전성 검정 변수

| 변수 | 값 | 무엇을 하는가 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- | --- |
| `LOCKED_PRIMARY_PATTERNS` | {"++++","++--","+--+","--+-","---+"} | 4개 branch 투표(+/-) 조합 16가지(2⁴) 중 실제로 "위험강등"으로 인정하는 5가지 | **2개 이상 동의.** 5개 패턴 모두 "+"(투표함)가 최소 2개 이상 포함 — "네 구간 중 최소 두 곳 이상이 동시에 같은 방향을 가리켜야 신뢰한다"는 다수결(consensus) 원칙. CNN 버전의 `soft_atleast2_prob`("2개 이상 동시 투표 확률" 학습)와 정확히 같은 설계 철학 | full_derivation…:149 |
| `NI_MARGIN` | 0.05 (5%p) | 위험강등으로 진짜 저-SMI 환자를 놓치는 비율(민감도 손실)의 95% 신뢰상한이 이 값을 넘으면 규칙 기각 | **임상 비열등성 기준.** 신약/신의료기기 임상시험에서 흔히 쓰는 "비열등성 마진" 관행을 차용 — 5%p라는 구체적 숫자 자체가 이 코드에서 통계적으로 유도된 것은 아니고, 임상 연구의 보수적 관례값을 채택한 것으로 보임 | full_derivation…:104 |
| `clopper_pearson_one_sided_upper` | 단측 95% 정확 신뢰상한 | 이항비율(놓친 환자 수/전체 위험군 수)의 신뢰구간을 정규근사가 아닌 정확한 베타분포 기반으로 계산 | **표본이 적을 때도 안전.** 강등 대상이 수십 명 단위로 적어 정규근사(대표본 가정) 대신 소표본에서도 보수적으로 정확한 Clopper-Pearson 방법을 선택. "단측"인 이유는 관심사가 "손실이 너무 크지 않은가"라는 한쪽 방향뿐이기 때문 | full_derivation…:501-506 |
| `MIN_DEESC_N` | 10명 | CNN-mimic 계열 탐색에서 강등 대상이 이 인원수 미만이면 규칙 무효 처리 | **최소표본 확보.** 10명 미만은 표본이 작을수록 신뢰구간이 넓어져 상한이 우연히 낮게 나오기 쉬워 신뢰하기 어려움 | aec_new_region_cnn_surrogate_mimic_gate.py:35 |
| `MAX_SENS_LOSS` | 0.08 (8%p) | CNN 계열 탐색에서 쓰는 민감도 손실 한도 | **1차 규칙보다 완화된 기준.** 1차(interpretable) 규칙의 NI_MARGIN=0.05보다 느슨한 8%p. CNN은 탐색 후보 수가 훨씬 많아(수십만 개) 5%p로는 통과 후보가 지나치게 적어질 수 있어, 2차/탐색적 분석 위상에 맞게 완화된 문턱을 씀 — 두 값의 차이가 "1차는 확정적 검정, 2차는 탐색적 재확인"이라는 위상 차이를 보여줌 | aec_new_region_cnn_surrogate_mimic_gate.py:36 |
| `exact_p` / McNemar류 이항검정 | `binomtest(min(a,b), a+b, 0.5, two-sided)` | "강등 후 좋아진 사람 수(a)"와 "나빠진 사람 수(b)"가 우연한 50:50에서 벗어나는지 검정 | **대응표본 특성 반영.** 같은 환자 집합을 강등 전/후로 비교하는 대응(paired) 데이터이므로 독립표본 검정 대신 "바뀐 사람들 중 어느 방향이 더 많았나"만 보는 이항검정을 씀 — McNemar 검정의 정확판(exact) | aec_lock_smoothed_deesc_gate.py:263-266 |

---

## 7. CNN-mimic 변수 — 학습 설정 하나하나

`MimicConfig` — [main_aec_new_region_cnn_surrogate_mimic_gate.py:56-73](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L56-L73). 두 프리셋(balanced/guarded)이 어떤 값에서 갈리는지가 핵심.

| 필드 | balanced | guarded | 왜 이렇게 다른가 |
| --- | --- | --- | --- |
| `hidden` | 10 | 12 | CNN 필터 개수(모델 용량). guarded가 조금 더 큰 모델 — 다만 dropout도 함께 올라가 "더 크지만 더 강하게 억제된" 모델 |
| `dropout` | 0.20 | 0.30 | 학습 중 뉴런을 임의로 꺼서 과적합을 막는 비율. guarded는 표본이 적은 상황(강등 후보 수십 명)에서 더 강하게 억제 |
| `lr` (learning rate) | 8.0e-4 | 6.0e-4 | 한 번에 가중치를 얼마나 크게 갱신할지. guarded가 더 작아 "더 천천히, 조심스럽게" 학습 |
| `weight_decay` | 1.0e-3 | 2.0e-3 | 가중치 크기 자체에 벌점을 주는 정규화(L2) 강도. guarded가 2배 커서 극단적인 가중치가 나오지 않도록 더 조임 |
| `consensus_weight` | 0.65 | 0.85 | "4구간 중 2개 이상 동의"라는 전체 합의를 개별 구간 판정보다 얼마나 더 중시할지. guarded가 더 커서 "낱개보다 전체 합의를 존중"하는 성향이 강함 — §6의 2-of-4 패턴 선택 철학과 일치 |
| `non_cpos_weight` | 0.04 | 0.02 | 임상적으로 아직 양성도 아닌 환자에 대한 오차를 얼마나 무시할지(작을수록 더 무시). guarded가 더 작아 "위험군이 아닌 사람의 판정 오차는 무시하고 임상양성군 판정에만 집중하라"는 방향을 더 세게 밀어붙임 |
| `max_epochs` / `patience` | 180 / 22 | 180 / 22 | 두 설정 동일. 최대 180번 반복 학습하되, 검증 손실이 22번 연속 개선 안 되면 조기 종료 |

> **실제로 guarded가 이겼다**: §10.4 결과에서 두 최종 승자(`internal_locked`, `internal_external_audit`) 모두 `surrogate_mimic_guarded` 설정에서 나왔습니다. 표본 수가 많지 않은 문제에서는 "더 크지만 더 억제된" 모델이 더 안정적으로 일반화되었다는 뜻으로 읽을 수 있습니다.

### 기타 CNN 관련 변수

| 변수 | 값 | 무엇을 하는가 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- | --- |
| `SEEDS` | [20260701, 20260711] | CNN 학습을 2개의 서로 다른 난수 초기값으로 반복 | **안정성 확인.** 신경망은 초기 난수에 따라 결과가 조금씩 달라져 2개 시드×5-fold=10개 모델의 평균으로 안정적인 확률 추정치를 얻음. 날짜 형식을 시드로 쓴 것은 재현성 기록 목적으로, 숫자 자체에 통계적 의미는 없음 | …mimic_gate.py:34 |
| `Conv1d(kernel_size=5, 3)` | 1층 커널5, 2층 커널3 | 구간 안의 이웃 5개(→3개) 지점을 한 번에 훑는 필터 크기 | **구간 길이에 맞춘 국소창.** R1(길이12) 기준 커널5는 구간의 약 40%를 한 번에 보는 크기 — 전역과 너무 국소적인 것 사이의 절충. 2층째에 커널을 줄인 것은 1층에서 뭉뚱그려진 특징을 더 좁게 다시 다듬기 위함 | …mimic_gate.py:159-160 |
| `threshold_vectors` 격자 (약 1,253개) | 균일 스캔(13개) + 5⁴×2종(1250개, 중복 제외) | CNN이 뽑은 확률을 다시 "투표"로 바꾸는 R1~R4별 임계값 후보 | **4구간 각각 다른 민감도.** R1~R4가 서로 다른 서술자·다른 위험 방향을 갖고 있어(§5) 최적 임계값도 구간마다 다를 수 있음. 5단계 격자를 2종 쓴 것은 촘촘함과 계산량 사이의 절충 | …mimic_gate.py:426-437 |

---

## 8. "교사(teacher)" 값이 왜 확정본과 다른가

CNN이 흉내내야 할 정답으로 쓰는 `TEACHER_BRANCHES`와, 최종 확정된 `LOCKED_PRIMARY_BRANCHES`는 값이 다릅니다.

| 구간 | TEACHER_BRANCHES (CNN이 흉내낸 값) | LOCKED_PRIMARY_BRANCHES (최종 확정값) | 차이 |
| --- | --- | --- | --- |
| R1 | endpoint_delta, sign−1, w0.35 | endpoint_delta, sign−1, w0.50 | 서술자·부호 동일, width만 다름 |
| R2 | level_mean, sign−1, w0.35 | level_mean, sign−1, w0.70 | width가 크게 다름(0.35→0.70) |
| R3 | endpoint_delta, sign−1, w0.70 | linear_slope, sign+1, w0.35 | 서술자·부호 모두 다름 |
| R4 | linear_slope, sign−1, w0.35 | endpoint_delta, sign−1, w0.50 | 서술자 다름 |

> **왜 다른가 — 시간 순서로 추정**: 코드 주석(…mimic_gate.py:45-46)에 `TEACHER_BRANCHES`는 "portable surrogate audit winner", 즉 `aec_new_region_surrogate_combo_gate.py`를 실행해서 나온 **더 이전(또는 다른 계열) 결과물**을 그대로 복사한 것이라고 적혀 있습니다. 반면 `LOCKED_PRIMARY_BRANCHES`는 `main_aec_full_derivation_pipeline.py`가 비열등성 검정까지 통과시켜 최종 확정한 값입니다. 즉 CNN은 "그 당시 기준으로 최선이었던 규칙"을 배웠고, 그 뒤 규칙이 한 번 더 다듬어져 지금의 확정본이 나온 것 — 정확히 어느 시점의 `new4_combo_summary.json` 실행인지는 코드만으로 완전히 특정되지 않아 "확인되지 않음"으로 남겨둡니다.

| 변수 | 값 | 왜 이 값인가 | 근거 |
| --- | --- | --- | --- |
| `CNN_S90_INDEX` | 2 | 저장된 확률 배열의 3개 운영점[S80,S85,S90] 중 인덱스 2 = S90. 단순히 배열 순서상의 위치 | full_derivation…:161 |
| `CNN_BRANCH_THRESHOLDS` | [0.80, 0.60, 0.90, 0.60] | **원본 스크린샷 재현용.** 임의의 값이 아니라, 협업자가 공유한 원본 리포트 스크린샷(`outputs/MD/144838527.png`, "Secondary: CNN-mimic Gate")의 수치(S90 강등 40/52명, TP lost 2/1)를 그대로 재현하도록 역산·고정된 값. 한때 §10.4의 "새 승자" 값으로 바꿨다가, 재현 결과가 원본과 달라져(강등 58/72명) 다시 이 값으로 되돌린 이력이 있음 | full_derivation…:152-163, §11.1 수정이력 |
| `CNN_SELECTED_PATTERNS` | {"+---","---+","-+-+","++++"} | 위와 동일한 이유로 원본 스크린샷 재현에 맞춰 고정된 패턴 4개(1차 규칙의 5패턴과는 다른 별개 집합) | full_derivation…:163 |

---

## 9. old_python 폴더의 원조 파일들

### `old_python/aec_region_search_surrogate.py`
**역할**: R1~R4의 원조 탐색기 — §3의 COARSE/FINE_STEP, LENGTHS, lo=33/flank=16의 1차 출처.

coarse(8칸 간격)로 128포인트 전체를 훑고, 그 결과 상위 30개 윈도우 주변만 `fine_windows_from_top(flank=16)`으로 다시 촘촘히(4칸 간격) 훑는 2단계 구조를 이 파일이 처음 구현했습니다. `main_aec_full_derivation_pipeline.py`는 이 적응형 탐색을 매번 다시 계산하는 대신, "좋은 후보가 33번 이후에 몰려 있더라"는 경험적 결론만 `lo=33` 하드코딩으로 가져와 재현을 단순화했습니다.

### `old_python/aec_new_region_guided_cnn_gate.py`
**역할**: 최종 CNN-mimic(`DirectVoteMimicCnn`) 이전의 1세대 CNN 게이트.

`TrainConfig`에 `consensus_weight=0.45`, `low_guard_weight=5.0`이라는, 최종본(`MimicConfig`)에는 없는 필드가 있습니다. 최종본에서 consensus_weight가 0.65/0.85로 올라간 것과 비교하면, "전체 합의(2-of-4)를 얼마나 존중할지"에 대한 가중치가 반복 실험을 거치며 점점 더 커지는 방향으로 조정되어 왔음을 보여줍니다 — 즉 최종 설정값(§7)은 하루아침에 정해진 것이 아니라 이 1세대 실험의 결과를 보고 튜닝된 값입니다.

---

## 10. 용어 사전

- **AEC (Automatic Exposure Control)**: CT 촬영 시 스캐너가 부위별로 자동 조절하는 X선량(관전류) 신호. 조직을 직접 찍은 영상이 아니라, "이 부위는 X선이 얼마나 필요한가"를 스캐너가 판단한 궤적이 간접적으로 환자 체형을 반영한다는 것이 이 연구의 전제.
- **SMI (Skeletal Muscle Index)**: TAMA(총 복부 근육 면적으로 추정) ÷ 키(m)². 근육량을 키에 대해 표준화한 지표로, 이 값이 낮으면 근감소증(저-SMI) 위험군으로 분류.
- **de-escalation (위험강등)**: 임상 모델이 일단 "위험군(양성)"으로 분류한 사람 중, AEC 형태 신호로 다시 봤을 때 "사실 위험도가 낮아 보이는" 사람들을 골라 위험도를 낮춰주는 절차. 새로운 사람을 위험군에 추가하는 것이 아니라, 이미 양성인 사람 중 일부를 음성 쪽으로 재분류하는 것만 가능.
- **noninferiority(비열등성) 검정**: "완전히 손실이 없다(=0)"를 증명하는 대신, "손실이 있더라도 미리 정한 한도(여기서는 5%p)를 넘지 않는다"를 통계적으로 확인하는 검정 방식.
- **Clopper-Pearson 신뢰구간**: 이항비율에 대해 정규분포 근사를 쓰지 않고 정확한 분포(베타분포)로 계산하는 신뢰구간. 표본이 적을 때도 왜곡 없이 보수적으로 안전한 상한을 준다.
- **OOF (Out-Of-Fold)**: 5-fold 교차검증에서, 각 환자에 대해 "그 환자가 학습에 쓰이지 않은 모델"의 예측만 모은 것. 내부 데이터로 정직한 성능을 추정하기 위한 표준 기법.
- **z-표준화(z-score standardization)**: 어떤 값에서 기준 집단(주로 internal)의 평균을 빼고 표준편차로 나눠 "평균 0·표준편차 1"의 공통 척도로 바꾸는 절차. external 데이터에도 internal 기준을 그대로 적용해 정보 누수를 막는다.
- **distillation(지식 증류)**: 이미 정답을 아는 단순한 모델(교사, 손코딩 4-region 규칙)의 판단을 더 유연한 모델(학생, CNN)이 따라 하도록 학습시키는 기법.
- **consensus(합의) / soft_atleast2_prob**: 4개 구간(R1~R4) 중 최소 몇 개가 동시에 투표했는가를 신경망이 미분 가능한 형태로 학습할 수 있도록 확률적으로 근사한 것.
- **BCE (Binary Cross-Entropy)**: 0/1을 맞히는 이진분류 문제에서 표준적으로 쓰는 손실함수.
- **Dropout / Weight Decay**: 둘 다 신경망의 과적합을 막는 정규화 기법. Dropout은 학습 중 일부 뉴런을 임의로 끄고, Weight Decay는 가중치 값이 너무 커지지 않도록 벌점을 준다.
- **Conv1d / kernel / pooling**: 1차원 데이터(128개 지점의 곡선)에 쓰는 합성곱 신경망 구조. kernel은 한 번에 보는 이웃 지점의 폭, pooling은 구간 전체에서 대표값(평균/최대)만 뽑아 요약하는 과정.
- **Fisher 정확검정 / McNemar류 이항검정**: 두 그룹(또는 강등 전/후)의 사건 발생 비율 차이가 우연이 아닌지 확인하는 통계 검정. 표본이 작을 때도 정확한 값을 주는 정확검정(exact test) 계열을 일관되게 사용.
- **민감도(sensitivity) / 특이도(specificity) / 정확도(accuracy)**: 민감도=진짜 위험군을 놓치지 않는 비율, 특이도=진짜 안전군을 안전하다고 맞히는 비율, 정확도=전체 중 맞힌 비율.
- **4비트 패턴 코드 (pattern mask)**: R1~R4 4개 구간의 +/- 투표를 4자리 이진수(0~15)로 압축해 표현하는 방식.

---

근거 파일: [python/main_aec_full_derivation_pipeline.py](python/main_aec_full_derivation_pipeline.py), [python/main_aec_new_region_cnn_surrogate_mimic_gate.py](python/main_aec_new_region_cnn_surrogate_mimic_gate.py), [python/aec_lock_smoothed_deesc_gate.py](python/aec_lock_smoothed_deesc_gate.py), [python/aec_new_region_surrogate_combo_gate.py](python/aec_new_region_surrogate_combo_gate.py), [old_python/aec_region_search_surrogate.py](old_python/aec_region_search_surrogate.py), [old_python/aec_new_region_guided_cnn_gate.py](old_python/aec_new_region_guided_cnn_gate.py).

> `old_python/aec_new_region_surrogate_combo_gate_with_r0.py`(R0=1–44 구간을 추가로 시도했던 중간 실험판)는 최종 게이트에 채택되지 않은 코드였으므로 삭제했습니다.
