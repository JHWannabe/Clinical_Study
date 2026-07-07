# main_aec_new_region_cnn_surrogate_mimic_gate.py 코드 설명 (Top-Down)

이 문서는 [main_aec_new_region_cnn_surrogate_mimic_gate.py](python/main_aec_new_region_cnn_surrogate_mimic_gate.py)를 **가장 바깥(실행 진입점)에서부터 안쪽(구체적 계산 함수)으로** 내려가며 이해하기 위한 설명입니다. [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md)와 짝을 이루는 문서로, 같은 스타일(의존성 트리 → depth별 함수 설명 → 산출물 → 임상적 배경)로 작성했습니다. 통계·딥러닝 비전공자도 읽을 수 있도록 전문용어는 풀어서 설명했고, 코드/산출물 어디에도 없는 내용은 추측하지 않고 "확인되지 않음"이라고 명시했습니다.

---

## 0. 큰 그림: 이 파일은 무엇을 하는가

**한 줄 요약**: "4개 영역(R1~R4) 형태 게이트"를 사람이 손으로 정한 수식(교사, teacher) 대신, **CNN(합성곱 신경망)이 곡선의 형태를 직접 보고 그 교사 규칙의 판정을 흉내내도록(mimic) 학습**시킨 뒤, 그렇게 학습된 CNN의 점수로 새로 규칙을 다시 탐색해서 안전 기준을 통과하는 최종 규칙을 찾는 스크립트입니다.

이 프로젝트에는 같은 목표("임상적으로 위험군으로 분류된 환자 중 실제로는 위험도가 낮은 사람을 형태학적으로 다시 골라내기")를 이루는 **두 가지 서로 다른 방법**이 있습니다.

| | **4-region AEC Gate** (주 방법) | **CNN-mimic Gate** (이 파일, 보조/2차 방법) |
| --- | --- | --- |
| 대표 파일 | [main_plot_external_s90_core_1x3_mean_curves.py](python/main_plot_external_s90_core_1x3_mean_curves.py) | **aec_new_region_cnn_surrogate_mimic_gate.py** (이 문서) |
| 판정 방식 | 사람이 R1~R4마다 "어떤 서술자(level/slope 등)를 어떤 부호·폭·가중치로 볼지" 직접 정한 **닫힌 수식** | 각 R1~R4 구간을 **작은 1D CNN**이 통째로 보고 "이 구간 형태가 저위험처럼 보이는가"를 스스로 점수화 |
| 학습 여부 | 학습 없음 (수식 고정, 값만 대입) | 있음 (교사 규칙의 판정을 정답으로 삼아 CNN을 distillation 방식으로 학습) |
| 해석 가능성 | 매우 높음 (수식 자체가 설명) | CNN 내부는 상대적으로 블랙박스, 대신 "교사와 얼마나 일치하는가(agreement)"로 검증 |
| 이 문서에서의 위치 | 논문의 **주 결과(primary)** | 논문의 **2차/강건성 확인용 결과(secondary / robustness check)**로 활용 가능 |

즉 이 파일은 완전히 새로운 아이디어가 아니라, **"손으로 정한 4-region 규칙이 우연이 아니라 데이터가 실제로 그런 형태 신호를 담고 있어서 나온 것인지"를 CNN으로 교차 검증**하는 역할을 합니다. 만약 CNN이 교사 규칙을 잘 흉내내고, CNN 기반 규칙도 내부/외부 데이터 모두에서 비슷한 안전 기준을 통과한다면, "4-region 규칙이 특정 손코딩의 우연이 아니라 재현 가능한 형태 신호"라는 근거가 보강됩니다.

### 0.1 "교사(teacher)"란 정확히 무엇인가

이 파일의 `TEACHER_BRANCHES`/`TEACHER_PATTERNS` ([main_aec_new_region_cnn_surrogate_mimic_gate.py:47-53](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L47-L53))는 새로 만든 값이 아니라, **`aec_new_region_surrogate_combo_gate.py`를 실행해서 나온 이전 결과(`outputs/aec_new_region_surrogate_combo_gate/new4_combo_summary.json`)를 그대로 복사**한 것입니다 (코드 주석 [L45-46](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L45-L46)). 즉 "교사"는 사람이 아니라 **이미 확정된 손코딩 4-region 규칙 그 자체**이고, 이 CNN은 그 교사의 입출력(어떤 환자를 강등시키는지)을 재현하도록 학습됩니다. 이런 학습 방식을 머신러닝에서는 **"지식 증류(knowledge distillation)"** 라고 부르며 — 크고 자유로운 모델(여기서는 CNN)이 이미 정답을 알고 있는 더 단순한 규칙(교사)의 판단을 "따라 하도록" 학습시키는 기법입니다.

주의: 이 교사 패턴(`--+-,---+,++-+,--++,++++`)은 [main_plot_external_s90_core_1x3_mean_curves.py](python/main_plot_external_s90_core_1x3_mean_curves.py)가 실제로 사용하는 `SELECTED_PATTERNS = {"++--", "--+-", "---+", "+--+", "++++"}` 및 `BRANCHES`의 위험(width)/서술자 값과 **완전히 동일하지 않습니다** (일부 서술자·폭이 다름 — 예: R1 서술자는 둘 다 `endpoint_delta`로 같지만 R2는 여기서는 `width=0.35`, plot 스크립트는 `width=0.70` 등). 즉 이 파일의 교사는 plot 스크립트가 최종적으로 채택한 "lock된" 버전이 아니라, **그보다 한 단계 이전(또는 다른 계열)의 surrogate combo 탐색 결과**로 보입니다. 정확히 어느 버전인지는 `aec_new_region_surrogate_combo_gate.py`의 산출 이력을 봐야 하며, 이 파일 자체에는 "portable surrogate audit winner"라는 주석([L45](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L45))만 있습니다. 논문에 두 결과를 나란히 쓸 경우 이 차이를 각주로 밝히는 것이 안전합니다.

---

## 1. 실행에 실제로 필요한 파일 의존성 트리

```
aec_new_region_cnn_surrogate_mimic_gate.py   ← 실행 진입점
│
├─ aec_lock_smoothed_deesc_gate.py
│   │  가져다 쓰는 것: DATA_DIR, clinical_scores, deesc_metric_row, load_dataset, make_single_deesc
│   │
│   ├─ aec_conditional_value.py        (clinical_scores가 내부적으로 사용)
│   ├─ aec_universal_boundary_gate.py  (clinical_scores가 내부적으로 사용)
│   └─ aec128_mass_feature_combinations.py  (이 실행 경로에서는 미사용 — 죽은 import, plot_external 문서 §5 참고)
│
└─ aec_region_constrained_cnn_gate.py
       가져다 쓰는 것: make_channels, standardize_channels_train_apply, stratified_folds
       │
       └─ (자체적으로 다시) aec_lock_smoothed_deesc_gate.py — 그 파일 자신의 main()용, 이 실행 경로엔 영향 없음
```

**중요한 차이점 (plot_external 스크립트와 비교)**: `main_plot_external_s90_core_1x3_mean_curves.py`는 `region_descriptor_matrix`/`z_train_apply`를 **`aec_new_region_surrogate_combo_gate.py`에서 import**해서 씁니다. 반면 이 파일(`aec_new_region_cnn_surrogate_mimic_gate.py`)은 **같은 이름의 함수를 자기 파일 안에 별도로 다시 구현**해 놓았습니다 ([region_descriptor_matrix L107](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L107), [z_train_apply L135](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L135)). 두 구현은 로직상 동일해 보이지만, **import 관계가 아니라 코드 중복(duplicate)** 이므로, 향후 한쪽만 수정하면 두 파일의 특징 계산이 서로 달라질 수 있다는 점을 알아둘 필요가 있습니다.

또한 `RegionBranch`라는 이름의 클래스가 **두 파일에 각각 따로 존재**합니다 — `aec_region_constrained_cnn_gate.py:124`의 것과 이 파일 [L226](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L226)의 것은 **서로 다른 클래스**이며, 이 파일이 실제로 CNN 계산에 쓰는 것은 **이 파일 자신에 정의된 버전**입니다 (import되는 것은 `make_channels`/`standardize_channels_train_apply`/`stratified_folds` 3개 함수뿐, 클래스는 import하지 않음).

---

## 2. main()의 전체 흐름 (Depth 0)

[main()](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L702)의 흐름을 순서대로 보면:

1. g1090(internal)/sdata(external) 로드, 임상 로지스틱 회귀 점수·S80/S85/S90 임계값 계산 (`aec_lock_smoothed_deesc_gate.py` 재사용)
2. **교사(teacher) 특징 계산** → 교사 규칙 그대로(정확히) 적용했을 때 내부/외부 성능을 `exact_surrogate_teacher_details.csv`로 저장 — "CNN 없이 손코딩 교사 규칙만 썼을 때의 기준선"
3. CNN 입력 채널(곡선 자체 + 1차 기울기 + 2차 곡률, 3채널) 준비
4. **두 가지 CNN 설정(`surrogate_mimic_balanced`, `surrogate_mimic_guarded`)** 각각에 대해:
   - 2개 시드 × 5-fold 교차검증으로 CNN을 "교사 투표를 흉내내도록" 학습 (distillation)
   - internal out-of-fold 확률 + external 확률을 저장 (`*_probabilities.npz`, `*_training_log.csv`)
5. 학습된 CNN 확률에 대해, **임계값(threshold) 조합 × 패턴(pattern) 조합을 전수 탐색**해서 "이 CNN 점수로 어떤 규칙을 만들면 안전 기준을 통과하는가"를 찾음 (`*_same_rule_candidates.csv`)
6. 모든 후보를 합쳐 정렬 → internal 안전기준을 통과하는 규칙 중 1위(`internal_locked`), internal+external 모두 통과하는 규칙 중 1위(`internal_external_audit`) 두 가지 "최종 승자"를 선정
7. 각 승자에 대해 상세 지표 CSV, 3-패널 그래프 PNG, "CNN이 교사와 얼마나 일치하는가(agreement)" 표를 저장
8. 전체 설정·승자 정보를 `surrogate_mimic_summary.json`으로 저장, 콘솔에 요약 출력

이 문서의 2~10절은 이 흐름의 각 부분을 담당하는 함수를 순서대로 설명합니다.

---

## 2단계 (Depth 1): 이 파일이 직접 import하는 함수

### 2.1 `load_dataset`, `clinical_scores` — [aec_lock_smoothed_deesc_gate.py](python/aec_lock_smoothed_deesc_gate.py)

[main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md) §2.1~2.2와 완전히 동일한 함수입니다 (엑셀 로드 + Gaussian smoothing + 환자별 정규화 + SMI 기반 저-SMI 라벨 / 임상 로지스틱 회귀 OOF+external 점수 및 S80~S90 임계값). 자세한 설명은 그 문서를 참고하세요.

### 2.2 `deesc_metric_row(dataset, rule, features, op, y, cpos, deesc)` — [aec_lock_smoothed_deesc_gate.py:269](python/aec_lock_smoothed_deesc_gate.py#L269)

특정 de-escalation(위험강등) 규칙을 적용하기 **전/후**의 민감도·특이도·정확도·균형정확도 변화를 계산하고, 그 변화가 우연히 나온 게 아닌지 확인하는 통계 검정(이항근사 p값, McNemar류 p값, Fisher 정확검정)까지 한 행(row)으로 정리하는 함수입니다. 이 파일의 `evaluate_gate_detail`(§9.4)이 각 운영점(S80/S85/S90) × 데이터셋(internal/external)마다 이 함수를 호출해 상세표를 만듭니다.

### 2.3 `make_single_deesc(clinical_z, feature_z, th, width, lam)` — [aec_lock_smoothed_deesc_gate.py:320](python/aec_lock_smoothed_deesc_gate.py#L320)

임상 점수 하나 + 형태 특징 하나로 만드는 **단일 특징 de-escalation 게이트**입니다. "임상 점수가 임계값(th) 근처에 있을 때만 형태 특징의 영향력을 가우시안 창(width)으로 살짝 열어주고, 그 결과가 다시 임계값 아래로 내려가면 강등"하는 규칙 — plot_external 문서의 `gate_scores`/`compute_external_s90_gate`와 정확히 같은 수식입니다. 이 파일에서는 **교사 규칙의 "정답(투표)"을 만드는 데** 쓰입니다 (`branch_votes`, §4.3).

### 2.4 `make_channels`, `standardize_channels_train_apply`, `stratified_folds` — [aec_region_constrained_cnn_gate.py](python/aec_region_constrained_cnn_gate.py)

- **`make_channels(norm)`** ([L97](python/aec_region_constrained_cnn_gate.py#L97)): 정규화된 AEC 곡선 1개에서 **3개 채널**(① 곡선 자체, ② 1차 기울기, ③ 2차 곡률)을 만들고, 각각을 환자별로 다시 표준화(row-z)합니다. "환자별로 다시 표준화"하는 이유는 CNN이 **곡선의 절대적인 높낮이가 아니라 순수한 "모양(shape)"만 보도록 강제**하기 위함입니다 (코드 주석: "형태만 보고 절대 레벨은 못 보게 강제").
- **`standardize_channels_train_apply(xg, xs)`** ([L104](python/aec_region_constrained_cnn_gate.py#L104)): internal(g1090) 채널의 평균/표준편차로 internal·external 데이터를 모두 표준화 — "학습 데이터 기준으로만 정규화 기준을 정하고, 외부 데이터에는 그 기준을 그대로 적용"하는 원칙(정보 누수 방지)을 지킴.
- **`stratified_folds(y, seed)`** ([L191](python/aec_region_constrained_cnn_gate.py#L191)): 저-SMI 비율을 유지하며 5-fold 교차검증용 인덱스를 나눔.

---

## 3. 이 파일 자체에서 다시 구현하는 특징 계산

### 3.1 `REGIONS` — [L38-43](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L38-L43)

```python
REGIONS = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}
```

plot_external 스크립트와 동일한 4개 구간 정의(치골결합부 위쪽~간 상단 축 위의 4개 구간)를 그대로 재사용합니다.

### 3.2 `d1`, `d2` — [L95-106](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L95-L106)

곡선의 1차 차분(기울기)과 2차 차분(곡률, 기울기의 기울기)을 계산하는 유틸리티. plot_external 문서와 개념은 같지만, 이 파일 안에 **독립적으로 다시 구현**되어 있습니다.

### 3.3 `region_descriptor_matrix(norm)` — [L107-134](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L107-L134)

4개 구간마다 **12개의 형태 서술자**(level_mean/level_sd/endpoint_delta/linear_slope/slope_mean/slope_sd/abs_slope_mean/abs_slope_max/curv_mean/curv_sd/abs_curv_mean/abs_curv_max)를 계산해 특징 표(DataFrame)를 만듭니다. plot_external 문서 §2.3과 완전히 같은 개념이며, "교사 규칙이 원래 어떤 숫자로 만들어졌는지"를 재현하기 위한 용도입니다. **이 함수의 출력 자체가 CNN의 입력은 아닙니다** — CNN 입력은 §4.4의 `make_channels` 3채널이고, 이 서술자 표는 오직 "교사가 어떤 환자를 강등시켰는지" 정답(distillation target)을 재계산하는 데만 쓰입니다.

### 3.4 `z_train_apply(xg_df, xs_df)` — [L135-148](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L135-L148)

internal 데이터 기준 중앙값으로 결측 대치 → internal 평균/표준편차로 두 데이터셋 모두 z-표준화. 단순 유틸이며 값 선택 로직은 없습니다.

---

## 4. "교사가 무엇에 투표했는가" 정답(distillation target) 만들기

CNN을 학습시키려면 "정답"이 있어야 합니다. 여기서 정답은 사람이 붙인 라벨이 아니라, **손코딩 교사 규칙이 각 환자·각 운영점(S80/S85/S90)·각 구간(R1~R4)마다 "강등에 투표했는가/안 했는가"**입니다.

### 4.1 `teacher_features(g, s)` — [L149-166](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L149-L166)

`region_descriptor_matrix` + `z_train_apply`로 만든 표준화 특징 중, `TEACHER_BRANCHES`에 지정된 **4개 특징만** 골라 부호(sign)를 곱해서 꺼냅니다. 이것이 교사 규칙이 실제로 참조하는 4개의 숫자입니다.

### 4.2 `clinical_positive_matrix(score, thresholds)` — [L167-171](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L167-L171)

각 환자가 S80/S85/S90 3개 운영점 각각에서 "임상적으로 양성(위험군)"인지 여부를 (환자 수 × 3) 불리언 행렬로 만듦.

### 4.3 `branch_votes(feature_z, clinical_z, thresholds)` — [L172-187](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L172-L187)

`make_single_deesc`(§2.3)를 **모든 운영점 × 모든 교사 branch**에 대해 반복 호출해서, (환자 수 × 3개 운영점 × 4개 구간) 크기의 **0/1 투표 행렬**을 만듭니다. 이것이 CNN이 배워야 할 "정답"입니다. 즉 "이 환자를, 이 운영점 기준으로, R1 구간이 강등에 투표했는가?"라는 질문에 대한 교사의 답이 4×3×N개 만들어집니다.

### 4.4 패턴을 숫자로 표현하기 — `pattern_str`/`pattern_mask_to_text`/`mask_from_patterns`/`popcount` ([L188-214](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L188-L214))

4개 구간의 +/- 투표를 4비트 이진수(0~15)로 표현하는 관례입니다 (예: `++--` = R1,R2 투표함/R3,R4 투표 안 함). plot_external 문서의 `pattern_from_votes`와 같은 개념을 비트 연산으로 더 빠르게 구현한 버전이며, 뒤의 대규모 전수탐색(§7~8)에서 속도를 위해 이 방식을 씁니다.

---

## 5. CNN 모델 구조

### 5.1 왜 CNN인가 (비전공자를 위한 비유)

R1~R4 손코딩 규칙은 "이 구간의 평균이 얼마나 낮은가", "이 구간의 기울기가 얼마나 가파른가"처럼, **사람이 미리 정한 한두 개의 통계량**만 봅니다. CNN은 그런 통계량을 미리 정하지 않고, **구간 안의 128개 지점 중 어디가 어떤 모양이든 상관없이 "저위험처럼 보이는 패턴"을 곡선을 직접 훑어보며 스스로 찾아내는 작은 필터 뭉치**입니다. 사진에서 고양이 귀 모양을 찾아내는 이미지 인식 CNN과 원리는 같고, 여기서는 128개 점으로 이루어진 1차원 곡선(1D)에서 "구간 형태 패턴"을 찾도록 축소한 버전입니다.

### 5.2 `RegionBranch` — [L226-247](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L226-L247)

```python
class RegionBranch(nn.Module):
    def __init__(self, hidden, dropout):
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.BatchNorm1d(hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden * 2, 1)
    def forward(self, x):
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)
```

- **입력**: 특정 구간(R1이면 45~56번 지점)의 3채널(곡선/기울기/곡률) 조각
- **합성곱 2겹**: 구간 안의 이웃한 몇 개 지점을 한 번에 훑으면서(kernel_size=5, 3) "국소적인 형태 패턴"을 감지하는 필터들을 학습
- **풀링(pooling)**: 구간 전체에서 "평균적으로 어땠는지"(mean)와 "가장 두드러졌던 지점은 어땠는지"(max) 두 가지를 뽑아냄
- **선형 head**: 이 두 값을 받아 "이 구간이 저위험 형태에 얼마나 가까운가"를 나타내는 **점수 하나(스칼라)**로 압축

즉 R1~R4마다 이런 `RegionBranch`가 **하나씩, 총 4개** 만들어지고, 각자 자기 담당 구간만 봅니다 (다른 구간 정보는 섞이지 않음 — "구간별로 독립적인 형태 판단"이라는 손코딩 규칙의 설계 철학을 CNN 구조에도 그대로 반영).

### 5.3 `DirectVoteMimicCnn` — [L248-291](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L248-L291)

이 클래스가 이 스크립트의 핵심 모델입니다. 4개의 `RegionBranch`(방금 설명한 CNN)를 감싸서, **교사 규칙과 똑같은 수학적 구조**로 최종 점수를 계산하도록 설계되어 있습니다.

```python
def forward(self, x, clinical_z):
    morph = [branch(구간별 조각) for branch, 구간 in zip(self.branches, ...)]  # CNN이 뽑은 4개 형태 점수
    delta = clinical_z - self.thresholds          # 임상점수가 임계값에서 얼마나 떨어져 있는가
    boundary = exp(-0.5 * (delta / width)^2)      # 임계값 근처에서만 커지는 가우시안 창 (교사 규칙의 그것과 동일)
    cpos = (delta >= 0)                            # 임상적으로 이미 양성인가
    feats = [형태점수, 형태점수*boundary, delta, boundary, cpos]  # 5개 특징
    return (feats * head_weight).sum() + head_bias  # 선형 결합 → 최종 로짓(logit)
```

풀어서 설명하면:

1. **형태 점수(morph)는 CNN이 학습으로 알아낸 값**이고, 손코딩 규칙의 "level_mean/endpoint_delta" 같은 고정 공식이 CNN 출력으로 대체된 것입니다.
2. 하지만 그 형태 점수를 **어떻게 조합해서 최종 판정을 내릴지의 틀(임상 임계값 근처에서만 형태가 영향을 주도록 하는 가우시안 boundary, 임상 양성 여부 cpos)은 교사 규칙의 수식 구조를 그대로 가져와서 고정**했습니다.
3. `head_weight`의 초깃값 ([L258-266](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L258-L266))도 무작위가 아니라 **교사 규칙이 원래 가지고 있던 부호(방향)로 미리 맞춰놓고 시작**합니다 (예: `head_weight[:, 1] = -1.5`처럼 "형태점수 × boundary" 항의 부호를 교사와 같은 방향으로 미리 설정).

**핵심 아이디어**: "CNN은 오직 구간별 형태 점수를 뽑는 부분만 자유롭게 학습하고, 그 점수를 임상 정보와 결합하는 방식은 교사가 이미 검증한 수식 틀을 그대로 쓴다." 이렇게 하면 CNN이 완전히 자유롭게 학습할 때보다 **원래 임상적 논리(임계값 근처에서만 형태를 보정한다)를 벗어나지 않으면서도**, 손코딩으로는 못 잡아내던 미묘한 형태 패턴을 추가로 잡아낼 여지를 줍니다.

### 5.4 `soft_atleast2_prob(logits)` — [L215-225](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L215-L225)

교사 패턴들(`--+-`, `---+`, `++-+`, `--++`, `++++`)을 보면 공통점이 있습니다 — **4개 중 최소 2개 이상의 구간이 투표한 경우만** 강등 후보로 인정된다는 것(단일 구간만 튄 경우는 제외). 이 함수는 "4개 로짓 중 정확히 0개 또는 1개만 양성일 확률"을 1에서 빼서, **"2개 이상이 동시에 양성일 확률"을 매끄럽게(미분 가능하게) 근사**합니다. 신경망은 미분 가능한 함수로만 학습이 되기 때문에, "몇 개 이상"이라는 딱딱한 개수 규칙을 이렇게 확률적으로 부드럽게 바꿔서 손실 함수에 넣습니다.

---

## 6. 학습 손실 함수 — `loss_fn` [L292-302](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L292-L302)

```python
def loss_fn(logits, target, cpos_weight, pos_weight, cfg):
    bce = binary_cross_entropy_with_logits(logits, target)       # 구간별로 "교사와 같은 투표를 냈는가"
    weight = cpos_weight * (1 + (pos_weight - 1) * target)       # 임상양성 환자에게 더 큰 가중치, 드문 양성 투표에 더 큰 가중치
    branch_loss = weighted mean(bce * weight)
    prob2 = soft_atleast2_prob(logits)
    target2 = (target.sum(-1) >= 2)                               # 교사가 실제로 "2개 이상 투표"했는가
    consensus_loss = BCE(prob2, target2)                          # CNN도 "2개 이상 투표" 확률을 맞추도록
    return branch_loss + cfg.consensus_weight * consensus_loss
```

두 부분의 합입니다:

1. **`branch_loss`**: R1~R4 **개별 구간마다** CNN이 교사와 같은 투표(1 또는 0)를 냈는지를 이진 분류 손실(BCE)로 학습. `non_cpos_weight`(예: 0.04)로 "임상적으로 아직 양성도 아닌 환자"의 오차는 거의 무시하고, 정작 중요한 "임상 양성 환자에서의 판단"에 학습을 집중시킵니다. 또한 교사가 투표하는 경우 자체가 드물기 때문에(클래스 불균형), `pos_weight`로 드문 양성 쪽에 더 큰 벌점을 줍니다.
2. **`consensus_loss`**: 개별 구간 하나하나가 아니라, **"2개 이상 구간이 동시에 투표하는 전체적인 합의(consensus)"가 교사와 일치하는지**를 별도로 한 번 더 확인. `consensus_weight`(balanced=0.65, guarded=0.85)로 이 항의 중요도를 조절 — guarded 설정이 이 항을 더 중요하게 취급해 "낱개 구간보다 전체 합의를 더 존중"합니다.

즉 이 손실 함수는 "교사가 낸 최종 판정(합의)"과 "교사가 각 구간에서 낸 개별 투표" **두 가지 수준을 모두 흉내내도록** CNN을 학습시킵니다.

---

## 7. 학습 절차: 교차검증과 두 가지 설정

### 7.1 `MimicConfig`/`CONFIGS` — [L57-75](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L57-L75)

두 가지 CNN 학습 설정을 나란히 돌립니다.

| 설정 | hidden | dropout | consensus_weight | non_cpos_weight | 성격 |
| --- | --- | --- | --- | --- | --- |
| `surrogate_mimic_balanced` | 10 | 0.20 | 0.65 | 0.04 | 표준 설정 |
| `surrogate_mimic_guarded` | 12 | 0.30 | 0.85 | 0.02 | 더 큰 모델 + 더 강한 드롭아웃(과적합 방지) + 전체 합의를 더 중시 + 비양성군 오차를 더 무시 → 상대적으로 "더 보수적/신중한" 학습 |

이름 그대로 `guarded`(보호/신중) 설정이 실제로 두 최종 승자 규칙 모두에서 선택되었습니다 (§10 결과 참고) — 이는 표본 수가 많지 않은 상황(강등 후보가 수십 명 단위)에서 더 규제가 강한 설정이 더 안정적으로 일반화되었다는 뜻으로 해석할 수 있습니다.

### 7.2 `train_fold` — [L303-375](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L303-L375)

한 fold(교차검증의 한 조각)에 대해 `DirectVoteMimicCnn`을 학습시킵니다. 매 epoch마다 검증 손실(validation loss)을 확인해서, **가장 좋았던 시점의 모델 가중치를 저장해 두었다가 학습이 끝나면 그 시점으로 복원**합니다(조기종료, early stopping — 계속 학습시키면 오히려 학습 데이터에만 맞춰져 새 데이터에 못 맞는 "과적합"이 생기므로, 검증 성능이 더 좋아지지 않으면 멈추는 안전장치). 최대 180 epoch까지 학습하되, 22번 연속 개선이 없으면 중단합니다(`patience`).

### 7.3 `crossfit_config` — [L376-408](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L376-L408)

**2개의 랜덤 시드 × 5-fold = 총 10번**의 독립적인 학습을 반복합니다. 왜 이렇게 여러 번 반복할까요 — 신경망은 초기 난수(seed)나 데이터를 5등분하는 방식에 따라 결과가 조금씩 달라질 수 있는데, 이를 여러 번 반복해 평균을 내면 "어쩌다 한 번 운 좋게/나쁘게 나온 결과"가 아니라 **더 안정적인 확률 추정치**를 얻을 수 있기 때문입니다.

- **internal(g1090) 확률**: 각 환자가 5-fold 중 정확히 한 번은 "검증군"으로 들어가므로, 그 환자에 대해서는 **자신이 학습에 쓰이지 않은 모델의 예측**만 모아 "OOF(out-of-fold)" 확률을 만듦 → 이렇게 하면 internal 데이터로도 정직한(과적합되지 않은) 성능 추정이 가능
- **external(sdata) 확률**: 10개 모델(2시드×5fold) 각각의 예측을 평균

---

## 8. 학습된 CNN 확률을 다시 "규칙"으로 바꾸기

CNN은 (환자 × 운영점 × 구간)마다 0~1 사이의 확률만 출력합니다. 이 확률 자체는 "규칙"이 아니므로, 이걸 다시 "몇 이상이면 투표로 칠지(threshold)"와 "어떤 투표 패턴 조합을 강등으로 인정할지(pattern mask)"로 바꿔야 실제 의사결정 규칙이 됩니다.

### 8.1 `threshold_vectors()` — [L426-437](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L426-L437)

R1~R4 각 구간마다 다른 확률 임계값을 시도해볼 수 있도록 후보 임계값 조합을 만듭니다:
- 4개 구간에 **같은 값**을 쓰는 균일 스캔: 0.35~0.95를 0.05 간격으로 (약 13개)
- 4개 구간에 **서로 다른 값**을 허용하는 두 종류의 5×5×5×5 격자(각 625개, 총 1250개, 일부 중복 제거)

총 약 1,253개의 임계값 조합 후보가 생깁니다.

### 8.2 `codes_from_prob`/`votes_to_codes` — [L409-425](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L409-L425)

확률(또는 이미 확정된 0/1 투표)을 임계값과 비교해 4비트 패턴 코드(0~15)로 압축.

### 8.3 `rank_codes_internal` — [L501-521](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L501-L521)

16가지 가능한 패턴 코드 각각에 대해, internal 데이터에서 "이 패턴을 강등 기준으로 썼을 때 얼마나 안전하고 유리한가"를 점수 매겨 **상위 8개**만 추립니다(전수 조합을 다 시도하면 계산량이 너무 커지므로, 유망한 후보로 먼저 좁히는 사전 필터).

### 8.4 `candidate_masks(top_codes)` — [L522-545](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L522-L545)

실제로 시도해볼 "패턴 조합(마스크)" 후보 집합을 만듭니다: 교사 마스크, 단일 패턴 16개, "몇 개 이상 구간이 투표하면 인정"/"정확히 몇 개일 때만 인정" 류의 규칙, 그리고 위 상위 8개 패턴을 2~5개씩 묶은 조합들.

### 8.5 안전성 판정 — `fast_dataset_summary`/`dataset_pass`/`exact_loss_p` ([L438-500](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L438-L500))

각 (임계값 조합, 패턴 마스크) 후보에 대해 **4-region 손코딩 규칙과 정확히 같은 안전 기준**을 적용합니다 ([MIN_DEESC_N=10, MAX_SENS_LOSS=0.08, L35-36](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L35-L36)):

- 강등 대상이 최소 10명 이상일 것 (너무 적은 표본으로 결정하지 않기 위함)
- 민감도(진짜 저-SMI 환자를 놓치지 않는 능력) 손실이 8%포인트를 넘지 않을 것
- 그 손실이 우연일 확률(p값)이 5% 이상일 것 (즉 "손실이 있다고 통계적으로 확신할 수 없는 수준"이어야 통과 — 다르게 말하면, 실제로 위험한 사람을 놓쳤다는 근거가 통계적으로 뚜렷하지 않아야 안전하다고 인정)
- 특이도(가짜 양성을 걸러내는 능력) 증가폭이 양수일 것
- 정확도 증가폭이 양수일 것

이 5가지를 모두 만족해야 `internal_pass`/`external_pass`가 `True`가 됩니다 — **plot_external 문서 §9.4에서 설명한 "de-escalation이 안전하려면"의 기준을 CNN 버전에도 동일하게 적용**한 것입니다.

### 8.6 `search_gates` — [L600-646](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L600-L646)

한 CNN 설정에 대해 **1,253개 임계값 조합 × (설정별로 다른 개수의) 패턴 마스크**를 전수 평가합니다. 진행 상황을 콘솔과 `progress.json`에 주기적으로 기록합니다(대규모 계산이라 시간이 걸림 — 아래 §11 참고).

---

## 9. 최종 규칙 선정과 리포팅

### 9.1 승자 선정 로직 — `main()` 후반부 ([L758-782](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L758-L782))

두 설정의 모든 후보를 합친 뒤:

- **`internal_locked`**: internal 안전기준을 통과(`internal_pass`)하는 후보 중 **internal 선택 점수**(정확도 증가 + 가중치를 준 최소 정확도/특이도 증가 - 민감도 손실 - 평균 강등 사건율)가 가장 높은 규칙. **external 통과 여부는 따지지 않음** — "internal 데이터만 보고 골랐을 때 가장 좋은 규칙"
- **`internal_external_audit`**: 그중에서도 **external까지 통과**한 후보만 추려서, external 정확도 증가/특이도 증가가 가장 높은 규칙 — "외부 검증까지 통과한, 더 엄격한 기준의 승자"

논문에서는 일반적으로 **`internal_external_audit` 쪽이 외부 타당성(external validity) 근거로 더 설득력 있는 결과**입니다. `internal_locked`는 "internal에만 맞춰 고른 규칙이 external에서는 기준을 못 넘을 수도 있다"는 것을 보여주는 대조군으로 유용합니다 (실제로 아래 §10 결과에서 그렇게 나타났습니다).

### 9.2 `evaluate_gate_detail`/`detail_for_winner` — [L576-599](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L576-L599), [L647-655](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L647-L655)

선정된 승자 규칙을 S80/S85/S90 각 운영점 × internal/external 각 데이터셋에 다시 적용해서, `deesc_metric_row`(§2.2)로 상세 지표(민감도/특이도/정확도 변화 + 각종 p값)를 계산 → `*_winner_details.csv`.

### 9.3 `plot_detail` — [L656-678](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L656-L678)

승자 규칙의 "정확도 증가/특이도 증가/민감도 손실"을 S80→S85→S90 순서로, internal(실선)과 external(점선)을 겹쳐 그린 3-패널 그래프 → `*_winner_plot.png`.

### 9.4 `agreement_table` — [L679-701](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L679-L701)

이 스크립트의 이름값을 검증하는 표입니다 — **"CNN이 정말로 교사를 잘 흉내냈는가"**를 임상 양성 환자들에 한해 두 가지 방식으로 측정합니다:
- **`branch_vote_agreement_cpos`**: R1~R4 개별 구간 투표가 교사와 얼마나 일치하는가 (구간별 일치율의 평균)
- **`consensus_agreement_cpos`**: "2개 이상 구간 투표"라는 최종 합의 판정 자체가 교사와 얼마나 일치하는가

---

## 10. 실제 결과 (2026-07-06 실행분 기준)

이 스크립트를 한 번 실행한 결과가 `outputs/aec_new_region_cnn_surrogate_mimic_gate/`에 저장되어 있어, 그 수치를 정리합니다. (**주의**: 이 결과는 코드가 만들어낸 하나의 실행 결과이며, 시드/설정을 바꾸면 조금씩 달라질 수 있습니다. 아래는 "이 코드가 실제로 어떤 답을 냈는가"의 스냅샷입니다.)

### 10.1 교사 규칙 자체의 성능 (CNN 없이, 손코딩 그대로) — `exact_surrogate_teacher_details.csv`

S90 기준으로, 임상 양성 환자 중 교사 규칙이 강등시킨 인원과 사건율:

| 데이터셋 | 강등 인원(deesc_n) | 그중 실제 저-SMI(deesc_events) | 강등군 사건율 | 정확도 증가 | 특이도 증가 | 민감도 손실 |
| --- | --- | --- | --- | --- | --- | --- |
| g1090 (internal) | 44 | 5 | 11.4% | +3.1%p | +4.1%p | 3.9%p |
| sdata (external) | 68 | 5 | 7.4% | +2.2%p | +6.3%p | 3.5%p |

→ 손코딩 교사 규칙 자체가 이미 internal/external 모두에서 "강등군의 실제 위험도가 낮다"는 방향으로 일관됨을 확인.

### 10.2 CNN 학습 결과 — 두 설정 모두 완주

`surrogate_mimic_balanced`/`surrogate_mimic_guarded` 각각 2시드×5fold=10개 모델이 모두 정상적으로 조기종료되며 학습 완료(`*_training_log.csv`). 검증 손실은 대체로 0.33~0.57 범위, 최적 epoch은 fold마다 32~179 epoch 사이에서 다양하게 나타남(즉 fold별로 필요한 학습량이 상당히 다름 — 데이터가 많지 않은 상황에서 흔한 현상).

### 10.3 탐색 규모

두 설정 합쳐 총 **582,485개의 (임계값, 패턴) 후보 규칙**을 탐색. 이 중 internal 안전기준을 통과한 것이 **211,260개(약 36%)**, internal+external 모두 통과한 것이 **164,541개(약 28%)**. (숫자가 매우 커 보이지만, 대부분은 서로 거의 같은 임계값을 조금씩만 바꾼 유사 규칙들이 중복 집계된 것입니다 — "안전 기준을 통과하는 규칙이 많다"는 것 자체가 "이 CNN 신호가 꽤 안정적으로 안전 조건을 만족시킨다"는 뜻으로 읽는 것이 맞고, "28만개의 서로 다른 독립적 발견"으로 해석하면 안 됩니다.)

### 10.4 두 승자 규칙 — 요약 ( `surrogate_mimic_summary.json` )

| | **internal_locked** | **internal_external_audit** |
| --- | --- | --- |
| 사용 설정 | `surrogate_mimic_guarded` | `surrogate_mimic_guarded` |
| 임계값 (R1,R2,R3,R4) | 0.65, 0.45, 0.75, 0.45 | 0.60, 0.80, 0.50, 0.50 |
| 인정 패턴 | `-+--,++--,--+-,---+,-+++` | `---+,+--+,-+-+,--++,++++` |
| internal 통과 | ✅ | ✅ |
| **external 통과** | **❌ (탈락)** | **✅** |
| internal 정확도 증가(평균) | +4.6%p | +3.6%p |
| external 정확도 증가(평균) | +4.8%p | **+6.7%p** |
| internal 특이도 증가(평균) | +5.7%p | +4.5%p |
| external 특이도 증가(평균) | +6.5%p | **+8.3%p** |
| external 최대 민감도 손실 | **7.1%p (0.08 한도에 근접)** | 2.8%p (여유 있음) |
| external 최소 유의성 p | **0.00195 (기준 0.05 미달 → 탈락 사유)** | 0.125 (기준 통과) |

**해석**: `internal_locked`는 internal 데이터만 보고 골랐더니 R2 임계값이 낮게(0.45) 잡히면서, external에서는 S80 운영점의 민감도 손실이 우연이라고 보기 어려운 수준(p=0.00195 < 0.05)까지 커져 안전기준을 통과하지 못했습니다. 반면 `internal_external_audit`는 R2 임계값을 훨씬 엄격하게(0.80) 잡은 덕분에 external에서도 안전기준을 여유 있게 통과했고, 오히려 external에서 더 큰 정확도/특이도 개선을 보였습니다. **이 대조는 "internal 데이터만으로 규칙을 고르면 외부 데이터에서 실패할 수 있다"는, 논문에서 강조하기 좋은 방법론적 근거**가 됩니다 (내부 검증만으로는 부족하고, 반드시 외부 데이터로 재확인해야 한다는 근거).

### 10.5 "CNN이 교사를 얼마나 잘 흉내냈는가" — agreement (`internal_external_audit_agreement.csv`)

| 데이터셋 | 운영점 | 구간별 투표 일치율 | 최종 합의(2-of-4) 일치율 |
| --- | --- | --- | --- |
| internal | S80/S85/S90 | 95.1~95.6% | 93.2~93.7% |
| external | S80/S85/S90 | 92.9~94.7% | 92.3~93.8% |

→ CNN이 정답(교사 판정)의 **약 93~96%를 그대로 재현**했습니다. 즉 CNN은 "교사를 거의 그대로 따라 하되, 나머지 4~7%의 경우에서만 다르게 판단"했고, 최종적으로 그 차이가 오히려 external 안전성 지표를 개선하는 방향으로 작용했습니다. 이는 "CNN이 손코딩 규칙을 대체로 재현하면서도, 손코딩이 놓쳤을 수 있는 미묘한 형태 신호를 일부 추가로 포착했다"는 근거로 해석할 수 있습니다(단, 이 해석은 이 한 번의 실행 결과에 기반한 것이며, 반복 실행/다른 시드로 재확인이 필요합니다 — 아래 §12 한계 참고).

---

## 11. 산출물(output) 파일 전체 정리

`OUT_DIR = outputs/aec_new_region_cnn_surrogate_mimic_gate/` ([L30](python/main_aec_new_region_cnn_surrogate_mimic_gate.py#L30))에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `progress.json` | 실행 중 진행 상황(단계, 완료된 fold/threshold 수, 예상 남은 시간) — 실행이 끝나면 `"stage": "done"`으로 표시 |
| `exact_surrogate_teacher_details.csv` | CNN 없이 손코딩 교사 규칙 그대로 적용했을 때의 성능(§10.1) |
| `surrogate_mimic_{balanced,guarded}_probabilities.npz` | 학습된 CNN이 출력한 확률. `prob_g`/`prob_s` 배열의 형태는 (환자 수 × 3개 운영점[S80,S85,S90] × 4개 구간[R1~R4]) — 이 배열이 뒤이은 모든 규칙 탐색의 원재료 |
| `surrogate_mimic_{balanced,guarded}_training_log.csv` / `surrogate_mimic_training_log.csv` | 시드×fold별 최적 epoch/검증손실/양성 가중치 기록 |
| `surrogate_mimic_{balanced,guarded}_same_rule_candidates.csv` | 각 설정에서 탐색한 모든 (임계값,패턴) 후보와 그 성능 — 매우 큼(1억 3천만 바이트대), 재현/재분석용이며 논문에 직접 넣을 표는 아님 |
| `surrogate_mimic_same_rule_all_candidates.csv` | 위 두 설정을 합친 전체 후보 (2.6억 바이트) |
| `surrogate_mimic_internal_passing_ranked.csv` / `surrogate_mimic_internal_external_passing_ranked.csv` | 안전기준을 통과한 후보만 골라 순위 매긴 표 |
| `internal_locked_winner_details.csv` / `internal_external_audit_winner_details.csv` | 두 승자 규칙의 S80/S85/S90 × internal/external 상세 지표 (§10.4 표의 출처) |
| `internal_locked_winner_plot.png` / `internal_external_audit_winner_plot.png` | 두 승자 규칙의 3-패널(정확도/특이도/민감도) 그래프 |
| `internal_locked_agreement.csv` / `internal_external_audit_agreement.csv` | CNN-교사 일치도 표 (§10.5 출처) |
| `surrogate_mimic_summary.json` | 모든 설정값, 교사 정의, 두 승자의 전체 지표를 담은 최종 요약 — **가장 먼저 열어봐야 할 파일** |

### 11.1 다른 파일과의 연결 — 경로 버그 수정 + 원본(MD) 재현 확인 (2026-07-07) ✅

프로젝트에는 이 CNN-mimic 결과를 "2차 분석(secondary)"으로 가져다 쓰는 통합 스크립트 [python/main_aec_full_derivation_pipeline.py](python/main_aec_full_derivation_pipeline.py)가 있습니다 (아래 절차를 확인하던 시점에는 `old_python/aec_full_derivation_pipeline.py`였으나, 이후 활성 폴더 `python/`으로 옮겨지고 `main_` 접두사가 붙었습니다 — 이 파일 자체의 top-down 설명은 [main_aec_full_derivation_pipeline.md](main_aec_full_derivation_pipeline.md) 참고). 이 파일의 `CNN_BRANCH_THRESHOLDS`/`CNN_SELECTED_PATTERNS`/`CNN_PROBABILITY_NPZ`가 실제로 무엇을 재현해야 하는지 확인하는 과정에서 아래와 같이 정리되었습니다.

**기준이 된 원본**: `outputs/MD/144838527.png` — 협업자가 KakaoTalk으로 공유한 "Secondary: CNN-mimic Gate" 결과 스크린샷("MD 원본"). S90 기준 목표값은 **de-escalated n: Internal 40 / External 52, TP lost: 2 / 1, De-escalated 사건율: 2/40=5.0% / 1/52=1.9%**.

**경로 버그 1건 수정**: `PROJECT_ROOT = SCRIPT_PATH.parents[2]`(Desktop 폴더) 기준으로 `DATA_DIR = PROJECT_ROOT / "work" / "data_cache"`처럼 `work`를 붙이는데, `CNN_PROBABILITY_NPZ`만 `PROJECT_ROOT / "outputs" / ...`로 `work`가 빠져 있어 실제 파일(`work/outputs/...`)을 절대 찾지 못했습니다. 이 때문에 임계값이 무엇이든 `compute_secondary_cnn_mimic()`이 항상 "파일 없음"으로 조용히 `None`을 반환해 CNN 2차 분석 자체가 전혀 실행되지 않고 있었습니다(에러 없이 `secondary_cnn_included: false`로 넘어감). → `PROJECT_ROOT / "work" / "outputs" / ...`로 수정.

**임계값/패턴/확률파일은 원래 값이 맞았음** (⚠️ 한 차례 잘못 고쳤다가 되돌림): 처음에는 `CNN_BRANCH_THRESHOLDS = [0.80, 0.60, 0.90, 0.60]` / `CNN_SELECTED_PATTERNS = {"+---","---+","-+-+","++++"}`가 §10.4의 두 "승자"(`internal_locked` 0.65/0.45/0.75/0.45, `internal_external_audit` 0.60/0.80/0.50/0.50) 중 어느 것과도 일치하지 않길래 §10.4 값으로 갱신했었으나, 이렇게 하면 S90 de-escalated n이 internal 58 / external 72로 나와 **원본 스크린샷(40/52)과 전혀 다른 결과**가 됩니다. `outputs/MD/144838527.png`를 직접 대조한 결과, **원래 하드코딩되어 있던 값(0.80/0.60/0.90/0.60, 4패턴, `surrogate_mimic_balanced` 확률)이 원본 스크린샷을 재현하는 올바른 값**이었습니다 — 즉 §10.4의 두 승자는 이 코드가 나중에 별도로 돌린 **더 큰 규모의 전수탐색(582,485개 후보)에서 새로 뽑은, 원본과는 다른(더 많이 강등시키는) 규칙**이고, 원본 리포트 값을 재현하려면 이 새 승자값을 쓰면 안 됩니다. 원래 값으로 되돌리고, 경로 버그만 수정했습니다.

**재현 확인 결과** (`compute_secondary_cnn_mimic()` 직접 호출, threshold=[0.80,0.60,0.90,0.60], patterns={+---,---+,-+-+,++++}, `surrogate_mimic_balanced` 확률):

| | Internal | External |
| --- | --- | --- |
| De-escalated n (목표: 40 / 52) | **40** (정확히 일치) | 51 (목표 52, 1명 차이) |
| De-escalated events (목표: 2 / 1) | **2** (정확히 일치) | **1** (정확히 일치) |
| TP lost (목표: 2 / 1) | **2** (정확히 일치) | **1** (정확히 일치) |

Internal은 완전히 일치, External은 40/52 중 52가 51로 1명 차이만 납니다(사건 수·TP lost는 정확히 일치). 이 정도 오차는 CNN 학습이 완전히 결정론적이지 않아(이번 세션에서 `.npz` 확률 파일이 다시 학습되어 저장됨 — 원본 스크린샷을 만들었을 때의 가중치와 100% 동일하지 않을 수 있음) 경계값 근처 환자 1명이 다르게 갈렸을 가능성이 높고, **설정(임계값/패턴/확률파일 종류) 자체는 올바르다**고 판단합니다.

**남은 참고사항**: `python/main_aec_full_derivation_pipeline.py`는 이 CNN-mimic 스크립트와 이름·목적이 다른 별도의 통합 파이프라인이므로, 논문에 실제로 쓸 1차 CNN-mimic 결과는 여전히 §10의 `surrogate_mimic_summary.json`(현재 활성 스크립트 `python/main_aec_new_region_cnn_surrogate_mimic_gate.py`의 산출물, §10.4의 새 승자값)을 출처로 삼고, `outputs/MD/144838527.png` 원본과 비교/재현이 필요할 때만 이 절의 원래 값(0.80/0.60/0.90/0.60, balanced)을 참고하십시오. 두 값 집합(§10.4 새 승자 vs 이 절의 원본 재현값)은 **서로 다른 목적의 서로 다른 규칙**이니 섞어 쓰지 않아야 합니다.

---

## 12. 임상적 배경 및 두 방법의 관계 (논문 작성을 위한 정리)

### 12.1 이 파일이 답하려는 임상적 질문

plot_external 문서 §9.4에서 설명한 것과 같은 임상적 문제(임상변수만으로는 민감도를 90%로 잡으면 위양성이 너무 많다 → AEC 형태 신호로 그중 진짜 저위험군을 다시 골라내자)에 대해, 이 파일은 **"그 형태 신호가 사람이 짠 4개의 손코딩 공식에만 의존하는 것인지, 아니면 CNN처럼 더 유연한 방법으로도 재현되는 진짜 신호인지"**를 검증합니다.

### 12.2 "surrogate(대리)"라는 이름의 의미

파일명과 변수명에 반복되는 "surrogate"는 통계에서 **"진짜 정답 대신 쓰는 대리 지표/모델"**을 뜻합니다. 여기서는 두 겹의 대리 관계가 있습니다:
1. AEC 곡선 자체가 이미 "환자 조직/근육량의 대리 신호"입니다 (스캐너의 자동 노출 조절값이지, 조직을 직접 잰 값이 아님 — plot_external 문서 §9.1 참고).
2. 이 CNN은 다시 "손코딩 4-region 규칙(교사)의 대리(surrogate)"로 학습됩니다 — CNN이 직접 저-SMI를 예측하도록 학습된 것이 아니라, **이미 검증된 규칙의 판정을 재현하도록** 학습되었다는 점이 중요합니다. 즉 이 CNN의 목적은 "더 정확한 새 모델을 만드는 것"이 아니라 "기존 손코딩 규칙이 재현 가능한 형태 신호에 기반하고 있음을 독립적인 방법으로 교차 확인하는 것"입니다.

### 12.3 두 방법을 논문에 함께 쓸 때 권장하는 서술 순서

1. **주 결과**: 4-region 손코딩 게이트(`main_plot_external_s90_core_1x3_mean_curves.py`) — 해석 가능하고 사전 등록(lock) 절차를 거친 결과를 1차 결과로 제시
2. **강건성/2차 확인**: CNN-mimic gate(이 파일) — "손코딩 규칙이 임의적인 것이 아니라, 서로 다른 방법론(학습 기반 CNN)으로도 유사한 방향의 안전한 강등 규칙이 재현된다"는 것을 보여주는 민감도 분석(sensitivity analysis)/강건성 확인(robustness check)으로 위치시키는 것을 권장
3. 이때 §10.5의 **"CNN이 교사 판정의 93~96%를 재현했다"** 는 수치와, §10.4의 **"internal만으로 고르면 external에서 실패할 수 있다(internal_locked vs internal_external_audit)"** 는 대조가 두 방법의 관계를 설명하는 핵심 근거가 됩니다.

### 12.4 이 문서에서 다루지 않은, 확인이 더 필요한 부분

- `TEACHER_BRANCHES`/`TEACHER_PATTERNS`가 정확히 `aec_new_region_surrogate_combo_gate.py`의 어느 실행/버전에서 나온 값인지는 이 파일 코드만으로는 완전히 특정되지 않습니다(§0.1 참고). 논문에 정확한 계보를 밝히려면 `aec_new_region_surrogate_combo_gate.py`의 산출 이력(`outputs/aec_new_region_surrogate_combo_gate/new4_combo_summary.json` 등)을 직접 대조해야 합니다.
- §11.1의 경로/상수 불일치는 반드시 해결하거나, 적어도 "이번 논문에 쓴 CNN-mimic 결과가 어느 산출물 파일에서 나왔는지"를 방법론 섹션에 명시적으로 적어두는 것이 안전합니다.
- 이 실행은 1회 실행 결과입니다. 재현성 확인을 위해 다른 시드로 재실행했을 때도 §10.4의 승자 임계값/패턴이 크게 달라지지 않는지 확인해 보는 것을 권장합니다(코드 자체는 `SEEDS = [20260701, 20260711]`로 이미 2개 시드를 쓰고 있지만, 이는 "같은 실행 안에서의 평균"이지 "전체 실행을 다른 초기조건으로 반복"한 것은 아닙니다).
