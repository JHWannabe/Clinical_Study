# main_plot_internal_s90_core_1x3_mean_curves.py 코드 설명 (Top-Down)

이 문서는 [python/main_plot_internal_s90_core_1x3_mean_curves.py](python/main_plot_internal_s90_core_1x3_mean_curves.py)를 **가장 바깥(실행 진입점)에서부터 안쪽(구체적 계산 함수)으로** 내려가며 이해하기 위한 설명입니다. 이 파일은 [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md)에서 설명한 external 스크립트의 **internal(g1090) 대응판**이므로, 겹치는 내용은 중복 설명하지 않고 "무엇이 같고 무엇이 다른지"에 집중합니다. 마지막 §11에 두 파일을 줄 단위로 대조해 **실제로 데이터 선택 외에 다른 차이가 없는지 검증한 결과**를 정리했습니다.

---

## 0. 큰 그림: 이 파일의 목적과 파일 의존성 트리

**목적**: external 스크립트가 g1090(internal)에서 고정("lock")한 S90 임계값과 R1~R4 게이트를 sdata(외부)에만 적용했다면, 이 스크립트는 **동일한 정의의 게이트를 g1090 자기 자신(out-of-fold 임상점수 + 훈련측 형태특징)에 적용**해서 "내부 기준에서는 이 그림이 어떻게 보이는가"를 같은 형식(1x3 / R4 tangent / 2x3 mirror)으로 재현하는 스크립트입니다.

**파일 의존성 트리는 external과 완전히 동일**합니다 (import 문이 한 글자도 다르지 않음 — §11 참고):

```text
main_plot_internal_s90_core_1x3_mean_curves.py   ← 실행 진입점
│
├─ aec_lock_smoothed_deesc_gate.py      (DATA_DIR, clinical_scores, load_dataset)
│   ├─ aec_conditional_value.py
│   ├─ aec_universal_boundary_gate.py
│   └─ aec128_mass_feature_combinations.py   (이 실행 경로에서는 미사용)
│
└─ aec_new_region_surrogate_combo_gate.py    (region_descriptor_matrix, z_train_apply)
       └─ (자체적으로 다시) aec_lock_smoothed_deesc_gate.py
```

각 파일의 역할은 external 문서 [§2~§7](main_plot_external_s90_core_1x3_mean_curves.md#2단계-depth-1-이-파일이-직접-import하는-4개-함수)에 이미 정리되어 있으므로 이 문서에서는 반복하지 않습니다. 이 문서가 추가로 설명할 것은 **같은 함수를 호출하면서도 "어느 그룹의 값"을 넘기는지가 internal/external에서 어떻게 갈리는가**입니다.

---

## 1단계 (Depth 0): main()의 흐름과 external과의 핵심 차이

[main()](python/main_plot_internal_s90_core_1x3_mean_curves.py#L285)의 흐름은 external과 단계 구성이 동일합니다:

1. g1090/sdata를 **둘 다** 로드한다 (`s`는 여전히 로드됨 — `clinical_scores(g, s)`가 임상 모델을 학습할 때 internal 5-fold OOF와 external 예측을 동시에 계산하는 함수라서, s 없이는 `clinical_scores` 자체가 성립하지 않음. 하지만 아래 그림/통계에 실제로 쓰이는 것은 `g`뿐).
2. `compute_internal_s90_gate(g, s)` 호출 → **g1090 자신의** "임상 양성 중 AEC 양성/음성"을 나눔.
3. PNG #1 (1x3), PNG #2 (1x3 + R4 tangent), PNG #3 (2x3 mirror), CSV, MIRROR_CSV, JSON — external과 동일한 6종 산출물을 `internal_*` 접두사로 저장.

**핵심 차이 — 어떤 배열을 그리는가**:

| | external | internal |
| --- | --- | --- |
| 그래프에 그리는 곡선 | `norm = s["norm"]` (sdata) | `norm = g["norm"]` (g1090) |
| 사건 라벨 | `y = s["y"]` | `y = g["y"]` |
| 게이트에 넣는 형태특징 z-score | `xs[:, idx]` (external 쪽) | `xg[:, idx]` (internal/훈련 쪽) |
| 게이트 임계값과 비교하는 임상 z-score | `c_s` | `c_g` |
| `clinical_pos` 정의 | `c_s >= threshold` | `c_g >= threshold` |

즉 `threshold`(S90 임계값 자체)와 `BRANCHES`/`SELECTED_PATTERNS`(게이트 정의)는 g1090에서 고정된 같은 값을 그대로 재사용하고, **"그 게이트를 누구에게 적용해서 보여주는가"만 s→g로 바뀝니다.**

---

## 2단계 (Depth 1): `compute_internal_s90_gate` vs `compute_external_s90_gate`

두 함수는 external 문서 [§2.2](main_plot_external_s90_core_1x3_mean_curves.md#22-clinical_scoresg-s--aec_lock_smoothed_deesc_gatepyl134)에서 설명한 `clinical_scores(g, s)`를 **똑같이 호출**해서 `xg, xs, c_g, c_s, thresholds`를 얻습니다. 차이는 그다음부터입니다 — external은 `xs`/`c_s`(4번째·5번째 반환값 중 s 쪽)를 골라 쓰고, internal은 `xg`/`c_g`를 골라 씁니다. 게이트 계산 로직(`gate_scores`, `pattern_from_votes`, `SELECTED_PATTERNS`와의 비교) 자체는 완전히 동일한 코드입니다.

`region_descriptor_matrix`/`z_train_apply`([아래 §참고](main_plot_external_s90_core_1x3_mean_curves.md#23-region_descriptor_matrixnorm--aec_new_region_surrogate_combo_gatepyl90))도 매 실행마다 `fg`(g1090)와 `fs`(sdata) 둘 다 계산하지만, internal 스크립트는 그중 `fg`/`xg` 쪽만 최종적으로 사용합니다.

---

## 3~9단계: external 문서와 동일 (재사용)

- **aec_conditional_value.py / aec_universal_boundary_gate.py / aec128_mass_feature_combinations.py / aec_new_region_surrogate_combo_gate.py 자신의 main() / aec_full_derivation_pipeline.py / 임상적 배경(AEC 정의, SMI/TAMA 컷오프, S80~S90의 의미, de-escalation 동기, R1~R4의 비-해부학적 성격)** — 이 모든 내용은 import 관계와 임상적 근거 모두 external과 100% 동일하므로, [main_plot_external_s90_core_1x3_mean_curves.md §3~§9](main_plot_external_s90_core_1x3_mean_curves.md)를 그대로 참고하면 됩니다. 이 문서에서 다시 옮겨 적지 않습니다.

---

## 10. 실제 산출 결과값 정리 (`outputs/aec_1x3_core_mean_curves/internal_*`)

### 10.1 임상양성 내 AEC 조건부 분리 (`internal_s90_core_1x3_summary.json` → `low_smi_conditional`)

| | AEC+ (강등 후보) | AEC- (유지) |
| --- | --- | --- |
| 인원(n) | 53 | 518 |
| 실제 저-SMI 사건 수 | 2 | 115 |
| 사건율 | **3.8%** | **22.2%** |
| Fisher 정확검정 p | **5.53×10⁻⁴** | (동일 검정) |

→ g1090(internal) 안에서도, S90 임상양성 571명 중 AEC 게이트가 골라낸 53명의 실제 저-SMI 유병률은 3.8%에 불과한 반면 나머지 518명은 22.2% — external(3.6% vs 26.9%, p=2.30×10⁻⁵)과 **방향과 크기가 일관됩니다.** 이는 g1090에서 먼저 확정(lock)된 규칙이므로 당연한 결과이지만("훈련 데이터에 다시 적용"), 같은 형식의 그림으로 재현해 두면 external 결과와 나란히 비교하기 쉽습니다.

### 10.2 브랜치별 투표 통과 인원 (`branches[].internal_vote_positive_n`, n=1090)

| 구간 | 서술자 | 투표 인원 | 비율 |
| --- | --- | --- | --- |
| R1 (45-56) | endpoint_delta | 519 | 47.6% |
| R2 (57-80) | level_mean | 525 | 48.2% |
| R3 (97-128) | linear_slope | 524 | 48.1% |
| R4 (117-128) | endpoint_delta | 532 | 48.8% |

### 10.3 세 가지 대비의 영역별 평균곡선 차이 (`internal_s90_core_1x3_mean_curve_summary.csv`)

| 대비 | 전체 평균 절대차 | R1 | R2 | R3 | R4 |
| --- | --- | --- | --- | --- | --- |
| Clinical+ vs Clinical− | 0.0272 | −0.0037 | −0.0439 | +0.0067 | +0.0072 |
| Low SMI+ vs Non-low SMI | 0.0175 | −0.0074 | −0.0291 | +0.0202 | +0.0343 |
| Clinical+/AEC− vs Clinical+/AEC+ | 0.0120 | −0.0134 | −0.0059 | +0.0221 | +0.0270 |

external과 마찬가지로 R3·R4(후반부)는 항상 "+"(높음), R1·R2(전반부)는 항상 "−" 방향 — 같은 형태 차이 패턴이 internal에서도 재현됩니다.

### 10.4 산출물 파일 한눈에 보기

| 파일 | 내용 |
| --- | --- |
| `internal_s90_core_1x3_mean_curves.png` | 1x3 그래프 (A 임상양성/음성, B 실제사건, C 임상양성 내 AEC 분리) — 모두 g1090 기준 |
| `internal_s90_core_1x3_mean_curves_with_r4_tangent.png` | 위 그래프 + R4 구간 적합 직선 주석 |
| `internal_s90_core_2x3_mean_and_mirror_deviation.png` | 3대비 각각의 평균곡선(위) + 기준곡선 대비 절대편차 미러 그래프(아래) |
| `internal_s90_core_1x3_mean_curve_summary.csv` | §10.3의 출처 |
| `internal_s90_core_2x3_mirror_deviation_summary.csv` | 미러 그래프의 영역별 평균/최대 절대편차 |
| `internal_s90_core_1x3_summary.json` | 게이트 정의 + §10.1/10.2의 출처 |

---

## 11. 검증: external → internal, 정말 "데이터만" 바뀌었는가

`diff -u main_plot_external_s90_core_1x3_mean_curves.py main_plot_internal_s90_core_1x3_mean_curves.py`로 두 파일을 줄 단위 대조한 결과, 변경 사항을 세 종류로 분류할 수 있습니다.

### 11.1 의도된 "데이터/대상 전환" 차이 (본질적 차이, 정상)

| 위치 | external | internal | 의미 |
| --- | --- | --- | --- |
| `compute_*_gate` 내부 | `xs[:, idx]`, `c_s`, `c_s >= threshold` | `xg[:, idx]`, `c_g`, `c_g >= threshold` | 게이트를 sdata 쪽 z-score에 적용하느냐, g1090 쪽 z-score에 적용하느냐 |
| `main()`의 `norm`/`y` | `s["norm"]`, `s["y"]` | `g["norm"]`, `g["y"]` | 곡선/사건 라벨을 어느 코호트에서 가져오는가 |
| 함수/변수 이름 | `compute_external_s90_gate`, `external_vote_positive_n` | `compute_internal_s90_gate`, `internal_vote_positive_n` | 위 전환을 반영한 이름 변경(로직 아님) |
| 출력 경로/제목/JSON 키 | `external_*`, `"External ..."`, `"external_dataset": "sdata"` | `internal_*`, `"Internal (g1090) ..."`, `"internal_dataset": "g1090"` | 파일명·라벨만 교체 |

이 네 그룹은 **"같은 계산을 어느 데이터에 적용/저장하느냐"만 바꾼 것**이며, 계산 자체(임계값 산출, 게이트 공식, 그림을 그리는 방식)는 100% 동일합니다. 요청하신 "internal은 데이터만 변경되는 것" 전제와 부합합니다.

### 11.2 순수 리팩터링(동작에 영향 없음)

- `from typing import cast` 및 `from matplotlib.axes import Axes` import 제거 → `ax: Axes` 타입힌트를 전부 `ax: plt.Axes`로 대체 (타입힌트 표기 방식 차이일 뿐, 런타임 동작 무관).
- `add_r4_tangent_annotation`에서 미사용 변수 `x_center = x.mean()` 삭제, `x_span` 정의 위치를 조금 아래로 이동 — 계산 결과에 영향 없음.
- `fisher_exact_conditional`의 반환문에서 `cast(float, ...)` 제거 — 정적 타입 힌트 캐스팅만 제거, 반환값 동일.
- 주석(BRANCHES/SELECTED_PATTERNS/REGION_SPANS 설명 주석 등)이 internal 파일에서 일부 생략·축약됨 — 문서화 차이일 뿐 로직 아님.

### 11.3 데이터 전환이 아닌 실제 동작 차이 — 1건 발견 및 수정 완료

```python
# 수정 전 internal (main_plot_internal_s90_core_1x3_mean_curves.py, 2x3 mirror 패널 하단)
ax_bottom.legend(frameon=False, loc="upper right", fontsize=8.5)
```

2x3 mirror 그래프의 아래쪽 서브플롯(편차 그래프) 범례 위치가 external(`"upper left"`)과 internal(`"upper right"`)에서 다르게 하드코딩되어 있었습니다. 데이터 선택과 무관한 **순수 시각적 차이**였으므로, "internal은 데이터만 바뀐다"는 전제에 맞춰 `"upper left"`로 통일했습니다. 수정 후 `internal_s90_core_2x3_mean_and_mirror_deviation.png`를 재생성해 반영했습니다.

### 11.4 결론

- 계산 로직(임상 모델, 게이트 공식, 임계값, 그림에 쓰이는 통계량)은 **internal이 external과 완전히 동일한 코드 경로**를 공유하며, 실제로 다른 것은 "어느 코호트의 배열을 넣는가"(§11.1)와 사소한 타입힌트/미사용 변수 정리(§11.2)뿐입니다.
- §11.3의 범례 위치 차이는 데이터와 무관한 코드 차이였으며, external과 동일하게(`loc="upper left"`) 맞춰 수정했습니다. 이제 두 스크립트는 데이터 선택(§11.1)과 이름/경로(§11.1) 외에는 동일합니다.
