# main_plot_external_s90_core_1x3_mean_curves.py 코드 설명 (Top-Down)

이 문서는 [main_plot_external_s90_core_1x3_mean_curves.py](python/main_plot_external_s90_core_1x3_mean_curves.py)를 **가장 바깥(실행 진입점)에서부터 안쪽(구체적 계산 함수)으로** 내려가며 이해하기 위한 설명입니다. 이 파일을 실행하려면 실제로 다른 어떤 `.py` 파일들이 필요한지, 그리고 그 파일들이 또 무엇에 의존하는지도 함께 정리했습니다.

---

## 0. 큰 그림: 이 파일의 목적과 파일 의존성 트리

**목적**: 여러 날짜에 걸쳐 이미 확정("locked")된 4개 영역(R1~R4) 형태 게이트를, 외부(sdata/Sinchon) 데이터 하나에 적용했을 때 결과 그림이 말이 되는지 확인하는 **최종 리포팅 스크립트**. 게이트를 새로 탐색하지 않고, 이미 정해진 값을 가져다 쓰기만 합니다.

**실행에 실제로 필요한 파일 의존성 트리** (import 기준):

```
plot_external_s90_core_1x3_mean_curves.py   ← 실행 진입점 (python plot_external_...py)
│
├─ aec_lock_smoothed_deesc_gate.py
│   │  가져다 쓰는 것: DATA_DIR, clinical_scores, load_dataset
│   │
│   ├─ aec_conditional_value.py
│   │      가져다 쓰는 것: clinical_estimator, clinical_matrix, make_folds,
│   │                     matrix_from_sheet, oof_and_external, zfit_apply
│   │
│   ├─ aec_universal_boundary_gate.py
│   │      가져다 쓰는 것: threshold_for_min_sensitivity
│   │
│   └─ aec128_mass_feature_combinations.py
│          가져다 쓰는 것: build_feature_bank
│          (주의: 이 함수는 aec_lock_smoothed_deesc_gate.py의 main()에서만 쓰이고,
│           plot 스크립트가 실제로 호출하는 clinical_scores/load_dataset 경로에서는
│           사용되지 않음 — import는 되지만 이 실행 경로엔 영향 없음)
│
└─ aec_new_region_surrogate_combo_gate.py
       가져다 쓰는 것: region_descriptor_matrix, z_train_apply
       │
       └─ (자체적으로 다시) aec_lock_smoothed_deesc_gate.py
              가져다 쓰는 것: DATA_DIR, clinical_scores, deesc_metric_row,
                             load_dataset, make_single_deesc
              (이것도 이 모듈 자신의 main()용이고, plot 스크립트가 쓰는
               region_descriptor_matrix/z_train_apply 경로에는 영향 없음)
```

**개념적으로(=값이 어디서 나왔는지 이해하려면) 필요한 파일** (import 관계는 없음):

```
main_aec_full_derivation_pipeline.py
  → BRANCHES 값(feature/sign/width/lambda)과 SELECTED_PATTERNS가 어떤 탐색·검증
    절차를 거쳐 "확정(lock)"되었는지 처음부터 끝까지 재현하는 독립 스크립트.
    다른 로컬 모듈을 import하지 않고 모든 로직을 자체 구현.
```

즉 **실행에 꼭 필요한 파일은 5개**(`aec_lock_smoothed_deesc_gate.py`, `aec_conditional_value.py`, `aec_universal_boundary_gate.py`, `aec128_mass_feature_combinations.py`, `aec_new_region_surrogate_combo_gate.py`)이고, **이해에 참고할 파일은 1개**(`main_aec_full_derivation_pipeline.py`)입니다.

---

## 1단계 (Depth 0): main_plot_external_s90_core_1x3_mean_curves.py의 `main()`

[main()](python/main_plot_external_s90_core_1x3_mean_curves.py#L299)의 흐름:

1. g1090(internal)/sdata(external) 로드
2. `compute_external_s90_gate(g, s)` 호출 → 외부 데이터의 "임상양성 중 AEC+/AEC-" 분류 확보
3. **PNG #1** (1x3): A) 임상 양성/음성, B) 실제 저-SMI/비저-SMI, C) 임상양성 내 AEC+/AEC- (Fisher p값 포함)
4. **PNG #2** (1x3 + R4 tangent): 패널 C에 R4 구간 적합 직선(기울기) 주석 추가
5. **PNG #3** (2x3 mirror): 3가지 대비 각각 위(평균곡선)/아래(기준곡선 대비 절대편차)
6. **CSV**: 3가지 대비의 전체/영역별 평균 차이 요약
7. **JSON**: 게이트 정의 + 조건부 사건율/Fisher p값 + 요약표

이 중 그림/표를 만드는 세부 함수는 모두 이 파일 자신 안에 정의되어 있습니다(다른 파일에서 가져오지 않음). 이 문서는 **2단계 이후, 즉 "다른 파일에서 가져오는 함수들"**에 집중하므로, 아래 §1.1에서 그 세부 함수들을 간단히만 정리하고 넘어갑니다.

### 1.1 그림/표를 만드는 세부 함수 (이 파일 자체 정의, import 없음)

| 함수 | 역할 |
| --- | --- |
| `mean_ci(x)` | 위치별 평균과 95% 신뢰구간(정규근사)을 계산 |
| `pattern_from_votes(votes)` | 4개 브랜치의 불리언 투표 행렬을 `"++--"` 같은 부호 문자열 패턴으로 변환 |
| `gate_scores(...)` | 임상 임계값 근처에서만 가우시안 가중치로 형태 특징 점수를 더하는 게이트 점수 계산 (§0의 `BRANCHES` 정의를 실제 수식으로 구현) |
| `compute_external_s90_gate(g, s)` | 위 함수들을 조합해 S90 임계값·4브랜치 투표·`SELECTED_PATTERNS` 판정까지 실행해, external 환자를 임상양성/음성 및 (임상양성 내) AEC 양성/음성으로 나눔 — `main()`이 그리는 모든 그림의 데이터 출처 |
| `add_regions(ax)` / `style_axis(ax)` | 그래프에 R1~R4 배경 밴드·라벨을 그리거나, 기준선·x축 범위·격자 등 공통 스타일을 적용 |
| `plot_group(ax, z, x, mask, label, color)` | mask로 선택된 그룹의 평균 곡선 + 95% 신뢰구간 음영을 그림 |
| `add_r4_tangent_annotation(...)` | R4 구간(117~128)에서 AEC+/AEC- 두 그룹 평균 곡선에 직선을 적합해 기울기를 텍스트로 표시 (PNG #2 전용) |
| `plot_mirror_deviation(...)` | 기준곡선 대비 두 그룹의 절대편차를 위/아래로 미러링해 그림 (PNG #3 하단 패널) |
| `panel_summary(...)` / `mirror_summary(...)` | 두 그룹의 평균곡선 차이(전체·영역별) 또는 편차(평균·최대)를 표 형태 행(row)으로 요약 — CSV/JSON 출력의 원천 |
| `fisher_exact_conditional(y, aec_pos, aec_neg)` | AEC 양성/음성군의 실제 사건율 차이에 대한 Fisher 정확검정 p값 계산 (패널 C에 표시되는 p값) |

---

## 2단계 (Depth 1): 이 파일이 직접 import하는 4개 함수

### 2.1 `load_dataset(path)` — [aec_lock_smoothed_deesc_gate.py:48](python/aec_lock_smoothed_deesc_gate.py#L48)

```python
def load_dataset(path: Path) -> dict:
    meta = pd.read_excel(path, sheet_name="metadata", ...)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", ...))
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=1.0, axis=1, mode="nearest")
    norm = row_norm(smooth_raw)
    ...
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "smooth_raw": smooth_raw, "norm": norm, "y": y, "sex": sex, "smi": smi}
```

- 엑셀의 `metadata` + `aec_128` 시트를 읽어 원자료 행렬을 만듦 (`matrix_from_sheet`로 결측치 처리는 **aec_conditional_value.py**에 위임)
- Gaussian smoothing(σ=1) → 환자별(row) 평균정규화(`row_norm`, 이 파일 자체 정의)
- TAMA/키로 SMI 계산 후 성별 기준 저-SMI 라벨 `y` 생성
- 반환된 dict의 `norm`이 plot 스크립트에서 `s["norm"]`, `g["norm"]`으로 쓰이는 정규화된 AEC-128 곡선

### 2.2 `clinical_scores(g, s)` — [aec_lock_smoothed_deesc_gate.py:134](python/aec_lock_smoothed_deesc_gate.py#L134)

```python
def clinical_scores(g, s):
    xg, xs, _ = clinical_matrix(g["meta"], s["meta"])          # → aec_conditional_value.py
    folds = make_folds(g["y"].astype(int), 5)                   # → aec_conditional_value.py
    clinical_oof, clinical_ext = oof_and_external(
        lambda seed: clinical_estimator(), xg, g["y"], xs, folds)  # → aec_conditional_value.py
    c_g, c_s, mu, sd = zfit_apply(clinical_oof, clinical_ext)   # → aec_conditional_value.py
    thresholds = {op: (threshold_for_min_sensitivity(g["y"], clinical_oof, target) - mu) / sd
                  for op, target in OPS}                        # → aec_universal_boundary_gate.py
    return clinical_oof, clinical_ext, c_g, c_s, thresholds
```

- age/height/weight/sex만으로 로지스틱 회귀 임상 모델을 5-fold OOF(internal) + external로 학습·예측
- internal 점수 기준으로 z-표준화(`c_g`, `c_s`)
- S80~S90 각 민감도 목표에 대응하는 표준화 임계값(`thresholds`) 계산
- plot 스크립트는 이 중 `c_s`(외부 임상 z-score)와 `thresholds["S90"]`만 사용

### 2.3 `region_descriptor_matrix(norm)` — [aec_new_region_surrogate_combo_gate.py:90](python/aec_new_region_surrogate_combo_gate.py#L90)

- 고정된 4개 구간 R1(45-56)/R2(57-80)/R3(97-128)/R4(117-128)마다 level/slope/curvature/endpoint_delta 등 12개 형태 서술자를 계산해 feature DataFrame 생성 (외부 모듈 의존 없이 numpy로 자체 계산)

### 2.4 `z_train_apply(xg_df, xs_df)` — [aec_new_region_surrogate_combo_gate.py:119](python/aec_new_region_surrogate_combo_gate.py#L119)

- internal(xg) 기준 중앙값으로 결측 대치 → internal 평균/표준편차로 두 데이터셋 모두 z-표준화 (단순 유틸, 값 선택 로직 없음)

---

## 3단계 (Depth 2): aec_conditional_value.py — 임상 모델 학습에 쓰이는 저수준 유틸

`clinical_scores`가 실제로 호출하는 함수들이 정의된 파일입니다. [aec_conditional_value.py](python/aec_conditional_value.py)

| 함수                                                      | 줄                                   | 역할                                                                                            |
| --------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `matrix_from_sheet(df)`                                 | [L30](python/aec_conditional_value.py#L30)   | AEC 컬럼을 숫자 행렬로 변환, 결측/비정상값을 열별(불가하면 전체) 중앙값으로 대체                |
| `clinical_matrix(train_meta, test_meta)`                | [L73](python/aec_conditional_value.py#L73)   | age/height/weight/sex_M 4개 임상변수 행렬 생성, train 기준 결측대체+표준화를 test에도 동일 적용 |
| `make_folds(y, k=5)`                                    | [L98](python/aec_conditional_value.py#L98)   | 클래스 비율 유지하며 5-fold 인덱스 분할                                                         |
| `clinical_estimator()`                                  | [L109](python/aec_conditional_value.py#L109) | `LogisticRegression(C=1e6, ...)` — 정규화 거의 없는 임상 전용 로지스틱 회귀                  |
| `oof_and_external(model_factory, xtr, ytr, xte, folds)` | [L135](python/aec_conditional_value.py#L135) | 폴드별 학습으로 internal out-of-fold 예측 + 전체 재학습 모델로 external 예측 반환               |
| `zfit_apply(train_score, test_score)`                   | [L149](python/aec_conditional_value.py#L149) | train 점수 평균/표준편차로 두 점수 모두 z-표준화                                                |

이 파일은 원래 자기 자신의 `main()`(별도의 "조건부 가치" 분석 프로토콜)도 갖고 있지만, plot 스크립트 경로에서는 오직 위 표의 6개 유틸 함수만 재사용됩니다.

---

## 4단계 (Depth 2): aec_universal_boundary_gate.py — 임계값 계산

| 함수                                                | 줄                                       | 역할                                                                                     |
| --------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| `threshold_for_min_sensitivity(y, score, target)` | [L32](python/aec_universal_boundary_gate.py#L32) | 목표 민감도(target) 이상을 유지하면서 특이도가 최대인 임계값 탐색 (없으면 분위수로 근사) |

`clinical_scores`가 S80/S85/S90 등 각 운영점(operating point)의 임계값을 정할 때 이 함수 하나만 씁니다.

---

## 5단계 (Depth 2): aec128_mass_feature_combinations.py — (이 실행 경로에서는 미사용)

| 함수                           | 역할                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_feature_bank(x_norm)` | 다중스케일 레벨/기울기/곡률, Haar 엣지, 해부학적 구간 간 쌍별 대비/비율, DCT/FFT/자기상관까지 포함하는 수천 차원짜리 초대형 특징 은행 생성 |

`aec_lock_smoothed_deesc_gate.py`가 이 함수를 import하긴 하지만, 이는 그 파일 자신의 `main()`(대규모 특징뱅크 기반 de-escalation 게이트 탐색, R1~R4와는 무관한 **별도 프로토콜**)에서만 쓰입니다. plot 스크립트가 실제로 호출하는 `load_dataset`/`clinical_scores` 경로에는 전혀 관여하지 않습니다 — import 트리에는 있지만 "죽은 경로"라고 이해하면 됩니다.

---

## 6단계 (Depth 1, 별도 가지): aec_new_region_surrogate_combo_gate.py 자신의 `main()`

이 파일은 plot 스크립트에 `region_descriptor_matrix`/`z_train_apply`만 제공하지만, **원래 자신의 목적은 R1~R4 게이트를 처음 탐색해서 확정하는 스크립트**였습니다 ([main()](python/aec_new_region_surrogate_combo_gate.py#L397)):

1. R1~R4 각 구간 × 12개 서술자 × sign(±1) × width × lambda 조합으로 단일-region branch 후보를 전수 스크리닝 (`branch_summary_row`)
2. 지역별 상위 후보 선택 (`select_branch_candidates`)
3. 4개 지역에서 하나씩 골라 만든 4-branch 조합 × 16가지 "+/-" 패턴 마스크를 전수 탐색 (`candidate_masks`, `combo_summary_row`)
4. internal(및 internal+external) 안전성 제약을 통과하는 규칙을 찾아 `new4_combo_summary.json` 등으로 저장

즉 `plot_external_s90_core_1x3_mean_curves.py`의 `BRANCHES`/`SELECTED_PATTERNS` 상수는, **이 파일의 `main()`을 실행해서 나온 결과값을 그대로 복사**해 놓은 것입니다. 이후 이 탐색 과정 전체가 [main_aec_full_derivation_pipeline.py](python/main_aec_full_derivation_pipeline.py) 한 파일로 다시 정리·통합되었습니다 (아래 7단계).

---

## 7단계 (참고, import 관계 없음): main_aec_full_derivation_pipeline.py — BRANCHES의 근본 도출 절차

이 파일은 plot 스크립트가 import하지는 않지만, **`BRANCHES` 값이 왜 그 숫자들인지**를 이해하려면 필요한 "derivation script"입니다. 다른 로컬 모듈에 의존하지 않고 전체 로직을 자체 구현합니다.

| 단계                   | 함수/구간                                                                                                 | 내용                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1. 데이터 로드         | `load_dataset` [L230](python/main_aec_full_derivation_pipeline.py#L230)                                              | g1090/sdata 로드, Gaussian smoothing + 환자별 정규화, SMI 기반 라벨        |
| 2. 임상 모델           | `make_context` [L345](python/main_aec_full_derivation_pipeline.py#L345)                                              | age/height/weight/sex 로지스틱 회귀, S80/S85/S90 임계값                    |
| 3. Region scout        | `run_region_scout` [L714](python/main_aec_full_derivation_pipeline.py#L714)                                          | 128포인트 전체에서 슬라이딩 윈도우 전수 스캔 → "왜 R1~R4를 보게 됐는가"   |
| 4. Region 고정         | `LOCKED_REGIONS` [L109](python/main_aec_full_derivation_pipeline.py#L109)                                            | scout 결과+해석을 바탕으로 R1(45-56)/R2(57-80)/R3(97-128)/R4(117-128) 확정 |
| 5. Branch search       | `screen_branch_candidates` [L768](python/main_aec_full_derivation_pipeline.py#L768)                                  | 4개 region × 12 descriptor × sign × width × lambda 전수 스크리닝       |
| 6. 4-branch 패턴 탐색  | `run_combo_pattern_search` [L978](python/main_aec_full_derivation_pipeline.py#L978)                                  | region별 후보 조합 × 5개 패턴 마스크 전수 탐색                            |
| 7. Noninferiority 검정 | `evaluate_deescalation`/`clopper_pearson_one_sided_upper` [L501](python/main_aec_full_derivation_pipeline.py#L501) | S90에서 민감도손실 95% 상한 ≤ 5%p 통과 규칙만 채택                        |
| 최종 결과              | `LOCKED_PRIMARY_BRANCHES` / `LOCKED_PRIMARY_PATTERNS` [L118](python/main_aec_full_derivation_pipeline.py#L118)     | plot 스크립트의`BRANCHES`/`SELECTED_PATTERNS`와 값이 동일              |

### 7-1. Region scout의 슬라이딩 윈도우 파라미터 (coarse/fine 스캔)

`candidate_windows` [L456](python/main_aec_full_derivation_pipeline.py#L456) / `run_region_scout` [L717-718](python/main_aec_full_derivation_pipeline.py#L717-L718):

```python
coarse = candidate_windows(step=8, lengths=[16, 24, 32])
fine   = candidate_windows(step=4, lengths=[12, 16, 20, 24, 28, 32], lo=33, hi=128)
```

- **coarse(성긴) 스캔**: 128포인트 전체(1~128)를 8칸 간격으로 이동하면서, 길이 16/24/32짜리 윈도우 후보를 생성 — 큰 그림에서 "어느 대략적인 구간이 쓸모 있는지" 넓게 훑는 1차 스캔
- **fine(촘촘한) 스캔**: 33~128 구간(초반부 제외)만, 4칸 간격으로 이동하면서 길이 12/16/20/24/28/32짜리 윈도우 후보를 추가 생성 — coarse보다 촘촘하게(간격 절반) 더 다양한 길이로 세밀하게 재탐색
- 이렇게 만들어진 모든 윈도우 후보 각각에 대해 12개 서술자(level/slope/curvature 등) × sign(±1) × width(0.35/0.50/0.70) × lambda(0.25/0.55/0.70) 조합으로 단일-feature de-escalation 게이트 성능을 채점(`summarize_single_feature_rule`)하고, internal 스코어 상위 결과들을 CSV(`01_region_scout_window_feature_ranked.csv`)로 저장
- **이 단계는 아직 R1~R4를 확정하지 않은 상태**입니다 — "8칸/4칸 간격으로 촘촘하게 훑어본 후보 목록"일 뿐이고, 그 결과 상위권에 몰려 있던 구간들을 사람이 해석해서 R1(45-56)/R2(57-80)/R3(97-128)/R4(117-128) 4개로 고정한 것이 다음 단계(`LOCKED_REGIONS`)입니다

**왜 fine 스캔은 33~128만 보는가 (검증된 근거)**:

`main_aec_full_derivation_pipeline.py`의 `lo=33, hi=128`은 임의 고정값이 아니라, 더 이전 단계인 원조 탐색 스크립트 [aec_region_search_surrogate.py](old_python/aec_region_search_surrogate.py)의 결과를 반영한 값입니다. 그 파일에서는 fine 스캔 범위가 고정 구간이 아니라, **coarse 스캔에서 internal 성적이 좋았던 상위 30개 윈도우 각각의 주변(±16, `flank=16`)만 다시 촘촘히 훑는 적응형 범위**였습니다 (`fine_windows_from_top`, [aec_region_search_surrogate.py:243-253](old_python/aec_region_search_surrogate.py#L243-L253)):

```python
def fine_windows_from_top(coarse: pd.DataFrame, flank: int = 16) -> list[tuple[int, int]]:
    top = coarse[coarse["internal_pass"]].sort_values("internal_selection_score", ascending=False).head(30)
    if top.empty:
        top = coarse.sort_values("internal_selection_score", ascending=False).head(30)
    windows: set[tuple[int, int]] = set()
    for r in top.itertuples():
        lo = max(1, int(r.window_start) - flank)
        hi = min(128, int(r.window_end) + flank)
        windows.update(candidate_windows(FINE_STEP, FINE_LENGTHS, lo, hi))
    return sorted(windows)
```

즉 coarse 스캔(8칸 간격)에서 internal 기준으로 쓸 만했던 후보 윈도우들이 실제로는 **거의 다 33번 지점 이후(중반~후반)에 몰려 있었다**는 뜻이고, R1(45)~R4(117-128)도 모두 그 범위 안에 들어옵니다. 반대로 1~32번 구간(craniocaudal index 기준 치골결합부 바로 위쪽 초반부)에서는 coarse 스캔에서 유의미한 후보가 나오지 않았다는 의미입니다.

`main_aec_full_derivation_pipeline.py`는 이 적응형 탐색을 매번 다시 계산하는 대신, 그 경험적 결과("좋은 후보는 33 이후에 몰려 있다")를 고정 경계 `lo=33`으로 하드코딩해 재현 파이프라인을 단순화한 것으로 보입니다. `COARSE_STEP=8`, `FINE_STEP=4`, `COARSE_LENGTHS=[16,24,32]`, `FINE_LENGTHS=[12,16,20,24,28,32]` 등 파라미터 자체도 `aec_region_search_surrogate.py:22-29`의 값과 동일합니다.

---

## 8. 정리: 파일별 역할 한눈에 보기

| 파일                                          | plot 스크립트와의 관계           | 핵심 역할                                                                                                        |
| --------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `plot_external_s90_core_1x3_mean_curves.py` | 본체                             | 확정된 게이트를 외부 데이터에 적용해 그림/표/JSON 생성                                                           |
| `aec_lock_smoothed_deesc_gate.py`           | 직접 import                      | 데이터 로딩(`load_dataset`) + 임상 점수/임계값(`clinical_scores`) 제공                                       |
| `aec_conditional_value.py`                  | 간접 import (2단계)              | 임상 로지스틱 회귀 학습에 필요한 저수준 유틸(폴드분할, OOF, z표준화 등)                                          |
| `aec_universal_boundary_gate.py`            | 간접 import (2단계)              | 목표 민감도 기준 임계값 탐색 함수 1개                                                                            |
| `aec128_mass_feature_combinations.py`       | 간접 import (2단계, 실질 미사용) | 대규모 특징뱅크 생성 — plot 경로에서는 호출 안 됨                                                               |
| `aec_new_region_surrogate_combo_gate.py`    | 직접 import                      | R1~R4 형태 특징 계산(`region_descriptor_matrix`) + z표준화(`z_train_apply`); 원래는 게이트 탐색 스크립트였음 |
| `main_aec_full_derivation_pipeline.py`      | import 없음 (개념적 참고)        | `BRANCHES`/`SELECTED_PATTERNS` 값이 어떤 탐색·통계검정을 거쳐 확정됐는지 재현하는 독립 문서형 스크립트      |

---

## 9. 임상적 배경 (Clinical Background)

코드 주석만으로는 "AEC가 정확히 뭘 재는 신호인지", "45.4/34.4 같은 숫자가 어디서 왔는지"가 드러나지 않아서, 프로젝트의 논문/인수인계 문서인 [handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md)를 근거로 확인한 내용입니다. `.py` 파일 어디에도 없는 정보라 출처를 명확히 표시합니다.

### 9.1 AEC란?

**AEC = Automatic Exposure Control** ([GPU_HANDOFF_PROMPT.md:507](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L507), 논문 제목: *"Automatic Exposure Control Profiles for Second-Stage Reclassification of Low Skeletal Muscle Index Risk on CT"*). CT 촬영 시 스캐너가 z축(머리-발 방향)을 따라 자동으로 조절하는 관전류(tube current) 프로파일 신호입니다. 즉 환자 조직 자체를 직접 찍은 영상이 아니라, **스캐너가 "이 부위는 이만큼의 X선량이 필요하다"고 자동 조절한 궤적**이 곧 환자의 체형/조직 분포를 간접적으로 반영한다는 아이디어입니다.

- `AEC profiles were resampled to 128 anatomically corresponding positions and normalized by each patient's mean AEC value` ([L519](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L519)) → 이것이 바로 `aec_128` 시트, 그리고 `load_dataset`의 환자별 평균정규화(`row_norm`/`patient_wise_mean_normalize`)의 근거
- **주의**: "AEC"가 "aortic", "abdominal", "erector spinae" 등 해부학 용어의 약자라는 근거는 코드/문서 어디에도 없습니다 (Automatic Exposure Control이 맞음)

### 9.2 SMI, TAMA, 저-SMI(low-SMI) 기준

```text
SMI = TAMA / (키[m])^2
남성 저-SMI: SMI < 45.4 cm²/m²
여성 저-SMI: SMI < 34.4 cm²/m²
```

([GPU_HANDOFF_PROMPT.md:39-52](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L39-L52))

- 이 컷오프는 **Derstine 기준(Derstine cutoffs)**을 그대로 채택한 것이며, 문서에 "Yoon criteria were considered earlier but discarded by user. Derstine low SMI is fixed."라고 명시되어 있어 — 다른 기준(Yoon)도 검토했지만 최종적으로 Derstine 기준으로 고정했다는 의사결정 이력이 있습니다.
- `TAMA`가 정확히 무엇의 약자인지는 코드/문서에 풀어쓰여 있지 않습니다. (관례적으로 Total Abdominal Muscle Area로 흔히 쓰이는 지표이지만, **이 프로젝트 파일 안에서 명시적으로 확인되지는 않았습니다** — 추정으로 단정하지 않습니다.)
- 코드에서는 [aec_lock_smoothed_deesc_gate.py:58](python/aec_lock_smoothed_deesc_gate.py#L58) 등에서 `y = np.where(sex == "M", smi < 45.4, smi < 34.4)`로 그대로 구현됩니다.
- g1090(internal) 저-SMI 유병률 11.8%(129/1090), sdata(external) 15.2%(141/926, 남 22.9%/여 8.6%) ([L75-87](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L75-L87))

### 9.3 S80/S85/S90은 "무엇의 80/85/90%"인가

`TARGET_OPS`/`OPS`에 등장하는 S80/S85/S90은 **임상 모델(나이·키·몸무게·성별)의 목표 민감도(sensitivity)**를 80%/85%/90%로 고정한 운영점(operating point)입니다 ([main_aec_full_derivation_pipeline.py:102](python/main_aec_full_derivation_pipeline.py#L102) `TARGET_OPS = [("S80", 0.80), ("S85", 0.85), ("S90", 0.90)]`). 즉 "저-SMI 환자를 최소 90% 잡아내려면 임상점수 임계값을 얼마로 잡아야 하는가"를 역산한 지점이 S90입니다. 민감도를 높게 고정할수록 임계값이 낮아져 임상 양성(clinical positive) 판정을 받는 환자가 늘어나고, 그만큼 위양성(false positive)도 늘어납니다 — 이 위양성을 줄이는 것이 AEC 게이트의 존재 이유입니다.

### 9.4 왜 "de-escalation(위험강등)"이 필요한가 — 임상적 동기

문서의 핵심 프레이밍 ([GPU_HANDOFF_PROMPT.md:11-15](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L11-L15)):

> AEC is not a standalone low-SMI detector. AEC is an acquisition-derived second-stage reclassification / triage biomarker that improves clinical utility among patients flagged by a simple clinical model.
>
> Clinical variables already identify many low-SMI patients but produce many false positives. AEC-derived shape signatures can reclassify clinically flagged patients, identifying a very-low-risk de-escalation group and a high-yield priority group in external validation.

즉:

1. 나이/키/몸무게/성별만으로 만든 **임상 모델은 민감도(90%)는 높지만, 그만큼 위양성이 많다** — "고위험"으로 분류된 사람 중 실제로는 저-SMI가 아닌 사람이 많이 섞여 있음
2. AEC는 **그 자체로 저-SMI를 처음부터 걸러내는 1차 검사(standalone detector)가 아니라**, 이미 임상적으로 "양성(위험)" 판정을 받은 사람들 중에서 **형태학적으로 "사실 위험도가 낮아 보이는" 하위군을 다시 골라내는 2차(second-stage) 재분류 검사**
3. 실제 관찰: "임상 양성 중 AEC 음성인 환자들은 실제 저-SMI 유병률이 매우 낮았다" ([L152-153](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L152-L153)) → 이런 환자들은 위험도를 낮춰도(de-escalate) 안전하다는 근거
4. 사전등록형 lock 프로토콜(g1090 internal에서 규칙 도출 → sdata external에서는 검증만)에서 41명을 강등했을 때 실제 저-SMI는 0명이었다는 구체적 수치도 문서에 기록됨 ([L159-161](handoff_extract/gpu_handoff_package/handoff/GPU_HANDOFF_PROMPT.md#L159-L161))

`plot_external_s90_core_1x3_mean_curves.py`의 패널 C("Clinical+ / AEC+" vs "Clinical+ / AEC-")가 정확히 이 임상적 스토리(임상양성 내에서 AEC로 재분류했을 때 실제 사건율이 갈라지는지)를 시각화하는 부분입니다.

### 9.5 R1~R4는 특정 해부학적 구조물(간, 요근 등)을 의미하는가?

**아니요 — 코드/문서 어디에도 R1~R4를 특정 장기·근육에 대응시키는 설명은 없습니다.** [main_aec_full_derivation_pipeline.py:27-38](python/main_aec_full_derivation_pipeline.py#L27-L38)에 명시된 대로, R1~R4는 순수하게 **데이터 기반 슬라이딩 윈도우 탐색(region scout) + 사람의 시각적 해석**으로 고정된 구간이며, "이 구간이 간(liver)/요근(psoas)/척추기립근(erector spinae)에 해당한다"는 해부학적 주석은 어디에도 없습니다. 코드에서 해부학적으로 명시된 것은 **곡선의 양 끝점뿐**입니다: `"Craniocaudal index: 1 inferior pubic margin -> 128 liver dome"` (예: [main_plot_external_s90_core_1x3_mean_curves.py:382](python/main_plot_external_s90_core_1x3_mean_curves.py#L382)) — 1번 지점이 치골결합부(pubic symphysis) 바로 위, 128번 지점이 간의 상단(liver dome)이라는 것만 확인됩니다. R1~R4가 이 축 위의 "어디쯤"인지는 알 수 있어도, 그것이 어떤 조직에 대응하는지는 이 프로젝트 파일들에서 확인되지 않았습니다.

---

## 10. 실제 산출 결과값 정리 (2026-07-07 재실행분, `outputs/aec_1x3_core_mean_curves/`)

이 스크립트를 실제로 돌려서 나온 external(sdata) 결과 파일의 핵심 수치를 정리합니다. (아래는 "이 코드가 실제로 낸 답"의 스냅샷이며, 데이터가 바뀌면 값도 바뀝니다.)

### 10.1 임상양성 내 AEC 조건부 분리 (`external_s90_core_1x3_summary.json` → `low_smi_conditional`) — 패널 C의 근거 수치

| | AEC+ (강등 후보) | AEC- (유지) |
| --- | --- | --- |
| 인원(n) | 56 | 480 |
| 실제 저-SMI 사건 수 | 2 | 129 |
| 사건율 | **3.6%** | **26.9%** |
| Fisher 정확검정 p | **2.30×10⁻⁵** | (동일 검정) |

→ 임상적으로 이미 "위험군(S90 임상양성)"으로 분류된 536명 중, AEC 형태 게이트가 그중 56명을 "사실 위험도가 낮다"고 재분류했더니, 그 56명의 실제 저-SMI 유병률은 3.6%에 불과했던 반면(유지군은 26.9%), 이 차이는 통계적으로 매우 유의(p<0.0001)합니다. **이 3.6% vs 26.9% 대비가 이 스크립트 전체의 핵심 임상적 결론**이며, 패널 C 그래프와 R4 접선 기울기 주석이 이 결과를 시각적으로 보여주는 장치입니다.

참고로 g1090(internal, `internal_s90_core_1x3_summary.json`)에서도 같은 방향의 결과가 재현됩니다: AEC+ 53명 중 사건 2명(3.8%) vs AEC- 518명 중 115명(22.2%), Fisher p=5.53×10⁻⁴ — internal에서 먼저 확정(lock)된 규칙이 external에서도 방향과 크기가 비슷하게 재현된다는 뜻입니다.

### 10.2 브랜치별 투표 통과 인원 (`branches[].external_vote_positive_n`)

External(n=926) 기준, R1~R4 각 브랜치가 개별적으로 "게이트 통과(위험 방향)"로 투표한 인원:

| 구간 | 서술자 | 투표 인원 | 비율 |
| --- | --- | --- | --- |
| R1 (45-56) | endpoint_delta | 387 | 41.8% |
| R2 (57-80) | level_mean | 393 | 42.4% |
| R3 (97-128) | linear_slope | 338 | 36.5% |
| R4 (117-128) | endpoint_delta | 450 | 48.6% |

이 개별 브랜치 투표 자체는 아직 최종 판정이 아니며, `SELECTED_PATTERNS`에 속하는 4비트 조합(`++++`, `++--`, `+--+`, `--+-`, `---+`)일 때만 최종 AEC+로 인정됩니다(`compute_external_s90_gate`).

### 10.3 세 가지 대비의 영역별 평균곡선 차이 (`external_s90_core_1x3_mean_curve_summary.csv`)

| 대비 | 전체 평균 절대차 | R1 | R2 | R3 | R4 |
| --- | --- | --- | --- | --- | --- |
| Clinical+ vs Clinical− | 0.0187 | −0.0095 | −0.0345 | +0.0184 | +0.0136 |
| Low SMI+ vs Non-low SMI | 0.0228 | −0.0261 | −0.0083 | +0.0377 | +0.0316 |
| Clinical+/AEC− vs Clinical+/AEC+ | 0.0117 | −0.0037 | −0.0067 | +0.0218 | +0.0300 |

R3·R4(후반부, 간 방향)는 항상 "+"(해당 그룹이 더 높음), R1·R2(전반부)는 항상 "−" 방향으로 일관되게 나타나, 세 가지 서로 다른 대비(임상양성/실제사건/AEC조건부)에서도 **같은 방향의 형태 차이**가 재현됩니다.

### 10.4 산출물 파일 한눈에 보기

| 파일 | 내용 |
| --- | --- |
| `external_s90_core_1x3_mean_curves.png` | 본문 §main() 1x3 그래프 (A 임상양성/음성, B 실제사건, C 임상양성 내 AEC 분리) |
| `external_s90_core_1x3_mean_curves_with_r4_tangent.png` | 위 그래프 + R4 구간 적합 직선(기울기) 주석 |
| `external_s90_core_2x3_mean_and_mirror_deviation.png` | 위 3대비 각각의 평균곡선(위) + 기준곡선 대비 절대편차 미러 그래프(아래) |
| `external_s90_core_1x3_mean_curve_summary.csv` | §10.3의 출처 — 전체/영역별 평균 차이 |
| `external_s90_core_2x3_mirror_deviation_summary.csv` | 미러 그래프의 영역별 평균/최대 절대편차 |
| `external_s90_core_1x3_summary.json` | 게이트 정의 + §10.1/10.2의 출처 — 가장 먼저 열어봐야 할 파일 |

같은 폴더의 `internal_*` 접두 파일들은 이 스크립트가 아니라 같은 구조의 **internal(g1090) 버전을 만드는 별도 스크립트**([main_plot_internal_s90_core_1x3_mean_curves.py](python/main_plot_internal_s90_core_1x3_mean_curves.py), 자세한 설명은 [main_plot_internal_s90_core_1x3_mean_curves.md](main_plot_internal_s90_core_1x3_mean_curves.md) 참고)의 산출물입니다 — 이 스크립트(`main_plot_external_s90_core_1x3_mean_curves.py`)는 external(sdata) 결과만 생성합니다.
