# main_aec_full_derivation_pipeline.py 코드 설명 (Top-Down)

이 문서는 [python/main_aec_full_derivation_pipeline.py](python/main_aec_full_derivation_pipeline.py)를 **가장 바깥(실행 진입점)에서부터 안쪽(구체적 계산 함수)으로** 내려가며 이해하기 위한 설명입니다. [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md), [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md)와 짝을 이루는 문서로, 같은 스타일(의존성/역할 정리 → depth별 함수 설명 → 산출물 → 다른 파일과의 관계)로 작성했습니다.

---

## 0. 큰 그림: 이 파일은 무엇을 하는가

**한 줄 요약**: 원자료(엑셀) 로딩부터 임상 모델, "왜 R1~R4를 보게 됐는가"의 region scout, branch/pattern 전수탐색, 공식 noninferiority(비열등성) lock, 그리고 2차 CNN-mimic 결과 재현까지 — **다른 로컬 모듈을 하나도 import하지 않고** 전체 분석 과정을 한 파일 안에 자체 구현(self-contained)한 "derivation(도출) 스크립트"입니다.

이 파일의 존재 이유는 새로운 분석이 아니라 **감사(audit) 가능성**입니다: [main_plot_external_s90_core_1x3_mean_curves.py](python/main_plot_external_s90_core_1x3_mean_curves.py)의 `BRANCHES`/`SELECTED_PATTERNS` 상수, [main_aec_new_region_cnn_surrogate_mimic_gate.py](python/main_aec_new_region_cnn_surrogate_mimic_gate.py)의 `TEACHER_BRANCHES`/`TEACHER_PATTERNS` — 이런 "이미 확정되어 다른 스크립트에 하드코딩된 숫자들"이 정확히 어떤 원자료, 어떤 탐색 절차, 어떤 통계 검정을 거쳐 나왔는지를 **처음부터 끝까지 재현**할 수 있게 해줍니다. 파일 자체의 모듈 docstring(코드 최상단 주석, [L1-73](python/main_aec_full_derivation_pipeline.py#L7-L73))에 10단계 분석 흐름이 이미 요약되어 있고, 이 문서는 그 각 단계를 실제 코드 위치와 함께 더 자세히 풀어씁니다.

### 0.1 실행 모드 2가지 — CLI

```
python main_aec_full_derivation_pipeline.py --mode reproduce     # 기본값, 빠름
python main_aec_full_derivation_pipeline.py --mode full-search   # 느림, 탐색표까지 전부 생성
```

| 모드 | 실행 함수 | 하는 일 | 소요 시간 |
| --- | --- | --- | --- |
| `reproduce` (기본) | [run_reproduce()](python/main_aec_full_derivation_pipeline.py#L1173-L1180) | 이미 정해진 `LOCKED_PRIMARY_BRANCHES`/`LOCKED_PRIMARY_PATTERNS`(1차 결과)와, 저장된 CNN 확률 파일(2차 결과)을 **그대로 적용**해 최종 지표·그림만 재생성 | 수 초~수십 초 (region scout/전수탐색 생략) |
| `full-search` | [run_full_search()](python/main_aec_full_derivation_pipeline.py#L1183-L1201) | region scout → branch 후보 스크리닝 → 4-branch × 패턴 전수탐색을 **처음부터 다시 실행**해서, "그 lock된 값이 실제로 이 탐색에서 나오는가"까지 검증 | 훨씬 오래 걸림 (코드 주석 경고, [L71-72](python/main_aec_full_derivation_pipeline.py#L71-L72)) |

두 모드 모두 마지막에 [write_final_outputs()](python/main_aec_full_derivation_pipeline.py#L620-L674)를 호출해 동일한 최종 산출물을 남기므로, `full-search`는 `reproduce`의 "상위 호환"입니다(중간 탐색표가 추가로 남는다는 차이뿐).

### 0.2 산출물 위치 — 다른 스크립트와 다른 규칙

다른 문서들(`main_plot_external_s90_core_1x3_mean_curves.py` 등)은 `outputs/<스크립트별_폴더>/`에 저장하지만, 이 파일은 **`OUT_DIR = SCRIPT_PATH.parent / "full_derivation_output"`** ([L97](python/main_aec_full_derivation_pipeline.py#L97)) — 즉 `work/outputs/` 아래가 아니라 **`python/full_derivation_output/`** (스크립트 자신과 같은 `python/` 폴더 바로 아래)에 저장합니다. 이 문서를 작성한 시점 기준 이 폴더에는 `reproduce` 모드 산출물만 있고(§8 참고), `full-search`의 중간 탐색표(`01_~04_` 파일들)는 아직 생성된 적이 없습니다.

---

## 1. 실행에 필요한 파일 — 의존성 없음

```
main_aec_full_derivation_pipeline.py   ← 실행 진입점, 로컬 모듈 import 없음
│
└─ (선택적 입력) outputs/aec_new_region_cnn_surrogate_mimic_gate/surrogate_mimic_balanced_probabilities.npz
       main_aec_new_region_cnn_surrogate_mimic_gate.py를 실행해서 나온 결과물.
       이 파일이 없어도 compute_secondary_cnn_mimic()이 None을 반환하고
       나머지 1차(interpretable) 결과는 정상적으로 재현됨 (§7 참고).
```

즉 이 스크립트는 다른 `.py` 파일을 전혀 import하지 않습니다 — `load_dataset`, `clinical_scores`류 로직, region descriptor 계산, gate 수식까지 **모두 이 파일 안에 독립적으로 다시 구현**되어 있습니다. 이는 [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md) §1에서 지적한 "코드 중복" 패턴과 같은 성격이지만, 이 파일의 경우는 의도된 설계입니다 — 감사자가 이 파일 하나만 읽고 실행해도 전체 파이프라인이 재현되도록, 일부러 다른 파일에 대한 의존을 없앤 것으로 보입니다. 다만 그 대가로 로직이 4곳(이 파일, `aec_lock_smoothed_deesc_gate.py`, `aec_conditional_value.py`, `aec_new_region_surrogate_combo_gate.py`)에 유사하게 흩어져 있으므로, 한쪽만 수정하면 서로 어긋날 수 있다는 점은 동일하게 유의해야 합니다.

유일한 예외가 위 트리의 `.npz` 파일입니다 — import는 아니지만, `compute_secondary_cnn_mimic()`이 **파일 경로로 다른 스크립트의 산출물을 읽어들이는** 유일한 외부 연결점입니다 (§7.2).

---

## 2. main()의 전체 흐름 (Depth 0)

[main()](python/main_aec_full_derivation_pipeline.py#L1204-L1211)은 `argparse`로 `--mode`만 읽고 [run_reproduce()](python/main_aec_full_derivation_pipeline.py#L1173) 또는 [run_full_search()](python/main_aec_full_derivation_pipeline.py#L1183)로 분기합니다.

- **`run_reproduce()`**: `make_context()` → `compute_locked_gate()` → `compute_secondary_cnn_mimic()` → `write_final_outputs()`. 마지막에 최종 지표 CSV를 콘솔에 그대로 출력합니다.
- **`run_full_search()`**: 위 흐름 앞에 `run_region_scout()` → `screen_branch_candidates()` → `precompute_branch_votes()` → `run_combo_pattern_search()` 4단계가 추가되고, 각 단계 진행 상황(`Stage n/5: ...`)을 콘솔에 출력합니다.

아래 §3~7은 이 흐름을 구성하는 함수들을 파일 안의 절 구분(코드 주석의 `# n. 제목` 배너)을 따라 순서대로 설명합니다.

---

## 3. 1절: 데이터 로딩과 전처리 — `load_dataset`

- [aec_columns(df)](python/main_aec_full_derivation_pipeline.py#L207-L209): `aec_` 로 시작하는 컬럼만 골라 번호순 정렬.
- [matrix_from_aec_sheet(df)](python/main_aec_full_derivation_pipeline.py#L212-L221): 결측/비정상값을 열별 중앙값(불가하면 전체 중앙값)으로 대체 — `aec_conditional_value.py`의 `matrix_from_sheet`와 로직상 동일한 자체 구현.
- [patient_wise_mean_normalize(x)](python/main_aec_full_derivation_pipeline.py#L224-L227): 환자별 평균으로 나눠 정규화(평균이 0이거나 비정상이면 1로 대체해 0-division 방지).
- [load_dataset(path)](python/main_aec_full_derivation_pipeline.py#L230-L250): 위 유틸을 조합해 `metadata`/`aec_128` 시트를 읽고, Gaussian smoothing(σ=`SMOOTHING_SIGMA`=1.0) → 환자별 정규화 → TAMA/키 기반 SMI 계산 → 성별 기준 저-SMI 라벨(`low_smi`, 남 SMI<45.4 / 여 SMI<34.4)까지 한 번에 만듭니다. `aec_lock_smoothed_deesc_gate.py`의 `load_dataset`과 같은 정의를 다시 구현한 버전입니다.

---

## 4. 2절: 임상 모델 — `make_context`

- [clinical_design_matrix(internal_meta, external_meta)](python/main_aec_full_derivation_pipeline.py#L258-L278): age/height/weight/sex_M 4개 임상변수 행렬 생성, internal(g1090) 기준 결측 대치 + 표준화를 external에도 동일 적용.
- [stratified_folds(y, k=5)](python/main_aec_full_derivation_pipeline.py#L281-L288): 클래스 비율을 유지하며 5-fold 인덱스 분할.
- [fit_clinical_scores(xg, yg, xs)](python/main_aec_full_derivation_pipeline.py#L291-L303): `LogisticRegression(C=1e6, ...)`(정규화 거의 없는 임상 전용 모델)으로 internal은 5-fold out-of-fold 점수, external은 전체 재학습 모델로 예측 점수를 계산.
- [z_standardize_by_internal(...)](python/main_aec_full_derivation_pipeline.py#L306-L311): internal 점수의 평균/표준편차로 두 점수 모두 z-표준화.
- [binary_metrics(y, pred_positive)](python/main_aec_full_derivation_pipeline.py#L314-L329): TP/FP/FN/TN과 민감도·특이도·정확도를 계산하는 공용 유틸 — 이 파일 전체에서 반복적으로 재사용됨.
- [threshold_for_min_sensitivity(y, score, target)](python/main_aec_full_derivation_pipeline.py#L332-L342): 목표 민감도 이상을 유지하면서 특이도가 최대인 임계값 탐색(없으면 분위수로 근사) — `aec_universal_boundary_gate.py`의 동명 함수와 동일한 로직.
- [make_context()](python/main_aec_full_derivation_pipeline.py#L345-L367): 위 함수들을 모두 묶어 internal/external 데이터, 임상 z-score, `TARGET_OPS`(S80/S85/S90)별 임계값과 임상양성 마스크(`cpos_g`/`cpos_s`)까지 담은 딕셔너리(`ctx`)를 만듭니다. 이후 거의 모든 함수가 이 `ctx`를 인자로 받습니다.

---

## 5. 3절: Region scout — "왜 R1~R4를 보게 되었나"

- [d1(x)](python/main_aec_full_derivation_pipeline.py#L375-L377) / [d2(x)](python/main_aec_full_derivation_pipeline.py#L380-L382): 곡선의 1차 차분(기울기)/2차 차분(곡률).
- [window_features(norm, windows)](python/main_aec_full_derivation_pipeline.py#L385-L412): 임의의 (start, end) 윈도우 목록 각각에 대해 level/slope/curvature 계열 **14개 서술자**를 계산 (`locked_region_descriptor_matrix`보다 2개 더 많음 — `level_min`/`level_max`가 scout 단계에만 추가로 있음).
- [candidate_windows(step, lengths, lo, hi)](python/main_aec_full_derivation_pipeline.py#L456-L461): `lo~hi` 구간을 `step` 간격으로 이동하며 주어진 `lengths` 길이의 윈도우 후보를 생성.
- [run_region_scout(ctx)](python/main_aec_full_derivation_pipeline.py#L714-L759): 아래 coarse/fine 두 스캔으로 만든 윈도우 후보 전체 × 12서술자 × sign(±1) × width(0.35/0.50/0.70) × lambda(0.25/0.55/0.70) 조합의 단일-feature 게이트 성능을 채점해 `01_region_scout_window_feature_ranked.csv`로 저장합니다(`full-search` 모드 전용, §0.2 참고).

### 5.1 coarse/fine 스캔 파라미터와 그 근거

```python
coarse = candidate_windows(step=8, lengths=[16, 24, 32])
fine   = candidate_windows(step=4, lengths=[12, 16, 20, 24, 28, 32], lo=33, hi=128)
```
([L717-718](python/main_aec_full_derivation_pipeline.py#L717-L718))

- **coarse(성긴) 스캔**: 128포인트 전체를 8칸 간격으로, 길이 16/24/32 윈도우로 넓게 훑는 1차 스캔.
- **fine(촘촘한) 스캔**: 33~128 구간(초반부 제외)만 4칸 간격, 길이 12~32까지 더 다양하게 재탐색.
- `lo=33`이 임의 고정값이 아니라는 근거는 더 이전 단계의 원조 스크립트 [old_python/aec_region_search_surrogate.py](old_python/aec_region_search_surrogate.py)에 있습니다 — 그 파일의 [fine_windows_from_top()](old_python/aec_region_search_surrogate.py#L243-L253)은 coarse 스캔에서 internal 성적이 좋았던 상위 30개 윈도우 주변(±16)만 다시 훑는 **적응형** 범위를 썼고, 그 결과가 "거의 다 33번 지점 이후에 몰려 있었다"는 경험적 사실을 이 파일은 `lo=33` 하드코딩으로 단순화해 재현합니다. (자세한 설명은 [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md) §7-1 참고 — 그 문서가 이 근거를 먼저 상세히 정리했습니다.)
- 이 단계는 아직 R1~R4를 확정하지 않습니다 — "촘촘하게 훑어본 후보 목록"일 뿐이고, 사람이 그 상위권 결과를 해석해서 다음 절의 `LOCKED_REGIONS`로 고정합니다.

---

## 6. 4~5절: Locked region 확정과 branch 후보 스크리닝

### 6.1 `LOCKED_REGIONS` — [L109-L114](python/main_aec_full_derivation_pipeline.py#L109-L114)

```python
LOCKED_REGIONS = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}
```

region scout(§5) 결과와 사람의 시각적 해석을 바탕으로 고정된 4개 구간 — 다른 두 문서에서도 동일하게 재사용되는 정의입니다.

### 6.2 서술자 계산과 표준화

- [locked_region_descriptor_matrix(norm)](python/main_aec_full_derivation_pipeline.py#L415-L439): `LOCKED_REGIONS` 4개 구간마다 12개 서술자(level_mean/level_sd/endpoint_delta/linear_slope/slope_mean/slope_sd/abs_slope_mean/abs_slope_max/curv_mean/curv_sd/abs_curv_mean/abs_curv_max) 계산 — `aec_new_region_surrogate_combo_gate.py`의 `region_descriptor_matrix`와 동일 개념의 자체 구현.
- [standardize_features_by_internal(xg_df, xs_df)](python/main_aec_full_derivation_pipeline.py#L442-L453): internal 기준 중앙값 대치 → internal 평균/표준편차로 두 데이터셋 모두 z-표준화.

### 6.3 Branch(단일 구간·단일 서술자) 게이트 수식

- [branch_gate_score(clinical_z, feature_z, threshold, sign, width, lam)](python/main_aec_full_derivation_pipeline.py#L469-L471): `boundary = exp(-0.5*((clinical_z-threshold)/width)^2)`; `score = clinical_z + lam*boundary*sign*feature_z` — 다른 두 문서의 `gate_scores`와 동일한 수식.
- [branch_vote(...)](python/main_aec_full_derivation_pipeline.py#L474-L475): `score < threshold`이면 투표(True).
- [summarize_single_feature_rule(...)](python/main_aec_full_derivation_pipeline.py#L682-L711): S80/S85/S90 3개 운영점 각각에 대해 이 branch를 적용했을 때의 정확도/특이도 증가, 민감도 손실, 강등 인원을 계산해 평균·최솟값으로 요약 — region scout(§5)와 branch 스크리닝(아래) 모두 이 함수를 점수 기준으로 사용합니다.

### 6.4 `BranchCandidate`와 `screen_branch_candidates`

- [BranchCandidate](python/main_aec_full_derivation_pipeline.py#L186-L199) (frozen dataclass): region/feature/sign/width/lambda/score를 담는 후보 하나. `label` 프로퍼티가 `"{feature}__sign{sign}__w{width}__lam{lam}"` 형식의 고유 문자열을 만듭니다.
- [screen_branch_candidates(ctx)](python/main_aec_full_derivation_pipeline.py#L768-L861): `LOCKED_REGIONS` 4구간 × `DESCRIPTORS` 12개 × `SIGNS`(±1) × `WIDTHS`(0.35/0.50/0.70) × `LAMBDAS`(0.25/0.40/0.55/0.70) = 구간당 최대 288개 조합을 전수 채점(`02_locked_region_branch_screen.csv`)한 뒤, **구간별 상위 6개**만 선택합니다. 그리고 `LOCKED_PRIMARY_BRANCHES`(§6.5)에 실제로 쓰인 branch가 상위 6개 안에 없더라도 **강제로 후보 목록에 끼워 넣어**(`03_selected_branch_candidates_for_combo_search.csv`), "manuscript rule이 실제로 탐색 공간의 어디에 있는지" 항상 보이도록 만듭니다 (코드 주석, [L823-L825](python/main_aec_full_derivation_pipeline.py#L823-L825)).

### 6.5 `LOCKED_PRIMARY_BRANCHES` / `LOCKED_PRIMARY_PATTERNS` — [L118-L149](python/main_aec_full_derivation_pipeline.py#L118-L149)

R1 `endpoint_delta`(sign −1, width 0.50) / R2 `level_mean`(sign −1, width 0.70) / R3 `linear_slope`(sign +1, width 0.35) / R4 `endpoint_delta`(sign −1, width 0.50), lambda는 4개 모두 0.25. 인정 패턴은 `{"++++", "++--", "+--+", "--+-", "---+"}`. 이 값들은 다른 두 문서의 `BRANCHES`/`SELECTED_PATTERNS`와 **정확히 동일**합니다 — 이 파일이 그 값의 "원본 출처"임을 보여주는 핵심 상수입니다.

---

## 7. 6~7절: 4-branch 패턴 전수탐색 → Noninferiority lock → 최종 재현

### 7.1 4비트 코드화와 카운트 벡터화

- [precompute_branch_votes(ctx, candidates, xg, xs, names)](python/main_aec_full_derivation_pipeline.py#L864-L879): 선택된 모든 branch 후보 각각에 대해, 3개 운영점 × internal/external 전체 환자의 투표(True/False)를 미리 계산해 배열로 저장 — 이후 반복 탐색에서 매번 다시 계산하지 않도록 하는 캐시.
- [code_from_four_votes(votes4)](python/main_aec_full_derivation_pipeline.py#L882-L887): 4개 branch의 투표를 4비트 정수(0~15)로 압축.
- [fast_counts_by_code(y, cpos, code)](python/main_aec_full_derivation_pipeline.py#L890-L898) / [evaluate_mask_from_counts(...)](python/main_aec_full_derivation_pipeline.py#L901-L921): 16개 코드별 사건/비사건 환자 수를 세어두면, 이후 어떤 "패턴 마스크"(16코드 중 어느 조합을 AEC+로 볼지)를 시도하든 **환자를 다시 순회하지 않고** 그 카운트만 더해서 지표를 계산할 수 있습니다.
- [evaluate_all_masks_from_counts(...)](python/main_aec_full_derivation_pipeline.py#L940-L965) / [pattern_selector_matrix(masks)](python/main_aec_full_derivation_pipeline.py#L935-L937): 위 아이디어를 **행렬곱으로 한 번에** 수천 개 패턴 마스크에 대해 벡터화 — 코드 주석(§한글, [L947-L948](python/main_aec_full_derivation_pipeline.py#L947-L948))이 "한 branch 조합 안에서 4,368개 pattern mask를 한 번에 평가한다"고 설명합니다. `pattern_masks_exactly_k(k=5)`([L968-L975](python/main_aec_full_derivation_pipeline.py#L968-L975))가 만드는 "16개 코드 중 정확히 5개를 고르는" 조합 수가 `C(16,5)=4368`입니다.

### 7.2 Noninferiority(비열등성) 판정 — `clopper_pearson_one_sided_upper`

- [clopper_pearson_one_sided_upper(k, n, alpha=0.05)](python/main_aec_full_derivation_pipeline.py#L501-L506) / 벡터화 버전 [clopper_pearson_one_sided_upper_array(...)](python/main_aec_full_derivation_pipeline.py#L924-L932): 민감도 손실(놓친 저-SMI 환자 수 `k` / 전체 저-SMI 환자 수 `n`)의 **95% 단측 신뢰상한**을 Clopper-Pearson 정확법으로 계산.
- [evaluate_deescalation(y, clinical_positive, aec_positive)](python/main_aec_full_derivation_pipeline.py#L509-L539): 하나의 규칙에 대해 강등 전/후 민감도·특이도·정확도, 그리고 `formal_NI_pass`(민감도 손실 95% 상한이 `NI_MARGIN`=0.05 이하인가)까지 한 번에 계산 — 이 파일에서 "안전하다"의 공식적 정의.
- [run_combo_pattern_search(ctx, candidates, votes)](python/main_aec_full_derivation_pipeline.py#L978-L1063): `LOCKED_REGIONS` 4구간에서 후보를 하나씩 골라 만든 모든 조합(region당 최대 7개 후보 → 최대 7⁴=2,401가지 조합) × 4,368개 패턴 마스크를 전수 평가해, **internal과 external 모두** `upper95 ≤ 0.05` & 특이도 증가 > 0 & 정확도 증가 > 0을 만족하는 조합만 `04_combo_pattern_search_s90_formal_NI_candidates.csv`에 남깁니다(+ 상위 200개 `04_..._top200.csv`, + `LOCKED_PRIMARY_BRANCHES`/`LOCKED_PRIMARY_PATTERNS`와 정확히 일치하는 행만 `04_locked_primary_rule_row_from_search.csv`). 이 마지막 파일이 "manuscript primary rule(`new4_combo_261089`)이 실제로 이 전수탐색에서도 통과 후보로 나오는가"를 직접 증명하는 산출물입니다.
- [exact_mcnemar_p(gain_n, loss_n)](python/main_aec_full_derivation_pipeline.py#L563-L567): 이항검정 기반 p값 유틸이 정의되어 있지만, **파일 전체에서 실제로 호출되는 곳이 없습니다** — 죽은 코드(다른 유사 스크립트에서 쓰던 것을 정리 과정에서 남겨둔 것으로 보임).

### 7.3 최종 재현 — `compute_locked_gate` / `compute_secondary_cnn_mimic` / `write_final_outputs`

- [compute_locked_gate(ctx)](python/main_aec_full_derivation_pipeline.py#L575-L600): `LOCKED_PRIMARY_BRANCHES`/`LOCKED_PRIMARY_PATTERNS`를 그대로 적용해 internal/external 각각의 AEC+/− 판정을 계산 — 이것이 **1차(primary) 결과**.
- [compute_secondary_cnn_mimic()](python/main_aec_full_derivation_pipeline.py#L603-L617): **2차(secondary) 결과**. `CNN_PROBABILITY_NPZ`([L160](python/main_aec_full_derivation_pipeline.py#L160), `outputs/aec_new_region_cnn_surrogate_mimic_gate/surrogate_mimic_balanced_probabilities.npz`)를 읽어, `CNN_S90_INDEX=2`(S80/S85/S90 중 3번째)로 S90 시점 확률만 뽑고, `CNN_BRANCH_THRESHOLDS=[0.80, 0.60, 0.90, 0.60]` 및 `CNN_SELECTED_PATTERNS={"+---","---+","-+-+","++++"}`로 투표·패턴 판정을 재현합니다. 파일이 없으면 `None`을 반환해 2차 결과 없이 1차만 진행됩니다.
  - 코드 주석([L152-L159](python/main_aec_full_derivation_pipeline.py#L152-L159))에 이 상수들의 유래가 명시되어 있습니다: `outputs/MD/144838527.png`("Secondary: CNN-mimic Gate" 원본 스크린샷, S90 internal 강등 40명/external ~51-52명, TP lost 2/1)을 재현하는 값이며, **`main_aec_new_region_cnn_surrogate_mimic_gate.py`가 나중에 별도로 돌린 더 큰 전수탐색의 "새 승자"(`internal_locked`/`internal_external_audit`, [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md) §10.4)로 바꾸면 안 된다**는 경고까지 코드 주석에 남아 있습니다. `CNN_PROBABILITY_NPZ` 경로에 `work`가 빠져 있던 버그가 있었고, 2026-07-07에 수정되었다는 경위는 [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md) §11.1에 정리되어 있습니다(그 절이 설명하는 파일이 바로 이 파일입니다).
- [conditional_low_smi_table(y, clinical_positive, aec_positive)](python/main_aec_full_derivation_pipeline.py#L542-L560): 임상양성 내 AEC+/AEC- 두 그룹의 실제 저-SMI 사건율과 Fisher 정확검정 p값 — 다른 두 문서의 §10.1과 같은 개념의 표.
- [make_main_figure(ctx, aec_positive_external, path)](python/main_aec_full_derivation_pipeline.py#L1122-L1165): `main_plot_external_s90_core_1x3_mean_curves.py`의 PNG #2(1x3 + R4 tangent)와 **거의 동일한 그림**을 다시 그리는 자체 구현(`mean_ci`/`add_region_spans`/`plot_group_mean`/`add_r4_tangent` 헬퍼 포함, [L1071-L1119](python/main_aec_full_derivation_pipeline.py#L1071-L1119)) — 다만 이 파일은 **1차 규칙(primary)만** 그리며, 2차(CNN) 그림은 별도로 그리지 않습니다.
- [write_final_outputs(ctx, primary_g, primary_s, cnn)](python/main_aec_full_derivation_pipeline.py#L620-L674): 위 결과들을 모아 §9의 4개 파일로 저장합니다.

---

## 8. 실제 산출 결과 (`reproduce` 모드 실행분, `python/full_derivation_output/`)

### 8.1 1차(primary)/2차(CNN) S90 지표 — `final_s90_primary_and_cnn_metrics.csv`

| model | cohort | 강등 인원 | 그중 저-SMI(TP lost) | 사건율 | 정확도 증가 | 특이도 증가 | 민감도 손실 | 95% 상한 (≤0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_interpretable_4region | Gangnam internal | 53 | 2 | 3.8% | +4.50%p | +5.31%p | 1.55%p | 0.0480 ✅ |
| primary_interpretable_4region | Sinchon external | 56 | 2 | 3.6% | +5.62%p | +6.88%p | 1.42%p | 0.0440 ✅ |
| secondary_CNN_mimic | Gangnam internal | 40 | 2 | 5.0% | +3.30%p | +3.95%p | 1.55%p | 0.0480 ✅ |
| secondary_CNN_mimic | Sinchon external | 51 | 1 | 2.0% | +5.29%p | +6.37%p | 0.71%p | 0.0332 ✅ |

4개 행 모두 `formal_NI_pass = True` — 1차·2차, internal·external 어디서도 공식 5%p 비열등성 기준을 통과합니다. CNN 쪽(2차)이 강등 인원은 더 적지만(더 보수적), 그만큼 민감도 손실 여유도 더 큽니다.

### 8.2 임상양성 내 AEC 조건부 분리 — `final_clinical_positive_aec_split.csv`

| model | cohort | AEC+ n (사건율) | AEC- n (사건율) | Fisher p |
| --- | --- | --- | --- | --- |
| primary_interpretable_4region | internal | 53 (3.8%) | 518 (22.2%) | 5.53×10⁻⁴ |
| primary_interpretable_4region | external | 56 (3.6%) | 480 (26.9%) | 2.30×10⁻⁵ |
| secondary_CNN_mimic | internal | 40 (5.0%) | 531 (21.7%) | 7.97×10⁻³ |
| secondary_CNN_mimic | external | 51 (2.0%) | 485 (26.8%) | 8.87×10⁻⁶ |

1차(primary) 행의 수치는 [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md) §10.1, [main_plot_internal_s90_core_1x3_mean_curves.md](main_plot_internal_s90_core_1x3_mean_curves.md) §10.1과 **정확히 일치**합니다 — 세 스크립트가 서로 다른 코드로 같은 결과를 재현하고 있다는 교차 확인입니다. 2차(CNN) 행은 [main_aec_new_region_cnn_surrogate_mimic_gate.md](main_aec_new_region_cnn_surrogate_mimic_gate.md) §11.1의 "재현 확인 결과"(internal 40명/2건, external 51명/1건, TP lost 2/1)와도 정확히 일치합니다.

### 8.3 임상 z-score 임계값 — `final_summary.json`

`clinical_thresholds_z`: S80=0.324, S85=0.199, S90=−0.036 (internal 5-fold OOF 임상 점수 기준 표준화 값).

---

## 9. 산출물 파일 정리 (`OUT_DIR = python/full_derivation_output/`)

| 파일 | 생성 모드 | 내용 |
| --- | --- | --- |
| `01_region_scout_window_feature_ranked.csv` | `full-search`만 | 모든 후보 윈도우 × 서술자 × sign/width/lambda 조합의 internal 점수 순위표 |
| `02_locked_region_branch_screen.csv` | `full-search`만 | `LOCKED_REGIONS` 4구간 × 12서술자 × sign/width/lambda 전체 스크리닝 결과 |
| `03_selected_branch_candidates_for_combo_search.csv` | `full-search`만 | 구간별 상위 6개 + 강제 포함된 locked branch 목록 |
| `04_combo_pattern_search_s90_formal_NI_candidates.csv` | `full-search`만 | 공식 5%p 비열등성을 통과(또는 locked mask와 일치)하는 4-branch×패턴 조합 전체 |
| `04_combo_pattern_search_s90_top200.csv` | `full-search`만 | 위 표의 상위 200행 |
| `04_locked_primary_rule_row_from_search.csv` | `full-search`만 | `LOCKED_PRIMARY_BRANCHES`/`PATTERNS`와 정확히 일치하는 행만 추출 — "manuscript rule이 탐색에서도 나온다"는 증거 |
| `final_s90_primary_and_cnn_metrics.csv` | 두 모드 공통 | §8.1의 출처 |
| `final_clinical_positive_aec_split.csv` | 두 모드 공통 | §8.2의 출처 |
| `final_external_s90_1x3_r4_tangent.png` | 두 모드 공통 | external 1차 규칙의 1x3(+R4 tangent) 그림 — `main_plot_external_s90_core_1x3_mean_curves.py`의 PNG #2와 같은 스타일 |
| `final_summary.json` | 두 모드 공통 | §8.3의 출처 — 1차/2차 규칙 정의와 임계값을 모두 담은 최종 요약, **가장 먼저 열어봐야 할 파일** |

---

## 10. 다른 파일과의 관계 정리

| 파일 | 관계 |
| --- | --- |
| [python/aec_lock_smoothed_deesc_gate.py](python/aec_lock_smoothed_deesc_gate.py), [python/aec_conditional_value.py](python/aec_conditional_value.py), [python/aec_universal_boundary_gate.py](python/aec_universal_boundary_gate.py) | import 관계 없음. 데이터 로딩/임상 모델 로직을 이 파일 안에 독립적으로 다시 구현(§3~4) |
| [python/aec_new_region_surrogate_combo_gate.py](python/aec_new_region_surrogate_combo_gate.py) | import 관계 없음. `region_descriptor_matrix`에 해당하는 로직을 `locked_region_descriptor_matrix`로 다시 구현(§6.2). 이 파일의 `main()`이 원래 4-branch 탐색을 처음 수행한 스크립트였고, 이 파일은 그 절차 전체를 통합·재현 |
| [python/main_plot_external_s90_core_1x3_mean_curves.py](python/main_plot_external_s90_core_1x3_mean_curves.py) / [python/main_plot_internal_s90_core_1x3_mean_curves.py](python/main_plot_internal_s90_core_1x3_mean_curves.py) | 이 두 스크립트의 `BRANCHES`/`SELECTED_PATTERNS` 상수의 **도출 근거**가 이 파일의 `LOCKED_PRIMARY_BRANCHES`/`LOCKED_PRIMARY_PATTERNS`(§6.5)입니다. 값은 동일하지만 import 관계는 없음(각자 하드코딩) |
| [python/main_aec_new_region_cnn_surrogate_mimic_gate.py](python/main_aec_new_region_cnn_surrogate_mimic_gate.py) | 이 파일이 저장한 `outputs/aec_new_region_cnn_surrogate_mimic_gate/surrogate_mimic_balanced_probabilities.npz`를 `compute_secondary_cnn_mimic()`이 파일 경로로 읽어들임(§7.3) — 유일한 실질적 데이터 의존 관계 |
| [old_python/aec_region_search_surrogate.py](old_python/aec_region_search_surrogate.py) | region scout의 coarse/fine 스캔 파라미터(`lo=33` 등)가 유래한 더 이전 단계의 원조 탐색 스크립트(§5.1) |

---

## 11. 임상적 배경

AEC의 정의, SMI/TAMA 컷오프, S80/S85/S90의 의미, de-escalation이 필요한 임상적 동기, R1~R4가 해부학적 구조물이 아니라 데이터 기반으로 고정된 구간이라는 점은 이 파일에서도 코드 수준에서 동일하게 확인되며(§4~6), 자세한 임상적 설명은 이미 [main_plot_external_s90_core_1x3_mean_curves.md](main_plot_external_s90_core_1x3_mean_curves.md) §9에 정리되어 있으므로 중복 서술하지 않습니다. 다만 이 파일의 모듈 docstring([L1-73](python/main_aec_full_derivation_pipeline.py#L7-L73))이 10단계 분석 흐름을 코드 저자 자신의 말로 요약한 유일한 곳이므로, 다른 두 문서와 표현이 다르게 느껴지는 부분이 있다면 이 docstring을 1차 출처로 우선하십시오.
