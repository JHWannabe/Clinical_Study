# `run_from_raw.py` 실행 흐름 완전 정리 (비전공자용)

이 문서는 `code/run_from_raw.py`를 실행했을 때 컴퓨터 안에서 실제로 무슨 일이,
어떤 순서로 벌어지는지를 **위에서 아래로(top-down)**, 하나도 건너뛰지 않고 설명합니다.
마지막에는 지금 `outputs/` 폴더에 실제로 만들어져 있는 파일들을 열어보고
"지금까지 어디까지 진행됐는지"도 분석해 두었습니다.

---

## 0. 이 스크립트가 하는 일 한 줄 요약

> 환자 CT에서 뽑은 "AEC 128포인트 곡선"이라는 신호 데이터를 이용해서,
> "이 환자가 근육량이 적은(저-SMI) 사람인지"를 예측하는 점수를 만들고,
> 그 점수가 임상적으로 위험군인 환자들 중에서도 의미 있게 구분이 되는지 검증하는
> 4단계 분석을 raw 엑셀 파일 2개만 가지고 처음부터 끝까지 자동으로 재현하는 스크립트입니다.

필요한 원본 파일은 단 2개입니다.

```
data/g1090.xlsx   ← 내부(Gangnam) 코호트, 모델을 만들고 잠그는(lock) 데 사용
data/sdata.xlsx   ← 외부(Sinchon) 코호트, 내부에서 정한 규칙을 "검증만" 하는 데 사용
```

실행 명령:

```bash
python code/run_from_raw.py
```

---

## 1. 전체 그림 (4단계 파이프라인)

```
[data/g1090.xlsx, data/sdata.xlsx]  (raw 엑셀 원본, 사람이 준비)
        │
        ▼
Stage 1  잠긴(locked) AEC 특징 찾기
        │  → outputs/aec_lock_smoothed_deesc_gate/locked_gate_features.csv
        ▼
Stage 2  영역별 CNN이 곡선 모양을 보고 "확률" 예측
        │  → outputs/aec_region_cnn_pattern_gate/direct_vote_probabilities.npz
        ▼
Stage 3  그 확률들을 하나의 "환자별 AEC 점수"로 합치기
        │  → outputs/aec_direct_vote_auc_boost/direct_vote_auc_boost_scores.csv
        ▼
Stage 4  임상 고위험군 안에서 AEC 점수로 저-SMI 환자를 더 잘 골라내는지 최종 검증
        │  → outputs/aec_final_global_quintile_phenotype/*.csv, *.png
        ▼
     최종 결과 (논문에 쓰이는 표/그림)
```

각 단계는 이전 단계가 만든 파일을 입력으로 받는 "릴레이" 구조입니다.
그래서 스크립트는 중간 CSV 파일들을 감추지 않고 그대로 디스크에 남겨서,
학생이 각 단계 결과를 직접 열어볼 수 있게 만들어져 있습니다.

---

## 2. 코드 최상단 — 무엇을 불러오는가 (1~76번째 줄)

```python
import argparse       # 커맨드라인 옵션(--force, --dry-run 등)을 읽기 위한 표준 도구
import importlib      # 패키지가 설치되어 있는지 동적으로 확인하기 위한 도구
import sys, time
from pathlib import Path
import numpy as np
```

파일 맨 위의 긴 주석(따옴표 3개로 감싼 docstring)은 사람이 읽는 설명서로,
4단계가 각각 무엇을 입력받고 무엇을 출력하는지 미리 요약해 둔 것입니다.
실제 동작에는 영향을 주지 않고, 코드를 읽는 사람을 위한 "지도" 역할만 합니다.

---

## 3. 작은 도우미 함수들 (82~192번째 줄)

프로그램이 실제로 `main()`을 실행하기 전에, 먼저 아래의 작은 부품(함수)들이 정의됩니다.
정의만 되고 아직 호출되지는 않습니다 (파이썬은 위에서 아래로 읽지만, 함수는 "정의"와 "호출"이 분리되어 있습니다).

| 함수 | 하는 일 (쉬운 말로) |
|---|---|
| `timestamp()` | 지금 몇 시 몇 분 몇 초인지 문자열로 반환 |
| `say(message)` | `[14:32:01] 메시지` 형태로 화면에 즉시 출력 (진행 상황 알림용) |
| `require_file(path, ...)` | 파일이 없으면 바로 에러를 내서 멈춤. pandas가 나중에 이상한 긴 에러를 내는 것보다 친절하게 "이 파일이 없어요"라고 알려주기 위함 |
| `check_python_dependencies()` | numpy, pandas, scipy, sklearn, statsmodels, torch, matplotlib, openpyxl이 설치되어 있는지 확인. 하나라도 없으면 설치를 요구하며 즉시 중단 |
| `run_step(step_name, marker, force, action)` | 아래에서 자세히 설명 |
| `reset_shared_random_state(mods)` | 아래에서 자세히 설명 |
| `stage_action(mods, action)` | 위 두 개를 묶어주는 포장지 |

### 3-1. `run_step` — "이미 했으면 건너뛰기" 로직

```python
def run_step(step_name, marker, force, action):
    if marker.exists() and not force:
        say(f"SKIP {step_name}: found {marker.name}")
        return
    ...
    action()
    ...
    if not marker.exists():
        raise RuntimeError(...)
```

- `marker`는 "이 단계가 이미 성공적으로 끝났다"는 증거가 되는 파일입니다.
  (예: Stage 1의 marker는 `locked_gate_features.csv`)
- 그 marker 파일이 **이미 존재하면**, 이번 단계는 통째로 건너뜁니다. (다시 몇 시간씩 CNN을 학습하지 않기 위한 캐싱)
- `--force` 옵션을 주면 강제로 marker를 지우고 다시 실행합니다.
- 단계를 실행한 뒤에도 marker 파일이 안 생겼다면 "결과물이 안 나왔다"는 뜻이므로 에러를 냅니다.

### 3-2. `reset_shared_random_state` — 재현성을 지키기 위한 안전장치

원래 연구자는 이 4단계를 각각 **별도의 파이썬 프로세스**로 하나씩 실행했습니다.
그러면 매번 프로그램이 새로 시작되므로 난수 생성기도 매번 초기화됩니다.

그런데 `run_from_raw.py`는 4단계를 **하나의 프로세스 안에서 연달아** 호출합니다.
이렇게 하면 예전 코드(`aec_conditional_value.py`)가 가진 전역 난수 생성기(`RNG`)가
이전 단계에서 이미 여러 번 사용된 상태로 남아있게 되고,
그 결과 환자를 5개 그룹(fold)으로 나누는 방식이 단계마다 미묘하게 달라질 수 있습니다.
이는 "같은 코드인데 실행할 때마다 결과가 조금씩 달라지는" 재현성 문제로 이어집니다.

그래서 각 단계를 실행하기 **직전에** 항상 같은 시드(seed)로 난수 생성기를 다시 만들어서,
매 단계가 항상 "방금 새로 시작한 것"과 같은 상태에서 시작하도록 강제합니다.

---

## 4. 경로 바꿔치기 (199~308번째 줄)

원래 연구용 코드들은 연구자 개인 컴퓨터의 경로(`C:\Users\user\OneDrive\...`)가
소스코드 안에 그대로 박혀 있었습니다. 학생마다 이 경로들을 일일이 손으로 고치게 하는 대신,
`run_from_raw.py`가 프로그램을 실행하는 시점에 그 경로들을 자동으로 학생의 폴더 구조로 바꿔치기합니다.

### 4-1. `import_pipeline_modules(code_dir)`

```python
sys.path.insert(0, str(code_dir))
import aec_conditional_value as conditional
import aec_universal_boundary_gate as universal
...
import aec_final_global_quintile_phenotype_pipeline as final_analysis
```

- `code/` 폴더를 파이썬이 모듈을 찾는 검색 경로의 맨 앞에 끼워 넣습니다.
  (혹시 컴퓨터 다른 곳에 같은 이름의 예전 파일이 있어도, 학생 폴더 안의 파일을 우선 사용하도록)
- `code/` 폴더 안에 있는 12개의 연구용 파이썬 파일을 전부 불러와서, 이름표를 붙인 딕셔너리(`mods`)로 돌려줍니다.
  이 시점에서는 아직 아무 계산도 하지 않고, "불러오기(import)"만 합니다.

### 4-2. `patch_module_paths(mods, project_root, data_dir, output_root)`

이 함수는 방금 불러온 각 모듈이 내부적으로 가지고 있는 `DATA_DIR`, `OUT_DIR` 같은
"경로 변수"들을 전부 학생의 실제 폴더(`data/`, `outputs/아무개폴더`)로 덮어씁니다.

즉,
- 4단계 각각의 출력 폴더 11개를 미리 만들어 둡니다 (`outputs/aec_lock_smoothed_deesc_gate` 등).
- Stage 1 모듈(`locked`)에게 "네 데이터는 `data/` 폴더에 있고, 네 결과는 여기에 저장해"라고 알려줍니다.
- Stage 2 모듈(`pattern_gate`)에게도 마찬가지로 알려주고, CNN 확률을 저장할 파일 경로(`PROB_CACHE`)도 지정해 줍니다.
- Stage 3 모듈(`auc_boost`)에게는 "네 입력은 Stage 2가 만든 그 확률 파일이야"라고 연결해 줍니다 (`PROB_PATH`).
- Stage 4 모듈(`final`)에게는 "네 입력은 Stage 3가 만든 점수 CSV야"라고 연결해 줍니다 (`DIRECT_VOTE_SCORE_CSV`).

이렇게 해서 4개의 스테이지가 서로의 결과물을 정확히 어디서 찾아야 하는지 자동으로 이어집니다.
학생은 경로를 손으로 편집할 필요가 없습니다.

---

## 5. `main()` 함수 — 실제 실행 순서 (315~401번째 줄)

`python code/run_from_raw.py`를 치면 맨 마지막 줄의

```python
if __name__ == "__main__":
    main()
```

에 의해 `main()`이 호출되고, 여기서부터 진짜 실행이 시작됩니다. 순서대로:

### 5-1. 커맨드라인 옵션 읽기

```
--data-dir     기본값: code/의 상위 폴더/data   (즉 data/g1090.xlsx, data/sdata.xlsx)
--output-dir   기본값: code/의 상위 폴더/outputs
--force        이미 끝난 단계도 강제로 다시 실행
--dry-run      실제로 모델을 학습하지 않고, 파일/경로만 점검
```

### 5-2. 경로 안내 출력

`project_root`, `data_dir`, `output_root`를 계산해서 화면에 출력합니다. (지금 코드가 어디를 보고 있는지 확인용)

### 5-3. 원본 파일 존재 확인

```python
require_file(data_dir / "g1090.xlsx", "internal/Gangnam raw Excel file")
require_file(data_dir / "sdata.xlsx", "external/Sinchon raw Excel file")
```

두 엑셀 파일이 `data/` 폴더에 실제로 있는지 먼저 확인합니다.

### 5-4. 필수 패키지 확인

`check_python_dependencies()`가 8개 패키지(numpy, pandas, scipy, sklearn, statsmodels, torch, matplotlib, openpyxl)를 import해 보고, 버전을 출력합니다.

### 5-5. 모듈 불러오기 + 경로 연결

앞서 설명한 `import_pipeline_modules`, `patch_module_paths`를 호출해서
12개 연구 모듈을 불러오고 경로를 학생 폴더에 맞게 다시 씁니다.

### 5-6. 예정된 출력 파일 미리 안내

4단계 각각의 최종 산출 파일 경로를 미리 화면에 찍어 줍니다.

### 5-7. `--dry-run`이면 여기서 종료

모델을 하나도 학습하지 않고 "점검만 끝났다"고 말하고 프로그램을 끝냅니다.

### 5-8. Stage 1~4를 순서대로 실행

```python
run_step("Stage 1/4: internal locked feature search",
         paths["lock_dir"] / "locked_gate_features.csv",
         args.force,
         stage_action(mods, mods["locked"].main))

run_step("Stage 2/4: region-guided CNN branch probabilities",
         paths["pattern_dir"] / "direct_vote_probabilities.npz",
         args.force,
         stage_action(mods, mods["pattern_gate"].main))

run_step("Stage 3/4: direct-vote AEC score generation",
         paths["boost_dir"] / "direct_vote_auc_boost_scores.csv",
         args.force,
         stage_action(mods, mods["auc_boost"].main))

run_step("Stage 4/4: final global quintile phenotype analysis",
         paths["final_dir"] / "01_quintile_vs_quartile_enrichment.csv",
         args.force,
         stage_action(mods, mods["final"].main))
```

각 줄은 "이 단계의 결과 marker 파일이 이미 있으면 건너뛰고, 없으면 해당 모듈의 `main()` 함수를 실제로 실행해서 만들어라"는 뜻입니다.

### 5-9. 완료 메시지

4단계가 모두 끝나면 "All stages complete."를 출력하고, 최종 결과 파일 위치를 다시 한번 알려줍니다.

---

## 6. 각 Stage 내부에서 실제로 일어나는 일 — 상세 설명

---

### Stage 1 — `aec_lock_smoothed_deesc_gate.py` — "잠긴(locked) AEC 특징 찾기"

**파일:** `code/aec_lock_smoothed_deesc_gate.py`
**입력:** `data/g1090.xlsx`, `data/sdata.xlsx`
**출력(마커):** `outputs/aec_lock_smoothed_deesc_gate/locked_gate_features.csv`

---

#### Stage 1을 한 줄로 이해하기

> "환자의 X선 노출 곡선(128개 숫자)을 부드럽게 다듬고, 그 모양에서 '이 환자가 근육이 적을 가능성'을
> 알려주는 핵심적인 특징을 찾은 다음, 그 규칙을 확정(lock)해서 다음 단계에 넘겨준다."

---

#### 배경 지식: AEC 128포인트 곡선이란?

CT 촬영을 할 때, 방사선 기계는 환자의 몸을 통과하는 X선 양을 자동으로 조절합니다 (Automatic Exposure Control, AEC).
환자의 몸이 두꺼울수록(예: 복부 지방이 많을수록) 기계가 더 강한 X선을 쏩니다.
이 조절값을 촬영 위치(위→아래로 128개 지점)마다 기록한 것이 "AEC 128포인트 곡선"입니다.

중요한 점은, 이 곡선의 **모양**이 단순히 "환자가 뚱뚱한지 말랐는지"뿐 아니라,
근육과 지방의 분포 방식에 따라서도 달라진다는 것입니다.
Stage 1은 바로 그 "곡선의 어떤 모양 특징이 근육량 부족(저-SMI)과 연관 있는가"를 찾습니다.

---

#### 세부 단계별 설명

**① 데이터 불러오기 + 다듬기** (`load_dataset`)

```python
smooth_raw = ndimage.gaussian_filter1d(raw, sigma=1.0, axis=1, mode="nearest")
norm = row_norm(smooth_raw)  # 각 환자 본인의 평균으로 나누기
y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
```

- 엑셀의 `metadata` 시트에서 나이·키·체중·성별·근육량(TAMA)을 읽고,
  `aec_128` 시트에서 환자별 128개 숫자 곡선을 읽습니다.
- **가우시안 스무딩(sigma=1):** 곡선에서 의미 없는 잡음(noise)을 살짝 지우는 작업입니다.
  마치 흔들린 사진을 약간 보정하는 것과 같습니다. `sigma=1`이면 아주 약하게 부드럽게 만드는 것입니다.
- **정규화(row_norm):** 각 환자의 곡선 전체를 그 환자 자신의 평균값으로 나눕니다.
  이렇게 하면 원래 신호 세기가 다른 환자들끼리도 "곡선의 모양"만 공평하게 비교할 수 있습니다.
  예를 들어, 키가 크거나 비만인 환자는 AEC 값 자체가 크지만, 정규화 후에는 모두 평균=1 부근으로 맞춰집니다.
- **저-SMI 정의:** 근골격 지수(SMI = 근육 면적 / 키²)가 남성은 45.4, 여성은 34.4 미만이면 저-SMI(`y=1`)로 분류합니다.
  이것이 이 프로젝트 전체에서 예측하고자 하는 "정답"(label)입니다.

---

**② 임상 점수 계산** (`clinical_scores`)

```python
xg, xs, _ = clinical_matrix(g["meta"], s["meta"])  # 나이, 키, 체중, 성별
folds = make_folds(g["y"], 5)                        # 환자를 5개 그룹으로 나눔
clinical_oof, clinical_ext = oof_and_external(...)   # 5-fold 교차검증
```

- 나이·키·체중·성별 **4개 변수만** 사용해서 "임상 점수"를 만듭니다. (AEC 곡선은 전혀 사용 안 함)
- **5-fold 교차검증(cross-validation):** 환자 전체를 5개 묶음(fold)으로 나누고, 4개로 모델을 학습한 뒤
  나머지 1개 묶음을 예측합니다. 이걸 5번 반복하면 모든 환자에 대해 "자기 자신을 보지 않은 상태에서"
  예측한 점수(OOF, Out-Of-Fold)를 얻을 수 있습니다.
  → 이 점수가 "임상 정보만으로 예측한 저-SMI 위험도"입니다.
- **민감도 기준선(S80~S90) 문턱값 계산:**
  민감도(sensitivity) = "실제 저-SMI 환자 중 몇 %를 위험군으로 잡아내는가"를 뜻합니다.
  S80이면 80% 이상의 저-SMI를 잡아낼 수 있는 점수 문턱값을 찾습니다.
  이 5가지 문턱값(S80, S82.5, S85, S87.5, S90)이 Stage 1의 "임상 고위험군" 정의에 사용됩니다.

---

**③ 후보 특징 대량 생성** (`build_candidate_bank`)

```python
dense = build_feature_bank(norm).add_prefix("smooth_norm__")
visual = build_visual_norm_bank(norm).add_prefix("smooth_visual__")
```

정규화된 128포인트 곡선에서 수학적으로 뽑아낼 수 있는 특징들을 **기계적으로 대량 생성**합니다.
이번 실행에서는 약 12,000개 이상의 특징이 생성됩니다.

생성되는 특징 종류:

| 특징 종류 | 예시 | 의미 |
|---|---|---|
| 구간 평균 | `norm_076_090_mean` | 76~90번 포인트 구간의 평균 높이 |
| 구간 표준편차 | `norm_082_085_sd` | 82~85번 구간이 얼마나 들쑥날쑥한가 |
| 구간 최소/최대 | `norm_094_099_max` | 94~99번 구간의 최고점 |
| 1차 기울기(slope) | `norm_slope_082_085_sd` | 82~85번 구간에서 곡선이 얼마나 가파르게 변하는가 |
| 2차 곡률(curvature) | `norm_curv_103_110_mean` | 103~110번 구간이 얼마나 휘어있는가 |
| 시각적 특징 | `visual_trough_depth__...` | 초반·중반·후반 구간 비교 (곡선의 골짜기 깊이) |
| 전체 진동성 | `visual_global_waviness_abs_slope_mean` | 곡선 전체가 얼마나 울퉁불퉁한가 |

> **비유:** 마치 사람의 필적(손글씨)에서 "획의 기울기", "필압의 변동", "글자 높이" 같은 특징을 수천 가지
> 자동으로 측정하는 것과 같습니다.

---

**④ 1차 거르기** (`prescreen_feature_indices`)

```python
score = global_score + 0.7 * cp_score + semantic
order = np.argsort(score)[::-1]
return order[:600]
```

12,000개 특징 중에서 상위 **600개**만 남깁니다. 선택 기준은:

- **global_score:** 전체 내부 코호트에서 "임상 점수로 설명 안 되는 나머지(잔차)"와 얼마나 상관이 있는가
- **cp_score:** 임상 고위험군 환자들 안에서 저-SMI를 얼마나 잘 구분하는가 (0.7배 가중치)
- **semantic:** 곡률, 기울기, 하르(haar), 진동성 같은 "해석 가능한 모양 특징"에 추가 점수(+0.08)를 줌
  → 해석이 어려운 복잡한 특징보다 "의미 있는 곡선 모양 특징"을 우선시하기 위한 조치

---

**⑤ 방향 정렬** (`risk_direction`)

각 특징이 "값이 클수록 저-SMI 위험이 높은가, 낮은가"를 결정합니다.

- 임상 잔차(저-SMI와 연관된 성분)와 양의 상관이면: 큰 값 = 위험
- 음의 상관이면: 작은 값 = 위험

이렇게 방향을 통일해야, 나중에 "이 특징이 위험 방향으로 얼마나 돌아섰는가"를 일관되게 비교할 수 있습니다.

---

**⑥ 단일 특징 스크리닝** (`feature_screen`)

```python
for j, name in enumerate(names):     # 600개 특징 각각에 대해
    for width in [0.35, 0.50, 0.70]: # 경계 폭 3가지
        for lam in [0.25, 0.40, 0.55, 0.70]:  # 혼합 강도 4가지
            # "이 특징을 이용해서 고위험군 일부를 안전하게 저위험으로 재분류할 수 있는가?" 검사
```

총 600 × 3 × 4 = **7,200가지 조합**을 시험합니다 (이것이 `internal_single_feature_screen.csv`의 7,200개 행).

핵심 아이디어 — "de-escalation(위험 강등)"이란:

- "임상 점수로는 고위험군에 들어오지만, AEC 특징을 보니 실제로는 저위험으로 봐도 될 것 같다" → 안전하게 고위험군에서 빼 줌
- 단, 이 과정에서 실제 저-SMI 환자를 놓치면 안 됨 (민감도 손실 < 0.08, 즉 최대 8% 이내)
- 특이도(실제 정상을 정상으로 판별)는 반드시 개선되어야 함
- de-escalate된 환자 수가 최소 25명은 되어야 함 (너무 적으면 우연일 수 있음)

이 조건을 통과하지 못하는 특징은 탈락시킵니다.

또한 각 특징에는 **`company_eta2`**라는 값도 계산됩니다:

- CT 기기 제조사(Siemens, Philips, GE)에 따라 이 특징 값이 얼마나 달라지는지 측정
- 값이 0에 가까울수록 "기기 종류와 무관한 순수한 신호" → 더 신뢰할 수 있는 특징
- 최종 점수에서 `-0.05 × company_eta2` 항을 빼서 장비 의존적인 특징을 불이익 줌

---

**⑦ 다양성 있는 후보 풀 만들기** (`diverse_combo_pool`)

600개 중 스크리닝을 통과한 특징들에서, 서로 너무 비슷하거나 한 종류가 너무 많아지지 않도록 골라:
최종 **18개**의 "다양한" 후보 특징을 선정합니다.

규칙:

- 서로 **상관관계 0.92 이상**인 특징들은 하나만 남김 (중복 정보 제거)
- 같은 종류(예: 기울기류, 곡률류)가 **5개를 넘으면** 초과분은 제거 (특정 종류로 쏠림 방지)

→ 이 18개가 `outputs/aec_lock_smoothed_deesc_gate/internal_combo_feature_pool.csv`로 저장됩니다.

---

**⑧ 조합 탐색 및 규칙 잠금** (`combo_search`) — **가장 시간이 오래 걸리는 단계**

```python
for m in range(1, MAX_COMBO_M + 1):  # 1개~4개씩 조합
    for combo in itertools.combinations(pool_indices, m):
        for k in range(1, m + 1):     # k개 이상 동의하면 de-escalate
```

18개 특징을 **1~4개씩 묶은 모든 조합**에 대해:

- "이 m개의 특징 중 k개 이상이 '안전하다'고 동의하면 de-escalation" 이라는 투표 규칙을 시험
- 위 스크리닝 기준(민감도 손실 < 0.08, 특이도 개선, de-escalate 수 ≥ 25, Fisher p 등)을 **내부 코호트에서만** 통과하는 조합 중
- 점수가 가장 높은 **하나의 조합 + k값**을 선택해서 **확정(lock)**합니다.

> **왜 "잠근다(lock)"고 하는가?**
> 이 규칙을 정할 때 외부 코호트(sdata.xlsx) 데이터는 전혀 보지 않습니다.
> 내부 데이터만으로 규칙을 확정한 뒤, 외부 데이터에 그 규칙을 그대로 적용해서
> "같은 규칙이 처음 보는 환자에게도 통하는가"를 검증합니다.
> 만약 외부 데이터를 보면서 규칙을 조정했다면 그건 "정답을 보고 시험을 푸는 것"이므로 연구 결과를 믿을 수 없습니다.

---

**⑨~⑪ 결과 정리 및 저장**

- 잠긴 규칙을 외부 코호트(sdata)에도 그대로 적용하여 성능표 생성
- "임상 점수만", "AEC 특징만", "임상+AEC 결합" 세 가지의 AUC(예측 정확도) 비교표 생성
- 그림 1장(`locked_gate_operating_points.png`)과 CSV/JSON 여러 개 저장

**최종 산출물(마커 파일):** `locked_gate_features.csv` — 잠긴 특징들의 목록

---

### Stage 2 — `aec_region_cnn_pattern_gate.py`
### "영역별 CNN이 곡선 모양을 학습"

**파일:** `code/aec_region_cnn_pattern_gate.py` + `aec_region_constrained_cnn_gate.py` + 관련 모듈들
**입력:** Stage 1의 `locked_gate_features.csv` + `data/g1090.xlsx`, `data/sdata.xlsx`
**출력(마커):** `outputs/aec_region_cnn_pattern_gate/direct_vote_probabilities.npz`

---

#### Stage 2를 한 줄로 이해하기

> "Stage 1이 손으로 고른 특징들을 보고, AI(CNN)가 '같은 곡선 영역에서 비슷한 위험 패턴을
> 스스로 학습'하게 한다. 그 결과로 각 환자·각 구간에 대한 위험 확률을 만들어낸다."

---

#### 배경 지식: CNN이란?

CNN(Convolutional Neural Network, 합성곱 신경망)은 원래 이미지 인식에 사용되는 AI 기법입니다.
이 프로젝트에서는 "128포인트 곡선"을 1차원 이미지처럼 취급하여, AI가 곡선의 특정 구간에서
패턴을 스스로 학습하게 합니다.

---

#### 세부 단계별 설명

**① 입력 채널 만들기** (`make_channels`)

각 환자의 곡선을 3가지 방식으로 변환하여 "3채널 입력"을 만듭니다:
- **채널 1 — 원본 곡선:** 정규화된 AEC 값 그 자체 (곡선의 높낮이)
- **채널 2 — 1차 기울기:** 인접한 두 포인트의 차이 (곡선이 올라가는지, 내려가는지)
- **채널 3 — 2차 곡률:** 기울기의 변화율 (곡선이 얼마나 급격하게 방향을 바꾸는가)

> **비유:** 심전도(ECG) 파형을 분석할 때, 파형 자체뿐 아니라 파형의 "속도"와 "가속도"도 함께 본다고 생각하세요.

또한 각 채널은 환자 개인 기준으로 표준화(mean=0, std=1)하여 모든 환자가 같은 스케일에서 비교되게 합니다.

---

**② 4개의 관심 영역(REGIONS) 정의**

```python
REGIONS = {
    "R1_slope_around_082_085": (76, 92),   # 82~85번 포인트 근처
    "R2_abs_slope_around_094_099": (88, 106),  # 94~99번 포인트 근처
    "R3_curv_around_103_110": (96, 118),   # 103~110번 포인트 근처
    "R4_curv_around_097_100": (90, 110),   # 97~100번 포인트 근처
}
```

Stage 1에서 가장 중요하다고 선택된 4개 특징이 집중된 **구간들 주변**을 조금 넓게 잡아서
CNN이 그 구간을 집중적으로 들여다보게 합니다.

> 왜 이 구간인가? Stage 1의 스크리닝 결과 상위권이 모두 AEC 곡선의 76~118번 포인트(전체 128개 중 후반부,
> CT 촬영 시 복부 하단 통과 구간에 해당)의 "기울기와 곡률"에 집중되어 있었기 때문입니다.
> CNN 영역 설계가 Stage 1의 통계 결과를 그대로 반영한 것입니다.

각 영역마다 **별도의 CNN "가지(branch)"** 가 하나씩 붙습니다 (총 4개의 가지).
각 가지는 그 구간만을 전담하여, "이 구간이 위험 신호를 보이는가"에 대한 확률을 출력합니다.

---

**③ 선생님 신호 만들기** (`locked_targets`, `exact_feature_votes`)

CNN이 학습할 "정답(교사 신호)"을 만드는 과정입니다:
- Stage 1에서 잠근 규칙을 그대로 사용해서, 각 환자가 "de-escalation 대상인가"를 결정합니다.
- 이것이 CNN의 학습 목표가 됩니다: "Stage 1이 판단한 것과 같은 패턴을 AI가 스스로 학습하라."
- 이 방식을 **Teacher Mimic(선생님 흉내)** 이라고 부릅니다.

---

**④ CNN 학습** (`crossfit_config`)

```python
CONFIGS = [
    TrainConfig("balanced_aux",   dropout=0.25, weight_decay=1e-3, lr=8e-4, low_weight=5.0, aux_weight=0.25),
    TrainConfig("low_smi_guard",  dropout=0.35, weight_decay=2e-3, lr=6e-4, low_weight=8.0, aux_weight=0.20),
]
SEEDS = [20260701, 20260711]
```

2가지 설정(`balanced_aux`, `low_smi_guard`) × 2가지 랜덤 시드 = **4가지 CNN**을 학습합니다.

각 CNN의 구조적 특징:
- **dropout:** 학습 중 일부 연결을 무작위로 끊어 과적합(overfitting) 방지 (값이 클수록 강한 규제)
- **low_weight:** 저-SMI 환자(소수 클래스)에 더 큰 학습 가중치 부여 (불균형 데이터 보정)
- **aux_weight:** 보조 손실 함수의 비중 (Stage 1 선생님 신호를 얼마나 따를지)
- **patience=16:** 검증 성능이 16에포크 동안 개선 안 되면 조기 종료

또한 **5-fold 교차검증**으로 학습하여, 각 환자에 대해 "자기 자신을 보지 않은" 예측값을 생성합니다.
이렇게 하면 모델이 훈련 데이터를 단순 암기하지 않고 일반화된 패턴을 학습했는지 알 수 있습니다.

**이미 계산된 결과가 있으면** (`PROB_CACHE` 파일이 존재하면) CNN을 다시 학습하지 않고 캐시 파일을 그대로 읽습니다.

---

**⑤ 확률 → 이진 패턴으로 변환** (`codes_from_prob`)

4개 영역 각각의 CNN 출력 확률을 문턱값과 비교해서:
- 확률 ≥ 문턱값: 이 영역이 "위험 패턴을 보임" → `+`
- 확률 < 문턱값: 이 영역이 "위험 패턴 없음" → `-`

환자마다 4개 영역의 +/- 조합이 생깁니다. 예: `(+, +, -, +)`, `(-, -, -, -)` 등
이론상 `2^4 = 16`가지 패턴이 가능합니다. 이를 정수 코드(0~15)로 표현합니다.

---

**⑥ 패턴 게이트 탐색** (`search_pattern_gate`)

"어떤 패턴들의 조합이 나오면 de-escalation으로 볼지"를 **내부 코호트에서만** 탐색합니다.
예: "패턴 0(----)와 패턴 2(-+--)가 나온 환자는 안전하게 저위험으로 재분류 가능"

Stage 1과 동일한 안전 기준(민감도 손실 < 0.08 등)을 통과하는 조합 중 가장 좋은 것을 선택합니다.

**최종 산출물(마커 파일):** `direct_vote_probabilities.npz`
- 저장 내용: (환자 수 × 임상 위험도 구간 5개 × CNN 영역 4개) 크기의 확률 배열
- 모든 환자(내부 1,090명 + 외부 926명)에 대한 값이 들어있습니다

---

### Stage 3 — `aec_direct_vote_auc_boost.py`
### "CNN 확률 → 환자별 AEC 점수"

**파일:** `code/aec_direct_vote_auc_boost.py`
**입력:** Stage 2의 `direct_vote_probabilities.npz`
**출력(마커):** `outputs/aec_direct_vote_auc_boost/direct_vote_auc_boost_scores.csv`

---

#### Stage 3을 한 줄로 이해하기

> "Stage 2에서 나온 4개 영역의 확률 덩어리를 다양한 방법으로 요약하여, 각 환자의 저-SMI 위험을
> 나타내는 '하나의 숫자(AEC 점수)'로 압축한다. 여러 방법을 시험해서 가장 좋은 것을 선택한다."

---

#### 배경 지식: 왜 여러 모델을 시험하는가?

Stage 2에서 나온 결과는 "각 환자, 각 임상 위험도 구간(5개), 각 CNN 영역(4개)"에 대한 확률 수치입니다.
이걸 합산하는 방법은 여러 가지가 있고(단순 평균, 투표, 머신러닝 모델 등),
어떤 방법이 가장 잘 작동하는지 사전에 알 수 없습니다.
그래서 13가지 후보 방법을 모두 시험해 보고, 내부 코호트에서 가장 좋은 것을 선택합니다.

---

#### 세부 단계별 설명

**① 투표 기반 특징 만들기** (`direct_vote_features`)

Stage 2의 확률 배열에서 수백 개의 파생 특징을 만듭니다:

| 특징 종류 | 예시 의미 |
|---|---|
| 원시 확률 | 4개 영역 × 5개 구간 = 20개 확률 그대로 |
| 연성 투표 | "4개 영역 중 2개 이상이 동의할 확률" (soft at-least-2) |
| 영역별 평균/표준편차 | 5개 임상 구간에 걸친 각 영역의 평균 위험도 |
| 임계값별 투표 수 | 확률 ≥ 0.50, 0.55, 0.60, 0.65, 0.70에서 각각 몇 개 영역이 동의하는가 |

---

**② 임상 정보 추가 옵션** (`add_clinical_features`)

일부 후보 모델은 CNN 투표 특징에 더해 임상 점수와의 거리, 임상 경계 근접도 같은 정보도 함께 사용합니다.
이렇게 하면 CNN만 쓸 때보다 좋아지는지 비교할 수 있습니다.

---

**③ 13개의 후보 모델 시험** (`CANDIDATES`)

```python
CANDIDATES = [
    Candidate("vote_only_logit_l2",          "vote", "logit_l2"),
    Candidate("vote_only_logit_l1",          "vote", "logit_l1"),  # ← 최종 선택
    Candidate("vote_only_svm_rbf",           "vote", "svm_rbf"),
    Candidate("vote_only_histgb",            "vote", "histgb"),
    Candidate("vote_only_extratrees",        "vote", "extratrees"),
    Candidate("vote_poly_logit_l2",          "vote_poly", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l2", "clinical_vote", "logit_l2"),
    Candidate("clinical_plus_vote_logit_l1", "clinical_vote", "logit_l1"),
    # ... 총 13개
]
```

사용되는 머신러닝 모델들:
- **로지스틱 회귀 L2 (logit_l2):** 변수들의 영향을 골고루 분산시키는 일반적인 선형 분류기
- **로지스틱 회귀 L1 (logit_l1):** 중요하지 않은 변수는 자동으로 0으로 만들어 중요한 특징만 선별 (변수 자동 선택)
- **SVM RBF:** 비선형 경계도 찾을 수 있는 서포트 벡터 머신
- **히스토그램 그래디언트 부스팅 (histgb):** 의사결정나무를 여러 개 순차적으로 쌓는 강력한 앙상블 방법
- **엑스트라 트리 (extratrees):** 랜덤하게 나무를 만들어 과적합을 줄이는 앙상블 방법

---

**④ 성능 비교 및 검정**

- **내부 AUC(AUROC):** 모델이 저-SMI vs 정상을 얼마나 잘 구분하는가 (1.0이 완벽, 0.5는 동전 던지기)
- **외부 AUC:** 내부에서 학습한 모델을 외부 코호트에 적용했을 때의 성능
- **부트스트랩 검정(2,000회):** "이 모델이 임상 점수 단독보다 통계적으로 유의미하게 나은가"를 2,000번
  재표본추출로 검증. 단순히 우연히 좋아 보이는 건지 구분합니다.

---

**⑤ 최종 선택: `vote_only_logit_l1`**

CNN 투표 특징만 사용하고, L1 로지스틱 회귀로 만든 점수입니다.
이 모델이 선택된 이유:
- 임상 변수 없이 AEC 곡선만으로 만들었으므로 "AEC 고유의 정보"를 반영
- L1 규제 덕분에 불필요한 특징은 0으로 처리되어 모델이 간결하고 해석 가능
- 내부/외부 코호트 모두에서 안정적인 성능

---

**⑥ 점수 CSV 저장**

모든 환자(내부 1,090명 + 외부 926명)에 대해:
- 환자 ID
- `clinical_score` (Stage 1에서 만든 임상 점수)
- 13개 후보 모델의 점수 (각각 별도 컬럼)
- 최종 사용되는 `vote_only_logit_l1` 컬럼

이것이 한 행에 한 환자씩 담긴 `direct_vote_auc_boost_scores.csv`입니다.

**최종 산출물(마커 파일):** `direct_vote_auc_boost_scores.csv`

---

### Stage 4 — `aec_final_global_quintile_phenotype_pipeline.py`
### "최종 검증 (논문 결과)"

**파일:** `code/aec_final_global_quintile_phenotype_pipeline.py`
**입력:** Stage 3의 `direct_vote_auc_boost_scores.csv` + `data/g1090.xlsx`, `data/sdata.xlsx`
**출력(마커):** `outputs/aec_final_global_quintile_phenotype/01_quintile_vs_quartile_enrichment.csv`

---

#### Stage 4를 한 줄로 이해하기

> "이 프로젝트의 핵심 주장을 검증한다: 임상적으로 이미 고위험군으로 분류된 환자들 안에서도,
> AEC 점수가 높은 그룹과 낮은 그룹은 실제 저-SMI 비율이 다르다(표현형 분리)."

---

#### 핵심 주장 — 이 연구가 진짜 하고 싶은 말

이 연구는 "AEC 점수가 임상 모델보다 낫다"고 주장하는 것이 **아닙니다**.

> **진짜 주장:**
> "임상 모델이 이미 '위험하다'고 분류한 환자들 중에서도,
> AEC 점수를 보면 **실제로 저-SMI인 그룹(AEC-high)** 과
> **실제로 정상인 그룹(AEC-low)** 을 더 세밀하게 나눌 수 있다."

즉, AEC 점수가 임상 점수를 대체하는 것이 아니라, **추가로 환자를 세분화**하는 역할을 한다는 것입니다.

---

#### 세부 단계별 설명

**① 환자 표 합치기** (`load_patient_table`)

원본 엑셀의 임상 정보(나이, BMI, 근육량 TAMA, 근육내 지방 IMATA 등)와
Stage 3에서 만든 점수(clinical_score, vote_only_logit_l1)를 환자 단위로 합칩니다.
- 내부 코호트: 1,090명이 정확히 맞는지 확인
- 외부 코호트: 926명이 정확히 맞는지 확인

---

**② 문턱값 잠그기** (`add_global_flags`)

> **중요한 원칙:** 문턱값은 반드시 **내부 코호트만** 보고 정한 다음, 외부 코호트에 그대로 적용합니다.

```
내부 코호트(g1090, 1,090명)에서:
    임상 고위험군 기준 = 내부 임상 점수 상위 20% (5분위 기준)
    
그 임상 고위험군 안에서:
    AEC-high = AEC 점수 상위 20%
    AEC-low  = AEC 점수 하위 20%
    
→ 이 두 컷오프(임상 80번째 백분위, AEC 80번째/20번째 백분위)를
  외부 코호트(sdata, 926명)에도 그대로 적용
```

> **왜 외부 코호트에서 컷오프를 다시 최적화하지 않는가?**
> 외부 데이터에서 컷오프를 다시 정하면 "외부 데이터에만 맞게 조정된" 결과가 나와서,
> 새로운 환자에게 적용할 때 성능이 보장되지 않습니다. 이를 **데이터 유출(data leakage)** 이라고 하며,
> 임상 연구에서 가장 피해야 할 설계 오류 중 하나입니다.

왜 20%(5분위)인가:
- "상위/하위 5분위"는 사전에 명확히 정의할 수 있는 직관적인 기준 (25%인 4분위보다 더 극단적)
- Youden 최적값이나 데이터 기반 최적 컷오프가 아니므로 과적합 위험 없음
- 25% 기준(4분위)도 민감도 분석(sensitivity analysis)으로 같이 보고함

---

**③ 핵심 표 만들기** (`enrichment_table` → `01_quintile_vs_quartile_enrichment.csv`)

**이 파일이 논문의 핵심 결과입니다.**

표의 구조:

| 코호트 | 그룹 | 환자 수 | 실제 저-SMI 비율 | Fisher 검정 p값 |
|---|---|---|---|---|
| 내부(g1090) | 임상 고위험군 전체 | N명 | X% | - |
| 내부(g1090) | AEC-high (상위 20%) | N명 | X% | p<0.05? |
| 내부(g1090) | AEC-low (하위 20%) | N명 | X% | p<0.05? |
| 외부(sdata) | 임상 고위험군 전체 | N명 | X% | - |
| 외부(sdata) | AEC-high | N명 | X% | p<0.05? |
| 외부(sdata) | AEC-low | N명 | X% | p<0.05? |

**Fisher 정확 검정:** 두 그룹의 비율 차이가 통계적으로 우연이 아닐 가능성을 검정합니다.
"AEC-high와 AEC-low 사이의 저-SMI 비율 차이가 우연의 일치가 아니다"를 p값으로 표현합니다.
20% 기준(본분석)과 25% 기준(민감도 분석) 두 가지 모두 보고합니다.

---

**④ 이진 규칙 비교표** (`02_or_and_diagnostic_metrics.csv`)

4가지 단순 이진 규칙의 성능 비교:
- **임상 양성(C+):** 임상 점수가 고위험군 컷오프 이상인 환자
- **AEC 양성(A+):** AEC 점수가 컷오프 이상인 환자
- **임상 OR AEC:** 둘 중 하나라도 양성인 환자 (민감도 높음)
- **임상 AND AEC:** 둘 다 양성인 환자 (특이도 높음)

각 규칙에 대해 민감도(sensitivity), 특이도(specificity), 양성예측도(PPV) 등을 계산합니다.

---

**⑤ 4칸 표 특성 분석** (`03_four_cell_characteristics.csv`)

환자를 2×2 = 4개 칸으로 나눕니다:

```
                  AEC 양성(A+)    AEC 음성(A-)
임상 양성(C+)  │  C+A+ 그룹    │  C+A- 그룹  │
임상 음성(C-)  │  C-A+ 그룹    │  C-A- 그룹  │
```

각 칸의 환자들의 평균 나이, BMI, 근육량(TAMA), 지방량(IMATA) 등을 비교합니다.
예: "C+A+ 그룹은 C+A- 그룹보다 근육량이 더 적을 것이다"는 가설을 검증합니다.

---

**⑥ 저-SMI 하위유형 분석** (`04_low_smi_subtype_characteristics.csv`, `05_low_smi_subtype_feature_tests.csv`)

실제로 저-SMI인 환자들만 모아서:
- AEC+ 그룹과 AEC- 그룹의 체형 특징이 통계적으로 다른지 검정
- 예: "AEC 양성으로 분류된 저-SMI 환자는 BMI가 낮은 '마른 근감소증' 체형이고,
        AEC 음성으로 분류된 저-SMI 환자는 지방이 많은 '비만성 근감소증' 체형이다"
  같은 임상적 해석을 뒷받침하는 근거를 제시합니다.

---

**⑦ 논문 그림 생성** (`figure_quintile_enrichment.png`)

내부/외부 각각에서 막대그래프를 그립니다:
- X축: "임상 고위험군 전체", "AEC-low", "AEC-high"
- Y축: 실제 저-SMI 비율(%)
- 오차 막대: 95% 신뢰구간

직관적으로 "AEC-high 그룹이 AEC-low 그룹보다 저-SMI 비율이 높다"는 것을 시각화합니다.

---

**⑧ 최종 요약 JSON 저장** (`final_summary.json`)

이번 분석의 핵심 결론(각 그룹의 저-SMI 비율, p값, 파일 경로 등)을 JSON 형식으로 정리합니다.

**최종 산출물(마커 파일):** `01_quintile_vs_quartile_enrichment.csv` — 이 프로젝트의 "논문에 실리는" 핵심 표

---

## 7. `outputs/` 폴더 실제 분석 (현재 상태 점검)

`run_from_raw.py`는 이미 한 번 실행이 시작된 흔적이 있어서, 현재 디스크에 실제로 만들어져 있는 파일을 열어
분석했습니다. 결과부터 말하면 **아직 Stage 1이 끝나지 않은 상태**입니다.

### 7-1. 지금 존재하는 파일 (Stage 1 중간 산출물만 존재)

```
outputs/aec_lock_smoothed_deesc_gate/
    internal_prescreen_feature_pool.csv     (601줄 = 헤더 1 + 후보 특징 600개)
    internal_single_feature_screen.csv      (7,201줄 = 헤더 1 + 특징×파라미터 조합 7,200개)
    internal_combo_feature_pool.csv         (19줄 = 헤더 1 + 최종 후보 특징 18개)

그 외 10개 출력 폴더(aec_conditional_value, aec_region_cnn_pattern_gate,
aec_direct_vote_auc_boost, aec_final_global_quintile_phenotype 등)는 전부 빈 폴더입니다.
```

즉, Stage 1의 ①~⑦단계까지는 끝났지만, 그 다음 단계인 **⑧ `combo_search`
(18개 특징으로 1~4개씩 조합해 최적의 k-of-m 규칙을 잠그는 단계)** 가 아직 끝나지 않아서
`locked_gate_features.csv`가 아직 생성되지 않았습니다. 이 파일이 없으므로 `run_step`의 marker 조건이
채워지지 않았고, 따라서 Stage 2·3·4는 아예 시작도 하지 않은 상태입니다.

(`combo_search`는 최대 4개씩 묶는 조합 수가 특징 18개 기준으로 수천 개에 이르고, 조합마다 5개 임상구간을
모두 계산하기 때문에 이 파이프라인에서 가장 시간이 오래 걸리는 단계 중 하나입니다.)

### 7-2. 지금까지 나온 숫자로 무엇을 알 수 있나

`internal_combo_feature_pool.csv`에 저장된 최종 후보 18개 특징을 보면, 점수(`screen_score`) 상위권은 다음과
같은 성격의 특징들이 차지하고 있습니다.

| 순위 | 특징 이름 (요약) | 특징 종류 |
|---|---|---|
| 1 | `norm_slope_082_085_sd` — 82~85번 구간 기울기의 변동성 | 부호 있는 기울기 |
| 2 | `visual_norm_abs_slope_094_099_max` — 94~99번 구간 기울기 절댓값 최대치 | 절댓값 기울기 |
| 3 | `visual_norm_slope_082_087_sd` — 82~87번 구간 기울기 변동성 | 부호 있는 기울기 |
| 4 | `norm_curv_103_110_mean` — 103~110번 구간 곡률 평균 | 곡률 |
| 5 | `norm_curv_097_100_sd` — 97~100번 구간 곡률 변동성 | 곡률 |

가장 눈에 띄는 점은, 상위 특징들이 대부분 **AEC 곡선의 80~110번 포인트 부근(전체 128포인트 중 후반부,
복부 하단 통과 구간에 해당)의 "기울기"와 "곡률"** 에 몰려 있다는 것입니다. 이는 Stage 2에서 CNN이 학습할
4개 영역(`R1: 76~92`, `R2: 88~106`, `R3: 96~118`, `R4: 90~110`)이 바로 이 구간을 감싸도록 미리 정해진 이유이기도
합니다 — 즉 Stage 2의 CNN 영역 설계는 Stage 1의 스크리닝 결과를 그대로 반영해서 만들어진 것입니다.

또한 `company_eta2`(스캐너 제조사에 따라 이 특징 값이 얼마나 갈리는지 나타내는 지표, 0에 가까울수록
"장비 종류와 무관하게 순수한 신호"에 가까움) 값을 보면, 1위 특징(`0.109`)과 5위 특징(`0.240`) 사이에
꽤 편차가 있어, 특징 선택 시 "장비 영향이 적은 특징"을 우선하려는 `screen_score` 계산식의
`- 0.05 * company_eta2` 항이 실제로 작동하고 있음을 확인할 수 있습니다.

`internal_single_feature_screen.csv`의 7,200개 행은 "특징 600개 × 폭(width) 3가지 × 람다(lambda) 4가지"의
모든 조합을 다 시험해 본 결과입니다 (600 × 3 × 4 = 7,200 — 정확히 일치). 즉 하나의 특징이라도
"경계 폭을 얼마나 넓게 볼지", "임상 점수와 얼마나 섞을지"를 바꿔가며 여러 버전으로 테스트되었다는 뜻입니다.

### 7-3. 지금 시점에서 아직 알 수 없는 것

- 어떤 "k개 중 m개 동의" 조합이 최종적으로 잠기는지 (`locked_gate_features.csv`, `locked_gate_summary.json`)
- 그 잠긴 규칙이 내부/외부 코호트에서 민감도·특이도를 얼마나 바꾸는지 (`locked_gate_operating_point_details.csv`)
- CNN이 학습한 영역별 확률 (Stage 2)
- 최종 AEC 점수 `vote_only_logit_l1` (Stage 3)
- 논문의 핵심 표인 임상고위험군 내 AEC-high vs AEC-low 저-SMI 비율 비교 (Stage 4)

이 결과들을 보려면 `combo_search`가 끝날 때까지 `python code/run_from_raw.py`를 계속 실행해 두거나,
다시 실행해서 이어서 진행해야 합니다. (`run_step`의 캐싱 로직 덕분에, 이미 끝난 하위 계산은 다시 반복되지
않고, 지금 멈춰 있는 지점부터 자연스럽게 이어집니다 — 단, Stage 1의 `main()` 자체가 끝까지 실행되어야
marker 파일이 생기므로, Stage 1 전체를 한 번에 끝까지 실행해야 다음 단계로 넘어갑니다.)

---

## 8. 전체 파이프라인이 다 끝나면 최종적으로 생기는 파일들

```
outputs/aec_lock_smoothed_deesc_gate/locked_gate_features.csv
outputs/aec_region_cnn_pattern_gate/direct_vote_probabilities.npz
outputs/aec_direct_vote_auc_boost/direct_vote_auc_boost_scores.csv
outputs/aec_final_global_quintile_phenotype/01_quintile_vs_quartile_enrichment.csv   ← 논문 핵심 표
outputs/aec_final_global_quintile_phenotype/figure_quintile_enrichment.png            ← 논문 핵심 그림
```

이 중 가장 중요한 결론 파일은 `01_quintile_vs_quartile_enrichment.csv`이며, 이 표의 각 행이
"임상 고위험군 안에서 AEC 점수가 높은 사람과 낮은 사람의 실제 저-SMI 비율 차이 및 통계적 유의성"을
내부(g1090)와 외부(sdata) 코호트 각각에 대해 20%와 25% 기준으로 보여줍니다.

---

## 9. 4단계 요약 비교표

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| **하는 일** | 곡선 특징 찾기 + 규칙 잠금 | CNN이 곡선 모양 학습 | CNN 확률 → AEC 점수 | AEC 점수로 환자 표현형 분리 |
| **핵심 기법** | 통계적 스크리닝 + 투표 규칙 | 합성곱 신경망(CNN) | 로지스틱 회귀(L1) | Fisher 정확 검정 |
| **입력** | 원본 엑셀 2개 | Stage 1 잠금 특징 | Stage 2 CNN 확률 | Stage 3 AEC 점수 |
| **출력** | `locked_gate_features.csv` | `direct_vote_probabilities.npz` | `direct_vote_auc_boost_scores.csv` | `01_...enrichment.csv` + 그림 |
| **소요 시간** | 가장 오래 걸림 (combo_search) | CNN 학습으로 오래 걸림 | 비교적 빠름 | 빠름 |
| **외부 데이터 역할** | 검증만 (규칙 정할 때 사용 안 함) | 검증만 | 검증만 | 최종 검증 |
