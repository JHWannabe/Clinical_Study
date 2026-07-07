# Codebook: `main()` — `main_aec_new_region_cnn_surrogate_mimic_gate.py`

---

## 배경 이해: 이 코드가 풀려는 문제

**AEC(Antibody Elution Curve)** 는 환자별로 측정된 128-포인트 시계열 곡선이다. 이 곡선의 형태를 보고 "이 환자는 치료 강도를 낮춰도 되는가(de-escalation)"를 판단하는 **게이트(gate) 규칙**을 자동으로 학습한다.

핵심 제약: **민감도 손실 ≤ 8%** — 실제 사건(event) 환자를 강등시켜 놓치는 비율을 8% 이내로 유지해야 한다. 의료적 안전 기준.

---

## 전역 상수 및 설정값

| 상수 | 값 | 의미 |
|---|---|---|
| `TARGET_OPS` | `["S80", "S85", "S90"]` | 평가할 3개 임상 운영점(sensitivity 목표) |
| `SEEDS` | `[20260701, 20260711]` | 재현성 보장용 시드 2개 |
| `MIN_DEESC_N` | `10` | 유효한 규칙의 최소 de-escalation 환자 수 |
| `MAX_SENS_LOSS` | `0.08` | 허용 최대 민감도 손실 (8%) |
| `DEVICE` | `cuda` or `cpu` | PyTorch 연산 디바이스 |

### REGIONS — AEC 곡선 구간 정의

| 이름 | 인덱스 범위 | 의미 |
|---|---|---|
| `R1_045_056` | 45~56 | AEC 곡선의 초기 하강 구간 |
| `R2_057_080` | 57~80 | 중간 안정화 구간 |
| `R3_097_128` | 97~128 | 말기 구간 (전체 포함) |
| `R4_117_128` | 117~128 | 말기 말단 구간 |

> 인덱스는 1-based. 코드에서 `slice(start-1, end)` 로 0-based 접근.

### TEACHER_BRANCHES — 교사(surrogate) 규칙 정의

이전 실험(`aec_new_region_surrogate_combo_gate`)에서 선정된 **고정 교사 규칙**. CNN이 이것을 흉내내도록 학습.

| 브랜치 | 구간 | 서술자 | 부호 | width | lambda |
|---|---|---|---|---|---|
| Branch 0 | R1 | `endpoint_delta` | −1 | 0.35 | 0.25 |
| Branch 1 | R2 | `level_mean` | −1 | 0.35 | 0.25 |
| Branch 2 | R3 | `endpoint_delta` | −1 | 0.70 | 0.25 |
| Branch 3 | R4 | `linear_slope` | −1 | 0.35 | 0.25 |

- **부호 −1**: 해당 서술자 값이 낮을수록 de-escalation 가능성이 높다는 방향성
- **width**: de-escalation 경계 가우시안의 폭 (클수록 임계값 부근에서 더 넓은 영역 커버)
- **lambda**: `make_single_deesc` 내부 가중치 파라미터

### TEACHER_PATTERNS — 교사가 de-escalation으로 판단하는 투표 패턴

```
["--+-", "---+", "++-+", "--++", "++++"]
```

- 4자리 각각이 Branch 0~3의 투표 여부 (`+` = 투표, `-` = 비투표)
- 이 5개 패턴에 해당하는 환자를 de-escalation 대상으로 분류
- 반드시 4개 지역이 모두 동의할 필요는 없고, **특정 조합 패턴**이면 강등 대상

### MimicConfig — CNN 하이퍼파라미터

| 파라미터 | balanced | guarded | 의미 |
|---|---|---|---|
| `hidden` | 10 | 12 | CNN 채널 수 |
| `dropout` | 0.20 | 0.30 | 드롭아웃 비율 |
| `lr` | 8e-4 | 6e-4 | 학습률 |
| `weight_decay` | 1e-3 | 2e-3 | L2 정규화 |
| `consensus_weight` | 0.65 | 0.85 | 2-of-4 합의 손실 가중치 |
| `non_cpos_weight` | 0.04 | 0.02 | 임상음성 환자 손실 가중치 (작게 → 임상양성 집중) |
| `max_epochs` | 180 | 180 | 최대 학습 에포크 |
| `patience` | 22 | 22 | 조기종료 인내 에포크 수 |
| `batch_size` | 96 | 96 | 미니배치 크기 |

> `guarded` 설정은 더 높은 정규화 + 더 강한 합의 강제 → 보수적인 모델

---

## Stage 0. 이전 실험에서 가져온 "교사(Teacher)" 규칙

**왜 "교사"인가?**

이전 실험(`aec_new_region_surrogate_combo_gate`)에서 이미 좋은 성능의 규칙을 찾아냈다. 그 규칙은 **해석 가능한 단일 서술자(예: endpoint_delta)**를 직접 사용하기 때문에 해석성은 높지만, 서술자가 고정되어 있어 다양한 곡선 형태를 포착하는 데 한계가 있다. 이 파일은 그 교사 규칙의 **"투표 결과"를 목표값**으로 삼아 CNN이 더 복잡한 형태 정보를 학습하도록 유도한다. 이것이 Knowledge Distillation.

**`sign: -1` 의 의미**

각 서술자의 낮은 값 = de-escalation 가능성이 높다는 임상적 방향성. 예를 들어 `endpoint_delta`가 음수(곡선이 하강)이면 치료 반응이 좋아진다는 신호. `sign=-1`을 곱해 "값이 낮을수록 양성 방향"으로 통일한다.

**`TARGET_OPS = ["S80", "S85", "S90"]` 의 의미**

80%, 85%, 90% sensitivity를 보장하는 3가지 임상 운영 지점. 하나의 규칙이 세 지점 모두에서 안전해야 한다. 가장 보수적인 조건(S80)은 민감도를 가장 높게 요구한다.

---

## Stage 1. 초기화 및 데이터 로드 (line 706–720)

### Step 1-1. 데이터 로드

```python
g = load_dataset(DATA_DIR / "g1090.xlsx")  # 내부 데이터 (학습/검증용)
s = load_dataset(DATA_DIR / "sdata.xlsx")  # 외부 데이터 (일반화 검증용)
```

**의미:** `g`는 모델을 학습하고 규칙을 선정하는 데 쓰인다. `s`는 선정된 규칙이 다른 환경에서도 작동하는지 독립 검증용이다. 두 데이터셋을 동시에 평가하는 것은 일반화 능력을 보장하기 위함이다.

### Step 1-2. 임상 점수 및 임계값 계산

```python
_clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
threshold_vec = np.array([thresholds[op] for op in TARGET_OPS], dtype=np.float32)
```

| 변수 | 타입 | 내용 |
|---|---|---|
| `c_g` | `(N_g,)` float | 내부 환자별 임상 점수 (교차검증 OOF 예측값) |
| `c_s` | `(N_s,)` float | 외부 환자별 임상 점수 |
| `thresholds` | `dict` | `{"S80": t1, "S85": t2, "S90": t3}` — 각 운영점의 임계값 |

**의미:** `c_g`, `c_s`는 환자별로 계산된 임상 점수(연속값). `thresholds["S80"]`은 전체 환자 중 80% 민감도를 달성하는 임계값이다. 이 값 이상인 환자 = 임상양성 = **원래 치료 강도를 유지해야 할 환자**. 이들 중 일부를 de-escalation 하는 것이 목표이므로, 임계값은 de-escalation 적용 모집단의 경계를 정의한다.

### Step 1-3. 임상양성 행렬

```python
cpos_g = clinical_positive_matrix(c_g, thresholds)  # (N_g, 3)
cpos_s = clinical_positive_matrix(c_s, thresholds)  # (N_s, 3)
# cpos[i, j] = True  if  c[i] >= thresholds[TARGET_OPS[j]]
```

**의미:** `cpos_g[i, j] = True` 이면 환자 i가 운영점 j(S80/S85/S90)에서 **임상양성**, 즉 현재 기준으로는 치료를 받아야 할 환자. de-escalation은 이 임상양성 환자 중에서만 수행 — 임상음성 환자를 강등시키는 것은 의미가 없다(이미 저위험).

### Step 1-4. 교사 특징 추출

```python
feat_g, feat_s, feature_labels = teacher_features(g, s)
# feat_g: (N_g, 4)   feat_s: (N_s, 4)
```

**`region_descriptor_matrix()` — 왜 구간을 나누는가?**

AEC 곡선의 어느 시점(구간)이 임상적으로 의미 있는지 모르므로 여러 구간을 동시에 분석한다. 각 구간은 치료 반응의 다른 국면(초기 반응 / 안정화 / 말기 반응)을 포착한다. 4구간 × 12서술자 = **48컬럼** 중 TEACHER_BRANCHES가 지정한 4개만 사용.

**`z_train_apply()` — 왜 내부 기준으로만 표준화하는가?**

외부 데이터를 표준화 통계 계산에 포함시키면 외부 데이터 정보가 내부 학습에 누출(data leakage)된다. 실제 배포 환경을 시뮬레이션하는 올바른 방식.

**`sign` 적용의 의미**

TEACHER_BRANCHES 4개 특징을 선택한 뒤 각 브랜치의 `sign(-1)`을 곱해 방향성 반전. "값이 낮을수록 de-escalation 가능성 높은 방향"으로 통일.

### Step 1-5. 교사 투표 타깃 생성

```python
target_g = branch_votes(feat_g, c_g, thresholds)  # (N_g, 3, 4)
target_s = branch_votes(feat_s, c_s, thresholds)  # (N_s, 3, 4)
```

`make_single_deesc()`를 각 브랜치 × 각 운영점에 대해 호출해 실제 교사 게이트의 투표 결과를 미리 계산한다.
- `target[i, op, j] = 1` → "교사 브랜치 j가 환자 i를 운영점 op에서 강등하라고 투표했다"
- 이것이 CNN이 흉내내야 할 **학습 목표값(distillation target)**

**왜 교사의 투표 자체를 타깃으로 쓰는가?**

교사의 최종 결정(de-escalation 여부)만 모방하면 중간 추론 과정이 전달되지 않는다. 브랜치별 투표를 개별적으로 모방하면 CNN이 각 지역의 역할을 각각 학습하게 되어 더 구조화된 지식 이전이 이루어진다.

### Step 1-6. CNN 입력 채널 생성

```python
xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
# shape: (N, 3, 128)
```

교사는 서술자(평균, 기울기 등 집약된 숫자)를 쓰지만, CNN은 **원시 파형 전체**를 입력으로 받는다. 3채널:

| 채널 | 내용 | 포착하는 정보 |
|---|---|---|
| Ch 0 | 원시 AEC 곡선 (row-z 정규화) | 레벨 정보 — "이 구간이 높은가 낮은가" |
| Ch 1 | 1차 차분 (기울기) | "상승 중인가 하강 중인가" |
| Ch 2 | 2차 차분 (곡률) | "변화가 빨라지는가 느려지는가" |

이 세 채널을 함께 쓰면 CNN이 서술자를 명시적으로 계산하지 않고도 형태 특징을 스스로 학습할 수 있다.

---

## Stage 2. 교사 게이트 기준선 평가 (line 722–734)

```python
teacher_mask = mask_from_patterns(TEACHER_PATTERNS)
exact_detail = evaluate_gate_detail(..., votes_to_codes(target_g), votes_to_codes(target_s))
exact_detail.to_csv(OUT_DIR / "exact_surrogate_teacher_details.csv")
```

**왜 이 단계가 필요한가?**

CNN을 학습하기 전에 교사 규칙이 이 데이터셋에서 실제로 어떤 성능을 보이는지 먼저 기록한다. 이후 CNN의 결과와 비교해 "CNN이 교사보다 나아졌는가, 비슷한가, 나빠졌는가"를 판단하는 **기준선(baseline)**이 된다.

**`votes_to_codes()` 의 의미**

```python
# 예시: target[i, op, :] = [1, 0, 1, 1] 이면
# code = 1*1 + 0*2 + 1*4 + 1*8 = 13
code = votes_to_codes(target_g)  # (N_g, 3)
```

4개 브랜치의 투표 결과를 하나의 정수(0~15)로 인코딩. 이 코드가 `TEACHER_PATTERNS`에 속하는지 비트마스크로 비교하면 O(1)에 de-escalation 여부 판단 가능.

---

## Stage 3. CNN 학습: Knowledge Distillation (line 736–747)

### Step 3-1. 모델 구조 — `RegionBranch`

```python
class RegionBranch(nn.Module):
    nn.Conv1d(3, hidden, kernel_size=5, padding=2)  # 넓은 수용영역
    nn.BatchNorm1d(hidden), nn.SiLU()
    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)  # 정밀 추출
    nn.BatchNorm1d(hidden), nn.SiLU()
    # avg+max 풀링 → Linear(hidden*2, 1)
```

- 각 지역 구간(R1~R4)마다 독립적인 CNN 브랜치 → 교사의 "지역별 투표" 구조를 구조적으로 반영
- `kernel_size=5` → 인접 5포인트의 패턴을 한 번에 볼 수 있어 형태 특징(완만한 하강, 급격한 상승 등) 포착
- **avg+max 풀링 병렬**: avg는 전체 구간의 평균적 특성, max는 극단적 특성을 모두 유지

### Step 3-2. 모델 구조 — `DirectVoteMimicCnn`

```python
def forward(self, x, clinical_z):
    morph_t = stack([branch_j(x[:,:,R_j]) for j in 0..3])  # (N, 4) 형태 점수

    delta    = clinical_z - thresholds     # (N, 3): 임상 거리
    boundary = exp(-0.5*(delta/width)²)   # (N, 3, 4): 가우시안 경계
    cpos     = (delta >= 0)               # (N, 3, 4): 임상양성 여부

    feats = stack([
        morph,            # f0: 순수 형태 점수
        morph * boundary, # f1: 경계 근방에서만 형태 강조
        delta,            # f2: 임상 점수 거리 (전체)
        boundary,         # f3: 가우시안 경계 가중치
        cpos,             # f4: 임상양성 여부
    ])
    logits = (feats * head_weight).sum(-1) + head_bias
    # output: (N, 3, 4) — (환자, 운영점, 지역)
```

**왜 형태 점수만 쓰지 않고 임상 특징을 추가하는가?**

교사의 de-escalation 게이트(`make_single_deesc`)는 단순히 형태 특징 하나만 보는 게 아니라:
1. **임상 점수가 임계값에 얼마나 가까운가** (`delta`) — 가까울수록 de-escalation이 더 의미 있다
2. **임계값 근방인가** (`boundary`) — 경계 근방 환자에서만 de-escalation 기회가 있다
3. **임상양성인가** (`cpos`) — 임상음성 환자에게는 게이트가 작동하지 않는다

이 구조를 CNN 헤드에 **명시적으로** 주입함으로써 교사의 의사결정 로직을 구조적으로 모방한다.

**헤드 초기값의 의미**

```python
head_weight[:, 1] = -1.5   # morph*boundary: 경계 근방 형태 점수가 낮으면 투표↑
head_weight[:, 2] = -2.0   # delta: 임상 점수 < 임계값(delta<0)이면 투표↑
head_weight[:, 3] =  0.5   # boundary: 경계 근방 자체가 양성 신호
head_weight[:, 4] =  0.5   # cpos: 임상양성이면 de-escalation 가능
head_bias         = -1.0   # 기본적으로 보수적 (투표 안 함 쪽으로 시작)
```

이 초기화는 교사 규칙의 부호 방향을 이미 알고 시작하도록 해 수렴을 빠르게 하고 잘못된 방향으로의 초반 수렴을 방지한다.

### Step 3-3. 손실 함수

```python
def loss_fn(logits, target, cpos_weight, pos_weight, cfg):
    # ① 브랜치별 BCE (교사 투표 모방)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    weight = cpos_weight * (1 + (pos_weight - 1) * target)
    branch_loss = (bce * weight).sum() / weight.sum()

    # ② 2-of-4 합의 BCE (consensus 모방)
    prob2 = soft_atleast2_prob(logits)
    target2 = (target.sum(dim=-1) >= 2).float()
    consensus_loss = (BCE(prob2, target2) * cpos_weight).sum() / cpos_weight.sum()

    return branch_loss + cfg.consensus_weight * consensus_loss
```

**① 브랜치별 BCE의 의미**

- `cpos_weight`: 임상양성=1.0, 임상음성=0.04(또는 0.02). 임상음성 환자는 어차피 de-escalation 대상이 아니므로 그들의 예측 오류는 거의 무시. 임상양성 환자의 정확한 모방에 집중.
- `pos_weight`: 양성 투표(de-escalation) 대 음성 투표 비율 보정. 교사가 de-escalation 하는 경우가 상대적으로 적으므로 **희소 양성 클래스를 과소 학습하지 않도록** 보정. 최대 40으로 클리핑(과도한 가중치는 학습 불안정 유발).

**② 2-of-4 합의 손실의 의미**

`TEACHER_PATTERNS`를 보면 교사는 "여러 지역이 동시에 동의해야" de-escalation한다. 브랜치별 BCE만 쓰면 각 브랜치를 개별적으로 맞추는 데는 성공해도, **조합 패턴**(2개 이상 투표)을 맞추지 못할 수 있다. 합의 손실이 이 조합 구조를 명시적으로 학습하게 한다.

```
P(≥2개 투표) = 1 - P(0개) - P(정확히 1개)
P(0개)       = ∏_j (1 - p_j)
P(정확히 1개) = Σ_j [ p_j × ∏_{k≠j} (1 - p_k) ]
```

미분 가능하므로 역전파를 통해 직접 학습 가능.

**`consensus_weight` 차이의 의미**

| 설정 | consensus_weight | 의미 |
|---|---|---|
| balanced | 0.65 | 브랜치별 모방과 합의 모방의 균형 |
| guarded | 0.85 | 합의 구조 모방을 더 강하게 강제 → 단독 브랜치가 규칙을 좌우하지 못함 |

### Step 3-4. 학습 가중치 설계

```python
wt_np = np.where(cpos[train_idx], 1.0, cfg.non_cpos_weight)
# 임상양성: 가중치 1.0, 임상음성: 가중치 0.04

pos = (target * wt_np).sum()
neg = ((1 - target) * wt_np).sum()
pw  = clip(neg / pos, 1.0, 40.0)  # 브랜치별 클래스 불균형 보정
```

두 가지 불균형을 동시에 보정:
1. **임상양성/음성 불균형** → `cpos_weight`로 임상음성을 크게 다운웨이팅
2. **투표 양성/음성 불균형** → `pos_weight`로 de-escalation(투표=1)이 드문 브랜치를 업웨이팅

### Step 3-5. 교차검증 + 다중 시드 설계

```python
for seed in SEEDS (2개):
    for fold in stratified_folds(y, seed) (5개):
        train_fold(seed=seed + fold_id * 101)  # fold마다 다른 시드
    oof_runs.append(oof)
    ext_runs.append(mean(ext_folds))

return mean(oof_runs), mean(ext_runs)  # 시드 평균
```

- **5-fold stratified CV**: 사건/비사건 비율을 각 fold에 균등하게 유지. OOF(Out-Of-Fold) 예측은 단순 hold-out보다 편향이 적다.
- **2개 시드**: 난수 시드에 의한 분산을 줄임. 두 시드의 평균을 취해 안정성 확보.
- **fold 시드 = `seed + fold_id * 101`**: 각 fold마다 다른 초기값 → 앙상블 다양성 확보.

### Step 3-6. 조기종료

```python
if val_loss < best_loss - 1e-4:
    best_state = model.state_dict() 저장
    patience = cfg.patience (22)
else:
    patience -= 1
    if patience <= 0: break

model.load_state_dict(best_state)  # 최적 상태로 복원
```

검증 손실이 `1e-4` 이상 개선되지 않은 에포크가 22회 누적되면 학습 중단. 학습 종료 후 최적 검증 손실 시점의 가중치로 복원.

---

## Stage 4. 게이트 탐색 (line 749–755)

### Step 4-1. 임계값 탐색 공간 설계

```python
def threshold_vectors():
    # 균일 스캔: (p,p,p,p) for p in 0.35~0.95       → 13개
    # 격자 A: product([0.45,0.55,0.65,0.75,0.85]^4) → 625개
    # 격자 B: product([0.50,0.60,0.70,0.80,0.90]^4) → 625개
    # 중복 제거 후 정렬 → 약 700~1,200개 조합
```

CNN은 (N, 3, 4) 확률값을 출력한다. 이 확률을 0/1 투표로 변환할 임계값을 찾아야 한다. 지역마다 다른 임계값을 쓸 수 있으므로 4차원 탐색. 균일 스캔은 빠른 전체 범위 커버, 격자 조합은 지역별 최적 조합 탐색.

### Step 4-2. 패턴 랭킹

```python
top_codes = rank_codes_internal(y, cpos_g, code_g)
# score = min(비사건 수) - 4×max(사건 수) - 18×mean(사건율)
```

좋은 de-escalation 패턴의 기준:

| 기준 | 가중치 | 이유 |
|---|---|---|
| 비사건 수 최대화 | +1 | 많은 비사건 환자를 강등 → 특이도 향상 |
| 사건 수 최소화 | ×4 (강한 패널티) | 실제 사건 환자를 놓치면 안 됨. 안전 우선 |
| 사건율 최소화 | ×18 (매우 강한 패널티) | 강등 집단의 질(순도) 요구 |

### Step 4-3. 마스크(패턴 조합) 생성

```python
masks = candidate_masks(top_codes)
```

어떤 패턴들의 조합을 de-escalation 기준으로 쓸지 탐색. 생성 마스크 집합:

| 마스크 종류 | 의미 |
|---|---|
| 교사 마스크 | 이전 실험 결과 재현 비교 기준 |
| 단일 패턴 16개 | 가장 단순한 규칙 |
| at_least_k | "k개 이상 투표하면 강등" 규칙 |
| exactly_k | "정확히 k개만 투표하면 강등" 규칙 |
| top_codes 조합 2~5개 | 데이터 기반으로 선정한 상위 패턴 묶음 |

### Step 4-4. 빠른 성능 평가

```python
def fast_dataset_summary(y, cpos, code, mask):
    for op_idx in range(3):  # S80, S85, S90
        deesc = cpos[:,op_idx] & isin(code[:,op_idx], selected_codes)
        # 임상양성이면서 해당 패턴인 환자 = 강등 대상

        tp_lost    = sum(deesc & y)    # 강등된 실제 사건 환자
        fp_removed = sum(deesc & ~y)   # 강등된 비사건 환자
```

각 운영점에서 de-escalation의 손익:

| 지표 | 수식 | 의미 |
|---|---|---|
| `sens_loss` | `tp_lost / total_pos` | 놓친 진양성 비율 (의료적 위험) |
| `spec_gain` | `fp_removed / total_neg` | 줄인 위양성 비율 (이득) |
| `acc_delta` | `(fp_removed - tp_lost) / n_total` | 순 정확도 변화 |
| `event_rate` | `tp_lost / n` | 강등 집단 내 실제 사건 비율 |
| `p_loss` | `2^(1 - tp_lost)` | 민감도 손실이 우연일 확률 상한 |

**`p_loss` 의 의미:** 놓친 사건이 1명뿐이라면 우연일 가능성이 크다(p=1.0). 여러 명이 놓쳐야 규칙이 실제로 민감도를 저하시키는 것으로 볼 수 있다.

---

## Stage 5. 규칙 통과 조건 및 선정 (line 757–773)

### Step 5-1. 통과 조건 설계

```python
def dataset_pass(row, prefix):
    return (
        min_deesc_n >= 10            # 최소 10명 이상 강등
        and max_sensitivity_loss <= 0.08  # 모든 운영점에서 민감도 손실 ≤ 8%
        and min_sensitivity_loss_p >= 0.05  # 손실이 통계적으로 우연이어야
        and min_specificity_gain > 0    # 최소한 위양성 감소 존재
        and min_accuracy_gain > 0       # 정확도가 나빠지면 안 됨
    )
```

각 조건의 의미:

| 조건 | 이유 |
|---|---|
| `min_deesc_n >= 10` | 강등 환자 수가 너무 적으면 규칙이 실질적 효과 없음. 9명 이하는 통계적으로도 불안정 |
| `max_sensitivity_loss <= 0.08` | 3개 운영점 **모두**에서 사건 환자 놓침 ≤ 8%. 가장 나쁜 운영점 기준. 안전 최우선 |
| `min_sensitivity_loss_p >= 0.05` | 손실이 우연일 가능성이 5% 이상 → 손실이 우연 수준으로 작다는 뜻 |
| `min_specificity_gain > 0` | 최소한 한 운영점에서 비사건 환자 감소 효과 확인 |
| `min_accuracy_gain > 0` | 전체 정확도가 오히려 나빠지는 규칙은 쓸모 없음 |

### Step 5-2. 선택 점수의 가중치 의미

```python
internal_selection_score = (
    mean_accuracy                     # 기본 정확도 (주성분)
    + 0.45 * min_accuracy_gain        # 최악 운영점의 정확도 향상 (안전 마진)
    + 0.20 * min_specificity_gain     # 최악 운영점의 특이도 향상
    - 0.20 * max_sensitivity_loss     # 최악 민감도 손실 패널티
    - 0.03 * mean_event_rate          # 강등 집단 내 사건율 패널티
)
```

정확도를 최대화하되, 안전(민감도 손실 최소화)을 부수적으로 고려하는 스코어. `min_accuracy_gain` 가중치(0.45)가 높은 이유: 평균이 좋아도 특정 운영점에서 나빠지는 규칙은 임상적으로 위험하다.

### Step 5-3. 두 가지 Winner 선정 전략

```python
"internal_locked":
    internal_passing.iloc[0]
    # internal_selection_score 1위
    # 외부 데이터는 보지 않음

"internal_external_audit":
    both_passing.iloc[0]
    # 내부 + 외부 모두 통과, external 정확도 향상 기준 정렬
```

| 전략 | 의미 | 특성 |
|---|---|---|
| `internal_locked` | 내부 데이터로만 규칙 확정 | 외부 데이터 미사용 = 진정한 독립 검증 유지 |
| `internal_external_audit` | 외부 일반화까지 확인된 규칙 | 더 신뢰할 수 있으나, 외부 데이터를 선정에 사용했으므로 진정한 독립 검증은 아님 |

---

## Stage 6. Winner 평가 및 결과 저장 (line 774–837)

### Step 6-1. 상세 지표 재계산

```python
detail = detail_for_winner(winner, g, s, cpos_g, cpos_s, prob_g, prob_s)
```

**왜 재계산하는가?** `search_gates()`에서 `fast_dataset_summary()`는 속도를 위해 집약된 지표만 저장했다. winner에 대해서만 `deesc_metric_row()`(더 상세한 지표 계산 함수)를 다시 실행해 임상에 보고할 수 있는 완전한 지표를 얻는다.

### Step 6-2. 교사 일치도 평가

```python
agreement_table(...)
# branch_vote_agreement_cpos:  임상양성 환자에서 CNN 투표 == 교사 투표 비율
# consensus_agreement_cpos:    임상양성 환자에서 CNN의 2-of-4 합의 == 교사의 합의 비율
```

CNN이 교사를 얼마나 잘 모방했는지 직접 측정. 일치도가 높으면 CNN이 교사의 의사결정 구조를 제대로 학습한 것. 낮다면 CNN이 다른 경로로 비슷한 성능을 달성한 것.

**임상양성 환자에 한정하는 이유**: 임상음성 환자는 애초에 게이트 대상이 아니므로, 그들의 일치도는 의미 없는 숫자를 높이는 효과만 있다.

### Step 6-3. 시각화

```python
# 3개 패널: Accuracy gain / Specificity gain / Sensitivity loss
# 내부(실선) vs 외부(점선), 운영점별(S80/S85/S90) 추이
```

세 패널이 함께 보여주는 것: "de-escalation 규칙이 정확도와 특이도를 향상시키면서(좋음) 민감도를 얼마나 희생하는가(비용)". 이 세 가지의 운영점별 추이가 안정적일수록 좋은 규칙이다.

### Step 6-4. 결과 파일 목록

| 파일 | 내용 |
|---|---|
| `exact_surrogate_teacher_details.csv` | 교사 게이트 기준선 상세 지표 |
| `{cfg}_training_log.csv` | fold별 학습 기록 |
| `{cfg}_probabilities.npz` | CNN 예측 확률 (prob_g, prob_s) |
| `surrogate_mimic_training_log.csv` | 전체 학습 로그 통합 |
| `{cfg}_same_rule_candidates.csv` | 설정별 후보 규칙 전체 |
| `surrogate_mimic_same_rule_all_candidates.csv` | 전체 후보 통합 정렬 |
| `surrogate_mimic_internal_passing_ranked.csv` | 내부 통과 규칙 순위 |
| `surrogate_mimic_internal_external_passing_ranked.csv` | 양쪽 통과 규칙 순위 |
| `{tag}_winner_details.csv` | winner 상세 de-escalation 지표 |
| `{tag}_winner_plot.png` | winner 시각화 |
| `{tag}_agreement.csv` | winner의 교사 투표 일치도 |
| `surrogate_mimic_summary.json` | 전체 파이프라인 요약 |
| `progress.json` | 실시간 진행 상황 |

---

## 전체 데이터 흐름 요약

```
AEC 곡선 (N, 128)
│
├─[교사 경로]─────────────────────────────────────────────────────
│  region_descriptor_matrix → 48개 서술자
│  → z-표준화 → TEACHER_BRANCHES 4개 선택 × sign 적용
│  → make_single_deesc() × 4 branches × 3 ops
│  → target: (N, 3, 4)  [교사 투표 행렬 = distillation 타깃]
│
├─[CNN 입력 경로]──────────────────────────────────────────────────
│  make_channels → 3채널 (원시/기울기/곡률)
│  → standardize → xg: (N, 3, 128)
│
└─[학습]──────────────────────────────────────────────────────────
   DirectVoteMimicCnn (2 configs × 2 seeds × 5 folds = 20회)
     4 RegionBranch (지역별 1D CNN → 형태 점수)
     + 임상 delta/boundary/cpos 특징
     → logits (N, 3, 4)
     → loss = branch_BCE + consensus_weight × consensus_BCE
   → prob_g, prob_s: (N, 3, 4)
          │
   [탐색]─────────────────────────────────────────────────────────
   ~700~1,200 임계값 조합 × 수십 패턴마스크 = 수만 개 후보
     codes_from_prob → 4비트 코드
     fast_dataset_summary → 민감도/특이도/정확도/건수
     dataset_pass → 통과/실패
          │
   [선정]─────────────────────────────────────────────────────────
   internal_locked         (내부만 통과 1위)
   internal_external_audit (내외부 통과 1위)
     → 상세 지표 + 교사 일치도 + 시각화 저장
```

---

## 전체 설계 철학 요약

| 설계 결정 | 이유 |
|---|---|
| Knowledge Distillation | 해석 가능한 교사 규칙의 구조를 유지하면서 CNN의 표현력을 활용 |
| 지역별 독립 브랜치 | 각 AEC 구간이 임상적으로 독립된 신호를 담고 있다는 가정 |
| 임상 특징 명시 주입 | 순수 CNN이 발견하기 어려운 임계값 경계 구조를 사전 지식으로 제공 |
| 2-of-4 합의 손실 | 하나의 지역만 보는 단순 규칙이 아닌, 다중 지역 동의를 강제 |
| 내부/외부 이중 검증 | 학습 데이터에만 맞는 규칙(과적합)을 걸러냄 |
| 민감도 손실 p값 검사 | 작은 샘플에서 우연히 발생한 손실을 진짜 위험으로 오인하지 않음 |
| 2 시드 × 5 fold | 무작위성에 의한 분산을 줄여 규칙 선정의 안정성 확보 |
| 임계값 전수 탐색 | CNN 확률을 투표로 변환하는 최적 임계값은 학습으로 결정할 수 없어 사후 탐색 |
