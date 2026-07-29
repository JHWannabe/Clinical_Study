# TP vs FP raw AEC-128 곡선 비교: p-value 산출 알고리즘

`code/compare_aec.py`의 `curve_diff_test()` (compare_aec.py:47-78)에서 사용한 방법.
포인트별(슬라이스 128개 각각)로 개별 검정하지 않고, 곡선 전체를 하나의 벡터로 취급하는
**whole-curve RMSD 기반 permutation test**.

## 절차

### 1. 검정통계량(관측값) 계산
- TP 그룹 128슬라이스 평균곡선 `mean_a`, FP 그룹 평균곡선 `mean_b`를 각각 구함
- `deviation = mean_a - mean_b` (슬라이스별 차이, 길이 128 벡터)
- `RMSD = sqrt(mean(deviation²))` → 두 평균곡선 간 거리를 스칼라 하나로 요약

### 2. Null 분포 생성 (permutation)
- TP/FP 라벨을 `n_perm=2000`회 무작위로 섞음 (`rng.shuffle`)
- 섞을 때마다 위와 동일한 방식으로 RMSD를 재계산 → `perm_stats` (길이 2000)
- 즉 "TP/FP 구분이 실제로는 의미 없다"는 귀무가설 하에서 RMSD가 우연히 어느 정도까지
  커질 수 있는지의 분포를 만듦

### 3. p-value
- `p = (관측 RMSD 이상인 permutation 개수 + 1) / (n_perm + 1)`
- 관측된 RMSD가 무작위 섞기로 나온 RMSD들보다 유의하게 크면 p가 작아짐 (one-sided 검정)

### 4. 부가 정보
- `peak_slice`: 두 곡선 차이(`|deviation|`)가 가장 큰 슬라이스 위치
- `peak_deviation`, `direction`: 그 지점에서 TP가 FP보다 큰지/작은지

## 설계 이유

128개 슬라이스를 독립적인 포인트로 나눠 각각 t-test하면 다중비교 문제(multiple
comparisons)가 생기고, 곡선의 "전체 모양 차이"라는 실제 관심 대상을 놓친다. RMSD
하나로 축약한 뒤 permutation으로 유의성을 재는 방식은 다중비교 보정 없이도
whole-curve 수준에서 타당한 p-value를 준다. AEC 곡선은 포인트별이 아니라 전체 곡선
단위로 분석한다는 기존 방침을 그대로 따른 것이다.

## 현재 결과 요약

| cohort | sex | n_TP | n_FP | curve RMSD | perm p | peak slice | peak Δ |
|---|---|---|---|---|---|---|---|
| gangnam | 전체 | 117 | 447 | 10.08 | 0.252 | 1 | -13.50 |
| sinchon | 전체 | 131 | 398 | 10.25 | 0.152 | 3 | -20.83 |
| gangnam | M | 72 | 217 | 28.58 | 0.0185 | 82 | -36.95 |
| gangnam | F | 45 | 230 | 4.68 | 0.842 | 2 | +10.74 |
| sinchon | M | 95 | 231 | 10.50 | 0.232 | 3 | -18.93 |
| sinchon | F | 36 | 167 | 15.48 | 0.248 | 1 | -27.50 |

Raw(전처리 없는) AEC 값 기준, gangnam 남성에서만 whole-curve 수준에서 유의한 차이
(p=0.0185)가 나타났고 나머지 조합(gangnam 전체/여성, sinchon 전체/남/여)은 유의하지
않았다.
