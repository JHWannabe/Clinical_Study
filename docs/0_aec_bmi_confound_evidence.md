# AEC-Clinical Confound 입증 근거

`1_aec_residual_reclassify.py`가 AEC-128 곡선을 clinical 변수(age/height/weight/sex)로
residualize하는 전제 — "clinical 변수가 AEC 곡선과 Low-SMI 사이의 confounder다" — 를
실제로 입증한 결과 정리. 코드 주석의 `project_aec_curve_bmi_confound` 메모가 가리키는
근거가 이 문서에 해당.

> Low-SMI 라벨은 이 문서 전체에서 `SMI = TAMA / (Height[m])²`(M<45.4, F<34.4 cutoff) —
> `clinic-only_baseline.py: load_cohort()` 및 `1_aec_residual_reclassify.py`가 실제
> 학습에 쓰는 정의 — 하나로 통일해 계산한다(internal Low-SMI n=129). 이전에는
> `aec_curve_comparison.py`가 metadata 시트의 별도 `SMI` 컬럼을 잘못 읽어 다른 인원수
> (n=291)로 계산됐던 적이 있었는데, 그 계산은 폐기하고 아래 내용을 전부 올바른 정의로
> 재계산했다.

## 0. Confound의 정의 (Greenland & Robins 기준)

변수 C가 노출(X) → 결과(Y) 관계의 confounder이려면 다음 3가지를 모두 만족해야 한다.

1. C가 노출(X)과 연관되어 있다
2. C가 결과(Y)와 연관되어 있다
3. C가 X→Y 인과경로의 매개변수(mediator)가 아니다

여기서는 X = AEC-128 곡선, Y = Low-SMI 여부, C = age/height/weight/sex(대표로 BMI 사용).

## 1. 조건 1 — clinical 변수 → AEC 곡선 연관

gangnam 코호트(internal, n=1090)의 환자별 정규화 AEC-128 곡선을, 아래 각 변수를 기준으로
두 그룹(또는 그 이상)으로 나눈 뒤 "두 그룹의 평균 곡선이 얼마나, 그리고 우연이 아니라고
할 만큼 다른가"를 매 변수마다 독립적으로 검정한 것. 검정 방법 자체(RMSD·순열검정)는 아래
"표 읽는 법"에 그대로 풀어 썼다. 산출 스크립트는 `code/baseline/aec_curve_comparison.py`의
`curve_diff_test()`, 원본 수치는 `outputs/0_clinic-only_baseline/aec_curve_comparison/
gangnam/00_group_diff_summary.csv`.

### 표 읽는 법 (RMSD / peak / 방향) — `aec_curve_comparison.py: curve_diff_test()`

- **RMSD (curve_rmsd)** — 실제 검정통계량이자 p-value가 매기는 대상. 두 그룹의 평균 AEC
  곡선(길이 128 벡터)을 슬라이스별로 뺀 뒤, 그 128개 차이값의 root-mean-square
  (`sqrt(mean(diff²))`). **128 slice 전체에 걸친 "곡선 하나 대 곡선 하나"의 거리**이지
  특정 slice 하나만 보는 게 아니다 — point-wise가 아니라 curve-level로 검정한다는 이
  프로젝트의 원칙(`feedback_aec_curve_wholistic`)이 여기 그대로 적용된 것.
- **p-value**의 출처 — RMSD 자체는 순열검정(permutation test, n_perm=2000)으로 유의성을
  매긴다: 그룹 라벨을 2000번 무작위로 섞어 매번 RMSD를 다시 계산하고, "관측 RMSD 이상으로
  큰 값이 나온 순열 비율"이 p-value다. 즉 "이 정도로 큰 곡선 전체 차이가 우연히 나올
  확률"을 직접 시뮬레이션한 것 — 정규분포 가정이 없는 비모수 검정.
- **peak_slice / peak_deviation / 방향** — p-value와는 별개인 **참고 정보**. 128개
  slice별 차이(`평균곡선A − 평균곡선B`) 중 절댓값이 가장 큰 지점(peak_slice)과 그 값
  (peak_deviation)을 보여준다. **방향**은 이 peak_deviation의 부호를 "그룹1 > 그룹2" /
  "그룹1 < 그룹2"로 풀어 쓴 것 — "두 곡선이 가장 크게 벌어지는 지점에서 어느 그룹이
  더 높은가"를 뜻한다(3그룹 이상 비교는 부호가 없어 방향 표기 없음).

### 1-1. Residualizer 입력 4변수 (`raw_clinical_matrix`와 정확히 일치)

| 변수 | 그룹(n) | RMSD | peak slice | peak deviation | 방향 | p |
| --- | --- | --- | --- | --- | --- | --- |
| 성별 | Male(390) / Female(700) | 0.1178 | 108 | 0.1713 | Male > Female | 0.0005 |
| 나이 | ≤median(549) / >median(541) | 0.0816 | 8 | 0.1291 | ≤median > >median | 0.0005 |
| 신장 | ≤median(574) / >median(516) | 0.0515 | 110 | −0.0863 | ≤median < >median | 0.0005 |
| 체중 | ≤median(545) / >median(545) | 0.0911 | 8 | 0.1198 | ≤median > >median | 0.0005 |

4변수 전부 AEC 곡선 형태와 p<0.001 수준으로 유의하게 연관 — residualizer가 회귀 대상으로
삼는 4개 변수와 정확히 일치.

### 1-2. 파생/추가 변수 (모델 입력은 아니지만 같은 CSV에 있는 진단용 비교)

| 변수 | 그룹(n) | RMSD | peak slice | peak deviation | 방향 | p |
| --- | --- | --- | --- | --- | --- | --- |
| BMI | ≤median(547) / >median(543) | 0.0751 | 5 | 0.1105 | ≤median > >median | 0.0005 |
| TAMA | ≤median(547) / >median(543) | 0.0245 | 86 | −0.0383 | ≤median < >median | 0.0005 |
| Low-SMI | Low(129) / Non-low(961) | 0.0210 | 120 | 0.0401 | Low > Non-low | 0.0475 |
| Scan length(z_range) | ≤median(574) / >median(516) | 0.0694 | 94 | −0.0943 | ≤median < >median | 0.0005 |
| Slice thickness | ≤median(557) / >median(533) | 0.0574 | 106 | −0.0778 | ≤median < >median | 0.0005 |
| Vendor | Siemens(568)/GE(318)/Philips(202) | 0.0304 | 120 | 0.0430 | (3군, 방향 없음) | 0.0005 |
| BMI 4구간 | Underweight(57)/Normal(429)/Overweight(280)/Obese(324) | 0.0559 | 9 | 0.0772 | (4군, 방향 없음) | 0.0005 |
| 성별×Low-SMI | M/Low(76), M/Non-low(314), F/Low(53), F/Non-low(647) | 0.0701 | 112 | 0.0996 | (4군, 방향 없음) | 0.0005 |

(Low-SMI/성별×Low-SMI 두 행은 SMI 정의 버그 수정 후 재계산된 값 — 위 정정 안내 참고.
나머지 행은 SMI에 의존하지 않아 버그의 영향을 받지 않았다.)

BMI/TAMA는 height·weight의 파생치라 조건 1의 추가 근거로 쓸 수 있지만 모델(residualizer/
stage-1) 입력은 아니다. BMI = weight/height²는 이미 입력에 포함된 height·weight 두
변수의 결정론적 함수이므로 "모델이 모르는 변수"를 끌어온 게 아니다 — 다만 residualizer는
height·weight를 **선형**으로만 회귀시키므로, BMI가 만드는 **비선형** confound가
완전히 제거된다는 보장은 없다(3절 이후에서 재확인 필요성 언급).
Scan length/Slice thickness/Vendor는 clinical 변수가 아니라
**기술적(acquisition) covariate** — AEC 곡선이 촬영 조건에도 민감하다는 것을 보여줄 뿐,
confound 논증(조건 1~3)의 직접 근거는 아니고 residualizer가 다루지 않는 별도의 잔여
변동원으로 남는다는 점을 알려주는 참고 정보.

## 2. 조건 2 — clinical 변수 → Low-SMI 연관

Se/Sp 표(아래)는 threshold sweep에서 나온 point estimate일 뿐 그 자체로는 가설검정이
아니다 — "유의미하다"는 claim은 별도의 유의성 검정으로 뒷받침해야 한다. `clinic-only_
baseline.py`에 `auc_significance_stats()` / `plot_roc_curve()`를 추가해 internal OOF
확률(n=1090, 양성 n=129/음성 n=961)로 정식 계산하도록 파이프라인에 반영했다.

- **AUC = 0.8285** (95% CI [0.7913, 0.8628], positive/negative 층화 bootstrap 3000회)
- **Mann-Whitney U 검정**(AUC≠0.5, 즉 "점수 분포가 양/음성 간 무작위와 다르다"에 대한
  정식 검정): **p = 7.47×10⁻³⁴**
- 산출물: `outputs/0_clinic-only_baseline/clinical_only_roc_curve.png` (ROC curve figure),
  `outputs/0_clinic-only_baseline/clinical_only_auc_significance.csv` (auc/ci/p 원본 수치)

CI 하한(0.791)이 0.5와 크게 떨어져 있고 p값이 사실상 0에 가까우므로, clinical
4변수(age/height/weight/sex)가 Low-SMI를 우연 수준을 훨씬 넘어 판별한다는 것이 정식
검정으로 확인된다 — 이게 조건 2("C가 Y와 연관")의 실제 근거.

참고용 point estimate (`outputs/0_clinic-only_baseline/clinical_only_sensitivity_comparison.csv`):

| target Se | internal Sp | internal PPV | external Sp | external PPV |
| --- | --- | --- | --- | --- |
| 0.90 | 53.49% | 20.74% | 49.30% | 24.76% |

n: internal 1090, external 926.

## 3. 조건 3 관련 계층별 재검토 (BMI 4구간 stratification)

이 절이 확인하는 것은 "Low-SMI~AEC 관계가 BMI 구간에 따라 방향이 반전되는
Simpson's paradox 패턴이 있는가"이다. 결과부터 말하면 **방향 반전은 관찰되지
않는다** — 아래 3-1절 참고. 조건 3("매개변수가 아님")의 실제 근거는 이 절이 아니라
4절(SMI 통제 후에도 BMI 효과가 남는지 검정)이다.

### 3-0. 애초에 무엇을 확인하려던 절차였나

**정의**: 전체(pooled) 데이터에서 관찰되는 두 변수(X, Y) 간 연관성의 방향이, 제3의 변수(C)로
층화(stratify)했을 때 하위집단에서 반대로 나타나거나 사라지는 현상(Simpson 1951; 고전적
예시는 UC Berkeley 입학 성비 사례, Bickel et al. 1975)을 Simpson's paradox라 부른다. C가
X·Y 둘 다와 연관되어 있고 계층별 표본 크기·비율이 불균등할 때, pooled association이 각
계층 고유의 관계를 나타내지 못하고 왜곡될 수 있다는 것이 이 절이 원래 확인하려던
절차였다: ① BMI를 무시한 pooled 검정, ② BMI 4구간 각각의 내부에서 독립적으로 반복한
stratified 검정, ③ 계층 간 방향·유의성 비교, ④ pooled 방향이 표본이 가장 큰 계층과
반대인지 확인.

### 3-1. 수치 근거

WHO 아시아 기준 BMI 4구간(Underweight <18.5, Normal 18.5-22.9, Overweight 23-24.9, Obese
≥25 kg/m²) 각각의 내부에서, Low-SMI 환자군과 Non-low-SMI 환자군의 평균 AEC 곡선을
독립적으로 비교한 결과(RMSD/방향 표기는 1절 "표 읽는 법" 참고):

| BMI 구간 | Low-SMI 환자 수 | Non-low-SMI 환자 수 | 방향(어느 쪽 곡선이 더 높은가) | RMSD | p |
| --- | --- | --- | --- | --- | --- |
| Underweight | 29 | 28 | Low SMI < Non-low SMI | 0.049 | 0.117 (n.s.) |
| Normal | 69 | 360 | Low SMI **>** Non-low SMI | 0.039 | **0.0035** |
| Overweight | 17 | 263 | Low SMI **>** Non-low SMI | 0.065 | **0.0065** |
| Obese | 14 | 310 | Low SMI < Non-low SMI | 0.032 | 0.246 (n.s.) |

Pooled(marginal, BMI 무시) 검정은 RMSD=0.021, **p=0.047**(경계적 유의), 방향은
"Low-SMI 환자의 곡선이 Non-low-SMI보다 **높다**"(peak slice 120번 기준) — 즉 pooled
방향과 Normal·Overweight 두 계층의 방향이 **동일**하다. Normal(n=429, 전체의 39%로
표본이 가장 큰 계층)과 Overweight(n=280) 모두 p<0.05로 유의하지만 부호가 서로 반대가
아니므로, 원래 보고했던 "방향 반전(Simpson's paradox)" 패턴은 **재현되지 않는다**.
Underweight·Obese 두 계층은 pooled와 반대 부호의 점추정치를 보이지만 유의하지 않다
(각각 Low-SMI 환자 수가 29명, 14명으로 작아 검정력이 낮다는 점도 함께 고려해야 함) —
이 역시 "반전"이 아니라 표본 크기에 따른 잡음에 가깝다.

**해석**: 이 절의 방향-반전 논리로 조건 3(진짜 confound인지, 매개변수는 아닌지 확인)을
보강하려던 시도는 성립하지 않는다. 다만 이는 "confound가 없다"는 뜻이 아니다 —
①(1절, clinical→AEC 연관, p<0.001)과 ②(2절, clinical→SMI AUC=0.83)는 그대로 성립하며,
매개변수가 아니라는 조건 3의 실질적 근거는 4절(SMI를 통계적으로 고정한 뒤에도 BMI
효과가 남는지)이 담당한다. 다만 pooled Low-SMI~AEC 연관 자체가 p=0.047(경계적 유의)에
불과하다는 점은, "AEC 곡선이 clinical 변수를 거치지 않고도 어느 정도 Low-SMI 신호를
담고 있다"는 주장의 강도를 낮춰서 해석해야 함을 시사한다 — Stage-2가
clinical-residualized AEC 정보로 얻을 수 있는 개선 폭에 대한 기대치도 이에 맞춰
보수적으로 잡아야 한다.

## 4. Mediator가 아님을 확인 (조건 3 보강)

BMI가 confounder가 아니라 사실은 "SMI를 거쳐서만 AEC에 영향을 주는 매개변수(mediator)"일
가능성도 배제해야 한다 — 매개변수라면 SMI를 통계적으로 고정(통제)한 순간 BMI와 AEC의
관계는 사라져야 한다. 그래서 SMI 값을 좁은 구간으로 맞춰 SMI 차이를 통제한 부분집합
(Non-low-SMI만, n=961: Underweight 28 / Normal 360 / Overweight 263 / Obese 310) 안에서,
BMI 4구간 간 AEC 곡선 차이를 다시 검정했다(`15_aec_curve_bmi4_smi_controlled.png` 산출).
결과는 SMI를 통제한 뒤에도 BMI 4구간 간 AEC 곡선 차이가 여전히 유의했다(RMSD=0.063,
peak slice 11, **p=0.0005**) — 즉 BMI는 SMI를 거치는 경로만으로 AEC에 영향을 주는 게
아니라, SMI와 무관한 별도 경로(예: 체지방 두께에 의한 X선 감쇠 변화)로도 AEC에 영향을
준다는 뜻이다. 매개변수였다면 이 통제 후 차이는 사라졌어야 하므로, 이 결과는 BMI가
매개변수가 아니라 진짜 confounder라는 조건 3을 뒷받침한다.

## 5. 외부 정합성

`docs/related_research.md` 1절 — 이홍선 교수님 내부자료(`260506_이홍선교수님.pdf`)에서도
"handcrafted AEC feature는 BMI를 넣으면 추가 효과가 거의 사라진다"는 독립적 결론이 이미
존재 — 같은 confound 패턴이 별도 데이터/분석에서도 재현됨.

## 6. 요약

| 조건 | 무엇을 검정했나 | 결과 |
| --- | --- | --- |
| 1. C-X 연관 | age/height/weight/sex 각각으로 나눈 두 그룹의 AEC 곡선 비교 | 전부 p<0.001 |
| 2. C-Y 연관 | clinical 4변수 모델의 internal OOF AUC 유의성(Mann-Whitney U) | AUC=0.8285 [0.7913,0.8628], p=7.5e-34 |
| 3. 방향 반전 | BMI 4구간 각각의 내부에서 Low-SMI vs Non-low-SMI AEC 곡선 비교 | 방향 반전 없음 (Normal/Overweight 모두 유의, 같은 방향); pooled는 p=0.047로 경계적 유의 |
| 3-보강. non-mediator | SMI를 통제한 부분집합에서 BMI 4구간 간 AEC 곡선 비교 | SMI 통제 후에도 p<0.001 |

조건 1·2는 명확히 충족되고, 조건 3("매개변수가 아님")은 3절이 아니라 3-보강(4절)의
SMI-통제 재검정으로 충족된다 — 3절의 stratified 비교는 방향 반전을 보여주지 못했으므로
독립적 보강 근거로 쓰지 않는다. 조건 1·2·3-보강 세 갈래가 실측 데이터로 충족되므로,
AEC 곡선을 clinical 변수로 residualize하는 전처리(`1_aec_residual_reclassify.py` Step 2)는
임의의 선택이 아니라 confound 제거라는 명시적 근거를 가진 절차다 — 다만 pooled
Low-SMI~AEC 연관이 p=0.047로 경계적 유의에 그친다는 점(3-1절 해석 참고)은
residualization으로 얻는 이득의 기대치를 보수적으로 잡아야 함을 시사한다.

## 7. 주제별 레퍼런스 논문

이 문서의 각 절에서 쓴 통계적 개념·방법론의 외부 출처. Simpson's paradox 원 논문 등
`aec_residual_related_papers.md` 5절에 이미 있는 것은 중복 기재하지 않고 표시만 했다.

### 0절 — Confound의 정의

- Greenland S, Robins JM, Pearl J. *Confounding and Collapsibility in Causal
  Inference.* Statistical Science. 1999;14(1):29–46. — 0절의 3조건 정의를 인과추론
  틀에서 엄밀하게 정식화한 원 논문.
- VanderWeele TJ, Shpitser I. *On the Definition of a Confounder.* Annals of
  Statistics. 2013;41(1):196–220. — 전통적 3조건 체크리스트가 실패하는 반례를 지적하고
  더 엄밀한 정의를 제시. 0절이 교과서적 체크리스트에 의존하고 있다는 한계를 밝히는 근거.
- Rothman KJ, Greenland S, Lash TL. *Modern Epidemiology*, 3rd ed. Lippincott
  Williams & Wilkins, 2008. — 역학 실무에서 쓰는 "노출과 연관 / 결과와 연관 / 매개변수
  아님" 3조건 체크리스트의 표준 교과서 출처.

### 1절 — 곡선 전체(RMSD) 순열검정, 연속형 confounder를 선형으로만 통제하는 것의 한계

- Ramsay JO, Silverman BW. *Functional Data Analysis*, 2nd ed. Springer, 2005. —
  128-slice 곡선을 포인트별이 아니라 함수(curve) 단위로 다루는 통계적 근거
  (`feedback_aec_curve_wholistic`와 직결).
- Ernst MD. *Permutation Methods: A Basis for Exact Inference.* Statistical
  Science. 2004;19(4):676–685. — 정규분포 가정 없이 그룹 라벨을 섞어 검정통계량의
  귀무분포를 직접 시뮬레이션하는 순열검정의 이론적 근거.
- Cochran WG. *Analysis of Covariance: Its Nature and Uses.* Biometrics.
  1957;13(3):261–281. — 공변량을 회귀로 통제(잔차화)한 뒤 비교하는 ANCOVA 접근의
  classical 원 논문 — residualizer 설계 자체의 통계적 뿌리.
- Brenner H, Blettner M. *Controlling for Continuous Confounders in
  Epidemiologic Research.* Epidemiology. 1997;8(4):429–434. — 연속형 confounder를
  선형항으로만 넣었을 때 남는 잔여 confounding(residual confounding) 문제 —
  BMI=weight/height²라는 비선형 항이 height·weight의 선형회귀만으로 완전히
  제거되지 않을 수 있다는 1절의 caveat을 뒷받침.

### 2절 — AUC의 유의성 검정(Mann-Whitney U)과 bootstrap CI

- Hanley JA, McNeil BJ. *The Meaning and Use of the Area under a Receiver
  Operating Characteristic (ROC) Curve.* Radiology. 1982;143(1):29–36. — AUC가
  "무작위로 뽑은 양성이 무작위로 뽑은 음성보다 높은 점수를 받을 확률"이며 이것이
  Mann-Whitney U 통계량과 동치임을 보인 원 논문 — 2절 검정 선택의 근거.
- Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman &
  Hall/CRC, 1993. — 2절에서 AUC 95% CI를 구할 때 쓴 percentile bootstrap 방법론의
  표준 출처.

### 3절 — Simpson's Paradox / 방향 반전(effect modification) 참고 문헌

3절이 확인하려 한 방향-반전 패턴은 이 데이터에서 재현되지 않았다(3-1절). 아래 문헌은
3절이 시도한 절차의 개념적 배경으로 남겨둔다.

- Simpson EH (1951), Norton & Divine (2015) — 이미 `aec_residual_related_papers.md`
  5절에 있음.
- Bickel PJ, Hammel EA, O'Connell JW. *Sex Bias in Graduate Admissions: Data
  from Berkeley.* Science. 1975;187(4175):398–404. — Simpson's paradox의 가장 널리
  인용되는 실제 사례(대학원 입학 데이터) — 전체 합격률은 성별 차이가 있어 보이지만
  학과별로 나누면 반대/무관하게 나오는 고전적 예시.
- VanderWeele TJ. *On the Distinction Between Interaction and Effect
  Modification.* Epidemiology. 2009;20(6):863–871. — Simpson's paradox와
  effect modification의 구분 근거.

### 4절 — Mediator와 Confounder의 구분

- Baron RM, Kenny DA. *The Moderator-Mediator Variable Distinction in Social
  Psychological Research.* Journal of Personality and Social Psychology.
  1986;51(6):1173–1182. — mediator/moderator/confounder를 통계적으로 구분하는 가장
  널리 인용되는 원 논문 — 4절이 "SMI 통제 후에도 BMI 효과가 남는지"를 검정 기준으로
  삼은 논리의 근거.
- VanderWeele TJ. *Mediation Analysis: A Practitioner's Guide.* Annual Review
  of Public Health. 2016;37:17–32. — 위 1986년 틀을 현대적 인과추론 관점에서
  업데이트한 실무 가이드.

## 참고 메모

- `project_aec_curve_bmi_confound` — 이 문서가 뒷받침하는 원 메모
- `model_algorithm` — Step 2 residualization 설계 원칙 문서
- `aec_residual_related_papers` — residualization 방법론 자체의 외부 문헌 근거
- `related_research` — 이홍선 교수님 내부자료 등 외부 정합성 자료
