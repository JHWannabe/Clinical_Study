# Stage 1 (Clinical-only) Sensitivity 90% 임계값 선택 근거 조사

`code/baseline/clinic-only_baseline.py`가 `TARGET_SENSITIVITIES = [0.85, 0.90, 0.95]`
중 90%를 채택하는 설계에 대해 (1) 왜 100%가 아닌 90%인가, (2) 왜 Specificity가
낮은가에 대한 근거 자료 정리. 외부 논문은 2022년 이후 발표분을 우선하되, 방법론
원 출처(Youden 1950, Obuchowski 2005, Pepe 2003)는 `related_research.md`의
Newcombe(1998) 인용 전례와 동일하게 foundational 인용으로 포함.

## 0. 근거 데이터 (`outputs/0_clinic-only_baseline/clinical_only_sensitivity_comparison.csv`)

| target Se | internal Sp | internal PPV | external Sp | external PPV |
| --- | --- | --- | --- | --- |
| 0.85 | 65.04% | 24.66% | 58.47% | 27.56% |
| 0.90 | 53.59% | 20.78% | 49.04% | 24.67% |
| 0.95 | 40.17% | 17.62% | 33.38% | 20.52% |

n: internal 1090, external 926.

Se 5%p 구간별 Sp 손실 기울기(%p Sp / 1%p Se):

| 구간 | internal 기울기 | external 기울기 |
| --- | --- | --- |
| Se 0.85→0.90 | −2.29 | −1.89 |
| Se 0.90→0.95 | −2.68 | −3.13 |
| 기울기 비 (후구간/전구간) | 1.17배 | 1.66배 |

PPV 손실: internal −3.88%p(0.85→0.90) → −3.16%p(0.90→0.95); external −2.89%p → −4.15%p.

**해석**: 100%에 가까워질수록 동일한 1%p Se 이득에 대한 Sp 손실 기울기가 internal
1.17배, external 1.66배로 커진다. 아래 1~2절의 논거는 이 실측 기울기를 뒷받침하는
문헌적 근거다.

## 1. Sensitivity를 100%가 아닌 90%로 설정하는 근거

### 1-1. ROC 구조: Sp(Se)는 두 클래스 score 분포가 겹치는 한(AUC<1) 단조 감소

- Youden WJ. *Index for rating diagnostic tests.* Cancer. 1950. — Youden's J, 최적
  조작점 개념의 원 출처.
- Zweig MH, Campbell G. *Receiver-operating characteristic (ROC) plots: a
  fundamental evaluation tool in clinical medicine.* Clin Chem. 1993.

두 문헌은 정성적 원리(trade-off의 존재)만 제공하며, 기울기의 크기는 제공하지 않는다
— 위 0절의 기울기 수치(1.17배/1.66배)가 이 구조적 성질의 실측치.

### 1-2. 100% 근접 구간에서 Sp 손실이 커지는 패턴은 유사 도메인 정량 보고와 일치

[Performance of artificial intelligence in diabetic retinopathy screening: a
systematic review and meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10296189/)
(2023):

- pooled Se = 88.0% (95% CI 87.5–88.4%), pooled Sp = 91.2% (95% CI 91.1–91.3%)
- 알고리즘 서브그룹: CNN Se=95.2%/Sp=92.2%, 일반 NN Se=84.2%/Sp=92.3%, 기타
  Se=79.9%/Sp=85.7%
- FNR=12%, FPR=8.8%로 명시 보고 — 100% Se를 목표치로 설정하지 않음

주의: 이 수치는 서로 다른 알고리즘 간 비교(판별력 자체가 다름)이며, 본 연구처럼
동일 모델의 threshold sweep과는 조건이 다르다. Se가 오를 때 Sp가 함께 오르는
구간이 존재하는 것은 알고리즘 AUC 차이가 섞여 있기 때문 — **동일 모델·동일
threshold sweep 조건에서의 대가는 0절의 자체 실측 기울기가 더 정확한 근거.**

### 1-3. 100% empirical sensitivity threshold는 extreme order statistic — 통계적으로 불안정

[Pugh S, Fosdick BK, Nehring M, Gallichotte EN, VandeWoude S, Wilson A. *Estimating
cutoff values for diagnostic tests to achieve target specificity using extreme
value theory.* BMC Med Res Methodol. 2024.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10851584/)

- target 0.995에서 empirical quantile 기반 cutoff는 편향·분산이 크고, target 0.95
  에서는 empirical quantile로 충분(논문 결론).
- 본 연구 internal n=1090 기준 100% Se를 요구하면 threshold는 양성 케이스 중 점수
  최하위 1건에 고정된다 — 표본 크기 1의 order statistic이며, 이 추정치의 분산 자체를
  계산할 방법이 없다(외부 코호트 n=926 일반화 보장 불가).

### 1-4. Serial(AND-rule) 2-stage 구조: Se_combined = Se1×Se2, Sp_combined = Sp1+(1−Sp1)×Sp2

- Obuchowski NA. *Clinical Evaluation of Diagnostic Tests.* AJR. 2005.
- Pepe MS. *The Statistical Evaluation of Medical Tests for Classification and
  Prediction.* Oxford, 2003 (Ch.5, Combining Tests).

Sp1 값에 관계없이 Sp_combined ≥ Sp1이 항상 성립 — Stage 1이 Sp를 확보할 필요가
없다는 것은 정성적 주장이 아니라 이 부등식의 직접적 귀결. 또한 Se1을 0.90→0.95로
올리면 internal PPV가 20.78%→17.62%(−3.16%p)로 낮아진다 — Stage 2가 학습하는
screen-positive 그룹의 양성 base rate가 낮아져, Stage 2 모델의 minority class
표본 수가 줄고 분산이 커진다. `model_algorithm.md` 2절의 "표본이 작은 그룹에 대한
재분류는 통계적으로 불안정" 논리와 동일 구조.

## 2. Specificity가 낮은 이유

### 2-1. 정의상 귀결

Sp=53.59%(internal)/49.04%(external, Se=90% threshold)는 ROC 궤적상 threshold
선택에 의해 정해지는 값 — 1절의 산술적 결과이며 별도의 모델 결함이 아니다.

### 2-2. Clinical-only 변수의 설명력 상한 — 수치 근거

[Development and Validation of a Skeletal Muscle Prediction Equation From
Anthropometric and Demographic Data](https://www.jamda.com/article/S1525-8610(25)00582-1/fulltext)
(JAMDA, 2025)

- adjusted R² = 0.90, SEE = 1.34 kg, validation r = 0.952 (개발군 n=4013, 검증군
  n=1003)
- R²=0.90 → 미설명분산 10%. 예측오차(SEE=1.34kg)가 진단 컷오프 폭과 같은 자릿수일
  경우, 컷오프 근방 케이스의 오분류는 회귀계수 추정 오류가 아니라 잔차 자체에서
  발생한다 — age/height/weight/sex만으로는 일정 오분류율이 구조적으로 남는다는
  정량적 근거.
- **주의**: 이 R²는 SMI 연속값 회귀 기준이며 본 연구의 binary low-SMI classification
  성능 지표(Se/Sp)와 동일 척도가 아니다 — 직접 등치 불가, 방향성 근거로만 사용.

[Development of Formulas for Calculating L3 Skeletal Muscle Mass Index and
Visceral Fat Area Based on Anthropometric Parameters](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249379/)
(Front Nutr, 2022) — 동일 계열 문헌. `related_research.md` 2-3절에 이미
clinical-only baseline 비교 대상으로 인용됨.

### 2-3. Serial 설계상 Stage 1의 낮은 Sp는 설계 파라미터

1-4절과 동일 인용(Obuchowski 2005; Pepe 2003)의 AND-rule 부등식
Sp_combined ≥ Sp1에 따르면, Stage 1의 낮은 Sp는 시스템 전체 Sp의 제약이 아니라
시작점이다. `model_algorithm.md` 1절의 "2차 분류 목표=screen-positive 그룹 PPV
최대화=전체 코호트 Sp 개선"과 수식적으로 동치.

## 3. 요약

Sensitivity 90% 채택 근거:

1. 자체 sweep에서 Se 90→95%p 구간 Sp 손실 기울기가 Se 85→90%p 구간 대비 internal
   1.17배, external 1.66배로 커짐 (0절).
2. 100% Se threshold는 internal n=1090 기준 표본 1개짜리 order statistic으로
   분산 추정이 불가능(Pugh 2024).
3. Serial 구조 Sp_combined = Sp1+(1−Sp1)Sp2 (Obuchowski 2005/Pepe 2003)에서
   Sp 하한 확보는 Stage 1의 역할이 아님 — Stage 2(AEC residual)가 담당.

이 threshold에서 관측된 Sp 53.6%(internal)/49.0%(external)는 (a) 위 threshold
선택의 산술적 결과이자 (b) clinical-only 변수의 유한한 설명력(JAMDA 2025, adjusted
R²=0.90)에서 기인한 구조적 하한이다.

## 4. Stage 2 학습 시 양성비율(class balance)을 Stage 1과 유사하게 유지하는 근거

Stage 2는 Stage 1 screen-positive 그룹(양성비율=Stage 1 PPV, internal 20.78%/
external 24.67%)을 그대로 학습에 사용하며, SMOTE 등으로 50:50 인위적 리밸런싱을
하지 않는다. 이 설계를 뒷받침하는 근거.

### 4-1. 리밸런싱은 calibration을 훼손하고 discrimination은 개선하지 못한다 — 정량 근거

[van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. *The harm of class
imbalance corrections for risk prediction models: illustration and simulation using
logistic regression.* J Am Med Inform Assoc. 2022;29(9):1525–1534.](https://academic.oup.com/jamia/article/29/9/1525/6605096)

Monte Carlo 시뮬레이션 + 난소암 진단 실데이터로 random undersampling/oversampling/
SMOTE를 비교:

| 조건 | Calibration intercept | Calibration slope | AUROC |
| --- | --- | --- | --- |
| 원본 비율(uncorrected) | −0.05 ~ 0.03 | ≈1.0 | 기준 |
| Undersampling/Oversampling/SMOTE | **−1.32 ~ −1.50** | 1.0 미만(확률 과대추정) | 원본 대비 개선 없음 |
| event fraction 1% 극단 사례 | median intercept **≤ −4.5** | — | — |

결론: 리밸런싱은 판별력(AUC)은 그대로 둔 채 확률 보정(calibration)만 심하게
훼손한다(minority class 확률 과대추정, intercept 최대 −4.5). 저자 권고: **"리밸런싱
대신 threshold를 이동시켜라"** — Se/Sp 조정은 threshold-shifting만으로 리밸런싱과
동일한 효과를 얻을 수 있음을 명시.

→ 본 연구의 th2 sweep 방식(원본 양성비율 유지 + threshold만 이동, `model_algorithm.md`
"임계값 선택 기준" 절)이 이 권고와 정확히 일치.

### 4-2. 후속/보강 문헌

- [Understanding random resampling techniques for class imbalance correction and
  their consequences on calibration and discrimination of clinical risk prediction
  models.](https://pubmed.ncbi.nlm.nih.gov/38848886/) J Biomed Inform. 2024. — 다른
  리샘플링 변형까지 확장 검증, 동일 결론(discrimination 개선 없음, calibrated
  prediction 목표 시 정당화 어려움).
- [Resampling methods for class imbalance in clinical prediction models: A scoping
  review protocol.](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0330050)
  PLOS One. 2025. — minority-class prevalence 30% 미만 임상 이진분류 15년치
  문헌(2009–2024)을 정리하는 프로토콜. 아직 review 결과는 아니나 문제의식(근거가
  흩어져 있음)이 위 2022/2024 결론과 같은 방향.

### 4-3. 이론적 배경 (foundational, 2022 이전)

[Saerens M, Latinne P, Decaestecker C. *Adjusting the Outputs of a Classifier to
New a Priori Probabilities: A Simple Procedure.* Neural Computation.
2002;14(1):21–41.](https://pubmed.ncbi.nlm.nih.gov/11747533/) — 학습 데이터의 class
prior(양성비율)를 인위적으로 바꾸면 분류기가 출력하는 posterior probability가 그
바뀐 prior에 종속되어, 실제 배포 환경(원래 prior)에서는 재보정이 필요해진다는 것을
최초로 정식화. SMOTE로 50:50을 만들면 Stage 2 모델이 "50% 유병률 세계"의 확률을
출력하고, 이를 실제 유병률(≈20%)로 되돌리려면 별도 보정이 필요해짐 — van den
Goorbergh(2022)가 실증한 miscalibration 현상의 이론적 근거.

### 4-4. 본 연구 데이터와의 정합성

Stage 2 입력 그룹의 실제 양성비율(=Stage 1 PPV)은 internal 20.78%, external
24.67%(Δ3.89%p) — 두 코호트 간 유병률 차이가 크지 않으므로 SMOTE로 재조정할
필요성 자체가 약하다. van den Goorbergh(2022)의 심각한 calibration 붕괴 조건은
event fraction 1% 수준의 극단적 불균형이며, Δ3.89%p는 이와 거리가 멀다 — 현재
방식(원본 비율 유지 + th2 sweep)이 문헌상 권고와 부합.

## 참고 메모

- `model_algorithm` — 2단계 설계 원칙 문서
- `related_research` — SMI 예측식 baseline 문헌 목록 (2-3절)
- `feedback_low_smi_noninferiority_criteria` — acceptance criteria
