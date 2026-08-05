# code/*.py 6개 파일 — 방법론별 레퍼런스 분류

`code/` 아래 6개 스크립트(`step0_aec_clinic_correlation.py`, `step0_clinic-only_baseline.py`,
`step1_clinic_aec_mean.py`, `step2_clinic_aec_back_half.py`, `step2_clinic_aec_segment_select.py`,
`step3_clinic_aec_shape.py`)가 실제로 쓰는 방법론 요소를 분해하고, 각 요소에 대응하는
레퍼런스를 분류한다. AEC 신호 자체(raw/patient-wise 정규화, 구간평균의 functional-data
해석)에 대한 근거는 이미 `docs/aec_preprocessing_related_research.md`에 정리되어
있으므로 여기서는 **그 문서가 다루지 않는 나머지 방법론 요소**(clinic4→체성분 예측,
K-fold OOF, bootstrap CI, internal/external 검증 프로토콜, CV 내부 feature selection
leakage 방지, 잔차 진단, 곡선 구간평균/형태 feature의 일반 시계열 이론 근거, 구간
선택 필터법)를 새로 조사해 보강했다.

## 방법론 요소 목록 (A~K)

| 코드 | 방법론 요소 | 실제 사용 함수/코드 | 레퍼런스 출처 |
| --- | --- | --- | --- |
| A | AEC-128을 구간평균(raw)으로 축약해 회귀/상관 predictor로 사용 | `segment_means()` | 기존 문서 B, D, E4 |
| B | clinic4(성별/나이/키/몸무게)만으로 CT 체성분(근육·지방 면적) 예측 | `LinearRegression()` on clinic4 | **신규** |
| C | K-fold CV + out-of-fold(OOF) 예측으로 internal 성능 산출 | `cross_val_predict(cv=KFold(...))` | **신규** |
| D | R²의 bootstrap 95% CI | `regression_significance_stats()`의 `boot_r2` | **신규** |
| E | internal(CV)로 학습 → external에 동결 모델 1회 검증하는 프로토콜 | `model.fit(int); model.predict(ext)` 패턴 전체 | **신규** ([[feedback_internal_external_validation_discipline]]과 정합) |
| F | CV fold 내부에서만 feature selection(SelectKBest) 수행해 leakage 방지 | `step2_clinic_aec_segment_select.py`의 `build_pipeline()`(Pipeline+ColumnTransformer) | **신규** |
| G | 잔차 진단(Q-Q, Shapiro-Wilk, Scale-Location) | `_draw_residual_row()`의 `stats.probplot`, `stats.shapiro` | **신규** |
| H | 확립된 임상 cutoff이 없는 연속형 체성분 feature를 logistic regression용 이분형 outcome으로 정의 | 미구현 (`step0_clinic-only_baseline.py`에 logistic regression 확장 검토 중) | **신규, 적용 예정** |
| I | 곡선을 N구간 평균으로 압축하는 일반 시계열 이론(PAA) — 구간평균·앞뒤50%비율의 근거 | `segment_means()`, `step3_clinic_aec_shape.py`의 `upper_lower_ratio` | **신규** |
| J | 곡선의 SD·Skewness 등 통계적 모멘트를 요약 feature로 사용 | `step3_clinic_aec_shape.py`의 `shape_features()` | **신규** |
| K | F-검정 기반 filter법으로 관련 변수(구간)만 순위화해 선택 | `step2_clinic_aec_segment_select.py`의 `SelectKBest(f_regression)` | **신규** (F를 보완 — F는 leakage 방지, K는 선택 기준 자체의 근거) |

---

## B. clinic4 → CT 체성분 예측 (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Cao Y et al. *Development of Formulas for Calculating L3 Skeletal Muscle Mass Index and Visceral Fat Area Based on Anthropometric Parameters.* Front Nutr. 2022;9:910771.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249379/) | 성별/나이/키/몸무게만으로 L3 단면 골격근량지수(SMI)·내장지방면적(VFA)을 예측하는 선형회귀식 직접 제시 — `step0_clinic-only_baseline.py`가 하는 것과 **동일한 과제**(구조: 344명 훈련/134명 검증). SMI 식 adj R²=0.597, VFA 식 adj R²=0.581. 체중이 VFA 예측 R²의 80% 이상, 성별이 SMI 예측 R²의 약 40%를 설명 — 귀하 코드의 clinic4 baseline이 어느 정도 R²를 내는 게 타당한지 비교 기준으로 인용 가능 |
| [*Computed tomography-based muscle and fat composition in a Dutch population: a cross-sectional study.* Insights Imaging. 2025;16:XX.](https://link.springer.com/article/10.1186/s13244-025-02114-2) | CT 기반 근육/지방 조성의 성별·연령별 정상 참고범위(population reference) 제시 — clinic4 예측의 residual이 정상 변동 범위 내인지 해석할 때 참고 |

**적용 파일**: `step0_clinic-only_baseline.py`(직접 대응), `step1_clinic_aec_mean.py`/`step2_clinic_aec_back_half.py`/`step2_clinic_aec_segment_select.py`/`step3_clinic_aec_shape.py`(모두 이 clinic4 baseline에 AEC를 추가하는 구조이므로 baseline 정당성 근거로 공유)

---

## C. K-fold Cross-Validation + Out-of-Fold 예측 (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Kohavi R. *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* Proc. 14th IJCAI. 1995:1137-1143.](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf) | k-fold CV(특히 10-fold)가 모델 선택/정확도 추정에서 leave-one-out이나 단순 홀드아웃보다 편향-분산 트레이드오프가 우수함을 실증. `KFold(n_splits=5)` + `cross_val_predict`로 산출하는 OOF R²의 방법론적 근거 |

**적용 파일**: 6개 스크립트 중 회귀를 수행하는 5개(`step0_clinic-only_baseline.py`, `step1_clinic_aec_mean.py`, `step2_clinic_aec_back_half.py`, `step2_clinic_aec_segment_select.py`, `step3_clinic_aec_shape.py`) 전부

---

## D. R²의 Bootstrap 95% 신뢰구간 (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Efron B. *Estimating the Error Rate of a Prediction Rule: Improvement on Cross-Validation.* J Am Stat Assoc. 1983;78(382):316-331.](https://www.jstor.org/stable/2288636) | Bootstrap 기반 예측오차 추정의 원조 |
| [Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall/CRC, 1993.](https://doi.org/10.1201/9780429246593) | Percentile bootstrap CI의 표준 교과서 근거 — `boot_r2`를 `rng.integers(0, n, size=(n_boot, n))`로 재표집 후 `np.percentile(..., [2.5, 97.5])`로 CI를 구하는 방식이 정확히 이 percentile method |

**적용 파일**: 회귀를 수행하는 5개 스크립트 전부 (`regression_significance_stats()` 함수가 5개 파일에 동일하게 복제되어 있음)

---

## E. Internal(CV)-External(동결 모델 1회 검증) 프로토콜 (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Steyerberg EW. *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating.* 2nd ed. Springer, 2019.](https://link.springer.com/book/10.1007/978-3-030-16399-0) | Internal validation(같은 모집단 내 CV/bootstrap)과 external validation(다른 모집단, 모델 재학습 없이 1회 평가)을 구분하는 표준 프레임워크 원전 |
| [Moons KGM et al. *Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): Explanation and Elaboration.* Ann Intern Med. 2015;162(1):W1-W73.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772854/) | TRIPOD 가이드라인 — internal/temporal/geographic(external) validation을 명확히 구분해 보고하도록 요구. Gangnam(internal, CV)/Sinchon(external, 동결 모델) 2-코호트 구조가 이 분류의 geographic external validation에 해당 |
| [Ramspek CL et al. *External validation of prognostic models: what, why, how, when and where?* Clin Kidney J. 2021;14(1):49-58.](https://pubmed.ncbi.nlm.nih.gov/33564405/) | External validation을 "모델을 재적합하지 않고 새 코호트에 그대로 적용"으로 명확히 정의 — `model.fit(internal)` 후 `model.predict(external)`만 하고 재학습하지 않는 코드 패턴과 정확히 대응 |

**적용 파일**: 회귀를 수행하는 5개 스크립트 전부. [[feedback_internal_external_validation_discipline]] 메모리(여러 configuration을 external로 비교 채택하면 안 된다는 규율)의 학술적 근거이기도 함

---

## F. CV Fold 내부 Feature Selection — Leakage 방지 (신규)

`step2_clinic_aec_segment_select.py`는 `SelectKBest(f_regression)`를 `Pipeline` 안에 넣어
`cross_val_predict`가 각 fold의 **학습 데이터에서만** top-k 구간을 선택하도록 강제한다
(스크립트 주석에도 "leakage를 막는다"고 명시). 이 설계가 방지하는 정확한 실수에 대한
근거:

| 문헌 | 핵심 내용 |
| --- | --- |
| [Vabalas A, Gowen E, Poliakoff E, Casson AJ. *Machine learning algorithm validation with a limited sample size.* PLOS ONE. 2019;14(11):e0224365.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365) | 표본 수백~수천 명대의 소표본에서 일반 K-fold CV가 성능을 얼마나 낙관적으로 편향시키는지 시뮬레이션으로 정량화(2019) — 귀하 코호트(n~1000대) 규모와 직접 대응. slide2 item④의 주 인용(2026-08-03 세션에서 Ambroise & McLachlan 2002 대신 채택, "레퍼런스가 너무 오래됐다"는 지적에 따른 교체) |
| [Ambroise C, McLachlan GJ. *Selection bias in gene extraction on the basis of microarray gene-expression data.* Proc Natl Acad Sci USA. 2002;99(10):6562-6566.](https://www.pnas.org/doi/10.1073/pnas.102102699) | Feature selection을 전체 데이터로 먼저 수행한 뒤 CV를 돌리면 에러율이 심각하게 낙관적으로 편향됨을 microarray 데이터로 실증한 고전 사례 — feature-selection-then-CV 문제의 대표적 반증. 코드 설계(F 요소) 자체의 근거로는 여전히 유효, 참고용 병기 |
| [Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning.* 2nd ed. Springer, 2009. §7.10.2 "The Right and Wrong Way to Do Cross-validation".](https://hastie.su.domains/ElemStatLearn/) | "Right way"(feature selection을 CV 루프 안에서 fold마다 재수행) vs "Wrong way"(전체 데이터로 먼저 선택)를 예제로 대조 — `build_pipeline()`이 `ColumnTransformer`+`SelectKBest`를 `Pipeline`으로 묶어 `cross_val_predict`에 넘기는 구현이 정확히 "Right way" |

**적용 파일**: `step2_clinic_aec_segment_select.py`만 해당 (다른 5개는 feature selection 자체가 없음)

---

## G. 잔차 진단 (Q-Q, Shapiro-Wilk, Scale-Location) (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Shapiro SS, Wilk MB. *An Analysis of Variance Test for Normality (Complete Samples).* Biometrika. 1965;52(3/4):591-611.](https://www.jstor.org/stable/2333709) | `stats.shapiro()`로 잔차 정규성을 검정하는 원 논문 |
| [Belsley DA, Kuh E, Welsch RE. *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity.* Wiley, 1980.](https://onlinelibrary.wiley.com/doi/book/10.1002/0471725153) | Residuals-vs-fitted, Normal Q-Q, Scale-Location(표준화잔차 제곱근) 4종 진단 플롯 세트의 표준 근거 — R의 `plot.lm()` 기본 4분할도 이 체계를 따름, `_draw_residual_row()`의 4개 서브플롯과 1:1 대응 |

**적용 파일**: `step0_clinic-only_baseline.py`만 해당 (`plot_residual_diagnostics`가 이 파일에만 있음)

---

## H. Cutoff이 없는 연속형 체성분 feature의 logistic regression용 이분화 (신규, 적용 예정)

`step0_clinic-only_baseline.py`가 예측하는 IMATA/NAMA/LAMA/VAT/SAT/TAMA/Total Fat 중
TAMA(Low-SMI = TAMA/Height²)만 임상 cutoff이 확정돼 있고([[project_smi_label_bug_and_residual_evidence]]),
나머지는 sarcopenia SMI 같은 확립된 국제 기준이 없다. 이런 경우 logistic regression용
이분형 outcome을 어떻게 정의하는지에 대한 근거.

| 문헌 | 핵심 내용 |
| --- | --- |
| [Low appendicular skeletal muscle mass is associated with the risk of mortality among adults in the United States. Sci Rep. 2025.](https://www.nature.com/articles/s41598-025-94357-8) | Sex-specific 하위 20%(lowest quintile)를 low muscle mass cutoff으로 채택 |
| [Investigating the relationship between body roundness index and low muscle mass... Focus on visceral adipose tissue.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12364339/) | 동일하게 sex-specific quintile 기반 low muscle mass 정의를 사용, 여러 CT/DXA 체성분 연구에서 하위 quintile·quartile·tertile이 관례적으로 쓰임을 확인 |
| [Sex-specific CT-derived reference cutoffs for body composition in healthy Brazilian adults: a multicenter study. Sci Rep. 2026.](https://www.nature.com/articles/s41598-026-60677-6) | 젊고 건강한(young, normal-BMI) reference subgroup을 sex별로 정의해 근육량은 mean−1SD/−2SD, 지방량은 mean+1SD/+2SD를 cutoff으로 산출 — quantile 방식보다 방어 가능한 대안이나 reference subgroup 확보가 전제 |
| [Studenmund JR et al. Defining sarcopenia and myosteatosis: the necessity for consensus on a technical standard and standardized cut-off values.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8977957/) | Sarcopenia/myosteatosis cutoff 자체가 아직 국제 합의가 없다는 것이 학계 공식 입장 — quantile split은 "표준 부재 시의 관례"이지 정답은 아님을 뒷받침 |
| [Reference Values for Skeletal Muscle Mass – Current Concepts and Methodological Considerations. Nutrients. 2020;12(3):755.](https://www.mdpi.com/2072-6643/12/3/755) | 골격근량 reference value 산출 방법론(quantile-based vs population-reference-based)을 체계적으로 비교 |
| [Bennette C, Vickers A. Against quantiles: categorization of continuous variables in epidemiologic research, and its discontents. BMC Med Res Methodol. 2012;12:21.](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-12-21) | 연속변수를 quantile로 임의 이분화하면 통계적 검정력이 손실되고 cutoff이 표본(코호트)에 종속적이라 다른 코호트로 이식 불가함을 지적 — internal 코호트에서 정의한 quintile 절대값을 external에 그대로 고정 적용해야 하는 근거([[feedback_internal_external_validation_discipline]]과 동일 원칙) |

**적용 방법**: sex-specific, internal(Gangnam) 코호트 기준 하위 20%(quintile) 절대 threshold를
산출해 고정한 뒤, external(Sinchon)에는 그 절대값을 그대로 적용(재계산 금지). TAMA와 상관이
높은 NAMA/LAMA(|r|≈0.8)는 label과 순환관계라 우선순위 낮음 — IMATA/SAT/Total Fat부터 적용
([[feedback_no_circular_label_feature]]).

**적용 파일**: `step0_clinic-only_baseline.py` (logistic regression 확장 시)

### H-보충. 실제 slide15~16(`260807_Results of multiple Features.pptx`)가 쓴 방법: 전체 코호트 Mean±2SD → Mean±1SD (2026-08-04 작성, 2026-08-05 정정)

위 표는 quantile(하위 20%) 방식을 권고하지만, `260807_Results of multiple Features.pptx`
slide15(1차 시도)~slide16(Step4 최종)은 이미 **다른 방식**(진단 대상인 전체 임상 코호트
자체의 성별 mean±SD를 cutoff으로 직접 산출, quantile이 아님)을 적용해뒀다 — 즉 이 표의
권고안과 실제 슬라이드의 구현이 서로 다른 방법론이라는 점에 주의. mean±2SD 산출 방식 자체의
근거:

| 문헌 | 핵심 내용 |
| --- | --- |
| [Derstine BA, Holcombe SA, Ross BE, Wang NC, Su GL, Wang SC. Skeletal muscle cutoff values for sarcopenia diagnosis using T10 to L5 measurements in a healthy US population. Sci Rep. 2018;8:11369.](https://www.nature.com/articles/s41598-018-29825-5) | 건강 신장기증자 후보군(18–40세, n=735)의 성별 mean−2SD를 sarcopenia cutoff으로 정의(EWGSOP 권고 방식) — mean±2SD cutoff 산출법의 표준 원조 |
| [Yoon JK et al. Reference Values for Skeletal Muscle Mass at the Third Lumbar Vertebral Level Measured by Computed Tomography in a Healthy Korean Population. Endocrinol Metab (Seoul). 2021;36(3):672-677.](https://www.e-enm.org/journal/view.php?doi=10.3803%2FEnM.2021.1041) | 한국 건강 간기증자(n=659)에서 mean−2SD와 5th percentile을 병기해 같은 원리를 재현 |
| [Kim EH, Kim KW, Shin Y, et al. Reference Data and T-Scores of Lumbar Skeletal Muscle Area and Its Skeletal Muscle Indices Measured by CT Scan in a Healthy Korean Population. J Gerontol A Biol Sci Med Sci. 2021;76(2):265-271.](https://doi.org/10.1093/gerona/glaa065) | 한국인 11,845명 중 20–44세 young reference군 기준 T-score<−2.0(=mean−2SD와 동일 원리)을 sarcopenia 기준으로 제시 |

**해소된 간극(2026-08-05 정정)**: 이전 버전(작성 당시 slide19 기준)은 세 논문의 reference군이
**나이(20–44/18–40세)** 또는 **건강기증자 심사 통과** 기준으로 정의된 반면, slide15(당시
slide17~19)는 **WHO Asian BMI 정상체중(18.5–23.0)을 reference군 선정 기준으로 쓴 선례는
확인하지 못했다**는 간극을 지적했었다. 2026-08-05 세션에서 slide15의 모집단을 BMI 서브그룹이
아니라 **전체 임상 코호트 자체(남390·여700)**로 바꾸면서 이 간극은 사라졌다 — 다만 이는 세
논문의 "별도 건강 reference군에서 cutoff을 산출한 뒤 진단 대상 코호트에 적용"하는 구조와는
다른, "진단 대상 코호트 자체의 분포에서 mean±2SD를 직접 산출"하는 자기참조적(self-referential)
방식이라는 새로운 방법론적 차이가 남는다(별도 대조군 없음).

**대신 드러난 문제**: 전체 코호트에 2SD를 적용하자 TAMA·NAMA(낮으면 이상 방향)의 양성 인원이
남 4~11명·여 7~17명까지 줄었다(양성률 1.0~2.8%) — "높으면 이상"인 나머지 5종(3.1~5.4%)과도
뚜렷한 비대칭이 나타나는데, BMI 서브그룹을 없애도 폭만 줄었을 뿐 같은 방향의 비대칭이
재현된 것을 보면 원인은 서브그룹의 대표성 문제가 아니라 각 feature 분포의 방향별
왜도(skewness)로 재해석된다. 이 소수 양성 표본 문제가 SD 배수를 2→1로 낮춰 slide16으로
재산정하게 된 실제 계기다(TAMA를 SMI로 정의했을 때 겪은 external n_pos=2 소표본 아티팩트와
같은 계열의 위험, 아래 H-보충2 참고).

### H-보충2. `code/step4_clinic_aec_logistic.py`의 실제 구현: BMI 미필터 + Mean±1SD (2026-08-04 추가, 2026-08-05 정합성 확인 갱신)

git에 커밋된 적 없는 현재의 `code/step4_clinic_aec_logistic.py`는 BMI 필터링 없이 internal
코호트 전체를 성별로만 나눠 mean±SD를 산출한다. 2026-08-05 세션 이전에는 이 코드 구현과
`260807_Results of multiple Features.pptx` slide15~16(당시 slide16~17)의 설명(WHO Asian BMI
정상체중군 reference + mean±2SD)이 서로 다른 모집단을 쓴다는 정합성 문제가 있었으나, 이번
세션에서 slide15의 모집단을 코드와 동일한 "BMI 필터 없는 전체 internal 코호트"로 바꾸면서
이 정합성 문제는 해소됐다 — slide15(mean±2SD, 1차 시도)·slide16(mean±1SD, Step4 최종)·
code/step4_clinic_aec_logistic.py 셋 다 이제 같은 모집단(SD 배수만 다름)을 쓴다.
`260804_Step0-3 Research Progress Review.pptx`([[project_step0_3_progress_review_pptx]])
쪽은 이번 세션에서 손대지 않아 정합성 여부가 미확인 상태로 남아있다(별도로 재확인 필요).

이 실제 구현에서 SD 배수를 2 → 1로 바꾼 이유(TAMA external n_pos=2/498으로 AUC=1.000 소표본
아티팩트 발생, [[project_step4_tama_1sd_cutoff_switch]] 참고)와 별개로, mean±1SD 자체를
cutoff으로 쓰는 것이 방어 가능한지에 대한 근거:

| 문헌 | 핵심 내용 |
| --- | --- |
| [Baumgartner RN, Koehler KM, Gallagher D, et al. Epidemiology of Sarcopenia among the Elderly in New Mexico. Am J Epidemiol. 1998;147(8):755-763.](https://doi.org/10.1093/oxfordjournals.aje.a009520) | Sarcopenia 역학의 원조 정의 논문. Rosetta study 젊은 reference군(18–40세) 기준 SMI(ASM/Height²)가 mean −1SD~−2SD면 "Class I(경도)", <−2SD면 "Class II(중증)"로 severity를 등급화 — mean−1SD를 완화된 등급의 정당한 cutoff으로 쓴 최초·최다인용 선례 |
| WHO Study Group. Assessment of fracture risk and its application to screening for postmenopausal osteoporosis. WHO Technical Report Series 843. Geneva: WHO; 1994. | 골밀도 T-score 체계: 정상=T-score≥−1, osteopenia(저골량, 위험군)=−2.5<T-score<−1, osteoporosis(확진)=T-score≤−2.5. "−1SD=정상 이탈/위험군 분류, −2.5SD(≈−2SD)=확진" 구도가 여러 체성분/골격 지표에서 반복되는 표준 골격 |

**적용된 결론**: Baumgartner의 Class I/II 등급 구조와 WHO T-score 체계 모두 "−1SD는 스크리닝·
위험군 분류, −2SD 이상은 확진/중증"이라는 2단 구조를 쓴다. 이번 분석은 진단이 아니라
clinic4/AEC 예측모델 비교용 이분류 라벨이 목적이므로, EWGSOP의 진단 표준(−2SD)이 아니라
Baumgartner Class I 수준(−1SD)의 위험군 정의를 채택한 것으로 문서화한다 — 단, "진단 cutoff"
이 아니라 "스크리닝/위험군 분류"라는 점은 report 작성 시 명시해야 함.

---

## I. 곡선 구간평균(PAA) — 일반 시계열 이론 근거 (신규)

기존 "A" 요소는 `docs/aec_preprocessing_related_research.md`의 CT AEC/TCM
도메인 특이적 근거(patient-wise 정규화가 왜 필요한지)만 다루고, "왜 곡선을 여러
구간으로 나눠 평균 내는 축약 자체가 통계적으로 정당한지"에 대한 일반 시계열
이론 근거는 없었다. 이번에 slide2(`260807_Results of multiple Features.pptx`)를
Step1~3 방법론 근거로 재구성하면서 보강.

| 문헌 | 핵심 내용 |
| --- | --- |
| [Keogh E, Chakrabarti K, Pazzani M, Mehrotra S. *Dimensionality Reduction for Fast Similarity Search in Large Time Series Databases.* Knowledge and Information Systems. 2001;3(3):263-286.](https://www.cs.ucr.edu/~eamonn/kais_2000.pdf) | PAA(Piecewise Aggregate Approximation)를 정의·명명한 원 논문 — SVD/DFT/DWT 대비 구현이 단순하고 선형시간 인덱싱이 가능함을 제안. N=1이면 전체평균(Step1의 `step1_clinic_aec_mean.py`), N=2면 앞/뒤 절반(Step2의 `step2_clinic_aec_back_half.py`·Step3의 `step3_clinic_aec_shape.py`의 `upper_lower_ratio`), N=2~128로 늘려가는 것이 Step2 구간수 sweep(`step2_clinic_aec_segment_select.py`) 그 자체. **"PAA"라는 용어 자체를 정의하는 근거는 이 논문이 유일** — 2026-08-03 세션에서 arXiv 전문(HTML)을 직접 검색해 아래 Middlehurst 2024가 "PAA"/"Piecewise Aggregate Approximation"이라는 표현을 전혀 쓰지 않는다는 것을 확인했으므로, 이 논문이 여전히 주 인용이어야 함 |
| [Middlehurst M, Schäfer P, Bagnall A. *Bake off redux: a review and experimental evaluation of recent time series classification algorithms.* Data Mining and Knowledge Discovery. 2024;38(4):1958-2031.](https://arxiv.org/abs/2304.13029) | **"PAA"라는 용어는 원문에 등장하지 않음(전문 검색으로 확인)** — 다만 interval-based 분류기(TSF: "mean, variance and slope" over 구간, CIF/DrCIF: Catch22 features)와 dictionary-based 분류기(BOSS/WEASEL의 SAX 기반 discretization, SAX 자체가 PAA를 내부적으로 사용)가 2024년 현재도 표준 벤치마크에 포함돼 있음을 근거로, "구간별 요약통계 압축"이라는 발상 자체는 최신 문헌에서도 여전히 쓰인다는 보조 근거로만 인용(PAA의 정의 근거로는 사용하지 않음) |

**적용 파일**: `step1_clinic_aec_mean.py`, `step2_clinic_aec_back_half.py`, `step2_clinic_aec_segment_select.py`(구간평균 산출부), `step3_clinic_aec_shape.py`(`upper_lower_ratio`)

---

## J. 곡선 형태 Feature(SD·Skewness) — 시계열 통계적 모멘트 (신규)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Barandas M, Folgado D, Fernandes L, et al. *TSFEL: Time Series Feature Extraction Library.* SoftwareX. 2020;11:100456.](https://www.sciencedirect.com/science/article/pii/S2352711020300017) | 시간·통계·주파수 영역 60여 종 시계열 특징추출 방법을 제공하는 라이브러리 논문(2020) — 평균·표준편차·왜도(skewness) 등 통계적 모멘트가 시계열 요약 feature의 표준 항목임을 재확인. `step3_clinic_aec_shape.py`의 `shape_features()`가 산출하는 SD·Skewness가 이 표준 항목에 해당 |
| [Christ M, Braun N, Neuffer J, Kempa-Liehr AW. *Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package).* Neurocomputing. 2018;307:72-77.](https://www.sciencedirect.com/science/article/pii/S0925231218304843) | tsfresh 원 논문(2018) — 63종 특징추출 방법(총 794개 feature) 정리. 위 TSFEL과 같은 계열의 선행 라이브러리로 참고용 병기(2026-08-03 세션에서 slide2의 주 인용을 Barandas 2020으로 교체) |

**적용 파일**: `step3_clinic_aec_shape.py`만 해당 (SD·Skewness를 직접 산출하는 유일한 스크립트)

---

## K. 구간 선택 Filter법(F-검정 기반) — 선택 기준 자체의 근거 (신규)

F 요소(Ambroise & McLachlan 2002)는 "구간을 CV 밖에서 먼저 고르면 안 된다"는
**leakage 방지** 근거이고, 이 K 요소는 "그 선택을 어떤 통계량으로 하는지"
(F-검정 기반 순위화) 자체의 근거로 서로 보완 관계다.

| 문헌 | 핵심 내용 |
| --- | --- |
| [Li J, Cheng K, Wang S, et al. *Feature Selection: A Data Perspective.* ACM Comput Surv. 2017;50(6):94.](https://dl.acm.org/doi/10.1145/3136625) | Similarity-based/information-theoretic/sparse-learning/statistical-based 4개 범주로 변수선택법을 정리한 최신 대표 리뷰 — 단변량 통계량(F-검정 등)으로 변수를 순위화해 상위 k개만 남기는 filter법(statistical-based)의 표준 근거. `step2_clinic_aec_segment_select.py`의 `SelectKBest(f_regression)`이 이 filter법의 직접 구현 |
| [Guyon I, Elisseeff A. *An Introduction to Variable and Feature Selection.* J Mach Learn Res. 2003;3:1157-1182.](https://jmlr.org/papers/special/feature03.html) | Filter/wrapper/embedded 3분류 체계를 정립한 고전 리뷰(2003) — 위 Li 2017의 선행 근거로 참고용 병기(2026-08-03 세션에서 slide2의 주 인용을 Li 2017로 교체) |

**적용 파일**: `step2_clinic_aec_segment_select.py`만 해당 (`SelectKBest`를 쓰는 유일한 스크립트)

---

## 파일별 최종 레퍼런스 매핑

| 파일 | A (AEC 구간평균) | B (clinic4→체성분) | C (K-fold OOF) | D (bootstrap CI) | E (internal/external) | F (CV 내부 selection) | G (잔차진단) | H (logistic cutoff) | I (PAA 일반이론) | J (곡선 형태 모멘트) | K (F-검정 filter) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `step0_aec_clinic_correlation.py` | ✅ | — | — | — | — | — | — | — | — | — | — |
| `step0_clinic-only_baseline.py` | — | ✅ | ✅ | ✅ | ✅ | — | ✅ | 🔜 예정 | — | — | — |
| `step1_clinic_aec_mean.py` | ✅ | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | — | — | ✅(N=1 PAA) | — | — |
| `step2_clinic_aec_back_half.py` | ✅(반구간=Bostani 2015 regional split과 동일 구조) | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | — | — | ✅(N=2 PAA) | — | — |
| `step2_clinic_aec_segment_select.py` | ✅ | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | ✅ | — | — | ✅(N=2~128 PAA sweep) | — | ✅ |
| `step3_clinic_aec_shape.py` | ✅(half-ratio) | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | — | — | ✅(N=2 half-ratio) | ✅ | — |

"A" 열의 상세 근거(Kalra 2004, McCollough 2006, Li 2017, Bostani 2015, AAPM 204,
Li 2022, Ramsay FDA 등)는 `docs/aec_preprocessing_related_research.md`를 그대로
참조한다 — 중복 기재하지 않음. "I" 열은 그와 별개로 구간평균 축약 자체의 일반
시계열 이론(PAA) 근거이며, `docs/260807_Results of multiple Features.pptx`
slide2(Step1~3 방법론 근거)에 반영되어 있다.
