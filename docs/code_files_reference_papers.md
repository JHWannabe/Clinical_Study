# code/*.py 방법론별 레퍼런스 분류 (260813 재편 기준)

> **개정 이력**: 이 문서는 과거 6개 스크립트(`step0_aec_clinic_correlation.py`,
> `step0_clinic-only_baseline.py`, `step1_clinic_aec_mean.py`,
> `step2_clinic_aec_back_half.py`, `step2_clinic_aec_segment_select.py`,
> `step3_clinic_aec_shape.py`) 기준으로 작성됐었다. 이 6개는 [[project_260810_step_renumbering]]
> → [[project_260813_step_renumbering_v2]] 재편으로 전부 삭제/재구성되어 `code/0807/`에만
> 아카이브로 남아있고, 현재 `code/` 최상위 파이프라인은 다음 6개다:
> `step0_output_feature_correlation.py`, `step1_aec_fpca.py`, `step2_clinic_aec_ratio.py`,
> `step3_clinic_aec_logistic.py`, `step4_aec_diagnostics.py`, `step6_aec_deep_learning.py`
> (step5는 의도적 결번). 이번 개정은 그 현재 파이프라인 기준으로 요소를 재분류한 것이다.
>
> AEC 신호 자체(raw/patient-wise 정규화)에 대한 근거는 여전히
> `docs/aec_preprocessing_related_research.md`가 담당하며 중복 기재하지 않는다.

## 방법론 요소 목록

| 코드 | 방법론 요소 | 현재 사용 파일 | 상태 |
| --- | --- | --- | --- |
| B | clinic4(성별/나이/키/몸무게)만으로 CT 체성분 예측 | step1~4 baseline 비교대상 | 유지 |
| C | K-fold CV + out-of-fold(OOF) 예측 | step1, step2, step3, step4, step6 | 유지 |
| D | R²/AUC 등의 bootstrap 신뢰구간 | step2, step3, step4, step6 | 유지 |
| E | internal(CV) 학습 → external 동결 모델 1회 검증 프로토콜 | step1~4, step6 전부 | 유지 |
| H | 확립된 cutoff 없는 연속형 체성분의 logistic 이분화(mean±1SD) | step3 | 유지 |
| J | 곡선의 SD·Skewness 등 통계적 모멘트를 형태 feature로 사용 | step2, step3, step4 | 유지 |
| L | **FPCA**(Functional PCA)로 곡선을 소수 score로 압축 | step1(직접), step2~4(형태후보 중 하나로 결합) | **신규** |
| M | **DeLong paired AUC test**(같은 환자집합 내 두 모델 AUC 비교) | step3 | **신규, 레퍼런스 미비** |
| N | **Benjamini-Hochberg FDR 보정**(9개 feature 동시검정) | step3 | **신규, 레퍼런스 미비** |
| O | 1D CNN / Transformer 멀티태스크 딥러닝 | step6 | **신규, 레퍼런스 미비** |
| ~~A~~ | AEC-128을 구간평균(raw)으로 축약해 predictor로 사용 | 없음(0807 전용) | **폐기** — step1이 FPCA로 대체 |
| ~~F~~ | CV fold 내부에서만 feature selection(SelectKBest) | 없음(0807 전용) | **폐기** — 현재 selection 코드 없음 |
| ~~G~~ | 잔차 진단(Q-Q, Shapiro-Wilk, Scale-Location) | 없음(0807 전용) | **폐기** — 현재 어느 파일에도 없음 |
| ~~I~~ | 곡선 구간평균(PAA) 일반 시계열 이론 | 없음(0807 전용) | **폐기** — L(FPCA)로 대체 |
| ~~K~~ | F-검정 기반 filter법(SelectKBest) | 없음(0807 전용) | **폐기** — F와 함께 폐기 |

옛 A/F/G/I/K의 원 문헌·상세 내용은 git 이력(이 파일의 이전 버전) 또는 `code/0807/`
스크립트 자체에서 여전히 확인 가능하며, 필요시 그쪽을 참조한다.

---

## B. clinic4 → CT 체성분 예측

| 문헌 | 핵심 내용 |
| --- | --- |
| [Cao Y et al. *Development of Formulas for Calculating L3 Skeletal Muscle Mass Index and Visceral Fat Area Based on Anthropometric Parameters.* Front Nutr. 2022;9:910771.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249379/) | 성별/나이/키/몸무게만으로 L3 SMI·VFA를 예측하는 선형회귀식 직접 제시(344명 훈련/134명 검증, SMI adj R²=0.597, VFA adj R²=0.581) — clinic4 baseline이 어느 정도 R²를 내는 게 타당한지 비교 기준 |
| [*Computed tomography-based muscle and fat composition in a Dutch population: a cross-sectional study.* Insights Imaging. 2025;16:XX.](https://link.springer.com/article/10.1186/s13244-025-02114-2) | CT 기반 근육/지방 조성의 성별·연령별 정상 참고범위 — residual 해석 참고 |

**적용 파일**: step1~4 전부(clinic4 baseline으로 공유)

---

## C. K-fold Cross-Validation + Out-of-Fold 예측

| 문헌 | 핵심 내용 |
| --- | --- |
| [Kohavi R. *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* Proc. 14th IJCAI. 1995:1137-1143.](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf) | k-fold CV(특히 10-fold)가 모델 선택/정확도 추정에서 우수함을 실증. `KFold(n_splits=5)` + OOF 산출의 방법론적 근거 |

**적용 파일**: step1, step2, step3, step4, step6

---

## D. Bootstrap 신뢰구간

| 문헌 | 핵심 내용 |
| --- | --- |
| [Efron B. *Estimating the Error Rate of a Prediction Rule: Improvement on Cross-Validation.* J Am Stat Assoc. 1983;78(382):316-331.](https://www.jstor.org/stable/2288636) | Bootstrap 기반 예측오차 추정의 원조 |
| [Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall/CRC, 1993.](https://doi.org/10.1201/9780429246593) | Percentile bootstrap CI의 표준 교과서 근거 |

**적용 파일**: step2, step3, step4, step6

---

## E. Internal(CV)-External(동결 모델 1회 검증) 프로토콜

| 문헌 | 핵심 내용 |
| --- | --- |
| [Steyerberg EW. *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating.* 2nd ed. Springer, 2019.](https://link.springer.com/book/10.1007/978-3-030-16399-0) | Internal/external validation 구분 표준 프레임워크 |
| [Moons KGM et al. *TRIPOD: Explanation and Elaboration.* Ann Intern Med. 2015;162(1):W1-W73.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772854/) | internal/temporal/geographic(external) validation 구분 보고 요구 — Gangnam/Sinchon 2-코호트 구조가 geographic external validation |
| [Ramspek CL et al. *External validation of prognostic models: what, why, how, when and where?* Clin Kidney J. 2021;14(1):49-58.](https://pubmed.ncbi.nlm.nih.gov/33564405/) | External validation = "재적합 없이 새 코호트에 그대로 적용" — `model.fit(internal); model.predict(external)` 패턴과 대응 |

**적용 파일**: step1~4, step6 전부. [[feedback_internal_external_validation_discipline]]의 학술적 근거

---

## H. 확립된 cutoff 없는 연속형 체성분의 logistic 이분화 (mean±1SD)

| 문헌 | 핵심 내용 |
| --- | --- |
| [Baumgartner RN, Koehler KM, Gallagher D, et al. *Epidemiology of Sarcopenia among the Elderly in New Mexico.* Am J Epidemiol. 1998;147(8):755-763.](https://doi.org/10.1093/oxfordjournals.aje.a009520) | Rosetta study reference군 기준 SMI mean −1SD~−2SD="Class I(경도)", <−2SD="Class II(중증)" — mean−1SD를 완화된 등급의 정당한 cutoff으로 쓴 최초·최다인용 선례 |
| WHO Study Group. *Assessment of fracture risk...* WHO Technical Report Series 843. Geneva: WHO; 1994. | T-score 체계: 정상≥−1, 위험군 −2.5~−1, 확진≤−2.5 — "−1SD=스크리닝/위험군, −2SD 이상=확진"의 표준 골격 |

**적용된 결론**: 진단이 아니라 clinic4/AEC 예측모델 비교용 이분류 라벨이 목적이므로
Baumgartner Class I 수준(−1SD)의 위험군 정의를 채택([[project_step4_tama_1sd_cutoff_switch]]).
**적용 파일**: step3

---

## J. 곡선 형태 Feature(SD·Skewness) — 시계열 통계적 모멘트

| 문헌 | 핵심 내용 |
| --- | --- |
| [Barandas M, Folgado D, Fernandes L, et al. *TSFEL: Time Series Feature Extraction Library.* SoftwareX. 2020;11:100456.](https://www.sciencedirect.com/science/article/pii/S2352711020300017) | 평균·SD·Skewness 등이 시계열 요약 feature의 표준 항목임을 재확인 |
| [Christ M, Braun N, Neuffer J, Kempa-Liehr AW. *tsfresh.* Neurocomputing. 2018;307:72-77.](https://www.sciencedirect.com/science/article/pii/S0925231218304843) | tsfresh 원 논문 — 같은 계열 선행 라이브러리, 참고용 병기 |

**적용 파일**: step2, step3, step4(SD·Skewness를 AEC 형태후보로 산출·재사용)

---

## L. FPCA(Functional PCA) — 신규

| 문헌 | 핵심 내용 |
| --- | --- |
| [Ramsay JO, Silverman BW. *Functional Data Analysis.* 2nd ed. Springer, 2005.](https://link.springer.com/book/10.1007/b98888) | FPCA 표준 교과서 — 곡선을 평균함수+고유함수(φ1, φ2...) 가중합으로 분해하는 방법론 원전. `step1_aec_fpca.py`가 `sklearn.decomposition.PCA`로 하는 이산근사(128 등간격 포인트)가 정확히 이 방법 |
| [Ramsay JO. *Curve registration.* J R Stat Soc Series B. 1998;60(2):351-363.](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00129) | 곡선을 포인트별이 아닌 전체 단위로 다루는 프레임워크([[feedback_aec_curve_wholistic]])와 정합 |

**적용 파일**: step1(직접 FPCA 컴포넌트 수 탐색), step2~4(AEC 형태후보 5종 중 `aec_fpca`로 결합 사용).
`docs/aec_preprocessing_related_research.md` D절에 이미 있던 인용을 여기 방법론 관점에서 재인용.

---

## M. DeLong Paired AUC Test — 신규, 레퍼런스 미비

`step3_clinic_aec_logistic.py`가 clinic4 vs clinic4_aec_best의 AUC 차이를 같은 환자집합에 대해
검정하는 `delong_paired_auc_test()`를 쓴다(`docs/step3_delong_significance_analysis.md` 참고).
**아직 정식 문헌을 조사하지 않았다** — 다음 세션에서 아래를 확인해 채울 것:

- DeLong ER, DeLong DM, Clarke-Pearson DL. *Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach.* Biometrics. 1988;44(3):837-845. (원 알고리즘, 정확한 서지정보 재확인 필요)
- Sun X, Xu W. *Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves.* IEEE Signal Process Lett. 2014;21(11):1389-1393. (코드 주석에 언급된 fast implementation, 재확인 필요)

**적용 파일**: step3

---

## N. Benjamini-Hochberg FDR 보정 — 신규, 레퍼런스 미비

step3이 9개 feature 동시검정에 대해 BH-FDR(α=0.05)을 적용한다. **아직 정식 인용 미기재**:

- Benjamini Y, Hochberg Y. *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* J R Stat Soc Series B. 1995;57(1):289-300. (원 논문, 정확한 서지정보 재확인 필요)

**적용 파일**: step3

---

## O. 1D CNN / Transformer 멀티태스크 딥러닝 — 신규, 레퍼런스 미비

`step6_aec_deep_learning.py`는 raw AEC-128+clinic4를 입력으로 7개 체성분을 동시에 예측하는
회귀+분류 멀티태스크 CNN/Transformer를 internal 5-fold OOF → external 동결 검증한다(이홍선 교수 요청).
**아직 정식 문헌 조사가 안 된 세 갈래**:

1. FPCA→DNN 비교 근거는 이미 있음 — [*Deep Learning for Functional Data Analysis with Adaptive Basis Layers.* arXiv:2106.10414](https://arxiv.org/pdf/2106.10414), [Wang et al. *Functional data analysis using deep neural networks.* WIREs Comput Stat. 2024.](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.70001) (`docs/aec_preprocessing_related_research.md`)
2. CNN/Transformer 아키텍처 자체(1D 시계열 분류/회귀 벤치마크)는 `project_reference_paper_screening` 메모에 "1D-CNN 3(Foumani TSER 서베이)", "Transformer 1(ConvTran, per-class n<50 급락 — 표본 리스크)"으로만 메모돼 있고 정식 서지정보가 없음 — 원문 재확인 필요
3. 멀티태스크(shared trunk로 여러 feature 동시 예측, "n~1000대 소표본 표본효율" 근거)에 대한 문헌은 전혀 조사되지 않음 — Caruana 1997 *Multitask Learning* 등 고전부터 확인 필요

**적용 파일**: step6

---

## 파일별 최종 레퍼런스 매핑

| 파일 | B | C | D | E | H | J | L | M | N | O |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `step0_output_feature_correlation.py` | — | — | — | — | — | — | — | — | — | — |
| `step1_aec_fpca.py` | ✅(baseline 비교대상) | ✅ | — | ✅ | — | — | ✅(직접) | — | — | — |
| `step2_clinic_aec_ratio.py` | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | ✅ | ✅(형태후보) | — | — | — |
| `step3_clinic_aec_logistic.py` | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(형태후보) | ✅ | ✅ | — |
| `step4_aec_diagnostics.py` | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | ✅ | ✅(형태후보) | — | — | — |
| `step6_aec_deep_learning.py` | ✅(baseline 비교대상) | ✅ | ✅ | ✅ | — | — | — | — | — | ✅ |

`step0_output_feature_correlation.py`는 EDA 상관분석뿐이라 이 표의 방법론 요소를 쓰지 않음
— 필요한 근거는 AEC 신호 자체를 다루는 `docs/aec_preprocessing_related_research.md`.
