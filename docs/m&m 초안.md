# Introduction

## Automatic Exposure Control

CT image noise is determined by the degree of x-ray attenuation along the beam path through the scanned region, so a fixed tube current, expressed in mA, produces heterogeneous noise whenever body habitus or regional thickness varies across a scan. To compensate, modern multidetector CT scanners employ automatic exposure control (AEC), which adjusts tube current in real time to maintain a user-specified target noise level, using regional x-ray attenuation information derived from a low-dose localizer, or scout, image acquired immediately before the diagnostic acquisition. AEC modulation is categorized as angular, or x-y, modulation, which varies tube current with projection angle; longitudinal modulation along the craniocaudal, or z, axis, which varies tube current from head to feet; or combined x-y-z modulation, which integrates both [1].

Regional attenuation is estimated from this low-dose localizer image. The localizer is a simple projection image obtained by scanning the patient in the anteroposterior or lateral direction at a fixed, low tube current, from which an approximate x-ray attenuation profile along the z-axis can be derived from the pixel-intensity distribution. The scanner combines this attenuation estimate with a user-specified target noise metric, such as noise index, standard deviation, or reference mAs, to precompute a tube-current profile as a function of z-axis position; once the diagnostic acquisition begins, tube current is adjusted in real time according to this precomputed profile. In some vendor implementations, an additional correction is applied using measured attenuation from the initial half-rotation of the scan.

In abdominal CT, tube-current modulation is frequently implemented as combined x-y-z modulation, reflecting not only the change in thickness along the z-axis but also left-right and anteroposterior asymmetry within each cross-section. Tube current is reduced at regions of relatively low attenuation, such as the mid-abdomen, which is predominantly soft tissue, and increased at regions of high attenuation, such as the pelvis, which contains dense bone, maintaining consistent image quality throughout the scan. As a result, the trajectory of recorded tube-current values along the scan direction, the tube-current modulation curve, reflects the attenuation characteristics of the patient's cross-section, and hence body habitus and body composition, at each z-axis position. This principle is consistent with prior clinical validation of z-axis tube-current modulation in the abdomen and pelvis [2] and with dose-management reviews describing how tube current is adjusted to patient size and attenuation [3], supporting the use of the AEC curve as a surrogate marker of body habitus.

The CT scanners included in this study implement combined x-y-z modulation; tube-current values extracted from the DICOM header field X Ray Tube Current therefore do not represent pure longitudinal modulation alone, but rather the output of combined modulation, incorporating both angular modulation reflecting within-slice asymmetry and longitudinal modulation along the craniocaudal axis. Because only a single value is recorded per slice, or rotation, angular variation within a given rotation is not separately resolved and is already integrated into the representative value for that rotation. Accordingly, while the AEC-128 curve in this study varies as a function of craniocaudal position, the value at each point reflects the output of combined x-y-z modulation, incorporating within-slice asymmetry in addition to purely longitudinal information.

This study exploits this tube-current modulation principle to construct AEC-128, a standardized, fixed-length representation of the tube-current curve over a defined anatomical range; the construction procedure is detailed in the Method section. This approach opportunistically extracts body-habitus-related information from tube-current data that is automatically recorded during image acquisition, without requiring separate analysis of the CT image pixel data itself.

Tube voltage, expressed as kVp, or exposure time, that is, gantry rotation time, could in principle be modulated to achieve the same purpose, but tube-current modulation has been adopted as the standard AEC implementation because of the limitations of these alternatives. Changing tube voltage alters the x-ray energy spectrum, that is, beam quality itself, which in turn changes tissue contrast, the degree of contrast enhancement, and beam-hardening artifact patterns, requiring separate recalibration for each examination. Increasing exposure time, or gantry rotation time, prolongs acquisition, increasing the risk of motion blur from respiration and bowel peristalsis, and rotation speed is subject to mechanical constraints that limit arbitrary adjustment. Tube current, by contrast, can be adjusted in an approximately linear fashion to control the number of x-ray photons, and hence dose, without altering the beam's energy spectrum or scan speed, preserving image contrast and temporal-resolution characteristics while compensating solely for differences in cross-sectional thickness. Because all subjects in this study were scanned at a fixed tube voltage of 100 kVp under the inclusion criteria described in Materials, differences in tube-current values across patients can be attributed to differences in body habitus rather than to differences in acquisition protocol.

# Materials

This retrospective cohort study comprised two independent institutional cohorts: an internal cohort from Gangnam Severance Hospital and an external cohort from Sinchon Severance Hospital. Because the two cohorts were drawn from different hospitals and different scanner populations, the model trained on the internal cohort was applied, without retraining, to the external cohort to assess external validity, following established internal/external validation principles for clinical prediction models [4-6]. CT examinations were performed between January 2018 and June 2020 for the internal cohort and throughout 2019 for the external cohort.

Inclusion criteria were as follows:

1. Both clinical data and an abdominal CT examination were available for the same patient.
2. Only CT examinations acquired at a tube voltage of 100 kVp were included; in the internal cohort, patients were further restricted to those scanned on the four most commonly used CT scanner models at that site, namely Siemens Sensation 64, GE Revolution CT, Philips Ingenuity Core 128, and Siemens SOMATOM Definition AS+, whereas in the external cohort no scanner-model restriction was applied except exclusion of the negligibly represented Canon scanners, of which only three were present.
3. Patients younger than 20 years of age were excluded from both cohorts.

After application of these criteria, the final study population consisted of 1,079 patients in the internal cohort and 922 patients in the external cohort; the patient selection process for both cohorts is summarized in Figure 1.

![Figure 1](../outputs/figure_patient_selection_flow/patient_selection_flow.png)

**Figure 1.** Flow diagram of patient selection for the internal and external cohorts.

Age, sex, height, and weight were abstracted from the same clinical database used for patient matching and served as the baseline clinical predictors, collectively termed clinic4. Outcomes were defined as the presence or absence of a documented diagnosis of hypertension, diabetes mellitus, and chronic kidney disease — HTN, DM, and CKD, respectively — in the clinical database. Hypertension was present in 324 of 1,079 internal patients (30.0%) and 424 of 922 external patients (46.0%); diabetes mellitus in 202 internal patients (18.7%) and 285 external patients (30.9%); and chronic kidney disease in 83 internal patients (7.7%) and 168 external patients (18.2%).

Baseline characteristics of the internal and external cohorts are summarized in Table 1. There were no missing values for age, sex, height, weight, BMI, comorbidity, or scanner vendor in either cohort.

**Table 1. Baseline Characteristics of the Patients**

| Characteristic | Internal cohort<br>(n = 1,079) | External cohort<br>(n = 922) | *p*-value |
|---|---|---|---|
| Age, years, mean ± SD | 57.1 ± 12.1 | 59.5 ± 12.7 | <0.001 |
| Age, years, range | 20–91 | 20–90 | |
| Sex, n (%) |  |  | <0.001 |
| — Female | 692 (64.1) | 497 (53.9) |  |
| — Male | 387 (35.9) | 425 (46.1) |  |
| Height, cm, mean ± SD | 162.2 ± 8.2 | 162.5 ± 8.6 | 0.401 |
| Weight, kg, mean ± SD | 61.9 ± 10.6 | 61.8 ± 11.4 | 0.873 |
| BMI, kg/m², mean ± SD | 23.5 ± 3.1 | 23.3 ± 3.4 | 0.383 |
| Hypertension, n (%) | 324 (30.0) | 424 (46.0) | <0.001 |
| Diabetes mellitus, n (%) | 202 (18.7) | 285 (30.9) | <0.001 |
| Chronic kidney disease, n (%) | 83 (7.7) | 168 (18.2) | <0.001 |
| CT scanner vendor, n (%) |  |  | <0.001 |
| — Siemens | 560 (51.9) | 568 (61.6) |  |
| — GE | 317 (29.4) | 219 (23.8) |  |
| — Philips | 202 (18.7) | 135 (14.6) |  |

AEC modulates the tube current (mA) along the craniocaudal z-axis during each abdominal CT acquisition, keeping image noise roughly constant from slice to slice. Because the current needed to achieve this depends on how much the patient attenuates the beam at each position, the resulting tube-current modulation curve carries information about the patient's cross-sectional attenuation, and by extension body habitus, at each slice position. We represented this curve at a fixed length of 128 points per patient, termed AEC-128, and used it as the imaging-derived predictor in this study.

# Methods

## AEC-128 Construction

To construct the fixed-length AEC-128 feature, the axial source series was first identified for each patient, excluding coronal/sagittal reformations, scout images, and dose reports, based on slice count, DICOM Image Type, Series Description, and image orientation. An open-source, deep-learning-based segmentation tool, Total Segmentator [7], was then applied to localize the liver and the bilateral hip bones, with the cranial-most liver slice and caudal-most hip slice defining the upper and lower boundaries of the analysis range; segmentation was used solely for this anatomical landmark localization, while tube-current values themselves were obtained directly from the DICOM header field X Ray Tube Current. Tube-current values within this hip-bone-to-liver range were extracted in craniocaudal order, and because the resulting slice count varied across patients, each sequence was resampled by linear interpolation to a fixed length of 128 points, yielding the AEC-128 curve used as the analytic unit in subsequent modeling.

## Clinical Baseline Model

For each disease outcome, we fit a baseline logistic regression model, termed clinic4, using the four clinical predictors introduced above — age, sex, height, and weight — since these are the clinical variables routinely available at the time of imaging and therefore give a realistic baseline against which any imaging-derived contribution can be judged. Sex, being binary, was entered directly as a 0/1 indicator without scaling; age, height, and weight, being continuous and on different physical scales, were standardized to zero mean and unit variance using a scaler fit on the internal cohort and applied, without refitting, to the external cohort, so that no single variable would dominate the regression by scale alone. We evaluated this baseline by internal AUC under 5-fold stratified cross-validation — reduced to as few as two folds for outcomes whose minority class contained fewer than five patients — and by external frozen AUC for HTN, DM, and CKD; together these numbers served as the reference against which the incremental value of AEC-128 was assessed.

## AEC-Derived Features

Raw AEC-128 curves were used without further normalization, since this variation already enters through clinic4 and an additional cohort-level scaling step risked removing the body-habitus signal that AEC-128 is intended to capture. Derived AEC features, described below, were standardized with their own scaler, fit on the internal cohort and applied frozen to the external cohort, kept separate from the clinical-feature scaler so that features on different physical scales did not dominate the L2-regularized logistic regression.

Three curve-level summary statistics were additionally computed per patient directly from the 128-point curve, following standard practice in time-series feature extraction [8]. The standard deviation, or SD, across all 128 points captured the overall range of tube-current modulation. The skewness of the point distribution captured the asymmetry of the modulation profile along the craniocaudal axis. The ratio of the mean tube current in the cranial half of the curve to that in the caudal half captured the relative attenuation burden of the upper versus lower abdomen.

Component scores from the functional principal component analysis described below, PC1 through PC3, were included as a fourth AEC feature candidate, providing a lower-dimensional representation of curve shape beyond these three summary statistics; during internal cross-validation, PCA was refit on each training fold and the resulting transform applied to the held-out fold, to prevent validation-curve information from leaking into the component estimation.

## Functional Principal Component Analysis

The AEC-128 curve is a 128-dimensional variable per patient, but adjacent points are strongly autocorrelated, making it a functional data object rather than a set of independent measurements. Entering all 128 points as independent variables would incur substantial multicollinearity and overfitting risk; functional principal component analysis, abbreviated FPCA [9], was therefore applied to summarize the curve's shape information in a lower-dimensional form. FPCA extends the principle of principal component analysis, or PCA, to functional data, approximating each curve as a linear combination of a mean curve and a series of orthogonal components that explain the greatest inter-curve variation; each patient's curve is then summarized by its projection onto these components, the component score, or FPC score.

This procedure was carried out as follows. The mean curve μ(z), for z = 1, ..., 128, was first computed across all curves in the internal cohort. Each patient's deviation curve was then obtained by subtracting this mean from the raw curve, dᵢ(z) = xᵢ(z) − μ(z). The 128 × 128 covariance matrix C of the deviation curves was eigen-decomposed via singular value decomposition (SVD) [10], using scikit-learn's PCA implementation for numerical stability; this yields the decomposition C ≈ ΦΛΦᵀ. Here, Φ is the 128×k matrix whose columns are the retained orthonormal eigenvectors, termed eigenfunctions and denoted φₖ, while Λ is the k×k diagonal matrix of the corresponding eigenvalues, denoted λₖ and ranked by the proportion of variance explained. The value of k, the number of retained components, is determined below via an elbow-based criterion, and truncating the product to the top k components gives the rank-k approximation of C used below to assess reconstruction quality. Finally, each patient's deviation curve was projected onto each component to obtain the component score, scoreᵢ,ₖ = Σ_z dᵢ(z)·φₖ(z).

![Figure 2](../outputs/figure1_fpca_computation/figure1_combined.png)

**Figure 2.** Computation of FPCA on the internal-cohort AEC-128 curves. (A) A random sample of patient curves, shown in gray, and the resulting mean curve μ(z), shown in black. (B) Deviation curve dᵢ(z) = xᵢ(z) − μ(z) for a representative patient, projected onto φ₁ to yield the component score scoreᵢ,₁ = Σᵩ dᵢ(z)φ₁(z) = 1,318. (C) The eigenvalue scree plot used for elbow-based component selection, with k = 3. (D, E) The 128×128 covariance matrix, eigen-decomposed via SVD to obtain (F) the resulting eigenfunctions φ₁–φ₃, shown as μ(z) ± √λₖ·φₖ(z).

When fit to the internal cohort, PC1 alone accounted for 84.2% of the total variance, with PC1–2 explaining 94.7% cumulatively and PC1–3 explaining 97.3%. We determined the number of retained components using the Kneedle algorithm [11], applied to the explained-variance scree plot: after normalizing the component index and explained-variance ratio to a 0–1 scale, the elbow was defined as the point of maximal perpendicular distance from the line connecting the first and last components. This elbow coincided with PC3, fixing k = 3 as introduced above, so the first three component scores — PC1 through PC3 — were retained as the representative AEC curve descriptors. To illustrate reconstruction accuracy at the individual level, we reconstructed the curve of one patient with large PC1–PC3 scores and reconstruction fit in the top quartile, with component values of PC1 +1,318, PC2 +1,175, and PC3 −247, from these three scores alone; the resulting R² between the reconstructed and original curve was 0.994, indicating that the 128-point curve can be closely approximated by three component scores.

We estimated the components, or eigenfunctions, from the internal cohort only. External-cohort curves were then projected onto these same fixed components — without any refitting — to obtain their component scores, keeping the two cohorts on a common basis for comparison. This captures morphological variation in curve shape that individual summary statistics such as the mean, SD, or skewness do not capture on their own; it enters the predictive models as an input variable, consistent with prior applications of FPCA to CT-derived curves for clinical outcome prediction [12].

## AEC Feature Selection and Evaluation

Five AEC feature candidates were compared as additions to clinic4: SD alone, skewness alone, the upper/lower ratio alone, FPCA scores alone, and all shape features combined with FPCA scores. For each disease, all five candidates were evaluated by internal AUC under the same 5-fold stratified cross-validation scheme, using logistic regression [13], and the candidate with the highest mean internal AUC was selected independently for that disease, since the shape of the AEC-clinical association was not assumed to be shared across HTN, DM, and CKD. External data played no role in this selection: model comparison relied on internal cross-validation only, and the external cohort was reserved for a single, frozen final evaluation of the selected model, consistent with the internal/external validation roles described above.

For each disease, we compared the resulting model — combining clinic4 with the best-performing AEC feature set — against clinic4 by internal out-of-fold AUC and by external AUC from the internal-trained model applied once, without retraining, following the two-cohort clinic-model-versus-clinic-plus-curve-model design used in prior curve-based disease prediction studies [14]. We tested the AUC difference between the two models with the paired DeLong test [15], appropriate because both models were evaluated on the same patients. We chose a classification threshold by maximizing Youden's J statistic [16] on internal out-of-fold predictions and applied it, fixed, to compute sensitivity, specificity, and accuracy in both cohorts. To assess whether the AEC contribution was consistent across CT manufacturers, we additionally recomputed AUC within each scanner subgroup containing at least 30 patients with both outcome classes represented, using the same fixed threshold. Throughout, we considered a two-sided p-value below 0.05 statistically significant, and we performed all analyses in Python 3.13 using scikit-learn 1.7, pandas 2.3, NumPy 2.3, and SciPy 1.16.

# References

1. McCollough CH, Bruesewitz MR, Kofler JM Jr. Techniques and applications of automatic tube current modulation for CT. *Radiology*. 2006;240(3):611-622. doi:10.1148/radiol.2333031150
2. Kalra MK, Maher MM, Toth TL, et al. Comparison of Z-axis automatic tube current modulation technique with fixed tube current CT scanning of abdomen and pelvis. *Radiology*. 2004;232:347-353. doi:10.1148/radiol.2322031304
3. McCollough CH, Bruesewitz MR, Kofler JM Jr. CT dose reduction and dose management tools: overview of available options. *RadioGraphics*. 2006;26(2):503-512. doi:10.1148/rg.262055138
4. Steyerberg EW. *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating.* 2nd ed. Springer; 2019.
5. Moons KGM, Altman DG, Reitsma JB, et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): explanation and elaboration. *Ann Intern Med*. 2015;162(1):W1-W73. doi:10.7326/M14-0698
6. Ramspek CL, Jager KJ, Dekker FW, Zoccali C, van Diepen M. External validation of prognostic models: what, why, how, when and where? *Clin Kidney J*. 2021;14(1):49-58. doi:10.1093/ckj/sfaa188
7. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. *Radiol Artif Intell*. 2023;5(5):e230024. doi:10.1148/ryai.230024
8. Barandas M, Folgado D, Fernandes L, et al. TSFEL: time series feature extraction library. *SoftwareX*. 2020;11:100456. doi:10.1016/j.softx.2020.100456
9. Ramsay JO, Silverman BW. *Functional Data Analysis.* 2nd ed. Springer; 2005.
10. Eckart C, Young G. The approximation of one matrix by another of lower rank. *Psychometrika*. 1936;1(3):211-218. doi:10.1007/BF02288367
11. Satopaa V, Albrecht J, Irwin D, Raghavan B. Finding a "Kneedle" in a haystack: detecting knee points in system behavior. In: *2011 31st International Conference on Distributed Computing Systems Workshops*. IEEE; 2011:166-171. doi:10.1109/ICDCSW.2011.20
12. Shalmon T, Salazar P, Horie M, et al. Predefined and data driven CT densitometric features predict critical illness and hospital length of stay in COVID-19 patients. *Sci Rep*. 2022;12(1):8143. doi:10.1038/s41598-022-12311-4
13. Kohavi R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: *Proceedings of the 14th International Joint Conference on Artificial Intelligence*. 1995:1137-1143.
14. Jeon ET, Park H, Lee JK, et al. Deep learning-based COPD exacerbation prediction using flow-volume and volume-time curve imaging: retrospective cohort study. *J Med Internet Res*. 2025;27:e69785. doi:10.2196/69785
15. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*. 1988;44(3):837-845. doi:10.2307/2531595
16. Youden WJ. Index for rating diagnostic tests. *Cancer*. 1950;3(1):32-35. doi:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3

---

## Reference Citation Map (작업용 메모 — 원고 본문 아님)

각 참고문헌이 본문 어디서, 어떤 근거로 인용되었는지 정리한 작업용 표. 번호는 위 References의 원 번호(1–16, Introduction 포함) 기준.

| # | 문헌 | 인용 위치 | 용도 |
|---|---|---|---|
| 1 | McCollough 2006, Techniques and Applications of Automatic Tube Current Modulation for CT | Introduction – Automatic Exposure Control, 1번째 단락 끝 `[1]` | AEC 변조를 각(angular)/종축(longitudinal)/복합(x-y-z) 세 방식으로 분류하는 정의 자체의 출처 |
| 2 | Kalra 2004, Z-axis automatic tube current modulation 비교 | Introduction, 3번째 단락 `[2]` | 복부·골반 z축 관전류 변조에 대한 선행 임상 검증 — AEC 곡선이 체형을 반영한다는 주장의 실증적 근거 |
| 3 | McCollough 2006, CT dose reduction and dose management tools | Introduction, 3번째 단락 `[3]` | 관전류가 환자 체형·감쇠에 맞춰 조정되는 방식을 설명하는 선량관리 리뷰 — 위 [2]와 함께 "AEC 곡선 = 체형 surrogate" 주장을 뒷받침 |
| 4 | Steyerberg, *Clinical Prediction Models* | Materials, 1번째 단락 `[4-6]` | internal cohort로 학습한 모델을 재학습 없이 external cohort에 적용하는 internal/external validation 설계의 원칙적 근거 |
| 5 | Moons et al., TRIPOD | Materials, 1번째 단락 `[4-6]` (4와 동일 문장) | 예측모델 보고 가이드라인(TRIPOD) — validation 설계 정당화에 함께 인용 |
| 6 | Ramspek et al., External validation of prognostic models | Materials, 1번째 단락 `[4-6]` (4와 동일 문장) | external validation의 "무엇을·왜·어떻게·언제·어디서" 원칙 — validation 설계 정당화에 함께 인용 |
| 7 | Wasserthal et al., TotalSegmentator | Method – AEC-128 Construction `[7]` | 간·양측 골반뼈 세그멘테이션에 사용한 오픈소스 딥러닝 도구 자체의 출처 |
| 8 | Barandas et al., TSFEL | Method – AEC-Derived Features `[8]` | SD·skewness·상하위 비율 등 곡선 단위 요약통계 추출이 "표준적인 시계열 요약특징 관행"을 따른다는 근거 |
| 9 | Ramsay & Silverman, *Functional Data Analysis* | Method – Functional Principal Component Analysis, 1번째 단락 `[9]` | FPCA 기법 자체의 원 출처(교과서) |
| 10 | Eckart & Young, The approximation of one matrix by another of lower rank | Method – FPCA, 2번째 단락 `[10]` | 공분산행렬의 SVD 분해(C ≈ ΦΛΦᵀ)를 상위 3개 성분으로 절단한 것이 최적 rank-3 근사임을 보장하는 정리(Eckart–Young theorem)의 원 출처 |
| 11 | Satopaa et al., Kneedle 알고리즘 | Method – FPCA, PCA 결과 단락 `[11]` | PC 성분 수(k=3)를 scree plot elbow로 결정할 때 사용한 Kneedle 알고리즘의 출처 |
| 12 | Shalmon et al., COVID-19 CT densitometric features | Method – FPCA, 마지막 단락 `[12]` | CT 유래 곡선에 FPCA를 적용해 임상 아웃컴을 예측한 선행 사례 — 본 연구 접근법의 선례 |
| 13 | Kohavi, Cross-Validation and Bootstrap | Method – AEC Feature Selection and Evaluation, 1번째 단락 `[13]` | 5개 AEC 후보 특징을 stratified cross-validated internal AUC로 비교하는 검증 방법론의 근거 |
| 14 | Jeon et al., COPD 예측(2병원 코호트 설계) | Method – AEC Feature Selection and Evaluation, 2번째 단락 `[14]` | clinic-model vs clinic+curve-model을 internal/external 2개 코호트로 비교하는 설계를 그대로 차용한 선행 연구 |
| 15 | DeLong et al., ROC AUC 비교 | Method, 2번째 단락 `[15]` | clinic4 vs clinic4+AEC 모델의 AUC 차이를 검정한 paired DeLong test의 원 출처 |
| 16 | Youden, Index for rating diagnostic tests | Method, 2번째 단락 `[16]` | internal out-of-fold 예측값에서 분류 임계값을 정할 때 사용한 Youden's J statistic의 원 출처 |

**패턴 요약**
- 1–3: Introduction — AEC 원리·분류 정의 및 "AEC 곡선=체형 surrogate" 주장의 이론·실증적 근거
- 4–6: Materials — internal/external validation 설계 정당화(한 문장에 묶어 인용)
- 7–12: Method(AEC-128/FPCA 파이프라인) — 세그멘테이션 도구 → 요약통계 관행 → FPCA 기법 → SVD 저랭크근사 정리 → elbow 알고리즘 → 선행 적용 사례, 파이프라인 순서와 일치
- 13–16: Method(모델 평가·검정) — 교차검증 → 2코호트 설계 → DeLong test → Youden's J, 평가 절차 순서와 일치

영문 Materials/Method-only 버전(`m&m 초안_영문.docx`)에는 Introduction 전용인 1–3번이 빠지고 나머지 4–16번이 1–13번으로 재번호화되어 들어가 있다.
