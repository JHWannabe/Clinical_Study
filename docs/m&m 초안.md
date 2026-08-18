# Introduction

## Automatic Exposure Control

CT image noise is determined by the degree of x-ray attenuation along the beam path through the scanned region, so a fixed tube current, expressed in mA, produces heterogeneous noise whenever body habitus or regional thickness varies across a scan. To compensate, modern multidetector CT scanners employ automatic exposure control (AEC), which adjusts tube current in real time to maintain a user-specified target noise level, using regional x-ray attenuation information derived from a low-dose localizer, or scout, image acquired immediately before the diagnostic acquisition. AEC modulation is categorized as angular, or x-y, modulation, which varies tube current with projection angle; longitudinal modulation along the craniocaudal, or z, axis, which varies tube current from head to feet; or combined x-y-z modulation, which integrates both [1].

Regional attenuation is estimated from this low-dose localizer image. The localizer is a simple projection image obtained by scanning the patient in the anteroposterior or lateral direction at a fixed, low tube current, from which an approximate x-ray attenuation profile along the z-axis can be derived from the pixel-intensity distribution. The scanner combines this attenuation estimate with a user-specified target noise metric, such as noise index, standard deviation, or reference mAs, to precompute a tube-current profile as a function of z-axis position; once the diagnostic acquisition begins, tube current is adjusted in real time according to this precomputed profile. In some vendor implementations, an additional correction is applied using measured attenuation from the initial half-rotation of the scan.

In abdominal CT, tube-current modulation is frequently implemented as combined x-y-z modulation, reflecting not only the change in thickness along the z-axis but also left-right and anteroposterior asymmetry within each cross-section. Tube current is reduced at regions of relatively low attenuation, such as the mid-abdomen, which is predominantly soft tissue, and increased at regions of high attenuation, such as the pelvis, which contains dense bone, maintaining consistent image quality throughout the scan. As a result, the trajectory of recorded tube-current values along the scan direction, the tube-current modulation curve, reflects the attenuation characteristics of the patient's cross-section, and hence body habitus and body composition, at each z-axis position. This principle is consistent with prior clinical validation of z-axis tube-current modulation in the abdomen and pelvis [2] and with dose-management reviews describing how tube current is adjusted to patient size and attenuation [3], supporting the use of the AEC curve as a surrogate marker of body habitus.

The CT scanners included in this study implement combined x-y-z modulation; tube-current values extracted from the DICOM header field XRayTubeCurrent therefore do not represent pure longitudinal modulation alone, but rather the output of combined modulation, incorporating both angular modulation reflecting within-slice asymmetry and longitudinal modulation along the craniocaudal axis. Because only a single value is recorded per slice, or rotation, angular variation within a given rotation is not separately resolved and is already integrated into the representative value for that rotation. Accordingly, while the AEC-128 curve in this study varies as a function of craniocaudal position, the value at each point reflects the output of combined x-y-z modulation, incorporating within-slice asymmetry in addition to purely longitudinal information.

This study exploits this tube-current modulation principle to construct AEC-128, a standardized, fixed-length representation of the tube-current curve over a defined anatomical range; the construction procedure is detailed in the Method section. This approach opportunistically extracts body-habitus-related information from tube-current data that is automatically recorded during image acquisition, without requiring separate analysis of the CT image pixel data itself.

Tube voltage, expressed as kVp, or exposure time, that is, gantry rotation time, could in principle be modulated to achieve the same purpose, but tube-current modulation has been adopted as the standard AEC implementation because of the limitations of these alternatives. Changing tube voltage alters the x-ray energy spectrum, that is, beam quality itself, which in turn changes tissue contrast, the degree of contrast enhancement, and beam-hardening artifact patterns, requiring separate recalibration for each examination. Increasing exposure time, or gantry rotation time, prolongs acquisition, increasing the risk of motion blur from respiration and bowel peristalsis, and rotation speed is subject to mechanical constraints that limit arbitrary adjustment. Tube current, by contrast, can be adjusted in an approximately linear fashion to control the number of x-ray photons, and hence dose, without altering the beam's energy spectrum or scan speed, preserving image contrast and temporal-resolution characteristics while compensating solely for differences in cross-sectional thickness. Because all subjects in this study were scanned at a fixed tube voltage of 100 kVp under the inclusion criteria described in Materials, differences in tube-current values across patients can be attributed to differences in body habitus rather than to differences in acquisition protocol.

# Materials

This retrospective cohort study comprised two independent institutional cohorts: an internal cohort from Gangnam Severance Hospital and an external cohort from Sinchon Severance Hospital. Because the two cohorts were drawn from different hospitals and different scanner populations, the model trained on the internal cohort was applied, without retraining, to the external cohort to assess external validity, following established internal/external validation principles for clinical prediction models [4-6]. CT examinations for both cohorts were performed between January 2018 and June 2020.

Inclusion criteria were as follows:

1. Both clinical data and an abdominal CT examination were available for the same patient.
2. Only CT examinations acquired at a tube voltage of 100 kVp were included; in the internal cohort, patients were further restricted to those scanned on the four most commonly used CT scanner models at that site, namely Siemens Sensation 64, GE Revolution CT, Philips Ingenuity Core 128, and Siemens SOMATOM Definition AS+, whereas in the external cohort no scanner-model restriction was applied except exclusion of the negligibly represented Canon scanners, of which only three were present.
3. Patients younger than 20 years of age were excluded from both cohorts.

After application of these criteria, the final study population consisted of 1,079 patients in the internal cohort and 922 patients in the external cohort.

Age, sex, height, and weight were abstracted from the same clinical database used for patient matching and served as the baseline clinical predictors, collectively termed clinic4. Outcomes were defined as the presence or absence of a documented diagnosis of hypertension, diabetes mellitus, and chronic kidney disease — HTN, DM, and CKD, respectively — in the clinical database, rather than by dichotomizing a continuous laboratory measurement. Hypertension was present in 324 of 1,079 internal patients (30.0%) and 424 of 922 external patients (46.0%); diabetes mellitus in 202 internal patients (18.7%) and 285 external patients (30.9%); and chronic kidney disease in 83 internal patients (7.7%) and 168 external patients (18.2%).

Baseline characteristics of the internal and external cohorts are summarized in Table 1. There were no missing values for age, sex, height, weight, BMI, comorbidity, or scanner vendor in either cohort.

**Table 1. Baseline Characteristics of the Patients**

| Characteristic | Internal cohort<br>(n = 1,079) | External cohort<br>(n = 922) | *p*-value |
|---|---|---|---|
| Age, years, mean ± SD | 57.1 ± 12.1 | 59.5 ± 12.7 | <0.001 |
| Age, years, range | 20–91 | 20–90 | |
| Sex, female, n (%) | 692 (64.1) | 497 (53.9) | <0.001 |
| Sex, male, n (%) | 387 (35.9) | 425 (46.1) | <0.001 |
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

SD, standard deviation; BMI, body mass index. Continuous variables were compared with Welch's *t*-test; categorical variables were compared with the chi-square test.

During each abdominal CT acquisition, AEC modulates tube current in mA along the craniocaudal z-axis to maintain a constant image noise level; the resulting tube-current modulation curve reflects the patient's cross-sectional attenuation, and hence body habitus, at each slice position. In this study, a fixed-length, 128-point representation of this curve, termed AEC-128, was used as the imaging-derived predictor.

# Method

## AEC-128 Construction

To construct the fixed-length AEC-128 feature, the axial source series was first identified for each patient, excluding coronal/sagittal reformations, scout images, and dose reports, based on slice count, DICOM ImageType, SeriesDescription, and image orientation. An open-source, deep-learning-based segmentation tool, TotalSegmentator [7], was then applied to localize the liver and the bilateral hip bones, with the cranial-most liver slice and caudal-most hip slice defining the upper and lower boundaries of the analysis range; segmentation was used solely for this anatomical landmark localization, while tube-current values themselves were obtained directly from the DICOM header field XRayTubeCurrent. Tube-current values within this pubis-to-liver range were extracted in craniocaudal order, and because the resulting slice count varied across patients from 101 to 238 slices, each sequence was resampled by linear interpolation to a fixed length of 128 points, yielding the AEC-128 curve used as the analytic unit in subsequent modeling.

## Clinical Baseline Model

For each disease outcome, a baseline logistic regression model, termed clinic4, was built from the four clinical predictors described above: age, sex, height, and weight. Sex, a binary variable, was entered directly as a 0/1 indicator without scaling, while age, height, and weight were standardized to zero mean and unit variance using a scaler fit on the internal cohort and applied, without refitting, to the external cohort. This baseline was evaluated by internal cross-validated and external frozen AUC for HTN, DM, and CKD, and served as the reference against which the incremental value of AEC-128 was measured.

## AEC-Derived Features

Raw AEC-128 curves were used without further normalization, since sex, age, height, and weight variation already enters through the clinical predictors and an additional cohort-level scaling step risked removing the body-habitus signal that AEC-128 is intended to capture. Derived AEC features, described below, were standardized with their own scaler, fit on the internal cohort and applied frozen to the external cohort, kept separate from the clinical-feature scaler so that features on different physical scales did not dominate the L2-regularized logistic regression.

Three curve-level summary statistics, following standard time-series summary-feature practice [8], were computed per patient directly from the 128-point curve: the standard deviation, or SD, across all 128 points, reflecting the overall range of tube-current modulation; the skewness of the point distribution, reflecting asymmetry of the modulation profile along the craniocaudal axis; and the ratio of the mean tube current over the cranial half of the curve to the mean over the caudal half, reflecting the relative attenuation burden of the upper versus lower abdomen.

Component scores from the functional principal component analysis described below, PC1 through PC3, were included as a fourth AEC feature candidate, providing a lower-dimensional representation of curve shape beyond these three summary statistics; during internal cross-validation, PCA was refit on each training fold and the resulting transform applied to the held-out fold, to prevent validation-curve information from leaking into the component estimation.

## Functional Principal Component Analysis

The AEC-128 curve is a 128-dimensional variable per patient, but adjacent points are strongly autocorrelated, making it a functional data object rather than a set of independent measurements. Entering all 128 points as independent variables would incur substantial multicollinearity and overfitting risk; functional principal component analysis, abbreviated FPCA [9], was therefore applied to summarize the curve's shape information in a lower-dimensional form. FPCA extends the principle of principal component analysis, or PCA, to functional data, approximating each curve as a linear combination of a mean curve and a series of orthogonal components that explain the greatest inter-curve variation; each patient's curve is then summarized by its projection onto these components, the component score, or FPC score.

Specifically, first, the mean curve μ(z), for z = 1, ..., 128, was computed across all curves in the internal cohort. Second, for each patient, a deviation curve dᵢ(z) = xᵢ(z) − μ(z) was obtained by subtracting the mean curve from the raw curve. Third, the 128 × 128 covariance matrix of the deviation curves was eigen-decomposed, computed in practice via singular value decomposition, or SVD, for numerical stability using scikit-learn's PCA implementation, to obtain components, termed eigenfunctions and denoted φₖ, and eigenvalues, denoted λₖ, ranked by the proportion of variance explained. Fourth, each patient's deviation curve was projected onto each component to yield the component score, scoreᵢ,ₖ = Σ_z dᵢ(z)·φₖ(z).

**Figure 1.** Computation of FPCA on the internal-cohort AEC-128 curves. (A) A random sample of patient curves, shown in gray, and the resulting mean curve μ(z), shown in black. (B) Deviation curve dᵢ(z) = xᵢ(z) − μ(z) for a representative patient. (C, F) The 128×128 covariance matrix, eigen-decomposed via SVD to obtain (E) the eigenvalue scree plot used for elbow-based component selection, with k = 3, and (G) the resulting eigenfunctions φ₁–φ₃, shown as μ(z) ± √λₖ·φₖ(z). (D) Projection of the representative patient's deviation curve onto φ₁ yields the component score scoreᵢ,₁ = Σᵩ dᵢ(z)φ₁(z) = 1,318.

Fitting PCA to the internal cohort showed that PC1 explained 84.3% of total variance, PC1–2 cumulatively explained 94.8%, and PC1–3 cumulatively explained 97.4%. The number of retained components was determined from the elbow of the explained-variance scree plot, using the Kneedle algorithm [10]: component index and individual explained-variance ratio were each normalized to the 0–1 range, and the elbow was defined as the point of maximal perpendicular distance from the line connecting the first and last components. This elbow coincided with PC3; three component scores, PC1 through PC3, were therefore fixed as the representative AEC curve descriptors. As an illustrative check of patient-level reconstruction accuracy, one patient with large PC1–PC3 scores and reconstruction fit in the top quartile, with component values of PC1 +1,318, PC2 +1,175, and PC3 −247, was reconstructed from these three scores; the coefficient of determination, R², between the reconstructed and original curve was 0.992, confirming that the 128-point raw curve can be closely approximated by just three component scores.

Components, or eigenfunctions, were estimated from the internal cohort only; curves from the external cohort were projected onto these fixed components without refitting to obtain component scores, ensuring that the two cohorts were compared on a common basis. This procedure quantifies morphological variation in curve shape that is not captured by individual summary statistics such as the mean, standard deviation, or skewness, for use as an input variable in the predictive models, consistent with prior applications of FPCA to CT-derived curves for clinical outcome prediction [11].

## AEC Feature Selection and Evaluation

Five AEC feature candidates were compared as additions to clinic4: SD alone, skewness alone, the upper/lower ratio alone, FPCA scores alone, and all shape features combined with FPCA scores. For each disease, all five candidates were evaluated by stratified cross-validated internal AUC using logistic regression [12], and the candidate with the highest mean internal AUC was selected independently for that disease, since the shape of the AEC-clinical association was not assumed to be shared across HTN, DM, and CKD. External data played no role in this selection: model comparison relied on internal cross-validation only, and the external cohort was reserved for a single, frozen final evaluation of the selected model, consistent with the internal/external validation roles described above.

For each disease, the resulting model combining clinic4 with the best-performing AEC feature set was compared against clinic4 by internal out-of-fold AUC and by external AUC from the internal-trained model applied once, without retraining, following the two-cohort clinic-model-versus-clinic-plus-curve-model design used in prior curve-based disease prediction studies [13]. The AUC difference between the two models was tested with the paired DeLong test [14], appropriate because both models were evaluated on the same patients. A classification threshold was chosen by maximizing Youden's J statistic [15] on internal out-of-fold predictions and then applied, fixed, to compute sensitivity, specificity, and accuracy in both cohorts. To assess whether the AEC contribution was consistent across CT manufacturers, AUC was additionally recomputed within each scanner subgroup containing at least 30 patients with both outcome classes represented, using the same fixed threshold.

# References

1. McCollough CH, Bruesewitz MR, Kofler JM Jr. Techniques and Applications of Automatic Tube Current Modulation for CT. *Radiology*. 2006;240(3):611-622.
2. Kalra MK, Maher MM, Toth TL, et al. Comparison of Z-axis automatic tube current modulation technique with fixed tube current CT scanning of abdomen and pelvis. *Radiology*. 2004;232:347-353.
3. McCollough CH, Bruesewitz MR, Kofler JM Jr. CT dose reduction and dose management tools: overview of available options. *RadioGraphics*. 2006;26(2):503-512.
4. Steyerberg EW. *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating.* 2nd ed. Springer; 2019.
5. Moons KGM, Altman DG, Reitsma JB, et al. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): Explanation and Elaboration. *Ann Intern Med*. 2015;162(1):W1-W73.
6. Ramspek CL, Jager KJ, Dekker FW, Zoccali C, van Diepen M. External validation of prognostic models: what, why, how, when and where? *Clin Kidney J*. 2021;14(1):49-58.
7. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiol Artif Intell*. 2023;5(5):e230024.
8. Barandas M, Folgado D, Fernandes L, et al. TSFEL: Time Series Feature Extraction Library. *SoftwareX*. 2020;11:100456.
9. Ramsay JO, Silverman BW. *Functional Data Analysis.* 2nd ed. Springer; 2005.
10. Satopaa V, Albrecht J, Irwin D, Raghavan B. Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior. In: *2011 31st International Conference on Distributed Computing Systems Workshops*. IEEE; 2011:166-171.
11. Ahn Y, Lee SM, Noh HN, et al. Predefined and data driven CT densitometric features predict critical illness and hospital length of stay in COVID-19 patients. *Sci Rep*. 2022;12:8916.
12. Kohavi R. A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. In: *Proceedings of the 14th International Joint Conference on Artificial Intelligence*. 1995:1137-1143.
13. Jeon ET, et al. Deep Learning-Based COPD Exacerbation Prediction Using Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study. *J Med Internet Res*. 2025;27:e69785.
14. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics*. 1988;44(3):837-845.
15. Youden WJ. Index for rating diagnostic tests. *Cancer*. 1950;3(1):32-35.
