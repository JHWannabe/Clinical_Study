# Materials

This retrospective cohort study comprised two independent institutional cohorts: an internal cohort from Gangnam Severance Hospital and an external cohort from Sinchon Severance Hospital. Because the two cohorts were drawn from different hospitals and different scanner populations, the model trained on the internal cohort was applied, without retraining, to the external cohort to assess external validity. CT examinations for both cohorts were performed between January 2018 and June 2020.

Inclusion criteria were as follows:

1. Both clinical data and an abdominal CT examination were available for the same patient.
2. Only CT examinations acquired at a tube voltage of 100 kVp were included.
3. In the internal cohort only, patients were restricted to those scanned on the four most commonly used CT scanner models at that site: Siemens Sensation 64, GE Revolution CT, Philips Ingenuity Core 128, and Siemens SOMATOM Definition AS+. No such restriction was applied to the external cohort.

After application of these criteria, the final study population consisted of 1,090 patients in the internal cohort and 926 patients in the external cohort.

Age, sex, height, and weight were abstracted from the same clinical database used for patient matching and served as the baseline clinical predictors, collectively termed clinic4. Outcomes were defined as the presence or absence of a documented diagnosis of hypertension (HTN), diabetes mellitus (DM), and chronic kidney disease (CKD) in the clinical database, rather than by dichotomizing a continuous laboratory measurement. Hypertension was present in 329 of 1,090 internal patients (30.2%) and 425 of 926 external patients (45.9%); diabetes mellitus in 204 internal patients (18.7%) and 285 external patients (30.8%); and chronic kidney disease in 83 internal patients (7.6%) and 168 external patients (18.1%).

Baseline characteristics of the internal and external cohorts are summarized in Table 1. There were no missing values for age, sex, height, weight, BMI, comorbidity, or scanner vendor in either cohort.

**Table 1. Baseline Characteristics of the Patients**

| Characteristic | Internal cohort<br>(n = 1,090) | External cohort<br>(n = 926) | *p*-value |
|---|---|---|---|
| Age, years, mean ± SD (range) | 57.0 ± 12.2 (14–91) | 59.5 ± 12.7 (19–90) | <0.001 |
| Sex, female, n (%) | 700 (64.2) | 498 (53.8) | <0.001 |
| Sex, male, n (%) | 390 (35.8) | 428 (46.2) | <0.001 |
| Height, cm, mean ± SD | 162.2 ± 8.2 | 162.5 ± 8.6 | 0.381 |
| Weight, kg, mean ± SD | 61.9 ± 10.6 | 61.8 ± 11.4 | 0.848 |
| BMI, kg/m², mean ± SD | 23.5 ± 3.1 | 23.3 ± 3.4 | 0.337 |
| Hypertension, n (%) | 329 (30.2) | 425 (45.9) | <0.001 |
| Diabetes mellitus, n (%) | 204 (18.7) | 285 (30.8) | <0.001 |
| Chronic kidney disease, n (%) | 83 (7.6) | 168 (18.1) | <0.001 |
| CT scanner vendor, n (%) |  |  | <0.001 |
| — Siemens | 568 (52.1) | 569 (61.4) |  |
| — GE | 318 (29.2) | 219 (23.6) |  |
| — Philips | 202 (18.5) | 135 (14.6) |  |
| — Canon | 2 (0.2) | 3 (0.3) |  |

SD, standard deviation; BMI, body mass index. Continuous variables were compared with Welch's *t*-test; categorical variables were compared with the chi-square test.

During each abdominal CT acquisition, **automatic exposure control (AEC)** modulates the tube current (mA) along the craniocaudal (z) axis to maintain a constant image noise level; the resulting tube-current modulation curve reflects the patient's cross-sectional attenuation, and hence body habitus, at each slice position. In this study, a fixed-length, 128-point representation of this curve (AEC-128) was used as the imaging-derived predictor.


# Method

## AEC-128 Curve Construction

To construct the fixed-length AEC-128 feature, the axial source series was first identified for each patient, excluding coronal/sagittal reformations, scout images, and dose reports, based on slice count, DICOM ImageType, SeriesDescription, and image orientation. An open-source, deep-learning-based segmentation tool (TotalSegmentator) was then applied to localize the liver and the bilateral hip bones, with the cranial-most liver slice and caudal-most hip slice defining the upper and lower boundaries of the analysis range; segmentation was used solely for this anatomical landmark localization, while tube-current values themselves were obtained directly from the DICOM header (XRayTubeCurrent). Tube-current values within this pubis-to-liver range were extracted in craniocaudal order, and because the resulting slice count varied across patients from 101 to 238 slices, each sequence was resampled by linear interpolation to a fixed length of 128 points, yielding the AEC-128 curve used as the analytic unit in subsequent modeling.

## Clinical Baseline Model

For each disease outcome, a baseline logistic regression model (clinic4) was built from the four clinical predictors described above: age, sex, height, and weight. Sex, a binary variable, was entered directly as a 0/1 indicator without scaling, while age, height, and weight were standardized to zero mean and unit variance using a scaler fit on the internal cohort and applied, without refitting, to the external cohort. This baseline was evaluated by internal cross-validated and external frozen AUC for HTN, DM, and CKD, and served as the reference against which the incremental value of AEC-128 was measured.

## AEC-Derived Feature Extraction

Raw AEC-128 curves were used without further normalization, since sex, age, height, and weight variation already enters through the clinical predictors and an additional cohort-level scaling step risked removing the body-habitus signal that AEC-128 is intended to capture. Derived AEC features, described below, were standardized with their own scaler, fit on the internal cohort and applied frozen to the external cohort, kept separate from the clinical-feature scaler so that features on different physical scales did not dominate the L2-regularized logistic regression.

Three curve-level summary statistics were computed per patient directly from the 128-point curve: the standard deviation (SD) across all 128 points, reflecting the overall range of tube-current modulation; the skewness of the point distribution, reflecting asymmetry of the modulation profile along the craniocaudal axis; and the ratio of the mean tube current over the cranial half of the curve to the mean over the caudal half, reflecting the relative attenuation burden of the upper versus lower abdomen.

Functional principal component analysis (FPCA) was applied to the internal cohort's AEC-128 curves to extract a lower-dimensional representation of curve shape beyond these three summary statistics. The number of retained components was determined from the scree plot of individual explained-variance ratio using the Kneedle elbow-detection method: both axes were min-max normalized, and the elbow was defined as the component at maximal perpendicular distance from the chord connecting the first and last points of the scree curve. This variance-based criterion, rather than downstream prediction accuracy, avoided choosing the component count on the same outcome later used for AUC comparison, which would otherwise inflate estimated performance. With the component count fixed, PCA was fit on the internal cohort's raw curves and applied, without refitting, to project the external cohort's curves onto the same components; during internal cross-validation, PCA was instead refit on each training fold and the resulting transform applied to the held-out fold, to prevent validation-curve information from leaking into the component estimation.

## Selection and Evaluation of AEC Feature Sets

Five AEC feature candidates were compared as additions to clinic4: SD alone, skewness alone, the upper/lower ratio alone, FPCA scores alone, and all shape features combined with FPCA scores. For each disease, all five candidates were evaluated by stratified cross-validated internal AUC using logistic regression, and the candidate with the highest mean internal AUC was selected independently for that disease, since the shape of the AEC-clinical association was not assumed to be shared across HTN, DM, and CKD. External data played no role in this selection: model comparison relied on internal cross-validation only, and the external cohort was reserved for a single, frozen final evaluation of the selected model, consistent with the internal/external validation roles described above.

For each disease, the resulting clinic4+AEC(best) model was compared against clinic4 by internal out-of-fold AUC and by external AUC from the internal-trained model applied once, without retraining. The AUC difference between the two models was tested with the paired DeLong test, appropriate because both models were evaluated on the same patients. A classification threshold was chosen by maximizing Youden's J statistic on internal out-of-fold predictions and then applied, fixed, to compute sensitivity, specificity, and accuracy in both cohorts. To assess whether the AEC contribution was consistent across CT manufacturers, AUC was additionally recomputed within each scanner subgroup containing at least 30 patients with both outcome classes represented, using the same fixed threshold.

