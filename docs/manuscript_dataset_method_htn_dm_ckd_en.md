# Dataset / Method Draft (English) — Predicting HTN/DM/CKD: clinic4 vs clinic4+AEC

> Based on: `code/step2_clinic_aec_disease_select.py`, `code/step3_clinic_aec_disease_logistic.py`
> (run on 2026-08-14, outputs in `outputs/step2_disease_select/`, `outputs/step3_disease_logistic/`)
>
> **Primary Method-section template**: Jeon ET, Park H, Lee JK, Heo EY, Lee CH, Kim DK, Kim DH,
> Lee HW. "Deep Learning–Based Chronic Obstructive Pulmonary Disease Exacerbation Prediction
> Using Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study." *J Med Internet
> Res* 2025;27:e69785 (hereafter "Jeon 2025"). Its two-hospital design (a development cohort
> plus an independent external-validation cohort), its comparison of a clinical-variables-only
> logistic model ("Clin model") against a model that adds a curve-derived score ("AI-PFT-Clin
> model"), and its use of the DeLong method to test the AUROC difference map almost one-to-one
> onto the present study's clinic4-vs-clinic4+AEC design, so it was adopted as the primary
> template. (This corresponds to `Research-Reference-Paper/papers/Curve-Regression 3.pdf`.
> The pre-existing memory `[[project_reference_paper_screening]]` had incorrectly recorded this
> file as "a duplicate export of the same paper as Curve-Regression 1 — a spirometry study with
> clinical variables, 2 hospitals, n=10,492." In fact Curve-Regression 1/2/3/4 are four entirely
> different papers; that memory is stale and should be corrected in a future session.)
>
> See `manuscript_dataset_method_htn_dm_ckd_ko.md` for the Korean draft.

---

## Dataset

### Cohorts and patient-selection process

This is a **retrospective cohort study** using two tertiary-hospital cohorts. As in Jeon 2025's
two-hospital design (a development cohort, BRMH, n=1,000, vs. an external-validation cohort,
SNUH, n=489), we used an independent development ("internal") cohort and an independent
external-validation cohort — Gangnam Severance Hospital (internal) and Sinchon Severance Hospital
(external). Because the two cohorts were drawn from different hospitals and different scanner
populations, freezing the internal-trained model and applying it once to the external cohort
allowed us to assess cross-institutional generalization.

Starting from the raw extracts obtained through the hospitals' data-service teams, the final
analysis cohort was defined through the following three-step selection process.

1. **Requiring both structured clinical data and CT DICOM data**: A patient-visit-level CT DICOM
   archive (folder names encoding a study registration number) was automatically cross-checked
   against a clinical database using the same registration-number system (a multi-sheet workbook
   covering demographics, height/weight, smoking/alcohol history, past medical history,
   admission/ER records, mortality, follow-up diagnoses — DM, HTN, dyslipidemia, osteoporosis, MI,
   stroke, etc. — and lab values), keeping only patients present in both sources (verified
   exhaustively via `check_patientid_registration.py`, which checks that every folder-derived
   patientID appears in the registration-number column of each sheet). **This same clinical
   database supplied both the items used to construct clinic4 (age, sex, height, weight) and the
   diagnosis history for this study's outcome variables — hypertension (HTN), diabetes (DM), and
   chronic kidney disease (CKD) — so cohort selection and outcome-variable extraction drew on the
   identical clinical database.**
2. **CT acquisition-protocol (kVp) filtering**: Both cohorts were restricted to scans acquired at
   a tube voltage (kVp) of 100, because a different tube voltage changes the absolute scale of the
   tube-current (mA) modulation curve, making the AEC-128 signal not directly comparable across
   cohorts. In the final dataset, kVp is confirmed to be uniformly 100 in both cohorts.
3. **Scanner-model ratio filtering (Gangnam cohort only)**: To minimize the influence of rarely
   used scanner models, the Gangnam cohort was further filtered toward models with high usage
   share. As a result, 99.2% (1,081/1,090) of the final Gangnam cohort is concentrated in four
   major models (Sensation 64, 34.9%; Revolution CT, 29.1%; Ingenuity Core 128, 18.5%; SOMATOM
   Definition AS+, 16.7%). The Sinchon cohort, by contrast, retained this additional filter-free
   heterogeneity, spanning 15 scanner models (SOMATOM Definition Flash, 24.6%; SOMATOM Definition
   AS+, 17.9%; Revolution EVO, 16.1%; iCT 256, 14.6%; SOMATOM Force, 13.3%; and others) — so that
   the external cohort would provide a stricter test of generalization across scanner differences.

The final patient counts passing all selection steps were 1,090 (Gangnam, internal) and 926
(Sinchon, external).

### Clinical variables (clinic4)

From the same clinical database used for matching in step 1 above, age (PatientAge), sex
(PatientSex), height (Height), and weight (Weight) — four variables collectively termed "clinic4"
— were extracted as the baseline clinical predictor set, playing the same role as Jeon 2025's
"Clin model" (a baseline logistic regression using only clinical variables such as prior AE-COPD
history, dyspnea, and inhaled-treatment use). Age, height, and weight were standardized (z-score)
using parameters fit on the internal cohort and then applied, frozen, to the external cohort; sex
was coded as a binary indicator (male = 1).

### Outcome variables (HTN, DM, CKD)

The outcome variables were the already-diagnosed binary (0/1) hypertension (HTN), diabetes
mellitus (DM), and chronic kidney disease (CKD) labels, drawn from the same clinical database used
for cohort selection — specifically its follow-up diagnosis sheets, which record diagnosis status
per disease based on presence/absence of a first-diagnosis date; no new dichotomization of a
continuous measure was performed. Prevalence in each cohort was as follows.

| Disease | Internal (n=1,090) | External (n=926) |
| --- | --- | --- |
| HTN | 329 (30.2%) | 425 (45.9%) |
| DM | 204 (18.7%) | 285 (30.8%) |
| CKD | 83 (7.6%) | 168 (18.1%) |

### The AEC-128 curve

For each patient, the CT scanner records an **Automatic Exposure Control (AEC) tube-current (mA)
modulation curve** during the chest-abdomen scan. This curve is the tube current the scanner
adjusts in real time along the z-axis (craniocaudal slice position) to keep image quality
constant, and it inherently encodes the patient's cross-sectional attenuation characteristics —
i.e., body habitus and size — at each position. Multiple scanner manufacturers are represented in
the data (e.g., SOMATOM Definition AS+, Ingenuity Core 128, Revolution CT), which enabled a
scanner-subgroup analysis in addition to the cross-institutional validation.

**Curve-generation procedure**:

1. **Axial-series selection**: Among the multiple series mixed together in a patient folder
   (the true axial acquisition, coronal/sagittal reformats, scout images, dose reports, etc.), we
   excluded series with fewer than 20 slices (treated as scout/dose-report series), series whose
   DICOM ImageType contained DERIVED/REFORMATTED or whose SeriesDescription contained a
   reformat-related keyword as a whole token (MPR/COR/CORONAL/SAG/SAGITTAL/SPO), and series that
   were not axial by ImageOrientationPatient — leaving only axial series.
2. **Landmark localization with TotalSegmentator**: For each remaining axial series,
   TotalSegmentator segmented the liver, left hip, and right hip. The most superior slice of the
   liver mask was set as the upper (liver) landmark and the most inferior slice of the combined
   hip mask as the lower (pubis) landmark, subject to QC criteria (liver mask ≥3,000 voxels and
   its median slice position more superior than the 30th percentile of the slice range). **This
   segmentation step is used solely to localize the anatomical crop boundaries — the AEC signal
   itself is not a segmentation output but is read directly from the DICOM header's tube-current
   field (XRayTubeCurrent).**
3. **Cropping to the pubis–liver range**: Only series with both landmarks detected and in valid
   order (pubis position ≤ liver position) were retained ("ok" status); the tube-current (mA)
   values within that range were extracted in pubis-to-liver order.
4. **Resampling to 128 points**: Because the raw cropped slice count varies by patient (101–238
   slices overall: 110–238 in Gangnam, 101–205 in Sinchon; medians 143–149), each patient's cropped
   tube-current array was resampled via linear interpolation to 128 points, yielding the `aec_128`
   curve used as the final unit of analysis.

Just as Jeon 2025 used the flow-volume/volume-time curves from spirometry — a signal already
being measured — as an additional biomarker, the AEC-128 curve in this study is likewise an
opportunistic signal obtained as a byproduct of a CT scan that has already been acquired,
requiring no additional radiation exposure. As step 2 above makes clear, however, TotalSegmentator
segmentation is involved in fixing the crop boundaries, so the more precise claim is not that "no
segmentation is involved at all" but that segmentation here is used only for landmark
localization rather than as the biomarker itself — the quantitative signal (the AEC value) is read
directly from the DICOM header.

---

## Method

### Model specification

Following Jeon 2025's comparison of a "Clin model" (clinical variables only) against an
"AI-PFT-Clin model" (clinical variables plus a curve-derived score), we compared two logistic
regression models: (1) a baseline model using clinic4 alone, and (2) an augmented model
(clinic4+AEC(best)) that adds shape features derived from the AEC-128 curve to clinic4. Just as
Jeon 2025 reported that adding pulmonary-function-curve information to clinical variables
improved AE-COPD prediction, we tested the analogous hypothesis that adding the CT scanner's AEC
curve itself as an opportunistic biomarker improves diagnostic classification performance beyond
a clinical-only model.

### AEC shape features and selection of the best combination

Five candidate representations were derived from the raw AEC-128 curve: (i) standard deviation
(SD), (ii) skewness, (iii) the ratio of the upper-half to lower-half 50% segment means, (iv)
functional PCA (FPCA) scores, and (v) a combination of all four. FPCA was obtained by fitting PCA
to the raw AEC-128 curves of the internal cohort. The number of components was chosen not by
downstream predictive performance (AUC) but by the cumulative-explained-variance-ratio criterion
commonly used for choosing the number of PCA/FPCA components (Jolliffe 2002; Ramsay & Silverman
2005), taken as the smallest n at which the **cumulative explained variance ratio first reached
99.5%** (n=7 in this run) so as to minimize information loss from the curve. Thresholds typically
illustrated in the literature fall in the 70–95% range (Jolliffe 2002); the specific value of
99.5% is therefore a project-specific choice by the authors rather than a threshold recommended
by either reference. Among the five candidates, the combination with the highest mean internal 5-fold stratified
cross-validated out-of-fold (OOF) AUC averaged across HTN, DM, and CKD was selected as the final
clinic4+AEC(best) (FPCA(PC1–7) alone was selected in this run); this selection step used only
internal data and never referenced the external cohort.

### Internal cross-validation and frozen external validation

Analogous to Jeon 2025's split of the BRMH cohort into training (60%), internal validation (20%),
and internal test (20%) sets, with SNUH held out as a separate external-validation set, we applied
stratified K-fold cross-validation (K=5, automatically reduced when a class had fewer members) to
the internal cohort and computed the internal AUC from out-of-fold predicted probabilities. For
models that included FPCA, PCA was refit independently within each fold's training partition only,
to prevent the leakage that would occur if curve information from the validation fold contaminated
the eigenfunction estimate. The external cohort was scored once, using the single model frozen
after training on the full internal cohort; consistent with the internal-selection principle
already established, external results were never used to select the model or the feature
combination.

### Statistical comparison

Following Jeon 2025's use of the **DeLong method** to test the AUROC difference between the Clin
model and the AI-PFT-Clin model, and its use of a Youden-index cutoff derived from the internal
validation set to compute sensitivity/specificity, we tested the AUC difference between clinic4
and clinic4+AEC(best) — two prediction scores obtained from the same patients — with a **paired
DeLong test** (DeLong et al. 1988; Sun & Xu 2014 algorithm), performed independently for the
internal and external cohorts. The classification threshold was set at the point maximizing
**Youden's J index** (sensitivity + specificity − 1) on the internal OOF ROC curve, then applied,
fixed, to the external cohort to compute sensitivity, specificity, and accuracy (significance
threshold P<0.05, as in Jeon 2025). Paralleling Jeon 2025's age/sex/smoking subgroup analyses, we
additionally recomputed AUC within scanner-manufacturer subgroups (restricted to scanners with
≥30 patients and both outcome classes present), re-slicing the already-computed predicted
probabilities without retraining, to confirm that results were not confined to a particular
scanner.

### Summary of key results (for reference; values may change on re-run)

| Disease | Internal AUC (clinic4→+AEC) | Internal DeLong P | External AUC (clinic4→+AEC) | External DeLong P |
| --- | --- | --- | --- | --- |
| HTN | 0.808 → 0.815 | 0.183 (n.s.) | 0.715 → 0.728 | **0.031** |
| DM | 0.727 → 0.751 | **0.018** | 0.662 → 0.697 | **0.002** |
| CKD | 0.790 → 0.803 | 0.335 (n.s.) | 0.622 → 0.635 | 0.140 (n.s.) |

DM showed a significant improvement in both the internal and external cohorts, and HTN was
significant only externally. CKD did not reach significance in either cohort in this run.
Unlike the pattern seen when predicting continuous body-composition measures — where adding AEC
consistently worsened performance — the direction of effect for these diagnostic classification
tasks was consistently favorable, though statistical significance varied by disease. This
resembles Jeon 2025's finding that the AUROC improvement for severe AE-COPD (0.675→0.713) in the
external validation cohort was smaller than for moderate-to-severe AE-COPD; outcome variables
with lower prevalence (CKD: 7.6–18.1%) appear to have lower statistical power to reach
significance.

---

## References (basis for the Method section)

1. **[Primary template]** Jeon ET, Park H, Lee JK, Heo EY, Lee CH, Kim DK, Kim DH, Lee HW. Deep
   Learning–Based Chronic Obstructive Pulmonary Disease Exacerbation Prediction Using
   Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study. *J Med Internet Res*
   2025;27:e69785. doi:10.2196/69785
2. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
   receiver operating characteristic curves: a nonparametric approach. *Biometrics*
   1988;44(3):837-845.
3. Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas under
   correlated receiver operating characteristic curves. *IEEE Signal Process Lett*
   2014;21(11):1389-1393.
4. Ramsay JO, Silverman BW. *Functional Data Analysis*. 2nd ed. Springer; 2005. (basis for FPCA)
5. Jolliffe IT. *Principal Component Analysis*. 2nd ed. Springer; 2002. (basis for choosing the
   number of components via the cumulative-explained-variance criterion — Ch.6; note that the
   thresholds illustrated there are typically 70–95%, so the specific 99.5% value used here is
   an author-specified choice, not a value recommended by this or the FPCA reference)
6. (Optional) "A new method for estimating patient body weight using CT dose modulation data" —
   can be cited in the Dataset section as evidence that the AEC/tube-current-modulation curve
   itself carries body-habitus information.

> **Caveat**: the values in the table above reflect a single script run on 2026-08-14 and may
> change on re-run due to randomness in cross-validation fold assignment or data updates (as
> noted in the script's own comments). Re-confirm with the final run before manuscript
> submission.
