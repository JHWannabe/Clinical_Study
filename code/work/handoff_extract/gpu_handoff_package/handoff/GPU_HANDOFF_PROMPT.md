# GPU Handoff Prompt: AEC / Low SMI Reclassification Project

이 파일은 `g1090.xlsx`로 모델을 만들고 `sdata.xlsx`에서 외부검증한 AEC 기반 low SMI 연구를 GPU 컴퓨터에서 이어가기 위한 전체 handoff 문서다. 새 에이전트는 이 문서만 읽고도 현재 연구 framing, 고정 조건, 이미 실패한 접근, 현재 가장 방어 가능한 결과, 다음 분석 방향을 이해해야 한다.

## 1. Research Objective

목표는 CT acquisition 과정에서 생성되는 AEC profile이 low skeletal muscle index (low SMI)를 찾는 데 clinical model을 보완할 수 있는지 평가하는 것이다.

핵심 framing은 이미 정해졌다.

> AEC is not a standalone low-SMI detector. AEC is an acquisition-derived second-stage reclassification / triage biomarker that improves clinical utility among patients flagged by a simple clinical model.

즉 논문은 "AEC가 AUC를 크게 올린다"가 아니라 다음 방향으로 가야 한다.

> Clinical variables already identify many low-SMI patients but produce many false positives. AEC-derived shape signatures can reclassify clinically flagged patients, identifying a very-low-risk de-escalation group and a high-yield priority group in external validation.

## 2. Required Data Files

Upload these Excel files to the GPU computer:

- `g1090.xlsx`: training/development cohort
- `sdata.xlsx`: external test cohort

Original local paths were:

- `C:\Users\user\OneDrive\1. RESEARCH\radiation\g1090.xlsx`
- `C:\Users\user\OneDrive\1. RESEARCH\radiation\sdata.xlsx`

Expected workbook sheets:

- `metadata`
- `aec_128`
- `aec_cropped`

The existing scripts assume these sheet names.

## 3. Fixed Outcome Definition

Outcome is fixed as low SMI using Derstine cutoffs.

Formula:

```text
SMI = TAMA / (height in meters)^2
```

Cutoffs:

```text
Male low SMI   = SMI < 45.4 cm2/m2
Female low SMI = SMI < 34.4 cm2/m2
```

Do not change this unless explicitly instructed. Yoon criteria were considered earlier but discarded by user. Derstine low SMI is fixed.

## 4. Fixed Clinical Model

Clinical model is fixed as:

```text
age + sex + height + weight
```

Important constraints:

- Do not include scanner, vendor, phase, protocol, or acquisition scanner variables in the main model.
- Do not include TAMA after BMI/sex/age because that breaks the intended workflow.
- Scanner/phase may be used only for subgroup reporting or sensitivity analysis, not as model input.

## 5. Cohort Summary

Training cohort:

```text
g1090: n = 1090
low SMI events = 129
prevalence = 11.8%
```

External test cohort:

```text
sdata: n = 926
low SMI events = 141
prevalence = 15.2%
male: 98 / 428 = 22.9%
female: 43 / 498 = 8.6%
```

## 6. Current Best Result

Current best no-scanner model was originally:

```text
clinical score + female-boundary gated direct AEC curve linear SVM score
```

But this is considered too complex / potentially post-hoc for main manuscript. It should be treated as a benchmark or supplement, not necessarily the final main method.

External sdata performance:

| Model | AUC | Sensitivity | Specificity | PPV | TP | FP |
|---|---:|---:|---:|---:|---:|---:|
| Clinical only | 0.8337 | 0.8865 | 0.5618 | 0.2665 | 125 | 344 |
| AEC boundary gate benchmark | 0.8420 | 0.8652 | 0.6815 | 0.3280 | 122 | 250 |

Key comparison:

```text
AUC: 0.8337 -> 0.8420
Sensitivity: 0.8865 -> 0.8652
Specificity: 0.5618 -> 0.6815
PPV: 0.2665 -> 0.3280
FP: 344 -> 250
TP: 125 -> 122
```

Bootstrap:

```text
Delta AUC mean +0.0083
95% CI approximately [-0.0012, 0.0191]
p(delta <= 0) approximately 0.044
Delta specificity approximately +0.120
Delta PPV approximately +0.061
Delta FP approximately -94
```

Interpretation:

```text
AUC gain is modest.
Clinical utility signal is false-positive reduction / reclassification.
```

## 7. Most Important Reclassification Result

At train-derived clinical and AEC thresholds:

| Group | n | low SMI | Prevalence |
|---|---:|---:|---:|
| clinical+ / AEC+ | 366 | 121 | 33.1% |
| clinical+ / AEC- | 103 | 4 | 3.9% |
| clinical- / AEC+ | 6 | 1 | 16.7% |
| clinical- / AEC- | 451 | 15 | 3.3% |

This is the strongest result.

Main interpretation:

```text
Among patients flagged by the clinical model, AEC-negative patients had very low observed low-SMI prevalence.
AEC can de-escalate a subset of clinical positives and enrich a high-priority subset.
```

Strict train-defined de-escalation:

```text
Clinical-positive patients de-escalated: 41
Observed low SMI in those 41 on sdata: 0
False positives removed: 41
```

High-priority enrichment:

```text
Clinical-positive high-priority group: n = 146
low SMI = 76
PPV = 52.1%
```

Clinical-positive baseline PPV was 26.7%, so high-priority enrichment roughly doubles observed PPV.

## 8. Decision Curve Result

AEC gate had consistently higher fixed-rule net benefit than clinical alone across threshold probabilities 0.10-0.40.

Net interventions avoided per 100 patients vs clinical:

```text
pt 0.10: +7.2
pt 0.20: +8.9
pt 0.30: +9.4
pt 0.40: +9.7
```

Use this to support workflow utility.

## 9. Current Methodological Concern

The old benchmark uses:

```text
female Gaussian boundary gate + lambda 0.25
```

This may look over-engineered or post-hoc:

- Why female only?
- Why Gaussian?
- Why lambda 0.25?
- Was it selected by looking at external test?

The user wants the method to be much simpler and more classic.

## 10. Recommended Main Method Going Forward

Do not present a complex nonlinear gated model as the main method.

Recommended final main framing:

```text
Classic diagnostic gray-zone / reflex-test framework
```

Proposed workflow:

```text
Step 1. Fit clinical model using age, sex, height, weight.
Step 2. Define clinically indeterminate patients near the train-derived clinical threshold.
Step 3. Apply scanner-free AEC expert only as a second-stage reclassification marker in this gray zone.
Step 4. Evaluate external test performance using specificity, PPV, false-positive reduction, reclassification tables, and DCA.
```

Preferred language:

> AEC was evaluated as a second-stage reclassification test for clinically indeterminate patients near the clinical decision threshold.

Avoid main-text language:

> female Gaussian boundary-gated nonlinear dynamic AEC rescue rule with lambda 0.25

## 11. Simplified Alternative Tested

A simpler hard indeterminate-zone version was tested.

Definition:

```text
Clinical indeterminate zone:
|standardized clinical score - clinical threshold| <= width
```

Candidate main model:

```text
hard indeterminate zone, female, width 0.50, lambda 0.25
```

Performance:

| Model | AUC | Sensitivity | Specificity | PPV | TP | FP |
|---|---:|---:|---:|---:|---:|---:|
| Clinical only | 0.834 | 0.887 | 0.562 | 0.267 | 125 | 344 |
| Hard indeterminate zone, female, width 0.50 | 0.842 | 0.865 | 0.676 | 0.324 | 122 | 254 |
| Old Gaussian female gate | 0.842 | 0.865 | 0.682 | 0.328 | 122 | 250 |

This is nearly identical to the Gaussian benchmark but easier to explain.

All-sex hard zone was also tested:

```text
Hard indeterminate zone, all sex, width 0.50:
AUC 0.839
Sensitivity 0.801
Specificity 0.721
PPV 0.340
FP 219
```

All-sex version reduces FP more but loses too much sensitivity. For low-SMI screening/triage, sensitivity near 0.80 is probably too weak.

## 12. Classic Additive Models Tested and Mostly Failed

These were tested:

- Clinical + AEC score logistic regression
- Clinical + AEC + sex interaction logistic regression
- AEC PCA + logistic regression
- Clinical + AEC PCA + logistic regression

Results were weak.

Examples:

| Model | AUC | Sensitivity | Specificity | PPV | FP |
|---|---:|---:|---:|---:|---:|
| Clinical logistic | 0.834 | 0.879 | 0.596 | 0.281 | 317 |
| AEC expert only | 0.566 | 0.482 | 0.597 | 0.177 | 316 |
| Clinical + AEC score logistic | 0.830 | 0.879 | 0.578 | 0.273 | 331 |
| Clinical + AEC + sex interaction logistic | 0.830 | 0.887 | 0.577 | 0.274 | 332 |

PCA/logistic results:

```text
AEC PCA logistic AUC roughly 0.57-0.61
Clinical + AEC PCA logistic often worse or unstable
```

Conclusion:

```text
Fully classic additive logistic modeling is methodologically clean but does not preserve the AEC signal.
The more defensible classic framing is a diagnostic gray-zone / reflex-test design, not an additive omnibus classifier.
```

## 13. AEC Expert Feature Definition

AEC expert input:

```text
aec_128 normalized 128 positions
+
aec_cropped normalized 128 positions
= 256 direct AEC curve features
```

Preprocessing:

```text
1. Resample AEC profile to 128 positions.
2. Patient-wise mean normalization:
   normalized AEC = AEC / patient mean AEC
3. Centering:
   centered AEC = normalized AEC - 1
```

The goal is to capture shape rather than absolute mA/protocol level.

Linear SVM expert:

```text
Input: 256 direct curve features
Feature selection: SelectKBest, k=128
Classifier: class-balanced linear SVM
Output: AEC expert decision score
```

Standalone AEC expert performance:

```text
OOF AUC: 0.578
External AUC: 0.566
```

Interpretation:

```text
AEC expert is weak alone.
It should not be sold as detector.
It is a shape-derived second-stage signal.
```

## 14. Scripts Already Created

Root workspace on original computer:

```text
C:\Users\user\Documents\Codex\2026-06-21\new-chat
```

Important scripts:

- `work/g1090_sdata_aec_assault.py`: broad original model grid and feature definitions.
- `work/g1090_sdata_dynamic_gating_no_scanner.py`: scanner-free dynamic gating, current best benchmark.
- `work/g1090_sdata_torch_deep_no_scanner.py`: PyTorch deep sequence models tested earlier.
- `work/g1090_sdata_missed_case_rescue.py`: missed-case audit and rescue attempts.
- `work/g1090_sdata_orthogonal_ensemble_no_scanner.py`: orthogonal-view ensemble attempts.
- `work/g1090_sdata_grand_perspective_ensemble.py`: grand ensemble; did not beat AEC boundary gate.
- `work/g1090_sdata_make_paper_figures.py`: calibration, DCA, reclassification, subgroup forest, waterfall figures.
- `work/g1090_sdata_visualize_aec_expert.py`: AEC expert schematic, coefficient map, feature matrix thumbnail.
- `work/g1090_sdata_simplify_aec_gate.py`: hard-zone simplified alternatives and classic score comparisons.
- `work/g1090_sdata_classic_aec_pca_logistic.py`: PCA/logistic classic model tests.

Important output folders:

- `work/analysis_g1090_sdata_paper_figures`
- `work/analysis_g1090_sdata_aec_expert_visual`
- `work/analysis_g1090_sdata_simplified_aec_gate`
- `work/analysis_g1090_sdata_classic_aec_pca`
- `work/analysis_g1090_sdata_robustness`
- `work/analysis_g1090_sdata_more_more`

## 15. Existing Figures

Already generated paper-style figures:

- `figure_1_calibration.png`
- `figure_2_decision_curve.png`
- `figure_3_reclassification_heatmap.png`
- `figure_4_subgroup_forest_specificity.png`
- `figure_5_clinical_positive_waterfall.png`
- `figure_6_aec_expert_method.png`
- `figure_7_aec_feature_matrix_thumbnail.png`

Use these to understand the current story. Regenerate on GPU if paths change.

## 16. How to Run Existing Scripts on New Machine

Install common dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl torch
```

Then edit these constants in scripts if paths change:

```python
ROOT = Path("path/to/workspace")
DATA_DIR = Path("path/to/folder/containing/g1090_and_sdata")
```

Run:

```bash
python work/g1090_sdata_simplify_aec_gate.py
python work/g1090_sdata_make_paper_figures.py
python work/g1090_sdata_visualize_aec_expert.py
```

If doing GPU deep learning:

```bash
python work/g1090_sdata_torch_deep_no_scanner.py
```

But do not assume GPU deep learning will improve the main claim. Prior PyTorch AEC sequence models did not beat the simpler benchmark.

## 17. What the GPU Computer Should Do Next

Primary next task:

```text
Rebuild the analysis around the classic gray-zone / reflex-test framework.
```

Specific tasks:

1. Lock the primary clinical model:

```text
age + sex + height + weight
```

2. Lock the external test:

```text
g1090 for model development
sdata for final external validation only
```

3. Define AEC expert using scanner-free normalized direct AEC curve.

4. Evaluate several pre-specified simple gray-zone definitions using training/CV only:

```text
clinical score within 0.5 SD of threshold
clinical score within 0.75 SD of threshold
clinical predicted probability 0.10-0.30
clinical predicted probability 0.15-0.35
```

5. Select one primary rule based on training/CV only, then report locked sdata result.

6. Report:

- AUC
- sensitivity
- specificity
- PPV
- NPV
- TP / FP / FN / TN
- reclassification 2x2
- de-escalation group event rate
- high-priority group PPV
- DCA
- subgroup robustness by sex and scanner
- bootstrap CIs

7. Put complex Gaussian/female/lambda model in supplement only, as sensitivity analysis.

## 18. What Not To Do

Do not chase tiny AUC gains by creating many post-hoc gates.

Do not present AEC as a strong standalone detector.

Do not use scanner/vendor/phase as main model features.

Do not use TAMA in the predictor side.

Do not overclaim:

Bad:

> AEC substantially improves low-SMI detection.

Good:

> AEC provides acquisition-derived second-stage reclassification among clinically flagged or clinically indeterminate patients.

## 19. Manuscript Language to Use

Potential title:

```text
Automatic Exposure Control Profiles for Second-Stage Reclassification of Low Skeletal Muscle Index Risk on CT
```

Core result sentence:

```text
Although AEC-derived signatures provided only modest improvement in overall discrimination, they meaningfully reclassified patients flagged by the clinical model, reducing false positives and identifying both a very-low-risk de-escalation group and a high-yield priority group in external validation.
```

Methods sentence:

```text
AEC profiles were resampled to 128 anatomically corresponding positions and normalized by each patient's mean AEC value to emphasize relative z-axis modulation patterns rather than absolute exposure level. A scanner-free AEC expert was derived from the 256-position direct AEC curve and evaluated as a second-stage reclassification marker rather than a standalone classifier.
```

Gray-zone sentence:

```text
To avoid a complex weighting function, the primary AEC-assisted analysis used a simple clinical indeterminate zone around the clinical decision threshold, consistent with a reflex-test framework.
```

## 20. Bottom-Line Recommendation

The best path is:

```text
Main manuscript:
clinical model + classic gray-zone/reflex-test AEC reclassification

Supplement:
old Gaussian gate, all-sex hard zone, PCA/logistic additive failures, deep learning attempts
```

The strongest defensible claim:

```text
AEC does not replace clinical modeling and does not dramatically improve AUC.
AEC can reduce false positives and prioritize clinical-positive patients by using acquisition-derived z-axis body attenuation patterns already present in CT acquisition data.
```

