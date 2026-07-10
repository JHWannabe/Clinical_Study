# Error feature analysis: TP/FN/TN/FP

Clinical-only classifier (`new_hypothesis.py`) analyzed on internal OOF predictions.

## Group sizes

| group   |   n |
|:--------|----:|
| TN      | 515 |
| FP      | 446 |
| TP      | 117 |
| FN      |  12 |

## Feature means by group

| group   |   age |   height |   weight |   bmi |   smi |   score |
|:--------|------:|---------:|---------:|------:|------:|--------:|
| FN      | 52.5  |   161.8  |    65.12 | 24.72 | 31.55 |   -3.48 |
| FP      | 56.95 |   165.26 |    60.35 | 21.96 | 46.03 |   -1.75 |
| TN      | 56.48 |   158.46 |    64.13 | 25.4  | 47.23 |   -3.86 |
| TP      | 59.74 |   166.96 |    57.86 | 20.62 | 36.5  |   -0.88 |

## Sex composition by group (fraction Male)

| group   |   frac_male |
|:--------|------------:|
| FN      |       0.333 |
| FP      |       0.489 |
| TN      |       0.186 |
| TP      |       0.615 |

## Full-data LR coefficients (standardized features: age, height, weight, sex_M)

|           |   coefficient |
|:----------|--------------:|
| age       |        0.3709 |
| height    |        1.5221 |
| weight    |       -1.6415 |
| sex_M     |        0.4019 |
| intercept |       -2.6765 |

## Correlation of OOF score with derived BMI / raw features

|        |   correlation_with_score |
|:-------|-------------------------:|
| bmi    |                  -0.7764 |
| height |                   0.4766 |
| weight |                  -0.3204 |

## TP vs FN (among actual low-SMI positives): Welch t-test

| feature   |   TP_mean |   FN_mean |   diff |     t |      p |
|:----------|----------:|----------:|-------:|------:|-------:|
| age       |     59.74 |     52.5  |   7.24 |  1.23 | 0.2409 |
| height    |    166.96 |    161.8  |   5.16 |  1.75 | 0.1045 |
| weight    |     57.86 |     65.12 |  -7.27 | -1.96 | 0.0725 |
| bmi       |     20.62 |     24.72 |  -4.1  | -4.5  | 0.0006 |

## TN vs FP (among actual negatives): Welch t-test

| feature   |   TN_mean |   FP_mean |   diff |      t |      p |
|:----------|----------:|----------:|-------:|-------:|-------:|
| age       |     56.48 |     56.95 |  -0.47 |  -0.61 | 0.5398 |
| height    |    158.46 |    165.26 |  -6.8  | -14.26 | 0      |
| weight    |     64.13 |     60.35 |   3.79 |   5.71 | 0      |
| bmi       |     25.4  |     21.96 |   3.44 |  22    | 0      |

## Sex distribution chi-square: TP vs FN

| group   |   F |   M |
|:--------|----:|----:|
| FN      |   8 |   4 |
| TP      |  45 |  72 |

chi2=2.51, p=0.1134

## Sex distribution chi-square: TN vs FP

| group   |   F |   M |
|:--------|----:|----:|
| FP      | 228 | 218 |
| TN      | 419 |  96 |

chi2=97.97, p=0.0000