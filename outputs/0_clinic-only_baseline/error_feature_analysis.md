# Error feature analysis: TP/FN/TN/FP

Clinical-only classifier analyzed on internal OOF predictions.

## Group sizes

| group   |   n |
|:--------|----:|
| TN      | 514 |
| FP      | 447 |
| TP      | 117 |
| FN      |  12 |

## Feature means by group

| group   |   age |   height |   weight |   bmi |   smi |   score |
|:--------|------:|---------:|---------:|------:|------:|--------:|
| FN      | 52.5  |   161.8  |    65.12 | 24.72 | 31.55 |    0.04 |
| FP      | 57.03 |   165.22 |    60.31 | 21.95 | 45.97 |    0.17 |
| TN      | 56.4  |   158.48 |    64.17 | 25.4  | 47.28 |    0.03 |
| TP      | 59.74 |   166.96 |    57.86 | 20.62 | 36.5  |    0.32 |

## Sex composition by group (fraction Male)

| group   |   frac_male |
|:--------|------------:|
| FN      |       0.333 |
| FP      |       0.485 |
| TN      |       0.189 |
| TP      |       0.615 |

## Full-data LR coefficients (standardized features: age, height, weight, sex_M)

|           |   coefficient |
|:----------|--------------:|
| age       |        0.7805 |
| height    |        0.38   |
| weight    |        1.5382 |
| sex_M     |       -1.6348 |
| intercept |       -2.9516 |

## Correlation of OOF score with derived BMI / raw features

|        |   correlation_with_score |
|:-------|-------------------------:|
| bmi    |                  -0.6167 |
| height |                   0.4173 |
| weight |                  -0.2492 |

## TP vs FN (among actual low-SMI positives): Welch t-test

| feature   |   TP_mean |   FN_mean |    diff |       t |      p |
|:----------|----------:|----------:|--------:|--------:|-------:|
| age       |   59.735  |   52.5    |  7.235  |  1.2324 | 0.2409 |
| height    |  166.957  |  161.8    |  5.1573 |  1.754  | 0.1045 |
| weight    |   57.8594 |   65.125  | -7.2656 | -1.9577 | 0.0725 |
| bmi       |   20.6244 |   24.7241 | -4.0997 | -4.5037 | 0.0006 |

## TN vs FP (among actual negatives): Welch t-test

| feature   |   TN_mean |   FP_mean |    diff |        t |          p |
|:----------|----------:|----------:|--------:|---------:|-----------:|
| age       |   56.4027 |   57.0313 | -0.6286 |  -0.8235 | 4.1050e-01 |
| height    |  158.479  |  165.224  | -6.7449 | -14.1204 | 3.1831e-41 |
| weight    |   64.1702 |   60.3099 |  3.8603 |   5.8304 | 7.5656e-09 |
| bmi       |   25.4034 |   21.9537 |  3.4497 |  22.0888 | 9.9866e-88 |

## TP vs FP (among model-predicted positives): Welch t-test

| feature   |   TP_mean |   FP_mean |    diff |       t |          p |
|:----------|----------:|----------:|--------:|--------:|-----------:|
| age       |   59.735  |   57.0313 |  2.7037 |  1.8326 | 6.8734e-02 |
| height    |  166.957  |  165.224  |  1.7338 |  2.3224 | 2.1301e-02 |
| weight    |   57.8594 |   60.3099 | -2.4505 | -2.2271 | 2.7244e-02 |
| bmi       |   20.6244 |   21.9537 | -1.3293 | -4.7962 | 3.7189e-06 |

## Sex distribution chi-square: TP vs FN

| group   |   F |   M |
|:--------|----:|----:|
| TP      |  45 |  72 |
| FN      |   8 |   4 |

chi2=2.51, p=0.1134

## Sex distribution chi-square: TN vs FP

| group   |   F |   M |
|:--------|----:|----:|
| TN      | 417 |  97 |
| FP      | 230 | 217 |

chi2=94.36, p=2.6337e-22

## Sex distribution chi-square: TP vs FP

| group   |   F |   M |
|:--------|----:|----:|
| TP      |  45 |  72 |
| FP      | 230 | 217 |

chi2=5.76, p=1.6433e-02
