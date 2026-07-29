# TP vs FP: TotalSegmentator body-composition features (baseline, internal cohort)

Clinic-only baseline (Se>=90%) TP (n=117) vs FP (n=447), features from gangnam_totalseg.xlsx.
Welch's t-test (unequal variance) and Mann-Whitney U, 95% CI on TP-FP mean difference, Cohen's d.

## Two circularity checks

**1) Label circularity.** The label is SMI = TAMA / Height^2, so a feature correlated with TAMA itself (full n=1090 cohort) separates TP from FP largely by construction. Flagged at |r(TAMA, feature)|>0.4:

- Label-circular: NAMA_sum_cm2, LAMA_sum_cm2, 내장지방_sum_cm2, 총근육량
- Independent of TAMA: IMATA_sum_cm2, 피하지방_sum_cm2, 총지방량

`총근육량` = `NAMA_sum_cm2 + LAMA_sum_cm2 + IMATA_sum_cm2` exactly (verified, max abs diff ~1e-12) -- it is the whole-scan analogue of TAMA.

**2) Classifier-input circularity.** TP and FP are both baseline "predicted positive", but they still differ on the classifier's own inputs themselves (weight/height/age/sex; e.g. TP mean weight 57.9kg vs FP 60.3kg, p=0.027) -- per [[feedback_no_circular_restratification]]. A feature merely correlated with those inputs (e.g. subcutaneous fat vs weight, r=0.19) can show a raw TP/FP gap driven by that, not by anything specific to TP/FP. `p_adj_weight_height_age_sex` / `p_adj_mwu` are the TP/FP test p-values on residuals after regressing each feature on weight+height+age+sex_M.

- Survives adjustment (independent of TAMA **and** p_adj<0.05): none
- Everything else is either label-circular, classifier-input-circular, or not significant to begin with -- raw p-values for those should not be read as new findings about FP patients.

| feature          |   n_TP |   n_FP |   TP_mean |     TP_sd |   FP_mean |    FP_sd |   diff_TP_minus_FP |   ci95_lower |   ci95_upper |      t |          p |      p_mwu |   cohens_d |   corr_with_TAMA |   p_adj_weight_height_age_sex |   p_adj_mwu | label_circular   | survives_adjustment   |
|:-----------------|-------:|-------:|----------:|----------:|----------:|---------:|-------------------:|-------------:|-------------:|-------:|-----------:|-----------:|-----------:|-----------------:|------------------------------:|------------:|:-----------------|:----------------------|
| IMATA_sum_cm2    |    117 |    447 |   1080.08 |   549.958 |   1074.27 |  484.428 |              5.813 |     -103.492 |      115.119 |  0.104 | 0.9171     | 0.61779    |      0.012 |            0.182 |                    0.32292    |  0.19099    | no               | no                    |
| NAMA_sum_cm2     |    117 |    447 |  12865.4  |  3818.39  |  15290.7  | 4222.12  |          -2425.34  |    -3220.28  |    -1630.4   | -5.98  | 1.0341e-08 | 7.3786e-08 |     -0.586 |            0.802 |                    1.7082e-12 |  2.4619e-16 | yes              | yes                   |
| LAMA_sum_cm2     |    117 |    447 |   3715.39 |  1768.81  |   3996.08 | 1935.82  |           -280.682 |     -648.016 |       86.651 | -1.498 | 0.13584    | 0.092991   |     -0.148 |            0.451 |                    0.45626    |  0.67405    | yes              | no                    |
| 내장지방_sum_cm2 |    117 |    447 |   7656.71 |  5721.75  |   7845.58 | 5186.58  |           -188.862 |    -1331.72  |      953.998 | -0.324 | 0.74642    | 0.36856    |     -0.036 |            0.445 |                    0.11925    |  0.058741   | yes              | no                    |
| 피하지방_sum_cm2 |    117 |    447 |  14390.4  |  6378.25  |  16366.7  | 5825.25  |          -1976.3   |    -3251.99  |     -700.601 | -3.036 | 0.0027711  | 0.010394   |     -0.333 |           -0.047 |                    0.12637    |  0.017819   | no               | no                    |
| 총근육량         |    117 |    447 |  17660.8  |  4584.62  |  20361    | 5195.84  |          -2700.21  |    -3660.49  |    -1739.92  | -5.511 | 1.0831e-07 | 1.9818e-06 |     -0.532 |            0.841 |                    2.0268e-11 |  1.5193e-15 | yes              | yes                   |
| 총지방량         |    117 |    447 |  22047.1  | 10678.8   |  24212.3  | 9187.41  |          -2165.16  |    -4279.32  |      -50.989 | -2.007 | 0.046366   | 0.040419   |     -0.228 |            0.17  |                    0.083213   |  0.019547   | no               | no                    |
