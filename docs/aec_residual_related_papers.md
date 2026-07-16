# AEC 잔차화(residual) 관련 참고 문헌 목록

[cnn_variant_comparison.md](cnn_variant_comparison.md#2-1-공통-입력-표현aec-128-잔차-곡선-관련-참고-문헌) 2-1절에 실은 표를 리스트 형태로
정리한 것. AEC(자동노출제어) 곡선 자체를 covariate 잔차화해 재분류에 쓴 선행연구는 찾지
못했다 — 아래는 "covariate를 회귀로 제거한 잔차를 1D 신호/곡선 단위로 다운스트림 모델에
쓰는" 같은 방법론을 다른 도메인·다른 신호에 적용한 유사 사례, 그리고 이 프로젝트가 쓰는
전처리 단계(4.2절)와 다른 대안적 전처리 기법 문헌이다.

## 1. Covariate 잔차화(linear residualization) 방법론

- Snoek, L., Miletić, S., & Scholte, H. S. (2019). How to Control for Confounds in Decoding
  Analyses of Neuroimaging Data. *NeuroImage*, 184, 741–760.
  https://doi.org/10.1016/j.neuroimage.2018.10.024
- Chaibub Neto, E. (2021). Causality-Aware Counterfactual Confounding Adjustment as an
  Alternative to Linear Residualization in Anticausal Prediction Tasks Based on Linear
  Learners. *arXiv:2011.04605* (ICML 2021).
  https://arxiv.org/abs/2011.04605

## 2. 잔차(gap)를 그 자체로 다운스트림 바이오마커로 쓰는 접근

- Cole, J. H., Poudel, R. P. K., Tsagkrasoulis, D., et al. (2017). Predicting Brain Age with
  Deep Learning from Raw Imaging Data Results in a Reliable and Heritable Biomarker.
  *NeuroImage*, 163, 115–124.
  https://doi.org/10.1016/j.neuroimage.2017.07.059
- Brain Age Residual Biomarker (BARB): Leveraging MRI-Based Models to Detect Latent Health
  Conditions in U.S. Veterans (2025). *arXiv:2501.05970*.
  https://arxiv.org/abs/2501.05970

## 3. Covariate-adjusted 잔차를 곡선(functional data) 단위로 다루는 연구

- Li, P.-L., Chiou, J.-M., & Shyr, Y. (2017). Functional Data Classification Using
  Covariate-Adjusted Subspace Projection. *Computational Statistics & Data Analysis*, 115,
  21–34.
  https://doi.org/10.1016/j.csda.2017.05.007
- Covariate-Adjusted Functional Data Analysis for Structural Health Monitoring (2024).
  *arXiv:2408.02106* (Data-Centric Engineering, Cambridge Univ. Press).
  https://arxiv.org/abs/2408.02106

## 4. AEC가 아닌 다른 1D 신호에 쓰이는 대안적 전처리(스케일/베이스라인 보정)

- Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by
  Simplified Least Squares Procedures. *Analytical Chemistry*, 36(8), 1627–1639.
  https://doi.org/10.1021/ac60214a047
- Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard Normal Variate
  Transformation and De-Trending of Near-Infrared Diffuse Reflectance Spectra. *Applied
  Spectroscopy*, 43(5), 772–777.
  https://doi.org/10.1366/0003702894202201
- Yan, C. (2025). A Review on Spectral Data Preprocessing Techniques for Machine Learning
  and Quantitative Analysis. *iScience*, 28(7), 112759.
  https://doi.org/10.1016/j.isci.2025.112759

## 5. BMI 교란(Simpson's paradox) 제거의 통계적 근거

- Simpson, E. H. (1951). The Interpretation of Interaction in Contingency Tables. *Journal
  of the Royal Statistical Society: Series B*, 13(2), 238–241.
  https://doi.org/10.1111/j.2517-6161.1951.tb00088.x
- Norton, H. J., & Divine, G. (2015). Simpson's Paradox … and How to Avoid It.
  *Significance*, 12(4), 40–43.
  https://doi.org/10.1111/j.1740-9713.2015.00844.x
