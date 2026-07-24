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
  — **데이터 유형**: fMRI voxel decoding(시뮬레이션 + 실제 neuroimaging), 자연 이미지나 1D
  신호가 아님. **주의**: 이 논문의 핵심 결론은 confound regression이 안전한 해법이라는 게
  아니라, confound regression이 **음의 편향**(실제보다 성능을 과소평가, 일부 시나리오에서
  chance 이하)을, post-hoc counterbalancing은 반대로 양의 편향을 유발할 수 있다는 경고다.
  본 프로젝트의 slice-wise OLS residualization도 이 편향 위험에서 자유롭지 않을 수 있음 —
  인용 시 "confound regression 정당화" 근거가 아니라 "위험 인지" 근거로 표기할 것.
- Chaibub Neto, E. (2021). Causality-Aware Counterfactual Confounding Adjustment as an
  Alternative to Linear Residualization in Anticausal Prediction Tasks Based on Linear
  Learners. *arXiv:2011.04605* (ICML 2021).
  https://arxiv.org/abs/2011.04605
  — **데이터 유형**: 순수 synthetic 데이터(regression MSE, classification accuracy)만 사용,
  실제 1D 신호·이미지·곡선 예시 없음. Linear residualization의 대안(counterfactual
  adjustment)을 이론적으로 비교하는 논문이라는 점만 인용 근거로 삼을 것.

이 두 문헌은 confound-regression *일반론*(및 그 위험/대안)의 근거일 뿐, AEC-128 같은
**1D 곡선(functional data) 자체**를 covariate로 조정한 논문은 아니다 — 그 직접적 대응은
아래 3절(Li et al. 2017, Wittenberg et al. 2024)이다.

## 2. 잔차(gap)를 그 자체로 다운스트림 바이오마커로 쓰는 접근

- Cole, J. H., Poudel, R. P. K., Tsagkrasoulis, D., et al. (2017). Predicting Brain Age with
  Deep Learning from Raw Imaging Data Results in a Reliable and Heritable Biomarker.
  *NeuroImage*, 163, 115–124.
  https://doi.org/10.1016/j.neuroimage.2017.07.059
- Brain Age Residual Biomarker (BARB): Leveraging MRI-Based Models to Detect Latent Health
  Conditions in U.S. Veterans (2025). *arXiv:2501.05970*.
  https://arxiv.org/abs/2501.05970

## 3. Covariate-adjusted 잔차를 곡선(functional data) 단위로 다루는 연구

**본 프로젝트의 slice-wise OLS residualization(`aec_i ~ CLIN_COLS`, 128 슬라이스 각각 회귀)에
가장 직접적으로 대응하는 두 문헌.**

- Li, P.-L., Chiou, J.-M., & Shyr, Y. (2017). Functional Data Classification Using
  Covariate-Adjusted Subspace Projection. *Computational Statistics & Data Analysis*, 115,
  21–34.
  https://doi.org/10.1016/j.csda.2017.05.007
  — covariate가 response function(곡선)의 **평균 함수(mean function)** 에 함수회귀로 영향을
  준다고 모델링하고, covariate-adjusted mean function을 곡선에서 뺀 잔차(Karhunen–Loève
  기반 서브스페이스)로 분류. 본 프로젝트의 `aec_i ~ CLIN_COLS` 적합 및 `residual = aec_i -
  predicted` 계산과 구조적으로 동일한 절차.
- Wittenberg, P., Neumann, L., Mendler, A., & Gertheiss, J. (2024). Covariate-Adjusted
  Functional Data Analysis for Structural Health Monitoring. *arXiv:2408.02106* (Data-Centric
  Engineering, Cambridge Univ. Press).
  https://arxiv.org/abs/2408.02106
  — 구조물 센서의 1D 시계열(곡선)에서 온도 등 환경 covariate로 인한 변동을 function-on-
  function regression으로 제거하고, 잔차 곡선을 이상탐지 신호로 사용. "정상 covariate
  변동을 제거한 잔차가 이상/신호를 담는다"는 논지가 본 프로젝트의 residualized-AEC-sum
  아이디어(잔차 곡선 합산 스칼라를 재분류 신호로 사용)와 동일.

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
