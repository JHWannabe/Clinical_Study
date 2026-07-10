# model_algorithm.md 관련 선행 연구 자료 조사

`model_algorithm.md`(2단계 low-SMI 분류 모델 설계)와 관련하여 조사한 내부 자료 및 외부 논문
정리. 외부 논문은 2022년 이후 발표분으로 한정.

## 1. 내부 선행 자료 (저장소 밖, `Desktop/2026-1_Study/연구자료/`)

| 파일 | 관련성 | 핵심 내용 |
| --- | --- | --- |
| `260506_이홍선교수님.pdf` | 직접 관련 | AEC curve 기반 low SMI 예측 연구 정리. handcrafted AEC feature는 BMI를 넣으면 추가 효과가 거의 사라진다는 결론(`project_aec_curve_bmi_confound` 메모의 BMI confound 문제의식과 직결). 강남 train / 신촌 external validation 구조, AUC/AUPRC/Brier 평가, incremental value 판정 기준(ΔAUC ≥0.04 강한 결과 등)을 제시. 다음 단계로 raw AEC curve 1D CNN 검토를 제안.
| `AEC_virtual_phenotype_analysis_instructions.pdf` | 직접 관련 | AEC → virtual body-composition phenotype(TAMA/NAMA/SMI/BMI 예측) → clinical + phenotype discordance로 low SMI 예측하는 2-stage 설계 지시서. Cross-fitting(outer 5-fold, inner CV) 원칙, negative control(shuffled AEC), 금지사항(actual TAMA 등 직접 넣지 않기) — `model_algorithm.md`의 "리키지 방지" 절(1차/2차 CV 폴드 일관성)과 유사한 구조.
| `251127_Tabular Data Analysis.pdf` | 간접 참고 | 같은 연구실(Gangnam Severance)의 골다공증 tabular 분류 발표자료. Age/BMI 기반 logistic regression, stratified analysis, SMOTE 등 방법론 참고용. AEC/low SMI와 직접 관련은 없음.
| `CAM.pdf` (Ismail Fawaz et al., *Deep learning for time series classification: a review*, 2019) | 방법론 참고 | ResNet/FCN 등 raw curve 기반 1D CNN 아키텍처, Class Activation Map 해석기법. 이홍선 교수님 메모의 "1D CNN으로 raw AEC curve 넣기" 제안의 아키텍처적 배경.

## 2. 외부 논문 조사 — 문제 자체 ("AEC curve → low SMI")

`model_algorithm.md`의 핵심 아이디어(스캐너 AEC/tube current curve 자체를 low SMI 예측
변수로 사용)와 정확히 일치하는 기존 학술 논문은 검색에서 발견되지 않음. 인접한 세 갈래의
선행 문헌만 존재:

### 2-1. Opportunistic CT sarcopenia screening (이미지/세그멘테이션 기반 — 반대 방향 접근)
- [Implementation of Fully Automated AI-Integrated System for Body Composition Assessment on CT for Opportunistic Sarcopenia Screening](https://formative.jmir.org/2025/1/e69940) (JMIR Form Res, 2025)
- [End-to-end automated body composition analyses with integrated quality control for opportunistic assessment of sarcopenia in CT](https://link.springer.com/article/10.1007/s00330-021-08313-x) (Eur Radiol, 2021)

→ 실제 CT 영상을 세그멘테이션해서 TAMA/SMI를 구하는 방식. `model_algorithm.md`가 명시적으로
배제하는 "L3 image 정보 사용"과 반대 방향이라 대조군으로만 인용 가치.

### 2-2. AEC / Tube Current Modulation 기술 문헌 (선량 최적화 목적, 근육량 예측과 무관)
- [Patient-specific radiation risk-based tube current modulation for diagnostic CT](https://pubmed.ncbi.nlm.nih.gov/35421263/) (Med Phys, 2022)
- [Introduction to CT Automatic Exposure Control (Mayo Clinic)](https://www.mayo.edu/research/documents/care-dose-4d-ct-automatic-exposure-control-system/doc-20086815)

→ AEC/TCM이 체형(감쇠)에 반응한다는 공학적 근거는 있으나, 이를 virtual body-composition
phenotype으로 재구성해 low SMI를 예측한다는 논문은 없음 — 이 gap이 `model_algorithm.md`
접근의 novelty.

### 2-3. Clinical/anthropometric-only SMI 예측식 (baseline 비교 대상)
- [Development and Validation of a Skeletal Muscle Prediction Equation From Anthropometric and Demographic Data](https://www.jamda.com/article/S1525-8610(25)00582-1/fulltext) (J Am Med Dir Assoc, 2025)
- [Development of Formulas for Calculating L3 Skeletal Muscle Mass Index and Visceral Fat Area Based on Anthropometric Parameters](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249379/) (Front Nutr, 2022)

→ Age/Sex/Weight/Height만으로 SMI를 예측하는 공식들 — `model_algorithm.md`의 clinical-only
baseline(1차 스크리닝)과 같은 역할을 하는 선행 연구군.

## 3. 외부 논문 조사 — 방법론 (곡선 featurizer: FPCA / band / cluster_band / combo)

### 3-1. 전역 Functional PCA (FPCA)
- [Functional classwise principal component analysis: a classification framework for functional data analysis](https://arxiv.org/pdf/2106.13959) (Data Min Knowl Discov, 2022) — 클래스별 FPCA 기저로 곡선을 분류에 특화된 저차원 피처로 요약. `stage2_aec_residual_reclassify.py`의 "분산 90/95%, k=4~6" FPCA 접근과 직접 비교 가능.
- [Functional Data Analysis: An Introduction and Recent Developments](https://onlinelibrary.wiley.com/doi/full/10.1002/bimj.202300363) (Biometrical Journal, 2024) — FDA/FPCA 최신 리뷰.
- [Multilevel Longitudinal Functional Principal Component Model](https://onlinelibrary.wiley.com/doi/10.1002/sim.10207) (Statistics in Medicine, 2024) — 생체 센서 곡선에 FPCA 적용 실전 사례.
- [Graphical Principal Component Analysis of Multivariate Functional Time Series](https://www.tandfonline.com/doi/full/10.1080/01621459.2024.2302198) (JASA, 2024) — 다변량 함수형 시계열 PCA 확장.

### 3-2. 데이터 기반 인접 구간 클러스터링 (cluster_band에 해당)
- [Review of Clustering Methods for Functional Data](https://dl.acm.org/doi/10.1145/3581789) — Zhang & Parnell, ACM TKDD, 2023. "유한차원 변환 후 클러스터링" vs "무한차원(곡선 자체) 클러스터링"으로 방법론을 체계화한 최신 서베이. `cluster_band` 설계 근거의 정식 인용처로 적합.
- [Distance-based Clustering of Functional Data with Derivative Principal Component Analysis](https://www.tandfonline.com/doi/full/10.1080/10618600.2024.2366499) (J Comput Graph Stat, 2024) — 곡선 형태 정보를 보존하는 클러스터링 + PCA 결합.
- [Penalized model-based clustering of complex functional data](https://link.springer.com/article/10.1007/s11222-023-10288-2) (Statistics and Computing, 2023).
- [Clustering multivariate functional data using the epigraph and hypograph indices](https://arxiv.org/pdf/2307.16720) (Statistics and Computing, 2023, Pulido/Franco-Pereira/Lillo) — cluster_band의 대안적 접근.

참고: `cluster_band`가 실제로 사용하는 "인접 slice만 병합되도록 chain-graph connectivity로
제약한 Ward 클러스터링" 알고리즘 자체는 adjacency-constrained hierarchical clustering
(Ward, ARXIV:1902.01596 / R 패키지 `adjclust`, 2019)에서 처음 제안되었으나 2022년 이전
발표라 위 목록에서는 제외. 위의 2023 Zhang & Parnell 리뷰가 이를 포괄하는 최신 분류체계를
제공하므로 정식 인용 시 함께 참고할 것.

### 3-3. Band + PCA 결합 (combo)
- [Revisiting PCA for Time Series Reduction in Temporal Dimension](https://arxiv.org/pdf/2412.19423) (arXiv, 2024) — 시계열 분류·회귀에서 PCA 기반 시간축 축소가 다운샘플링/1D-CNN 축소층보다 우수함을 실증. combo 방식(band+PCA)이 PCA 단독보다 나을 수 있다는 관찰과 같은 방향.
- [Segmentation over Complexity: Evaluating Ensemble and Hybrid Approaches for Anomaly Detection in Industrial Time Series](https://arxiv.org/html/2510.26159) (arXiv, 2025) — 세그먼트 기반 피처 + PCA/트리 앙상블 하이브리드의 최신 실증.

## 4. 종합

- `model_algorithm.md`가 다루는 문제(AEC curve → virtual phenotype → low SMI, BMI confound
  제거)는 출판된 논문으로는 아직 다뤄지지 않은 조합이며, 내부 미팅 자료(이홍선 교수님
  정리본, 분석 지시서)가 가장 가까운 "선행 연구"임. 외부 논문은 개별 구성요소(AEC 공학,
  opportunistic 세그멘테이션, anthropometric SMI 공식)에 대한 background로만 활용 가능.
- 곡선 featurizer 방법론 중 `cluster_band`는 이름 있는 기존 방법론(adjacency-constrained
  hierarchical/Ward clustering)과 정확히 일치 — 2023 Zhang & Parnell 리뷰를 정식 인용처로
  사용 가능. FPCA 단독이 band/cluster_band보다 못한 이유(국소 신호가 전역 성분에 희석됨)도
  2022~2024년 FDA 문헌들이 공통적으로 지적하는 한계와 일치.

## 참고 메모
- `model_algorithm` — 본 조사의 대상 문서
- `feedback_no_legacy_pipeline_reuse`
- `project_aec_curve_bmi_confound`
