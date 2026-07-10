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

### 3-4. 평가 방법론 — 비열등성(Non-Inferiority) 검정 및 재분류개선(NRI)

두 `stage2_*.py` 스크립트 모두 임계값(th2) 선택과 최종 채택/기각 판정을
`noninferiority_test_sensitivity()`(Newcombe Method 10 기반 페어드 비율 차이 CI)로
전환했다 — `model_algorithm.md`의 "임계값 선택 기준" 절 및
[residual_reclassify_algorithm.md](residual_reclassify_algorithm.md) 4.5.1절 참고. 2022년
이후 문헌은 검색되지 않아 원 논문(1998년, 2022년 이전 발표)을 직접 인용하되, 최신 적용 사례로
아래를 함께 참고한다.

- Newcombe RG. *Interval estimation for the difference between independent proportions:
  comparison of eleven methods.* Stat Med. 1998 — "Method 10"(paired proportions, score-based
  CI)의 원 출처. 2022년 이전 발표라 본 조사의 연도 기준(2022년 이후)에는 포함되지 않지만,
  구현이 이 방법을 정확히 따르므로 정식 인용처로 필요.
- [Comparison of the sensitivity and specificity of two diagnostic tests: paired-sample
  confidence intervals](https://pmc.ncbi.nlm.nih.gov/articles/PMC10039285/) (BMC Med Res
  Methodol, 2023) — 동일 환자에 대한 두 진단 검사(여기서는 Stage-1-only vs Stage-1+Stage-2)의
  민감도/특이도 차이를 paired 신뢰구간으로 비교하는 최신 방법론 리뷰. Newcombe Method 10을
  포함한 여러 접근을 비교하며, `noninferiority_test_sensitivity`가 다루는 문제(같은 환자 집합에
  대한 전/후 민감도 비교)와 정확히 일치.
- [Net Reclassification Improvement (NRI): a graphical approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC10022374/)
  (Diagn Progn Res, 2023) — `plot_clinical_vs_aec_table`가 계산하는 Net NRI(특이도 개선 flip
  수 - 민감도 악화 flip 수)의 표준 정의와 해석 가이드. Stage-2가 Stage-1-negative를 positive로
  뒤집지 않는 단방향 구조(By design c=0)이므로, 일반적인 양방향 NRI 공식의 특수 케이스에 해당.
- [Statistical guidance for reporting studies evaluating diagnostic tests or prediction
  models](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02184-9)
  (BMC Med Res Methodol, 2024) — 진단/예측 모델의 비열등성·재분류 지표 보고 시 권고되는
  통계적 관행 일반 가이드.

이 방법론 전환의 실질적 효과: 점추정치 기준 acceptance criteria(민감도 하락 ≤5%p)에서
CI 기반 비열등성 검정으로 바뀌면서, `model_algorithm.md` "구현 파일별 차이" 절에 기록된
채택 spec_delta 수치가 구버전(점추정치 기준) 대비 낮아졌다 — 더 보수적인 기준으로 모델/임계값을
선택하기 때문이며, 두 방법 모두 통계적으로 정당하지만 CI 기반 쪽이 표본 크기에 따른 불확실성을
명시적으로 반영한다는 점에서 더 엄격하다.

## 4. 종합

- `model_algorithm.md`가 다루는 문제(AEC curve → virtual phenotype → low SMI, BMI confound
  제거)는 출판된 논문으로는 아직 다뤄지지 않은 조합이며, 내부 미팅 자료(이홍선 교수님
  정리본, 분석 지시서)가 가장 가까운 "선행 연구"임. 외부 논문은 개별 구성요소(AEC 공학,
  opportunistic 세그멘테이션, anthropometric SMI 공식)에 대한 background로만 활용 가능.
- 곡선 featurizer 방법론 중 `cluster_band`는 이름 있는 기존 방법론(adjacency-constrained
  hierarchical/Ward clustering)과 정확히 일치 — 2023 Zhang & Parnell 리뷰를 정식 인용처로
  사용 가능. FPCA 단독이 band/cluster_band보다 못한 이유(국소 신호가 전역 성분에 희석됨)도
  2022~2024년 FDA 문헌들이 공통적으로 지적하는 한계와 일치.
- 평가 방법론(3-4절)은 원 논문(Newcombe 1998) 자체는 2022년 이전이지만, paired 민감도/특이도
  비교와 NRI 보고에 대한 2023~2024년 방법론 문헌이 구현 방식(페어드 CI, 단방향 flip 구조,
  Net NRI)을 그대로 뒷받침한다 — acceptance criteria를 점추정치에서 CI 기반으로 강화한 최근
  변경의 통계적 근거로 활용 가능.

## 참고 메모
- `model_algorithm` — 본 조사의 대상 문서
- `feedback_no_legacy_pipeline_reuse`
- `project_aec_curve_bmi_confound`
