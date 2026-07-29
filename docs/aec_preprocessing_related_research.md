# AEC-128 전처리(raw/patient-wise/global z-score) 관련 레퍼런스

AEC-128(`aec_1`...`aec_128`, `data/{gangnam,sinchon}.xlsx`의 `aec_128`/`aec_cropped`
시트)은 **CT 스캐너의 Automatic Exposure Control(AEC)이 z축(슬라이스 위치)을 따라
조절한 관전류(tube current, mA) 변조 곡선**이다 — BIA/EIM 전기임피던스도, 초음파도
아니다. `Series_Desc`/`Manufacturer`(SOMATOM Definition AS+, Ingenuity Core 128,
Revolution CT 등 실제 CT 기종명)/`z_range`/`n_slices_cropped` 컬럼이 DICOM
메타데이터임을 근거로 2026-07-27 세션에서 확인(사용자 확인 완료). 원본은 환자마다
슬라이스 수가 다르며(`n_slices_cropped` 121~154 등) 이를 128포인트로 리샘플링한
것이 `aec_1..128`이다. 자세한 배경은 `[[project_aec_signal_definition]]` 메모리
참고.

## A. CT AEC/TCM(Automatic Exposure Control / Tube Current Modulation) 도메인 기초 문헌

| 문헌 | 핵심 내용 |
| --- | --- |
| [Kalra MK, Maher MM, Toth TL, et al. *Comparison of Z-axis automatic tube current modulation technique with fixed tube current CT scanning of abdomen and pelvis.* Radiology. 2004;232:347-353.](https://pubmed.ncbi.nlm.nih.gov/15286306/) | z축 AEC/TCM 기법의 원조 임상 검증 논문 |
| [McCollough CH, Bruesewitz MR, Kofler JM Jr. *CT dose reduction and dose management tools: overview of available options.* RadioGraphics. 2006;26(2):503-512.](https://pubs.rsna.org/doi/10.1148/rg.262055138) | AEC/TCM을 포함한 CT 선량관리 기법 전반 리뷰 (foundational) |
| [McCollough CH et al. *Techniques and Applications of Automatic Tube Current Modulation for CT.* Radiology. 2006;240(3):611-622.](https://pubs.rsna.org/doi/abs/10.1148/radiol.2333031150) | AEC/TCM의 원리·구현 방식(각도별/z축별 변조) 총정리 |

## B. TCM mA 곡선의 patient-wise 정규화 — CT 도메인 자체의 전례 (가장 직접 관련)

| 문헌 | Raw mA(또는 CTDIvol) 취급 | Patient-wise에 대응하는 방법 | Global에 대응하는 부분 | 핵심 수치/결론 |
|---|---|---|---|---|
| [Li, Yang, Liu 2017. Med Phys. 44:5413-5422.](https://pubmed.ncbi.nlm.nih.gov/28681439/) | Raw mA(z) 곡선 형태 자체를 문제 삼음 — "DL(z)(실제 선량 종방향 분포)의 형태가 mA(z) 곡선 형태와 크게 다르다"(30cm 수조 기준) | 명시적 정규화 알고리즘은 제시하지 않음 — "water equivalent diameter + 임상 mA 곡선으로 DL(z)를 평가해야 한다"고 필요성만 논증 | 41개 수조 샘플에서 CV 5.5~70.0% 산포만 보고, population 모델 없음 | raw mA 곡선을 그대로 곡선비교/특징추출에 쓰면 안 된다는 반증(counter-example) 논문 — 귀하 프로젝트의 raw AEC 조건이 왜 약한 신호(shape n.s., level만 유의)로 나왔는지의 이론적 배경으로 인용 가능 |
| [Bostani, McMillan et al. 2015. Med Phys. 42(2):958-968.](https://pubmed.ncbi.nlm.nih.gov/25652508/) | Raw 관전류를 effective diameter(기하학적 단면 지름, 감쇠 무시)로만 정규화하던 기존 관행의 한계 지적 | Water-equivalent diameter(조직 감쇠 반영) 기반 국소(regional) 정규화 제안 — 스캔 범위 내 "local/regional 평균" vs "middle-slice 단일값" vs "전체 스캔 평균" 3가지 정규화 단위를 비교 | "전체 스캔 범위 평균(global)"과 "중간 단면(middle-slice, 단일 대표값)"을 대조군으로 명시 | Regional(=patient-wise, 위치별) 정규화가 middle-slice/global보다 장기선량 예측력이 유의하게 높음(특히 흉부 — 폐 저감쇠로 위치별 편차 큼) — 곡선 전체를 patient-wise로 정규화하는 게 대표값 하나로 축약하는 것보다 낫다는 직접 증거 |
| [Boone et al. AAPM Report No. 204, 2011.](https://www.aapm.org/pubs/reports/rpt_220.pdf) | Raw CTDIvol = 스캐너 팬텀(32cm/16cm) 출력값, 환자 실제 크기 무시 | SSDE = CTDIvol × f(size) — f(size)는 환자 개인의 water-equivalent diameter를 입력받아 patient-wise로 적용되는 스케일링 계수 | f(size) 함수 자체는 Monte Carlo 시뮬레이션으로 도출된 population-level 회귀식(모든 환자에 동일 함수 적용) | 하이브리드 구조: "함수는 global(모집단에서 1회 추정), 입력은 patient-wise(개별 환자 크기)" — 전체 데이터로 추정한 정규화 함수를 환자별 공변량에 적용하는 제3의 방식을 시도할 근거 |
| [Li, Marschall, Yang, Liu 2022. Med Phys. 49(2):1303-1311.](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.15402) | DICOM 헤더에서 kV, mA, CTDIvol을 슬라이스별로 그대로 추출(raw 출발점은 귀하 파이프라인과 동일) | 환자별 water-equivalent diameter DW(z)를 슬라이스마다 계산해 SSDE(z) 산출 — position-wise + patient-wise 이중 정규화. 65명 환자, 흉부/복부-골반 CT, Monte Carlo로 검증 | 스캔 범위 평균(scan-range average) SSDE를 global 대조 지표로 병기 | 귀하 파이프라인과 데이터 흐름이 구조적으로 가장 유사(DICOM 헤더 → 슬라이스별 raw 값 → 환자별 크기로 슬라이스별 정규화 → 곡선). patient-wise 정규화를 슬라이스 레벨에서 실제 구현하는 재현 가능한 레시피 |

**종합 판단**: raw만 쓰지 말라는 근거는 Li 2017(mA 곡선 형태와 실제 물리량(dose)
형태가 불일치, 환자 크기 커질수록 왜곡 증가), patient-wise가 global/middle-slice보다
낫다는 근거는 Bostani 2015(regional 정규화가 통계적으로 유의하게 우수), patient-wise를
"무엇으로 나눌지"의 표준은 AAPM 204(water-equivalent diameter), 슬라이스별
patient-wise 정규화의 재현 가능한 구현 예시는 Li 2022(귀하 파이프라인과 거의 1:1 대응).

## C. TCM/mA z-profile을 ML/DL 입력으로 쓴 선례

- [Medrano MJ et al. *Scout-Dose-TCM: Direct and Prospective Scout-Based Estimation of Personalized Organ Doses from Tube Current Modulated CT Exams.* arXiv:2506.24062, 2025.](https://arxiv.org/abs/2506.24062) — scout 이미지 + TCM 곡선을 함께 모델 입력으로 사용, DCT(discrete cosine transform) basis function으로 곡선을 재표현 — raw/patient-wise/global 3종 외의 대안 전처리 사례(정규화 디테일은 원문 확인 필요)
- [*Role of Machine Learning-Based CT Body Composition in Risk Prediction and Prognostication: Current State and Future Directions.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10000509/) — CT 영상 기반 근육량(SMA/SMI 등) 자동 산출 ML 적용 리뷰(참고용, TCM 곡선 자체는 다루지 않음)

## D. 방법론 자체(modality 무관) — raw/patient-wise/global 프레임워크의 일반론

- [Apicella A, Isgrò F, Pollastro A, Prevete R. *On the Effects of Data Normalisation for Domain Adaptation on EEG Data.* Engineering Applications of Artificial Intelligence, 2023.](https://arxiv.org/abs/2210.01081) — raw/subject-wise/all-subject 정규화 비교, 정규화 스키마 선택만으로 도메인 적응 기법을 능가하는 경우 다수 관찰 — Gangnam↔Sinchon 코호트 문제에 적용 가능
- [Apicella A et al. *Toward cross-subject and cross-session generalization in EEG-based emotion recognition: Systematic review, taxonomy, and methods.* Neurocomputing, 2024.](https://arxiv.org/abs/2212.08744) — raw/subject-wise/all-subject 정규화 taxonomy 리뷰
- [*Cross-Subject EEG-Based Emotion Recognition Through Neural Networks With Stratified Normalization.* Frontiers in Neuroscience, 2021.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.626277/full) — subject-wise와 population-wise 중간(계층화) 정규화
- [Barnes RJ, Dhanoa MS, Lister SJ. *Standard Normal Variate Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra.* Appl Spectrosc. 1989;43(5):772-777.](https://journals.sagepub.com/doi/10.1366/0003702894202201) — SNV 원 논문(=patient-wise 정규화의 수학적 원형)
- [Geladi P, MacDougall D, Martens H. *Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat.* Appl Spectrosc. 1985;39(3):491-500.](https://journals.sagepub.com/doi/10.1366/0003702854248656) — MSC 원 논문(=global reference 정규화의 원형)
- [Rinnan Å, van den Berg F, Engelsen SB. *Review of the most common pre-processing techniques for near-infrared spectra.* TrAC Trends Anal Chem. 2009;28(10):1201-1222.](https://www.sciencedirect.com/science/article/abs/pii/S0165993609001629) — raw/SNV/MSC/derivative/baseline correction/smoothing 총정리 리뷰
- [Johnson WE, Li C, Rabinovic A. *Adjusting batch effects in microarray expression data using empirical Bayes methods.* Biostatistics. 2007;8(1):118-127.](https://academic.oup.com/biostatistics/article/8/1/118/252073) — ComBat 원 논문. 스캐너 제조사(Siemens/Philips/GE)별 systematic offset 보정에 적용 가능 — `Manufacturer` 컬럼과 직접 대응
- [Ramsay JO, Silverman BW. *Functional Data Analysis.* 2nd ed. Springer, 2005.](https://link.springer.com/book/10.1007/b98888) / [Ramsay JO. *Curve registration.* J R Stat Soc Series B. 1998;60(2):351-363.](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00129) — 곡선을 포인트별이 아닌 전체 단위로 다루는 통계 프레임워크([[feedback_aec_curve_wholistic]])와 정합
- [Ulyanov D, Vedaldi A, Lempitsky V. *Instance Normalization: The Missing Ingredient for Fast Stylization.* arXiv:1607.08022, 2016.](https://arxiv.org/abs/1607.08022) — patient-wise 정규화의 CNN-레이어 버전
- [Ioffe S, Szegedy C. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML, 2015 (arXiv:1502.03167).](https://arxiv.org/abs/1502.03167) — global z-score의 CNN-레이어 버전

## E. 곡선 단위 처리를 위한 딥러닝 아키텍처 레퍼런스 (모달리티 무관, 2026-07-27 추가)

"공간축 1D 신호에 대한 딥러닝 전처리 방법"을 조사한 결과. A-D의 CT AEC/TCM
도메인 문헌과 별개로, **아키텍처 자체**(베이스라인 보정, 정합, 증강)가 모달리티에
무관하게 적용 가능한 레퍼런스만 선별했다. BIA/EIM 진단 특이적 문헌(EIM 임상
논문 등)은 AEC의 실제 도메인이 CT TCM으로 확정된 이상 오도메인이라 제외.

### E1. 베이스라인 보정 / 노이즈 제거 (Denoising Autoencoder 계열)

| 문헌 | 핵심 내용 |
| --- | --- |
| [*Automatic Baseline Correction of 1D Signals Using a Parameter-Free Deep Convolutional Autoencoder Algorithm.* Appl Sci. 2025;15(22):12069.](https://doi.org/10.3390/app152212069) | 1D CNN autoencoder로 파라미터 튜닝 없이 베이스라인(드리프트) 추정 — mA 곡선의 스캐너별 오프셋/드리프트 보정에 참고 가능 |
| [*Deep learning baseline correction method via multi-scale analysis and regression.* Chemometrics and Intelligent Laboratory Systems. 2023;234:104768.](https://www.sciencedirect.com/science/article/abs/pii/S0169743923000291) | Coarse-to-fine 멀티스케일 회귀로 베이스라인 추정, 전통적 asymmetric least squares 대비 비교 |
| [*Denoising and Baseline Correction of Low-Scan FTIR Spectra: A Benchmark of Deep Learning Models Against Traditional Signal Processing.* Bioengineering. 2026;13(3):347 (arXiv:2601.20905).](https://www.mdpi.com/2306-5354/13/3/347) | CNN/U-Net/Transformer 계열을 SNR·MAE 기준 벤치마크 — 저SNR·저해상도 1D 신호(귀하의 128포인트 곡선)에 어떤 구조가 유리한지 정량 근거 |

### E2. 정규화 아키텍처 리뷰 (SNV/MSC 등 전통 기법의 DL 대응)

| 문헌 | 핵심 내용 |
| --- | --- |
| [*A review on spectral data preprocessing techniques for machine learning and quantitative analysis.* iScience. 2025;28(X) (S2589-0042(25)01020-X).](https://www.cell.com/iscience/fulltext/S2589-0042(25)01020-X) | SNV/MSC/min-max/batch norm을 체계적으로 비교한 최신 리뷰 |
| [Zhang G, Abdulla W. *Optimizing Hyperspectral Imaging Classification Performance with CNN and Batch Normalization.* J Spectral Imaging. 2023.](https://journals.sagepub.com/doi/10.1177/27551857231204622) | Batch Norm이 SNV/min-max 대비 분류 성능 변동성을 유의하게 낮춤을 정량 보고 |

### E3. 곡선 정합 (Curve Registration) — Section D의 Ramsay 통계 프레임워크의 DL 구현

| 문헌 | 핵심 내용 |
| --- | --- |
| [*DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification.* arXiv:2501.18116, 2025.](https://arxiv.org/pdf/2501.18116) | Elastic warping(위상 정합)과 분류를 동시 학습. Amplitude(값 크기) vs Phase(피크 위치) 변동 분리 — 환자 간 z축 스캔 범위/시작점 차이로 인한 phase 변동 보정에 이론적으로 적합 |
| [*Deep Learning of Warping Functions for Shape Analysis.* PMC7520101.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7520101/) | Fisher-Rao metric 기반 elastic registration을 DL로 근사, 1D 함수 기준 약 3000배 속도 향상 |
| [*Time-warping analysis for biological signals: methodology and application.* Sci Rep. 2025;15:95108.](https://www.nature.com/articles/s41598-025-95108-5) | 생체신호 phase/amplitude 분리 정합 방법론, 그룹 간 곡선 비교에 통계적 프레임 제공 |

### E4. 함수형 데이터 분석(FDA) + DNN — point-wise 대신 basis 표현 ([[feedback_aec_curve_wholistic]] 정합)

| 문헌 | 핵심 내용 |
| --- | --- |
| [*Deep Learning for Functional Data Analysis with Adaptive Basis Layers.* arXiv:2106.10414.](https://arxiv.org/pdf/2106.10414) | 128개 포인트를 고정 basis(FPCA)가 아닌 학습 가능한 basis로 투영 후 DNN 입력 |
| [Wang et al. *Functional data analysis using deep neural networks.* WIREs Comput Stat. 2024.](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.70001) | FPCA→DNN 구조(FDNN) 개관, 1D/2D functional regression·classification 리뷰 |
| [*Deep Neural Network Classifier for Multi-dimensional Functional Data.* arXiv:2205.08592.](https://arxiv.org/pdf/2205.08592) | 함수형 주성분(FPC) 기반 curve-level classification — 기존 stage2 AEC PCA 피처 접근과 이론적으로 대응 |

### E5. 데이터 증강 (1D CNN 학습용, 공간축 순서 보존 필요)

| 문헌 | 핵심 내용 |
| --- | --- |
| [*Data Augmentation techniques in time series domain: A survey and taxonomy.* arXiv:2206.13508.](https://arxiv.org/html/2206.13508v4) | Jittering/Scaling/Magnitude·Window Warping/Permutation 분류 체계 |
| [*An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks.* PLOS ONE. 2021.](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0254841) | Window Warping이 VGG/ResNet/LSTM 전반에서 평균 순위 1위(가장 범용적) |

**주의**: 위 증강 기법은 시간축 데이터 기준으로 설계되어 permutation처럼 순서를
깨는 기법이 포함된다. AEC는 z축(해부학적 위치) 순서가 의미를 가지므로
**scaling·jittering·국소 magnitude warping만 선별 적용**하고 permutation은
배제할 것.

## 참고 (도메인 오인 — BIA/EIM 전제 조사, 폐기됨)

세션 초반 AEC를 BIA/EIM 전기임피던스 전류로 오인하고 조사한 레퍼런스(Sanchez &
Rutkove 2017, Rutkove 2009, Kortman et al. 2014/2015, Cheng et al. 2022 Sensors,
Piccoli et al. 1994 BIVA)는 이번 CT AEC/TCM 확인으로 **도메인이 맞지 않아 폐기**.
필요시 대화 이력 참고.
