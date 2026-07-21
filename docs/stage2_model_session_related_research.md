# stage2_model.py 세션 비교 관련 레퍼런스 조사

이번 세션에서 `code/stage2_model.py`(Stage-2 late-fusion 분류기)에 순서대로 적용/시도한
9가지 변경에 대한 근거 문헌 정리. **모든 외부 논문을 최근 10년(2016~2026) 이내 발표분으로
한정** — 방법론 원 논문(Newcombe 1998, Obuchowski 2005, Pepe 2003, DeLong 1988, Ioffe &
Szegedy 2015, Pascanu et al. 2013, Prechelt 1998, Saerens et al. 2002)은 다른 저장소 문서
(`related_research.md`, `stage1_sensitivity_threshold_related_research.md`)에서는 foundational
예외로 인용되어 있지만, 본 문서에서는 전부 10년 이내에 발표된 대체/후속 문헌으로 교체했다.
이미 이 저장소의 다른 문서가 다룬 항목(및 그 문서의 오래된 원 논문)은 새로 찾지 않고
교차 인용으로 표시하되, 본 문서 자체의 1차 인용은 모두 2016년 이후로 유지.

## 요약 표

| # | 변경 | 결과 (internal / external AUC) | 채택 여부 |
| --- | --- | --- | --- |
| 1 | `pos_weight` 클래스 가중 (BCEWithLogitsLoss) | 0.553 → 0.718 / - | ✅ 채택 |
| 2 | Threshold 선택을 F1-max → NI 조건 제약 하 spec-max로 변경 | (동일 모델, threshold만 재선택) NI PASS/PASS | ✅ 채택 |
| 3 | LR 스케줄(ReduceLROnPlateau) + gradient clipping | 0.718→0.721 / 0.763→0.756 (중립) | ✅ 유지(안정성 목적) |
| 4 | Validation-AUC 기반 epoch 선택 (inner split) | 0.721→0.649 / 0.756→0.762 (악화) | ❌ 되돌림 |
| 5 | 5-seed 앙상블 평균 | 0.721→0.724 / 0.756→0.771 | ✅ 채택 (현재 최고) |
| 6 | Curve-level 요약 통계 7개 + 1차/2차 미분 4개 추가 | 0.724→0.714 / 0.771→0.754 (악화) | ❌ 되돌림 |
| 7 | AecBranch 확장(GAP+GMP, BatchNorm, 채널 확대) | spec 0.580→0.558 / 0.507→0.000 (악화) | ❌ 되돌림 |
| 8 | Stage-1 vs Full-pipeline AUC 비교 (전체 코호트, DeLong 검정) | internal 0.828 vs 0.822 (p=0.32) / external 0.835 vs 0.844 (p=0.046) | ✅ 채택(평가 도구) |

## 1. 클래스 불균형 보정 — `pos_weight` (weighted BCE)

Stage-2 입력의 양성비율(TP/screen-positive)이 internal 20.7%(117/564)로 불균형 —
`BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`를 적용해 소수 클래스(TP) gradient를
증폭했다. 이 변경 하나로 internal AUC가 0.553→0.718로 가장 크게 개선됨(loss가
base-rate entropy 근처에서 정체되던 과소적합 문제 해소).

- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). [*Class-Balanced Loss Based
  on Effective Number of Samples*](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf).
  CVPR 2019, 9268–9277. — loss 단계에서 클래스 가중치를 주는 표준적 접근의 최신 정식화
  (effective number of samples).
- PyTorch `BCEWithLogitsLoss(pos_weight=...)` — 본 구현이 그대로 따르는 표준 라이브러리
  구현(공식 문서 기준 소수 클래스 가중은 `pos_weight`로 양성 항에만 곱해짐).

**주의(교차 인용)**: [stage1_sensitivity_threshold_related_research.md](stage1_sensitivity_threshold_related_research.md)
4절이 인용하는 van den Goorbergh et al. (2022, *JAMIA*)은 **리샘플링(SMOTE/oversampling)**이
calibration을 훼손한다는 결론이며, 본 세션의 `pos_weight`는 리샘플링이 아니라 손실함수
가중이다. 다만 원리적으로는 loss reweighting도 유효 class prior를 바꾸므로 동일한 calibration
왜곡 위험이 있다 — 아래 최근 문헌들이 이 prior-shift/재보정 문제를 다룬다:

- Tian, J., Liu, Y.-C., Glaser, N., Hsu, Y.-C., & Kira, Z. (2020). [*Posterior
  Re-calibration for Imbalanced Datasets*](https://arxiv.org/abs/2010.11820). arXiv:2010.11820.
  — 클래스 불균형 보정(리샘플링·재가중 포함)이 posterior 확률을 왜곡시키는 정도를 정식화하고
  재보정(recalibration) 절차를 제안. 본 세션의 `pos_weight`가 유발할 수 있는 확률 왜곡의
  최근 정량적 근거.
- Tasche, D. (2025). [*Recalibrating binary probabilistic classifiers*](https://arxiv.org/abs/2505.19068).
  arXiv:2505.19068. — distribution/covariate shift 관점에서 이진 분류기의 확률을 목표
  prior에 맞게 재보정하는 방법(CSPD, QMM)을 다룬 최신(2025) 논문. class prior를 바꾸는
  조정(본 세션의 `pos_weight` 포함)은 별도 재보정 없이는 확률값이 왜곡된 채 남는다는
  논지를 뒷받침.

본 파이프라인은 확률값 자체(calibration)가 아니라 AUC(순위 판별력)와 NI 제약 하의 threshold만
쓰므로 이 위험의 실질적 영향은 제한적이지만, 확률값을 직접 보고/사용할 계획이 있다면
재보정이 필요하다는 점을 명시해야 함.

## 2. Threshold 선택 기준 — F1-max → NI 조건 제약 하 specificity-max

기존 F1-최대화 threshold는 internal sensitivity를 0.907→0.605까지 무너뜨려 NI 기준을
위반했다. Threshold 탐색 목적함수를 "sens ≥ sens_before×0.95 및 spec ≥ spec_before를
만족하는 후보 중 spec 최대"로 바꿔 NI 준수를 threshold 선택 단계에서 강제했다.

이 항목은 이미 [related_research.md](related_research.md) 3-4절, 4절과
[stage1_sensitivity_threshold_related_research.md](stage1_sensitivity_threshold_related_research.md)
1-4절이 상세히 다룸(그 문서들의 1차 인용인 Newcombe 1998, Obuchowski 2005, Pepe 2003은
10년 초과라 본 문서에서는 재인용하지 않음) — 아래는 10년 이내 문헌으로 같은 논지를
뒷받침하는 최근 참고문헌:

- [Comparison of the sensitivity and specificity of two diagnostic tests: paired-sample
  confidence intervals](https://pmc.ncbi.nlm.nih.gov/articles/PMC10039285/) (BMC Med Res
  Methodol, 2023) — 동일 환자 집합에 대한 두 진단 전략(Stage-1-only vs Stage-1+Stage-2)의
  민감도/특이도 차이를 paired CI로 비교하는 최신 방법론. `related_research.md` 3-4절이 이미
  인용한 문헌을 본 절의 threshold 재설계 근거로도 재사용.
- Ganguly, I., & Huang, Y. (2025). [*Sequential Testing for Assessing the Incremental Value
  of Biomarkers Under Biorepository Specimen Constraints with Robustness to Model
  Misspecification*](https://arxiv.org/abs/2511.15918). arXiv:2511.15918. — 2단계 순차
  검사에서 새 바이오마커(본 연구의 Stage-2/AEC에 해당)가 기존 검사(Stage-1) 대비 주는
  증분 가치를 평가하는 최신 순차검정 프레임워크. Serial 2-stage 구조에서 두 번째 검사가
  담당해야 할 역할(첫 검사 대비 증분 개선)이 본 세션의 threshold 재설계 목표(Stage-2가
  specificity를 담당)와 같은 문제의식.

## 3. 학습 안정화 — LR 스케줄(ReduceLROnPlateau) + Gradient Clipping

`loss_curve.png`에서 학습 후반부 loss가 진동하는 것을 관찰하고, LR을 plateau 시
절반으로 줄이는 스케줄과 `clip_grad_norm_(max_norm=1.0)`을 추가했다. AUC는
통계적으로 구별 안 되는 수준(0.718→0.721 internal)이었지만 학습 곡선의 노이즈는
감소함.

- Ramaswamy, A. (2023). [*Gradient Clipping in Deep Learning: A Dynamical Systems
  Perspective*](https://www.scitepress.org/PublishedPapers/2023/116780/). 12th Int'l Conf.
  on Pattern Recognition Applications and Methods (ICPRAM 2023). — gradient clipping이
  학습 동역학(안정적 minima로의 수렴)에 미치는 영향을 동역학계 관점에서 분석한 최신 논문.
- Marshall, N., Xiao, K. L., Agarwala, A., & Paquette, E. (2024). [*To Clip or not to Clip:
  the Dynamics of SGD with Gradient Clipping in High-Dimensions*](https://arxiv.org/abs/2406.11733).
  arXiv:2406.11733. — 고차원에서 gradient clipping이 SGD 동역학에 미치는 영향을 정량
  분석. `clip_grad_norm_(max_norm=1.0)` 채택의 최신 이론적 배경.
- PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` — 표준 구현.

## 4. Validation-AUC 기반 epoch 선택 (시도 후 되돌림)

각 fold의 학습 데이터를 다시 85/15로 쪼개 inner validation AUC로 조기종료 시점을
정하도록 바꿨으나, inner validation이 fold당 ~68명(event ~14개)에 불과해 추정 분산이
커서 fold별 학습량이 들쭉날쭉해지고 internal AUC가 0.721→0.649로 악화 — 되돌림.

- Forouzesh, M., & Thiran, P. (2021). [*Disparity Between Batches as a Signal for Early
  Stopping*](https://arxiv.org/abs/2107.06665). arXiv:2107.06665. — 조기종료에 쓰이는
  신호(gradient disparity)가 표본이 적거나 라벨 노이즈가 있을 때 특히 유용/불안정해질 수
  있음을 다룸. 본 세션에서 관찰한 "inner validation이 너무 작으면(event~14개) 조기종료
  신호 자체가 신뢰할 수 없어진다"는 현상과 같은 문제의식 — 신호 선택(train loss vs. 작은
  val split의 AUC)이 조기종료 품질을 좌우한다는 논지를 뒷받침.

## 5. 5-Seed 앙상블 평균

각 fold(및 최종 refit)를 5개의 서로 다른 랜덤 시드로 독립 학습한 뒤 sigmoid 확률을
평균 — internal 0.721→0.724, external 0.756→0.771로 소폭 개선, 특히 external에서
threshold가 실제로 재분류를 수행하게 됨(이전엔 0/0 무효화 상태).

- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). [*Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles*](https://arxiv.org/abs/1612.01474). NeurIPS
  2017. — 서로 다른 랜덤 초기화로 학습한 모델들의 평균이 단일 모델보다 안정적인 예측/불확실성
  추정을 제공한다는 표준 근거. 본 세션은 uncertainty quantification이 아니라 point
  estimate(OOF AUC) 안정화 목적으로 같은 메커니즘(시드 다양성 평균)을 사용.

## 6. Curve-level 요약 통계 + 1차/2차 미분 (시도 후 되돌림)

AEC-128 곡선을 std/skewness/slope/quartile 평균(7개) 및 1차·2차 미분의 평균+표준편차
(4개, 총 11개)로 요약해 별도 branch로 fusion head에 추가했으나, 오히려
0.724→0.719(7개)→0.714(11개)로 단조 하락 — CNN branch(conv+GAP)가 이미 이 정보를
학습하고 있어 명시적 통계량 추가가 새 정보보다 차원(노이즈)만 늘린 것으로 해석하고
되돌림.

- Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., & Muller, P.-A. (2019). *Deep
  learning for time series classification: a review.* Data Mining and Knowledge Discovery,
  33(4), 917–963. — 이미 [related_research.md](related_research.md) 1절(내부 자료 표,
  `CAM.pdf`)에서 인용된 문헌. Raw curve를 1D-CNN에 직접 넣는 접근(본 AecBranch와 동일
  설계)의 아키텍처적 배경이자, handcrafted feature와 raw-signal CNN을 비교하는 표준 참고문헌.
- 곡선을 명시적 저차원 요약통계(FPCA/quartile 등)로 축약하는 대안 방법론은 이미
  [related_research.md](related_research.md) 3-1~3-3절에 상세 정리되어 있음 — 본 세션의
  실패 사례(명시적 요약 통계가 CNN 대비 이득 없음)는 그 문서가 논하는
  "국소 신호가 전역 성분에 희석됨"(3-2절 인용, Zhang & Parnell 2023) 문제의 한 변주로 볼 수 있음.

## 7. AecBranch 구조 확장 — GAP+GMP, BatchNorm, 채널 확대 (시도 후 되돌림)

Conv 채널을 8→16에서 16→32로 늘리고 BatchNorm과 GlobalMaxPool(피크 신호 보존
목적)을 추가했으나 internal spec 0.580→0.558, external spec 0.507→**0.000**(재분류
완전 무효화)로 악화 — n=564 규모에서 파라미터 증가가 과적합으로 이어진 것으로 판단하고
되돌림.

- 1D-CNN + Global Average/Max Pooling 결합 아키텍처 일반론(채널 확장 시 표본 대비
  파라미터 비율이 성능을 좌우한다는 논지)은 여러 도메인(활동 인식, 결함 진단 등)에서
  공통적으로 보고되나, 본 조사에서 "128-slice 생체 곡선, n<1000" 규모에 특화된
  2023–2024년 정량 논문은 찾지 못함 — 6절의 Ismail Fawaz(2019) 리뷰가 다루는 일반적
  용량-표본 트레이드오프로 대체 참고.
- Yong, H., Huang, J., Meng, D., Hua, X., & Zhang, L. (2020). [*Momentum Batch
  Normalization for Deep Learning with Small Batch Size*](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_14).
  ECCV 2020. — BatchNorm이 배치 크기가 작을수록 평균/분산 추정이 불안정해진다는 문제를
  다룬 논문. 본 학습 루프는 fold 전체(~450개)를 한 번에 forward하는 full-batch 학습이라
  "작은 배치" 문제 자체는 없지만, 그럼에도 spec이 악화된 것은 배치 크기가 아니라 표본
  자체(n=564)와 파라미터 수 증가의 상호작용 때문임을 시사.
- [Regularizing deep neural networks for medical image analysis with augmented batch
  normalization](https://www.sciencedirect.com/science/article/abs/pii/S156849462400111X)
  (Computers in Biology and Medicine / Medical Image Analysis 계열, 2024) — 의료 영상
  분석에서 BatchNorm 정규화 효과와 표본 크기 민감성을 다룬 최신 연구. 본 세션의 AecBranch
  확장 실패(과적합 의심)를 뒷받침하는 최근 도메인 근거.

## 8. Stage-1 vs Full-pipeline AUC 비교 (전체 코호트, DeLong 검정)

Screen-negative는 Stage-1 점수, screen-positive는 `Stage-1 threshold + Stage-2 점수`를
부여해 전체 코호트(1090/926명) 단위의 연속 점수를 구성 — Stage-1 단독 AUC와
Full-pipeline AUC를 같은 축에서 비교할 수 있게 함. DeLong 등의 방법으로 paired AUC
차이 유의성까지 검정(internal p=0.32 NS, external p=0.046).

- Zhu, H., Liu, S., Xu, W., Dai, J., & Benbouzid, M. (2024). [*Linearithmic and unbiased
  implementation of DeLong's algorithm for comparing the areas under correlated ROC
  curves*](https://www.sciencedirect.com/science/article/abs/pii/S0957417424000599).
  Expert Systems with Applications, 246, 123194. — DeLong(1988)의 원 방법(paired/correlated
  ROC AUC 비교)을 O(n log n)으로 재구현하고 편향을 교정한 최신(2024) 논문. 본 세션에서
  추가된 DeLong diff/p-value 출력이 구현하는 통계 검정의 최신 정식화이자 구현 참고문헌.
- 같은 표본(같은 환자)에서 나온 두 진단 전략의 AUC를 비교한다는 문제 설정은 §2에서 인용한
  [BMC Med Res Methodol(2023) paired 신뢰구간 문헌](https://pmc.ncbi.nlm.nih.gov/articles/PMC10039285/)이
  다루는 "paired sensitivity/specificity 비교"와 동일 계열 — DeLong은 그 이산(Se/Sp)
  버전이 아니라 연속(AUC) 버전에 해당.
- Screen-negative/positive를 하나의 순위로 잇는 구성 자체(전 단계 점수 + 상수 오프셋)는
  §2에서 인용한 Ganguly & Huang (2025)의 순차검정 증분가치 프레임워크의 문제의식(두 번째
  검사가 첫 검사 대비 주는 증분 판별력)을 연속 점수 버전으로 확장한 것 — 별도 논문 근거보다는
  §2에서 이미 확립된 원리의 직접적 응용.

## 9. Fusion 아키텍처 용어 재검토 — "late fusion"이 실제로는 intermediate fusion

`LateFusionNet`(코드 주석상 "late fusion")은 clinical/AEC 두 branch의 **임베딩을
concat한 뒤 공유 classification head를 함께 학습**하는 구조다. 최근 멀티모달 융합
분류체계에 따르면 이는 "late(decision) fusion"이 아니라 **intermediate/feature-level
fusion**에 해당한다 — late fusion은 별도로 학습된 두 모델의 최종 예측(확률)을 사후에
결합하는 방식을 가리킨다. 오히려 본 세션에서 추가한 `LateFusionEnsemble`(5개 독립
모델의 sigmoid 확률 평균)이 문헌상 정의의 "late fusion"에 더 가깝다.

- [A review of deep learning-based information fusion techniques for multimodal medical
  image classification](https://arxiv.org/html/2404.15022v1) (arXiv, 2024) — early/
  intermediate(feature-level)/late(decision-level) 3분류 정의.
- [The future of multimodal artificial intelligence models for integrating imaging and
  clinical metadata: a narrative review](https://www.dirjournal.org/articles/the-future-of-multimodal-artificial-intelligence-models-for-integrating-imaging-and-clinical-metadata-a-narrative-review/doi/dir.2024.242631)
  (Diagn Interv Radiol, 2024) — 같은 3분류를 임상 영상+메타데이터 맥락에서 재정리.
  Intermediate fusion이 임상+영상 결합에서 가장 흔히 쓰이는 방식이라는 서술과, 본
  아키텍처(임베딩 concat 후 공동 학습)가 부합.
- 메타분석 수치 참고(직접 비교 목적 아님): 2025년 Alzheimer 진단 transformer 융합
  메타분석에서 intermediate fusion AUC=0.931이 early(0.905)/late(0.912)보다 높게
  보고됨 — 도메인이 다르므로 본 연구 결과를 직접 뒷받침하진 않지만, "intermediate
  fusion을 우선 시도하는" 본 설계 선택과 방향이 일치.

**정정 권고**: 코드 주석/변수명(`LateFusionNet`, "late fusion")을 그대로 유지할지,
문헌상 정확한 용어인 "intermediate/joint fusion"으로 바꿀지는 다음에 확인이 필요함 —
본 조사에서는 현상 확인만 하고 리네이밍은 하지 않음.

## 참고 메모

- [related_research.md](related_research.md) — 2단계 설계 전체(AEC curve, NI test,
  clinical-only baseline)의 선행 조사
- [stage1_sensitivity_threshold_related_research.md](stage1_sensitivity_threshold_related_research.md) —
  Stage-1 threshold 및 class imbalance 관련 선행 조사
- [aec_residual_related_papers.md](aec_residual_related_papers.md) — AEC 잔차화 방법론
  (본 세션과는 다른 구파이프라인 대상이나 곡선 전처리 배경으로 참고 가능)
- [cnn_variant_comparison.md](cnn_variant_comparison.md) — 이전 파이프라인의 CNN 변형 비교
  (본 세션 §6~7의 "CNN이 이미 정보를 잡고 있다" 관찰과 대조해볼 가치)
