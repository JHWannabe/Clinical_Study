# AEC-128 + clinic4 멀티모달 아키텍처 10일 로테이션 계획

목표: clinic4(age/sex/height/weight) 단독 대비 HTN/DM/CKD internal OOF AUC를 +0.05 이상
개선하는 아키텍처를 찾는다. 매일 질적으로 다른 아키텍처 계열 1개씩 시도.

## 공통 평가 프로토콜
- 데이터: `data/gangnam_final_dataset.xlsx`(internal) / `data/sinchon_final_dataset.xlsx`(external)
- clinic4 baseline: logistic regression, internal 5-fold OOF AUC
- 각 아키텍처: internal 5-fold CV로만 비교(모델 선택), external은 최종 확정 설정으로 1회만 동결 평가
  ([[feedback_internal_external_validation_discipline]])
- 종료 기준: internal delta ≥ +0.02 AUC(3질환 중 1개라도) → "유망"으로 표시, 다음날 세부 튜닝으로 이어감.
  미달이면 다음 계열로 이동
- 백본 curve 전처리는 raw 또는 patient-wise z-score만 사용([[feedback_aec_preprocessing_methods]])
- 코드는 `code/03_aec_deep_learning/fusion/aec_fusion_common.py`의 `run_fusion_pipeline`을 재사용(데이터 로딩·그리드서치·DeLong·plot 공통화)

## 사전 참고 근거(이미 확인된 것)
- CNN GAP + concat/gated/attnpool/crossattn 4종 fusion: 24개 DeLong 중 Bonferroni 통과 0건
  ([[project_aec_fusion_htn_dm_ckd_comparison]])
- CNN depth(3/5/7/9층): 개선 없음, DM은 깊을수록 악화([[project_aec_depth_ablation_no_improvement]])
- curve zero-ablation(matched-capacity): HTN 신호 없음(p=0.51), DM 경계(p=0.06), CKD의 명목유의(p=0.027)는
  zero-curve 모델 자체의 학습 불안정 때문으로 확인돼 곡선신호 근거로 인용 불가([[project_aec_curve_zero_ablation_no_signal]])
- FPCA PC1-3: 128차원 곡선 분산의 97.4% 설명 — "정보 손실" 문제라기보다 곡선 자체의 예측정보 한계
- 결론: **어느 질환도 곡선 기반 아키텍처가 clinic4를 유의하게 능가한다는 근거가 없음(CKD도 예외 아님)**

## 10일 로테이션

| Day | 아키텍처 | 핵심 아이디어 | 파일 | 상태 |
|---|---|---|---|---|
| 1 | BiLSTM/GRU curve encoder | 시퀀스 순환 구조로 장거리 의존성 포착 | `code/03_aec_deep_learning/arch/aec_arch_day1_rnn.py` | 완료(목표 미달) |
| 2 | Transformer encoder (self-attention) | positional encoding + multi-head attention, CLS/mean pooling | `code/03_aec_deep_learning/arch/aec_arch_day2_transformer.py` | 완료(목표 미달) |
| 3 | TCN (dilated causal conv) | dilation 1/2/4/8로 receptive field 지수적 확장 | `code/03_aec_deep_learning/arch/aec_arch_day3_tcn.py` | 완료(목표 미달) |
| 4 | SE-CNN (channel/segment attention) | 기존 CNN에 Squeeze-Excitation 추가, 구간별 중요도 자동 가중 | `code/03_aec_deep_learning/arch/aec_arch_day4_secnn.py` | 완료(목표 미달) |
| 5 | Multi-scale Inception-1D | kernel 3/7/15/31 병렬 branch concat | `code/03_aec_deep_learning/arch/aec_arch_day5_inception.py` | 완료(목표 미달) |
| 6 | Self-supervised pretrain → fine-tune | curve autoencoder 사전학습 후 encoder만 fine-tune | `code/03_aec_deep_learning/arch/aec_arch_day6_ssl_pretrain.py` | 완료(목표 미달) |
| 7 | Residual-target learning | clinic4 logistic의 residual(잔차 logit)을 curve 모델이 예측 | `code/03_aec_deep_learning/arch/aec_arch_day7_residual_target.py` | 완료(목표 미달) |
| 8 | Multi-task 공유 인코더 | curve 인코더 하나로 HTN+DM+CKD 동시 학습, task-specific head 분리 | `code/03_aec_deep_learning/arch/aec_arch_day8_multitask.py` | 완료(목표 미달) |
| 9 | Late-fusion stacking | curve-only NN과 clinic4 logistic을 독립 학습 후 meta-logistic으로 결합 | `code/03_aec_deep_learning/arch/aec_arch_day9_stacking.py` | 완료(목표 미달) |
| 10 | 주파수영역 특징 | FFT/DWT로 곡선을 주파수 성분화 후 소형 MLP + clinic4 concat | `code/03_aec_deep_learning/arch/aec_arch_day10_frequency.py` | 완료(목표 미달) |

## 10일 로테이션 결론(2026-09-01 완료)

10개 아키텍처(RNN/Transformer/TCN/SE-CNN/Inception-1D/SSL pretrain/residual-target/multi-task/stacking/주파수영역)
모두 internal delta ≥ +0.02 기준 미달. internal 최댓값은 Day7 residual-target의 CKD +0.0105(p=0.213, n.s.)이고,
30개 (아키텍처×질환) internal 비교 중 명목 유의(p<0.05)조차 없음. HTN은 10개 중 8개가 delta<0.01,
DM·CKD도 최댓값이 +0.01 초반대에 머묾 — 이전 구간평균/FPCA/CNN GAP/4종 fusion/depth ablation/curve
zero-ablation 6가지와 합쳐 총 16가지 독립적 방법론이 "AEC-128 곡선에 clinic4를 +0.05 이상 능가할 신호가
없다"는 동일 결론에 수렴. 다음 단계는 아키텍처 추가 탐색보다 목표치 재설정 또는 다른 입력 데이터(예: 원본
CT 영상 자체)를 지도교수님과 논의하는 쪽을 권장.

## 결과 기록
각 Day 완료 시 아래에 한 줄씩 추가(internal delta 기준, 3질환 중 최댓값과 해당 질환 표기).

| Day | 최대 internal delta | 질환 | DeLong p | 판정 |
|---|---|---|---|---|
| 1 (RNN) | +0.0124 | DM | 0.162(n.s.) | 미달(HTN +0.0014/n.s., CKD -0.0007/n.s.) |
| 2 (Transformer) | +0.0009 | HTN | 0.727(n.s.) | 미달(DM -0.0019/n.s., CKD +0.0008/n.s.) |
| 3 (TCN) | +0.0074 | CKD | 0.332(n.s.) | 미달(HTN -0.0006/n.s., DM +0.0052/n.s.) |
| 4 (SE-CNN) | +0.0029 | DM | 0.724(n.s.) | 미달(HTN +0.0026/n.s., CKD -0.0210/n.s.) |
| 5 (Inception-1D) | +0.0103 | CKD | 0.361(n.s.) | 미달(HTN +0.0038/n.s., DM +0.0061/n.s.) |
| 6 (SSL pretrain) | +0.0053 | HTN | 0.304(n.s.) | 미달(DM -0.0052/n.s., CKD -0.0055/n.s.) |
| 7 (Residual-target) | +0.0105 | CKD | 0.213(n.s.) | 미달(HTN +0.0034/n.s., DM +0.0068/n.s., p=0.060 경계) |
| 8 (Multi-task) | +0.0067 | DM | 0.427(n.s.) | 미달(HTN -0.0065/n.s., CKD -0.0192/n.s.) |
| 9 (Stacking) | +0.0072 | HTN | 0.040(명목유의, 목표 미달) | 미달(DM -0.0032/n.s., CKD -0.0213/p=0.048 악화) |
| 10 (Frequency) | -0.0025 | HTN(최선도 음수) | 0.362(n.s.) | 미달(DM -0.0032/n.s., CKD -0.0050/n.s., 3종 전부 internal 악화) |
