# Dataset / Method 초안 (국문) — HTN/DM/CKD 예측: clinic4 vs clinic4+AEC

> 작성 기준: `code/step2_clinic_aec_disease_select.py`, `code/step3_clinic_aec_disease_logistic.py`
> (2026-08-14 실행 결과, `outputs/step2_disease_select/`, `outputs/step3_disease_logistic/`)
>
> **Method 섹션 주 템플릿**: Jeon ET, Park H, Lee JK, Heo EY, Lee CH, Kim DK, Kim DH, Lee HW.
> "Deep Learning–Based Chronic Obstructive Pulmonary Disease Exacerbation Prediction Using
> Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study." *J Med Internet Res*
> 2025;27:e69785. (이하 "Jeon 2025") — 2개 병원 코호트(개발/외부검증), 임상변수만 사용한
> Clin model vs 임상변수+곡선유래 score를 결합한 모델을 logistic regression으로 비교하고
> DeLong test로 AUROC 차이를 검정하는 구조가 본 연구(clinic4 vs clinic4+AEC)와 거의 1:1
> 대응하여 채택함. (`Research-Reference-Paper/papers/Curve-Regression 3.pdf`에 해당 —
> 기존 `[[project_reference_paper_screening]]` 메모리는 이 파일을 "Curve-Regression 1과
> 동일 논문의 다른 출력본(폐활량+임상변수, 2병원, n=10,492)"으로 잘못 기록했음. 실제로는
> Curve-Regression 1/2/3/4가 전부 서로 다른 논문이며, 이 메모리는 stale 상태이므로 향후
> 세션에서 재작성 필요.)
>
> 영문본은 `manuscript_dataset_method_htn_dm_ckd_en.md` 참조.

---

## Dataset

### 코호트 구성 및 환자 선정 과정

본 연구는 강남세브란스병원(internal)과 신촌세브란스병원(external)의 두 코호트를 사용한 **후향적 연구(retrospective cohort study)**이다. 두 코호트는 서로 다른 병원·스캐너 집단에서 수집되어, internal에서 학습된 모델을 external에 그대로 동결 적용(freeze)함으로써 기관 간 일반화 성능을 평가할 수 있도록 설계되었다.

최종 분석 대상 환자는 병원 데이터서비스팀을 통해 확보한 원시 데이터로부터 다음 세 단계의 선정 절차를 거쳐 확정되었다.

1. **임상 데이터와 CT DICOM 데이터의 동시 보유 확인**: 환자-내원 단위로 정리된 CT DICOM 아카이브(폴더명에 연구등록번호 포함)를, 동일 연구등록번호 체계를 쓰는 임상 데이터베이스("통합 문서" — 인구학적 정보, 신장/체중, 흡연·음주력, 과거력, 입원/응급실 기록, 사망 정보, 추적관찰 진단(DM/HTN/이상지질혈증/골다공증/심근경색/뇌졸중 등), LAB 수치 등 다수 시트로 구성)와 자동 대조하여, 두 데이터가 모두 존재하는 환자만 후보로 남겼다(`check_patientid_registration.py`로 폴더의 patientID가 임상DB 각 시트의 연구등록번호에 전부 존재하는지 전수 검증). **이 임상 데이터베이스에는 clinic4(연령·성별·신장·체중) 산출에 쓰인 항목뿐 아니라, 본 연구의 결과 변수인 고혈압(HTN)·당뇨병(DM)·만성신장질환(CKD) 진단 이력도 함께 포함되어 있어, 코호트 선정과 결과 변수 추출이 동일한 임상DB에서 이루어졌다.**
2. **CT 촬영 프로토콜(kVp) 필터링**: 두 코호트 모두에 대해 관전압(tube voltage, kVp)이 100인 검사만 포함하였다 — 관전압이 다르면 관전류(mA) modulation 곡선의 절대 스케일이 달라져 코호트 간 AEC-128 신호를 직접 비교하기 어려워지기 때문이다. 실제로 최종 데이터셋에서 두 코호트 모두 kVp는 100으로 완전히 균일함을 확인하였다.
3. **스캐너 기종 비율 필터링(강남 코호트에만 적용)**: 강남 코호트는 사용 빈도가 낮은 소수 스캐너 기종의 영향을 최소화하기 위해, 스캐너(Manufacturer) 사용 비율이 높은 기종 위주로 추가 필터링하였다. 그 결과 최종 강남 코호트(n=1,090)의 99.2%(1,081명)가 4개 주요 기종(Sensation 64 34.9%, Revolution CT 29.1%, Ingenuity Core 128 18.5%, SOMATOM Definition AS+ 16.7%)에 집중되어 있다. 반면 신촌 코호트는 이러한 추가 필터링 없이 15개 스캐너 기종(SOMATOM Definition Flash 24.6%, SOMATOM Definition AS+ 17.9%, Revolution EVO 16.1%, iCT 256 14.6%, SOMATOM Force 13.3% 등)에 걸쳐 상대적으로 이질적인 스캐너 구성을 유지하였다 — external 코호트가 기관 간 스캐너 차이에 대한 일반화 성능을 더 엄격하게 검증할 수 있도록 하기 위함이다.

위 선정 절차를 모두 통과한 최종 환자 수는 강남(internal) 1,090명, 신촌(external) 926명이다.

### 임상 변수 (clinic4)

위 선정 절차의 1단계에서 매칭에 사용한 동일 임상 데이터베이스로부터, 각 환자의 연령(PatientAge), 성별(PatientSex), 신장(Height), 체중(Weight) 4개 변수를 기본 임상 예측변수 세트("clinic4")로 추출하였다 — Jeon 2025의 "Clin model"(과거 AE-COPD력·호흡곤란·흡입치료 등 임상변수만 사용한 baseline logistic regression)과 대응되는 역할이다. 연령·신장·체중은 internal 코호트 기준으로 적합(fit)한 z-score로 표준화한 뒤 동일 변환을 external에 동결 적용하였고, 성별은 이진 변수(남성=1)로 부호화하였다.

### 결과 변수 (HTN, DM, CKD)

결과 변수는 코호트 선정에 사용한 것과 동일한 임상 데이터베이스의 추적관찰 진단 시트(질환별 최초진단일 유무로 진단 여부를 판정하는 f_u 시트)에 이미 진단 완료된 상태로 기록된 고혈압(HTN), 당뇨병(DM), 만성신장질환(CKD) 이진값(0/1)을 그대로 사용하였다(연속형 지표를 새로 이분화하는 절차는 거치지 않음).

Internal/External 코호트별 유병률은 다음과 같다.

| 질환 | Internal (n=1,090) | External (n=926) |
| ---- | ------------------ | ---------------- |
| HTN  | 329명 (30.2%)      | 425명 (45.9%)    |
| DM   | 204명 (18.7%)      | 285명 (30.8%)    |
| CKD  | 83명 (7.6%)        | 168명 (18.1%)    |

### AEC-128 곡선

각 환자는 흉복부 CT 촬영 시 스캐너가 기록한 **Automatic Exposure Control(AEC) tube-current(mA) modulation curve**를 보유한다. 이는 CT가 z축(두미 방향, craniocaudal) 슬라이스 위치별로 영상 품질을 일정하게 유지하기 위해 실시간으로 조절한 관전류(mA) 프로파일로, 환자 개인의 단면 감쇠(attenuation) 특성, 즉 체형·체격 정보를 내재적으로 담고 있다. 다수의 스캐너 제조사(Manufacturer; 예: SOMATOM Definition AS+, Ingenuity Core 128, Revolution CT 등)가 포함되어 있어, 기관 간 검증 외에 스캐너 서브그룹 분석도 가능하다.

**곡선 생성 절차**는 다음과 같다.

1. **Axial series 선별**: 환자 폴더에 섞여 있는 여러 series(실제 axial 촬영, coronal/sagittal 재구성, scout, dose report 등) 중, slice 수가 20 미만인 series(scout/dose report로 간주), DICOM ImageType에 DERIVED/REFORMATTED가 포함되거나 SeriesDescription에 재구성 관련 키워드(MPR/COR/CORONAL/SAG/SAGITTAL/SPO)가 단어 단위로 포함된 series, ImageOrientationPatient 기준 axial이 아닌 series를 모두 제외하고 axial series만 남겼다.
2. **TotalSegmentator를 이용한 landmark 확인**: 남은 axial series에 대해 TotalSegmentator로 liver·hip_left·hip_right 3개 구조물을 분할하고, liver 마스크의 z축 최상단 slice를 상한(liver landmark), hip(좌우 결합) 마스크의 z축 최하단 slice를 하한(pubis landmark)으로 지정하였다(liver 마스크 3,000 voxel 이상 및 마스크 중앙값 위치가 전체 slice의 30% 지점보다 상단인 경우만 유효한 liver로 인정하는 등 QC 기준 적용). **이 세그멘테이션은 어디까지나 크롭 구간의 해부학적 경계를 확인하기 위한 랜드마킹 용도이며, AEC 신호 자체는 세그멘테이션 출력이 아니라 DICOM 헤더의 관전류 값(XRayTubeCurrent)에서 직접 추출한다.**
3. **Pubis~Liver 구간 크롭**: liver·pubis 경계가 모두 확인되고 순서가 유효한(pubis 위치 ≤ liver 위치) series만 채택하여("ok" status), 해당 구간의 관전류(mA) 값을 pubis→liver 방향으로 순서대로 추출하였다.
4. **128포인트 정규화**: 환자별 원본 crop 구간의 slice 수는 101~238장으로 상이하므로(강남 110~238, 신촌 101~205, 중앙값 143~149), 각 환자의 crop된 관전류 배열을 선형보간(linear interpolation)하여 128개 지점으로 리샘플링한 `aec_128` 곡선을 최종 분석 단위로 사용하였다.

Jeon 2025가 폐활량계의 flow-volume/volume-time 곡선을 "이미 촬영/측정된 신호에서 유도한 추가 biomarker"로 활용했듯이, 본 연구의 AEC-128 곡선 역시 추가 방사선 노출 없이 이미 촬영된 CT의 스캐너 부산물(byproduct)로부터 얻어지는 기회주의적(opportunistic) 신호이다. 다만 위 2단계에서 보듯 크롭 구간 확정에는 TotalSegmentator 분할이 관여하므로, "분할이 전혀 불필요하다"기보다는 "분할 결과가 곧 바이오마커인 것이 아니라 랜드마크 확인용으로만 쓰이고, 정량적 신호(AEC 값) 자체는 DICOM 헤더에서 직접 얻는다"는 점이 정확한 서술이다.

---

## Method

### 예측 모델 구성

Jeon 2025의 "Clin model"(임상변수만) vs "AI-PFT-Clin model"(임상변수+곡선유래 score) 비교 구조를 그대로 채택하여, 두 가지 로지스틱 회귀 모델을 비교하였다:

(1) clinic4만을 예측변수로 사용하는 baseline 모델, (2) clinic4에 AEC-128 곡선에서 유도한 형태(shape) 특징을 추가한 확장 모델(clinic4+AEC(best))이다. Jeon 2025가 폐기능 곡선 정보를 임상변수에 추가함으로써 AE-COPD 예측력이 향상됨을 보고한 것과 같은 논리로, 본 연구는 CT 스캐너의 AEC 곡선 자체를 opportunistic biomarker로 사용하여 동일한 가설(체형 관련 영상 정보의 추가가 clinical-only 모델 대비 진단 분류 성능을 향상시키는지)을 검정하였다.

### AEC 형태 특징 및 최적 조합 선택

AEC-128 원곡선으로부터 다음 5개 후보 표현을 산출하였다: (i) 표준편차(SD), (ii) 왜도(Skewness), (iii) 상위/하위 50% 구간 평균비율, (iv) FPCA(functional PCA) score, (v) 위 4가지를 모두 결합한 조합. FPCA는 internal 코호트의 raw AEC-128 곡선에 PCA를 적합하여 산출하였으며, 주성분 개수는 하위 예측 성능(AUC)이 아니라 표준적인 FPCA/PCA 관행에 따라 **누적 explained variance ratio가 99.5%를 최초로 넘는 시점**으로 결정하였다
(본 실행에서 n=7). 5개 후보 중 HTN/DM/CKD 3개 질환의 internal 5-fold 층화 교차검증
평균 out-of-fold(OOF) AUC가 가장 높은 조합을 최종 clinic4+AEC(best)로 선택하였으며
(본 실행에서 FPCA(PC1-7) 단독 조합이 선택됨), 이 선택 과정은 오직 internal 데이터만
사용하고 external은 전혀 참조하지 않았다.

### 내부 교차검증 및 외부 동결 검증

Jeon 2025가 BRMH 코호트를 train(60%)/internal validation(20%)/internal test(20%)로
나누고 SNUH를 별도의 external validation set으로 사용한 것과 유사하게, 본 연구는
internal 코호트에 층화 K-fold 교차검증(K=5, 클래스 최소 표본수에 따라 자동 축소)을
적용하여 out-of-fold 예측확률로 internal AUC를 산출하였다. FPCA가 포함된 모델의 경우,
검증 fold의 곡선 정보가 주성분 추정에 유입되는 data leakage를 방지하기 위해 PCA
적합(fit)을 매 fold의 학습(train) 구간에서만 독립적으로 재수행하였다. External 코호트에는
internal 전체 데이터로 학습을 완료한 단일 동결(frozen) 모델을 1회만 적용하여 예측확률을
산출하였으며, 모델 선택 단계에서 이미 확립된 원칙대로 external 결과를 모델/조합 선택에
재사용하지 않았다(internal-선택 → external-1회검증 원칙).

### 통계적 비교

Jeon 2025가 Clin model과 AI-PFT-Clin model의 AUROC 차이를 **DeLong method**로 검정하고
internal validation set에서 도출한 Youden index 기준 cutoff으로 민감도·특이도를
산출한 것과 동일하게, 본 연구는 clinic4와 clinic4+AEC(best) 두 모델의 AUC 차이를 동일
환자 표본에서 산출된 두 예측 점수를 비교하는 **paired DeLong test**(DeLong et al. 1988;
Sun & Xu 2014 알고리즘)로 검정하였으며, internal/external 각각에 대해 독립적으로
수행하였다. 분류 임계값은 internal OOF ROC에서 **Youden's J index**(sensitivity+
specificity−1)를 최대화하는 지점으로 결정한 뒤 동일 임계값을 external에 고정 적용하여
민감도·특이도·정확도를 산출하였다(P<0.05를 유의성 기준으로 사용, Jeon 2025와 동일).
아울러 Jeon 2025의 연령/성별/흡연 서브그룹 분석에 대응하여, 스캐너 제조사별 서브그룹
(표본수 30명 이상, 두 클래스 모두 존재하는 스캐너만 포함)에서 재학습 없이 동일 예측확률을
재분할하여 AUC를 산출함으로써, 결과가 특정 스캐너에 국한되지 않는지 확인하였다.

### 주요 결과 요약 (참고, 재실행 시 값 변동 가능)

| 질환 | Internal AUC (clinic4→+AEC) | Internal DeLong p | External AUC (clinic4→+AEC) | External DeLong p |
| ---- | ---------------------------- | ----------------- | ---------------------------- | ----------------- |
| HTN  | 0.808 → 0.815               | 0.183 (n.s.)      | 0.715 → 0.728               | **0.031**   |
| DM   | 0.727 → 0.751               | **0.018**   | 0.662 → 0.697               | **0.002**   |
| CKD  | 0.790 → 0.803               | 0.335 (n.s.)      | 0.622 → 0.635               | 0.140 (n.s.)      |

DM은 internal/external 모두 유의한 개선을 보였고, HTN은 external에서만 유의하였다.
CKD는 이번 실행 기준으로 internal/external 모두 유의성에 도달하지 못했다(체성분 연속값
예측에서 AEC 추가가 일관되게 악화되던 이전 결과와 달리, 진단 분류 과제에서는 방향이
개선 쪽으로 일관되나 유의성은 질환별로 차이가 있음). Jeon 2025 역시 external validation
cohort에서 severe AE-COPD의 AUROC 개선폭(0.675→0.713)이 moderate-to-severe 대비
작았던 것과 유사하게, 유병률이 낮은 결과 변수(CKD, 7.6~18.1%)일수록 검정력이 낮아
유의성 도달이 더 어려운 경향이 관찰된다.

---

## 참고문헌 (Method 서술 근거)

1. **[주 템플릿]** Jeon ET, Park H, Lee JK, Heo EY, Lee CH, Kim DK, Kim DH, Lee HW. Deep
   Learning–Based Chronic Obstructive Pulmonary Disease Exacerbation Prediction Using
   Flow-Volume and Volume-Time Curve Imaging: Retrospective Cohort Study. *J Med Internet
   Res* 2025;27:e69785. doi:10.2196/69785
2. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
   receiver operating characteristic curves: a nonparametric approach. *Biometrics*
   1988;44(3):837-845.
3. Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the areas under
   correlated receiver operating characteristic curves. *IEEE Signal Process Lett*
   2014;21(11):1389-1393.
4. Ramsay JO, Silverman BW. *Functional Data Analysis*. 2nd ed. Springer; 2005. (FPCA 근거)
5. Jolliffe IT. *Principal Component Analysis*. 2nd ed. Springer; 2002. (누적 explained
   variance ratio 기준으로 성분 수를 정하는 표준 방법론 근거 — Ch.6, 다만 문헌이 예시하는
   임계값은 통상 70~95%이며 99.5%라는 수치 자체는 문헌 권고가 아니라 저자 지정값)
6. (선택) "A new method for estimating patient body weight using CT dose modulation data."
   — AEC/tube-current modulation curve가 체형 정보를 담고 있다는 근거로 Dataset 섹션에
   인용 가능.

> **주의**: 위 표의 수치는 2026-08-14 시점 스크립트 실행 결과이며, 랜덤성(교차검증 fold
> 분할)이나 데이터 갱신에 따라 재실행 시 달라질 수 있음(스크립트 주석에도 명시됨). 논문
> 제출 전 최종 실행 결과로 재확인 필요.
