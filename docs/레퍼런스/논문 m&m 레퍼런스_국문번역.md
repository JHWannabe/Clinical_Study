# 당뇨병 및 관련 심대사질환 위험도의 자동화 종합 CT 평가

**원문**: Chang Y, Yoon SH, Kwon R, et al. Automated Comprehensive CT Assessment of the Risk of Diabetes and Associated Cardiometabolic Conditions. *Radiology* 2024;312(2):e233410. https://doi.org/10.1148/radiol.233410

**저널**: Radiology (RSNA), 2024년 8월호, Original Research · Computer Applications
**편집**: Deputy Editor Vicky Goh / Scientific Editor Shannyn Wolfe (AJE)
**같은 호 에디토리얼**: Pickhardt PJ 논평 수록

---

## 초록 (Abstract)

**배경**: 다양한 임상 적응증으로 시행된 CT는 심대사질환(cardiometabolic disease)을 예측할 잠재력을 가진다. 그러나 개별 CT 파라미터의 예측능은 아직 충분히 탐구되지 않았다.

**목적**: 완전자동화된 CT 유래 지표(marker)가 당뇨병 및 동반 심대사 합병증을 예측하는 능력을 평가한다.

**대상 및 방법**: 본 후향적 연구는 2012년 1월~2015년 12월 사이 불소-18 FDG PET/CT를 포함한 건강검진을 받은 25세 이상 한국 성인을 대상으로 하였다. 완전자동화 CT 지표는 내장지방·피하지방·근육·골밀도·간지방(모두 신장의 제곱, m²로 정규화)과 대동맥 석회화를 포함한다. 예측 성능은 단면 분석에서는 ROC 곡선하면적(AUC), 생존분석에서는 Harrell C-index로 평가하였다.

**결과**:
- 단면분석: 32,166명 (평균연령 45±6세, 남성 28,833명)
- 코호트분석: 27,298명 (평균연령 44±5세, 남성 24,820명)
- 당뇨병 유병률(baseline) 6%, 발생률(중앙값 7.3년 추적) 9%
- **내장지방지수(visceral fat index)**가 유병/신규발생 당뇨병 모두에서 가장 높은 예측성능을 보임
  - 유병 당뇨병 AUC: 남성 0.70 (95% CI 0.68–0.71), 여성 0.82 (95% CI 0.78–0.85)
  - 신규발생 당뇨병 C-index: 남성 0.68 (95% CI 0.67–0.69), 여성 0.82 (95% CI 0.77–0.86)
- 내장지방+근육면적+간지방분율+대동맥석회화 **조합** 시 예측성능 향상
  - C-index: 남성 0.69 (95% CI 0.68–0.71), 여성 0.83 (95% CI 0.78–0.87)
- 대사증후군 식별 AUC(내장지방지수): 남성 0.81 (95% CI 0.80–0.81), 여성 0.90 (95% CI 0.88–0.91)
- CT 유래 지표는 초음파 진단 지방간, 관상동맥칼슘점수(CAC) >100, 근감소증, 골다공증도 식별 (AUC 0.80~0.95)

**결론**: 자동화된 다장기 CT 분석은 당뇨병 및 기타 심대사 동반질환 고위험군을 식별하였다.

---

## 서론 (Introduction)

머신러닝·딥러닝 알고리즘의 적용은 영상 기반 체성분 분석 영역을 혁신하여 수작업 의존도를 낮췄다(1,2). 그러나 체계적 통합의 부재로 임상 실무에서의 완전한 활용은 아직 제한적이다(1).

1차 임상 적응증 이외 목적으로 촬영된 CT를 기회주의적으로(opportunistically) 활용한 데이터는 우연 발견 골다공증 선별(3,4)과 대동맥 석회화·내장지방/피하지방·근육량·간지방 함량 정량화(5–8)에서 가능성을 보여주었다. 제3요추(L3) 레벨의 단일 CT 영상만으로도 내장지방·피하지방·근육량·골밀도에 대한 정밀 정보를 얻을 수 있다(9). CT는 심혈관질환 이벤트와 전체원인 사망률을 예측할 잠재력이 있어, 환자 위험 계층화(risk stratification)에서 CT의 실질적 효용 가능성을 시사한다(10–12).

그러나 전통적으로 재래식 검사법(conventional modality)으로 진단되어 온 심대사질환에 대해, 개별 영상 파라미터의 예측능은 아직 충분히 탐구되지 않았다. 제2형 당뇨병(DM)은 상당한 동반질환·합병증을 동반하며 흔히 늦게 진단되는 흔한 대사질환이다(13). 근육량 및 지방 분포 차이를 포함한 체성분은 제2형 당뇨병 및 관련 합병증을 예측할 잠재력을 가진다(14,15).

본 연구의 목적은 건강검진 프로그램에 참여한 한국 성인을 대상으로, 완전자동화 CT 유래 지표가 유병/신규발생 당뇨병 및 동반 심대사질환(대사증후군, 근감소증, 골다공증, 지방간, 관상동맥칼슘[CAC])을 식별하는 능력을 평가하는 것이다.

---

## 대상 및 방법 (Materials and Methods)

본 후향적 연구는 기관윤리위원회(IRB No. KBSMC 2022-04-028) 승인을 받았으며, 서면동의는 면제되었다.

### 대상자 (Patients)

Kangbuk Samsung Health Study(주로 회사원 및 그 배우자로 구성된 전향적 코호트, 대한민국 산업안전보건법에 따라 건강검진을 받는 대상자)의 하위집단을 분석하였다. 포함기준은 2012~2015년 사이 종합건강검진의 일환으로 불소-18(¹⁸F) FDG PET/CT를 시행받은 자였다(제외기준 상세는 Appendix S1). 방사선 노출 및 비용 문제로 건강검진에서의 ¹⁸F-FDG PET/CT 시행에 대한 논란이 있음에도, 한국에서는 암 선별검사 목적으로 본 검사가 시행되고 있다.

### 측정 (Measurements)

**PET/CT 영상 획득**: 최소 8시간 금식 후, PET/CT 시스템(Discovery 600; GE HealthCare)으로 조영제 없이 흉복부(torso) ¹⁸F-FDG PET/CT를 촬영하였다. 영상 획득 파라미터는 Appendix S1에 제시.

**체성분·간·복부대동맥 석회화 분석**: PET/CT 촬영 중 얻어진 비조영 흉복부 CT 영상을 미국 FDA 승인 상용 소프트웨어(version 1.2.0.0, DeepCatch, Medical IP; http://www.medicalip.com)로 처리하여(Appendix S1, Fig S2), 골격근·피하지방·내장지방의 단면적(cm²)을 신장의 제곱(m²)으로 정규화한 지수로 측정하였다. 본 연구는 이들 면적 지수, 내장/피하지방 비율, L3 레벨의 해면골(trabecular) 밀도에 초점을 맞추었는데, 이는 전체 사망률에 대한 강력한 예측가치가 보고되었기 때문이다(16). 또한 소프트웨어는 간의 체적 밀도를 자동 산출하고, CT 영상으로부터 딥러닝 기반 영상 합성을 통해 MRI 양성자밀도지방분율(PDFF) 추정값을 산출하였다(Appendix S1). 대동맥 석회화는 환자 CT 영상의 전체 대동맥 영역에서 Agatston 칼슘점수 방식으로 산출하였다.

**당뇨병 및 기타 변수 정의**: 신체계측, 복부초음파 소견, 혈청 생화학 측정치는 ¹⁸F-FDG PET/CT 촬영 이전 건강검진 프로그램의 일환으로 체계적으로 수집되었다. 인구학적 특성, 건강행태(흡연·음주·신체활동), 병력, 약물 복용은 표준화된 자기기입식 설문지로 평가하였다(Appendix S1).

10시간 금식 후 채혈한 혈액 검체로 공복혈당·당화혈색소(HbA1c)·지질프로필을 측정하였다. 제2형 당뇨병은 다음 중 1개 이상을 만족하는 경우로 정의: ①공복혈당 ≥126mg/dL(7.0mmol/L), ②HbA1c ≥6.5%(≥48mmol/mol), ③현재 인슐린 또는 혈당강하제 복용 중. 대사증후군 및 (임피던스 분석으로 평가한) 근감소증은 표준 기준으로 정의하였다(Appendix S1) (17–19). CAC CT 기반 CAC 점수는 0–100 또는 100 초과로 분류하였으며, 100 초과는 스타틴 투여 적응증 판단의 임계값을 의미한다(Appendix S1) (20,21).

### 통계 분석 (Statistical Analysis)

제2형 당뇨병 유병률 기반 단면연구 및 초기 비당뇨군에서의 제2형 당뇨병 발생 기반 코호트연구에서, 평균±SD·백분율·(해당시) 중앙값 및 IQR로 대상자 특성을 요약하였다. Robust Poisson 회귀모형(22,23)을 이용하여 각 파라미터의 사분위수(최하위 사분위수를 기준군으로) 비교를 통한 제2형 당뇨병 및 각 임상질환의 유병률비(prevalence ratio, PR)와 95% CI를 산출하였고, Cox 비례위험모형을 이용하여 각 파라미터의 나머지 세 사분위수를 최하위 사분위수(기준군) 대비 비교한 당뇨병 발생의 위험비(hazard ratio, HR)와 95% CI를 산출하였다(상세는 Appendix S1). CT 유래 지표와 당뇨병 위험 간의 관계를 평가하기 위해, 표본 분포의 5·27.5·50·72.5·95백분위수를 매듭(knot)으로 하는 제한적 3차 스플라인(restricted cubic spline)을 이용하여 CT 유래 지표와 당뇨병 위험 간 농도-반응 관계를 유연하게 추정하였다.

표준 또는 실무에서 흔히 쓰이는 지표 대비 영상 파라미터의 당뇨병 및 관련 심대사질환 예측 성능은 ROC 곡선하면적(AUC)으로 평가하였다. 복수의 CT 유래 지표를 결합한 모형에서는, 다변량 로지스틱 회귀모형 적합 후 사후추정 명령어 "predict"를 이용해 당뇨병 예측확률을 산출하였다. 신체계측 지표와 CT 유래 지표 간 AUC 차이는 Stata(StataCorp)의 "roccomp" 명령어로 평가하였다. 최적 모형은 AUC를 최대화하는 CT 유래 지표의 조합으로 결정하였다. 코호트연구에서는 생존분석용으로 고안된 지표인 Harrell C-index(24)를 이용해 재래식 지표와 CT 유래 영상 파라미터의 예측능을 비교하였다. CT 유래 지표의 예측값은 미국당뇨병학회(ADA) 당뇨병 위험점수, Leicester 당뇨병 위험점수 등 재래식 위험인자 모형의 예측값과 비교하였다. 다중검정 문제를 보정하기 위해 Bonferroni 보정(다중검정 보정에 흔히 쓰이는 방법)을 적용하였다. 본 연구는 일상적으로 수집된 건강검진 자료를 이용하였으며, 표본크기는 연구기간 중 등록된 환자수로 결정되었다. 통계분석은 두 명의 저자(Y.C., S.R.)가 Stata(version 17.0; StataCorp)로 수행하였으며, 통계적 유의성은 P<.05로 정의하였다.

---

## 결과 (Results)

### 대상자 특성

2012~2015년 사이 ¹⁸F-FDG PET/CT를 시행받은 34,368명 중, 당뇨병 관련 데이터 결측·영상 저장 불완전·암 병력·간경변·낮은 추정사구체여과율(eGFR)로 2,202명을 제외하여 단면연구 대상자는 **32,166명**이 되었다(Figure 1, Appendix S1). 코호트연구에서는 초기 비당뇨군 27,298명을 2022년 12월 31일까지 추적하여 신규발생 당뇨병을 관찰하였다.

**[플로우차트 — 대상자 선정 과정]**
- ¹⁸F-FDG PET/CT를 포함한 종합건강검진을 받은 한국 성인: n=34,368
- 제외(n=2,202; 2개 이상 기준 중복 해당자 2명 포함):
  - 당뇨병 관련 데이터(혈당/HbA1c/약물력) 결측: n=1,448
  - 암 병력: n=702
  - PET/CT상 악성종양 의심: n=9
  - 초음파상 간경변: n=30
  - eGFR<30 mL/min/1.73m²: n=11
  - AI 프로그램 추론 오류 또는 영상 저장 불완전: n=4
- 단면분석 대상: n=32,166
- 추가 제외(코호트연구, n=4,868; 2개 이상 기준 중복 해당자 150명 포함):
  - 기저시점 유병 당뇨병: n=1,873
  - 2022년 말까지 추적 방문 없음: n=3,145
- 코호트연구 대상(기저시점 비당뇨군): n=27,298

단면연구 대상 32,166명(평균연령 45±6세, 남성 28,833명·여성 3,333명)(Table 1). 기저시점 제2형 당뇨병 전체 유병률은 6%(남성 6%, 여성 4%). 비당뇨군은 당뇨군보다 젊었으며(남성 평균 44±5세 vs 47±6세, P<.001; 여성 47±9세 vs 57±10세, P<.001), 고혈압 동반율(남성 17% vs 40%, P<.001; 여성 11% vs 39%, P<.001)과 지질강하제 복용율(남성 4% vs 23%, P<.001; 여성 5% vs 29%, P<.001)이 낮았다. 비당뇨군은 체질량지수(BMI)·허리둘레·임피던스 유래 지방지수도 더 낮았다(P<.001).

당뇨군은 임피던스 기반 골격근지수 및 CT 유래 L3 근육면적지수가 증가되어 있었으나 근육밀도는 감소되어 있었다. 초기에는 당뇨군에서 근육면적지수가 더 높았으나, 연령·BMI 보정 후에는 이 차이가 사라졌다. 연령·BMI 보정 평균 근육면적지수는 남성 비당뇨군 52.9(95% CI 52.8–52.9) vs 당뇨군 52.2(95% CI 52.0–52.4); 여성은 비당뇨군 39.8(95% CI 39.6–39.9) vs 당뇨군 40.0(95% CI 39.4–40.7)으로 무시할 만한 차이를 보였다. 당뇨군은 피하지방지수·내장지방지수가 더 높았고, 내장/피하지방비가 더 크며, 간밀도가 낮고, 대동맥 석회화가 더 많았다(P<.001). 연령·성별에 따른 CT 유래 파라미터 분포는 Tables S1–S3 참조.

코호트연구 대상 27,298명(평균연령 43.8±4.8세, 남성 24,820명)(Table S4). 비당뇨군과 신규발생 당뇨군 간 신체계측·임피던스·CT 유래 지표 차이 패턴은 유병 당뇨병 비교와 유사하였다.

### 표1. 유병 당뇨병 여부에 따른 기저시점 대상자 특성 (요약)

| 특성 | 남성-비당뇨 (n=27,090) | 남성-당뇨 (n=1,743) | 여성-비당뇨 (n=3,203) | 여성-당뇨 (n=130) | P(남성) | P(여성) |
|---|---|---|---|---|---|---|
| 연령(세) | 44.1±5.0 | 47.3±6.3 | 46.6±8.6 | 57.1±9.8 | <.001 | <.001 |
| 현재흡연 | 33.4% | 37.9% | 2.7% | 2.9% | <.001 | .89 |
| 고혈압 | 17.2% | 39.6% | 10.8% | 39.2% | <.001 | <.001 |
| 대사증후군 | 26.2% | 67.7% | 6.7% | 42.3% | <.001 | <.001 |
| BMI(kg/m²) | 24.5±2.7 | 25.9±3.2 | 22.3±3.0 | 24.7±4.1 | <.001 | <.001 |
| 허리둘레(cm) | 86.1±7.3 | 90.2±8.1 | 76.9±8.2 | 83.9±10.6 | <.001 | <.001 |
| L3 근육면적지수(중앙값) | 52.3 | 54.3 | 39.2 | 42.1 | <.001 | <.001 |
| L3 근육밀도(HU, 중앙값) | 45.1 | 44.2 | 38.3 | 34.5 | <.001 | <.001 |
| L3 내장지방지수(중앙값) | 41.0 | 54.3 | 17.7 | 42.7 | <.001 | <.001 |
| L3 피하지방지수(중앙값) | 42.8 | 45.0 | 56.9 | 69.7 | <.001 | <.001 |
| 내장/피하지방비(중앙값) | 0.9 | 1.2 | 0.3 | 0.6 | <.001 | <.001 |
| 간밀도(HU, 중앙값) | 55.4 | 50.6 | 55.8 | 51.4 | <.001 | <.001 |
| 간 PDFF(중앙값) | 6.8 | 8.7 | 6.2 | 8.2 | <.001 | <.001 |
| 대동맥석회화(Agatston, 중앙값) | 9.5 | 57.2 | 5.7 | 122.5 | <.001 | <.001 |

*(전체 항목은 원문 Table 1 참조. ASM=사지골격근량, HEPA=건강증진 신체활동, HU=Hounsfield unit, SC=피하, VS=내장/피하지방)*

### 유병 당뇨병 및 기타 심대사질환과 CT 유래 파라미터의 관계

Table 2는 남녀 각각에서 유병 당뇨병과의 관련성에 대한 재래식/CT 유래 파라미터의 AUC를 제시한다. 남성에서는 개별 지표인 CT 유래 L3 내장지방지수, 간 PDFF(또는 간밀도), 대동맥석회화가 모두 BMI보다 높은 AUC를 보였다(P<.001). 여성에서는 세 지표 모두 더 높은 AUC를 보였으나, 내장지방지수와 대동맥석회화만 통계적으로 유의한 차이를 보였다(P<.001). 내장지방면적+피하지방+간PDFF+대동맥석회화의 조합이 남녀 모두에서 당뇨병 유병에 대한 최고 AUC를 나타냈다: **남성 0.75(95% CI 0.74–0.77), 여성 0.85(95% CI 0.82–0.89)**.

연령·검진센터·검진연도로 보정 후, 당뇨병 유병률비(최하위 사분위수 대비 최상위 사분위수, Table S5)는 남녀 간 다른 패턴을 보였다. 남성에서는 내장지방이 가장 높은 유병률비를 보였고 간PDFF가 뒤를 이었다. 여성에서는 내장지방지수가 가장 높았고, 내장/피하지방비·대동맥석회화·간PDFF 순이었다. CT 내장지방지수는 임피던스 유래 체성분 지표(골격근지수·지방량지수·체지방률)보다 일관되게 우수한 당뇨병 유병 예측력을 보였다(Table S6).

Table 3은 임상표준을 기준으로 한 CT 유래 지표의 다양한 동반질환 식별 성능을 보여준다.
- **지방간(초음파 진단)**: 간PDFF AUC 남성 0.81(95% CI 0.80–0.81), 여성 0.80(95% CI 0.78–0.82)
- **CAC>100**: 대동맥석회화 AUC 남성 0.84(95% CI 0.80–0.87), 여성 **0.95**(95% CI 0.89–1.00)
- **근감소증**(임피던스 기반 골격근지수 대비): L3 근육면적지수 AUC 남성 0.90(95% CI 0.89–0.91), 여성 0.88(95% CI 0.83–0.94)
- **골다공증**(척추 DXA T-score < −2.5): L3 해면골밀도 AUC 양성 모두 0.9 초과
- **대사증후군**: 내장지방지수 AUC 남성 0.81(95% CI 0.80–0.81), 여성 0.90(95% CI 0.88–0.91)

### 표2. 유병 당뇨병 식별의 CT 유래 파라미터 판별 성능 (핵심 발췌)

| 변수 | 남성 AUC (95% CI) | 여성 AUC (95% CI) |
|---|---|---|
| BMI (기준) | 0.64 (0.62–0.65) | 0.68 (0.64–0.73) |
| 허리둘레 | 0.65 (0.63–0.66) | 0.71 (0.67–0.76) |
| SMI(ASM/height²) | 0.57 (0.55–0.58) | 0.57 (0.50–0.63) |
| L3 근육면적지수 | 0.58 (0.56–0.59) | 0.65 (0.60–0.70) |
| **L3 내장지방지수** | **0.70 (0.68–0.71)** | **0.82 (0.78–0.85)** |
| L3 피하지방지수 | 0.54 (0.53–0.56) | 0.66 (0.62–0.70) |
| 내장/피하지방비 | 0.67 (0.66–0.69) | 0.80 (0.76–0.84) |
| 간 PDFF | 0.68 (0.66–0.69) | 0.73 (0.68–0.77) |
| 대동맥석회화 | 0.67 (0.66–0.69) | 0.78 (0.74–0.82) |
| **내장지방+대동맥석회화+간PDFF+피하지방(최고조합)** | **0.75 (0.74–0.77)** | **0.85 (0.82–0.89)** |

*(45개 항목 다중검정 Bonferroni 보정 기준 α/45=0.001; 전체 항목은 원문 Table 2 참조)*

### 신규발생 당뇨병과 CT 유래 파라미터의 관계

183,651인년(person-years)의 추적관찰(중앙값 7.3년, 최대 10.8년) 동안, 초기 비당뇨군 27,298명 중 2,456명이 신규 제2형 당뇨병을 발생하였으며, 전체 발생률은 1,000인년당 13.4(95% CI 12.9–13.9) [여성 5.4(95% CI 4.4–6.7), 남성 14.1(95% CI 13.6–14.7)]였다.

**내장지방지수는 남녀 모두에서 신규발생 제2형 당뇨병에 대한 최고 예측 단일 영상 지표**였으며, 재래식 지표·임피던스 유래 체성분 지표·ADA/Leicester 임상 위험모형을 모두 상회하였다(Tables 4, S7). 영상 지표를 조합하면 AUC가 증가하였으며, **내장지방지수+근육면적지수+간PDFF+대동맥석회화** 조합이 최고 성능을 보였다[C-index 남성 0.69(95% CI 0.68–0.71), 여성 0.83(95% CI 0.78–0.87)].

연령·검진센터·검진연도 보정 후, 가장 높은 위험비는 내장지방지수에서 관찰되었다. 최상위 사분위수 대 최하위 사분위수(기준) 비교시 다변량보정 신규발생 당뇨병 위험비는 남성 5.19(95% CI 4.52–5.96), 여성 44.12(95% CI 10.58–184.0)였다(Tables S8, S9). 스플라인 회귀분석에서 남성은 내장지방지수 전 범위에서 당뇨병 위험이 가장 뚜렷하게 증가하였다(Fig S3). 여성은 내장지방지수 20 미만 구간에서 급격히 위험이 증가한 후 완만하게 지속 증가하였다(Fig S4).

### 표4. 신규발생 당뇨병 식별의 다양한 지표별 예측능 (핵심 발췌)

| 변수 | 남성 C-index (95% CI) | 여성 C-index (95% CI) |
|---|---|---|
| ADA 점수 | 0.64 (0.63–0.65) | 0.80 (0.75–0.85, 기준) |
| Leicester UK 당뇨병 위험점수 | 0.65 (0.64–0.67, 기준) | 0.78 (0.73–0.83) |
| BMI | 0.66 (0.65–0.67) | 0.77 (0.72–0.82) |
| SMI(ASM/height²) | 0.61 (0.60–0.62) | 0.69 (0.62–0.75) |
| L3 근육면적지수 | 0.61 (0.60–0.62) | 0.72 (0.67–0.78) |
| **L3 내장지방지수** | **0.68 (0.67–0.69)** | **0.82 (0.77–0.86)** |
| L3 피하지방지수 | 0.59 (0.58–0.60) | 0.76 (0.70–0.81) |
| 간 PDFF | 0.63 (0.61–0.64) | 0.67 (0.60–0.74) |
| 대동맥석회화 | 0.56 (0.55–0.58) | 0.60 (0.53–0.67) |
| **내장지방+근육면적+간PDFF+대동맥석회화(최고조합)** | **0.69 (0.68–0.71)** | **0.83 (0.78–0.87)** |

*(48개 항목 다중검정 Bonferroni 보정 기준 α/48=0.001; 전체 항목은 원문 Table 4 참조)*

---

## 고찰 (Discussion)

심대사질환에 대한 개별 CT 파라미터의 예측능은 아직 충분히 탐구되지 않았다. 본 한국 성인 코호트연구에서, 자동화 CT 유래 체성분 파라미터는 남녀 모두에서 재래식 신체계측 및 임상 위험모형을 능가하는 우수한 유병/신규발생 당뇨병 예측인자였다. 내장지방지수는 유병 당뇨병에 대해 AUC 남성 0.70(95% CI 0.68–0.71)·여성 0.82(95% CI 0.78–0.85), 신규발생 당뇨병에 대해 C-index 남성 0.68(95% CI 0.67–0.69)·여성 0.82(95% CI 0.77–0.86)를 나타냈다. 내장지방·근육면적·간 양성자밀도지방분율·대동맥석회화를 결합한 CT 유래 지표 조합은 제2형 당뇨병 위험예측 모형의 성능을 향상시켰으며, 해당 동반질환들을 정확히 식별하였다.

제2형 당뇨병 환자에서는 초기 진단 시점에 동반질환·당뇨병 관련 합병증이 흔히 함께 발견되며, 이는 약물 선택에 영향을 미친다(13). CT 유래 영상 지표는 보다 맞춤화되고 정밀한 당뇨병 치료전략을 가능하게 할 수 있다. CT 유래 체성분과 당뇨병 진단을 연결지은 선행연구(14,25)를 바탕으로, 정기적인 당뇨병 평가가 이루어지는 실제 건강검진 환경에서 얻은 본 연구 결과는, Pickhardt가 이전에 권고한 바와 같이(12) 기회주의적 CT 선별을 통한 예방적 관리 및 위험평가 향상에 CT 영상이 기여할 잠재력을 부각시킨다.

본 연구에서 CT 유래 내장지방 단독으로도 재래식 제2형 당뇨병 예측모형을 능가하였으며, 다른 CT 유래 영상 지표와 결합 시 예측성능이 더욱 향상되었다. 이 지수는 대사증후군도 식별하였다(26). 또한, 허리-엉덩이비 또는 허리둘레로 측정한 복부비만은 BMI보다 당뇨병성 망막병증·당뇨병성 신장질환·심혈관질환을 더 잘 예측한다(27). CT가 내장지방을 정밀 정량화하는 기준표준(reference standard)임을 고려하면, 정확한 내장비만 평가는 당뇨병 위험과 그 합병증을 모두 예측할 수 있을 것이다. 제2형 당뇨병 환자의 55–70% 이상이 대사기능이상 관련 지방간질환(MASLD)도 동반하며(28), 간 관련 합병증 위험이 높아 이 질환에 대한 정기 선별검사를 권고하는 임상 가이드라인이 마련되어 있다(13). 제2형 당뇨병은 근감소증 위험 증가와 관련이 있다(29). 본 연구에서 제2형 당뇨병 환자는 CT상 근육량은 더 많았으나 근육밀도는 더 낮았으며, 이 차이는 BMI·연령 보정 후 사라졌다. 근감소성 지방증(myosteatosis)을 시사하는 낮은 근육밀도의 식별은 중요한데, 이는 근력 저하 및 사망률 증가와 부정적으로 연관되어 있어(30), 근육 건강을 종합적으로 평가하려면 근육량과 근육질(quality)을 모두 평가할 필요성을 뒷받침한다.

심혈관 사망률과 관련이 있고 CAC와 심혈관질환 위험인자를 공유하는(31) 대동맥석회화는, 본 연구에서 CAC>100을 정확히 식별하였다. 이는 강도 높은 치료 개시가 필요한(21,32) 죽상경화성 심혈관질환 고위험군(1,000인년당 20건 초과)의 제2형 당뇨병 환자를 식별할 잠재력을 가진다. 본 연구는 대동맥석회화를 신규발생 당뇨병의 예측인자로 확인하였으며, 이는 혈관 석회화가 심혈관질환 위험을 포괄하는 노화·전반적 건강상태의 통합적 지표로서 당뇨병 발생에 영향을 미침을 시사한다(33,34).

본 연구에서 CT 유래 지표는 남성보다 여성에서 당뇨병·대사증후군과 더 강한 관련성을 보였는데, 이는 여성에서 더 높은 예측정확도를 보고한 Pickhardt 등(26)의 결과와 일치한다. 이러한 성별 차이는 심대사질환의 성별 이형적(sexually dimorphic) 위험인자와 관련이 있을 수 있다(35). 폐경 전 여성은 에스트로겐의 영향으로 둔부-대퇴부 지방조직을 더 많이 축적하는 경향이 있으며, 이는 더 나은 인슐린 감수성 및 내장지방보다 피하지방으로의 지방저장 선호와 관련이 있다(35). 이러한 에스트로겐 관련 지방분포는 여성에서 CT 영상의 당뇨병 위험 예측능이 더 우수한 이유를 설명할 수 있으며(36), 위험평가에서 성별 특이 요인 및 CT로부터 얻은 정확한 지방분포 데이터를 고려하는 것의 중요성을 부각시킨다.

### 한계점 (Limitations)

1. **당뇨병 진단**: 반복검사를 요구하는 일반적 임상 실무와 달리, 공복혈당·HbA1c 단회 측정으로 제2형 당뇨병을 진단하였다. 다만 HbA1c는 스트레스·운동 등 즉각적 영향에 대한 저항성과 신뢰할 만한 검사전(preanalytical) 안정성을 지녀 제2형 당뇨병 발생률을 정확히 판정하는 데 도움이 된다(37). 유병/신규발생 당뇨병 대상자의 최저연령은 각각 37세·39세로, 이 연령군에서 제1형 당뇨병 가능성은 낮다.
2. **췌장지방 미분석**: 당뇨병 예측인자인(14) 췌장지방을 분석하지 못하였으며, 이는 향후 연구 방향을 시사한다.
3. **일반화 가능성 제한**: 젊고 중년의 한국인에 초점을 맞추어, 더 넓은 인구집단을 대표하지 못할 가능성이 있다. 또한 간초음파는 코호트 내에서 거의 전원 시행되었으나, 골밀도·CAC 검사 등은 참가자 선호에 따라 시행되는 경우가 많아 선택편향이 발생했을 가능성이 있다. 다만 참가자가 검사를 선택할 당시 이들 질환과 CT 유래 지표 간 연관성이 예상되지 않았다는 점에서, CAC·골다공증 진단에 대한 CT 유래 영상 지표의 진단적 유용성에 미치는 영향은 미미할 것으로 예상된다. 향후 연구는 비선택(unselected) 인구집단을 포함하여 본 연구 결과를 다양한 인구집단에서 검증해야 한다.

### 결론 (Conclusion)

CT 유래 파라미터, 특히 **내장지방 면적지수**는 남녀 모두에서 제2형 당뇨병(DM) 예측에 있어 전통적 방법을 능가하였다. 내장지방·피하지방 면적, 근육면적, 간지방분율, 대동맥석회화를 포함한 CT 유래 지표의 조합은 제2형 당뇨병 위험예측 성능을 향상시켰고, 여러 당뇨병 관련 동반질환에 대한 선별을 용이하게 하여 맞춤형 위험 계층화를 가능하게 하였다. 방사선 노출 감소 및 표적화된 다장기 평가를 통해 더 효율적이고 안전한 접근법을 달성하는 것은 여전히 필요하며, 본 연구 결과의 실제 임상 적용 가능성을 고려할 때는 신중을 기해야 한다.

---

## 저자 정보 및 이해상충

**공동 제1저자**: Yoosoo Chang, Soon Ho Yoon (동등 기여)

**소속기관**: Kangbuk Samsung Hospital 코호트연구센터·직업환경의학교실(성균관대 의대), 삼성융합의과학원 임상연구설계평가학과, 서울대병원 영상의학과, MEDICAL IP 연구개발부, 성균관대 의대 의학연구원, Kangbuk Samsung Hospital 핵의학교실, 에딘버러대 Usher Institute, 사우샘프턴대 영양대사학과, NIHR 사우샘프턴 생의학연구센터

**연구비**: SKKU Excellence in Research Award Research Fund(2022), 한국연구재단(NRF-2021R1A2C1012626), NIHR Southampton Biomedical Research Centre(NIHR 203319, C.D.B. 일부 지원). 연구비 지원기관은 연구설계·자료수집·분석·해석·논문작성에 관여하지 않았음.

**이해상충 주요사항**:
- S.H.Y.: MEDICAL IP 주식 및 스톡옵션 보유
- J.M.K., H.J.C.: MEDICAL IP 직원
- C.D.B.: Echosens로부터 소속기관에 지급된 연구비
- 그 외 저자: 관련 이해상충 없음

**감사의 글**: Kangbuk Samsung Health Study의 다른 연구자·스태프·참가자들에게 감사를 표함.

---

## 참고문헌 (References)

1. Pickhardt PJ, Summers RM, Garrett JW, et al. Opportunistic Screening: Radiology Scientific Expert Panel. *Radiology* 2023;307(5):e222044.
2. Bajwa J, Munir U, Nori A, Williams B. Artificial intelligence in healthcare: transforming the practice of medicine. *Future Healthc J* 2021;8(2):e188–e194.
3. Lee SJ, Graffy PM, Zea RD, Ziemlewicz TJ, Pickhardt PJ. Future Osteoporotic Fracture Risk Related to Lumbar Vertebral Trabecular Attenuation Measured at Routine Body CT. *J Bone Miner Res* 2018;33(5):860–867.
4. Pickhardt PJ, Pooler BD, Lauder T, del Rio AM, Bruce RJ, Binkley N. Opportunistic screening for osteoporosis using abdominal computed tomography scans obtained for other indications. *Ann Intern Med* 2013;158(8):588–595.
5. Graffy PM, Liu J, Pickhardt PJ, Burns JE, Yao J, Summers RM. Deep learning-based muscle segmentation and quantification at abdominal CT: application to a longitudinal adult screening cohort for sarcopenia assessment. *Br J Radiol* 2019;92(1100):20190327.
6. Graffy PM, Liu J, O'Connor S, Summers RM, Pickhardt PJ. Automated segmentation and quantification of aortic calcification at abdominal CT: application of a deep learning-based algorithm to a longitudinal screening cohort. *Abdom Radiol (NY)* 2019;44(8):2921–2928.
7. Pickhardt PJ, Graffy PM, Reeder SB, Hernando D, Li K. Quantification of Liver Fat Content With Unenhanced MDCT: Phantom and Clinical Correlation With MRI Proton Density Fat Fraction. *AJR Am J Roentgenol* 2018;211(3):W151–W157.
8. Lee YS, Hong N, Witanto JN, et al. Deep neural network for automatic volumetric segmentation of whole-body CT images for body composition assessment. *Clin Nutr* 2021;40(8):5038–5046.
9. Lee JH, Choi SH, Jung KJ, Goo JM, Yoon SH. High visceral fat attenuation and long-term mortality in a health check-up population. *J Cachexia Sarcopenia Muscle* 2023;14(3):1495–1507.
10. Pickhardt PJ, Graffy PM, Zea R, et al. Automated CT biomarkers for opportunistic prediction of future cardiovascular events and mortality in an asymptomatic screening population: a retrospective cohort study. *Lancet Digit Health* 2020;2(4):e192–e200.
11. Lee MH, Zea R, Garrett JW, Graffy PM, Summers RM, Pickhardt PJ. Abdominal CT Body Composition Thresholds Using Automated AI Tools for Predicting 10-year Adverse Outcomes. *Radiology* 2023;306(2):e220574.
12. Pickhardt PJ. Value-added Opportunistic CT Screening: State of the Art. *Radiology* 2022;303(2):241–254.
13. ElSayed NA, Aleppo G, Aroda VR, et al; American Diabetes Association. 4. Comprehensive Medical Evaluation and Assessment of Comorbidities: Standards of Care in Diabetes-2023. *Diabetes Care* 2023;46(Suppl 1):S49–S67.
14. Tallam H, Elton DC, Lee S, Wakim P, Pickhardt PJ, Summers RM. Fully Automated Abdominal CT Biomarkers for Type 2 Diabetes Using Deep Learning. *Radiology* 2022;304(1):85–95.
15. Al-Sofiani ME, Ganji SS, Kalyani RR. Body composition changes in diabetes and aging. *J Diabetes Complications* 2019;33(6):451–459.
16. Liu D, Garrett JW, Lee MH, Zea R, Summers RM, Pickhardt PJ. Fully automated CT-based adiposity assessment: comparison of the L1 and L3 vertebral levels for opportunistic prediction. *Abdom Radiol (NY)* 2023;48(2):787–795.
17. Chang Y, Kim BK, Yun KE, et al. Metabolically-healthy obesity and coronary artery calcification. *J Am Coll Cardiol* 2014;63(24):2679–2686.
18. Alberti KG, Eckel RH, Grundy SM, et al; International Association for the Study of Obesity. Harmonizing the metabolic syndrome. *Circulation* 2009;120(16):1640–1645.
19. Janssen I, Heymsfield SB, Ross R. Low relative skeletal muscle mass (sarcopenia) in older persons is associated with functional impairment and physical disability. *J Am Geriatr Soc* 2002;50(5):889–896.
20. Grundy SM, Stone NJ, Bailey AL, et al. 2018 AHA/ACC/... Guideline on the Management of Blood Cholesterol. *Circulation* 2019;139(25):e1082–e1143.
21. Hecht HS, Blaha MJ, Kazerooni EA, et al. CAC-DRS: Coronary Artery Calcium Data and Reporting System. *J Cardiovasc Comput Tomogr* 2018;12(3):185–191.
22. Zou G. A modified poisson regression approach to prospective studies with binary data. *Am J Epidemiol* 2004;159(7):702–706.
23. Barros AJ, Hirakata VN. Alternatives for logistic regression in cross-sectional studies. *BMC Med Res Methodol* 2003;3(1):21.
24. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. *Stat Med* 2011;30(10):1105–1117.
25. Zou X, Zhou X, Li Y, et al. Gender-specific data-driven adiposity subtypes using deep-learning-based abdominal CT segmentation. *Obesity (Silver Spring)* 2023;31(6):1600–1609.
26. Pickhardt PJ, Graffy PM, Zea R, et al. Utilizing Fully Automated Abdominal CT-Based Biomarkers for Opportunistic Screening for Metabolic Syndrome in Adults Without Symptoms. *AJR Am J Roentgenol* 2021;216(1):85–92.
27. Wan H, Wang Y, Xiang Q, et al. Associations between abdominal obesity indices and diabetic complications: Chinese visceral adiposity index and neck circumference. *Cardiovasc Diabetol* 2020;19(1):118.
28. Lomonaco R, Godinez Leiva E, Bril F, et al. Advanced Liver Fibrosis Is Common in Patients With Type 2 Diabetes Followed in the Outpatient Setting: The Need for Systematic Screening. *Diabetes Care* 2021;44(2):399–406.
29. Liccini A, Malmstrom TK. Frailty and Sarcopenia as Predictors of Adverse Health Outcomes in Persons With Diabetes Mellitus. *J Am Med Dir Assoc* 2016;17(9):846–851.
30. Nachit M, Horsmans Y, Summers RM, Leclercq IA, Pickhardt PJ. AI-based CT Body Composition Identifies Myosteatosis as Key Mortality Predictor in Asymptomatic Adults. *Radiology* 2023;307(5):e222008.
31. Wilson PW, Kauppila LI, O'Donnell CJ, et al. Abdominal aortic calcific deposits are an important predictor of vascular morbidity and mortality. *Circulation* 2001;103(11):1529–1534.
32. Grundy SM, Stone NJ, Bailey AL, et al. 2018 AHA/ACC/... Guideline on the Management of Blood Cholesterol. *J Am Coll Cardiol* 2019;73(24):e285–e350.
33. Shaw LJ, Raggi P, Berman DS, Callister TQ. Coronary artery calcium as a measure of biologic age. *Atherosclerosis* 2006;188(1):112–119.
34. Handy CE, Desai CS, Dardari ZA, et al. The Association of Coronary Artery Calcium With Noncardiovascular Disease: The Multi-Ethnic Study of Atherosclerosis. *JACC Cardiovasc Imaging* 2016;9(5):568–576.
35. Mauvais-Jarvis F, Bairey Merz N, Barnes PJ, et al. Sex and gender: modifiers of health, disease, and medicine. *Lancet* 2020;396(10250):565–582.
36. Goossens GH, Jocken JWE, Blaak EE. Sexual dimorphism in cardiometabolic health: the role of adipose tissue, muscle and liver. *Nat Rev Endocrinol* 2021;17(1):47–66.
37. Bonora E, Tuomilehto J. The pros and cons of diagnosing diabetes with A1C. *Diabetes Care* 2011;34(Suppl 2):S184–S190.

---

## 약어 (Abbreviations)

| 약어 | 원어 | 국문 |
|---|---|---|
| AUC | area under the receiver operating characteristic curve | ROC 곡선하면적 |
| BMI | body mass index | 체질량지수 |
| CAC | coronary artery calcium | 관상동맥칼슘 |
| DM | diabetes mellitus | 당뇨병 |
| FDG | fluorodeoxyglucose | 플루오로데옥시글루코스 |
| PDFF | proton density fat fraction | 양성자밀도지방분율 |
| ASM | appendicular skeletal muscle mass | 사지골격근량 |
| SMI | skeletal muscle index | 골격근지수 |
| HU | Hounsfield unit | 하운스필드 단위 |
| L3 | third lumbar vertebra | 제3요추 |
| SC | subcutaneous | 피하 |
| VS | visceral to subcutaneous fat | 내장/피하지방비 |
| HEPA | health-enhancing physical activity | 건강증진 신체활동 |
| eGFR | estimated glomerular filtration rate | 추정사구체여과율 |
| DXA | dual-energy x-ray absorptiometry | 이중에너지 X선 흡수계측법 |
| ADA | American Diabetes Association | 미국당뇨병학회 |

---

*본 문서는 docs\논문 m&m 레퍼런스.pdf(Radiology 2024;312(2):e233410)를 국문 번역한 것으로, 표는 지면 관계상 핵심 수치 위주로 발췌하였음. 전체 수치(Table S1–S9, Fig S1–S4 등 Supplemental material 포함)는 원문 PDF 참조.*
