# Baltic Dry Index (BDI) 예측을 위한 Temporal Fusion Transformer (TFT) 모델

본 프로젝트는 시계열 예측을 위한 Temporal Fusion Transformer (TFT) 모델을 사용하여 Baltic Dry Index (BDI)를 예측하는 방법을 제공합니다. 

## Baltic Dry Index (BDI) 소개

Baltic Dry Index (BDI)는 세계 해상 화물 운송 시장의 건화물 운송비를 나타내는 경제 지표입니다. BDI는 Capesize, Panamax, Supramax 및 Handysize 선박의 평균 운임을 기반으로 산출되며, 해운업계의 수요와 공급에 대한 중요한 지표로 사용됩니다. BDI가 급등하면 선박 용적이 부족하거나 수요가 높아지는 것을 나타내고, 반대로 BDI가 급락하면 선박 용적이 과잉되거나 수요가 낮아지는 것을 나타냅니다.

## 시계열 예측을 위한 Temporal Fusion Transformer (TFT) 모델

Temporal Fusion Transformer (TFT)는 시계열 예측 작업을 위해 특별히 설계된 최신 딥러닝 모델입니다. 이 모델은 Bryan Lim 등이 작성한 "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" 논문에서 소개되었습니다. (https://arxiv.org/abs/1912.09363)

## 모델 개요

TFT는 순환 신경망(RNN), 합성곱 신경망(CNN) 및 어텐션 메커니즘과 같은 다양한 신경망 구조의 장점을 결합하여 복잡한 시계열 데이터를 모델링합니다. 이 모델은 다양한 시간 해상도를 가진 여러 입력 특징을 처리할 수 있으며, 다중 수평 예측을 생성할 수 있습니다.

TFT 모델의 주요 구성 요소는 다음과 같습니다:

1. **입력 인코딩**: TFT는 게이트가 있는 선형 유닛(GLU)과 dilated convolution의 조합을 사용하여 각 입력 특징을 개별적으로 처리합니다. 이를 통해 모델은 복잡한 시간 패턴을 학습하고 다양한 특징 유형에 적응할 수 있습니다.

2. **시간적 융합**: 모델은 각각 장기 및 단기 시간 종속성을 모델링하기 위해 셀프 어텐션 및 로컬 어텐션 메커니즘의 조합을 사용합니다. 이를 통해 모델은 전역 및 로컬 컨텍스트를 효과적으로 포착할 수 있습니다.

3. **출력 디코딩**: TFT는 인코딩된 입력 특징의 다양한 부분에 주목하여 다중 수평 예측을 생성하는 멀티 헤드 어텐션 메커니즘을 사용합니다. 이로 인해 더 정확하고 해석 가능한 예측이 가능해집니다.

## 모델의 장점

- **해석 가능성**: TFT는 다양한 수준에서의 특징 중요도를 제공하여 사용자가 모델의 의사결정 과정을 이해할 수 있습니다.

- **유연성**: 이 모델은 다양한 시간 해상도를 가진 여러 입력 특징을 처리할 수 있으며, 다양한 시계열 예측 작업에 적용할 수 있습니다.

- **확장성**: TFT는 병렬 컴퓨팅 리소스를 사용하여 대규모 데이터셋에서 효율적으로 학습할 수 있어 실제 애플리케이션에 적합합니다.

## Dataset
  
| Feature | Description | Unit | Frequency |
| ------- | ----------- | ---- | --------- |
| BDI | Baltic Dry Index | Points | Daily |
| Freight Rates | Freight rates for major shipping routes | USD/ton | Daily |
| Global Iron Ore Prices | Iron ore prices in the global market | USD/ton | Daily |
| Crude Oil Prices | Crude oil prices in the global market | USD/barrel | Daily |
| Vessel Supply | Number of available vessels for shipping | Count | Daily |
| Port Congestion | Number of congested ports worldwide | Count | Daily |
| Shipbuilding Orders | Number of new shipbuilding orders placed | Count | Monthly |
| Economic Indicators | GDP, CPI, PPI, and other economic indicators for major economies | Various | Monthly/Quarterly |
| Seasonality | Seasonal factors affecting shipping demand | - | Annual |

# 프로젝트 타임테이블

본 프로젝트는 주요 단계별로 구성되어 있으며, 각 단계의 예상 소요 시간과 완료 날짜를 아래 타임테이블에 제시합니다.

| 단계 | 설명 | 예상 소요 시간 | 시작 날짜 | 완료 날짜 |
|:---:|:---|:---:|:---:|:---:|
| 1 | 데이터 수집 및 전처리 | 2주 | 2023-05-01 | 2023-05-14 |
| 2 | 모델 학습 및 하이퍼파라미터 튜닝 | 3주 | 2023-05-15 | 2023-06-04 |
| 3 | 모델 평가 및 결과 분석 | 1주 | 2023-06-05 | 2023-06-11 |
| 4 | 문서 작성 및 발표 자료 준비 | 1주 | 2023-06-12 | 2023-06-18 |
| 5 | 최종 발표 | - | 2023-06-19 | 2023-06-19 |

## 주요 이벤트 및 마일스톤

- **데이터 수집 및 전처리 완료**: 2023-05-14
- **모델 학습 및 하이퍼파라미터 튜닝 완료**: 2023-06-04
- **모델 평가 및 결과 분석 완료**: 2023-06-11
- **문서 작성 및 발표 자료 준비 완료**: 2023-06-18
- **최종 발표**: 2023-06-19
