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
| :-----: | :---------: | :--: | :-------: |
| BDI     | Baltic Dry Index | Points | Daily |
| SSEC    | Shanghai Stock Exchange Composite | Points | Daily |
| Nasdaq  | National Association of Securities Dealers Automated Quotations | Points | Daily |
| NOSB    | Manufacturers' New Orders: Ships and Boats | Millions of Dollars | Monthly |
| CRB     | Commodity Research Bureau Index | Points | Daily |
| Fed     | United States Fed Funds Rate | Percent | Monthly |
| OECD GDP growth | OECD GDP growth rate | Percent | Annual |
| EPU     | Economic Policy Uncertainty (USA, China) | Points | Monthly |
| Special Events | Subprime Mortgage, Covid-19, etc. | - | Occasional |
| Seasonality | Seasonal factors affecting shipping demand | - | Annual |


# 프로젝트 타임테이블

본 프로젝트는 4월 한달간 진행 하였으며, 이후 5월, 6월 추가 수정 작업이 예정되어 있습니다.

| 단계 | 설명 | 예상 소요 시간 | 시작 날짜 | 완료 날짜 |
|:---:|:---|:---:|:---:|:---:|
| 1 | 데이터 수집 및 전처리 | 1주 | 2023-04-03 | 2023-04-10 |
| 2 | 모델 학습 및 하이퍼파라미터 튜닝 | 1주 | 2023-04-10 | 2023-04-17 |
| 3 | 모델 평가 및 결과 분석 | 1주 | 2023-04-11 | 2023-04-21 |
| 4 | 문서 작성 및 발표 자료 준비 | 1주 | 2023-04-21 | 2023-04-27 |
| 5 | 최종 발표 | - | 2023-04-28 | 2023-04-28 |

## 주요 이벤트 및 마일스톤

- **데이터 수집 및 전처리 완료**: 2023-04-10
- **모델 학습 및 하이퍼파라미터 튜닝 완료**: 2023-04-17
- **모델 평가 및 결과 분석 완료**: 2023-04-21
- **문서 작성 및 발표 자료 준비 완료**: 2023-04-27
- **최종 발표**: 2023-04-28

## Reference 
Module : https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
Paper : https://arxiv.org/abs/1912.09363
