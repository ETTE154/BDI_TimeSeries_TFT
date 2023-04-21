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
| :-------: | :-----------: | :----: | :---------: |
| BDI | Baltic Dry Index | Points | Daily |
| Freight Rates | Freight rates for major shipping routes | USD/ton | Daily |
| Global Iron Ore Prices | Iron ore prices in the global market | USD/ton | Daily |
| Crude Oil Prices | Crude oil prices in the global market | USD/barrel | Daily |
| Vessel Supply | Number of available vessels for shipping | Count | Daily |
| Port Congestion | Number of congested ports worldwide | Count | Daily |
| Shipbuilding Orders | Number of new shipbuilding orders placed | Count | Monthly |
| Economic Indicators | GDP, CPI, PPI, and other economic indicators for major economies | Various | Monthly/Quarterly |
| Seasonality | Seasonal factors affecting shipping demand | - | Annual |

# 사용 모듈 설명
  - TemporalFusionTransformer 클래스는 PyTorch Forecasting 라이브러리에서 시계열 예측을 위한 Temporal Fusion Transformer (TFT) 모델을 구현한 것입니다.
  TFT는 다양한 특성을 갖는 시계열 데이터를 처리하고 예측하는 데 효과적인 딥러닝 아키텍처입니다.


##  PyTorch Forecasting

### Temporal Fusion Transformer

**Bases**: BaseModelWithCovariates

Temporal Fusion Transformer는 `BaseModelWithCovariates`를 상속한 클래스로, 시계열 예측을 위해 설계되었습니다. 이 클래스는 가능한 경우 `from_dataset()` 메서드를 사용하는 것이 좋습니다.

이 구현은 "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" 논문에 기반하며, 벤치마크에서 Amazon의 DeepAR보다 36-69% 우수한 성능을 보입니다.

### Enhancements compared to the original implementation

원래 구현에 비해 추가된 기능과 개선 사항은 다음과 같습니다 (기본 모델을 통해 추가된 기능 외에도 단조 제약 조건 등):

1. 정적 변수가 연속일 수 있습니다.
2. 여러 범주형 변수가 EmbeddingBag을 사용하여 요약될 수 있습니다.
3. 샘플 별로 인코더와 디코더 길이가 다릅니다.
4. 범주형 임베딩이 변수 선택 네트워크에 의해 변환되지 않습니다 (중복 작업이기 때문입니다).
5. 변수 선택 네트워크의 변수 차원이 선형 보간을 통해 확장되어 매개변수 수를 줄입니다.
6. 인코더와 디코더 사이에서 변수 선택 네트워크의 비선형 변수 처리가 공유될 수 있습니다 (기본적으로는 공유되지 않습니다).

### Hyperparameter Tuning

이 클래스의 하이퍼파라미터는 `optimize_hyperparameters()` 메서드를 사용하여 조정할 수 있습니다. 이를 통해 모델 성능을 최적화할 수 있습니다.

###  
| 매개변수                         | 기본값        | 설명                                                         |
|--------------------------------|-------------|------------------------------------------------------------|
| hidden_size                     | 16          | LSTM과 Fully Connected 레이어의 hidden size                 |
| lstm_layers                     | 1           | LSTM 레이어의 개수                                          |
| dropout                         | 0.1         | 드롭아웃 비율                                                |
| output_size                     | 7           | 출력 크기 또는 각 출력 특성에 대한 출력 크기 목록            |
| loss                            | None        | 손실 함수                                                   |
| attention_head_size             | 4           | Multi-head Attention 레이어의 헤드 크기                      |
| max_encoder_length              | 10          | 인코더의 최대 시퀀스 길이                                    |
| static_categoricals, ...        | []          | 입력 데이터의 다양한 유형의 특성들을 정의하는 리스트들      |
| categorical_groups              | {}          | 범주형 그룹화를 정의하는 딕셔너리                           |
| hidden_continuous_size          | 8           | 연속적인 변수의 hidden size                                 |
| hidden_continuous_sizes         | {}          | 각 연속적인 변수에 대한 hidden size를 정의하는 딕셔너리    |
| embedding_sizes                 | {}          | 각 범주형 변수에 대한 임베딩 크기를 정의하는 딕셔너리      |
| embedding_paddings              | []          | 패딩이 필요한 범주형 변수의 목록                            |
| embedding_labels                | {}          | 범주형 변수의 레이블을 정의하는 딕셔너리                   |
| learning_rate                   | 0.001       | 학습률                                                     |
| log_interval, log_val_interval  | -1, None    | 로그 간격 및 검증 로그 간격                                 |
| log_gradient_flow               | False       | 그래디언트 흐름을 기록할지 여부                             |
| reduce_on_plateau_patience      | 1000        | 학습률을 줄이기 위한 ReduceLROnPlateau 스케줄러의 patience |
| monotone_constaints             | {}          | 예측이 단조롭게 증가하거나 감소하도록 강제하는 제약 조건    |
| share_single_variable_networks  | False       | 모든 변수에 대해 동일한 가중치를 공유하는 네트워크 사용 여부|
| causal_attention                | True        | 인코더-디코더 어텐션에 인과 관계를 갖도록 할지 여부          |
| logging_metrics                 | None        | 모델 학습 중 로깅할 메트릭을 정의하는 ModuleList            |
| **kwargs                        | -           | 추가 인자를 전달하기 위한 키워드 인수                        |

### create_log(x, y, out, batch_idx, **kwargs)[source]
  - **훈련 및 검증 단계에서 사용되는 로그를 생성합니다.**

  - **Parameters:**
    - x (Dict[str, torch.Tensor]) - 데이터 로더에 의해 네트워크로 전달된 x
    - y (Tuple[torch.Tensor, torch.Tensor]) - 데이터 로더에 의해 손실 함수로 전달된 y
    - out (Dict[str, torch.Tensor]) - 네트워크의 출력
    - batch_idx (int) - 배치 번호
    - prediction_kwargs (Dict[str, Any], optional) - to_prediction()에 전달할 인수입니다. 기본값은 {}입니다.
    - quantiles_kwargs (Dict[str, Any], optional) - to_quantiles()에 전달할 인수입니다. 기본값은 {}입니다.

    - **Returns:**
<!-- 훈련 및 검증 단계에서 반환되는 로그 사전 -->

Return type:
Dict[str, Any]

expand_static_context(context, timesteps)[source]
정적 컨텍스트에 시간 차원을 추가합니다.

forward(x: Dict[str, Tensor]) → Dict[str, Tensor][source]
입력 차원: n_samples x time x variables

classmethod from_dataset(dataset: TimeSeriesDataSet, allowed_encoder_known_variable_names: List[str] | None = None, **kwargs)[source]
데이터셋에서 모델을 생성합니다.

Parameters:
dataset – 시계열 데이터셋
allowed_encoder_known_variable_names – 인코더에서 허용되는 알려진 변수 목록, 기본값은 모두 허용됩니다.
**kwargs – 모델의 하이퍼파라미터와 같은 추가 인수(__init__() 참조)

Returns:
TemporalFusionTransformer


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
