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

### create_log(x, y, out, batch_idx, **kwargs)
  - **훈련 및 검증 단계에서 사용되는 로그를 생성합니다.**

  - **Parameters:**
    - **x** (Dict[str, torch.Tensor]) - 데이터 로더에 의해 네트워크로 전달된 x
    - **y** (Tuple[torch.Tensor, torch.Tensor]) - 데이터 로더에 의해 손실 함수로 전달된 y
    - **out** (Dict[str, torch.Tensor]) - 네트워크의 출력
    - **batch_idx** (int) - 배치 번호
    - **prediction_kwargs** (Dict[str, Any], optional) - to_prediction()에 전달할 인수입니다. 기본값은 {}입니다.
    - **quantiles_kwargs** (Dict[str, Any], optional) - to_quantiles()에 전달할 인수입니다. 기본값은 {}입니다.

    ### **Returns:**
      - 훈련 및 검증 단계에서 반환되는 로그 사전

    ### **Return type:**
      - Dict[str, Any]

### expand_static_context(context, timesteps)
  - **정적 컨텍스트에 시간 차원을 추가합니다.**

### forward(x: Dict[str, Tensor]) → Dict[str, Tensor]
  - **입력 차원: n_samples x time x variables**

### classmethod from_dataset(dataset: TimeSeriesDataSet, allowed_encoder_known_variable_names: List[str] | None = None, **kwargs)

- **Parameters:**
    - **dataset** – 시계열 데이터셋
    - **allowed_encoder_known_variable_names** – 인코더에서 허용되는 알려진 변수 목록, 기본값은 모두 허용됩니다.
    - **kwargs** – 모델의 하이퍼파라미터와 같은 추가 인수(__init__() 참조)

    - **Returns:** 
       TemporalFusionTransformer

### get_attention_mask(encoder_lengths: LongTensor, decoder_lengths: LongTensor)
  - **셀프 어텐션 계층에 적용할 인과 마스크를 반환합니다.**

### interpret_output(out: Dict[str, Tensor], reduction: str = 'none', attention_prediction_horizon: int = 0) → Dict[str, Tensor]
   - **모델의 출력을 해석합니다.**

- **Parameters:**
    - **out** – forward()에 의해 생성된 출력
    - **reduction** – 배치에 대한 평균 없음을 위해 "none", 어텐션을 합산하기 위해 "sum", 인코딩 길이로 정규화하기 위해 "mean"
    - **attention_prediction_horizon** – 어텐션에 사용할 예측 지평선

- **Returns:**
  plot_interpretation()으로 그릴 수 있는 해석

### log_embeddings()
  - **텐서보드에 임베딩을 기록합니다.**

### log_interpretation(outputs)
  - **텐서보드에 해석 지표를 기록합니다.**

### on_epoch_end(outputs)
  - **훈련 또는 검증의 에포크 종료시 실행됩니다.**

### on_fit_end()
  - **맞춤(fit)이 완전히 끝난 후 호출됩니다.**
  - **DDP에서는 각 프로세스에서 호출됩니다.**

### plot_interpretation(interpretation: Dict[str, Tensor]) → Dict[str, Figure]
  - **모델을 해석하는 그림을 생성합니다.**

    - Attention
    - 변수 선택 가중치 / 중요도

  - **Parameters:**
    - interpretation – interpret_output()에서 얻은 값

  - **Returns:**
    - matplotlib 그림의 사전

### plot_prediction(x: Dict[str, Tensor], out: Dict[str, Tensor], idx: int, plot_attention: bool = True, add_loss_to_title: bool = False, show_future_observed: bool = True, ax=None, **kwargs) → Figure
실제 값과 예측 그리고 어텐션을 그립니다.

  - **Parameters:**
    - **x** (Dict[str, torch.Tensor]) – 네트워크 입력
    - **out** (Dict[str, torch.Tensor]) – 네트워크 출력
    - **idx** (int) – 샘플 인덱스
    - **plot_attention** – 보조 축에 어텐션을 그릴지 여부
    - **add_loss_to_title** – 제목에 손실을 추가할지 여부. 기본값은 False입니다.
    - **show_future_observed** – 미래 실제 값을 표시할지 여부. 기본값은 True입니다.
    - **ax** – 그림을 그릴 matplotlib 축

  - **Returns:**
    - matplotlib 그림

  - **Return type:**
    - plt.Figure
