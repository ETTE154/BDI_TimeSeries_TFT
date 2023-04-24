# 결과 분석 방법

### Attention 매커니즘
  - **TFT (Temporal Fusion Transformer) 모델의 어텐션 메커니즘에서 사용되는 쿼리(Query), 키(Key), 밸류(Value)는 다음과 같이 연산됩니다.**어텐션 메커니즘은 주어진 입력 시퀀스에 대해 다른 위치의 정보를 가중치를 부여하여 정보를 종합하는 과정입니다. 쿼리(Query), 키(Key), 밸류(Value)는 이 과정에서 핵심적인 역할을 수행합니다.

쿼리(Query), 키(Key), 밸류(Value) 계산:

**1. 먼저, 인코더 및 디코더의 각 위치에서 쿼리(Query), 키(Key), 밸류(Value)를 계산합니다. 이를 위해, 입력 벡터(인코더 및 디코더의 입력)에 대해 선형 변환을 수행합니다.**

Q = W_q * X
K = W_k * X
V = W_v * X

여기서,

**Q, K, V**는 각각 **쿼리(Query), 키(Key), 밸류(Value)** 행렬입니다.
**W_q, W_k, W_v**는 각각 **쿼리(Query), 키(Key), 밸류(Value)**를 계산하기 위한 가중치 행렬입니다.
**X**는 인코더 또는 디코더의 입력 행렬입니다.
어텐션 가중치 계산:

**2. 쿼리(Query)와 키(Key) 행렬의 내적을 통해 각 위치 간의 유사도를 계산합니다. 이후, 소프트맥스(softmax) 함수를 적용하여 어텐션 가중치를 계산합니다.**

**Attention_weights = softmax(Q * K^T / sqrt(d_k))**

여기서,

d_k는 키(Key) 및 쿼리(Query) 벡터의 차원입니다.
"^T"는 전치(transpose)를 의미합니다.
sqrt는 제곱근을 의미합니다.
어텐션 출력 계산:

**3. 어텐션 가중치와 밸류(Value) 행렬의 곱을 통해 어텐션 출력을 계산합니다.**

**Attention_output = Attention_weights * V**

어텐션 출력은 다음 계층으로 전달되거나, 여러 개의 어텐션 헤드가 있는 경우 이를 결합하여 최종 출력을 생성합니다.
이렇게 계산된 어텐션 메커니즘은 TFT 모델의 인코더와 디코더에서 입력 시퀀스의 정보를 종합하는 데 사용되어, 시계열 데이터의 다양한 패턴을 포착하고 예측 성능을 향상시킵니다.

### plot_prediction 함수는 실제 값과 예측 그리고 어텐션을 시각화합니다.

#### 매개변수:
  - **x** (Dict[str, torch.Tensor]): 네트워크 입력
  - **out** (Dict[str, torch.Tensor]): 네트워크 출력
  - **idx** (int): 샘플 인덱스
  - **plot_attention** (bool): 보조 축에 어텐션을 그릴지 여부, 기본값은 True입니다.
  - **add_loss_to_title** (bool): 제목에 손실을 추가할지 여부, 기본값은 False입니다.
  - **show_future_observed** (bool): 미래 실제 값을 표시할지 여부, 기본값은 True입니다.
  - **ax**: 그림을 그릴 matplotlib 축
  - **반환 값**:
    - plt.Figure: matplotlib 그림 함수는 먼저 일반적인 예측을 그립니다. 그런 다음 plot_attention이 True이면, 보조 축에 어텐션을 추가합니다. 
      이를 위해 interpret_output 함수를 사용하여 out의 해석을 얻고, 그 결과를 사용하여 어텐션을 그립니다. 마지막으로 tight_layout() 함수를 호출하여 그림의 레이아웃을 조정합니다.

#### Attention
  - 어텐션(Attention)은 딥러닝 모델, 특히 시퀀스 처리 작업에서 중요한 정보에 초점을 맞추도록 도움을 주는 메커니즘입니다.
    어텐션은 모델이 입력 시퀀스의 각 부분에 가중치를 할당하여 중요한 부분에 높은 가중치를 부여하고 상대적으로 덜 중요한 부분에 낮은 가중치를 부여합니다. 
    이를 통해 모델이 더욱 효과적으로 학습하고 예측을 수행할 수 있습니다.

#### plot_prediction 함수에서 어텐션을 그리는 과정은 다음과 같습니다:

  1. **interpret_output** 함수를 사용하여 네트워크 출력인 out의 해석을 얻습니다. 이 함수는 출력에 대한 어텐션 가중치를 계산합니다.
  2. 어텐션 가중치를 그래프로 그리기 위해, 먼저 기본 축(ax)에 대한 보조 축(ax2)을 생성합니다. 이렇게 하면 어텐션 그래프가 예측 그래프와 겹치지 않고 독립적으로 그려집니다.
  3. 보조 축에 **"Attention"** 레이블을 붙입니다.
  4. 인코더 길이를 사용하여 x축 범위를 설정하고, 어텐션 가중치를 y축 값으로 사용하여 그래프를 그립니다. 그래프의 투명도는 0.2로 설정하고, 선 색상은 검정색으로 지정합니다.
  5. **tight_layout**() 함수를 호출하여 그림의 레이아웃을 조정합니다.

  결과적으로, 이 함수는 예측 그래프와 함께 어텐션 가중치를 시각화하여, 어떤 입력 부분이 모델에 의해 더 중요하게 여겨지는지 이해하는 데 도움이 됩니다.
  
### Variables importance
#### Variable Selection Network 구조

Temporal Fusion Transformer (TFT) 모델의 Variable Selection Network는 입력 변수의 가변적인 중요도를 자동으로 학습하는 구조입니다. 이 네트워크는 각 입력 변수에 대해 가중치를 할당하고, 이 가중치를 이용하여 예측에 얼마나 영향력 있는지를 결정합니다. 이를 통해 모델은 불필요한 변수를 걸러내고, 관련성이 높은 변수에 집중할 수 있습니다.

Variable Selection Network는 크게 두 가지 부분으로 구성됩니다.

  1. **입력 게이팅(Input Gating) 메커니즘**: 각 입력 변수에 대해 게이트 가중치를 계산합니다. 이 가중치는 변수의 중요도를 나타내며, 이 가중치에 따라 변수가 얼마나 예측에 기여하는지 결정됩니다. 입력 게이팅 메커니즘은 각 변수에 대한 스칼라 가중치를 학습하며, 이 가중치를 변수의 값에 곱합니다. 이를 통해 중요한 변수는 강조되고, 덜 중요한 변수는 약화됩니다.

  2. **Temporal Variable Selection (TVS) 레이어**: 입력 시계열의 다양한 변수를 처리하기 위해 설계된 고유한 구조입니다. TVS 레이어는 인코더와 디코더의 모든 스텝에서 각 변수에 대해 별도의 가중치를 학습합니다. 이를 통해 모델은 시간에 따라 입력 변수의 중요도가 어떻게 변하는지를 고려할 수 있습니다.

Variable Selection Network는 이러한 구성요소를 사용하여 각 입력 변수의 중요도를 자동으로 학습합니다. 이 과정을 통해 TFT 모델은 다양한 입력 변수와 시간에 따라 변하는 중요도를 효과적으로 처리하며, 예측 성능을 높이는 데 도움이 됩니다.

#### best_tft.plot_interpretation(interpretation)
best_tft.plot_interpretation(interpretation) 함수는 Temporal Fusion Transformer (TFT) 모델의 해석 결과를 시각화합니다. 이 결과는 주로 다음 네 가지 범주로 나눌 수 있습니다.

  1. **Attention**: 이 그래프는 모델의 어텐션 메커니즘에 대한 정보를 보여줍니다. 어텐션 메커니즘은 인코더와 디코더 사이의 정보 흐름을 조절하며, 시계열 데이터의 특정 시점에 집중하여 예측에 도움이 되는 패턴을 찾습니다. 이 그래프에서는 각 시점의 어텐션 가중치가 어떻게 분포하는지 확인할 수 있습니다.

  2. **Static Variables**: 이 그래프는 static 변수의 중요도를 보여줍니다. Static 변수는 시간에 따라 변하지 않는 변수로, 주로 데이터의 고유한 특성을 나타냅니다. 이 그래프에서는 각 static 변수의 중요도를 비교하여 어떤 변수가 예측에 더 큰 영향을 미치는지 파악할 수 있습니다.

  3. **Encoder Variables**: 이 그래프는 인코더에서 사용되는 시계열 변수의 중요도를 보여줍니다. 인코더 변수는 시계열 데이터의 과거 정보를 나타내며, 인코더에서 처리되어 디코더로 전달됩니다. 이 그래프를 통해 각 인코더 변수가 모델의 예측에 얼마나 기여하는지를 평가할 수 있습니다.

  4. **Decoder Variables**: 이 그래프는 디코더에서 사용되는 시계열 변수의 중요도를 보여줍니다. 디코더 변수는 미래의 정보를 나타내며, 디코더에서 처리되어 최종 예측을 생성하는 데 사용됩니다. 이 그래프를 통해 각 디코더 변수가 모델의 예측에 얼마나 기여하는지를 평가할 수 있습니다.