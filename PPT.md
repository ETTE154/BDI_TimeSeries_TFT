# PTT 작성 내용 요약

## TFT 모델 소개

### TFT 모델
  - TFT는 순환 신경망(RNN), 합성곱 신경망(CNN) 및 어텐션 메커니즘과 같은 **다양한 신경망 구조의 장점을 결합**하여 복잡한 시계열 데이터를 모델링합니다.

### TFT 모델의 구성요소

  1. **입력 인코딩**: TFT는 게이트가 있는 선형 유닛(GLU)과 dilated convolution의 조합을 사용하여 각 입력 특징을 개별적으로 처리합니다. 이를 통해 모델은 **복잡한 시간 패턴을 학습하고 다양한 특징 유형에 적응**할 수 있습니다.

  2. **시간적 융합**: 모델은 각각 장기 및 단기 시간 종속성을 모델링하기 위해 셀프 어텐션 및 로컬 어텐션 메커니즘의 조합을 사용합니다. 이를 통해 모델은 **전역 및 로컬 컨텍스트를 효과적으로 포착**할 수 있습니다.

  3. **출력 디코딩**: TFT는 인코딩된 입력 특징의 다양한 부분에 주목하여 다중 수평 예측을 생성하는 멀티 헤드 어텐션 메커니즘을 사용합니다. 이로 인해 더 **정확하고 해석 가능한 예측이 가능**해집니다.

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
| Special Events | Subprime Mortgage, Covid-19, etc. | - | Occasional |
| Seasonality | Seasonal factors affecting shipping demand | - | Annual |
