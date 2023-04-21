#%%
import os
import warnings
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.data.encoders import EncoderNormalizer

torch.set_float32_matmul_precision('high')

data = pd.read_csv('04_19.csv')

# add time index
# date 의 형식을 datetime으로 바꿔준다.
data["date"] = pd.to_datetime(data["date"])
# data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
# data["time_idx"] -= data["time_idx"].min()

# add additional features
data["year"] = data.date.dt.year.astype(str).astype("category")  # categories have be strings
# data["log_volume"] = np.log(data.BDI + 1e-8)
# season 칼럼을 카테고리화 시킨다.
# data["season"] = data["season"].astype("category")
#%%
data.columns
#%%
data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = ['dot_com',
                '911',
                'China_WTO',
                'Iraq_war',
                'Indian_Ocean_Tsunami',
                'Grobal_Financial_Crisis',
                'China_Economic_policy_changes',
                'Lehman_bro',
                'Europe_Debt_Crisis',
                'Japan_Earthqauke',
                'Crude_oil_price_Collapse',
                'China_economic_Slowdown',
                'Covid_19',
                ]
season = ['spring', 'summer', 'fall', 'winter']
#%%
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
data[season] = data[season].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
max_prediction_length = 24
max_encoder_length = 52
training_cutoff = data["TimeSeries"].max() - max_prediction_length
data["constant_group_id"] = "constant_value"
# constant_group_id 카테고리화 시킨다.
data["constant_group_id"] = data["constant_group_id"].astype("category")
data.sample(30)
#%%
# ========================== EncoderNormalizer ========================== #

encoder_normalizer = EncoderNormalizer(
    method="robust",
    method_kwargs={
        "center": 0.5,
        "lower": 0.25,
        "upper": 0.75,
    }
)
# encoder_normalizer = EncoderNormalizer(
#     method="identity"
# )

# encoder_normalizer = EncoderNormalizer(
#     method="standard"
# )

# ========================== EncoderNormalizer ========================== #

training = TimeSeriesDataSet(
    data[lambda x: x.TimeSeries <= training_cutoff],
    time_idx="TimeSeries",
    target="BDI",
    #group_ids: 데이터를 그룹화하는 데 사용되는 ID
    group_ids = ['constant_group_id'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=12,
    max_prediction_length=max_prediction_length,
    
    #static_categoricals: 시간에 따라 변하지 않는 범주형 데이터
    static_categoricals=[],
    # static_reals: 시간에 따라 변하지 않는 실수형 데이터
    static_reals=['NOSB'],
    # variable_groups: 변수 그룹을 나타내는 데이터
    variable_groups={"special_days": special_days},
    # time_varying_known_categoricals: 시간에 따라 변하고 이미 알려진 범주형 데이터
    time_varying_known_categoricals=['month'],
    # time_varying_known_reals: 시간에 따라 변하고 이미 알려진 실수형 데이터
    time_varying_known_reals=['OECD_GDP_growth'],
    # time_varying_unknown_categoricals: 시간에 따라 변하지만 알려지지 않은 범주형 데이터
    time_varying_unknown_categoricals=['special_days'],
    # time_varying_unknown_reals: 시간에 따라 변하지만 알려지지 않은 실수형 데이터
    time_varying_unknown_reals=['SSEC', 'NASDAQ', 'CRB', 'BDI', 'FED'],
    target_normalizer=encoder_normalizer,
    # 불규칙한 데이터셋 허용
    # allow_missing_timesteps=True,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# TimeSeriesDataSet.from_dataset() 함수는 기존의 학습 데이터셋(training)을 기반으로 새로운 검증 데이터셋(validation)을 생성합니다.
# predict=True를 설정하여 각 시계열 데이터의 마지막 max_prediction_length 시점을 예측하도록 합니다.
# stop_randomization=True를 설정하여 검증 데이터셋을 생성하는 동안 시계열의 무작위 샘플링을 중지합니다. 이렇게 하면 검증 과정에서 일관성 있는 결과를 얻을 수 있습니다.
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
#%%
# 데이터 로더는 모델에 데이터를 공급하는 역할을 합니다.
# 데이터 로더를 사용하면 배치(batch) 단위로 데이터를 처리할 수 있어, 효율적인 학습과 검증이 가능합니다.

# batch_size는 한 번에 처리할 데이터의 개수를 의미하며, 이 예에서는 64로 설정되었습니다. 일반적으로 32 ~ 128 사이의 값을 사용합니다.
batch_size = 32
# train_dataloader는 학습 데이터셋(training)을 기반으로 생성되며, 
# train=True를 설정하여 학습 모드로 동작하게 합니다.
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# val_dataloader는 검증 데이터셋(validation)을 기반으로 생성되며, 
# train=False를 설정하여 검증 모드로 동작하게 합니다.
# 검증 데이터 로더의 배치 크기는 일반적으로 학습 데이터 로더의 배치 크기보다 크게 설정할 수 있습니다. 여기서는 학습 배치 크기의 10배인 640을 사용합니다.
# num_workers=0는 데이터 로딩을 처리할 병렬 작업자 수를 설정하는데, 이 경우에는 병렬 처리를 사용하지 않습니다.
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
# %%
# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()

# %%
# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator='gpu',
    devices= [0],
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)
#%%
tft = TemporalFusionTransformer.from_dataset(
    training,
    # 사용할 최적화 알고리즘
    optimizer=torch.optim.Adam,
    #  최적화 알고리즘의 학습률 (0.001~0.1)
    learning_rate=0.03,
    #  모델의 숨겨진 레이어 크기를 설정합니다. 이 값은 모델의 복잡성과 성능에 영향을 미칩니다.
    hidden_size=32,
    #  Attention 메커니즘에서 사용되는 헤드 수를 설정합니다. 큰 데이터셋의 경우 최대 4까지 설정할 수 있습니다.
    attention_head_size=3,
    # 드롭아웃 비율을 설정(0.1~0.3)
    dropout=0.1,
    #  연속 변수를 처리하는 데 사용되는 숨겨진 레이어의 크기(hidden_size 이하로 설정)
    hidden_continuous_size=16,
    #  출력 벡터의 크기를 설정합니다. 기본적으로는 7개의 분위수를 출력하도록 설정
    output_size=7,
    # 손실 함수
    loss=QuantileLoss(),
    # 검증 손실이 지정된 에폭 수 동안 개선되지 않을 경우 학습률을 줄이는 옵션
    reduce_on_plateau_patience=4,
    # embedding_sizes=embedding_sizes
)
#%%
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
# %%

# find optimal learning rate
res = trainer.tuner.lr_find(
    tft,
    # 학습 데이터를 불러오는 DataLoader
    train_dataloaders=train_dataloader,
    # 검증 데이터를 불러오는 DataLoader
    val_dataloaders=val_dataloader,
    # 학습률을 탐색할 최대 범위
    max_lr=9.0,
    # 학습률을 탐색할 최소 범위
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# %%
# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
#%%
trainer = pl.Trainer(
    # 모델이 학습을 수행할 최대 에포크 수
    max_epochs=100,
    # 학습에 사용할 가속기 gpu 또는 cpu
    accelerator= 'gpu',
    # 학습에 사용할 GPU의 수
    devices= [0],
    # 모델 요약 정보를 출력할지 여부
    enable_model_summary=True,
    # 그래디언트 클리핑 값을 설정(그래디언트 폭주를 방지하기 위해 그래디언트의 최대 크기를 0.1로 제한)
    gradient_clip_val=0.1,
    # 한 에포크에서 사용할 최대 학습 배치의 수를 설정
    limit_train_batches=128,  
    # #  네트워크나 데이터셋에 심각한 버그가 없는지 확인하기 위해 사용하는 빠른 개발 모드
    # fast_dev_run=True,  
    # 학습 중 사용할 콜백 목록
    callbacks=[lr_logger, early_stop_callback],
    # 학습 결과를 기록할 로거입니다. 이 예에서는 TensorBoardLogger를 사용하여 학습 결과를 TensorBoard에 기록합니다.
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # 학습률 설정
    learning_rate=0.03,
    # 은닉층 크기 설정
    hidden_size=64,
    # 어텐션 헤드 수 설정 (4 이하로 설정)
    attention_head_size=1,
    # 드롭아웃 비율 설정
    dropout=0.1,
    # 연속형 입력 변수에 대한 은닉층 크기 설정(hidden_size 이하로 설정)
    hidden_continuous_size=32,
    # 출력 크기 설정 (기본적으로 7개의 분위수 사용)
    output_size=7,
    # 손실 함수 설정
    loss=QuantileLoss(),
    # 로깅 간격 설정 (예: 10개 배치마다 로깅 수행)
    log_interval=10,
    # 검증 손실이 개선되지 않을 경우 학습률을 줄이기 위한 기다림 횟수 설정
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    # model_path: 최적화된 모델을 저장할 경로입니다.
    model_path="optuna_test",
    # n_trials: 최적화를 위해 시도할 횟수입니다.
    n_trials= 30,
    # max_epochs: 각 trial 동안 최대로 학습시킬 epoch의 수입니다.
    max_epochs=300,
    # gradient_clip_val_range: 기울기 클리핑 값의 범위입니다.
    # 기울기가 지정된 범위를 벗어나지 않도록 조정합니다.
    gradient_clip_val_range=(0.01, 1.0),
    # hidden_size_range: hidden layer의 크기 범위입니다.
    hidden_size_range=(8, 128),
    # hidden_continuous_size_range: 연속 특징을 처리하는 hidden layer의 크기 범위입니다.
    hidden_continuous_size_range=(8, 128),
    # attention_head_size_range: attention head의 개수 범위입니다.
    attention_head_size_range=(1, 4),
    # learning_rate_range: 학습률 범위입니다.
    learning_rate_range=(0.001, 0.1),
    # dropout_range: dropout 비율의 범위입니다.
    dropout_range=(0.1, 0.3),
    # trainer_kwargs: PyTorch Lightning의 Trainer에 전달되는 인자입니다.
    # 여기서는 limit_train_batches를 30으로 설정하여,
    # 각 epoch에서 30개의 배치만 사용하여 학습하도록 설정합니다.
    trainer_kwargs=dict(limit_train_batches=64),
    # reduce_on_plateau_patience: 학습률을 감소시키기 전의 지연 횟수입니다.
    reduce_on_plateau_patience=4,
    # use_learning_rate_finder: 이 값이 True일 경우, 내장된 학습률 찾기를 사용하고,
    # False일 경우 Optuna를 사용하여 최적의 학습률을 찾습니다
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)
#%%
# "test_study.pkl"이라는 이름의 바이너리 쓰기 모드("wb")로 파일을 엽니다.
# 파일 객체를 fout 변수에 할당합니다.
with open("test_study.pkl", "wb") as fout:
    # pickle 모듈의 dump() 함수를 사용하여 study 객체를 fout 파일 객체에 저장합니다.
    # 이렇게 저장된 파일은 추후 하이퍼파라미터 최적화를 계속할 때 또는 결과를 분석할 때 사용할 수 있습니다.
    pickle.dump(study, fout)

# study 객체의 best_trial 속성에서 최적의 하이퍼파라미터를 가져와 출력합니다.
print(study.best_trial.params)
# %%
# 성능이 가장 좋은 모델을 불러옵니다.
best_model_path = trainer.checkpoint_callback.best_model_path
# 가장 성능이 좋은 모델을 TemporalFusionTransformer 객체로 불러옵니다.
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# %%
# 검증 데이터셋에 대해 실제 값과 예측 값을 계산합니다.
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
# 계산된 실제 값과 예측 값의 차이를 절대값으로 취한 후, 이를 평균내어 평균 절대 오차를 구합니다.
(actuals - predictions).abs().mean()
 # %%
# 모델의 원시 예측 값을 가져옵니다.
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
#%%
raw_predictions
#%%
# 예측 결과를 시각화합니다. 여기서는 처음 1개의 예측 결과를 시각화합니다.
for idx in range(10):
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
#%%
# 검증 데이터셋에 대한 예측을 수행합니다.
predictions = best_tft.predict(val_dataloader)
# Symmetric Mean Absolute Percentage Error (SMAPE)를 계산하여 예측 오차를 구합니다.
# reduction="none"를 사용하면, 각 데이터 포인트에 대한 SMAPE 값을 반환합니다.
#%%
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
# 계산된 평균 오차를 기준으로 예측 결과를 정렬합니다.
# argsort(descending=True)를 사용하여 오차가 큰 순서대로 정렬한 인덱스를 반환합니다.
mean_losses
#%%
indices = mean_losses.argsort(descending=True)
# 오차가 가장 큰 예측 결과를 시각화합니다.
# 여기서는 가장 큰 오차를 가진 하나의 예측 결과만 시각화합니다.
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    );
# %%
from matplotlib import pyplot as plt
# 예측과 실제 값 간의 비교를 변수별로 수행하기 위해 필요한 값을 계산합니다.
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
# 변수별로 예측과 실제 값을 비교하는 시각화를 생성합니다.
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);
#%%
time_idx_first_prediction
#%%
# mode="quantiles": 이 인자는 예측 모드를 설정합니다. 이 경우에는 "quantiles"을 사용하여,
# 예측의 불확실성을 고려한 분위수를 반환하도록 합니다. 이를 통해 예측의 신뢰 구간을 추정할 수 있습니다.
best_tft.predict(
    training.filter(lambda x: (x.time_idx_first_prediction == 53)),
    mode="quantiles",
)
# %%
raw_prediction, x = best_tft.predict(
    # training.filter(lambda x: (x.time_idx_first_prediction == 15)): 이 부분은 training 데이터셋을 필터링하여
    # 첫 번째 예측 값이 시간 인덱스 15인 데이터만 선택합니다.
    training.filter(lambda x: (x.time_idx_first_prediction == 53)),
    # mode="raw": 이 인자는 예측 모드를 설정합니다. "raw"를 사용하면, 분위수를 고려하지 않고 원시 예측 값을 반환
    mode="raw",
    # return_x=True: 이 인자는 예측에 사용된 입력 데이터 x도 반환하도록 합니다. 이를 통해 나중에 예측 결과와 함께 시각화할 수 있습니다.
    return_x=True,
)
# 이 부분은 best_tft 모델의 plot_prediction() 메서드를 사용하여 입력 데이터 x와 원시 예측 값 raw_prediction을 시각화합니다
# idx=0 인자는 첫 번째 하위 시퀀스에 대한 예측 결과를 표시하도록 합니다.
best_tft.plot_prediction(x, raw_prediction, idx=0);
# %%
#  데이터셋에서 최근 24개월의 데이터를 인코더 데이터로 선택합니다 (max_encoder_length는 24로 설정되어 있음).
#  time_idx 값이 최대 TimeSeries24를 뺀 값보다 큰 데이터만 선택합니다.
encoder_data = data[lambda x: x.TimeSeries > x.TimeSeries.max() - max_encoder_length]
# 가장 최근 데이터 포인트를 선택합니다.
last_data = data[lambda x: x.TimeSeries == x.TimeSeries.max()]
# 가장 최근 데이터 포인트를 기반으로 디코더 데이터를 생성합니다.
# 이 과정에서, 가장 최근 데이터 포인트를 반복하고 월을 증가시켜 예측 범위 내의 모든 데이터 포인트를 생성합니다. 
# 이 예에서는 각각의 예측 월에 대해 last_data를 복사한 후, 월을 증가시키는 방식으로 디코더 데이터를 생성합니다.
# 실제 데이터셋에서는 공휴일이나 가격 등의 공변량을 고려하여 디코더 데이터를 생성해야 합니다. 하지만 이 예제에서는 간단한 시연을 위해 이러한 공변량을 고려하지 않고 진행합니다.
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)
#%%
# time_idx를 계산하여 디코더 데이터에 추가합니다. 
# 이렇게 하면 인코더 데이터의 time_idx와 일관성을 유지할 수 있습니다. 
# 디코더 데이터의 최소 time_idx 값과 인코더 데이터의 최대 time_idx 값 사이의 차이를 보정해줍니다.
decoder_data["TimeSeries"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["TimeSeries"] += encoder_data["TimeSeries"].max() + 1 - decoder_data["TimeSeries"].min()

# 추가 시간 특성을 조정합니다. 여기서는 "month" 특성을 문자열 타입의 범주형 변수로 변환합니다.
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# 인코더 데이터와 디코더 데이터를 결합하여 새로운 예측 데이터를 생성합니다.
# 이 데이터를 사용하여 시계열 예측 모델에 입력하여 미래 시점의 값을 예측할 수 있습니다.
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
# %%
# best_tft.predict() 함수를 사용하여 새로운 예측 데이터에 대한 예측을 수행합니다. mode="raw"로 설정하여 예측 결과의 원시 값을 반환하고, return_x=True로 설정하여 예측에 사용된 입력 데이터도 반환합니다.
new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

# 예측 결과를 시각화합니다. 이 예제에서는 첫 번째 데이터 포인트에 대한 예측 결과만 시각화합니다. show_future_observed=False로 설정하여 미래 시점의 실제 관측값을 표시하지 않습니다.
# 이는 미래 시점의 실제 값이 아직 알려지지 않았기 때문입니다.
for idx in range(1):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False);
# %%
# best_tft.interpret_output() 함수를 사용하여 예측 결과를 해석합니다.
# 이 함수는 모델의 출력을 해석하여 각 변수의 중요도를 계산합니다
# reduction: 배치(batch)에 대한 평균화 방법을 결정합니다.
# "none"은 배치에 대해 평균화를 수행하지 않음을 의미하며,
# "sum"은 attention 값의 합계를 구하고, "mean"은 인코딩 길이로 정규화합니다.
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
#%%
# best_tft.plot_interpretation() 함수를 사용하여 해석 결과를 시각화합니다. 
# 이 함수는 각 변수의 중요도를 막대 그래프로 표시하여, 
# 어떤 변수가 예측에 큰 영향을 미치는지 확인할 수 있습니다.
best_tft.plot_interpretation(interpretation)
# %%
# best_tft.predict_dependency() 함수를 사용하여 "discount_in_percent" 변수의 값에 따른 예측 결과의 종속성을 계산합니다.
# 이 함수는 변수의 값 범위를 입력으로 받아 (여기서는 0부터 30까지 30개의 값을 사용) 해당 범위에서의 예측 결과를 반환합니다. 
# show_progress_bar=True로 설정하여 진행 상황을 표시하고, mode="dataframe"로 설정하여 결과를 데이터프레임 형태로 반환합니다.
dependency = best_tft.predict_dependency(
    val_dataloader.dataset, "OECD_GDP_growth", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
)
# %%
# 반환된 데이터프레임을 그룹화하여 "discount_in_percent" 값별로
# 정규화된 예측값의 중앙값, 하위 25% 및 상위 75% 백분위수를 계산합니다.
agg_dependency = dependency.groupby("OECD_GDP_growth").normalized_prediction.agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
)
# 계산된 중앙값, 하위 25%, 상위 75% 백분위수를 그래프로 시각화합니다. 
# 이를 통해 "discount_in_percent" 값에 따른 예측 결과의 종속성을 확인할 수 있습니다.

ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3);
# %%
