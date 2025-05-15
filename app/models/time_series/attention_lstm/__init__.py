"""
Attention LSTM 모델 패키지 (Attention LSTM Model Package)

이 패키지는 Attention 메커니즘을 활용한 LSTM 기반 커피 가격 예측 모델을 제공합니다.

주요 모듈:
1. data_preprocessing.py: 데이터 로딩 및 전처리 기능
2. dataset.py: 시계열 데이터셋 클래스
3. model.py: 어텐션 LSTM 모델 아키텍처
4. training.py: 모델 학습 및 예측 기능
5. utils.py: 유틸리티 함수 (저장, 로딩 등)
6. main.py: 메인 실행 코드

간단한 사용 예시:
    from app.models.time_series.attention_lstm import main
    
    forecast_series, model, scaler = main.main(
        macro_data_path='path/to/macro_data.csv',
        climate_data_path='path/to/climate_data.csv',
        num_epochs=200
    )
"""

from .data_preprocessing import load_and_prepare_data, train_test_split, scale_data
from .dataset import MultiStepTimeSeriesDataset
from .model import AttentionLSTMModel, EntmaxAttention
from .training import train_model, predict_future_prices
from .utils import save_prediction_to_csv, save_model, load_model
from .main import main

__all__ = [
    # data_preprocessing.py
    'load_and_prepare_data', 'train_test_split', 'scale_data',
    
    # dataset.py
    'MultiStepTimeSeriesDataset',
    
    # model.py
    'AttentionLSTMModel', 'EntmaxAttention',
    
    # training.py
    'train_model', 'predict_future_prices',
    
    # utils.py
    'save_prediction_to_csv', 'save_model', 'load_model',
    
    # main.py
    'main'
]
