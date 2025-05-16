"""
이 파일은 이제 대체되었다.
새로운 모듈화된 코드는 attention_lstm/ 디렉토리에 있다.
하위 호환성을 위해 이 파일은 유지되며 새 모듈을 임포트한다.

사용 예시:
    from app.models.time_series.attention_lstm_model import main
    
    forecast_series, model, scaler = main(
        macro_data_path='path/to/macro_data.csv',
        climate_data_path='path/to/climate_data.csv',
        num_epochs=200
    )
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 현재 파일의 절대 경로를 가져온다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 새 패키지 경로를 sys.path에 추가한다.
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 새 모듈을 임포트한다.
from attention_lstm.data_preprocessing import load_and_prepare_data, train_test_split, scale_data, add_volatility_features
from attention_lstm.dataset import MultiStepTimeSeriesDataset
from attention_lstm.model import AttentionLSTMModel, EntmaxAttention
from attention_lstm.training import train_model, predict_future_prices
from attention_lstm.utils import save_prediction_to_csv, save_model, load_model
from attention_lstm.main import main

# 하위 호환성을 위해 모든 함수와 클래스를 현재 네임스페이스로 가져온다.
# 이렇게 하면 이전 코드가 그대로 작동한다.
__all__ = [
    'load_and_prepare_data', 'train_test_split', 'scale_data', 'add_volatility_features',
    'MultiStepTimeSeriesDataset', 'AttentionLSTMModel', 'EntmaxAttention',
    'train_model', 'predict_future_prices', 'save_prediction_to_csv', 'save_model', 'main'
]

# 기존 코드와의 호환성을 위해 함수 및 클래스 전달
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and predict coffee prices using Attention LSTM")
    parser.add_argument('--macro_data', type=str, default='./data/input/거시경제및커피가격통합데이터.csv',
                        help='Path to the macroeconomic data file')
    parser.add_argument('--climate_data', type=str, default='./data/input/기후데이터피쳐선택.csv',
                        help='Path to the climate data file')
    parser.add_argument('--output_path', type=str, default='./data/output/',
                        help='Path to save model and prediction outputs')
    parser.add_argument('--epochs', type=int, default=10, # epochs=200, test=10
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Hidden size of the LSTM model')
    parser.add_argument('--data_window', type=int, default=100,
                        help='Size of the input data window')
    parser.add_argument('--future_target', type=int, default=14,
                        help='Number of days to predict')
    parser.add_argument('--step', type=int, default=6,
                        help='Sampling step within the data window')
    
    args = parser.parse_args()
    
    forecast_series, model, scaler = main(
        args.macro_data,
        args.climate_data,
        args.output_path,
        args.data_window,
        args.future_target,
        args.step,
        args.batch_size,
        args.hidden_size,
        args.epochs
    )