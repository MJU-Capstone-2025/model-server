"""
Attention LSTM 모델 메인 실행 모듈 (Main Execution Module)

이 모듈은 Attention LSTM 모델 파이프라인을 실행하는 메인 코드다.
데이터 로딩, 전처리, 모델 학습, 예측, 및 결과 저장을 포함하는 전체 프로세스를 조정한다.

주요 기능:
1. 명령행 인수 처리
2. 데이터 로드 및 전처리
3. 모델 학습 및 예측 파이프라인 실행
4. 결과 저장 및 시각화

실행 방법:
    python app/models/time_series/attention_lstm/main.py --epochs 200 --batch_size 10
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime

# 패키지 경로 설정을 위한 코드
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 내부 모듈 import (상대 경로로 수정)
from .data_preprocessing import load_and_prepare_data, train_test_split, scale_data
from .dataset import MultiStepTimeSeriesDataset
from .model import AttentionLSTMModel
from .training import train_model, predict_future_prices
from .utils import save_prediction_to_csv, save_model


def main(macro_data_path, climate_data_path, output_path='./data/output/', 
            data_window=100, future_target=14, step=6, batch_size=10, 
            hidden_size=100, num_epochs=200):
    """
    모델 학습 및 예측을 실행하는 메인 함수다.
    
    Args:
        macro_data_path (str): 거시경제 데이터 파일 경로
        climate_data_path (str): 기후 데이터 파일 경로
        output_path (str): 출력 파일 저장 경로
        data_window (int): 입력 윈도우 크기
        future_target (int): 예측할 미래 일수
        step (int): 샘플링 간격
        batch_size (int): 미니배치 크기
        hidden_size (int): LSTM 은닉층 크기
        num_epochs (int): 학습 에폭 수
        
    Returns:
        tuple: (forecast_series, model, scaler) 형태의 튜플
    """
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 데이터 로드 및 전처리
    df = load_and_prepare_data(macro_data_path, climate_data_path)
    print(f"Data loaded with shape: {df.shape}")
    
    # 학습/테스트 분할
    train_df, test_df = train_test_split(df)
    print(f"Training data: {train_df.shape}, Test data: {test_df.shape}")
    
    # 데이터 스케일링
    scaled_train_df, scaler = scale_data(train_df)
    
    # 학습 데이터 준비
    X_train = scaled_train_df.values
    target_col = "Coffee_Price"
    y_train = scaled_train_df[target_col].values
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, future_target, step)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    input_size = X_train.shape[1]
    model = AttentionLSTMModel(
        input_size=input_size, 
        hidden_size=hidden_size, 
        target_size=future_target
    ).to(device)
    
    # 모델 학습
    print("Starting model training...")
    model = train_model(train_loader, model, device, num_epochs=num_epochs)
    
    # 미래 가격 예측
    print("Predicting future prices...")
    forecast_series, metrics = predict_future_prices(
        model, train_dataset.data, train_df, df, scaler, 
        target_col=target_col, days=future_target, device=device,
        output_path=output_path
    )
    
    # 예측 결과 저장
    prediction_path = save_prediction_to_csv(forecast_series, output_path=output_path)
    
    # 모델 저장
    hyperparameters = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': model.num_layers,
        'target_size': model.target_size
    }
    model_path = save_model(model, scaler, hyperparameters, metrics, output_path)
    
    return forecast_series, model, scaler


if __name__ == "__main__":
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
