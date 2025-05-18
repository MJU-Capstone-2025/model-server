"""
유틸리티 모듈 (Utilities Module)

이 모듈은 모델 예측 결과를 저장하고 시각화하는 유틸리티 기능을 제공한다.

주요 기능:
1. 예측 결과를 CSV 파일로 저장
2. 학습된 모델 저장
3. 저장된 모델 로드
4. 시각화 헬퍼 함수

사용 예시:
    from utils import save_prediction_to_csv, save_model
    
    # 예측 결과 저장
    prediction_path = save_prediction_to_csv(forecast_series)
    
    # 모델 저장
    model_path = save_model(model, scaler, hyperparameters, metrics)
"""

import pandas as pd
import torch
from datetime import datetime


def save_prediction_to_csv(forecast_series, filename=None, output_path='./data/output/'):
    """
    예측 결과를 CSV 파일로 저장한다.
    
    Args:
        forecast_series (pd.Series): 예측 결과 시리즈
        filename (str, optional): 저장할 파일명
        output_path (str): 출력 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    if filename is None:
        today = datetime.now().strftime('%Y%m%d')
        filename = f'{output_path}coffee_prediction_{today}.csv'
    
    # 예측 결과 DataFrame 생성 및 저장
    prediction_df = pd.DataFrame({
        'Date': forecast_series.index,
        'Prediction_Price': forecast_series.values
    })
    
    prediction_df.to_csv(filename, index=False)
    print(f"Prediction saved to: {filename}")
    
    return filename


def save_model(model, scaler, hyperparameters, metrics, output_path='./data/output/'):
    """
    학습된 모델과 관련 정보를 저장한다.
    
    Args:
        model (nn.Module): 학습된 모델
        scaler (MinMaxScaler): 데이터 스케일러
        hyperparameters (dict): 모델 하이퍼파라미터
        metrics (dict): 성능 지표
        output_path (str): 출력 경로
        
    Returns:
        str: 저장된 모델 파일 경로
    """
    today = datetime.now().strftime('%Y%m%d')
    model_path = f"{output_path}attention_model_v2_{today}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'hyperparameters': hyperparameters,
        'metrics': metrics
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(model_path, device='cpu'):
    """
    저장된 모델을 로드한다.
    
    Args:
        model_path (str): 모델 파일 경로
        device (str): 모델을 로드할 디바이스
        
    Returns:
        tuple: (model, scaler, hyperparameters, metrics) 형태의 튜플
    """
    from .model import AttentionLSTMModel
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 하이퍼파라미터 추출
    hyperparameters = checkpoint['hyperparameters']
    
    # 모델 초기화 및 가중치 로드
    model = AttentionLSTMModel(
        input_size=hyperparameters['input_size'],
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters.get('num_layers', 2),
        target_size=hyperparameters['target_size']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 스케일러 및 성능 지표 추출
    scaler = checkpoint['scaler']
    metrics = checkpoint.get('metrics', {})
    
    return model, scaler, hyperparameters, metrics