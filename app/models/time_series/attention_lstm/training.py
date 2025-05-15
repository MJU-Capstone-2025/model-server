"""
모델 학습 및 예측 모듈 (Model Training and Prediction Module)

이 모듈은 Attention LSTM 모델의 학습 및 예측 기능을 제공한다.
학습 루프, 손실 계산, 가중치 업데이트, 그리고 학습된 모델을 이용한 미래 가격 예측 기능을 포함한다.

주요 기능:
1. 모델 학습 (train_model)
2. 미래 가격 예측 (predict_future_prices)
3. 성능 지표 계산 (MAE, RMSE)

사용 예시:
    from training import train_model, predict_future_prices
    
    # 모델 학습
    model = train_model(train_loader, model, device, num_epochs=200)
    
    # 미래 가격 예측
    forecast_series, metrics = predict_future_prices(
        model, X_train, train_df, df, scaler, 
        target_col='Coffee_Price', days=14
    )
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta


def train_model(train_loader, model, device, num_epochs=200):
    """
    모델을 학습한다.
    
    Args:
        train_loader (DataLoader): 학습 데이터 로더
        model (nn.Module): 학습할 모델
        device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        num_epochs (int): 학습 에폭 수
        
    Returns:
        nn.Module: 학습된 모델
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # 모델 예측 (hidden state 제거)
            y_pred, _ = model(x_batch)
            
            # (선택적으로) 차원 조정
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            
            # Loss 계산
            loss = criterion(y_pred, y_batch)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    
    return model


def predict_future_prices(model, X_train, train_df, df, scaler, target_col='Coffee_Price', days=14, device='cpu', save_plot=True, output_path='./data/output/'):
    """
    미래 가격을 예측한다.
    
    Args:
        model (nn.Module): 학습된 모델
        X_train (torch.Tensor): 학습 데이터 입력 텐서
        train_df (pd.DataFrame): 학습 데이터 프레임
        df (pd.DataFrame): 전체 데이터 프레임
        scaler (MinMaxScaler): 데이터 스케일러
        target_col (str): 예측할 타겟 컬럼명
        days (int): 예측할 미래 일수
        device (str): 예측에 사용할 디바이스 ('cuda' 또는 'cpu')
        save_plot (bool): 시각화 결과 저장 여부
        output_path (str): 출력 파일 저장 경로
        
    Returns:
        tuple: (forecast_series, metrics) 형태의 튜플
            - forecast_series: 예측 결과 (pd.Series)
            - metrics: 성능 지표 (dict)
    """
    # 1. train_df 마지막 날짜의 위치를 df 전체에서 찾음
    last_train_idx = df.index.get_loc(train_df.index[-1])
    
    # 2. 그 다음 days일치 날짜를 df에서 추출
    prediction_dates = df.index[last_train_idx + 1 : last_train_idx + 1 + days]
    
    # 3. 실제값 가져오기 (테스트용)
    try:
        true_values = df.loc[prediction_dates, target_col].values
    except:
        true_values = None
        print("Warning: True values not available for prediction dates")
    
    # 마지막 시퀀스 예측
    last_seq = X_train[-1].unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        prediction, attn_weights = model(last_seq)
    
    # CPU로 이동 후 numpy 변환
    prediction = prediction.squeeze().cpu().numpy().reshape(-1, 1)
    
    # 역변환을 위해 dummy 피처 생성
    dummy = np.zeros((days, train_df.shape[1] - 1))  # target_col 제외 나머지
    prediction_combined = np.concatenate([prediction, dummy], axis=1)
    
    # target_col이 첫 번째 컬럼이라면 그대로 [:, 0]
    # (다른 위치일 경우 target_index로 변경 가능)
    prediction = scaler.inverse_transform(prediction_combined)[:, 0]
    
    # 예측값을 Series로 변환
    forecast_series = pd.Series(prediction, index=prediction_dates)
    
    if save_plot:
        # 결과 시각화
        plt.figure(figsize=(14, 6))
        plt.plot(df[target_col], label='Actual Coffee Price', color='blue')
        plt.plot(prediction_dates, forecast_series, label='Predicted Coffee Price', color='red', linestyle='dashed')
        
        plt.title('Coffee Price Prediction')
        plt.xlabel("Date")
        plt.ylabel("Coffee Price")
        
        # 최근 100일 + 예측 기간만 표시
        last_date = prediction_dates[-1]
        first_date = last_date - timedelta(days=100)
        plt.xlim(first_date, last_date + timedelta(days=7))
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 현재 날짜로 파일명 생성
        today = datetime.now().strftime('%Y%m%d')
        plt.savefig(f'{output_path}attention_prediction_{today}.png', dpi=300, bbox_inches='tight')
    
    # 성능 평가 (실제값이 있는 경우만)
    metrics = {}
    if true_values is not None:
        comparison_df = pd.DataFrame({
            "날짜": forecast_series.index,
            "실제값": true_values,
            "예측값": forecast_series.values
        })
        
        # 평가 지표 계산
        y_true = comparison_df["실제값"][:days]
        y_pred = comparison_df["예측값"][:days]
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        metrics = {
            "MAE": mae,
            "RMSE": rmse
        }
    
    return forecast_series, metrics
