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


class HuberLoss(nn.Module):
    """
    Huber Loss 함수 (MSE와 MAE의 장점을 결합한 강건한 손실 함수)
    
    Args:
        delta (float): Huber Loss의 delta 파라미터. 이 값보다 작은 오차는 MSE처럼,
                        큰 오차는 MAE처럼 계산됨 (기본값 1.0)
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        # 오차 계산
        error = torch.abs(y_pred - y_true)
        # delta보다 작은 오차는 MSE, 큰 오차는 MAE 방식으로 계산
        is_small_error = (error < self.delta).float()
        small_error_loss = 0.5 * error ** 2
        large_error_loss = self.delta * (error - 0.5 * self.delta)
        # 최종 손실 반환
        return torch.mean(is_small_error * small_error_loss + (1 - is_small_error) * large_error_loss)


class DirectionalLoss(nn.Module):
    """
    방향성 손실 함수 - 가격 변동 방향을 정확하게 예측하는 데 중점을 둔 손실 함수
    
    Args:
        alpha (float): 방향성과 크기 간의 가중치 (1에 가까울수록 방향성 중시)
        beta (float): MSE와 방향성 손실 간의 균형을 조절 (기본값 0.5)
    """
    def __init__(self, alpha=0.6, beta=0.5):
        super(DirectionalLoss, self).__init__()
        self.alpha = alpha  # 방향성과 크기 간의 가중치
        self.beta = beta    # MSE와 방향성 손실 간의 균형
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE 손실 계산
        mse_loss = self.mse(y_pred, y_true)
        
        # 방향성 손실 계산 (순차적 변화에 대해)
        if y_pred.size(1) > 1:  # 예측이 여러 시점에 대한 것일 경우
            # 인접 시점 간 변화 계산
            direction_true = y_true[:, 1:] - y_true[:, :-1]
            direction_pred = y_pred[:, 1:] - y_pred[:, :-1]
            
            # 방향성 일치 여부 계산 (양/음의 방향이 같은지)
            dir_match = (direction_true * direction_pred > 0).float()
            
            # 방향성 일치율 계산 (1일수록 방향성 일치도 높음)
            dir_match_rate = torch.mean(dir_match)
            
            # 방향성 손실 (1 - 일치율)
            direction_loss = 1 - dir_match_rate
            
            # 최종 손실 = beta * MSE + (1-beta) * 방향성 손실
            return self.beta * mse_loss + (1 - self.beta) * direction_loss
        
        # 단일 시점 예측일 경우 MSE만 반환
        return mse_loss


class CombinedLoss(nn.Module):
    """
    여러 손실 함수를 결합한 손실 함수
    
    Args:
        mse_weight (float): MSE 손실 가중치
        dir_weight (float): 방향성 손실 가중치
        mae_weight (float): MAE 손실 가중치
    """
    def __init__(self, mse_weight=0.4, dir_weight=0.4, mae_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.dir_weight = dir_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE 손실
        mse_loss = self.mse(y_pred, y_true)
        
        # MAE 손실
        mae_loss = torch.mean(torch.abs(y_pred - y_true))
        
        # 방향성 손실 계산 (여러 시점에 대한 예측일 경우)
        if y_pred.size(1) > 1:
            # 인접 시점 간 변화 계산
            direction_true = y_true[:, 1:] - y_true[:, :-1]
            direction_pred = y_pred[:, 1:] - y_pred[:, :-1]
            
            # 방향성 일치 여부 계산
            dir_match = (direction_true * direction_pred > 0).float()
            dir_match_rate = torch.mean(dir_match)
            
            # 방향성 손실
            direction_loss = 1 - dir_match_rate
        else:
            # 단일 시점 예측일 경우 방향성 손실은 0
            direction_loss = 0
            
        # 최종 손실 = 각 손실의 가중 합산
        return (self.mse_weight * mse_loss + 
                self.dir_weight * direction_loss + 
                self.mae_weight * mae_loss)


def train_model(train_loader, model, device, num_epochs=200, loss_fn='huber', delta=1.0, alpha=0.6):
    """
    모델을 학습한다.
    
    Args:
        train_loader (DataLoader): 학습 데이터 로더
        model (nn.Module): 학습할 모델
        device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        num_epochs (int): 학습 에폭 수
        loss_fn (str): 손실 함수 ('mse', 'huber', 'directional', 'combined')
        delta (float): Huber Loss에 사용될 델타 값
        alpha (float): DirectionalLoss에 사용될 방향성 가중치
        
    Returns:
        nn.Module: 학습된 모델
    """
    # 손실 함수 선택
    if loss_fn.lower() == 'huber':
        print(f"Using Huber Loss with delta={delta}")
        criterion = HuberLoss(delta=delta)
    elif loss_fn.lower() == 'directional':
        print(f"Using Directional Loss with alpha={alpha}")
        criterion = DirectionalLoss(alpha=alpha)
    elif loss_fn.lower() == 'combined':
        print(f"Using Combined Loss")
        criterion = CombinedLoss(mse_weight=0.4, dir_weight=0.4, mae_weight=0.2)
    else:
        print("Using MSE Loss")
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
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


def predict_future_prices(model, X_train, train_df, df, scaler, target_col='Coffee_Price', 
                            days=14, device='cpu', save_plot=True, output_path='./data/output/',
                            scale_price=True):
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
        scale_price (bool): 가격 특성 스케일링 여부
        
    Returns:
        tuple: (forecast_series, metrics) 형태의 튜플
            - forecast_series: 예측 결과 (pd.Series)
            - metrics: 성능 지표 (dict)
    """
    # 커피 선물 시장 캘린더 관련 기능 임포트
    try:
        from .market_calendar import adjust_forecast_for_market_calendar
    except ImportError:
        try:
            from models.time_series.attention_lstm_2.market_calendar import adjust_forecast_for_market_calendar
        except ImportError:
            from market_calendar import adjust_forecast_for_market_calendar
    
    # 1. train_df 마지막 날짜의 위치를 df 전체에서 찾는다
    last_train_idx = df.index.get_loc(train_df.index[-1])
    
    # 2. 그 다음 days일치 날짜를 df에서 추출한다
    prediction_dates = df.index[last_train_idx + 1 : last_train_idx + 1 + days]
    
    # 3. 실제값 가져오기 (테스트용)
    try:
        true_values = df.loc[prediction_dates, target_col].values
    except:
        true_values = None
        print("Warning: 예측 날짜에 대한 실제 값을 찾을 수 없다")
    
    # 마지막 시퀀스 예측
    last_seq = X_train[-1].unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        prediction, attn_weights = model(last_seq)
    
    # CPU로 이동 후 numpy 변환
    prediction = prediction.squeeze().cpu().numpy().reshape(-1, 1)
    
    # 가격을 스케일링한 경우, 역변환 필요
    if scale_price:
        # 역변환을 위해 dummy 피처 생성
        dummy = np.zeros((days, train_df.shape[1] - 1))  # target_col 제외 나머지
        prediction_combined = np.concatenate([prediction, dummy], axis=1)
        
        # target_col이 첫 번째 컬럼이라면 그대로 [:, 0]
        prediction = scaler.inverse_transform(prediction_combined)[:, 0]
    else:
        # 가격을 스케일링하지 않은 경우 예측값 그대로 사용
        prediction = prediction.flatten()
    
    # 예측값을 Series로 변환
    forecast_series = pd.Series(prediction, index=prediction_dates)
    
    # 주말 및 휴장일에는 전일 가격을 유지하도록 조정
    forecast_series = adjust_forecast_for_market_calendar(forecast_series)
    
    if save_plot:
        # 결과 시각화
        plt.figure(figsize=(14, 6))
        plt.plot(df[target_col], label='Actual Coffee Price', color='blue')
        plt.plot(prediction_dates, forecast_series, label='Predicted Coffee Price', color='red', linestyle='dashed')
        
        plt.title('Coffee Price Prediction (Enhanced Model)')
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
        plt.savefig(f'{output_path}coffee_prediction_{today}.png', dpi=300, bbox_inches='tight')
    
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


def predict_multiple_sequences(model, test_dataset, test_df, df, scaler, data_window, step, future_target, 
                                target_col='Coffee_Price', device='cpu', scale_price=True):
    """
    테스트 데이터셋에서 여러 시퀀스를 예측한다.
    
    Args:
        model (nn.Module): 학습된 모델
        test_dataset (Dataset): 테스트 데이터셋
        test_df (pd.DataFrame): 테스트 데이터프레임
        df (pd.DataFrame): 전체 데이터프레임
        scaler (MinMaxScaler): 데이터 스케일러
        data_window (int): 입력 윈도우 크기
        step (int): 샘플링 간격
        future_target (int): 예측할 미래 일수
        target_col (str): 예측할 타겟 컬럼명
        device (str): 예측에 사용할 디바이스 ('cuda' 또는 'cpu')
        scale_price (bool): 가격 특성 스케일링 여부
        
    Returns:
        tuple: (predictions, dates_list) 형태의 튜플
            - predictions: 각 시퀀스의 예측 결과 리스트 (pd.Series)
            - dates_list: 예측 날짜 리스트
    """
    model.eval()
    predictions = []
    prediction_dates_list = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x_input, _ = test_dataset[i]
            x_input = x_input.unsqueeze(0).to(device)

            y_pred, _ = model(x_input)
            y_pred = y_pred.squeeze().cpu().numpy().reshape(-1, 1)

            # 가격을 스케일링한 경우에만 역변환 적용
            if scale_price:
                # 역변환
                dummy = np.zeros((future_target, test_df.shape[1] - 1))
                combined = np.concatenate([y_pred, dummy], axis=1)
                y_inv = scaler.inverse_transform(combined)[:, 0]
            else:
                # 가격을 스케일링하지 않은 경우 예측값 그대로 사용
                y_inv = y_pred.flatten()

            # test_df에서 해당 시점의 실제 위치를 찾고, df 전체 인덱스로 변환
            base_test_index = i * step + data_window
            if base_test_index + future_target >= len(test_df):
                break

            start_timestamp = test_df.index[base_test_index]
            try:
                start_pos_in_df = df.index.get_loc(start_timestamp)
            except KeyError:
                continue  # 시작 타임스탬프가 df에 없는 경우 건너뛴다

            date_range = df.index[start_pos_in_df + 1 : start_pos_in_df + 1 + future_target]
            if len(date_range) != future_target:
                continue  # 필요한 미래 날짜 수가 충분하지 않은 경우 건너뛴다

            predictions.append(pd.Series(y_inv, index=date_range))
            prediction_dates_list.extend(date_range)
    
    return predictions, prediction_dates_list


def visualize_predictions(predictions, df, target_col='Coffee_Price', k=2, save_plot=True, output_path='./data/output/'):
    """
    예측 결과를 시각화한다.
    
    Args:
        predictions (list): 예측 결과 시리즈의 리스트
        df (pd.DataFrame): 전체 데이터프레임
        target_col (str): 타겟 컬럼명
        k (int): 시각화할 특정 예측 인덱스
        save_plot (bool): 시각화 결과 저장 여부
        output_path (str): 출력 파일 저장 경로
        
    Returns:
        pd.Series: 모든 예측의 평균값 (forecast_all)
    """
    # 특정 시점의 예측 시각화
    if len(predictions) > k:
        pred_k = predictions[k]
        start_date = pred_k.index[0]
        end_date = pred_k.index[-1]
        true_k = df[target_col].loc[start_date:end_date]

        plt.figure(figsize=(12, 4))
        plt.plot(true_k.index, true_k.values, label='Actual', color='blue')
        plt.plot(pred_k.index, pred_k.values, label='Predicted', color='red', linestyle='dashed')
        plt.title(f"{start_date.date()} to {end_date.date()}")
        plt.xlabel("Date")
        plt.ylabel("Coffee Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_plot:
            today = datetime.now().strftime('%Y%m%d')
            plt.savefig(f'{output_path}coffee_prediction_sample_{today}.png', dpi=300, bbox_inches='tight')
    
    # 전체 예측 결과 시각화 (겹치는 날짜는 평균값)
    forecast_all = pd.concat(predictions, axis=1).mean(axis=1)

    plt.figure(figsize=(14, 6))
    plt.plot(df[target_col], label='Actual Coffee Price', color='blue')
    plt.plot(forecast_all.index, forecast_all.values, label='Predicted Coffee Price (avg)', color='red', linestyle='dashed')

    plt.title('Coffee Price Prediction (Enhanced Model)')
    plt.xlabel('Date')
    plt.ylabel('Coffee Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_plot:
        today = datetime.now().strftime('%Y%m%d')
        plt.savefig(f'{output_path}coffee_prediction_{today}.png', dpi=300, bbox_inches='tight')
    
    return forecast_all