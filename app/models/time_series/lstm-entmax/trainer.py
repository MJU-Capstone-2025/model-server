"""
모델 학습 및 예측 함수들
"""
import pandas as pd
import numpy as np
import torch
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    from .losses import directional_loss, variance_loss
    from .data_loader import save_result
except ImportError:
    from losses import directional_loss, variance_loss
    from data_loader import save_result


def train_model(model, train_loader, test_loader, base_criterion, optimizer, scheduler, 
                num_epochs, alpha, beta, device):
    """
    모델을 학습하고 에포크별 손실을 반환합니다.
    
    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 훈련 데이터 로더
        test_loader (DataLoader): 검증 데이터 로더
        base_criterion (nn.Module): 기본 손실 함수 (MSE)
        optimizer (torch.optim.Optimizer): 옵티마이저
        scheduler (torch.optim.lr_scheduler): 학습률 스케줄러
        num_epochs (int): 학습 에포크 수
        alpha (float): 방향성 손실 가중치
        beta (float): 분산 손실 가중치
        device (str): 학습 장치 ('cuda' 또는 'cpu')
        
    Returns:
        tuple: (train_losses, test_losses) - 에포크별 손실 리스트
    """
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        epoch_loss = 0.0
        
        for x_seq, x_static, y_batch in train_loader:
            x_seq, x_static, y_batch = x_seq.to(device), x_static.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred, _ = model(x_seq, x_static)
            
            # 차원 조정
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            
            # 복합 손실 계산
            base_loss = base_criterion(y_pred, y_batch)
            dir_loss = directional_loss(y_pred, y_batch)
            var_loss = variance_loss(y_pred, y_batch)
            loss = base_loss + alpha * dir_loss + beta * var_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증 모드
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for x_test_seq, x_test_static, y_test in test_loader:
                x_test_seq, x_test_static, y_test = x_test_seq.to(device), x_test_static.to(device), y_test.to(device)
                y_test_pred, _ = model(x_test_seq, x_test_static)
                
                if y_test_pred.ndim == 3 and y_test_pred.shape[-1] == 1:
                    y_test_pred = y_test_pred.squeeze(-1)
                
                base_test_loss = base_criterion(y_test_pred, y_test)
                dir_test_loss = directional_loss(y_test_pred, y_test)
                var_test_loss = variance_loss(y_test_pred, y_test)
                total_test_loss = base_test_loss + alpha * dir_test_loss + beta * var_test_loss
                test_loss += total_test_loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f}")
    
    return train_losses, test_losses


def predict_and_inverse(model, test_loader, scaler, train_df, test_df, df, target_col, 
                       price_col, data_window, future_target, step, static_feat_idx):
    """
    테스트 데이터에 대해 예측을 수행하고 역정규화하여 실제 가격으로 변환합니다.
    
    Args:
        model (nn.Module): 학습된 모델
        test_loader (DataLoader): 테스트 데이터 로더
        scaler (StandardScaler): 정규화에 사용된 스케일러
        train_df (pd.DataFrame): 훈련 데이터프레임
        test_df (pd.DataFrame): 테스트 데이터프레임
        df (pd.DataFrame): 전체 원본 데이터프레임
        target_col (str): 예측 대상 컬럼명 (수익률)
        price_col (str): 가격 컬럼명
        data_window (int): 입력 시퀀스 길이
        future_target (int): 예측 구간 길이
        step (int): 슬라이딩 윈도우 스텝
        static_feat_idx (list): 정적 피처 인덱스
        
    Returns:
        tuple: (forecast_all, predictions) - 평균 예측값과 개별 예측값들
    """
    predictions = []
    return_idx = train_df.columns.get_loc(target_col)
    
    with torch.no_grad():
        for batch_idx, (x_seq, x_static, _) in enumerate(test_loader):
            x_seq, x_static = x_seq.to(model.fc[0].weight.device), x_static.to(model.fc[0].weight.device)
            y_pred_batch, _ = model(x_seq, x_static)
            y_pred_batch = y_pred_batch.cpu().numpy()
            
            for i in range(x_seq.size(0)):
                y_pred = y_pred_batch[i].reshape(-1)
                
                # 수익률을 역정규화
                dummy = np.zeros((future_target, len(train_df.columns)))
                dummy[:, return_idx] = y_pred
                return_inv = scaler.inverse_transform(dummy)[:, return_idx]
                
                # 글로벌 인덱스 계산
                global_idx = (batch_idx * test_loader.batch_size + i) * step + data_window
                if global_idx + future_target >= len(test_df):
                    break
                
                # 시작 시점의 가격 찾기
                start_timestamp = test_df.index[global_idx]
                start_pos_in_df = df.index.get_loc(start_timestamp)
                
                try:
                    start_price = df[price_col].iloc[start_pos_in_df]
                except IndexError:
                    continue
                
                # 예측 날짜 범위 설정
                date_range = df.index[start_pos_in_df + 1 : start_pos_in_df + 1 + future_target]
                if len(date_range) != future_target:
                    continue
                
                # 수익률을 가격으로 변환
                price_pred = [start_price]
                for r in return_inv:
                    price_pred.append(price_pred[-1] * (1 + r))
                price_pred = price_pred[1:]
                
                predictions.append(pd.Series(price_pred, index=date_range))
    
    # 모든 예측값의 평균 계산
    forecast_all = pd.concat(predictions, axis=1).mean(axis=1)
    return forecast_all, predictions

def predict_long_future(model, df, scaler, static_feat_idx, data_window, total_days, horizon, price_col, target_col):
    """
    horizon 단위로 반복 예측하여 total_days만큼 미래를 예측
    """
    all_future_prices = []
    all_future_dates = []
    last_df = df.copy()
    for i in range(0, total_days, horizon):
        cur_horizon = min(horizon, total_days - i)
        future_price_series, future_dates, price_future = predict_future(
            model, last_df, last_df, scaler, static_feat_idx, data_window, cur_horizon, price_col, target_col
        )
        all_future_prices.extend(price_future)
        all_future_dates.extend(future_dates)
        for date, price in zip(future_dates, price_future):
            row = last_df.iloc[-1].copy()
            row[price_col] = price
            row[target_col] = np.nan
            row.name = date
            last_df = pd.concat([last_df, pd.DataFrame([row])])
    return pd.Series(all_future_prices, index=all_future_dates), pd.DatetimeIndex(all_future_dates), all_future_prices

def predict_future(model, test_df, train_df, scaler, static_feat_idx, data_window, 
                  future_target, price_col, target_col):
    """
    현재 시점 이후의 미래 구간에 대한 예측을 수행합니다.
    
    Args:
        model (nn.Module): 학습된 모델
        test_df (pd.DataFrame): 테스트 데이터프레임
        train_df (pd.DataFrame): 훈련 데이터프레임
        scaler (StandardScaler): 정규화 스케일러
        static_feat_idx (list): 정적 피처 인덱스
        data_window (int): 입력 시퀀스 길이
        future_target (int): 예측 구간 길이
        price_col (str): 가격 컬럼명
        target_col (str): 예측 대상 컬럼명
        
    Returns:
        tuple: (future_price_series, future_dates, price_future) - 미래 가격 시리즈, 날짜, 가격 리스트
    """
    # 피처 분리
    all_columns = train_df.drop(columns=[target_col]).columns.tolist()
    static_columns = [all_columns[i] for i in static_feat_idx]
    seq_columns = [col for col in all_columns if col not in static_columns]
    
    # 입력 데이터 준비
    x_seq_input = test_df.iloc[-data_window:][seq_columns].values
    x_seq_input = torch.tensor(x_seq_input, dtype=torch.float32).unsqueeze(0).to(model.fc[0].weight.device)
    
    x_static_input = test_df.iloc[-1][static_columns].values
    x_static_input = torch.tensor(x_static_input, dtype=torch.float32).unsqueeze(0).to(model.fc[0].weight.device)
    
    # 미래 예측 수행
    with torch.no_grad():
        y_pred_future, _ = model(x_seq_input, x_static_input)
        y_pred_future = y_pred_future.squeeze(0).cpu().numpy()
    
    # future_target이 모델 출력보다 작으면 앞부분만 사용
    if future_target < len(y_pred_future):
        y_pred_future = y_pred_future[:future_target]
    
    # 수익률 역정규화
    dummy = np.zeros((future_target, len(train_df.columns)))
    return_idx = train_df.columns.get_loc(target_col)
    dummy[:, return_idx] = y_pred_future
    return_inv = scaler.inverse_transform(dummy)[:, return_idx]
    
    # 수익률을 가격으로 변환
    start_price = test_df[price_col].iloc[-1]
    price_future = [start_price]
    for r in return_inv:
        price_future.append(price_future[-1] * (1 + r))
    price_future = price_future[1:]
    
    # 미래 날짜 생성
    last_date = test_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_target, freq='D')
    future_price_series = pd.Series(price_future, index=future_dates)
    
    return future_price_series, future_dates, price_future


def evaluate_and_save(df, forecast_all, predictions, price_col, future_dates, price_future, data_path=None):
    """
    예측 결과를 평가하고 CSV 파일로 저장합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        forecast_all (pd.Series): 테스트 구간 예측값
        predictions (list): 개별 예측 시리즈 리스트
        price_col (str): 가격 컬럼명
        future_dates (pd.DatetimeIndex): 미래 예측 날짜
        price_future (list): 미래 예측 가격 리스트
        data_path (str, optional): 저장할 파일 경로. None이면 기본 경로 사용.
    """
    # 성능 평가
    actual = df.loc[forecast_all.index, price_col]
    predicted = forecast_all
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    print(f"=== 모델 성능 평가 ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    # 결과 데이터프레임 생성
    pred_series = pd.concat(predictions).sort_index()
    true_series = pd.Series(df[price_col], index=pred_series.index)
    
    # 테스트 구간 결과
    result_df = pd.DataFrame({
        "Date": pred_series.index, 
        "Predicted_Price": pred_series.values, 
        "Actual_Price": true_series.values
    })
    result_df["Date"] = pd.to_datetime(result_df["Date"])
    result_df = result_df.sort_values("Date").reset_index(drop=True)
    
    # 미래 구간 결과
    future_df = pd.DataFrame({
        "Date": future_dates, 
        "Predicted_Price": price_future, 
        "Actual_Price": [None] * len(price_future)
    })
    
    # 전체 결과 결합 및 저장
    X_all = pd.concat([
        pd.DataFrame(result_df, columns=["Date", "Predicted_Price", "Actual_Price"]),
        pd.DataFrame(future_df, columns=["Date", "Predicted_Price", "Actual_Price"])
    ], axis=0)
    X_all.drop_duplicates(subset=["Date"], inplace=True)
    save_result(X_all, data_path) 