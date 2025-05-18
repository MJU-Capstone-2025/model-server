"""
잔차 기반 학습 및 예측 모듈 (Residual-based Training and Prediction Module)

이 모듈은 이전 예측 오차(residual)를 활용한 모델 학습 및 예측 기능을 제공한다.
잔차를 활용하여 모델이 예측 오류를 학습하고 보정하도록 한다.

주요 기능:
1. 잔차 기반 모델 학습 (train_with_residuals)
2. 잔차 정보를 활용한 미래 가격 예측 (predict_with_residuals)
3. 성능 향상 효과 평가 (compare_models)

사용 예시:
    from residual_training import train_with_residuals, predict_with_residuals, compare_models
    
    # 잔차 기반 모델 학습
    residual_model, residuals = train_with_residuals(
        train_loader, residual_model, device, num_epochs=200
    )
    
    # 잔차 기반 미래 가격 예측
    forecast, metrics = predict_with_residuals(
        residual_model, X_train, residuals, train_df, df, scaler, 
        target_col='Coffee_Price', days=14
    )
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_residuals(model, dataset, device):
    """
    데이터셋에 대한 모델의 예측 오차(잔차)를 계산한다.
    
    Args:
        model (nn.Module): 예측에 사용할 모델
        dataset (Dataset): 예측할 데이터셋
        device (str): 모델이 로드된 디바이스 ('cuda' 또는 'cpu')
        
    Returns:
        np.ndarray: 예측 오차(잔차) 배열
    """
    model.eval()
    residuals = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # 단일 샘플에 대한 예측
            inputs = dataset[i][0].unsqueeze(0).to(device)
            targets = dataset[i][-1].unsqueeze(0).to(device)
            
            # 잔차 데이터셋인 경우 잔차 정보 건너뛰기
            if len(dataset[i]) == 3:
                # No residuals for initial calculation
                outputs, _ = model(inputs, None)
            else:
                outputs, _ = model(inputs)
                
            # 예측 오차(잔차) 계산: 실제값 - 예측값
            residual = (targets - outputs).cpu().numpy()
            residuals.append(residual.squeeze())
    
    return np.array(residuals)


def train_with_residuals(train_loader, model, device, num_epochs=200, 
                            loss_fn='huber', delta=1.0, alpha=0.6, 
                            residual_update_freq=10):
    """
    잔차 정보를 활용하여 모델을 학습한다.
    
    Args:
        train_loader (DataLoader): 잔차 데이터를 포함한 학습 데이터 로더
        model (nn.Module): 학습할 모델
        device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        num_epochs (int): 학습 에폭 수
        loss_fn (str): 손실 함수 ('mse', 'huber', 'directional', 'combined')
        delta (float): Huber Loss에 사용될 델타 값
        alpha (float): DirectionalLoss에 사용될 방향성 가중치
        residual_update_freq (int): 잔차 업데이트 주기 (에폭 단위)
        
    Returns:
        tuple: (model, final_residuals) - 학습된 모델과 최종 잔차
    """
    from .training import HuberLoss, DirectionalLoss, CombinedLoss
    
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
    
    # 현재 배치의 잔차 정보를 저장할 딕셔너리
    residuals_dict = {}
    
    # 데이터셋에서 첫 번째 배치를 추출하여 residual 사용 여부 확인
    sample_batch = next(iter(train_loader))
    use_residuals = len(sample_batch) == 3
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if use_residuals:
                x_batch, r_batch, y_batch = batch
                x_batch, r_batch, y_batch = x_batch.to(device), r_batch.to(device), y_batch.to(device)
            else:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                r_batch = None
            
            optimizer.zero_grad()
            
            # 모델 예측
            y_pred, _ = model(x_batch, r_batch)
            
            # (선택적으로) 차원 조정
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            
            # Loss 계산
            loss = criterion(y_pred, y_batch)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            # 현재 배치의 잔차 계산 및 저장
            with torch.no_grad():
                residuals = y_batch - y_pred
                for i, res in enumerate(residuals):
                    batch_index = batch_idx * train_loader.batch_size + i
                    if batch_index < len(train_loader.dataset):
                        residuals_dict[batch_index] = res.detach().cpu().numpy()
            
            epoch_loss += loss.item()
        
        # 잔차 업데이트 주기에 따라 데이터셋 업데이트
        if use_residuals and (epoch + 1) % residual_update_freq == 0:
            # 잔차 배열 생성
            new_residuals = np.zeros((len(train_loader.dataset), y_batch.size(1)))
            for idx, res in residuals_dict.items():
                if idx < len(new_residuals):
                    new_residuals[idx] = res
            
            # 디버깅: 데이터 형태 출력
            print(f"DEBUG: new_residuals shape: {new_residuals.shape}")
            print(f"DEBUG: dataset residuals shape: {train_loader.dataset.residuals.shape}")
            
            # 데이터셋의 잔차 업데이트 - 안전하게 업데이트하기
            try:
                # 먼저 update_residuals 메소드 사용 시도
                train_loader.dataset.update_residuals(new_residuals)
                print("DEBUG: Residuals updated via update_residuals method")
            except Exception as e:
                print(f"WARNING: Error updating residuals with update_residuals method: {e}")
                # 실패 시 직접 새 잔차 설정 시도
                if hasattr(train_loader.dataset, 'residuals'):
                    try:
                        # 차원 확인 및 조정 후 업데이트
                        train_loader.dataset.residuals = torch.tensor(new_residuals, dtype=torch.float32)
                        print(f"DEBUG: Updated residuals directly")
                    except Exception as e2:
                        print(f"ERROR: Failed to update residuals directly: {e2}")
            
            # 잔차 딕셔너리 초기화
            residuals_dict = {}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    
    # 최종 잔차 배열 생성
    final_residuals = np.zeros((len(train_loader.dataset), y_batch.size(1)))
    for idx, res in residuals_dict.items():
        if idx < len(final_residuals):
            final_residuals[idx] = res
            
    return model, final_residuals


def predict_with_residuals(model, X_data, residuals, train_df, df, scaler,
                            target_col='Coffee_Price', days=14, device='cpu', 
                            save_plot=True, output_path='./data/output/',
                            scale_price=True):
    """
    이전 잔차를 활용하여 미래 가격을 예측한다.
    
    Args:
        model (nn.Module): 학습된 모델
        X_data (torch.Tensor): 입력 데이터 텐서
        residuals (np.ndarray): 이전 예측의 잔차
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
    """
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
    
    # 최근 잔차 데이터 준비 (최대 5개)
    if residuals is not None and len(residuals) > 0:
        recent_residuals = residuals[-5:] if len(residuals) >= 5 else np.pad(
            residuals, ((5 - len(residuals), 0), (0, 0)), 'constant'
        )
        residual_tensor = torch.tensor(recent_residuals, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        residual_tensor = None
    
    # 마지막 시퀀스 예측
    last_seq = X_data[-1].unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        prediction, attn_weights = model(last_seq, residual_tensor)
    
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
    
    if save_plot:
        # 결과 시각화
        plt.figure(figsize=(14, 6))
        plt.plot(df[target_col], label='Actual Coffee Price', color='blue')
        plt.plot(prediction_dates, forecast_series, label='Predicted with Residuals', color='red', linestyle='dashed')
        
        plt.title('Coffee Price Prediction with Residual Learning')
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
        plt.savefig(f'{output_path}coffee_prediction_residual_{today}.png', dpi=300, bbox_inches='tight')
    
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


def compare_models(base_model, residual_model, X_data, residuals, test_df, df,
                    scaler, target_col='Coffee_Price', device='cpu', 
                    save_path='./data/output/'):
    """
    기본 모델과 잔차 활용 모델의 성능을 비교한다.
    
    Args:
        base_model (nn.Module): 기본 모델
        residual_model (nn.Module): 잔차 활용 모델
        X_data (torch.Tensor): 입력 데이터 텐서
        residuals (np.ndarray): 잔차 데이터
        test_df (pd.DataFrame): 테스트 데이터프레임
        df (pd.DataFrame): 전체 데이터프레임
        scaler (MinMaxScaler): 데이터 스케일러
        target_col (str): 타겟 컬럼명
        device (str): 사용할 디바이스 ('cuda' 또는 'cpu')
        save_path (str): 결과 저장 경로
        
    Returns:
        pd.DataFrame: 성능 비교 결과
    """
    from .training import predict_future_prices
    
    # 기본 모델 예측
    base_forecast, base_metrics = predict_future_prices(
        base_model, X_data, test_df, df, scaler,
        target_col=target_col, save_plot=False, device=device
    )
    
    # 잔차 활용 모델 예측
    residual_forecast, residual_metrics = predict_with_residuals(
        residual_model, X_data, residuals, test_df, df, scaler,
        target_col=target_col, save_plot=False, device=device
    )
    
    # 실제 값 가져오기
    last_test_idx = df.index.get_loc(test_df.index[-1])
    prediction_dates = df.index[last_test_idx + 1 : last_test_idx + 1 + len(base_forecast)]
    true_values = df.loc[prediction_dates, target_col]
    
    # 결과 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(true_values.index, true_values, label='Actual', color='blue', linewidth=2)
    plt.plot(base_forecast.index, base_forecast, label='Base Model', color='red', linestyle='dashed')
    plt.plot(residual_forecast.index, residual_forecast, label='Residual Model', color='green', linestyle='dotted')
    
    plt.title('Model Comparison: Base vs Residual Enhanced')
    plt.xlabel('Date')
    plt.ylabel('Coffee Price')
    plt.legend()
    plt.grid(True)
    
    # 성능 개선 계산
    mae_improvement = ((base_metrics['MAE'] - residual_metrics['MAE']) / base_metrics['MAE']) * 100
    rmse_improvement = ((base_metrics['RMSE'] - residual_metrics['RMSE']) / base_metrics['RMSE']) * 100
    
    # 성능 정보 추가
    plt.figtext(0.15, 0.15, 
                f"Base Model: MAE={base_metrics['MAE']:.4f}, RMSE={base_metrics['RMSE']:.4f}\n" +
                f"Residual Model: MAE={residual_metrics['MAE']:.4f}, RMSE={residual_metrics['RMSE']:.4f}\n" +
                f"Improvement: MAE {mae_improvement:.2f}%, RMSE {rmse_improvement:.2f}%",
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    today = datetime.now().strftime('%Y%m%d')
    plt.savefig(f'{save_path}model_comparison_{today}.png', dpi=300, bbox_inches='tight')
    
    # 성능 비교 테이블 생성
    comparison_data = {
        'Model': ['Base Model', 'Residual Model'],
        'MAE': [base_metrics['MAE'], residual_metrics['MAE']],
        'RMSE': [base_metrics['RMSE'], residual_metrics['RMSE']],
        'MAE Improvement': [0, mae_improvement],
        'RMSE Improvement': [0, rmse_improvement]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # CSV로 저장
    comparison_df.to_csv(f'{save_path}model_comparison_{today}.csv', index=False)
    
    return comparison_df
