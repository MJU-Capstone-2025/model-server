"""
잔차 기반 어텐션 LSTM 파이프라인 (Residual-Based Attention LSTM Pipeline)

이 모듈은 이전 예측 오차(잔차)를 활용하는 Attention LSTM 모델의 파이프라인을 제공한다.
데이터셋 생성부터 모델 학습, 예측, 평가까지 전체 프로세스를 통합적으로 실행한다.

주요 기능:
1. 잔차 활용 데이터셋 구성
2. 잔차 기반 모델 학습
3. 모델 성능 비교 및 평가

사용 예시:
    python -m app.models.time_series.attention_lstm_2.residual_main --epochs 200 --batch_size 10 --compare True
    또는
    cd app
    python -m models.time_series.attention_lstm_2.residual_main --epochs 200 --batch_size 10 --compare True
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd  # 추가된 임포트
import matplotlib.pyplot as plt
from collections import defaultdict

# 패키지 경로 설정을 위한 코드
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

# 직접 실행 시 현재 디렉토리를 Python 경로에 추가
if __name__ == "__main__" and os.path.basename(os.getcwd()) == "app":
    sys.path.insert(0, os.getcwd())

# 절대 경로로 임포트 시도
try:
    # 패키지로 임포트될 때 사용하는 상대 경로
    from .data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
    from .dataset import MultiStepTimeSeriesDataset
    from .residual_dataset import ResidualTimeSeriesDataset
    from .model import AttentionLSTMModel
    from .residual_model import ResidualAttentionLSTM
    from .training import train_model, predict_future_prices
    from .residual_training import train_with_residuals, predict_with_residuals, compare_models, calculate_residuals
    from .utils import save_prediction_to_csv, save_model, save_test_predictions_to_csv
except ImportError:
    # 직접 실행될 때 사용하는 절대 경로
    try:
        from models.time_series.attention_lstm_2.data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
        from models.time_series.attention_lstm_2.dataset import MultiStepTimeSeriesDataset
        from models.time_series.attention_lstm_2.residual_dataset import ResidualTimeSeriesDataset
        from models.time_series.attention_lstm_2.model import AttentionLSTMModel
        from models.time_series.attention_lstm_2.residual_model import ResidualAttentionLSTM
        from models.time_series.attention_lstm_2.training import train_model, predict_future_prices
        from models.time_series.attention_lstm_2.residual_training import train_with_residuals, predict_with_residuals, compare_models, calculate_residuals
        from models.time_series.attention_lstm_2.utils import save_prediction_to_csv, save_model, save_test_predictions_to_csv
    except ModuleNotFoundError:
        # 디렉토리 내에서 직접 실행할 때
        from data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
        from dataset import MultiStepTimeSeriesDataset
        from residual_dataset import ResidualTimeSeriesDataset
        from model import AttentionLSTMModel
        from residual_model import ResidualAttentionLSTM
        from training import train_model, predict_future_prices
        from residual_training import train_with_residuals, predict_with_residuals, compare_models, calculate_residuals
        from utils import save_prediction_to_csv, save_model, save_test_predictions_to_csv


def main_residual_pipeline(macro_data_path, climate_data_path, output_path='./data/output/', 
                            data_window=50, future_target=14, step=1, batch_size=10, 
                            hidden_size=100, num_epochs=100, scale_price=True, 
                            loss_fn='huber', delta=1.0, alpha=0.6, 
                            residual_window=5, compare=True):
    """
    잔차 기반 모델 파이프라인을 실행한다.
    
    Args:
        macro_data_path (str): 거시경제 데이터 파일 경로
        climate_data_path (str): 기후 데이터 파일 경로
        output_path (str): 출력 파일 저장 경로
        data_window (int): 입력 윈도우 크기 (기본값 50)
        future_target (int): 예측할 미래 일수 (기본값 14)
        step (int): 샘플링 간격 (기본값 1)
        batch_size (int): 미니배치 크기 (기본값 10)
        hidden_size (int): LSTM 은닉층 크기 (기본값 100)
        num_epochs (int): 학습 에폭 수 (기본값 100)
        scale_price (bool): 가격 특성(Coffee_Price) 스케일링 여부
        loss_fn (str): 손실 함수 ('mse', 'huber', 'directional', 'combined')
        delta (float): Huber Loss에 사용될 델타 값
        alpha (float): DirectionalLoss에 사용될 방향성 가중치
        residual_window (int): 잔차 윈도우 크기
        compare (bool): 기본 모델과 잔차 모델 비교 여부
        
    Returns:
        tuple: (base_model, residual_model, forecast) 형태의 튜플
    """
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"> Using device: {device}")
    
    # 데이터 로드 및 전처리
    df = load_and_prepare_data(macro_data_path, climate_data_path)
    print(f"> Data loaded with shape: {df.shape}")
    
    # 학습/테스트 분할
    train_df, test_df = train_test_split(df)
    print(f"> Training data: {train_df.shape}, Test data: {test_df.shape}")
    
    # 데이터 스케일링
    if scale_price:
        print("> Scaling all features including Coffee_Price...")
        scaled_train_df, scaled_test_df, scaler = scale_data(train_df, test_df, preserve_return=True)
    else:
        print("> Excluding Coffee_Price from scaling to preserve original price values...")
        scaled_train_df, scaled_test_df, scaler = scale_data_except_price(train_df, test_df)
    
    # 학습 데이터 준비
    X_train = scaled_train_df.values
    X_test = scaled_test_df.values
    target_col = "Coffee_Price"
    y_train = scaled_train_df[target_col].values
    y_test = scaled_test_df[target_col].values
    
    # 단계 1: 기본 모델 (잔차를 사용하지 않는 모델) 학습
    print("\n================ STEP 1: Training Base Model ================")
    # 기본 데이터셋 생성
    base_train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, future_target, step)
    base_train_loader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    
    # 기본 모델 초기화
    input_size = X_train.shape[1]
    base_model = AttentionLSTMModel(
        input_size=input_size, 
        hidden_size=hidden_size, 
        target_size=future_target
    ).to(device)
    
    # 기본 모델 학습
    print(f"> Starting base model training with {num_epochs} epochs...")
    base_model = train_model(base_train_loader, base_model, device, num_epochs=num_epochs, 
                            loss_fn=loss_fn, delta=delta, alpha=alpha)
    
    # 단계 2: 기본 모델로 잔차 계산
    print("\n================ STEP 2: Calculating Residuals ================")
    residuals = calculate_residuals(base_model, base_train_dataset, device)
    print(f"> Calculated residuals with shape: {residuals.shape}")
    
    # 단계 3: 잔차 활용 모델 학습
    print("\n================ STEP 3: Training Residual Model ================")
    # 잔차 데이터셋 생성
    residual_train_dataset = ResidualTimeSeriesDataset(
        X_train, y_train, residuals=residuals,
        data_window=data_window, target_size=future_target, step=step,
        residual_window=residual_window, use_residuals=True
    )
    residual_train_loader = DataLoader(residual_train_dataset, batch_size=batch_size, shuffle=True)
    # 잔차 모델 초기화
    residual_model = ResidualAttentionLSTM(
        input_size=input_size,
        residual_size=future_target,  # 잔차 차원을 future_target(14)로 변경
        hidden_size=hidden_size,
        target_size=future_target
    ).to(device)
    
    # 잔차 모델 학습
    print(f"> Starting residual model training with {num_epochs} epochs...")
    residual_model, final_residuals = train_with_residuals(
        residual_train_loader, residual_model, device, 
        num_epochs=num_epochs, loss_fn=loss_fn, delta=delta, alpha=alpha,
        residual_update_freq=10  # 10 에폭마다 잔차 업데이트
    )
    
    # 단계 4: 예측 및 결과 비교
    print("\n================ STEP 4: Making Predictions ================")
    # 기본 모델 예측
    base_forecast, base_metrics = predict_future_prices(
        base_model, base_train_dataset.data, train_df, df, scaler, 
        target_col=target_col, days=future_target, device=device,
        output_path=output_path, scale_price=scale_price
    )
    
    # 잔차 활용 모델 예측
    residual_forecast, residual_metrics = predict_with_residuals(
        residual_model, residual_train_dataset.data, final_residuals,
        train_df, df, scaler, target_col=target_col, days=future_target, 
        device=device, output_path=output_path, scale_price=scale_price
    )
    
    # 모델 저장
    base_hyperparams = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': base_model.num_layers,
        'target_size': base_model.target_size,
        'data_window': data_window,
        'step': step,
        'future_target': future_target
    }
    
    residual_hyperparams = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'target_size': future_target,
        'data_window': data_window,
        'residual_window': residual_window,
        'step': step,
        'future_target': future_target
    }
    
    # 기본 모델 저장
    base_model_path = save_model(
        base_model, scaler, base_hyperparams, base_metrics,
        output_path, model_prefix="base_model"
    )
    
    # 잔차 모델 저장
    residual_model_path = save_model(
        residual_model, scaler, residual_hyperparams, residual_metrics,
        output_path, model_prefix="residual_model"
    )
    
    print(f"> Base model saved to: {base_model_path}")
    print(f"> Residual model saved to: {residual_model_path}")
    
    # 단계 5: 모델 성능 비교 (선택 사항)
    if compare:
        print("\n================ STEP 5: Comparing Models ================")
        comparison_df = compare_models(
            base_model, residual_model, base_train_dataset.data, 
            final_residuals, train_df, df, scaler,
            target_col=target_col, device=device, save_path=output_path
        )
        
        # 향상도 출력
        mae_improvement = comparison_df['MAE Improvement'].iloc[1]
        rmse_improvement = comparison_df['RMSE Improvement'].iloc[1]
        print(f"\n> Performance Improvement Summary:")
        print(f"> MAE: Improved by {mae_improvement:.2f}%")
        print(f"> RMSE: Improved by {rmse_improvement:.2f}%")
    
    # 단계 6: 테스트 세트 전체에 대해 반복적으로 예측 수행
    print("\n================ STEP 6: Making Full Test Set Predictions ================")
    base_pred_dict = defaultdict(list)
    residual_pred_dict = defaultdict(list)

    for start_idx in range(0, len(X_test) - data_window - future_target + 1, 1):  # step=1
        test_window = X_test[start_idx:start_idx + data_window]
        test_window = test_window[np.newaxis, :, :]  # 배치 차원 추가

        # 기본 모델 예측
        base_pred, _ = base_model(torch.tensor(test_window, dtype=torch.float32).to(device))
        base_forecast = base_pred.cpu().detach().numpy()

        # 잔차 모델 예측
        residual_pred, _ = residual_model(torch.tensor(test_window, dtype=torch.float32).to(device))
        residual_forecast = residual_pred.cpu().detach().numpy()

        # === 역스케일링 ===
        if scale_price:
            dummy = np.zeros((future_target, X_test.shape[1] - 1))
            base_inv = scaler.inverse_transform(np.concatenate([base_forecast.reshape(-1, 1), dummy], axis=1))[:, 0]
            residual_inv = scaler.inverse_transform(np.concatenate([residual_forecast.reshape(-1, 1), dummy], axis=1))[:, 0]
        else:
            base_inv = base_forecast.flatten()
            residual_inv = residual_forecast.flatten()

        # 날짜 계산
        start_date = test_df.index[start_idx + data_window]
        forecast_dates = pd.date_range(start=start_date, periods=future_target)

        # 날짜별로 예측값 누적
        for d, b, r in zip(forecast_dates, base_inv, residual_inv):
            base_pred_dict[d].append(b)
            residual_pred_dict[d].append(r)

    # 날짜별 평균 계산 및 저장
    all_dates = sorted(set(base_pred_dict.keys()) & set(residual_pred_dict.keys()))
    all_test_predictions = []
    all_test_dates = []
    for d in all_dates:
        base_avg = np.mean(base_pred_dict[d])
        residual_avg = np.mean(residual_pred_dict[d])
        all_test_predictions.append((d, base_avg, residual_avg))
        all_test_dates.append(d)

    test_predictions_path = os.path.join(output_path, "test_predictions.csv")
    save_test_predictions_to_csv(all_test_dates, 
                                [pred[1] for pred in all_test_predictions], 
                                [pred[2] for pred in all_test_predictions], 
                                test_predictions_path)
    print(f"Test set predictions with dates saved to: {test_predictions_path}")
    
    return base_model, residual_model, residual_forecast


def plot_test_predictions_with_actual(csv_path, macro_data_path, climate_data_path, output_path='./data/output/'):
    """
    test_predictions.csv와 실제 커피 가격을 한 그래프에 시각화한다.
    """
    # 예측 결과 CSV 읽기
    df_pred = pd.read_csv(csv_path, parse_dates=['Date'])
    # 실제 데이터 로드
    df_actual = load_and_prepare_data(macro_data_path, climate_data_path)
    # 실제값에서 예측 구간만 추출
    df_actual = df_actual.loc[df_pred['Date'].min():df_pred['Date'].max()]
    # 그래프 그리기
    plt.figure(figsize=(16, 7))
    plt.plot(df_pred['Date'], df_pred['Base Model Predictions'], label='Base Model', color='red', linestyle='--')
    plt.plot(df_pred['Date'], df_pred['Residual Model Predictions'], label='Residual Model', color='green', linestyle=':')
    plt.plot(df_actual.index, df_actual['Coffee_Price'], label='Actual Price', color='blue', linewidth=2)
    plt.title('Test Set Prediction vs Actual Coffee Price')
    plt.xlabel('Date')
    plt.ylabel('Coffee Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'test_predictions_plot.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--macro_data', type=str, default='./data/input/거시경제및커피가격통합데이터.csv', help='Path to macro data')
    parser.add_argument('--climate_data', type=str, default='./data/input/기후데이터피쳐선택.csv', help='Path to climate data')
    parser.add_argument('--output_path', type=str, default='./data/output/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--data_window', type=int, default=50, help='Data window size')
    parser.add_argument('--future_target', type=int, default=14, help='Future prediction days')
    parser.add_argument('--hidden_size', type=int, default=100, help='LSTM hidden size')
    parser.add_argument('--scale_price', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Whether to scale price feature (True/False)')
    parser.add_argument('--loss_fn', type=str, default='huber', 
                        choices=['mse', 'huber', 'directional', 'combined'], 
                        help='Loss function to use (mse, huber, directional, combined)')
    parser.add_argument('--delta', type=float, default=1.0, help='Delta parameter for Huber Loss')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for Directional Loss')
    parser.add_argument('--residual_window', type=int, default=5, help='Residual window size')
    parser.add_argument('--compare', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Whether to compare models (True/False)')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    base_model, residual_model, forecast = main_residual_pipeline(
        args.macro_data, 
        args.climate_data,
        output_path=args.output_path,
        data_window=args.data_window,
        future_target=args.future_target,
        step=1,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_epochs=args.epochs,
        scale_price=args.scale_price,
        loss_fn=args.loss_fn,
        delta=args.delta,
        alpha=args.alpha,
        residual_window=args.residual_window,
        compare=args.compare
    )

    # test_predictions.csv 시각화
    test_pred_csv = os.path.join(args.output_path, 'test_predictions.csv')
    plot_test_predictions_with_actual(
        test_pred_csv,
        args.macro_data,
        args.climate_data,
        output_path=args.output_path
    )
