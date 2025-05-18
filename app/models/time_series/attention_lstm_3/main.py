"""
Attention LSTM 모델 메인 실행 모듈 (Main Execution Module)

이 모듈은 Attention LSTM 모델 파이프라인을 실행하는 메인 코드다.
데이터 로딩, 전처리, 모델 학습, 예측, 및 결과 저장을 포함하는 전체 프로세스를 조정한다.
버전 2는 더 세밀한 특성 엔지니어링과 최적화된 하이퍼파라미터를 사용한다.

주요 기능:
1. 명령행 인수 처리
2. 데이터 로드 및 전처리
3. 모델 학습 및 예측 파이프라인 실행
4. 결과 저장 및 시각화

실행 방법:
    python -m app.models.time_series.attention_lstm_2.main --epochs 200 --batch_size 10 --scale_price False
    또는
    cd app
    python -m models.time_series.attention_lstm_2.main --epochs 200 --batch_size 10 --scale_price False
"""

import argparse
import torch
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime

# 패키지 경로 설정을 위한 코드
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

# 직접 실행 시 현재 디렉토리를 Python 경로에 추가
if __name__ == "__main__" and os.path.basename(os.getcwd()) == "app":
    sys.path.insert(0, os.getcwd())

# 절대 경로로 변경
try:
    # 패키지로 임포트될 때 사용하는 상대 경로
    from .data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
    from .dataset import MultiStepTimeSeriesDataset
    from .model import AttentionLSTMModel
    from .training import train_model, predict_future_prices, predict_multiple_sequences, visualize_predictions
    from .utils import save_prediction_to_csv, save_model
except ImportError:
    # 직접 실행될 때 사용하는 절대 경로
    try:
        from models.time_series.attention_lstm_2.data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
        from models.time_series.attention_lstm_2.dataset import MultiStepTimeSeriesDataset
        from models.time_series.attention_lstm_2.model import AttentionLSTMModel
        from models.time_series.attention_lstm_2.training import train_model, predict_future_prices, predict_multiple_sequences, visualize_predictions
        from models.time_series.attention_lstm_2.utils import save_prediction_to_csv, save_model
    except ModuleNotFoundError:
        # 디렉토리 내에서 직접 실행할 때
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        from data_preprocessing import load_and_prepare_data, train_test_split, scale_data, scale_data_except_price
        from dataset import MultiStepTimeSeriesDataset
        from model import AttentionLSTMModel
        from training import train_model, predict_future_prices, predict_multiple_sequences, visualize_predictions
        from utils import save_prediction_to_csv, save_model


def main(macro_data_path, climate_data_path, output_path='./data/output/', 
         data_window=50, future_target=14, step=1, batch_size=10, 
         hidden_size=100, num_epochs=100, do_multiple_predictions=True, 
         scale_price=True, loss_fn='huber', delta=1.0, alpha=0.6):
    """
    모델 학습 및 예측을 실행하는 메인 함수다.
    
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
        do_multiple_predictions (bool): 여러 시퀀스에 대한 예측 수행 여부 (기본값 True)
        scale_price (bool): 가격 특성(Coffee_Price) 스케일링 여부 (기본값 True)
        loss_fn (str): 손실 함수 ('mse', 'huber', 'directional', 'combined')
        delta (float): Huber Loss에 사용될 델타 값
        alpha (float): DirectionalLoss에 사용될 방향성 가중치
        
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
    
    # 데이터 스케일링 - 가격 스케일링 여부에 따라 다른 함수 사용
    if scale_price:
        print("Scaling all features including Coffee_Price...")
        scaled_train_df, scaled_test_df, scaler = scale_data(train_df, test_df, preserve_return=True)
    else:
        print("Excluding Coffee_Price from scaling to preserve original price values...")
        scaled_train_df, scaled_test_df, scaler = scale_data_except_price(train_df, test_df)
    
    # 학습 데이터 준비
    X_train = scaled_train_df.values
    X_test = scaled_test_df.values
    target_col = "Coffee_Price"
    y_train = scaled_train_df[target_col].values
    y_test = scaled_test_df[target_col].values
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, future_target, step)
    test_dataset = MultiStepTimeSeriesDataset(X_test, y_test, data_window, future_target, step)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    input_size = X_train.shape[1]
    model = AttentionLSTMModel(
        input_size=input_size, 
        hidden_size=hidden_size, 
        target_size=future_target
    ).to(device)
    
    # 모델 학습
    print(f"Starting model training with {num_epochs} epochs...")
    model = train_model(train_loader, model, device, num_epochs=num_epochs, 
                       loss_fn=loss_fn, delta=delta, alpha=alpha)
    
    # 미래 가격 예측
    print("Predicting future prices...")
    forecast_series, metrics = predict_future_prices(
        model, train_dataset.data, train_df, df, scaler, 
        target_col=target_col, days=future_target, device=device,
        output_path=output_path, scale_price=scale_price
    )
    
    # 여러 시퀀스에 대한 예측 수행 (옵션)
    if do_multiple_predictions and len(test_dataset) > 0:
        print(f"Performing predictions on {len(test_dataset)} test sequences...")
        predictions, dates_list = predict_multiple_sequences(
            model, test_dataset, test_df, df, scaler,
            data_window, step, future_target, target_col, device,
            scale_price=scale_price
        )
        
        if predictions:
            print(f"Generated {len(predictions)} sequence predictions")
            # 예측 결과 시각화 및 평균 계산
            forecast_all = visualize_predictions(predictions, df, target_col, save_plot=True, output_path=output_path)
            
            # 예측 결과 저장
            prediction_path = save_prediction_to_csv(forecast_all, output_path=output_path)
            print(f"Multiple predictions saved to: {prediction_path}")
        else:
            print("No valid multiple predictions generated")
    
    # 단일 예측 저장
    prediction_path = save_prediction_to_csv(forecast_series, output_path=output_path)
    print(f"Single prediction saved to: {prediction_path}")
    
    # 모델 저장
    hyperparameters = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': model.num_layers,
        'target_size': model.target_size,
        'data_window': data_window,
        'step': step,
        'future_target': future_target
    }
    model_path = save_model(model, scaler, hyperparameters, metrics, output_path)
    print(f"Model saved to: {model_path}")
    
    return forecast_series, model, scaler


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

    args = parser.parse_args()
    
    # 현재 작업 디렉토리 확인 및 파일 경로 조정
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # 파일 경로 확인 및 설정
    for path_arg in ['macro_data', 'climate_data', 'output_path']:
        path = getattr(args, path_arg)
        if not os.path.isabs(path):
            if path.startswith('./'):
                path = path[2:]  # './' 제거
            
            if os.path.basename(cwd) == 'app':
                # app 디렉토리 내에서 실행 시
                abs_path = os.path.join(cwd, path)
            else:
                # 프로젝트 루트 디렉토리에서 실행 시
                abs_path = os.path.join(cwd, 'app', path)
                
            if path_arg != 'output_path' and not os.path.exists(abs_path):
                print(f"Warning: File {abs_path} does not exist")
            
            setattr(args, path_arg, abs_path)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Starting main with epochs={args.epochs}, batch_size={args.batch_size}, scale_price={args.scale_price}, loss_fn={args.loss_fn}, delta={args.delta}, alpha={args.alpha}")
    
    # 메인 함수 실행
    forecast, model, scaler = main(
        args.macro_data,
        args.climate_data,
        args.output_path,
        data_window=args.data_window,
        future_target=args.future_target,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_epochs=args.epochs,
        scale_price=args.scale_price,
        loss_fn=args.loss_fn,
        delta=args.delta,
        alpha=args.alpha
    )
    
    # 모델은 이미 main 함수 내에서 저장되었습니다.
    print(f"Script execution complete.")