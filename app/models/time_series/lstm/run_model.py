"""
LSTM + Attention + Entmax 모델을 이용한 커피 생두 가격 예측 모듈

실행 방법:
python -m src.models.lstm_attention.run_model --loss_fn mse --epochs 10
"""

import os
import sys
import torch
import numpy as np
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

# 패키지 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

# 패키지 구조에 맞는 상대 import
from .data_preprocessor import *
from .model import *
from .utils import *
from .debug import *

def parse_arguments():
    """명령줄 인자 파싱 함수"""
    parser = argparse.ArgumentParser(description='커피 생두 가격 예측 모델링')
    
    # 손실 함수 선택 (mse 또는 huber)
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'huber'],
                        help='손실 함수 유형 (mse 또는 huber)')
    
    # Huber loss의 delta 값 설정 (huber loss 사용 시에만 적용)
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber loss의 delta 값 (huber loss 사용 시에만 적용)')
    
    # 훈련 에폭 수 설정
    parser.add_argument('--epochs', type=int, default=5,
                        help='훈련 에폭 수')
    
    # 학습률 설정
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률')
    
    parser.add_argument('--online', action='store_true',
                    help='온라인 업데이트 방식으로 예측 수행')

    return parser.parse_args()

def main(loss_fn='mse', delta=1.0, epochs=5, lr=0.001, online=False):
    """
    메인 실행 함수
    
    Args:
        loss_fn (str): 손실 함수 유형 ('mse' 또는 'huber')
        delta (float): Huber 손실 함수의 delta 값 (huber 사용 시에만 적용)
        epochs (int): 훈련 에폭 수
        lr (float): 학습률
    
    Returns:
        dict: 모델링 결과
    """
    try:
        start_time = time.time()
        print(f"🚀 커피 생두 가격 예측 모델링 시작")
        print(f"📊 설정 - 손실 함수: {loss_fn}, Delta: {delta}, 에폭: {epochs}, 학습률: {lr}")
        
        # 1. 데이터 로드
        weather_data = load_weather_data()
        weather_data = remove_lag(weather_data)
        
        # 2. 데이터 전처리
        weather_data = leave_PRECTOTCORR_columns(weather_data)  # 기후 데이터 중에서 강수량 + 필요한 컬럼만 남김
        
        # 3. 범주형 변수 인코딩 후 변동성 관련 파생 피처 추가
        weather_data = encode_categorical_features(weather_data)  # 먼저 범주형 인코딩
        weather_data = add_volatility_features(weather_data)      # 그 다음 변동성 특성 추가
        
        # 4. train/test split
        train_data, test_data = split_data(weather_data, train_ratio=0.8)  # 80% train, 20% test
        
        # 5. 데이터 형태 디버깅 (None 체크가 있는 함수 사용)
        debug_data_shape(train_data, test_data)  # 로더는 아직 없으므로 인자 제거
        
        # 6. 데이터 준비
        train_loader, test_loader, scaler, test_dates, seq_length, pred_length = prepare_data_for_model(train_data, test_data)
        
        # 이제 로더가 준비되었으므로 더 자세한 디버깅 정보 출력
        debug_data_shape(train_data, test_data, train_loader, test_loader)
        
        # 7. 모델 설정 - Softmax Attention 모델 사용
        input_dim = train_data.shape[1]  # 특성 개수
        model, device = setup_model(input_dim, use_entmax=False)
        
        # 8. 모델 훈련 - 수정된 파라미터 사용
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            test_loader, 
            epochs=epochs,          # 파라미터로 전달받은 에폭 수 사용
            lr=lr,                  # 파라미터로 전달받은 학습률 사용
            device=device, 
            loss_fn_type=loss_fn,   # 파라미터로 전달받은 손실 함수 사용
            delta=delta             # 파라미터로 전달받은 delta 값 사용
        )
        
        # 9. 모델 평가
        if online:
            print("🔄 온라인 업데이트 방식으로 예측 수행 중...")

            test_data_array = test_data.values if hasattr(test_data, 'values') else test_data

            predictions, actuals = online_update_prediction(
                model=model,
                test_data=scaler.transform(test_data_array),
                scaler=scaler,
                seq_length=seq_length,
                pred_length=pred_length,
                device=device,
                lr=lr,
                loss_fn=loss_fn
            )

            attention_weights = None  # 온라인 방식에서는 attention 저장하지 않음
            mae = np.mean(np.abs(predictions.flatten() - actuals.flatten()))
            rmse = np.sqrt(np.mean((predictions.flatten() - actuals.flatten())**2))

        else:
            predictions, actuals, attention_weights, mae, rmse = predict_and_evaluate(
                model, test_loader, scaler, device
            )
        
        # 10. 결과 저장 - 폴더 생성 및 결과 저장
        # 저장 폴더 이름 생성: loss 함수 및 에폭 정보 포함
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = f"coffee_price_model_{loss_fn}_epochs{epochs}_{timestamp}"
        
        # 11. 모델 및 결과 저장
        result_dir = save_model_results(
            model, 
            train_losses, 
            val_losses, 
            predictions, 
            actuals, 
            test_dates=test_dates,
            folder_name=folder_name
        )
        
        # 12. 성능 요약 시각화 - 동일한 폴더에 저장
        visualization_summary(
            predictions, 
            actuals, 
            train_losses, 
            val_losses, 
            mae, 
            rmse, 
            test_dates=test_dates,
            folder_name=folder_name
        )
        
        # 13. 슬라이딩 윈도우 예측 (선택적)
        # 시간이 오래 걸릴 수 있으므로 필요한 경우에만 실행
        run_sliding = True
        if run_sliding:
            try:
                test_data_array = test_data.values if hasattr(test_data, 'values') else test_data
                sliding_predictions = run_sliding_window_prediction(
                    model, 
                    scaler.transform(test_data_array), 
                    scaler, 
                    seq_length, 
                    pred_length, 
                    device
                )
                
                # 슬라이딩 윈도우 예측 결과 시각화
                plot_sliding_window_predictions(
                    sliding_predictions, 
                    max_samples=5, 
                    save_path=os.path.join(result_dir, 'sliding_window_predictions.png')
                )
            except Exception as e:
                print(f"⚠️ 슬라이딩 윈도우 예측 중 오류 발생: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"🏁 모델링 완료 - 소요 시간: {elapsed_time:.2f}초")
        print(f"📂 결과 저장 위치: {result_dir}")
        
        # 하이퍼파라미터 정보 저장
        hyperparams = {
            'loss_function': loss_fn,
            'delta': delta if loss_fn == 'huber' else 'N/A',
            'epochs': epochs,
            'learning_rate': lr,
            'mae': mae,
            'rmse': rmse
        }
        
        # 하이퍼파라미터 파일 저장
        with open(os.path.join(result_dir, 'hyperparameters.txt'), 'w') as f:
            for param, value in hyperparams.items():
                f.write(f"{param}: {value}\n")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'actuals': actuals,
            'attention_weights': attention_weights,
            'mae': mae,
            'rmse': rmse,
            'scaler': scaler,
            'hyperparams': hyperparams
        }
    
    except Exception as e:
        import traceback
        print(f"❌ 프로세스 실행 중 오류 발생: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 모델링 실행 (명령줄 인자 사용)
    results = main(
        loss_fn=args.loss_fn,
        delta=args.delta,
        epochs=args.epochs,
        lr=args.lr,
        online=args.online
    )