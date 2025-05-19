import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 패키지 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def ensure_array(data):
    """단일 값이나 배열을 적절한 형태로 변환"""
    if isinstance(data, (np.float32, np.float64, float, int)):
        # 단일 값인 경우 1차원 배열로 변환
        return np.array([data])
    elif isinstance(data, np.ndarray):
        # 이미 배열인 경우 그대로 반환
        return data
    elif isinstance(data, list):
        if not data:  # 빈 리스트
            return np.array([])
        if isinstance(data[0], (np.ndarray, list)):
            # 2차원 이상 리스트
            return np.array(data)
        else:
            # 1차원 리스트
            return np.array(data)
    else:
        # 다른 타입의 경우 문자열로 변환하여 경고
        print(f"⚠️ 예상치 못한 데이터 타입: {type(data)}. 빈 배열로 대체합니다.")
        return np.array([])

def plot_training_loss(train_losses, val_losses, save_path=None):
    """훈련 손실 그래프 시각화"""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ 학습 손실 그래프 저장 완료: {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"⚠️ 학습 손실 시각화 중 오류 발생: {e}")

def plot_prediction_results(predictions, actuals, test_dates=None, save_path=None, title='커피 생두 가격 예측 결과'):
    """예측 결과 시각화"""
    try:
        # 데이터 형태 확인 및 변환
        predictions_arr = ensure_array(predictions)
        actuals_arr = ensure_array(actuals)
        
        if len(predictions_arr) == 0 or len(actuals_arr) == 0:
            print("⚠️ 결과가 비어 있어 시각화를 건너뜁니다.")
            return
        
        # 배열 형태 확인 및 첫 번째 샘플 추출
        if predictions_arr.ndim > 1:
            pred_data = predictions_arr[0]
            actual_data = actuals_arr[0]
        else:
            pred_data = predictions_arr
            actual_data = actuals_arr
        
        # 첫 번째 예측 시각화
        plt.figure(figsize=(12, 6))
        
        # 날짜 정보가 있는 경우
        if test_dates is not None and len(test_dates) >= len(pred_data):
            plt.plot(test_dates[:len(actual_data)], actual_data, label='실제 가격', color='blue')
            plt.plot(test_dates[:len(pred_data)], pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('날짜')
        else:
            # 날짜 정보가 없는 경우 인덱스 사용
            x_range = np.arange(len(pred_data))
            plt.plot(x_range, actual_data, label='실제 가격', color='blue')
            plt.plot(x_range, pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('예측 일수')
        
        plt.title(title)
        plt.ylabel('가격')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ 예측 결과 그래프 저장 완료: {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"⚠️ 예측 결과 시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())

def plot_sliding_window_predictions(sliding_predictions, max_samples=5, save_path=None):
    """슬라이딩 윈도우 예측 결과 시각화"""
    try:
        if not sliding_predictions:
            print("⚠️ 슬라이딩 윈도우 예측 결과가 없습니다.")
            return
        
        # 최대 표시할 샘플 수 결정
        n_samples = min(max_samples, len(sliding_predictions))
        
        # 결과 시각화
        plt.figure(figsize=(15, 3 * n_samples))
        
        for i, pred_data in enumerate(sliding_predictions[:n_samples]):
            plt.subplot(n_samples, 1, i+1)
            plt.plot(pred_data['actual'], label='실제 가격', color='blue')
            plt.plot(pred_data['prediction'], label='예측 가격', color='red', linestyle='--')
            plt.title(f'슬라이딩 윈도우 예측 #{i+1}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ 슬라이딩 윈도우 예측 그래프 저장 완료: {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"⚠️ 슬라이딩 윈도우 예측 시각화 중 오류 발생: {e}")

def save_model_results(model, train_losses, val_losses, predictions, actuals, test_dates=None, folder_name=None):
    """모델 및 결과 저장 (개선된 버전)"""
    print(f"⏳ 모델 및 결과 저장 중...")
    
    try:
        # 저장 폴더 생성
        if folder_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"model_results_{timestamp}"
        
        # 기본 저장 경로 설정
        result_dir = os.path.join(app_dir, 'data', 'output', 'results', folder_name)
        
        # 폴더가 존재하지 않으면 생성
        os.makedirs(result_dir, exist_ok=True)
        
        # 모델 저장
        model_path = os.path.join(result_dir, 'coffee_price_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장 완료: {model_path}")
        
        # 손실 그래프 저장
        loss_path = os.path.join(result_dir, 'training_loss.png')
        plot_training_loss(train_losses, val_losses, save_path=loss_path)
        
        # 예측 결과 시각화 및 저장
        predictions = ensure_array(predictions)
        actuals = ensure_array(actuals)
        
        # 첫 번째 샘플 시각화
        sample_path = os.path.join(result_dir, 'prediction_sample.png')
        plot_prediction_results(
            predictions, 
            actuals, 
            test_dates=test_dates,
            save_path=sample_path, 
            title='커피 생두 가격 예측 결과 (첫 번째 샘플)'
        )
        
        # 성능 메트릭 저장
        try:
            pred_arr = predictions[0] if predictions.ndim > 1 else predictions
            actual_arr = actuals[0] if actuals.ndim > 1 else actuals
            
            mae = np.mean(np.abs(pred_arr - actual_arr))
            rmse = np.sqrt(np.mean((pred_arr - actual_arr)**2))
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse
            }
            
            # 메트릭 파일 저장
            with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                    
            print(f"✅ 성능 메트릭 저장 완료: MAE={mae:.4f}, RMSE={rmse:.4f}")
        except Exception as e:
            print(f"⚠️ 성능 메트릭 저장 중 오류 발생: {e}")
        
        print(f"✅ 모델 및 결과 저장 완료: {result_dir}")
        return result_dir
    
    except Exception as e:
        print(f"❌ 모델 결과 저장 중 예외 발생: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 실패해도 기본 폴더 경로 반환
        default_dir = os.path.join(app_dir, 'data', 'output', 'results', 'fallback_results')
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

def visualization_summary(predictions, actuals, train_losses, val_losses, mae, rmse, test_dates=None, folder_name=None):
    """모델 성능 요약 시각화 - 학습 및 예측 결과를 한 페이지에 표시"""
    try:
        # 저장 폴더 생성
        if folder_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"model_results_{timestamp}"
        
        # 기본 저장 경로 설정
        result_dir = os.path.join(app_dir, 'data', 'output', 'results', folder_name)
        
        # 폴더가 존재하지 않으면 생성
        os.makedirs(result_dir, exist_ok=True)
        
        # 저장 경로 설정
        save_path = os.path.join(result_dir, 'model_performance_summary.png')
        
        # 데이터 준비
        predictions_arr = ensure_array(predictions)
        actuals_arr = ensure_array(actuals)
        
        if len(predictions_arr) == 0 or len(actuals_arr) == 0:
            print("⚠️ 결과가 비어 있어 시각화를 건너뜁니다.")
            return result_dir
        
        # 첫 번째 샘플 추출
        if predictions_arr.ndim > 1:
            pred_data = predictions_arr[0]
            actual_data = actuals_arr[0]
        else:
            pred_data = predictions_arr
            actual_data = actuals_arr
        
        # 2x1 그리드 생성
        plt.figure(figsize=(15, 10))
        
        # 1. 학습 손실 그래프
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('훈련 및 검증 손실')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 2. 예측 결과 그래프
        plt.subplot(2, 1, 2)
        
        # 날짜 정보가 있는 경우
        if test_dates is not None and len(test_dates) >= len(pred_data):
            plt.plot(test_dates[:len(actual_data)], actual_data, label='실제 가격', color='blue')
            plt.plot(test_dates[:len(pred_data)], pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('날짜')
        else:
            # 날짜 정보가 없는 경우 인덱스 사용
            x_range = np.arange(len(pred_data))
            plt.plot(x_range, actual_data, label='실제 가격', color='blue')
            plt.plot(x_range, pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('예측 일수')
        
        plt.title(f'예측 결과 (MAE: {mae:.4f}, RMSE: {rmse:.4f})')
        plt.ylabel('가격')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"✅ 모델 성능 요약 시각화 완료: {save_path}")
        return result_dir
    
    except Exception as e:
        print(f"⚠️ 성능 요약 시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 실패해도 기본 폴더 경로 반환
        default_dir = os.path.join(app_dir, 'data', 'output', 'results', 'fallback_results')
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

def run_sliding_window_visualization(model, test_data, scaler, seq_length, pred_length, device, folder_name=None):
    """슬라이딩 윈도우 방식으로 예측 및 시각화 진행 (model.py에서 이동)"""
    from .model import run_sliding_window_prediction
    
    print(f"⏳ 슬라이딩 윈도우 예측 진행 중...")
    
    try:
        # 저장 폴더 생성
        if folder_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"model_results_{timestamp}"
        
        # 기본 저장 경로 설정
        result_dir = os.path.join(app_dir, 'data', 'output', 'results', folder_name)
        
        # 폴더가 존재하지 않으면 생성
        os.makedirs(result_dir, exist_ok=True)
        
        # 저장 경로 설정
        save_path = os.path.join(result_dir, 'sliding_window_predictions.png')
        
        # 슬라이딩 윈도우 예측 (model.py에서 데이터 생성)
        sliding_predictions = run_sliding_window_prediction(
            model, test_data, scaler, seq_length, pred_length, device
        )
        
        # 시각화 (utils.py에서 시각화)
        plot_sliding_window_predictions(
            sliding_predictions, 
            max_samples=5, 
            save_path=save_path
        )
        
        print(f"✅ 슬라이딩 윈도우 예측 완료: {save_path}")
        return sliding_predictions, result_dir
    
    except Exception as e:
        print(f"⚠️ 슬라이딩 윈도우 시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 실패해도 기본 폴더 경로 반환
        default_dir = os.path.join(app_dir, 'data', 'output', 'results', 'fallback_results')
        os.makedirs(default_dir, exist_ok=True)
        return None, default_dir