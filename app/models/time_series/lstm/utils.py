import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

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
    """슬라이딩 윈도우 예측 결과 시각화 - Date 정보 포함 버전"""
    try:
        if not sliding_predictions:
            print("⚠️ 슬라이딩 윈도우 예측 결과가 없습니다.")
            return
        
        # 최대 표시할 샘플 수 결정
        n_samples = min(max_samples, len(sliding_predictions))
        
        # 더 큰 그림 크기와 해상도 설정
        plt.figure(figsize=(15, 7 * n_samples), dpi=150)
        
        # 전체 데이터의 최소/최대값 찾기 (일관된 Y축 설정용)
        all_actual_values = []
        all_prediction_values = []
        
        for i in range(n_samples):
            all_actual_values.extend(sliding_predictions[i]['actual'])
            all_prediction_values.extend(sliding_predictions[i]['prediction'])
        
        global_actual_min = min(all_actual_values)
        global_actual_max = max(all_actual_values)
        global_pred_min = min(all_prediction_values)
        global_pred_max = max(all_prediction_values)
        
        # 2개의 subplot: 왼쪽은 실제값+예측값, 오른쪽은 예측값만
        for i, pred_data in enumerate(sliding_predictions[:n_samples]):
            # 각 윈도우마다 2개의 그래프 (1x2 그리드)
            plt.subplot(n_samples, 2, i*2+1)  # 왼쪽: 실제값+예측값
            plt.subplot(n_samples, 2, i*2+2)  # 오른쪽: 예측값만
            
            # 실제 데이터와 예측 데이터의 범위 확인
            actual_min = np.min(pred_data['actual'])
            actual_max = np.max(pred_data['actual'])
            pred_min = np.min(pred_data['prediction'])
            pred_max = np.max(pred_data['prediction'])
            
            # 디버깅용 출력
            print(f"Plot #{i} - actual 범위: {actual_min:.2f}~{actual_max:.2f}, 예측 범위: {pred_min:.2f}~{pred_max:.2f}")
            
            # 날짜 데이터 포맷맷
            dates = None
            date_labels = None
            if 'dates' in pred_data:
                dates = pred_data['dates']
                # 간략한 날짜 표시를 위해 MM-DD 형식으로 변환
                date_labels = [d.strftime('%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]
            
            # ----- 첫 번째 서브플롯: 실제값+예측값 (Y축 범위 다름) -----
            ax1 = plt.subplot(n_samples, 2, i*2+1)
            
            # X축 값 (날짜 또는 인덱스)
            x_values = np.arange(len(pred_data['actual'])) if dates is None else np.arange(len(dates))
            
            # 실제 데이터 (파란색)
            ax1.plot(x_values, pred_data['actual'], 'b-', label='실제 가격', linewidth=2.5)
            
            # 예측 데이터 (빨간색 점선)
            ax1.plot(x_values[:len(pred_data['prediction'])], pred_data['prediction'], 'r--', 
                    label='예측 가격', linewidth=2.5, marker='o', markersize=5)
            
            # 그래프 설정
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper right')
            
            title_text = f'윈도우 #{i+1} - 실제값 & 예측값\n(실제범위: {actual_min:.2f}~{actual_max:.2f}, 예측범위: {pred_min:.2f}~{pred_max:.2f})'
            if dates is not None:
                title_text += f'\n기간: {dates[0].strftime("%Y-%m-%d") if hasattr(dates[0], "strftime") else dates[0]} ~ {dates[-1].strftime("%Y-%m-%d") if hasattr(dates[-1], "strftime") else dates[-1]}'
            
            ax1.set_title(title_text, fontsize=11)
            
            # 각 윈도우마다 개별 Y축 범위 설정 (실제값과 예측값을 모두 잘 보이게)
            buffer = max(actual_max - actual_min, pred_max - pred_min) * 0.1
            ax1.set_ylim(min(actual_min, pred_min) - buffer, max(actual_max, pred_max) + buffer)
            
            # X축 날짜 설정 (날짜가 있는 경우)
            if date_labels is not None:
                # 모든 날짜를 표시하는 대신 적절한 간격으로 표시 (5개 정도)
                n_ticks = min(5, len(date_labels))
                tick_indices = np.linspace(0, len(date_labels)-1, n_ticks, dtype=int)
                
                ax1.set_xticks(tick_indices)
                ax1.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45)
                ax1.set_xlabel('날짜', fontsize=10)
            else:
                ax1.set_xlabel('예측 기간 (일)', fontsize=10)
            
            ax1.set_ylabel('가격', fontsize=10)
            
            # ----- 두 번째 서브플롯: 예측값만 (모든 그래프 Y축 범위 동일) -----
            ax2 = plt.subplot(n_samples, 2, i*2+2)
            
            # 예측 데이터만 (빨간색 굵은 선) - 날짜가 있으면 날짜 사용
            x_pred_values = np.arange(len(pred_data['prediction'])) if dates is None else x_values[:len(pred_data['prediction'])]
            
            ax2.plot(x_pred_values, pred_data['prediction'], 'r-', linewidth=3, 
                     label='예측 가격', marker='o', markersize=6)
            
            # 그래프 설정
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 모든 그래프에 동일한 Y축 범위 적용 (예측값만 표시)
            pred_buffer = (global_pred_max - global_pred_min) * 0.1
            ax2.set_ylim(global_pred_min - pred_buffer, global_pred_max + pred_buffer)
            
            # 제목에 날짜 정보 추가
            title_text2 = f'윈도우 #{i+1} - 예측값만\n(시작 인덱스: {pred_data.get("start_idx", "N/A")})'
            if dates is not None:
                end_date_idx = min(len(pred_data['prediction'])-1, len(dates)-1)
                start_date = dates[0].strftime("%Y-%m-%d") if hasattr(dates[0], "strftime") else dates[0]
                end_date = dates[end_date_idx].strftime("%Y-%m-%d") if hasattr(dates[end_date_idx], "strftime") else dates[end_date_idx]
                title_text2 += f'\n예측 기간: {start_date} ~ {end_date}'
            
            ax2.set_title(title_text2, fontsize=11)
            
            # X축 날짜 설정 (날짜가 있는 경우)
            if date_labels is not None:
                pred_date_labels = date_labels[:len(pred_data['prediction'])]
                
                # 적절한 간격으로 표시 (5개 정도)
                n_ticks = min(5, len(pred_date_labels))
                tick_indices = np.linspace(0, len(pred_date_labels)-1, n_ticks, dtype=int)
                
                ax2.set_xticks(tick_indices)
                ax2.set_xticklabels([pred_date_labels[i] for i in tick_indices], rotation=45)
                ax2.set_xlabel('날짜', fontsize=10)
            else:
                ax2.set_xlabel('예측 기간 (일)', fontsize=10)
            
            ax2.set_ylabel('예측 가격', fontsize=10)
            
            # 각 포인트에 값 표시 (처음, 중간, 마지막)
            for idx in [0, len(pred_data['prediction'])//2, -1]:
                ax2.annotate(f'{pred_data["prediction"][idx]:.1f}', 
                          xy=(x_pred_values[idx], pred_data['prediction'][idx]), 
                          xytext=(0, 10), textcoords='offset points',
                          ha='center', fontsize=9, 
                          bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            # 구분선 추가
            if i < n_samples-1:
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=3)
        
        plt.tight_layout()
        
        # 전체 제목 추가
        plt.suptitle('슬라이딩 윈도우 방식 예측 결과 (왼쪽: 실제값+예측값, 오른쪽: 예측값만)', fontsize=14, y=0.995)
        plt.subplots_adjust(top=0.98)
        
        if save_path:
            # 고해상도로 저장
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 슬라이딩 윈도우 예측 그래프 저장 완료: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️ 슬라이딩 윈도우 예측 시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())

def save_model_results(model, train_losses, val_losses, predictions, actuals, test_dates=None, folder_name=None):
    """모델 및 결과 저장"""
    print(f"⏳ 모델 및 결과 저장 중...")
    from .utils import get_result_dir
    try:
        # 저장 폴더 생성
        result_dir = get_result_dir(folder_name)
        
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
    from .utils import get_result_dir
    try:
        # 저장 폴더 생성
        result_dir = get_result_dir(folder_name)
        
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

def get_result_dir(folder_name=None):
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    if folder_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"model_results_{timestamp}"
    result_dir = os.path.join(app_dir, 'data', 'output', 'results', folder_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def plot_sliding_window_from_csv(csv_path, max_windows=5, save_path=None):
    df = pd.read_csv(csv_path)
    if 'window_id' not in df.columns:
        print("CSV에 window_id 컬럼이 없습니다.")
        return

    grouped = df.groupby('window_id')
    n_samples = min(max_windows, len(grouped))

    plt.figure(figsize=(15, 7 * n_samples), dpi=150)

    for i, (window_id, group) in enumerate(list(grouped)[:n_samples]):
        dates = group['date'].tolist()
        prediction = group['prediction'].tolist()
        actual = group['actual'].tolist()

        # 실제값+예측값
        ax1 = plt.subplot(n_samples, 2, i*2+1)
        ax1.plot(dates, actual, 'b-', label='실제 가격', linewidth=2.5)
        ax1.plot(dates, prediction, 'r--', label='예측 가격', linewidth=2.5, marker='o', markersize=5)
        ax1.set_title(f'윈도우 #{window_id} - 실제값 & 예측값')
        ax1.set_xlabel('날짜')
        ax1.set_ylabel('가격')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xticks(range(len(dates)))
        ax1.set_xticklabels(dates, rotation=45)

        # 예측값만
        ax2 = plt.subplot(n_samples, 2, i*2+2)
        ax2.plot(dates, prediction, 'r-', linewidth=3, label='예측 가격', marker='o', markersize=6)
        ax2.set_title(f'윈도우 #{window_id} - 예측값만')
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('예측 가격')
        ax2.grid(True)
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(dates, rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 슬라이딩 윈도우 예측 그래프 저장 완료: {save_path}")
    plt.close()