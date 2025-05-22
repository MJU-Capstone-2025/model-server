import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import traceback

# 패키지 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def save_predictions_to_csv(all_predictions, test_dates=None, folder_name=None):
    """
    예측 결과를 CSV 파일로 저장하는 함수
    
    Args:
        all_predictions (list): 예측 정보가 담긴 딕셔너리의 리스트
        test_dates (list, optional): 테스트 날짜 리스트
        folder_name (str, optional): 결과를 저장할 폴더 이름
    
    Returns:
        str: 저장된 CSV 파일 경로 또는 None (오류 발생 시)
    """
    try:
        import numpy as np
        result_dir = get_result_dir(folder_name)
        csv_path = os.path.join(result_dir, 'sliding_window_predictions.csv')
        # test_dates를 string으로 변환
        if test_dates is not None:
            test_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in test_dates]
        
        all_data = []
        for idx, pred_data in enumerate(all_predictions):
            window_id = idx + 1
            start_idx = pred_data['start_idx']
            end_idx = pred_data['end_idx']
            seq_length = pred_data.get('seq_length', 50)
            dates = pred_data.get('dates', None)
            
            for step in range(len(pred_data['prediction'])):
                data_row = {
                    'window_id': window_id,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'step': step + 1,
                    'prediction': pred_data['prediction'][step],
                    'actual': pred_data['actual'][step],
                }
                
                # 날짜 정보가 있는 경우 string으로 저장
                if dates is not None and step < len(dates):
                    data_row['date'] = dates[step]
                elif test_dates is not None:
                    date_idx = start_idx + seq_length + step
                    if date_idx < len(test_dates):
                        data_row['date'] = test_dates[date_idx]
                    else:
                        # test_dates 범위를 넘으면 마지막 날짜 이후로 하루씩 증가
                        from datetime import datetime, timedelta
                        last_date_str = test_dates[-1]
                        try:
                            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                        except Exception:
                            # fallback: 날짜 포맷이 다르면 그냥 string으로 둠
                            data_row['date'] = ''
                            all_data.append(data_row)
                            continue
                        extra_days = date_idx - (len(test_dates) - 1)
                        new_date = last_date + timedelta(days=extra_days)
                        data_row['date'] = new_date.strftime('%Y-%m-%d')
                else:
                    data_row['date'] = ''
                
                # 실제값: test_dates 범위 내면 그대로, 아니면 nan
                if test_dates is not None:
                    date_idx = start_idx + seq_length + step
                    if date_idx < len(test_dates):
                        data_row['actual'] = pred_data['actual'][step]
                    else:
                        data_row['actual'] = np.nan
                else:
                    data_row['actual'] = pred_data['actual'][step]
                
                all_data.append(data_row)
        
        df = pd.DataFrame(all_data)
        if 'date' in df.columns:
            cols = ['window_id', 'date', 'step', 'prediction', 'actual', 'start_idx', 'end_idx']
            df = df[[col for col in cols if col in df.columns]]
        
        os.makedirs(result_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ 예측 결과 CSV 저장 완료: {csv_path}")
        return csv_path
    
    except Exception as e:
        print(f"❌ CSV 파일 저장 중 오류 발생: {e}")
        traceback.print_exc()
        return None

def ensure_array(data):
    """단일 값이나 배열을 적절한 형태로 변환"""
    if isinstance(data, (np.float32, np.float64, float, int)):
        return np.array([data])
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        if not data:
            return np.array([])
        if isinstance(data[0], (np.ndarray, list)):
            return np.array(data)
        else:
            return np.array(data)
    else:
        print(f"⚠️ 예상치 못한 데이터 타입: {type(data)}. 빈 배열로 대체합니다.")
        return np.array([])

def plot_training_loss(train_losses, val_losses, save_path=None):
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
    """예측 결과 시각화 (전체 + 첫 2주)"""
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
        
        # 전체 구간 시각화
        plt.figure(figsize=(12, 6))
        if test_dates is not None and len(test_dates) >= len(pred_data):
            plt.plot(test_dates[:len(actual_data)], actual_data, label='실제 가격', color='blue')
            plt.plot(test_dates[:len(pred_data)], pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('날짜')
        else:
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

        # 테스트 데이터셋의 첫 2주(14일) 구간만 시각화
        if test_dates is not None and len(test_dates) >= 14:
            plt.figure(figsize=(12, 6))
            plt.plot(test_dates[:14], actual_data[:14], label='실제 가격', color='blue')
            plt.plot(test_dates[:14], pred_data[:14], label='예측 가격', color='red', linestyle='--')
            plt.xlabel('날짜')
            plt.title(title + ' (테스트 첫 2주)')
            plt.ylabel('가격')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_path:
                base, ext = os.path.splitext(save_path)
                save_path2 = base + '_first2weeks' + ext
                plt.savefig(save_path2)
                print(f"✅ 예측 결과 (테스트 첫 2주) 그래프 저장 완료: {save_path2}")
            plt.close()
    except Exception as e:
        print(f"⚠️ 예측 결과 시각화 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())

def save_model_results(model, train_losses, val_losses, predictions, actuals, test_dates=None, folder_name=None):
    print(f"⏳ 모델 및 결과 저장 중...")
    from .utils import get_result_dir
    try:
        result_dir = get_result_dir(folder_name)
        model_path = os.path.join(result_dir, 'coffee_price_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장 완료: {model_path}")
        loss_path = os.path.join(result_dir, 'training_loss.png')
        plot_training_loss(train_losses, val_losses, save_path=loss_path)
        predictions = ensure_array(predictions)
        actuals = ensure_array(actuals)
        sample_path = os.path.join(result_dir, 'prediction_sample.png')
        plot_prediction_results(
            predictions, 
            actuals, 
            test_dates=test_dates,
            save_path=sample_path, 
            title='커피 생두 가격 예측 결과 (첫 번째 샘플)'
        )
        try:
            pred_arr = predictions[0] if predictions.ndim > 1 else predictions
            actual_arr = actuals[0] if actuals.ndim > 1 else actuals
            mae = np.mean(np.abs(pred_arr - actual_arr))
            rmse = np.sqrt(np.mean((pred_arr - actual_arr)**2))
            metrics = {
                'MAE': mae,
                'RMSE': rmse
            }
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
        default_dir = os.path.join(app_dir, 'data', 'output', 'results', 'fallback_results')
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

def visualization_summary(predictions, actuals, train_losses, val_losses, mae, rmse, test_dates=None, folder_name=None):
    from .utils import get_result_dir
    try:
        result_dir = get_result_dir(folder_name)
        save_path = os.path.join(result_dir, 'model_performance_summary.png')
        predictions_arr = ensure_array(predictions)
        actuals_arr = ensure_array(actuals)
        if len(predictions_arr) == 0 or len(actuals_arr) == 0:
            print("⚠️ 결과가 비어 있어 시각화를 건너뜁니다.")
            return result_dir
        if predictions_arr.ndim > 1:
            pred_data = predictions_arr[0]
            actual_data = actuals_arr[0]
        else:
            pred_data = predictions_arr
            actual_data = actuals_arr
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('훈련 및 검증 손실')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        if test_dates is not None and len(test_dates) >= len(pred_data):
            plt.plot(test_dates[:len(actual_data)], actual_data, label='실제 가격', color='blue')
            plt.plot(test_dates[:len(pred_data)], pred_data, label='예측 가격', color='red', linestyle='--')
            plt.xlabel('날짜')
        else:
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
        default_dir = os.path.join(app_dir, 'data', 'output', 'results', 'fallback_results')
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

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
    
    # 윈도우 ID 85~89만 시각화
    selected_windows = [1, 2, 3, 4, 5, 85, 86, 87, 88, 89]
    filtered_df = df[df['window_id'].isin(selected_windows)]
    grouped = filtered_df.groupby('window_id')
    n_samples = len(selected_windows)
    
    plt.figure(figsize=(15, 7 * n_samples), dpi=150)
    
    for i, (window_id, group) in enumerate(grouped):
        dates = group['date'].tolist()
        prediction = group['prediction'].tolist()
        actual = group['actual'].tolist()
        
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