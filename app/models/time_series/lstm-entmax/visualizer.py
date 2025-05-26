"""
시각화 함수들
"""
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_loss(train_losses, test_losses):
    """
    훈련 및 검증 손실을 시각화합니다.
    
    Args:
        train_losses (list): 에포크별 훈련 손실
        test_losses (list): 에포크별 검증 손실
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction(df, forecast_all, start=None, end=None, future_series=None):
    """
    실제값과 예측값을 시각화합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        forecast_all (pd.Series): 예측값 시리즈
        start (datetime, optional): 시작 날짜
        end (datetime, optional): 종료 날짜
        future_series (pd.Series, optional): 미래 예측값 시리즈
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df['Coffee_Price'], label='Actual Coffee Price', color='blue')
    plt.plot(forecast_all.index, forecast_all.values, label='Predicted', color='red', linestyle='dashed')
    
    if future_series is not None:
        plt.plot(future_series.index, future_series.values, 
                label='Predicted (Future)', color='orange', linestyle='dotted')
    
    if start and end:
        plt.xlim(start, end)
    
    plt.title('Coffee Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Coffee Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 