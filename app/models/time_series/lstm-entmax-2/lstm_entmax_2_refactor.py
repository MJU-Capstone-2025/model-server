import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from entmax import Entmax15

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 유틸리티 함수들
# ============================================================================

def get_device():
    """
    사용 가능한 GPU 또는 CPU 장치를 반환합니다.
    
    Returns:
        str: 'cuda' (GPU 사용 가능시) 또는 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 데이터 로딩 및 저장 함수들
# ============================================================================

def load_eco_data(data_path=None):
    """
    커피, 원유, 환율 등 경제 데이터를 로드합니다.
    
    Args:
        data_path (str, optional): 데이터 파일 경로. None이면 기본 경로 사용.
        
    Returns:
        pd.DataFrame: 경제 데이터가 포함된 데이터프레임
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', 'coffee_oil_exchange_daily.csv')
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', 'coffee_oil_exchange_daily.csv')
    
    df = pd.read_csv(data_path)
    return df

def load_weather_data(data_path=None):
    """
    날씨 및 기후 관련 데이터를 로드합니다.
    
    Args:
        data_path (str, optional): 데이터 파일 경로. None이면 기본 경로 사용.
        
    Returns:
        pd.DataFrame: 날씨 데이터가 포함된 데이터프레임
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '비수확기평균커피가격통합데이터.csv')
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '비수확기평균커피가격통합데이터.csv')
    
    df = pd.read_csv(data_path)
    return df

def save_result(result_df, data_path=None):
    """
    예측 결과를 CSV 파일로 저장합니다.
    
    Args:
        result_df (pd.DataFrame): 저장할 예측 결과 데이터프레임
        data_path (str, optional): 저장할 파일 경로. None이면 기본 경로 사용.
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'output', 'prediction_result.csv')
    
    result_df.to_csv(data_path, index=False)
    print(f"예측 결과가 {data_path}에 저장되었습니다.")

# ============================================================================
# 데이터 전처리 함수들
# ============================================================================

def preprocess_data():
    """
    원시 데이터를 로드하고 모델 학습에 필요한 피처들을 생성합니다.
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임 (Date가 인덱스로 설정됨)
    """
    # 경제 데이터 로드 및 기본 전처리
    df = load_eco_data()
    df = df.ffill()  # forward fill (이전 값으로 채우기)
    df = df.bfill()  # backward fill (다음 값으로 채우기)
    df = df[['Date', 'Coffee_Price', 'Crude_Oil_Price', 'USD_BRL']]
    
    # 수익률 및 변동성 피처 생성
    df['Coffee_Price_Return'] = df['Coffee_Price'].pct_change()
    df['abs_return'] = df['Coffee_Price_Return'].abs()
    df['volatility_5d'] = df['Coffee_Price_Return'].rolling(window=5).std()
    df['volatility_10d'] = df['Coffee_Price_Return'].rolling(window=10).std()
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']
    
    # 모멘텀 피처 생성
    df['momentum_1d'] = df['Coffee_Price'].diff(1)
    df['momentum_3d'] = df['Coffee_Price'].diff(3)
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)
    
    # 볼린저 밴드 및 Z-스코어 피처
    rolling_mean = df['Coffee_Price'].rolling(window=20).mean()
    rolling_std = df['Coffee_Price'].rolling(window=20).std()
    df['bollinger_width'] = (2 * rolling_std) / rolling_mean
    df['return_zscore'] = (df['Coffee_Price_Return'] - df['Coffee_Price_Return'].rolling(20).mean()) / \
                          (df['Coffee_Price_Return'].rolling(20).std() + 1e-6)
    
    # 날씨 데이터 병합
    weather_data = load_weather_data()
    columns_to_drop = ['Coffee_Price', 'Coffee_Price_Return', 'Crude_Oil_Price', 
                      'USD_KRW', 'USD_BRL', 'USD_COP']
    weather_data.drop(columns=[c for c in columns_to_drop if c in weather_data.columns], inplace=True)
    df = pd.merge(df, weather_data, on='Date', how='inner')
    
    # 최종 정리
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df

def split_and_scale(df, target_col, static_feat_count, window, horizon, step):
    """
    데이터를 훈련/테스트로 분할하고 정규화를 수행합니다.
    
    Args:
        df (pd.DataFrame): 전처리된 데이터프레임
        target_col (str): 예측 대상 컬럼명
        static_feat_count (int): 정적 피처의 개수
        window (int): 입력 시퀀스 길이
        horizon (int): 예측 구간 길이
        step (int): 슬라이딩 윈도우 스텝 크기
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx)
    """
    n = len(df)
    train_size = int(n * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 정규화 (타겟 변수 제외)
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), 
        columns=train_df.columns, 
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), 
        columns=test_df.columns, 
        index=test_df.index
    )
    
    # 타겟 변수는 원본 값 유지
    train_scaled[target_col] = train_df[target_col]
    test_scaled[target_col] = test_df[target_col]
    
    # 피처와 타겟 분리
    y_train = train_scaled[target_col].values
    y_test = test_scaled[target_col].values
    X_train = train_scaled.drop(columns=[target_col]).values
    X_test = test_df.drop(columns=[target_col]).values
    
    # 정적 피처 인덱스 설정
    static_feat_idx = list(range(X_train.shape[1] - static_feat_count, X_train.shape[1]))
    
    return X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx

# ============================================================================
# 데이터셋 클래스
# ============================================================================

class MultiStepTimeSeriesDataset(torch.utils.data.Dataset):
    """
    다중 스텝 시계열 예측을 위한 PyTorch 데이터셋 클래스.
    시퀀스 피처와 정적 피처를 분리하여 처리합니다.
    """
    
    def __init__(self, X, y, window_size, horizon, step, static_feat_idx):
        """
        Args:
            X (np.ndarray): 입력 피처 배열
            y (np.ndarray): 타겟 배열
            window_size (int): 입력 시퀀스 길이
            horizon (int): 예측 구간 길이
            step (int): 슬라이딩 윈도우 스텝 크기
            static_feat_idx (list): 정적 피처의 인덱스 리스트
        """
        self.seq_feat_idx = [i for i in range(X.shape[1]) if i not in static_feat_idx]
        
        # 시퀀스 데이터 생성
        self.data = [
            X[i:i+window_size, self.seq_feat_idx] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # 정적 피처 데이터 생성
        self.static_feats = [
            X[i, static_feat_idx] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # 라벨 데이터 생성
        self.labels = [
            y[i+window_size:i+window_size+horizon] 
            for i in range(0, len(X) - window_size - horizon + 1, step)
        ]
        
        # NumPy 배열로 변환
        self.data = np.array(self.data)
        self.static_feats = np.array(self.static_feats)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.static_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# ============================================================================
# 모델 관련 클래스들
# ============================================================================

def compute_attention_entropy(attn_weights):
    """
    어텐션 가중치의 엔트로피를 계산합니다.
    
    Args:
        attn_weights (torch.Tensor): 어텐션 가중치 텐서
        
    Returns:
        float: 평균 엔트로피 값
    """
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=1)
    return entropy.mean().item()

class EntmaxAttention(nn.Module):
    """
    Entmax 기반 어텐션 메커니즘을 구현한 클래스.
    """
    
    def __init__(self, hidden_size, attn_dim=64):
        """
        Args:
            hidden_size (int): LSTM 은닉 상태 크기
            attn_dim (int): 어텐션 차원 크기
        """
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output (torch.Tensor): LSTM 출력 텐서 [batch_size, seq_len, hidden_size]
            
        Returns:
            tuple: (context_vector, attention_weights)
        """
        scores = self.score_layer(lstm_output).squeeze(-1)
        weights = self.entmax(scores)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionLSTMModel(nn.Module):
    """
    Entmax 어텐션과 정적 피처를 결합한 LSTM 모델.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, target_size=14, 
                 dropout=0.1, static_feat_dim=9):
        """
        Args:
            input_size (int): 입력 피처 차원
            hidden_size (int): LSTM 은닉 상태 크기
            num_layers (int): LSTM 레이어 수
            target_size (int): 예측 구간 길이
            dropout (float): 드롭아웃 비율
            static_feat_dim (int): 정적 피처 차원
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # 어텐션 메커니즘
        self.attention = EntmaxAttention(hidden_size)
        
        # 게이트 메커니즘 (컨텍스트와 마지막 은닉 상태 결합)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
        # 최종 예측 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, target_size)
        )
    
    def forward(self, x_seq, x_static, hidden_states=None):
        """
        Args:
            x_seq (torch.Tensor): 시퀀스 입력 [batch_size, seq_len, input_size]
            x_static (torch.Tensor): 정적 피처 입력 [batch_size, static_feat_dim]
            hidden_states (tuple, optional): 초기 은닉 상태
            
        Returns:
            tuple: (predictions, attention_weights)
        """
        batch_size = x_seq.size(0)
        
        # 초기 은닉 상태 설정
        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            hidden_states = (h0, c0)
        
        # LSTM 순전파
        lstm_out, _ = self.lstm(x_seq, hidden_states)
        
        # 어텐션 적용
        context, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        
        # 게이트를 통한 컨텍스트와 마지막 은닉 상태 결합
        combined = torch.cat([context, last_hidden], dim=1)
        alpha = self.gate(combined)
        fused = alpha * context + (1 - alpha) * last_hidden
        
        # 정적 피처와 결합하여 최종 예측
        fused_with_static = torch.cat([fused, x_static], dim=1)
        out = self.fc(fused_with_static)
        
        return out, attn_weights

# ============================================================================
# 손실 함수들
# ============================================================================

def directional_loss(y_pred, y_true):
    """
    방향성 손실: 예측값과 실제값의 변화 방향이 일치하는지 측정합니다.
    
    Args:
        y_pred (torch.Tensor): 예측값
        y_true (torch.Tensor): 실제값
        
    Returns:
        torch.Tensor: 방향성 손실값
    """
    pred_diff = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
    true_diff = torch.sign(y_true[:, 1:] - y_true[:, :-1])
    return torch.mean((pred_diff != true_diff).float())

def variance_loss(y_pred, y_true):
    """
    분산 손실: 예측값과 실제값의 분산 차이를 측정합니다.
    
    Args:
        y_pred (torch.Tensor): 예측값
        y_true (torch.Tensor): 실제값
        
    Returns:
        torch.Tensor: 분산 손실값
    """
    return torch.abs(torch.std(y_pred) - torch.std(y_true))

# ============================================================================
# 시각화 함수들
# ============================================================================

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

# ============================================================================
# 모델 학습 및 예측 함수들
# ============================================================================

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

def evaluate_and_save(df, forecast_all, predictions, price_col, future_dates, price_future):
    """
    예측 결과를 평가하고 CSV 파일로 저장합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        forecast_all (pd.Series): 테스트 구간 예측값
        predictions (list): 개별 예측 시리즈 리스트
        price_col (str): 가격 컬럼명
        future_dates (pd.DatetimeIndex): 미래 예측 날짜
        price_future (list): 미래 예측 가격 리스트
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
    full_df = pd.concat([result_df, future_df], ignore_index=True)
    full_df.drop_duplicates(subset=["Date"], inplace=True)
    save_result(full_df)

# ============================================================================
# 메인 실행 함수
# ============================================================================

def main():
    """
    커피 가격 예측 모델의 전체 파이프라인을 실행합니다.
    
    주요 단계:
    1. 데이터 로드 및 전처리
    2. 데이터 분할 및 정규화
    3. 데이터셋 및 데이터로더 생성
    4. 모델 초기화 및 학습
    5. 테스트 구간 예측
    6. 미래 구간 예측
    7. 결과 평가 및 저장
    """
    print("=== 커피 가격 예측 모델 시작 ===")
    
    # 1. 환경 설정
    device = get_device()
    print(f"사용 장치: {device}")
    
    # 2. 데이터 전처리
    print("데이터 전처리 중...")
    df = preprocess_data()
    print(f"전처리된 데이터 크기: {df.shape}")
    
    # 3. 하이퍼파라미터 설정
    target_col = "Coffee_Price_Return"
    price_col = "Coffee_Price"
    static_feat_count = 9
    data_window = 100
    future_target = 14
    step = 1
    
    # 4. 데이터 분할 및 정규화
    print("데이터 분할 및 정규화 중...")
    X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx = split_and_scale(
        df, target_col, static_feat_count, data_window, future_target, step
    )
    
    # 5. 데이터셋 및 데이터로더 생성
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, future_target, step, static_feat_idx)
    test_dataset = MultiStepTimeSeriesDataset(X_test, y_test, data_window, future_target, step, static_feat_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. 모델 초기화
    x_seq, x_static, _ = train_dataset[0]
    input_size = x_seq.shape[1]
    static_feat_dim = x_static.shape[0]
    
    model = AttentionLSTMModel(
        input_size=input_size, 
        target_size=future_target, 
        static_feat_dim=static_feat_dim
    ).to(device)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. 학습 설정
    base_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10)
    
    num_epochs = 5
    alpha, beta = 0.2, 0.1  # 방향성 손실, 분산 손실 가중치
    
    # 8. 모델 학습
    print("모델 학습 시작...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, base_criterion, optimizer, scheduler, 
        num_epochs, alpha, beta, device
    )
    
    # 9. 학습 곡선 시각화
    plot_loss(train_losses, test_losses)
    
    # 10. 테스트 구간 예측
    print("테스트 구간 예측 중...")
    forecast_all, predictions = predict_and_inverse(
        model, test_loader, scaler, train_df, test_df, df, target_col, 
        price_col, data_window, future_target, step, static_feat_idx
    )
    
    # 11. 테스트 구간 결과 시각화
    plot_prediction(df, forecast_all, start=pd.to_datetime('2023-07-01'), end=pd.to_datetime('2025-04-01'))
    
    # 12. 미래 구간 예측
    print("미래 구간 예측 중...")
    future_price_series, future_dates, price_future = predict_future(
        model, test_df, train_df, scaler, static_feat_idx, data_window, 
        future_target, price_col, target_col
    )
    
    # 13. 전체 결과 시각화 (테스트 + 미래)
    plot_prediction(
        df, forecast_all, 
        start=pd.to_datetime('2023-07-01'), 
        end=future_price_series.index[-1], 
        future_series=future_price_series
    )
    
    # 14. 결과 평가 및 저장
    print("결과 평가 및 저장 중...")
    evaluate_and_save(df, forecast_all, predictions, price_col, future_dates, price_future)
    
    print("=== 커피 가격 예측 모델 완료 ===")

if __name__ == "__main__":
    main()
