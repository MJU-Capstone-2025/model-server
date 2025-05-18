"""
Attention LSTM 모델 통합 버전 - 시계열 커피 가격 예측
이 스크립트는 Google Colab에서 실행할 수 있도록 모든 기능을 통합한 버전입니다.
다양한 손실 함수, 스케일링 옵션, 그리고 향상된 예측 기능을 제공합니다.

주요 기능:
1. 여러 손실 함수 지원 (MSE, Huber, Directional, Combined)
2. 가격 스케일링 옵션 (활성화/비활성화)
3. Entmax 기반 어텐션 메커니즘
4. 다양한 시각화 및 평가 도구

원본 구조: 
- data_preprocessing.py - 데이터 전처리
- dataset.py - 데이터셋 클래스
- model.py - 모델 아키텍처
- training.py - 학습 및 예측
- utils.py - 유틸리티 기능
- main.py - 실행 로직
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# entmax 설치 확인
try:
    from entmax import Entmax15
except ImportError:
    print("⚠️ entmax 패키지를 찾을 수 없습니다. 설치를 시도합니다...")
    try:
        # import sys
        # pip install entmax
        # from entmax import Entmax15
        print("✅ entmax 패키지 설치 완료")
    except:
        print("❌ entmax 설치 실패. 대체 구현을 사용합니다.")
        # Entmax15의 대체 구현으로 softmax를 사용
        class Entmax15(nn.Module):
            def __init__(self, dim=None):
                super().__init__()
                self.dim = dim
            
            def forward(self, x):
                return nn.functional.softmax(x, dim=self.dim)

# Google Drive 마운트 함수 (필요시 호출)
def mount_drive():
    """Google Drive를 마운트합니다 (Colab에서 실행 시)"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive 마운트 완료")
        return True
    except:
        print("Google Drive 마운트 실패 또는 필요 없음")
        return False

#--------------------------- 1. 데이터 전처리 모듈 ---------------------------#

def add_volatility_features(df):
    """
    주가 변동성 관련 피처를 추가합니다.
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: 변동성 피처가 추가된 데이터프레임
    """
    # 절대 수익률
    df['abs_return'] = df['Coffee_Price_Return'].abs()

    # 5일, 10일 변동성 (rolling std)
    df['volatility_5d'] = df['Coffee_Price_Return'].rolling(window=5).std()
    df['volatility_10d'] = df['Coffee_Price_Return'].rolling(window=10).std()

    # 5일 평균 수익률
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)

    # Bollinger Band Width (상대 변동성)
    rolling_mean = df['Coffee_Price'].rolling(window=20).mean()
    rolling_std = df['Coffee_Price'].rolling(window=20).std()
    df['bollinger_width'] = (2 * rolling_std) / rolling_mean

    # Return Z-score (비정상 변동 탐지)
    df['return_zscore'] = (df['Coffee_Price_Return'] - df['Coffee_Price_Return'].rolling(20).mean()) / \
                        (df['Coffee_Price_Return'].rolling(20).std() + 1e-6)
    
    # 추가된 변동성 특성 (버전 2)
    df['momentum_1d'] = df['Coffee_Price'].diff(1)  # 1일 가격 변화량
    df['momentum_3d'] = df['Coffee_Price'].diff(3)  # 3일 가격 변화량
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']  # 단기/중기 변동성 비율
    
    return df


def load_and_prepare_data(macro_data_path, climate_data_path):
    """
    데이터를 로드하고 전처리합니다.
    
    Args:
        macro_data_path (str): 거시경제 데이터 파일 경로
        climate_data_path (str): 기후 데이터 파일 경로
        
    Returns:
        pd.DataFrame: 전처리된 통합 데이터프레임
    """
    # 거시경제 데이터 로딩
    df = pd.read_csv(macro_data_path)
    
    # 변동성 피처 추가
    df = add_volatility_features(df)
    
    # 기후 데이터 로딩 및 병합
    we = pd.read_csv(climate_data_path)
    we.drop(columns=['Coffee_Price'], inplace=True, errors='ignore')
    df = pd.merge(df, we, on='Date', how='left') # -> left join으로 2023년 데이터까지만 사용
    
    # 결측치 제거 및 날짜 인덱스 설정
    df = df.dropna() # -> 결측치를 제거한다
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df


def train_test_split(df, train_size=0.8):
    """
    데이터를 학습 및 테스트 세트로 분할합니다.
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        train_size (float): 학습 데이터 비율 (0-1 사이)
        
    Returns:
        tuple: (train_df, test_df) 형태의 튜플
    """
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def scale_data(train_df, test_df=None, preserve_return=True, preserve_price=False):
    """
    학습 데이터를 스케일링합니다. Coffee_Price_Return 특성은 원본값 보존이 가능하며
    Coffee_Price 특성도 원본값으로 보존할 수 있습니다.
    
    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        test_df (pd.DataFrame, optional): 테스트 데이터프레임
        preserve_return (bool): Coffee_Price_Return 특성을 원본값으로 보존할지 여부
        preserve_price (bool): Coffee_Price 특성을 원본값으로 보존할지 여부
        
    Returns:
        tuple: 스케일링된 데이터와 스케일러를 포함한 튜플
               test_df가 제공된 경우: (scaled_train_df, scaled_test_df, scaler)
               아닌 경우: (scaled_train_df, scaler)
    """
    scaler = MinMaxScaler()
    
    # 원본 데이터 보존
    if preserve_return and 'Coffee_Price_Return' in train_df.columns:
        return_train = train_df["Coffee_Price_Return"].copy()
    if preserve_price and 'Coffee_Price' in train_df.columns:
        price_train = train_df["Coffee_Price"].copy()
        
    # 훈련 데이터 스케일링
    scaled_train_df = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )
    
    # 원본값 복원
    if preserve_return and 'Coffee_Price_Return' in train_df.columns:
        scaled_train_df["Coffee_Price_Return"] = return_train
    if preserve_price and 'Coffee_Price' in train_df.columns:
        scaled_train_df["Coffee_Price"] = price_train
    
    if test_df is not None:
        # 테스트 데이터도 있는 경우
        if preserve_return and 'Coffee_Price_Return' in test_df.columns:
            return_test = test_df["Coffee_Price_Return"].copy()
        if preserve_price and 'Coffee_Price' in test_df.columns:
            price_test = test_df["Coffee_Price"].copy()
            
        scaled_test_df = pd.DataFrame(
            scaler.transform(test_df),
            columns=test_df.columns,
            index=test_df.index
        )
        
        # 테스트 원본값 복원
        if preserve_return and 'Coffee_Price_Return' in test_df.columns:
            scaled_test_df["Coffee_Price_Return"] = return_test
        if preserve_price and 'Coffee_Price' in test_df.columns:
            scaled_test_df["Coffee_Price"] = price_test
            
        return scaled_train_df, scaled_test_df, scaler
    
    return scaled_train_df, scaler


def scale_data_except_price(train_df, test_df=None):
    """
    가격(Coffee_Price)을 제외한 모든 특성만 스케일링합니다.
    가격은 원본 그대로 유지하여 모델이 실제 가격 스케일에서 학습하도록 합니다.
    
    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        test_df (pd.DataFrame, optional): 테스트 데이터프레임
        
    Returns:
        tuple: 스케일링된 데이터와 스케일러를 포함한 튜플
               test_df가 제공된 경우: (scaled_train_df, scaled_test_df, scaler)
               아닌 경우: (scaled_train_df, scaler)
    """
    # 가격 컬럼만 제외한 스케일링
    return scale_data(train_df, test_df, preserve_return=True, preserve_price=True)

#--------------------------- 2. 데이터셋 모듈 ---------------------------#

class MultiStepTimeSeriesDataset(Dataset):
    """
    다중 스텝 시계열 예측을 위한 PyTorch 데이터셋 클래스
    
    이 클래스는 시계열 데이터를 슬라이딩 윈도우 방식으로 처리하여
    입력 시퀀스(X)와 타겟 시퀀스(y)를 생성합니다.
    """
    
    def __init__(self, dataset, target, data_window, target_size, step, single_step=False):
        """
        데이터셋 초기화
        
        Args:
            dataset (np.ndarray): 입력 데이터 배열
            target (np.ndarray): 타겟 데이터 배열
            data_window (int): 입력 윈도우 크기 (과거 몇 개의 데이터를 볼 것인지)
            target_size (int): 예측할 미래 기간
            step (int): 샘플링 간격 (입력 윈도우 내 데이터 간격)
            single_step (bool): 단일 스텝 예측 여부
        """
        self.data, self.labels = [], []

        start_index = data_window
        end_index = len(dataset) - target_size  # 미래 예측을 고려해 끝점 조정

        for i in range(start_index, end_index):
            indices = range(i - data_window, i, step)  # X 데이터 생성 (샘플링 적용)
            self.data.append(dataset[indices])

            if single_step:
                self.labels.append(target[i + target_size])  # 단일 값 예측
            else:
                self.labels.append(target[i:i + target_size])  # 다중 스텝 예측

        # 리스트를 PyTorch Tensor로 변환
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        """데이터셋 길이를 반환합니다"""
        return len(self.data)

    def __getitem__(self, idx):
        """특정 인덱스의 샘플을 반환합니다"""
        return self.data[idx], self.labels[idx]

#--------------------------- 3. 모델 아키텍처 모듈 ---------------------------#

class EntmaxAttention(nn.Module):
    """
    Entmax15 기반의 어텐션 메커니즘
    
    기존 Softmax 어텐션보다 더 희소한(sparse) 어텐션 가중치를 산출하여
    더 선택적인 특성을 학습할 수 있도록 합니다.
    """
    
    def __init__(self, hidden_size, attn_dim=64):
        """
        EntmaxAttention을 초기화합니다.
        
        Args:
            hidden_size (int): LSTM의 은닉 상태 크기
            attn_dim (int): 어텐션 내부 표현 차원
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
        순방향 계산을 수행합니다.
        
        Args:
            lstm_output (torch.Tensor): LSTM 출력, 형태 (batch_size, seq_len, hidden_size)
            
        Returns:
            tuple: (context, weights) 형태의 튜플
                - context: 어텐션 컨텍스트 벡터, 형태 (batch_size, hidden_size)
                - weights: 어텐션 가중치, 형태 (batch_size, seq_len)
        """
        # lstm_output: (B, T, H)
        scores = self.score_layer(lstm_output).squeeze(-1)  # (B, T)
        weights = self.entmax(scores)  # sparse attention weights
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights


class AttentionLSTMModel(nn.Module):
    """
    어텐션 메커니즘이 통합된 LSTM 모델
    
    특징:
    1. 다층(multi-layer) LSTM 
    2. EntmaxAttention을 통한 중요 시점 강조
    3. 게이팅 메커니즘을 통한 컨텍스트와 최종 은닉 상태 결합
    4. 비선형 예측 헤드
    """
    
    def __init__(self, input_size, hidden_size=100, num_layers=2, target_size=14, dropout=0.2):
        """
        AttentionLSTMModel을 초기화합니다.
        
        Args:
            input_size (int): 입력 특성 차원
            hidden_size (int): LSTM 은닉 상태 차원
            num_layers (int): LSTM 층 수
            target_size (int): 예측할 미래 시점 수
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_size = target_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attention = EntmaxAttention(hidden_size)

        # Gating layer to mix context and last_hidden
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        # Nonlinear prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, target_size)
        )

    def forward(self, x, hidden_states=None):
        """
        순방향 계산을 수행합니다.
        
        Args:
            x (torch.Tensor): 입력 데이터, 형태 (batch_size, seq_len, input_size)
            hidden_states (tuple, optional): 초기 은닉 상태 튜플 (h0, c0)
            
        Returns:
            tuple: (predictions, attn_weights) 형태의 튜플
                - predictions: 모델의 예측값, 형태 (batch_size, target_size)
                - attn_weights: 어텐션 가중치, 형태 (batch_size, seq_len)
        """
        batch_size = x.size(0)

        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden_states = (h0, c0)

        lstm_out, _ = self.lstm(x, hidden_states)  # (B, T, H)

        context, attn_weights = self.attention(lstm_out)  # (B, H)
        last_hidden = lstm_out[:, -1, :]  # (B, H)

        # Gated fusion of context and last hidden
        combined = torch.cat([context, last_hidden], dim=1)  # (B, 2H)
        alpha = self.gate(combined)  # (B, 1)
        fused = alpha * context + (1 - alpha) * last_hidden  # (B, H)

        out = self.fc(fused)  # (B, target_size)

        return out, attn_weights

#--------------------------- 4. 학습 및 예측 모듈 ---------------------------#

class HuberLoss(nn.Module):
    """
    Huber Loss 함수 (MSE와 MAE의 장점을 결합한 강건한 손실 함수)
    
    Args:
        delta (float): Huber Loss의 delta 파라미터. 이 값보다 작은 오차는 MSE처럼,
                        큰 오차는 MAE처럼 계산됨 (기본값 1.0)
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        # 오차 계산
        error = torch.abs(y_pred - y_true)
        # delta보다 작은 오차는 MSE, 큰 오차는 MAE 방식으로 계산
        is_small_error = (error < self.delta).float()
        small_error_loss = 0.5 * error ** 2
        large_error_loss = self.delta * (error - 0.5 * self.delta)
        # 최종 손실 반환
        return torch.mean(is_small_error * small_error_loss + (1 - is_small_error) * large_error_loss)


class DirectionalLoss(nn.Module):
    """
    방향성 손실 함수 - 가격 변동 방향을 정확하게 예측하는 데 중점을 둔 손실 함수
    
    Args:
        alpha (float): 방향성과 크기 간의 가중치 (1에 가까울수록 방향성 중시)
        beta (float): MSE와 방향성 손실 간의 균형을 조절 (기본값 0.5)
    """
    def __init__(self, alpha=0.6, beta=0.5):
        super(DirectionalLoss, self).__init__()
        self.alpha = alpha  # 방향성과 크기 간의 가중치
        self.beta = beta    # MSE와 방향성 손실 간의 균형
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE 손실 계산
        mse_loss = self.mse(y_pred, y_true)
        
        # 방향성 손실 계산 (순차적 변화에 대해)
        if y_pred.size(1) > 1:  # 예측이 여러 시점에 대한 것일 경우
            # 인접 시점 간 변화 계산
            direction_true = y_true[:, 1:] - y_true[:, :-1]
            direction_pred = y_pred[:, 1:] - y_pred[:, :-1]
            
            # 방향성 일치 여부 계산 (양/음의 방향이 같은지)
            dir_match = (direction_true * direction_pred > 0).float()
            
            # 방향성 일치율 계산 (1일수록 방향성 일치도 높음)
            dir_match_rate = torch.mean(dir_match)
            
            # 방향성 손실 (1 - 일치율)
            direction_loss = 1 - dir_match_rate
            
            # 최종 손실 = beta * MSE + (1-beta) * 방향성 손실
            return self.beta * mse_loss + (1 - self.beta) * direction_loss
        
        # 단일 시점 예측일 경우 MSE만 반환
        return mse_loss


class CombinedLoss(nn.Module):
    """
    여러 손실 함수를 결합한 손실 함수
    
    Args:
        mse_weight (float): MSE 손실 가중치
        dir_weight (float): 방향성 손실 가중치
        mae_weight (float): MAE 손실 가중치
    """
    def __init__(self, mse_weight=0.4, dir_weight=0.4, mae_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.dir_weight = dir_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        # MSE 손실
        mse_loss = self.mse(y_pred, y_true)
        
        # MAE 손실
        mae_loss = torch.mean(torch.abs(y_pred - y_true))
        
        # 방향성 손실 계산 (여러 시점에 대한 예측일 경우)
        if y_pred.size(1) > 1:
            # 인접 시점 간 변화 계산
            direction_true = y_true[:, 1:] - y_true[:, :-1]
            direction_pred = y_pred[:, 1:] - y_pred[:, :-1]
            
            # 방향성 일치 여부 계산
            dir_match = (direction_true * direction_pred > 0).float()
            dir_match_rate = torch.mean(dir_match)
            
            # 방향성 손실
            direction_loss = 1 - dir_match_rate
        else:
            # 단일 시점 예측일 경우 방향성 손실은 0
            direction_loss = 0
            
        # 최종 손실 = 각 손실의 가중 합산
        return (self.mse_weight * mse_loss + 
                self.dir_weight * direction_loss + 
                self.mae_weight * mae_loss)


def train_model(train_loader, model, device, num_epochs=200, loss_fn='huber', delta=1.0, alpha=0.6):
    """
    모델을 학습합니다.
    
    Args:
        train_loader (DataLoader): 학습 데이터 로더
        model (nn.Module): 학습할 모델
        device (str): 학습에 사용할 디바이스 ('cuda' 또는 'cpu')
        num_epochs (int): 학습 에폭 수
        loss_fn (str): 손실 함수 ('mse', 'huber', 'directional', 'combined')
        delta (float): Huber Loss에 사용될 델타 값
        alpha (float): DirectionalLoss에 사용될 방향성 가중치
        
    Returns:
        nn.Module: 학습된 모델
    """
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
        
    # 가격 스케일링 여부에 따라 학습률 조정 (비스케일링 시 더 낮은 학습률 사용)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # 모델 예측 (hidden state 제거)
            y_pred, _ = model(x_batch)
            
            # (선택적으로) 차원 조정
            if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
            
            # Loss 계산
            loss = criterion(y_pred, y_batch)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    
    return model


def predict_future_prices(model, X_train, train_df, df, scaler, target_col='Coffee_Price', 
                          days=14, device='cpu', save_plot=True, output_path='./',
                          scale_price=True):
    """
    미래 가격을 예측합니다.
    
    Args:
        model (nn.Module): 학습된 모델
        X_train (torch.Tensor): 학습 데이터 입력 텐서
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
            - forecast_series: 예측 결과 (pd.Series)
            - metrics: 성능 지표 (dict)
    """
    # 1. train_df 마지막 날짜의 위치를 df 전체에서 찾음
    last_train_idx = df.index.get_loc(train_df.index[-1])
    
    # 2. 그 다음 days일치 날짜를 df에서 추출
    prediction_dates = df.index[last_train_idx + 1 : last_train_idx + 1 + days]
    
    # 3. 실제값 가져오기 (테스트용)
    try:
        true_values = df.loc[prediction_dates, target_col].values
    except:
        true_values = None
        print("Warning: 예측 날짜에 대한 실제 값을 찾을 수 없습니다")
    
    # 마지막 시퀀스 예측
    last_seq = X_train[-1].unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        prediction, attn_weights = model(last_seq)
    
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
        plt.plot(prediction_dates, forecast_series, label='Predicted Coffee Price', color='red', linestyle='dashed')
        
        plt.title('Coffee Price Prediction (Enhanced Model)')
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
        plt.savefig(f'{output_path}coffee_prediction_{today}.png', dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(12, 4))
        plt.plot(true_values, label='Actual', color='blue')
        plt.plot(prediction, label='Predicted', color='red', linestyle='dashed')
        plt.title(f"Sample Prediction: {prediction_dates[0].date()} to {prediction_dates[-1].date()}")
        plt.xlabel("Days")
        plt.ylabel("Coffee Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_path}coffee_prediction_sample_{today}.png', dpi=300, bbox_inches='tight')
    
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
        
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["MAPE"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 방향성 정확도 (Direction Accuracy)
        direction_true = np.diff(y_true)
        direction_pred = np.diff(y_pred)
        dir_match = (direction_true * direction_pred > 0)
        metrics["Direction_Accuracy"] = np.mean(dir_match) * 100
        
        print(f"성능 평가:")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
    
    return forecast_series, metrics


def predict_multiple_sequences(model, test_dataset, test_df, df, scaler,
                               data_window, step, future_target, target_col, device,
                               scale_price=True):
    """
    테스트 데이터셋의 여러 시퀀스에 대해 예측을 수행합니다.
    
    Args:
        model (nn.Module): 학습된 모델
        test_dataset (Dataset): 테스트 데이터셋
        test_df (pd.DataFrame): 테스트 데이터프레임
        df (pd.DataFrame): 전체 데이터프레임
        scaler (MinMaxScaler): 데이터 스케일러
        data_window (int): 입력 윈도우 크기
        step (int): 샘플링 간격
        future_target (int): 예측할 미래 시점 수
        target_col (str): 예측 대상 컬럼명
        device (str): 예측에 사용할 디바이스
        scale_price (bool): 가격 특성 스케일링 여부
        
    Returns:
        tuple: (predictions, dates_list) 형태의 튜플
            - predictions: 예측 시리즈 리스트
            - dates_list: 각 예측에 해당하는 날짜 리스트
    """
    model.eval()
    predictions = []
    dates_list = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x_input, _ = test_dataset[i]
            x_input = x_input.unsqueeze(0).to(device)
            
            # 모델 추론
            y_pred, _ = model(x_input)
            y_pred = y_pred.squeeze().cpu().numpy().reshape(-1, 1)
            
            # 역변환 (가격 스케일링 여부에 따라 다름)
            if scale_price:
                dummy = np.zeros((future_target, test_df.shape[1] - 1))
                combined = np.concatenate([y_pred, dummy], axis=1)
                y_inv = scaler.inverse_transform(combined)[:, 0]
            else:
                y_inv = y_pred.flatten()
            
            # test_df에서 해당 시점의 실제 위치를 찾고, df 전체 인덱스로 변환
            base_test_index = i * step + data_window
            if base_test_index + future_target >= len(test_df):
                break
            
            start_timestamp = test_df.index[base_test_index]
            try:
                start_pos_in_df = df.index.get_loc(start_timestamp)
                date_range = df.index[start_pos_in_df + 1 : start_pos_in_df + 1 + future_target]
                
                if len(date_range) == future_target:
                    predictions.append(pd.Series(y_inv, index=date_range))
                    dates_list.extend(date_range)
            except KeyError:
                # 날짜가 전체 df에 없는 경우 건너뜀
                continue
    
    return predictions, dates_list


def visualize_predictions(predictions, df, target_col, save_plot=True, output_path='./'):
    """
    여러 예측 결과를 시각화하고 평균 예측값을 계산합니다.
    
    Args:
        predictions (list): 예측 시리즈 리스트
        df (pd.DataFrame): 전체 데이터프레임
        target_col (str): 예측 대상 컬럼명
        save_plot (bool): 시각화 결과 저장 여부
        output_path (str): 출력 파일 저장 경로
        
    Returns:
        pd.Series: 평균 예측값
    """
    # 예측값 통합 (겹치는 날짜는 평균으로 처리)
    all_preds_df = pd.concat(predictions, axis=1)
    forecast_all = all_preds_df.mean(axis=1)
    
    # 예측 품질 통계
    overlap_counts = (~all_preds_df.isna()).sum(axis=1)
    mean_overlap = overlap_counts.mean()
    print(f"평균 {mean_overlap:.1f}개의 예측이 각 날짜에 겹침")
    
    if save_plot:
        plt.figure(figsize=(14, 6))
        plt.plot(df[target_col], label='Actual Coffee Price', color='blue')
        plt.plot(forecast_all.index, forecast_all.values, 
                 label='Predicted Coffee Price (avg)', color='red', linestyle='dashed')
        
        plt.title('Coffee Price Prediction (Enhanced Model)')
        plt.xlabel('Date')
        plt.ylabel('Coffee Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 현재 날짜로 파일명 생성
        today = datetime.now().strftime('%Y%m%d')
        plt.savefig(f'{output_path}coffee_prediction_{today}.png', dpi=300, bbox_inches='tight')
    
    return forecast_all

#--------------------------- 5. 유틸리티 모듈 ---------------------------#

def save_prediction_to_csv(forecast_series, filename=None, output_path='./'):
    """
    예측 결과를 CSV 파일로 저장합니다.
    
    Args:
        forecast_series (pd.Series): 예측 결과 시리즈
        filename (str, optional): 저장할 파일명
        output_path (str): 출력 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    if filename is None:
        today = datetime.now().strftime('%Y%m%d')
        filename = f'{output_path}coffee_prediction_{today}.csv'
    
    # 예측 결과 DataFrame 생성 및 저장
    prediction_df = pd.DataFrame({
        'Date': forecast_series.index,
        'Prediction_Price': forecast_series.values
    })
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    prediction_df.to_csv(filename, index=False)
    print(f"Prediction saved to: {filename}")
    
    return filename


def save_model(model, scaler, hyperparameters, metrics, output_path='./'):
    """
    학습된 모델과 관련 정보를 저장합니다.
    
    Args:
        model (nn.Module): 학습된 모델
        scaler (MinMaxScaler): 데이터 스케일러
        hyperparameters (dict): 모델 하이퍼파라미터
        metrics (dict): 성능 지표
        output_path (str): 출력 경로
        
    Returns:
        str: 저장된 모델 파일 경로
    """
    today = datetime.now().strftime('%Y%m%d')
    model_path = f"{output_path}attention_model_v2_{today}.pt"
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'date': today
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(model_path, device='cpu'):
    """
    저장된 모델을 로드합니다.
    
    Args:
        model_path (str): 모델 파일 경로
        device (str): 모델을 로드할 디바이스
        
    Returns:
        tuple: (model, scaler, hyperparameters, metrics) 형태의 튜플
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 하이퍼파라미터 추출
    hyperparameters = checkpoint['hyperparameters']
    
    # 모델 초기화 및 가중치 로드
    model = AttentionLSTMModel(
        input_size=hyperparameters['input_size'],
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers'],
        target_size=hyperparameters['target_size']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['scaler'], hyperparameters, checkpoint['metrics']


#--------------------------- 6. 메인 실행 모듈 ---------------------------#

def main(macro_data_path, climate_data_path, output_path='./', 
         data_window=50, future_target=14, step=1, batch_size=10, 
         hidden_size=100, num_epochs=100, do_multiple_predictions=True, 
         scale_price=True, loss_fn='huber', delta=1.0, alpha=0.6):
    """
    모델 학습 및 예측을 실행하는 메인 함수입니다.
    
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
        output_path=output_path, save_plot=True, scale_price=scale_price
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
        'future_target': future_target,
        'scale_price': scale_price,
        'loss_fn': loss_fn
    }
    model_path = save_model(model, scaler, hyperparameters, metrics, output_path)
    print(f"Model saved to: {model_path}")
    
    return forecast_series, model, scaler


# 코랩 실행을 위한 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--macro_data', type=str, 
                      default='/content/drive/MyDrive/캡스톤/거시경제및커피가격통합데이터.csv', 
                      help='Path to macro data')
    parser.add_argument('--climate_data', type=str, 
                      default='/content/drive/MyDrive/캡스톤/기후데이터피쳐선택.csv', 
                      help='Path to climate data')
    parser.add_argument('--output_path', type=str, 
                      default='/content/drive/MyDrive/캡스톤/output/', 
                      help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--data_window', type=int, default=50, help='Data window size')
    parser.add_argument('--future_target', type=int, default=14, help='Future prediction days')
    parser.add_argument('--hidden_size', type=int, default=100, help='LSTM hidden size')
    parser.add_argument('--scale_price', type=lambda x: (str(x).lower() == 'true'), default=True, 
                      help='Whether to scale price feature (True/False)')
    parser.add_argument('--loss_fn', type=str, default='combined', 
                      choices=['mse', 'huber', 'directional', 'combined'], 
                      help='Loss function to use (mse, huber, directional, combined)')
    parser.add_argument('--delta', type=float, default=1.0, help='Delta parameter for Huber Loss')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for Directional Loss')
    parser.add_argument('--mount_drive', type=lambda x: (str(x).lower() == 'true'), default=True,
                      help='Whether to mount Google Drive (Colab only)')
    
    args = parser.parse_args()
    
    # Google Drive 마운트 (코랩에서만)
    if args.mount_drive:
        is_mounted = mount_drive()
        if not is_mounted:
            print("Note: Google Drive not mounted, using local paths")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Starting main with epochs={args.epochs}, batch_size={args.batch_size}, scale_price={args.scale_price}, loss_fn={args.loss_fn}")
    
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
    
    print("Script execution complete.")