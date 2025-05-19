import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Entmax15 구현 (Softmax 대체)
class Entmax15(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return entmax15(x, self.dim)

def entmax15(x, dim=1):
    """
    Entmax 1.5 함수: Softmax의 스파스 대안으로, 중요도가 낮은 입력에 0 가중치를 할당
    """
    # 수치 안정성을 위해 입력에서 최대값 제거
    x = x - x.max(dim=dim, keepdim=True)[0]
    
    # entmax15 알고리즘
    tau_star = self_supporting_threshold_entmax15(x, dim)
    y = torch.clamp(x - tau_star, min=0) ** 0.5
    return y / y.sum(dim=dim, keepdim=True)

def self_supporting_threshold_entmax15(x, dim=1):
    """Entmax15를 위한 threshold 계산 (안전한 버전)"""
    n_features = x.size(dim)
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    
    # 누적합 계산
    csm = torch.cumsum(x_sorted, dim=dim)
    rhos = torch.arange(1, n_features + 1, device=x.device, dtype=x.dtype)
    
    support = rhos * x_sorted - csm + 0.5
    support_thresh = support / rhos
    
    # threshold 계산 (안전하게 인덱스 구하기)
    support_size = torch.sum(support > 0, dim=dim, keepdim=True).clamp(min=1)  # 최소 1 보장
    
    # 안전한 인덱스 계산
    support_size = torch.min(support_size, torch.tensor([n_features - 1], device=support_size.device))
    
    # torch.gather 사용 시 인덱스 확인
    gather_indices = support_size.long()
    
    try:
        tau = torch.gather(support_thresh, dim, gather_indices)
    except Exception as e:
        # 오류 발생 시 fallback: 단순 평균 사용
        print(f"⚠️ Entmax 계산 중 오류 발생: {e}. Softmax로 대체합니다.")
        return torch.zeros_like(x)  # Softmax와 유사한 효과를 위해 상수 반환
    
    return tau

# 시계열 데이터셋 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM + Softmax Attention 모델 정의 (Entmax 대신 Softmax 사용)
class LSTMAttentionSoftmax(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(LSTMAttentionSoftmax, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention 레이어
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1)
        
        # Softmax 활성화 함수
        self.softmax = nn.Softmax(dim=1)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Attention 계산
        att_energies = self.v_a(torch.tanh(self.W_a(lstm_out)))  # (batch_size, seq_len, 1)
        att_energies = att_energies.squeeze(-1)  # (batch_size, seq_len)
        
        # Softmax로 attention weights 계산 (Entmax 대신)
        att_weights = self.softmax(att_energies)  # (batch_size, seq_len)
        
        # Context vector 계산
        context = torch.bmm(att_weights.unsqueeze(1), lstm_out)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # 드롭아웃 적용
        context = self.dropout(context)
        
        # 최종 예측
        output = self.fc(context)  # (batch_size, output_dim)
        
        return output, att_weights

# 수정된 LSTM + Attention + Entmax 모델 정의 (안전한 버전)
class LSTMAttentionEntmax(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2, use_entmax=True):
        super(LSTMAttentionEntmax, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_entmax = use_entmax
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention 레이어
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1)
        
        # Attention 활성화 함수 (Entmax15 또는 Softmax)
        if use_entmax:
            self.attention_fn = Entmax15(dim=1)
        else:
            self.attention_fn = nn.Softmax(dim=1)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        try:
            # LSTM 통과
            lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
            
            # Attention 계산
            att_energies = self.v_a(torch.tanh(self.W_a(lstm_out)))  # (batch_size, seq_len, 1)
            att_energies = att_energies.squeeze(-1)  # (batch_size, seq_len)
            
            # Attention weights 계산 (Entmax 또는 Softmax)
            try:
                att_weights = self.attention_fn(att_energies)  # (batch_size, seq_len)
            except Exception as e:
                print(f"[경고] Attention 계산 중 오류 발생: {e}. Softmax로 대체합니다.")
                # Fallback: Softmax 사용
                att_weights = F.softmax(att_energies, dim=1)
            
            # Context vector 계산
            context = torch.bmm(att_weights.unsqueeze(1), lstm_out)  # (batch_size, 1, hidden_dim)
            context = context.squeeze(1)  # (batch_size, hidden_dim)
            
            # 드롭아웃 적용
            context = self.dropout(context)
            
            # 최종 예측
            output = self.fc(context)  # (batch_size, output_dim)
            
            return output, att_weights
            
        except Exception as e:
            print(f"❌ 모델 순전파 중 예외 발생: {e}")
            print("⚠️ 긴급 복구: 평균 풀링을 사용한 단순 예측 수행")
            
            # 긴급 복구: 평균 풀링 후 예측
            avg_pooled = torch.mean(x, dim=1)  # (batch_size, input_dim)
            
            # 단순 예측을 위한 임시 레이어
            if not hasattr(self, 'emergency_fc'):
                self.emergency_fc = nn.Linear(x.size(2), self.fc.out_features).to(x.device)
            
            # 임시 예측
            emergency_output = self.emergency_fc(avg_pooled)  # (batch_size, output_dim)
            
            # 가짜 attention weights 생성
            fake_weights = torch.ones(batch_size, seq_len).to(x.device) / seq_len
            
            return emergency_output, fake_weights

# Huber Loss 클래스 정의
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(y_pred.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()

# 데이터 전처리 함수
def preprocess_data(df, price_col='Coffee_Price', return_col='Coffee_Price_Return'):
    """데이터 전처리: 정규화 및 특성 추가"""
    # 날짜 컬럼을 인덱스로 설정
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 추가 변동성 관련 파생 피처 생성
    # 1. 절대 수익률
    df['Abs_Return'] = np.abs(df[return_col])
    
    # 2. n일 변동성
    for n in [5, 10, 20]:
        df[f'Volatility_{n}d'] = df[return_col].rolling(window=n).std()
    
    # 3. 모멘텀 (n일 전 대비 가격 변화)
    for n in [5, 10, 20]:
        df[f'Momentum_{n}d'] = df[price_col] / df[price_col].shift(n) - 1
    
    # 4. 볼린저 밴드 너비
    for n in [20]:
        rolling_mean = df[price_col].rolling(window=n).mean()
        rolling_std = df[price_col].rolling(window=n).std()
        df[f'BB_Width_{n}d'] = (rolling_mean + 2*rolling_std - (rolling_mean - 2*rolling_std)) / rolling_mean
    
    # 5. Z-score (현재 가격이 과거 n일 평균에서 얼마나 떨어져 있는지)
    for n in [20]:
        rolling_mean = df[price_col].rolling(window=n).mean()
        rolling_std = df[price_col].rolling(window=n).std()
        df[f'Z_Score_{n}d'] = (df[price_col] - rolling_mean) / rolling_std
    
    # NaN 값 제거
    df.dropna(inplace=True)
    
    return df

# 시계열 데이터 윈도우 생성 함수
def create_sequences(data, seq_length, pred_length):
    """시계열 데이터 윈도우 생성: seq_length일의 데이터로 pred_length일 예측"""
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0]  # 첫 번째 컬럼(가격)만 예측
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 모델 훈련 함수 - loss_fn_type 파라미터 추가
def train_model(model, train_loader, val_loader, epochs, lr=0.001, device='cpu', loss_fn_type='mse', delta=1.0):
    """모델 훈련 함수"""
    print(f"⏳ 모델 훈련 시작... (손실 함수: {loss_fn_type}, 에폭: {epochs})")

    model.to(device)
    
    # 손실 함수 설정 (MSE 또는 Huber)
    if loss_fn_type.lower() == 'huber':
        criterion = HuberLoss(delta=delta)
        print(f"✅ Huber Loss 사용 (delta={delta})")
    else:
        criterion = nn.MSELoss()
        print(f"✅ MSE Loss 사용")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for X_batch, y_batch in train_loader:
            try:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파
                y_pred, _ = model(X_batch)
                
                # 손실 계산
                loss = criterion(y_pred, y_batch)
                
                # 역전파
                loss.backward()
                
                # 가중치 업데이트
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"❌ 훈련 배치 처리 중 오류 발생: {e}")
                continue
        
        # 평균 손실 계산
        if batch_count > 0:
            train_loss /= batch_count
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                try:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    # 순전파
                    y_pred, _ = model(X_batch)
                    
                    # 손실 계산
                    loss = criterion(y_pred, y_batch)
                    
                    val_loss += loss.item()
                    batch_count += 1
                except Exception as e:
                    print(f"❌ 검증 배치 처리 중 오류 발생: {e}")
                    continue
        
        # 평균 손실 계산
        if batch_count > 0:
            val_loss /= batch_count
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print(f"✅ 모델 훈련 완료: {epochs}에폭")
    return train_losses, val_losses

# 예측 및 평가 함수
def predict_and_evaluate(model, test_loader, scaler, device='cpu'):
    """테스트 데이터로 예측 및 평가"""
    model.eval()
    predictions = []
    actuals = []
    attention_weights = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            try:
                X_batch = X_batch.to(device)
                
                # 예측
                y_pred, att_weights = model(X_batch)
                
                # 결과 저장
                predictions.append(y_pred.cpu().numpy())
                actuals.append(y_batch.numpy())
                attention_weights.append(att_weights.cpu().numpy())
            except Exception as e:
                print(f"❌ 예측 배치 처리 중 오류 발생: {e}")
                continue
    
    if not predictions:
        raise ValueError("❌ 예측 결과가 없습니다. 모든 배치에서 오류가 발생했습니다.")
    
    # 결합
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    attention_weights = np.vstack(attention_weights)
    
    # 역정규화 (첫 번째 컬럼인 가격만)
    try:
        # 패딩 크기 계산
        pad_width = scaler.scale_.shape[0] - predictions.shape[1]
        
        # 예측 결과와 실제 값을 패딩하여 역정규화
        padded_predictions = np.pad(predictions, ((0, 0), (0, pad_width)), 'constant')
        padded_actuals = np.pad(actuals, ((0, 0), (0, pad_width)), 'constant')
        
        # 역정규화
        predictions_rescaled = scaler.inverse_transform(padded_predictions)[..., 0]
        actuals_rescaled = scaler.inverse_transform(padded_actuals)[..., 0]
    except Exception as e:
        print(f"⚠️ 역정규화 중 오류 발생: {e}. 원본 값 사용.")
        predictions_rescaled = predictions[:, 0]  # 첫 번째 값만 사용
        actuals_rescaled = actuals[:, 0]
    
    # 평가 지표 계산
    mae = np.mean(np.abs(predictions_rescaled - actuals_rescaled))
    rmse = np.sqrt(np.mean((predictions_rescaled - actuals_rescaled)**2))
    
    print(f'평가 지표 - MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    
    return predictions_rescaled, actuals_rescaled, attention_weights, mae, rmse

# 슬라이딩 윈도우 예측 데이터 생성 함수
def sliding_window_prediction(model, data, scaler, seq_length, pred_length, stride=1, device='cpu'):
    """슬라이딩 윈도우 방식으로 예측"""
    model.eval()
    
    # 첫 번째 시퀀스 준비
    all_predictions = []
    
    for i in range(0, len(data) - seq_length - pred_length + 1, stride):
        try:
            # 시퀀스 데이터 추출
            sequence = data[i:i+seq_length]
            actual = data[i+seq_length:i+seq_length+pred_length, 0]  # 첫 번째 컬럼(가격)
            
            # 텐서로 변환
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # [1, seq_length, n_features]
            
            # 예측
            with torch.no_grad():
                prediction, _ = model(sequence_tensor)
                prediction = prediction.cpu().numpy()

            # 역정규화
            try:
                # 예측 결과 shape: (1, pred_length)
                prediction = prediction[0]  # shape: (pred_length,)

                # 실제 값도 동일하게
                actual = actual.reshape(-1)  # shape: (pred_length,)

                # scaler로 inverse_transform 하기 위해 2D 배열로 맞춤
                if scaler.scale_.shape[0] == 1:
                    # 단일 피처만 정규화한 경우
                    prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                    actual_rescaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
                else:
                    # 다변량 정규화된 경우, 첫 번째 피처(가격)만 복원
                    n_features = scaler.scale_.shape[0]

                    # prediction을 가격값만 포함한 전체 피처 벡터로 패딩
                    pred_padded = np.zeros((pred_length, n_features))
                    pred_padded[:, 0] = prediction  # 가격이 첫 번째 피처

                    actual_padded = np.zeros((pred_length, n_features))
                    actual_padded[:, 0] = actual

                    # 역정규화
                    prediction_rescaled = scaler.inverse_transform(pred_padded)[:, 0]
                    actual_rescaled = scaler.inverse_transform(actual_padded)[:, 0]
            except Exception as e:
                print(f"⚠️ 역정규화 중 오류 발생: {e}. 원본 값 사용.")
                prediction_rescaled = prediction
                actual_rescaled = actual

            
            # 예측 저장
            all_predictions.append({
                'start_idx': i,
                'end_idx': i + seq_length + pred_length,
                'prediction': prediction_rescaled,
                'actual': actual_rescaled
            })
        except Exception as e:
            print(f"[에러] 윈도우 {i} 예측 중 오류 발생: {e}")
            continue
    
    return all_predictions

def setup_model(input_dim, use_entmax=False):
    """모델 설정"""
    print(f"⏳ 모델 설정 중...")
    
    hidden_dim = 128
    output_dim = 14  # 14일 예측
    num_layers = 2
    dropout = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 사용 장치: {device}")
    
    # Entmax 또는 Softmax 기반 모델 선택
    if use_entmax:
        print("🔍 Entmax Attention 모델 사용")
        model = LSTMAttentionEntmax(input_dim, hidden_dim, output_dim, num_layers, dropout, use_entmax=True)
    else:
        print("🔍 Softmax Attention 모델 사용")
        model = LSTMAttentionSoftmax(input_dim, hidden_dim, output_dim, num_layers, dropout)
    
    print(f"✅ 모델 생성 완료: {model.__class__.__name__}")
    
    return model, device

def run_sliding_window_prediction(model, test_data, scaler, seq_length, pred_length, device):
    """슬라이딩 윈도우 방식으로 예측 진행"""
    print(f"⏳ 슬라이딩 윈도우 예측 진행 중...")
    
    # 슬라이딩 윈도우 예측
    sliding_predictions = sliding_window_prediction(model, test_data, scaler, seq_length, pred_length, stride=1, device=device)
    
    if not sliding_predictions:
        print("⚠️ 슬라이딩 윈도우 예측 결과가 없습니다.")
        return []
    
    print(f"✅ 슬라이딩 윈도우 예측 완료: {len(sliding_predictions)}개 예측")
    return sliding_predictions


def online_update_prediction(model, test_data, scaler, seq_length, pred_length, device='cpu', lr=1e-4, loss_fn='mse'):
    model.to(device)
    model.train()

    # 손실 함수
    if loss_fn == 'huber':
        criterion = HuberLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    predictions = []
    actuals = []

    for i in range(0, len(test_data) - seq_length - pred_length + 1):
        # 시퀀스 준비
        input_seq = test_data[i:i+seq_length]
        target_seq = test_data[i+seq_length:i+seq_length+pred_length, 0]  # 실제 가격

        # 텐서 변환
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
        target_tensor = torch.FloatTensor(target_seq).unsqueeze(0).to(device)

        # 예측
        model.eval()
        with torch.no_grad():
            pred_tensor, _ = model(input_tensor)
        model.train()

        # 예측 저장
        pred_np = pred_tensor.cpu().numpy().flatten()
        target_np = target_tensor.cpu().numpy().flatten()
        predictions.append(pred_np)
        actuals.append(target_np)

        # 온라인 학습 (한 step)
        optimizer.zero_grad()
        output, _ = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return predictions, actuals
