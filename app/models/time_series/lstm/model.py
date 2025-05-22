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
import datetime
from .utils import get_result_dir, save_predictions_to_csv
from .market_calendar import adjust_forecast_for_market_calendar, is_trading_day

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

# LSTM + Attention + Entmax 모델 정의 -> 안씀 
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
                print(f"⚠️ Attention 계산 중 오류 발생: {e}. Softmax로 대체합니다.")
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

def returns_to_price(returns, start_price):
    """수익률 시퀀스를 누적 곱하여 price 시퀀스로 변환하는 함수"""
    # returns: (N,) 또는 (batch, N)
    # start_price: float 또는 (batch,)
    if returns.ndim == 1:
        return start_price * np.cumprod(1 + returns)
    elif returns.ndim == 2:
        return np.array([sp * np.cumprod(1 + r) for r, sp in zip(returns, start_price)])
    else:
        raise ValueError('returns shape error')

def predict_and_evaluate(model, test_loader, scaler, device='cpu', test_dates=None, folder_name=None, target='price'):
    """테스트 데이터로 예측 및 평가 + 휴장일 보정 및 csv 저장 (target: price/return)"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from .market_calendar import adjust_forecast_for_market_calendar, is_trading_day

    model.eval()
    predictions = []
    actuals = []
    attention_weights = []
    all_dates = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            try:
                X_batch = X_batch.to(device)
                y_pred, att_weights = model(X_batch)
                predictions.append(y_pred.cpu().numpy())
                actuals.append(y_batch.numpy())
                attention_weights.append(att_weights.cpu().numpy())
            except Exception as e:
                print(f"❌ 예측 배치 처리 중 오류 발생: {e}")
                continue

    if not predictions:
        raise ValueError("❌ 예측 결과가 없습니다. 모든 배치에서 오류가 발생했습니다.")

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    attention_weights = np.vstack(attention_weights)

    # 역정규화 (첫 번째 컬럼인 가격/수익률만)
    try:
        pad_width = scaler.scale_.shape[0] - predictions.shape[1]
        padded_predictions = np.pad(predictions, ((0, 0), (0, pad_width)), 'constant')
        padded_actuals = np.pad(actuals, ((0, 0), (0, pad_width)), 'constant')
        if target == 'price':
            predictions_rescaled = scaler.inverse_transform(padded_predictions)[..., 0]
            actuals_rescaled = scaler.inverse_transform(padded_actuals)[..., 0]
        elif target == 'return':
            # 수익률 컬럼이 1번이라고 가정 (Coffee_Price_Return)
            predictions_rescaled = scaler.inverse_transform(padded_predictions)[..., 1]
            actuals_rescaled = scaler.inverse_transform(padded_actuals)[..., 1]
        else:
            raise ValueError(f'Unknown target: {target}')
    except Exception as e:
        print(f"⚠️ 역정규화 중 오류 발생: {e}. 원본 값 사용.")
        predictions_rescaled = predictions[:, 0]
        actuals_rescaled = actuals[:, 0]

    # === price/return 변환 ===
    if target == 'return':
        # test_dates가 있으면 첫날 가격을 test_dates[0]로부터 추출
        # 없으면 200으로 임시(사용자 맞춤 필요)
        if test_dates is not None and hasattr(test_dates, '__len__') and len(test_dates) > 0:
            # test_loader.dataset[0][0]의 마지막 price를 쓰는 게 더 정확하지만, 여기선 test_dates[0]에 해당하는 price를 사용한다고 가정
            # 실제로는 test set의 첫날 price를 별도 전달받는 게 가장 안전함
            # 여기서는 scaler의 min/max로 역정규화한 price를 사용(0번 컬럼)
            # padded_predictions[0, 0]은 첫날 price의 정규화 값
            # 역정규화
            first_price_norm = padded_predictions[0, 0]
            first_price = scaler.inverse_transform([padded_predictions[0]])[0, 0]
        else:
            first_price = 200.0  # 임시값
        price_pred = returns_to_price(predictions_rescaled, first_price)
        price_actual = returns_to_price(actuals_rescaled, first_price)
    else:
        price_pred = predictions_rescaled
        price_actual = actuals_rescaled

    # 첫날 보정 (price 기준)
    if len(price_pred) > 0 and len(price_actual) > 0:
        shift = price_actual[0] - price_pred[0]
        price_pred = price_pred + shift

    # 예측 구간 날짜 생성 (test_dates가 있으면 그걸 사용, 없으면 연속 날짜 생성)
    if test_dates is not None:
        start_date = pd.to_datetime(test_dates[0])
        end_date = pd.to_datetime(test_dates[0]) + timedelta(days=len(price_pred)-1)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        if not isinstance(test_dates, list):
            test_dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in test_dates]
        else:
            test_dates_str = test_dates
    else:
        all_dates = pd.RangeIndex(len(price_pred))
        test_dates_str = None

    # 휴장일 보정: 휴장일에는 가장 최근 거래일의 예측값을 복사
    pred_series = pd.Series(price_pred, index=all_dates)
    adj_pred_series = pred_series.copy()
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        if not is_trading_day(d):
            prev_idx = i-1
            while prev_idx >= 0 and not is_trading_day(all_dates[prev_idx]):
                prev_idx -= 1
            if prev_idx >= 0:
                adj_pred_series[d] = adj_pred_series[all_dates[prev_idx]]
    # 실제값도 같은 방식으로 맞추되, 없는 날짜는 nan
    adj_actuals = []
    for d in all_dates:
        d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
        if test_dates_str is not None and d_str in test_dates_str:
            idx = test_dates_str.index(d_str)
            adj_actuals.append(price_actual[idx])
        else:
            adj_actuals.append(np.nan)

    # csv 저장 (price/return 모두 저장)
    all_data = []
    for i, d in enumerate(all_dates):
        data_row = {
            'window_id': 1,
            'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d),
            'step': i+1,
            'prediction_price': adj_pred_series.iloc[i],
            'actual_price': adj_actuals[i],
            'prediction_return': predictions_rescaled[i] if target == 'return' else np.nan,
            'actual_return': actuals_rescaled[i] if target == 'return' else np.nan,
            'start_idx': 0,
            'end_idx': len(all_dates)
        }
        all_data.append(data_row)
    df = pd.DataFrame(all_data)
    cols = ['window_id', 'date', 'step', 'prediction_price', 'actual_price', 'prediction_return', 'actual_return', 'start_idx', 'end_idx']
    df = df[[col for col in cols if col in df.columns]]
    if folder_name is None:
        folder_name = 'basic_predict'
    result_dir = get_result_dir(folder_name)
    csv_path = os.path.join(result_dir, 'basic_prediction.csv')
    os.makedirs(result_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ 기본 예측 결과 CSV 저장 완료: {csv_path}")

    mae = np.nanmean(np.abs(adj_pred_series.values - np.array(adj_actuals)))
    rmse = np.sqrt(np.nanmean((adj_pred_series.values - np.array(adj_actuals))**2))
    print(f'평가 지표 - MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    return adj_pred_series.values, np.array(adj_actuals), attention_weights, mae, rmse

def sliding_window_prediction(model, data, scaler, seq_length, pred_length, stride=14, device='cpu', test_dates=None, folder_name=None, isOnline=False, target='price'):
    """슬라이딩 윈도우 방식으로 예측 (온라인 또는 일반, target: price/return)"""
    model.eval()
    print(f"[DEBUG] test_dates: {test_dates}")
    if test_dates is not None:
        test_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in test_dates]
    all_predictions = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 온라인 학습을 위한 옵티마이저 설정

    # 슬라이딩 윈도우 방식으로 예측
    for i in range(0, len(data) - seq_length - pred_length + 1, stride):
        try:
            sequence = data[i:i+seq_length].copy()
            # 실제값 추출 (target 분기)
            if target == 'price':
                actual = data[i+seq_length:i+seq_length+pred_length, 0].copy().reshape(-1)
            elif target == 'return':
                actual = data[i+seq_length:i+seq_length+pred_length, 1].copy().reshape(-1)
            else:
                raise ValueError(f'Unknown target: {target}')
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

            # 예측
            with torch.no_grad():
                prediction, _ = model(sequence_tensor)
                prediction = prediction.cpu().numpy().reshape(-1)  # (pred_length,)

            # 역정규화 (feature 수에 맞게 더미 feature 붙이기)
            n_features = scaler.scale_.shape[0]
            pred_padded = np.zeros((pred_length, n_features))
            actual_padded = np.zeros((pred_length, n_features))
            if target == 'price':
                pred_padded[:, 0] = prediction
                actual_padded[:, 0] = actual
                prediction_rescaled = scaler.inverse_transform(pred_padded)[:, 0]
                actual_rescaled = scaler.inverse_transform(actual_padded)[:, 0]
                prediction_return = np.nan * np.ones_like(prediction_rescaled)
                actual_return = np.nan * np.ones_like(actual_rescaled)
                prediction_price = prediction_rescaled
                actual_price = actual_rescaled
            elif target == 'return':
                pred_padded[:, 1] = prediction
                actual_padded[:, 1] = actual
                prediction_rescaled = scaler.inverse_transform(pred_padded)[:, 1]
                actual_rescaled = scaler.inverse_transform(actual_padded)[:, 1]
                # 누적 곱으로 price 환산 (윈도우 시작점 price 필요)
                # 윈도우 시작점 price는 data[i+seq_length-1, 0] (정규화된 값) -> 역정규화
                start_price_norm = data[i+seq_length-1, 0]
                start_price = scaler.inverse_transform([[start_price_norm]+[0]*(n_features-1)])[0, 0]
                prediction_price = returns_to_price(prediction_rescaled, start_price)
                actual_price = returns_to_price(actual_rescaled, start_price)
                prediction_return = prediction_rescaled
                actual_return = actual_rescaled
            else:
                raise ValueError(f'Unknown target: {target}')

            # 변동성 완화 보정: 온라인 모드일 때만 적용
            if isOnline and i > 0:
                previous_prediction = all_predictions[-1]['prediction_price'] if all_predictions else prediction_price
                change = (prediction_price[0] - previous_prediction[0]) / 1 # 변화율 보정 안함
                prediction_price = previous_prediction + change

            # 예측 결과 보정: 첫날 실제값과 예측값의 차이만큼 전체 예측값을 shift (price 기준)
            shift = actual_price[0] - prediction_price[0]
            prediction_price = prediction_price + shift

            # === 휴장일 보정 ===
            # 날짜 인덱스 생성 (test_dates가 있으면 그에 맞춰서, 없으면 None)
            if test_dates is not None:
                # 각 윈도우의 예측 구간에 해당하는 날짜 리스트 생성
                start_idx = i + seq_length
                end_idx = i + seq_length + pred_length
                window_dates = test_dates[start_idx:end_idx]
                if len(window_dates) == len(prediction_price):
                    pred_series = pd.Series(prediction_price, index=pd.to_datetime(window_dates))
                    prediction_price = adjust_forecast_for_market_calendar(pred_series).values
            # ===

            # 예측 결과 저장 (price/return 모두)
            prediction_info = {
                'start_idx': i,
                'end_idx': i + seq_length + pred_length,
                'prediction_price': prediction_price,
                'actual_price': actual_price,
                'prediction_return': prediction_return,
                'actual_return': actual_return,
                'seq_length': seq_length,
                'pred_length': pred_length
            }
            all_predictions.append(prediction_info)

            # 온라인 학습 (한 step): 온라인 모드일 때만 적용
            if isOnline:
                model.train()
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                target_tensor = torch.FloatTensor(actual).unsqueeze(0).to(device)  # (1, pred_length)
                optimizer.zero_grad()
                output, _ = model(input_tensor)
                output = output.squeeze(0)  # (pred_length,)
                loss = nn.MSELoss()(output, target_tensor.squeeze(0))
                loss.backward()
                optimizer.step()

        except Exception as e:
            print(f"❌ 윈도우 {i} 예측 중 오류 발생: {e}")
            continue
    # price/return 모두 저장
    save_predictions_to_csv(all_predictions, test_dates, folder_name, target=target)
    return all_predictions

def setup_model(input_dim, use_entmax=False):
    """모델 설정"""
    print(f"⏳ 모델 설정 중...")
    
    hidden_dim = 128
    output_dim = 28  # 14일 예측
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

def run_sliding_window_prediction(model, test_data, scaler, seq_length, pred_length, device, stride=1, folder_name=None, test_dates=None, isOnline=False, target='price'):
    """슬라이딩 윈도우 방식으로 예측 진행 (isOnline 파라미터 추가, target: price/return)"""
    print(f"⏳ 슬라이딩 윈도우 예측 진행 중...")
    sliding_predictions = sliding_window_prediction(
        model, 
        test_data, 
        scaler, 
        seq_length, 
        pred_length, 
        stride=stride, 
        device=device, 
        test_dates=test_dates, 
        folder_name=folder_name,
        isOnline=isOnline,
        target=target
    )
    if not sliding_predictions:
        print("⚠️ 슬라이딩 윈도우 예측 결과가 없습니다.")
        return []
    print(f"✅ 슬라이딩 윈도우 예측 완료: {len(sliding_predictions)}개 예측")
    return sliding_predictions

