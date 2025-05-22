import pandas as pd
import numpy as np
import os

def load_eco_data(data_path=None):
    """_summary_
    거시경제 및 커피 가격 통합 데이터셋을 로드하고, 결측치를 처리한 후 반환합니다.
    Args:
        data_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if data_path is None:
        # 기본 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '거시경제및커피가격통합데이터.csv')
        
        # 파일 존재 여부 확인 및 디버깅 정보 출력
        print(f"현재 디렉터리: {current_dir}")
        print(f"앱 디렉터리: {app_dir}")
        print(f"찾고 있는 파일 경로: {data_path}")
        print(f"파일 존재 여부: {os.path.exists(data_path)}")
        
        # 파일이 없으면 상대 경로로 다시 시도
        if not os.path.exists(data_path):
            # 모델 서버 루트 디렉토리 기준으로 시도
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '거시경제및커피가격통합데이터.csv')
            print(f"루트 디렉토리 기준으로 시도: {data_path}")
            print(f"파일 존재 여부: {os.path.exists(data_path)}")


    print(f"⏳ 데이터 로드 중...")

    try:
        df = pd.read_csv(data_path)
        print(f"✅ 경제 데이터 로드 성공: {df.shape}")
    except Exception as e:
        print(f"❌ 경제 데이터 로드 실패: {e}")
        raise

    return df

def load_weather_data(data_path=None):
    """
    날씨 및 lag 특성이 포함된 데이터셋을 로드하고, coffee_label.csv와 Date 기준으로 병합하여 저장.
    
    Args:
        data_path (str, optional): 날씨 데이터 파일 경로
        label_path (str, optional): 라벨 데이터 파일 경로
    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    if data_path is None:
        # 기본 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '비수확기평균커피가격통합데이터.csv')
        
        # 파일 존재 여부 확인 및 디버깅 정보 출력
        print(f"현재 디렉터리: {current_dir}")
        print(f"앱 디렉터리: {app_dir}")
        print(f"찾고 있는 파일 경로: {data_path}")
        print(f"파일 존재 여부: {os.path.exists(data_path)}")
        
        # 파일이 없으면 상대 경로로 다시 시도
        if not os.path.exists(data_path):
            # 모델 서버 루트 디렉토리 기준으로 시도
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '거시경제및커피가격통합데이터.csv')
            print(f"루트 디렉토리 기준으로 시도: {data_path}")
            print(f"파일 존재 여부: {os.path.exists(data_path)}")


    print(f"⏳ 데이터 로드 중...")

    try:
        df = pd.read_csv(data_path)
        print(f"✅ 날씨 데이터 로드 성공: {df.shape}")
    except Exception as e:
        print(f"❌ 날씨 데이터 로드 실패: {e}")
        raise

    return df

df = load_eco_data()

df['abs_return'] = df['Coffee_Price_Return'].abs()
df["log_return"] = np.log(df["Coffee_Price"]) - np.log(df["Coffee_Price"].shift(1))
df["target_volatility_14d"] = df["log_return"].rolling(window=14).std().shift()
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

df['momentum_1d'] = df['Coffee_Price'].diff(1)
df['momentum_3d'] = df['Coffee_Price'].diff(3)
df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']

we = load_weather_data()

we.drop(columns=['Coffee_Price', 'Coffee_Price_Return','Crude_Oil_Price','USD_KRW','USD_BRL','USD_COP'], inplace=True)
df = pd.merge(df, we, on='Date', how='left')
df = df.dropna()

print(f"데이터: {df.shape}")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

n = len(df)
train_size = int(n * 0.9)

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

from sklearn.preprocessing import MinMaxScaler

return_train = train_df["Coffee_Price_Return"].copy()
return_test = test_df["Coffee_Price_Return"].copy()


log_return_train = train_df["log_return"].copy()
log_return_test = test_df["log_return"].copy()

# target_train = train_df["target_volatility_14d"].copy()
# target_test = test_df["target_volatility_14d"].copy()

scaler_seq = MinMaxScaler()
train_df = pd.DataFrame(scaler_seq.fit_transform(train_df),
                        columns=train_df.columns,
                        index=train_df.index)

test_df = pd.DataFrame(scaler_seq.transform(test_df),
                        columns=test_df.columns,
                        index=test_df.index)

train_df["Coffee_Price_Return"] = return_train
test_df["Coffee_Price_Return"] = return_test

train_df["log_return"] = log_return_train
test_df["log_return"] = log_return_test

# train_df["target_volatility_14d"] = target_train
# test_df["target_volatility_14d"] = target_test
     

print("Train target min:", train_df['target_volatility_14d'].min())
print("Train target max:", train_df['target_volatility_14d'].max())
print("Test target min:", test_df['target_volatility_14d'].min())
print("Test target max:", test_df['target_volatility_14d'].max())

import matplotlib.pyplot as plt

plt.hist(train_df["target_volatility_14d"], bins=50, alpha=0.5, label='Train')
plt.hist(test_df["target_volatility_14d"], bins=50, alpha=0.5, label='Test')
plt.legend()
plt.title("Target Volatility Distribution (MinMax)")
plt.show()

import numpy as np
import torch
from torch.utils.data import Dataset

class MultiStepTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window_size, step, static_feat_idx):
        self.data = []
        self.labels = []
        self.static_feats = []

        self.seq_feat_idx = [i for i in range(X.shape[1]) if i not in static_feat_idx]

        for i in range(0, len(X) - window_size, step):
            x_seq = X[i:i+window_size, self.seq_feat_idx]         # (T, D_seq)
            x_static = X[i, static_feat_idx]                      # (D_static,)
            y_target = y[i + window_size]                         # 미래 14일 변동성 (미리 shift(-13) 해둔 상태)

            if np.isnan(y_target):  # 결측치 제거
                continue

            self.data.append(x_seq)
            self.static_feats.append(x_static)
            self.labels.append(y_target)

        self.data = np.array(self.data)
        self.static_feats = np.array(self.static_feats)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_seq = torch.tensor(self.data[idx], dtype=torch.float32)
        x_static = torch.tensor(self.static_feats[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # (scalar)
        return x_seq, x_static, y
     

X_train = train_df.values
target_col = "target_volatility_14d"
y_train = train_df[target_col].values

X_test = test_df.values
y_test = test_df[target_col].values
     

data_window = 100  # 최근 데이터를 입력으로 사용
step = 1  # 단위로 샘플링
     

static_feat_idx = list(range(X_train.shape[1] - 9, X_train.shape[1]))

train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, step, static_feat_idx)
test_dataset = MultiStepTimeSeriesDataset(X_test, y_test, data_window, step, static_feat_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def compute_attention_entropy(attn_weights):
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=1)  # (B,)
    return entropy.mean().item()


from entmax import Entmax15

class EntmaxAttention(nn.Module):
    def __init__(self, hidden_size, attn_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)

    def forward(self, lstm_output):
        # lstm_output: (B, T, H)
        scores = self.score_layer(lstm_output).squeeze(-1)  # (B, T)
        weights = self.entmax(scores)  # sparse attention weights
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights
     

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3, static_feat_dim=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attention = EntmaxAttention(hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        # 정적 피처 인코더 추가
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feat_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # FC 입력 차원 변경: fused(H) + static_encoded(64)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1)
        )

    def forward(self, x_seq, x_static, hidden_states=None):
        batch_size = x_seq.size(0)

        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            hidden_states = (h0, c0)

        lstm_out, _ = self.lstm(x_seq, hidden_states)

        if self.training:
            lstm_out = lstm_out + 0.05 * torch.randn_like(lstm_out)

        context, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat([context, last_hidden], dim=1)
        alpha = self.gate(combined)
        fused = alpha * context + (1 - alpha) * last_hidden

        # 정적 피처 인코딩 및 결합
        static_encoded = self.static_encoder(x_static)  # (B, 64)
        fused_with_static = torch.cat([fused, static_encoded], dim=1)  # (B, H+64)

        out = self.fc(fused_with_static).squeeze(-1)
        return out, attn_weights
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
     

def weighted_mse_loss(y_pred, y_true, temp=5.0):
    # 개별 loss 계산: (B,)
    sample_losses = (y_pred - y_true) ** 2

    # softmax 기반 가중치: 큰 loss에 더 높은 weight
    weights = torch.softmax(sample_losses * temp, dim=0)  # temp 높을수록 sharpen
    weighted_loss = torch.sum(weights * sample_losses)

    return weighted_loss
     

from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X_all = train_df.values
y_all = train_df[target_col].values

static_feat_idx = list(range(X_all.shape[1] - 9, X_all.shape[1]))
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
    print(f"\n===== Fold {fold + 1} =====")

    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]

    X_val_fold = X_all[val_idx]
    y_val_fold = y_all[val_idx]

    train_dataset = MultiStepTimeSeriesDataset(
        X_train_fold, y_train_fold,
        window_size=data_window,
        step=step,
        static_feat_idx=static_feat_idx
    )
    val_dataset = MultiStepTimeSeriesDataset(
        X_val_fold, y_val_fold,
        window_size=data_window,
        step=step,
        static_feat_idx=static_feat_idx
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = AttentionLSTMModel(
        input_size=X_all.shape[1] - 9,
        static_feat_dim=9
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(5):
        model.train()
        for x_seq, x_static, y in train_loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred, _ = model(x_seq, x_static)
            # weighted MSE 적용
            loss = weighted_mse_loss(y_pred, y, temp=5.0)
            loss.backward()
            optimizer.step()

        # Validation 평가 (여기는 일반 MSE로 측정)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_seq, x_static, y in val_loader:
                x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
                y_pred, _ = model(x_seq, x_static)
                val_loss = torch.mean((y_pred - y) ** 2).item()
                val_losses.append(val_loss)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")

    fold_results.append(avg_val_loss)

print("\nFold별 평균 Loss:", fold_results)
print("전체 평균 Validation Loss:", np.mean(fold_results))


import matplotlib.pyplot as plt

plt.plot(y_all, label="target")
for i, (_, val_idx) in enumerate(tscv.split(X_all)):
    plt.axvspan(val_idx[0], val_idx[-1], color=f"C{i}", alpha=0.2, label=f"Fold {i+1}")
plt.legend()
plt.title("Validation")
plt.show()


x_seq, x_static, _ = train_dataset[0]

input_size = x_seq.shape[1]
model = AttentionLSTMModel(input_size=input_size, static_feat_dim=9).to(device)
base_criterion = nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
     

# ✅ 최종 학습 루프 (weighted loss 포함)
for epoch in range(50):  # ← epoch 수 늘림
    model.train()
    epoch_loss = 0.0

    for x_seq, x_static, y in train_loader:
        x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x_seq, x_static)
        loss = weighted_mse_loss(y_pred, y, temp=5.0)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Final Training] Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

num_epochs = 5

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    all_train_preds = []
    all_train_targets = []

    for x_seq, x_static, y_batch in train_loader:
        x_seq, x_static, y_batch = x_seq.to(device), x_static.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred, _ = model(x_seq, x_static)  # y_pred: (B,)

        loss = base_criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        epoch_loss += loss.item()
        all_train_preds.append(y_pred.detach().cpu())
        all_train_targets.append(y_batch.detach().cpu())

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 통계 출력
    all_train_preds_tensor = torch.cat(all_train_preds, dim=0)
    all_train_targets_tensor = torch.cat(all_train_targets, dim=0)
    y_pred_mean = all_train_preds_tensor.mean().item()
    y_pred_std = all_train_preds_tensor.std().item()
    y_true_mean = all_train_targets_tensor.mean().item()
    y_true_std = all_train_targets_tensor.std().item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_test_seq, x_test_static, y_test in test_loader:
            x_test_seq, x_test_static, y_test = x_test_seq.to(device), x_test_static.to(device), y_test.to(device)
            y_test_pred, _ = model(x_test_seq, x_test_static)

            test_loss += base_criterion(y_test_pred, y_test).item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    scheduler.step(avg_test_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
          f"y_pred mean: {y_pred_mean:.4f}, std: {y_pred_std:.4f} | "
          f"y_true mean: {y_true_mean:.4f}, std: {y_true_std:.4f}")
    
with torch.no_grad():
    # 1개 샘플 선택 후 배치 차원 유지
    sample_seq, sample_static, _ = train_dataset[0]
    sample_seq = sample_seq.unsqueeze(0).to(device)      # (1, T, D_seq)
    sample_static = sample_static.unsqueeze(0).to(device)  # (1, D_static)

    _, attn_weights = model(sample_seq, sample_static)
    print("Attention std:", attn_weights.std().item())
    print("Attention weights:", attn_weights)

input_seq1, static_feat1, y_true1 = test_dataset[0]
input_seq2, static_feat2, y_true2 = test_dataset[10]

# 배치 차원 추가 (B=1)
input_seq1 = input_seq1.unsqueeze(0).to(device)
input_seq2 = input_seq2.unsqueeze(0).to(device)
static_feat1 = static_feat1.unsqueeze(0).to(device)
static_feat2 = static_feat2.unsqueeze(0).to(device)

# 모델 호출 시 두 인자 모두 전달
out1, _ = model(input_seq1, static_feat1)
out2, _ = model(input_seq2, static_feat2)

print("Prediction difference:", torch.norm(out1 - out2).item())

with torch.no_grad():
    out, _ = model.lstm(sample_seq)
    std_across_time = out.std(dim=1).mean().item()
    print("LSTM output std (across time steps):", std_across_time)

print("y_pred mean:", y_pred.mean().item())
print("y_batch mean:", y_batch.mean().item())
     
predictions = []
true_values = []
date_ranges = []
future_target = 14

target_col = "target_volatility_14d"
target_idx = test_df.columns.get_loc(target_col)

with torch.no_grad():
    for batch_idx, (x_seq, x_static, y_true) in enumerate(test_loader):
        x_seq = x_seq.to(device)
        x_static = x_static.to(device)

        y_pred, _ = model(x_seq, x_static)               # (B,)
        y_pred = y_pred.squeeze(-1).cpu().numpy()        # (B,)
        y_true = y_true.cpu().numpy()                    # (B,)

        for i in range(x_seq.size(0)):
            global_idx = batch_idx * test_loader.batch_size + i
            target_idx_in_df = global_idx * step + data_window + future_target - 1

            if target_idx_in_df >= len(test_df):
                continue

            dummy_pred = np.zeros((1, test_df.shape[1]))
            dummy_true = np.zeros((1, test_df.shape[1]))

            dummy_pred[0, target_idx] = y_pred[i]
            dummy_true[0, target_idx] = y_true[i]

            # 역정규화
            y_pred_inv = scaler_seq.inverse_transform(dummy_pred)[0, target_idx]
            y_true_inv = scaler_seq.inverse_transform(dummy_true)[0, target_idx]

            predictions.append(y_pred_inv)
            true_values.append(y_true_inv)
            date_ranges.append(test_df.index[target_idx_in_df])

print("y_pred std:", np.std(predictions))
print("y_true std:", np.std(true_values))

import matplotlib.pyplot as plt
import pandas as pd

pred_series = pd.Series(predictions, index=date_ranges)
true_series = pd.Series(true_values, index=date_ranges)

plt.figure(figsize=(12, 5))
plt.plot(true_series.sort_index(), label='Actual 14-day Volatility')
plt.plot(pred_series.sort_index(), label='Predicted 14-day Volatility', linestyle='--')
plt.title("Predicted vs Actual 14-day Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (std of log returns)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df_preds = pd.DataFrame({
    "date": date_ranges,
    "pred_vol": predictions,
    "true_vol": true_values
})

# 해당 날짜의 실제 가격 정보 추가
df_preds["price"] = df.loc[df_preds["date"], "Coffee_Price"].values

# 인덱스를 날짜로 지정
df_preds.set_index("date", inplace=True)

print(df_preds.head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ① 예측 변동성 기반 가격 시뮬레이션 함수
def simulate_price_curve(start_date, predicted_volatility, base_price, num_days=14):
    np.random.seed(0)
    simulated_returns = np.random.normal(loc=0.0, scale=predicted_volatility, size=num_days)
    prices = base_price * np.exp(np.cumsum(simulated_returns))
    date_range = pd.date_range(start=start_date, periods=num_days)
    return pd.Series(prices, index=date_range)

def plot_price_simulation_curve(df, index=0):
    date = df.index[index]
    vol = df.iloc[index]["pred_vol"]
    price = df.iloc[index]["price"]
    sim_curve = simulate_price_curve(date, vol, price)
    plt.figure(figsize=(10, 4))
    plt.plot(sim_curve.index, sim_curve.values, label='Simulated Price')
    plt.title(f"Simulated Price from {date.date()} (vol={vol:.4f})")
    plt.xlabel("Date")
    plt.ylabel("Simulated Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_price_simulation_curve(df_preds, index=0)


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = np.array(predictions)
y_true = np.array(true_values)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")