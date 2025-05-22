"""
LSTM 모델 2: 변동성 예측 모델 (리팩토링 버전)
커피 가격의 변동성을 예측하여 리스크 관리에 활용하는 모델

주요 기능:
- 거시경제 및 기후 데이터를 활용한 변동성 예측
- EntMax Attention 메커니즘 적용
- 정적/동적 피처 분리 처리
- 시계열 교차검증
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from entmax import Entmax15

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 한글 폰트 설정 제거 (범용성)
plt.rcParams['axes.unicode_minus'] = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_eco_data(data_path=None):
    """거시경제 및 커피 가격 통합 데이터셋 로드"""
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '거시경제및커피가격통합데이터.csv')
        
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '거시경제및커피가격통합데이터.csv')

    print(f"⏳ 경제 데이터 로드 중: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ 경제 데이터 로드 성공: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ 경제 데이터 로드 실패: {e}")
        raise

def load_weather_data(data_path=None):
    """기후 데이터셋 로드"""
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
        data_path = os.path.join(app_dir, 'data', 'input', '비수확기평균커피가격통합데이터.csv')
        
        if not os.path.exists(data_path):
            root_dir = os.path.abspath(os.path.join(current_dir, '../../../../'))
            data_path = os.path.join(root_dir, 'app', 'data', 'input', '비수확기평균커피가격통합데이터.csv')

    print(f"⏳ 기후 데이터 로드 중: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ 기후 데이터 로드 성공: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ 기후 데이터 로드 실패: {e}")
        raise

def create_features(df):
    """파생 변수 생성"""
    print("⏳ 파생 변수 생성 중...")
    
    # 기본 변동성 관련 피처
    df['abs_return'] = df['Coffee_Price_Return'].abs()
    df["log_return"] = np.log(df["Coffee_Price"]) - np.log(df["Coffee_Price"].shift(1))
    
    # 타겟 변수 (shit에 의해서 미래 데이터를 참고하는 경향이 있음)
    df["target_volatility_14d"] = df["log_return"].rolling(window=14).std()\
        # .shift(-13)
    
    # 변동성 피처들
    df['volatility_5d'] = df['Coffee_Price_Return'].rolling(window=5).std()
    df['volatility_10d'] = df['Coffee_Price_Return'].rolling(window=10).std()
    
    # 모멘텀 피처들
    df['momentum_1d'] = df['Coffee_Price'].diff(1)
    df['momentum_3d'] = df['Coffee_Price'].diff(3)
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)
    
    # 볼린저 밴드 및 기타 피처
    rolling_mean = df['Coffee_Price'].rolling(window=20).mean()
    rolling_std = df['Coffee_Price'].rolling(window=20).std()
    df['bollinger_width'] = (2 * rolling_std) / rolling_mean
    
    df['return_zscore'] = (df['Coffee_Price_Return'] - df['Coffee_Price_Return'].rolling(20).mean()) / \
                          (df['Coffee_Price_Return'].rolling(20).std() + 1e-6)
    
    df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_10d'] + 1e-6)
    
    print(f"✅ 파생 변수 생성 완료: {df.shape}")
    return df

def prepare_data():
    """데이터 로드 및 전처리"""
    print("🚀 데이터 준비 시작")
    
    # 데이터 로드
    df = load_eco_data()
    we = load_weather_data()
    
    # 파생 변수 생성
    df = create_features(df)
    
    # 기후 데이터 병합 (중복 컬럼 제거)
    weather_cols_to_drop = ['Coffee_Price', 'Coffee_Price_Return', 'Crude_Oil_Price', 
                           'USD_KRW', 'USD_BRL', 'USD_COP']
    we = we.drop(columns=[col for col in weather_cols_to_drop if col in we.columns])
    df = pd.merge(df, we, on='Date', how='left')
    
    # 결측치 제거
    df = df.dropna()
    print(f"✅ 최종 데이터 형태: {df.shape}")
    
    return df

class MultiStepTimeSeriesDataset(torch.utils.data.Dataset):
    """시계열 데이터셋 클래스"""
    
    def __init__(self, X, y, window_size, step, static_feat_idx):
        self.data = []
        self.labels = []
        self.static_feats = []
        self.seq_feat_idx = [i for i in range(X.shape[1]) if i not in static_feat_idx]

        for i in range(0, len(X) - window_size, step):
            x_seq = X[i:i+window_size, self.seq_feat_idx]
            x_static = X[i, static_feat_idx]
            y_target = y[i + window_size]

            if np.isnan(y_target):
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
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_seq, x_static, y

class EntmaxAttention(nn.Module):
    """EntMax Attention 메커니즘"""
    
    def __init__(self, hidden_size, attn_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)

    def forward(self, lstm_output):
        scores = self.score_layer(lstm_output).squeeze(-1)
        weights = self.entmax(scores)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        return context, weights

class AttentionLSTMModel(nn.Module):
    """Attention 기반 LSTM 변동성 예측 모델"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3, static_feat_dim=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention 메커니즘
        self.attention = EntmaxAttention(hidden_size)

        # Gate 메커니즘
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        # 정적 피처 인코더
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feat_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # 최종 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x_seq, x_static, hidden_states=None):
        batch_size = x_seq.size(0)

        # 초기 hidden state 설정
        if hidden_states is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x_seq.device)
            hidden_states = (h0, c0)

        # LSTM 통과
        lstm_out, _ = self.lstm(x_seq, hidden_states)

        # Attention 적용
        context, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        
        # Gate를 통한 정보 결합
        combined = torch.cat([context, last_hidden], dim=1)
        alpha = self.gate(combined)
        fused = alpha * context + (1 - alpha) * last_hidden

        # 정적 피처 인코딩 및 결합
        static_encoded = self.static_encoder(x_static)
        fused_with_static = torch.cat([fused, static_encoded], dim=1)

        # 최종 예측
        out = self.fc(fused_with_static).squeeze(-1)
        return out, attn_weights

def weighted_mse_loss(y_pred, y_true, temp=5.0):
    """가중 MSE 손실 함수"""
    sample_losses = (y_pred - y_true) ** 2
    weights = torch.softmax(sample_losses * temp, dim=0)
    weighted_loss = torch.sum(weights * sample_losses)
    return weighted_loss

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """모델 훈련"""
    print(f"⏳ 모델 훈련 시작 (에폭: {num_epochs})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        epoch_loss = 0.0
        
        for x_seq, x_static, y_batch in train_loader:
            x_seq, x_static, y_batch = x_seq.to(device), x_static.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred, _ = model(x_seq, x_static)
            loss = weighted_mse_loss(y_pred, y_batch, temp=5.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, x_static, y_batch in val_loader:
                x_seq, x_static, y_batch = x_seq.to(device), x_static.to(device), y_batch.to(device)
                y_pred, _ = model(x_seq, x_static)
                val_loss += nn.MSELoss()(y_pred, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}")
    
    print("✅ 모델 훈련 완료")
    return train_losses, val_losses

def cross_validate_model(X_all, y_all, static_feat_idx, data_window=100, step=1):
    """시계열 교차검증"""
    print("⏳ 교차검증 시작")
    
    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        print(f"\n===== Fold {fold + 1} =====")
        
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]
        
        # 데이터셋 생성
        train_dataset = MultiStepTimeSeriesDataset(
            X_train_fold, y_train_fold, data_window, step, static_feat_idx
        )
        val_dataset = MultiStepTimeSeriesDataset(
            X_val_fold, y_val_fold, data_window, step, static_feat_idx
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 모델 생성 및 훈련
        model = AttentionLSTMModel(
            input_size=X_all.shape[1] - len(static_feat_idx),
            static_feat_dim=len(static_feat_idx)
        ).to(device)
        
        _, val_losses = train_model(model, train_loader, val_loader, num_epochs=5)
        fold_results.append(min(val_losses))
    
    print(f"\n✅ 교차검증 완료")
    print(f"Fold별 최고 성능: {fold_results}")
    print(f"평균 검증 성능: {np.mean(fold_results):.4f}")
    
    return fold_results

def evaluate_model(model, test_loader, scaler, test_df, target_col="target_volatility_14d"):
    """모델 평가 및 결과 분석"""
    print("⏳ 모델 평가 중...")
    
    model.eval()
    predictions = []
    true_values = []
    date_ranges = []
    
    target_idx = test_df.columns.get_loc(target_col)
    data_window = 100
    step = 1
    
    with torch.no_grad():
        for batch_idx, (x_seq, x_static, y_true) in enumerate(test_loader):
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            
            y_pred, _ = model(x_seq, x_static)
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            
            for i in range(x_seq.size(0)):
                global_idx = batch_idx * test_loader.batch_size + i
                target_idx_in_df = global_idx * step + data_window
                
                if target_idx_in_df >= len(test_df):
                    continue
                
                # 역정규화
                dummy_pred = np.zeros((1, test_df.shape[1]))
                dummy_true = np.zeros((1, test_df.shape[1]))
                dummy_pred[0, target_idx] = y_pred[i]
                dummy_true[0, target_idx] = y_true[i]
                
                y_pred_inv = scaler.inverse_transform(dummy_pred)[0, target_idx]
                y_true_inv = scaler.inverse_transform(dummy_true)[0, target_idx]
                
                predictions.append(y_pred_inv)
                true_values.append(y_true_inv)
                date_ranges.append(test_df.index[target_idx_in_df])
    
    # 성능 지표 계산
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mae = mean_absolute_error(true_values, predictions)
    
    print(f"✅ 평가 완료 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return predictions, true_values, date_ranges, rmse, mae

def visualize_results(predictions, true_values, date_ranges, title="변동성 예측 결과"):
    """결과 시각화"""
    pred_series = pd.Series(predictions, index=date_ranges)
    true_series = pd.Series(true_values, index=date_ranges)
    
    plt.figure(figsize=(15, 8))
    plt.plot(true_series.sort_index(), label='실제 변동성', linewidth=2)
    plt.plot(pred_series.sort_index(), label='예측 변동성', linestyle='--', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('변동성', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def simulate_price_curve(start_date, predicted_volatility, base_price, num_days=14):
    """변동성 기반 가격 시뮬레이션"""
    np.random.seed(42)
    simulated_returns = np.random.normal(loc=0.0, scale=predicted_volatility, size=num_days)
    prices = base_price * np.exp(np.cumsum(simulated_returns))
    date_range = pd.date_range(start=start_date, periods=num_days)
    return pd.Series(prices, index=date_range)

def main():
    """메인 실행 함수"""
    print("🚀 LSTM 변동성 예측 모델 시작")
    
    # 데이터 준비
    df = prepare_data()
    
    # 날짜 처리 및 인덱스 설정
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 데이터 분할
    n = len(df)
    train_size = int(n * 0.9)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # 정규화 (일부 컬럼 제외)
    scaler = MinMaxScaler()
    
    # 제외할 컬럼들 백업
    return_train = train_df["Coffee_Price_Return"].copy()
    return_test = test_df["Coffee_Price_Return"].copy()
    log_return_train = train_df["log_return"].copy()
    log_return_test = test_df["log_return"].copy()
    
    # 정규화 적용
    train_df_scaled = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )
    test_df_scaled = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index
    )
    
    # 제외 컬럼들 복원
    train_df_scaled["Coffee_Price_Return"] = return_train
    test_df_scaled["Coffee_Price_Return"] = return_test
    train_df_scaled["log_return"] = log_return_train
    test_df_scaled["log_return"] = log_return_test
    
    # 정적 피처 인덱스 (마지막 9개)
    static_feat_idx = list(range(train_df_scaled.shape[1] - 9, train_df_scaled.shape[1]))
    
    # 데이터셋 생성
    data_window = 100
    step = 1
    target_col = "target_volatility_14d"
    
    X_train = train_df_scaled.values
    y_train = train_df_scaled[target_col].values
    X_test = test_df_scaled.values
    y_test = test_df_scaled[target_col].values
    
    # 교차검증 (선택사항)
    print("교차검증을 수행하시겠습니까? (y/n): ", end="")
    if input().lower() == 'y':
        cross_validate_model(X_train, y_train, static_feat_idx, data_window, step)
    
    # 최종 모델 훈련
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, data_window, step, static_feat_idx)
    test_dataset = MultiStepTimeSeriesDataset(X_test, y_test, data_window, step, static_feat_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 모델 생성 및 훈련
    model = AttentionLSTMModel(
        input_size=X_train.shape[1] - len(static_feat_idx),
        static_feat_dim=len(static_feat_idx)
    ).to(device)
    
    train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=20)
    
    # 모델 평가
    predictions, true_values, date_ranges, rmse, mae = evaluate_model(
        model, test_loader, scaler, test_df, target_col
    )
    
    # 결과 시각화
    visualize_results(predictions, true_values, date_ranges)
    
    # 샘플 가격 시뮬레이션
    if len(predictions) > 0:
        sample_date = date_ranges[0]
        sample_vol = predictions[0]
        sample_price = test_df.loc[sample_date, 'Coffee_Price'] if sample_date in test_df.index else 200
        
        sim_prices = simulate_price_curve(sample_date, sample_vol, sample_price)
        
        plt.figure(figsize=(12, 6))
        plt.plot(sim_prices.index, sim_prices.values, linewidth=2)
        plt.title(f"시뮬레이션된 가격 곡선 (시작일: {sample_date.date()}, 변동성: {sample_vol:.4f})")
        plt.xlabel('날짜')
        plt.ylabel('시뮬레이션 가격')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print(f"🏁 분석 완료 - 최종 성능: RMSE={rmse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()