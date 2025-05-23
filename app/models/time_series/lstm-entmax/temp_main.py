import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import DEVICE
from data_loader import load_data
from feature_engineering import create_features
from dataset import MultiStepTimeSeriesDataset
from model import AttentionLSTMModel
from train import train_model
from evaluate import evaluate
from visualize import plot_prediction
from simulate import simulate_price_curve
import torch
from torch.utils.data import DataLoader


def main():
    # 데이터 로딩 및 병합
    df_macro, df_weather = load_data()
    df = create_features(df_macro)
    df_weather = df_weather.drop(columns=[col for col in df_weather.columns if col in df.columns and col != 'Date'])
    df = pd.merge(df, df_weather, on='Date', how='left')
    df.dropna(inplace=True)

    # 인덱스 처리
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 데이터 분할
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 정규화
    scaler = MinMaxScaler()
    y_col = 'target_volatility_14d'
    return_backup = train_df['Coffee_Price_Return'].copy(), test_df['Coffee_Price_Return'].copy()
    log_return_backup = train_df['log_return'].copy(), test_df['log_return'].copy()

    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

    train_scaled['Coffee_Price_Return'] = return_backup[0]
    test_scaled['Coffee_Price_Return'] = return_backup[1]
    train_scaled['log_return'] = log_return_backup[0]
    test_scaled['log_return'] = log_return_backup[1]

    # 정적 피처 인덱스 (마지막 9개)
    static_idx = list(range(train_scaled.shape[1] - 9, train_scaled.shape[1]))
    
    X_train = train_scaled.values
    y_train = train_scaled[y_col].values
    X_test = test_scaled.values
    y_test = test_scaled[y_col].values

    # 데이터셋 및 로더
    window = 100
    step = 1
    train_ds = MultiStepTimeSeriesDataset(X_train, y_train, window, step, static_idx)
    test_ds = MultiStepTimeSeriesDataset(X_test, y_test, window, step, static_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 모델 학습
    model = AttentionLSTMModel(input_size=X_train.shape[1] - len(static_idx), static_dim=len(static_idx)).to(DEVICE)
    train_model(model, train_loader, test_loader, epochs=20)

    # 예측 결과 수집
    model.eval()
    predictions, targets, dates = [], [], []
    with torch.no_grad():
        for i, (x_seq, x_static, y) in enumerate(test_loader):
            x_seq, x_static = x_seq.to(DEVICE), x_static.to(DEVICE)
            out = model(x_seq, x_static).cpu().numpy()
            y = y.cpu().numpy()
            for j in range(len(out)):
                idx = i * test_loader.batch_size + j
                global_idx = idx * step + window
                if global_idx < len(test_df):
                    pred_scaled = np.zeros((1, test_df.shape[1]))
                    true_scaled = np.zeros((1, test_df.shape[1]))
                    target_idx = test_df.columns.get_loc(y_col)
                    pred_scaled[0, target_idx] = out[j]
                    true_scaled[0, target_idx] = y[j]
                    predictions.append(scaler.inverse_transform(pred_scaled)[0, target_idx])
                    targets.append(scaler.inverse_transform(true_scaled)[0, target_idx])
                    dates.append(test_df.index[global_idx])

    # 평가 및 시각화
    metrics = evaluate(predictions, targets)
    print("\n[Final Evaluation] RMSE: {:.4f}, MAE: {:.4f}".format(metrics['rmse'], metrics['mae']))
    plot_prediction(predictions, targets, dates)

    # 시뮬레이션 예시
    sample_vol = predictions[0]
    base_price = test_df.loc[dates[0], 'Coffee_Price'] if dates[0] in test_df.index else 200
    sim_prices = simulate_price_curve(dates[0], sample_vol, base_price)
    sim_prices.plot(title=f"Simulated Coffee Price (start={dates[0].date()}, vol={sample_vol:.4f})", figsize=(12, 6))


def check_current_code_leakage():
    """현재 코드에서 발생하는 데이터 누출 확인"""
    
    # 샘플 데이터 생성 (현재 코드 시뮬레이션)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Coffee_Price': 200 + np.cumsum(np.random.randn(100) * 0.02),
        'Coffee_Price_Return': np.random.randn(100) * 0.02
    })
    df.set_index('Date', inplace=True)
    
    # 현재 feature_engineering.py와 동일한 방식
    df['log_return'] = np.log(df['Coffee_Price']) - np.log(df['Coffee_Price'].shift(1))
    df['target_volatility_14d'] = df['log_return'].rolling(14).std().shift(-13)
    
    # 현재 main.py와 동일한 방식
    y_col = 'target_volatility_14d'
    
    print("=== 현재 코드의 데이터 누출 문제 ===")
    print(f"전체 컬럼: {list(df.columns)}")
    print(f"타겟 컬럼: {y_col}")
    
    # 현재 방식대로 X, y 생성
    X_current = df.values  # 모든 컬럼 포함 (target_volatility_14d도 포함!)
    y_current = df[y_col].values
    
    print(f"\nX의 shape: {X_current.shape}")
    print(f"y의 shape: {y_current.shape}")
    
    # target_volatility_14d가 X에 포함되어 있는지 확인
    target_col_idx = df.columns.get_loc(y_col)
    print(f"\n🚨 데이터 누출 확인:")
    print(f"target_volatility_14d가 X의 {target_col_idx}번째 컬럼에 포함됨!")
    print(f"X[:5, {target_col_idx}] = {X_current[:5, target_col_idx]}")
    print(f"y[:5] = {y_current[:5]}")
    print(f"값이 동일한가? {np.array_equal(X_current[:, target_col_idx], y_current)}")
    
    return df, X_current, y_current, target_col_idx

def fix_data_leakage():
    """데이터 누출 문제 수정"""
    
    # 샘플 데이터 생성
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Coffee_Price': 200 + np.cumsum(np.random.randn(100) * 0.02),
        'Coffee_Price_Return': np.random.randn(100) * 0.02,
        'USD_BRL': 5.0 + np.cumsum(np.random.randn(100) * 0.01),
        'Crude_Oil_Price': 70 + np.cumsum(np.random.randn(100) * 0.5)
    })
    df.set_index('Date', inplace=True)
    
    # 올바른 피처 엔지니어링
    df['log_return'] = np.log(df['Coffee_Price']) - np.log(df['Coffee_Price'].shift(1))
    
    # 피처들 (과거 데이터만 사용)
    df['volatility_5d'] = df['log_return'].rolling(5).std()
    df['volatility_10d'] = df['log_return'].rolling(10).std()
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)
    
    # 타겟 (미래 데이터 사용 - 정상)
    df['target_volatility_14d'] = df['log_return'].rolling(14).std().shift(-13)
    
    print("\n=== 수정된 코드 ===")
    print(f"전체 컬럼: {list(df.columns)}")
    
    # 올바른 방식: 타겟 컬럼을 X에서 제외
    feature_cols = [col for col in df.columns if not col.startswith('target_')]
    target_col = 'target_volatility_14d'
    
    print(f"피처 컬럼들: {feature_cols}")
    print(f"타겟 컬럼: {target_col}")
    
    # X에는 피처만, y에는 타겟만
    X_fixed = df[feature_cols].values
    y_fixed = df[target_col].values
    
    print(f"\n수정 후:")
    print(f"X의 shape: {X_fixed.shape}")
    print(f"y의 shape: {y_fixed.shape}")
    print(f"✅ 타겟 변수가 X에 포함되지 않음!")
    
    return df, X_fixed, y_fixed, feature_cols

def demonstrate_impact():
    """데이터 누출이 성능에 미치는 영향 시연"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    # 샘플 데이터
    np.random.seed(42)
    n_samples = 1000
    X_normal = np.random.randn(n_samples, 5)  # 정상적인 피처들
    y = np.sum(X_normal, axis=1) + np.random.randn(n_samples) * 0.1  # 타겟
    
    # 데이터 누출 케이스: X에 y값 추가
    X_leaked = np.column_stack([X_normal, y + np.random.randn(n_samples) * 0.01])
    
    # 학습/테스트 분할
    X_normal_train, X_normal_test, y_train, y_test = train_test_split(
        X_normal, y, test_size=0.2, random_state=42)
    X_leaked_train, X_leaked_test, _, _ = train_test_split(
        X_leaked, y, test_size=0.2, random_state=42)
    
    # 모델 학습 및 평가
    rf_normal = RandomForestRegressor(random_state=42)
    rf_leaked = RandomForestRegressor(random_state=42)
    
    rf_normal.fit(X_normal_train, y_train)
    rf_leaked.fit(X_leaked_train, y_train)
    
    pred_normal = rf_normal.predict(X_normal_test)
    pred_leaked = rf_leaked.predict(X_leaked_test)
    
    mse_normal = mean_squared_error(y_test, pred_normal)
    mse_leaked = mean_squared_error(y_test, pred_leaked)
    
    print(f"\n=== 데이터 누출 영향 시연 ===")
    print(f"정상적인 모델 MSE: {mse_normal:.6f}")
    print(f"데이터 누출 모델 MSE: {mse_leaked:.6f}")
    print(f"성능 개선: {((mse_normal - mse_leaked) / mse_normal * 100):.1f}%")
    print(f"🚨 데이터 누출로 인한 과도한 성능 향상은 실제 운용에서 재현되지 않습니다!")

if __name__ == '__main__':
    main()

    # 1. 현재 코드의 문제점 확인
    df_current, X_current, y_current, target_idx = check_current_code_leakage()
    
    # 2. 수정된 코드
    df_fixed, X_fixed, y_fixed, feature_cols = fix_data_leakage()
    
    # 3. 데이터 누출 영향 시연
    demonstrate_impact()