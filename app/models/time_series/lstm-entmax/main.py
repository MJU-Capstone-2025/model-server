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


if __name__ == '__main__':
    main()