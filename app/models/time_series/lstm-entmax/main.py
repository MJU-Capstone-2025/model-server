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

    # 피처와 타겟 분리
    target_col = 'target_volatility_14d'
    
    # 타겟으로 시작하는 모든 컬럼 찾기
    target_cols = [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    print(f"타겟 컬럼들: {target_cols}")
    print(f"피처 컬럼 수: {len(feature_cols)}")
    print(f"사용할 타겟: {target_col}")
    
    # 데이터 분할
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 피처만 정규화
    scaler = MinMaxScaler()
    
    # 피처 데이터만 추출하여 정규화
    train_features = train_df[feature_cols]
    test_features = test_df[feature_cols]
    
    # 타겟 데이터 별도 저장 (정규화 안함)
    train_target = train_df[target_col]
    test_target = test_df[target_col]
    
    # 피처만 정규화
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features), 
        columns=feature_cols, 
        index=train_features.index
    )
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features), 
        columns=feature_cols, 
        index=test_features.index
    )
    
    print(f"정규화 전 피처 shape: {train_features.shape}")
    print(f"정규화 후 피처 shape: {train_features_scaled.shape}")

    # 정적 피처 인덱스 (마지막 9개라고 가정)
    # 피처 컬럼 기준으로 정적 인덱스 계산
    static_feature_names = [col for col in feature_cols if any(
        keyword in col.lower() for keyword in ['season', 'harvest', 'climate', 'country']
    )]
    
    if len(static_feature_names) == 0:
        # 정적 피처가 명시적으로 없다면 마지막 몇 개를 정적으로 가정
        static_idx = list(range(max(0, len(feature_cols) - 9), len(feature_cols)))
    else:
        static_idx = [feature_cols.index(col) for col in static_feature_names if col in feature_cols]
    
    print(f"정적 피처 인덱스: {static_idx}")
    
    # X, y 데이터 준비
    X_train = train_features_scaled.values
    y_train = train_target.values
    X_test = test_features_scaled.values
    y_test = test_target.values
    
    print(f"최종 X_train shape: {X_train.shape}")
    print(f"최종 y_train shape: {y_train.shape}")
    
    # ✅ 데이터 누출 검증
    print(f"\n=== 데이터 누출 검증 ===")
    print(f"피처에 타겟 포함 여부: {target_col in feature_cols}")
    print(f"✅ 데이터 누출 없음!" if target_col not in feature_cols else "🚨 데이터 누출 발견!")

    # 데이터셋 및 로더
    window = 100
    step = 1
    train_ds = MultiStepTimeSeriesDataset(X_train, y_train, window, step, static_idx)
    test_ds = MultiStepTimeSeriesDataset(X_test, y_test, window, step, static_idx)
    
    # shuffle=False로 시계열 순서 유지
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 모델 학습
    input_size = X_train.shape[1] - len(static_idx)  # 시계열 피처 개수
    static_dim = len(static_idx)  # 정적 피처 개수
    
    model = AttentionLSTMModel(
        input_size=input_size, 
        static_dim=static_dim
    ).to(DEVICE)
    
    print(f"모델 입력 크기 - 시계열: {input_size}, 정적: {static_dim}")
    
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
                    # 타겟은 별도 정규화하지 않았으므로 그대로 사용
                    predictions.append(out[j])
                    targets.append(y[j])
                    dates.append(test_df.index[global_idx])

    # NaN 값 제거
    valid_indices = []
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if not (np.isnan(pred) or np.isnan(target)):
            valid_indices.append(i)
    
    predictions = [predictions[i] for i in valid_indices]
    targets = [targets[i] for i in valid_indices]
    dates = [dates[i] for i in valid_indices]
    
    print(f"유효한 예측 개수: {len(predictions)}")

    # 평가 및 시각화
    if len(predictions) > 0:
        metrics = evaluate(predictions, targets)
        print("\n[Final Evaluation] RMSE: {:.4f}, MAE: {:.4f}".format(metrics['rmse'], metrics['mae']))
        plot_prediction(predictions, targets, dates)

        # 시뮬레이션 예시
        if len(predictions) > 0:
            sample_vol = predictions[0]
            base_price = test_df.loc[dates[0], 'Coffee_Price'] if dates[0] in test_df.index else 200
            sim_prices = simulate_price_curve(dates[0], sample_vol, base_price)
            sim_prices.plot(title=f"Simulated Coffee Price (start={dates[0].date()}, vol={sample_vol:.4f})", figsize=(12, 6))
    else:
        print("⚠️ 유효한 예측 결과가 없습니다. 데이터를 확인해주세요.")


if __name__ == '__main__':
    main()