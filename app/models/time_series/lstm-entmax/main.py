"""
커피 가격 예측 모델 메인 실행 파일

Entmax 어텐션을 활용한 LSTM 기반 커피 가격 예측 모델의 전체 파이프라인을 실행합니다.
"""

import pandas as pd
import torch
import torch.nn as nn

# 로컬 모듈 import (절대 import로 변경)
try:
    # 패키지로 실행될 때
    from .utils import get_device
    from .preprocessor import preprocess_data, split_and_scale
    from .dataset import MultiStepTimeSeriesDataset
    from .models import AttentionLSTMModel
    from .trainer import train_model, predict_and_inverse, predict_future, evaluate_and_save
    from .visualizer import plot_loss, plot_prediction
except ImportError:
    # 직접 실행될 때
    from utils import get_device
    from preprocessor import preprocess_data, split_and_scale
    from dataset import MultiStepTimeSeriesDataset
    from models import AttentionLSTMModel
    from trainer import train_model, predict_and_inverse, predict_future, evaluate_and_save
    from visualizer import plot_loss, plot_prediction


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