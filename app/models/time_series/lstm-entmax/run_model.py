"""
커피 가격 예측 모델 메인 실행 파일

Entmax 어텐션을 활용한 LSTM 기반 커피 가격 예측 모델의 전체 파이프라인을 실행합니다.
"""

import pandas as pd
import torch
import torch.nn as nn
import argparse

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


def parse_arguments():
    """
    명령행 인수를 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인수들
    """
    parser = argparse.ArgumentParser(description='커피 가격 예측 모델 실행')
    
    # 데이터 관련 파라미터
    parser.add_argument('--window', type=int, default=100, 
                       help='입력 시퀀스 길이 (기본값: 100)')
    parser.add_argument('--horizon', type=int, default=14, 
                       help='예측 구간 길이 (기본값: 14)')
    parser.add_argument('--step', type=int, default=1, 
                       help='슬라이딩 윈도우 스텝 크기 (기본값: 1)')
    
    # 모델 관련 파라미터
    parser.add_argument('--hidden_size', type=int, default=64, 
                       help='LSTM 은닉 상태 크기 (기본값: 64)')
    parser.add_argument('--num_layers', type=int, default=2, 
                       help='LSTM 레이어 수 (기본값: 2)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='드롭아웃 비율 (기본값: 0.1)')
    
    # 학습 관련 파라미터
    parser.add_argument('--epochs', type=int, default=5, 
                       help='학습 에포크 수 (기본값: 5)')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='훈련 배치 크기 (기본값: 64)')
    parser.add_argument('--test_batch_size', type=int, default=32, 
                       help='테스트 배치 크기 (기본값: 32)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='학습률 (기본값: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.2, 
                       help='방향성 손실 가중치 (기본값: 0.2)')
    parser.add_argument('--beta', type=float, default=0.1, 
                       help='분산 손실 가중치 (기본값: 0.1)')
    
    # 기타 파라미터
    parser.add_argument('--no_plot', action='store_true', 
                       help='시각화 생략')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], 
                       help='사용할 장치 (기본값: auto)')
    parser.add_argument('--examples', action='store_true', 
                       help='사용 예시 출력 후 종료')
    
    args = parser.parse_args()
    
    # 사용 예시 출력 후 종료
    if args.examples:
        print_usage_examples()
        exit(0)
    
    return args


def print_usage_examples():
    """
    사용 예시를 출력합니다.
    """
    print("\n=== 사용 예시 ===")
    print("1. 기본 설정으로 실행:")
    print("   python run_model.py")
    print()
    print("2. 에포크 수와 윈도우 크기 변경:")
    print("   python run_model.py --epochs 10 --window 50")
    print()
    print("3. 모델 구조 변경:")
    print("   python run_model.py --hidden_size 128 --num_layers 3 --dropout 0.2")
    print()
    print("4. 학습 설정 변경:")
    print("   python run_model.py --batch_size 32 --lr 0.0001 --alpha 0.3 --beta 0.15")
    print()
    print("5. 시각화 없이 실행:")
    print("   python run_model.py --no_plot")
    print()
    print("6. CPU 강제 사용:")
    print("   python run_model.py --device cpu")
    print()
    print("7. 전체 옵션 확인:")
    print("   python run_model.py --help")
    print()


def main():
    """
    커피 가격 예측 모델의 전체 파이프라인을 실행합니다.
    
    주요 단계:
    1. 명령행 인수 파싱
    2. 데이터 로드 및 전처리
    3. 데이터 분할 및 정규화
    4. 데이터셋 및 데이터로더 생성
    5. 모델 초기화 및 학습
    6. 테스트 구간 예측
    7. 미래 구간 예측
    8. 결과 평가 및 저장
    """
    # 1. 명령행 인수 파싱
    args = parse_arguments()
    
    print("=== 커피 가격 예측 모델 시작 ===")
    print(f"설정된 하이퍼파라미터:")
    print(f"  - 윈도우 크기: {args.window}")
    print(f"  - 예측 구간: {args.horizon}")
    print(f"  - 에폭 수: {args.epochs}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - 학습률: {args.lr}")
    print(f"  - 은닉 크기: {args.hidden_size}")
    print(f"  - LSTM 레이어: {args.num_layers}")
    
    # 2. 환경 설정
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    print(f"사용 장치: {device}")
    
    # 3. 데이터 전처리
    print("데이터 전처리 중...")
    df = preprocess_data()
    print(f"전처리된 데이터 크기: {df.shape}")
    
    # 4. 고정 파라미터 설정
    target_col = "Coffee_Price_Return"
    price_col = "Coffee_Price"
    static_feat_count = 9
    
    # 5. 데이터 분할 및 정규화
    print("데이터 분할 및 정규화 중...")
    X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx = split_and_scale(
        df, target_col, static_feat_count, args.window, args.horizon, args.step
    )
    
    # 6. 데이터셋 및 데이터로더 생성
    train_dataset = MultiStepTimeSeriesDataset(X_train, y_train, args.window, args.horizon, args.step, static_feat_idx)
    test_dataset = MultiStepTimeSeriesDataset(X_test, y_test, args.window, args.horizon, args.step, static_feat_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    
    # 7. 모델 초기화
    x_seq, x_static, _ = train_dataset[0]
    input_size = x_seq.shape[1]
    static_feat_dim = x_static.shape[0]
    
    model = AttentionLSTMModel(
        input_size=input_size, 
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        target_size=args.horizon, 
        dropout=args.dropout,
        static_feat_dim=static_feat_dim
    ).to(device)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. 학습 설정
    base_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10)
    
    # 9. 모델 학습
    print("모델 학습 시작...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, base_criterion, optimizer, scheduler, 
        args.epochs, args.alpha, args.beta, device
    )
    
    # 10. 학습 곡선 시각화
    if not args.no_plot:
        plot_loss(train_losses, test_losses)
    
    # 11. 테스트 구간 예측
    print("테스트 구간 예측 중...")
    forecast_all, predictions = predict_and_inverse(
        model, test_loader, scaler, train_df, test_df, df, target_col, 
        price_col, args.window, args.horizon, args.step, static_feat_idx
    )
    
    # 12. 테스트 구간 결과 시각화
    if not args.no_plot:
        plot_prediction(df, forecast_all, start=pd.to_datetime('2023-07-01'), end=pd.to_datetime('2025-04-01'))
    
    # 13. 미래 구간 예측
    print("미래 구간 예측 중...")
    future_price_series, future_dates, price_future = predict_future(
        model, test_df, train_df, scaler, static_feat_idx, args.window, 
        args.horizon, price_col, target_col
    )
    
    # 14. 전체 결과 시각화 (테스트 + 미래)
    if not args.no_plot:
        plot_prediction(
            df, forecast_all, 
            start=pd.to_datetime('2023-07-01'), 
            end=future_price_series.index[-1], 
            future_series=future_price_series
        )
    
    # 15. 결과 평가 및 저장
    print("결과 평가 및 저장 중...")
    evaluate_and_save(df, forecast_all, predictions, price_col, future_dates, price_future)
    
    print("=== 커피 가격 예측 모델 완료 ===")


if __name__ == "__main__":
    main() 