"""
데이터 전처리 함수들
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    from .data_loader import load_eco_data, load_weather_data
except ImportError:
    from data_loader import load_eco_data, load_weather_data


def preprocess_data():
    """
    원시 데이터를 로드하고 모델 학습에 필요한 피처들을 생성합니다.
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임 (Date가 인덱스로 설정됨)
    """
    # 경제 데이터 로드 및 기본 전처리
    df = load_eco_data()
    df = df.ffill()  # forward fill (이전 값으로 채우기)
    df = df.bfill()  # backward fill (다음 값으로 채우기)
    df = df[['Date', 'Coffee_Price', 'Crude_Oil_Price', 'USD_BRL']]
    
    # 수익률 및 변동성 피처 생성
    df['Coffee_Price_Return'] = df['Coffee_Price'].pct_change()
    df['abs_return'] = df['Coffee_Price_Return'].abs()
    df['volatility_5d'] = df['Coffee_Price_Return'].rolling(window=5).std()
    df['volatility_10d'] = df['Coffee_Price_Return'].rolling(window=10).std()
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']
    
    # 모멘텀 피처 생성
    df['momentum_1d'] = df['Coffee_Price'].diff(1)
    df['momentum_3d'] = df['Coffee_Price'].diff(3)
    df['momentum_5d'] = df['Coffee_Price'] - df['Coffee_Price'].shift(5)
    
    # 볼린저 밴드 및 Z-스코어 피처
    rolling_mean = df['Coffee_Price'].rolling(window=20).mean()
    rolling_std = df['Coffee_Price'].rolling(window=20).std()
    df['bollinger_width'] = (2 * rolling_std) / rolling_mean
    df['return_zscore'] = (df['Coffee_Price_Return'] - df['Coffee_Price_Return'].rolling(20).mean()) / \
                          (df['Coffee_Price_Return'].rolling(20).std() + 1e-6)
    
    # 날씨 데이터 병합
    weather_data = load_weather_data()
    columns_to_drop = ['Coffee_Price', 'Coffee_Price_Return', 'Crude_Oil_Price', 
                      'USD_KRW', 'USD_BRL', 'USD_COP']
    weather_data.drop(columns=[c for c in columns_to_drop if c in weather_data.columns], inplace=True)
    df = pd.merge(df, weather_data, on='Date', how='inner')
    
    # 최종 정리
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df


def split_and_scale(df, target_col, static_feat_count, window, horizon, step):
    """
    데이터를 훈련/테스트로 분할하고 정규화를 수행합니다.
    
    Args:
        df (pd.DataFrame): 전처리된 데이터프레임
        target_col (str): 예측 대상 컬럼명
        static_feat_count (int): 정적 피처의 개수
        window (int): 입력 시퀀스 길이
        horizon (int): 예측 구간 길이
        step (int): 슬라이딩 윈도우 스텝 크기
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx)
    """
    n = len(df)
    train_size = int(n * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # 정규화 (타겟 변수 제외)
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), 
        columns=train_df.columns, 
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), 
        columns=test_df.columns, 
        index=test_df.index
    )
    
    # 타겟 변수는 원본 값 유지
    train_scaled[target_col] = train_df[target_col]
    test_scaled[target_col] = test_df[target_col]
    
    # 피처와 타겟 분리
    y_train = train_scaled[target_col].values
    y_test = test_scaled[target_col].values
    X_train = train_scaled.drop(columns=[target_col]).values
    X_test = test_df.drop(columns=[target_col]).values
    
    # 정적 피처 인덱스 설정
    static_feat_idx = list(range(X_train.shape[1] - static_feat_count, X_train.shape[1]))
    
    return X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx 