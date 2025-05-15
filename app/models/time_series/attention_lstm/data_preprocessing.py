"""
데이터 전처리 모듈 (Data Preprocessing Module)

이 모듈은 커피 가격 예측 모델에 사용되는 데이터 전처리 기능을 포함하고 있습니다.
주요 기능:
1. 거시경제 및 기후 데이터 로딩
2. 데이터 전처리 (결측치 처리, 날짜 인덱스 설정 등)
3. 변동성 관련 피처 생성
4. 데이터 스케일링
5. 학습/테스트 데이터 분할

사용 예시:
    from data_preprocessing import load_and_prepare_data, train_test_split, scale_data
    
    # 데이터 로드 및 전처리
    df = load_and_prepare_data(macro_data_path, climate_data_path)
    
    # 학습/테스트 분할
    train_df, test_df = train_test_split(df)
    
    # 데이터 스케일링
    scaled_train_df, scaler = scale_data(train_df)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_volatility_features(df):
    """
    주가 변동성 관련 피처를 추가합니다.
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: 변동성 피처가 추가된 데이터프레임
    """
    # 절대 수익률
    df['abs_return'] = df['Coffee_Price_Return'].abs()

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
    return df


def load_and_prepare_data(macro_data_path, climate_data_path):
    """
    데이터를 로드하고 전처리합니다.
    
    Args:
        macro_data_path (str): 거시경제 데이터 파일 경로
        climate_data_path (str): 기후 데이터 파일 경로
        
    Returns:
        pd.DataFrame: 전처리된 통합 데이터프레임
    """
    # 거시경제 데이터 로딩
    df = pd.read_csv(macro_data_path)
    
    # 변동성 피처 추가
    df = add_volatility_features(df)
    
    # 기후 데이터 로딩 및 병합
    we = pd.read_csv(climate_data_path)
    we.drop(columns=['Coffee_Price'], inplace=True, errors='ignore')
    df = pd.merge(df, we, on='Date', how='left')
    
    # 결측치 제거 및 날짜 인덱스 설정
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df


def train_test_split(df, train_size=0.8):
    """
    데이터를 학습 및 테스트 세트로 분할합니다.
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        train_size (float): 학습 데이터 비율 (0-1 사이)
        
    Returns:
        tuple: (train_df, test_df) 형태의 튜플
    """
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def scale_data(train_df):
    """
    학습 데이터를 스케일링합니다.
    
    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        
    Returns:
        tuple: (scaled_train_df, scaler) 형태의 튜플
    """
    scaler = MinMaxScaler()
    scaled_train_df = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )
    return scaled_train_df, scaler
