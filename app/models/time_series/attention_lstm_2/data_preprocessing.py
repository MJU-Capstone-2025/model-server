"""
데이터 전처리 모듈 (Data Preprocessing Module)

이 모듈은 커피 가격 예측 모델에 사용되는 데이터 전처리 기능을 포함하고 있다.
주요 기능:
1. 거시경제 및 기후 데이터 로딩
2. 데이터 전처리 (결측치 처리, 날짜 인덱스 설정 등)
3. 변동성 관련 피처 생성 (업데이트된 버전)
4. 데이터 스케일링 - 수익률 정보 보존 방식
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
    주가 변동성 관련 피처를 추가한다.
    
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
    
    # 추가된 변동성 특성 (버전 2)
    df['momentum_1d'] = df['Coffee_Price'].diff(1)  # 1일 가격 변화량
    df['momentum_3d'] = df['Coffee_Price'].diff(3)  # 3일 가격 변화량
    df['volatility_ratio'] = df['volatility_5d'] / df['volatility_10d']  # 단기/중기 변동성 비율
    
    return df


def load_and_prepare_data(macro_data_path, climate_data_path):
    """
    데이터를 로드하고 전처리한다.
    
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
    df = pd.merge(df, we, on='Date', how='left') # -> left join으로 2023년 데이터까지만 사용
    
    # 결측치 제거 및 날짜 인덱스 설정
    df = df.dropna() # -> 결측치를 제거한다
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df


def train_test_split(df, train_size=0.8):
    """
    데이터를 학습 및 테스트 세트로 분할한다.
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        train_size (float): 학습 데이터 비율 (0-1 사이)
        
    Returns:
        tuple: (train_df, test_df) 형태의 튜플
    """
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"test_df start date: {test_df.index[0]}")
    print(f"test_df end date: {test_df.index[-1]}")
    return train_df, test_df


def scale_data(train_df, test_df=None, preserve_return=True, preserve_price=False):
    """
    학습 데이터를 스케일링한다. Coffee_Price_Return 특성은 원본값 보존이 가능하다.
    Coffee_Price 특성도 원본값으로 보존할 수 있다.
    
    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        test_df (pd.DataFrame, optional): 테스트 데이터프레임
        preserve_return (bool): Coffee_Price_Return 특성을 원본값으로 보존할지 여부
        preserve_price (bool): Coffee_Price 특성을 원본값으로 보존할지 여부
        
    Returns:
        tuple: 스케일링된 데이터와 스케일러를 포함한 튜플
        test_df가 제공된 경우: (scaled_train_df, scaled_test_df, scaler)
        아닌 경우: (scaled_train_df, scaler)
    """
    scaler = MinMaxScaler()
    
    # 원본 데이터 보존
    if preserve_return and 'Coffee_Price_Return' in train_df.columns:
        return_train = train_df["Coffee_Price_Return"].copy()
    if preserve_price and 'Coffee_Price' in train_df.columns:
        price_train = train_df["Coffee_Price"].copy()
        
    # 훈련 데이터 스케일링
    scaled_train_df = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index
    )
    
    # 원본값 복원
    if preserve_return and 'Coffee_Price_Return' in train_df.columns:
        scaled_train_df["Coffee_Price_Return"] = return_train
    if preserve_price and 'Coffee_Price' in train_df.columns:
        scaled_train_df["Coffee_Price"] = price_train
    
    if test_df is not None:
        # 테스트 데이터도 있는 경우
        if preserve_return and 'Coffee_Price_Return' in test_df.columns:
            return_test = test_df["Coffee_Price_Return"].copy()
        if preserve_price and 'Coffee_Price' in test_df.columns:
            price_test = test_df["Coffee_Price"].copy()
            
        scaled_test_df = pd.DataFrame(
            scaler.transform(test_df),
            columns=test_df.columns,
            index=test_df.index
        )
        
        # 테스트 원본값 복원
        if preserve_return and 'Coffee_Price_Return' in test_df.columns:
            scaled_test_df["Coffee_Price_Return"] = return_test
        if preserve_price and 'Coffee_Price' in test_df.columns:
            scaled_test_df["Coffee_Price"] = price_test
            
        return scaled_train_df, scaled_test_df, scaler
    
    return scaled_train_df, scaler


def scale_data_except_price(train_df, test_df=None):
    """
    가격(Coffee_Price)을 제외한 모든 특성만 스케일링한다.
    가격은 원본 그대로 유지하여 모델이 실제 가격 스케일에서 학습하도록 한다.
    
    Args:
        train_df (pd.DataFrame): 학습 데이터프레임
        test_df (pd.DataFrame, optional): 테스트 데이터프레임
        
    Returns:
        tuple: 스케일링된 데이터와 스케일러를 포함한 튜플
                test_df가 제공된 경우: (scaled_train_df, scaled_test_df, scaler)
                아닌 경우: (scaled_train_df, scaler)
    """
    # 가격 컬럼만 제외한 스케일링
    return scale_data(train_df, test_df, preserve_return=True, preserve_price=True)