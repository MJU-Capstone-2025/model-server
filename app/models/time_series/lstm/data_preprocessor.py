"""
날씨 데이터 로더 모듈

이 모듈은 weather_with_lag 데이터를 로드하기 위한 기능을 제공.
"""

import pandas as pd
import os
from .model import *
from sklearn.preprocessing import MinMaxScaler

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
        data_path = os.path.join(app_dir, 'data', 'input', 'data.csv')


    print(f"⏳ 데이터 로드 중...")

    try:
        df_weather = pd.read_csv(data_path)
        print(f"✅ 날씨 데이터 로드 성공: {df_weather.shape}")
    except Exception as e:
        print(f"❌ 날씨 데이터 로드 실패: {e}")
        raise

    return df_weather

def remove_lag(df):
    """
    lag 변수를 제거
    """
    lag_cols = [col for col in df.columns if 'lag' in col]
    df = df.drop(columns=lag_cols)
    print(f"✅ lag features 제거 성공: {df.shape}")
    return df

def leave_PRECTOTCORR_columns(df):
    """
    'PRECTOTCORR' 관련 컬럼들,
    'season_tag', 'until' 관련 컬럼들,
    'Date', 'Coffee_Price', 'Coffee_Price_Return' 컬럼만 남기고 나머지 컬럼 제거
    """
    # 'PRECTOTCORR', 'season_tag', 'until' 관련 컬럼들
    prectotcorr_cols = [col for col in df.columns if 'PRECTOTCORR' in col]
    t2m_cols = [col for col in df.columns if 'T2M' in col]
    season_tag_cols = [col for col in df.columns if 'season_tag' in col]
    until_cols = [col for col in df.columns if 'until' in col]
    
    # 남길 컬럼들
    keep_cols = ['Date', 'Coffee_Price', 'Coffee_Price_Return', 
                 'Crude_Oil_Price', 'USD_BRL'] \
        + prectotcorr_cols \
        + until_cols \
        + season_tag_cols \
        # + t2m_cols
    
    # 남길 컬럼들로 필터링
    df = df[keep_cols]

    print(f"✅ PRECTOTCORR 관련 + 실시간 수집 가능한 경제 데이터들만 남기기 성공: {df.shape}")
    return df

def split_data(df, train_ratio=0.8):
    """
    데이터를 train과 test로 분할
    """
    # 데이터 분할
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    print(f"✅ 데이터 분할 성공: {train_data.shape}, {test_data.shape}")
    return train_data, test_data


def add_volatility_features(df, price_col='Coffee_Price', return_col='Coffee_Price_Return'):
    """변동성 관련 파생 피처 추가"""
    
    # 날짜 컬럼을 인덱스로 설정
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 1. 절대 수익률
    df['Abs_Return'] = np.abs(df[return_col])
    
    # 2. n일 변동성
    for n in [5, 10, 20]:
        df[f'Volatility_{n}d'] = df[return_col].rolling(window=n).std()
    
    # 3. 모멘텀 (n일 전 대비 가격 변화)
    for n in [5, 10, 20]:
        df[f'Momentum_{n}d'] = df[price_col] / df[price_col].shift(n) - 1
    
    # 4. 볼린저 밴드 너비
    for n in [20]:
        rolling_mean = df[price_col].rolling(window=n).mean()
        rolling_std = df[price_col].rolling(window=n).std()
        df[f'BB_Width_{n}d'] = (rolling_mean + 2*rolling_std - (rolling_mean - 2*rolling_std)) / rolling_mean
    
    # 5. Z-score (현재 가격이 과거 n일 평균에서 얼마나 떨어져 있는지)
    for n in [20]:
        rolling_mean = df[price_col].rolling(window=n).mean()
        rolling_std = df[price_col].rolling(window=n).std()
        df[f'Z_Score_{n}d'] = (df[price_col] - rolling_mean) / rolling_std
    
    # NaN 값 제거
    df.dropna(inplace=True)
    
    print(f"✅ 변동성 관련 파생 피처 추가 성공: {df.shape}")
    return df

def prepare_data_for_model(train_data, test_data, target='price'):
    """모델 학습을 위한 데이터 준비 (target: price/return)"""
    # 날짜 정보 저장
    train_dates = train_data.index
    test_dates = test_data.index
    # 정규화를 위한 스케일러
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    # 시퀀스 생성 (50일 데이터로 14일 예측)
    seq_length = 50
    pred_length = 14
    X_train, y_train = create_sequences(train_scaled, seq_length, pred_length, target=target, df=train_data)
    X_test, y_test = create_sequences(test_scaled, seq_length, pred_length, target=target, df=test_data)
    print(f"✅ 시퀀스 생성 완료 - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✅ 시퀀스 생성 완료 - X_test: {X_test.shape}, y_test: {y_test.shape}")
    # 데이터셋 및 데이터로더 생성
    train_dataset = TimeSeriesDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TimeSeriesDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler, test_dates[-len(X_test):], seq_length, pred_length

def encode_categorical_features(df):
    """범주형 특성을 원-핫 인코딩"""
    
    # season_tag 컬럼 찾기
    season_tag_cols = [col for col in df.columns if 'season_tag' in col]
    
    if not season_tag_cols:
        print("⚠️ 범주형 특성이 없습니다.")
        return df
    
    # 각 season_tag 컬럼에 대해 원-핫 인코딩 진행
    for col in season_tag_cols:
        # 고유값 출력
        unique_values = df[col].unique()
        
        # 원-핫 인코딩
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        
        # 원본 데이터프레임에 원-핫 인코딩 컬럼 추가
        df = pd.concat([df, dummies], axis=1)
        
        # 원본 범주형 컬럼 제거
        df = df.drop(columns=[col])
    
    print(f"✅ 범주형 특성 인코딩 성공: {df.shape}")
    return df

def create_sequences(data, seq_length, pred_length, target='price', df=None):
    """시계열 데이터 윈도우 생성: seq_length일의 데이터로 pred_length일 예측 (target: price/return)"""
    xs, ys = [], []
    # target 인덱스 결정
    if df is not None:
        columns = list(df.columns)
        price_col_idx = columns.index('Coffee_Price') if 'Coffee_Price' in columns else 0
        return_col_idx = columns.index('Coffee_Price_Return') if 'Coffee_Price_Return' in columns else 1
    else:
        price_col_idx = 0
        return_col_idx = 1
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length)]
        if target == 'price':
            y = data[(i + seq_length):(i + seq_length + pred_length), price_col_idx]
        elif target == 'return':
            y = data[(i + seq_length):(i + seq_length + pred_length), return_col_idx]
        else:
            raise ValueError(f"Unknown target: {target}")
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

