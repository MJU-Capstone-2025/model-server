"""
날씨 데이터 로더 모듈

이 모듈은 weather_with_lag 데이터를 로드하기 위한 기능을 제공.
"""

import pandas as pd
import os

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
    season_tag_cols = [col for col in df.columns if 'season_tag' in col]
    until_cols = [col for col in df.columns if 'until' in col]
    
    # 남길 컬럼들
    keep_cols = ['Date', 'Coffee_Price', 'Coffee_Price_Return'] \
        + prectotcorr_cols + season_tag_cols + until_cols
    
    # 남길 컬럼들로 필터링
    df = df[keep_cols]
    
    print(f"✅ PRECTOTCORR 관련 컬럼들만 남기기 성공: {df.shape}")
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