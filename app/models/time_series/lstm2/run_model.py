import pandas as pd
import numpy as np
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
        data_path = os.path.join(app_dir, 'data', 'input', '거시경제및커피가격통합데이터.csv')


    print(f"⏳ 데이터 로드 중...")

    try:
        df_weather = pd.read_csv(data_path)
        print(f"✅ 날씨 데이터 로드 성공: {df_weather.shape}")
    except Exception as e:
        print(f"❌ 날씨 데이터 로드 실패: {e}")
        raise

    return df_weather

df = load_weather_data()
print(df.head())