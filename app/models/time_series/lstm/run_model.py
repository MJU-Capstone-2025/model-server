"""
LSTM 모델 메인 모듈

"""

import os
import sys
import pandas as pd

# 패키지 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if app_dir not in sys.path:
    sys.path.append(app_dir)

# 패키지 구조에 맞는 상대 import만 사용
from .data_preprocessor import *

def main():
    """
    메인 함수.
    """
    # 1. 날씨 데이터 로드
    weather_data = load_weather_data()
    weather_data = remove_lag(weather_data)
    
    # 2. 데이터 전처리
    weather_data = leave_PRECTOTCORR_columns(weather_data) # 기후 데이터 중에서 강수량 + 필요한 컬럼만 남김
    
    # 3. train/test split
    train_data, test_data = split_data(weather_data, train_ratio=0.8) # 80% train, 20% test
    
    return weather_data

if __name__ == "__main__":
    # 데이터 로드 실행
    weather_data = main()