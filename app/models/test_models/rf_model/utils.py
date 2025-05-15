"""
유틸리티 함수 모음
"""
import pandas as pd
from .config import MARKET_HOLIDAYS

def is_market_closed(date):
    """
    주어진 날짜가 주말이거나 공휴일인지 확인합니다.
    
    Args:
        date: 확인할 날짜
        
    Returns:
        boolean: 휴장일이면 True, 아니면 False
    """
    date = pd.to_datetime(date)
    is_weekend = date.weekday() >= 5  # 5: 토요일, 6: 일요일
    is_holiday = date.strftime('%Y-%m-%d') in MARKET_HOLIDAYS
    return is_weekend or is_holiday
