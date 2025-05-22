"""
커피 선물 시장 캘린더 모듈 (Coffee Futures Market Calendar)

이 모듈은 커피 선물 시장의 거래일/휴장일을 처리하는 기능을 제공합니다.
주말(토요일, 일요일) 및 공휴일에는 시장이 열리지 않으므로, 
이러한 날짜에 대해서는 이전 거래일의 가격을 유지하도록 함수를 제공합니다.

주요 기능:
1. 주말 및 휴장일 판별
2. 예측 가격 시리즈 조정 (주말/휴장일에는 전일 가격 유지)
"""

import pandas as pd
from datetime import datetime, timedelta

def is_weekend(date):
    """
    주어진 날짜가 주말(토요일, 일요일)인지 확인합니다.
    
    Args:
        date (datetime or Timestamp): 확인할 날짜
        
    Returns:
        bool: 주말이면 True, 아니면 False
    """
    return date.weekday() >= 5  # 5=토요일, 6=일요일

def is_holiday(date):
    """
    주어진 날짜가 미국 커피 선물 시장의 휴장일인지 확인합니다.
    현재는 주요 미국 공휴일만 포함되어 있습니다.
    
    Args:
        date (datetime or Timestamp): 확인할 날짜
        
    Returns:
        bool: 휴장일이면 True, 아니면 False
    """
    # 연도에 관계없이 반복되는 주요 미국 공휴일 (월/일)
    # 정확한 휴장일 목록은 ICE(Intercontinental Exchange) 공식 캘린더 참조 필요
    us_holidays = [
        (1, 1),    # 새해 (New Year's Day)
        (7, 4),    # 독립기념일 (Independence Day)
        (12, 25),  # 크리스마스 (Christmas Day)
    ]
    
    # 고정 휴일 확인
    if (date.month, date.day) in us_holidays:
        return True
    
    # 추가적인 휴장일 확인 로직은 필요에 따라 확장 가능
    # (예: 마틴 루터 킹 데이, 메모리얼 데이 등)
    
    return False

def is_trading_day(date):
    """
    주어진 날짜가 거래일인지 확인합니다.
    
    Args:
        date (datetime or Timestamp): 확인할 날짜
        
    Returns:
        bool: 거래일이면 True, 아니면 False
    """
    return not (is_weekend(date) or is_holiday(date))

def adjust_forecast_for_market_calendar(forecast_series):
    """
    커피 선물 시장 캘린더에 맞게 예측 가격을 조정합니다.
    주말이나 휴장일에는 이전 거래일의 가격을 유지합니다.
    
    Args:
        forecast_series (pd.Series): 날짜 인덱스를 가진 예측 가격 시리즈
        
    Returns:
        pd.Series: 시장 캘린더에 맞게 조정된 가격 시리즈
    """
    adjusted_forecast = forecast_series.copy()
    dates = forecast_series.index
    
    # 첫 번째 날짜가 주말/휴장일이면, 해당 날짜에 대해서는 원래 예측값 사용
    # (이전 거래일의 가격 정보가 없으므로)
    
    # 두 번째 날짜부터 순회하며 비거래일은 이전 거래일의 가격으로 조정
    for i in range(1, len(dates)):
        current_date = dates[i]
        
        # 주말이나 휴장일이면 이전 가격 사용
        if not is_trading_day(current_date):
            adjusted_forecast[current_date] = adjusted_forecast[dates[i-1]]
    
    return adjusted_forecast
