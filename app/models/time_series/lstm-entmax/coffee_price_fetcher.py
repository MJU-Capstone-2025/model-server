"""
커피 선물 가격을 가져오는 간단한 모듈 (yfinance 전용)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def get_coffee_price_for_dates(dates):
    """
    특정 날짜들에 대한 커피 선물 가격을 가져옵니다.
    
    Args:
        dates (list or pd.DatetimeIndex): 가격을 가져올 날짜들
    
    Returns:
        dict: {날짜: 가격} 형태의 딕셔너리
    """
    try:
        
        print("Yahoo Finance에서 커피 선물 가격 가져오는 중...")
        
        # 날짜 범위 계산
        dates = pd.to_datetime(dates)
        start_date = dates.min() - timedelta(days=30)  # 여유분 추가
        end_date = dates.max() + timedelta(days=1)
        
        # 커피 선물 데이터 가져오기 (KC=F: Coffee C Futures)
        coffee = yf.Ticker("KC=F")
        hist = coffee.history(start=start_date, end=end_date)
        
        if hist.empty:
            print("커피 선물 데이터를 가져올 수 없습니다.")
            return {}
        
        # 날짜별 가격 딕셔너리 생성
        price_dict = {}
        hist_dates = hist.index.date
        hist_prices = hist['Close'].values
        
        for target_date in dates:
            target_date_only = target_date.date()
            
            # 정확한 날짜 매칭
            if target_date_only in hist_dates:
                idx = list(hist_dates).index(target_date_only)
                price_dict[target_date_only] = hist_prices[idx]
            else:
                # 가장 가까운 이전 날짜의 가격 사용 사용하여 빈 곳 채움움
                available_dates = [d for d in hist_dates if d <= target_date_only]
                if available_dates:
                    closest_date = max(available_dates)
                    idx = list(hist_dates).index(closest_date)
                    price_dict[target_date_only] = hist_prices[idx]
                else:
                    price_dict[target_date_only] = None

                # 오늘 이후는 전부 None으로 변경
                if target_date_only > datetime.now().date():
                    price_dict[target_date_only] = None
        
        valid_count = sum(1 for price in price_dict.values() if price is not None)
        print(f"{valid_count}/{len(dates)}개 날짜의 커피 가격 수집 완료")
        
        return price_dict
        
    except ImportError:
        print("yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance' 실행하세요.")
        return {}
    except Exception as e:
        print(f"커피 가격 가져오기 실패: {e}")
        return {}


def enhance_predictions_with_actual_prices(future_df):
    """
    예측 결과에 실제 커피 가격을 추가합니다.
    
    Args:
        future_df (pd.DataFrame): 예측 결과 DataFrame
    
    Returns:
        pd.DataFrame: 실제 가격이 추가된 DataFrame
    """
    print("예측 결과에 실제 커피 가격 추가 중...")
    
    # 예측 날짜들에 대한 실제 가격 가져오기
    prediction_dates = pd.to_datetime(future_df['Date'])
    price_dict = get_coffee_price_for_dates(prediction_dates)
    
    if not price_dict:
        print("실제 커피 가격을 가져올 수 없어 예측값만 저장합니다.")
        # Actual_Price 컬럼이 없으면 추가
        if 'Actual_Price' not in future_df.columns:
            future_df['Actual_Price'] = None
        return future_df
    
    # DataFrame에 실제 가격 추가
    actual_prices = []
    for date in prediction_dates:
        date_only = date.date()
        actual_prices.append(price_dict.get(date_only, None))
    
    future_df['Actual_Price'] = actual_prices
    
    # 통계 출력
    valid_count = sum(1 for price in actual_prices if price is not None)
    print(f"{valid_count}개 날짜에 실제 커피 가격 추가 완료")
    
    return future_df