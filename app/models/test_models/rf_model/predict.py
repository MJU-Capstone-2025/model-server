"""
예측 및 결과 시각화 관련 기능
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# 현재 모듈의 디렉토리를 가져옵니다
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import TRUE_PRICE_PATH, PREDICT_DAYS
from utils import is_market_closed
from data import preprocess_data, generate_future_rows_with_lag

def predict_future_prices(pipe, full_data, cat_cols, num_cols, days=57):
    """
    미래 가격을 예측하고 실제 가격으로 보정하여 저장합니다.
    """
    # 오늘 날짜 설정 및 파일명 생성
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)  # 어제 날짜 계산
    today_str = today.strftime('%Y%m%d')
    prediction_filename = f'./data/output/prediction_{today_str}.csv'
    print(f"📅 오늘 날짜: {today.date()}")
    print(f"📅 어제 날짜: {yesterday.date()} (시차 보정 시작일)")
    
    # 실제 가격 데이터 로드
    try:
        true_prices = pd.read_csv(TRUE_PRICE_PATH)
        true_prices['Date'] = pd.to_datetime(true_prices['Date'])
    except FileNotFoundError:
        print(f"⚠️ 실제 가격 파일 {TRUE_PRICE_PATH}를 찾을 수 없습니다.")
        true_prices = pd.DataFrame(columns=['Date', 'True_Price'])
        true_prices['Date'] = pd.to_datetime(true_prices['Date'])
    
    # 마지막 실제 가격 찾기 (어제 또는 그 이전의 가장 최근 데이터)
    if not true_prices.empty and 'True_Price' in true_prices.columns:
        # 어제 이전의 가격 데이터만 필터링
        past_prices = true_prices[true_prices['Date'] <= yesterday]
        valid_true_prices = past_prices[past_prices['True_Price'].notna()]
        
        if not valid_true_prices.empty:
            latest_date = valid_true_prices['Date'].max()
            latest_price = valid_true_prices.loc[valid_true_prices['Date'] == latest_date, 'True_Price'].iloc[0]
            print(f"✅ {latest_date.date()} 기준 실제 가격: {latest_price:.2f} USD")
            
            # 마지막 실제 가격과 어제 날짜 간의 차이 확인
            date_gap = (yesterday - latest_date).days
            if date_gap > 5:  # 5일 이상 차이나면 경고
                print(f"⚠️ 경고: 최근 실제 가격이 {date_gap}일 전 데이터입니다.")
        else:
            latest_date = None
            latest_price = None
            print("⚠️ 실제 가격 데이터가 없습니다.")
    else:
        latest_date = None
        latest_price = None
        print("⚠️ 실제 가격 데이터가 없습니다.")
    
    # 예측 시작일: 어제부터 시작 (시차 보정)
    start_date = yesterday
    print(f"📈 예측 시작일: {start_date.date()} (어제부터 시작)")
    
    # 예측
    fut = generate_future_rows_with_lag(full_data, start_date, days)
    fut = preprocess_data(fut)
    Xf = fut[cat_cols + num_cols]
    fut['Prediction_Price'] = pipe.predict(Xf)
    
    # 예측 결과 데이터프레임 준비
    pred = fut[['Date', 'Prediction_Price']]
    
    # 예측값 보정
    if latest_price is not None:
        future_preds = pred.copy()
        if not future_preds.empty:
            first_pred = future_preds['Prediction_Price'].iloc[0]
            
            # 차이값 계산 (실제 가격 - 예측값)
            difference = latest_price - first_pred
            
            # 모든 미래 예측값에 차이값을 더함
            pred['Prediction_Price'] += difference
            
            # 급격한 변화 완화
            prev_price = latest_price
            for idx, row in pred.iterrows():
                current_price = row['Prediction_Price']
                price_change = abs(current_price - prev_price)
                
                if price_change > 20:  # $20 이상 변화하는 경우
                    # 변화폭을 30%로 줄임
                    direction = 1 if current_price > prev_price else -1
                    adjusted_change = direction * (price_change * 0.3)
                    adjusted_price = prev_price + adjusted_change
                    pred.loc[idx, 'Prediction_Price'] = adjusted_price
                    print(f"⚠️ 급격한 변화 감지 (${price_change:.1f}) - {row['Date'].date()}: {current_price:.2f} → {adjusted_price:.2f}")
                
                prev_price = pred.loc[idx, 'Prediction_Price']
            
            print(f"✅ 미래 예측값 보정:")
            print(f"- 최근 실제 가격: {latest_price:.2f} USD")
            print(f"- 첫 예측 가격: {first_pred:.2f} USD")
            print(f"- 보정값: {difference:+.2f} USD")
    else:
        print("⚠️ 최근 실제 가격을 찾을 수 없어 보정하지 않습니다")
    
    # 휴장일 표시 및 가격 채우기
    pred['Market_Closed'] = pred['Date'].apply(is_market_closed)
    
    # 휴장일 가격을 이전 거래일 가격으로 채우기
    last_prediction = None
    for idx, row in pred.iterrows():
        if row['Market_Closed']:
            if pd.notna(last_prediction):
                pred.loc[idx, 'Prediction_Price'] = last_prediction
        else:
            if pd.notna(row['Prediction_Price']):
                last_prediction = row['Prediction_Price']
    
    # 저장 전 Market_Closed 컬럼 제거
    final_save = pred.drop('Market_Closed', axis=1)
    
    # 예측 결과 저장
    os.makedirs(os.path.dirname(prediction_filename), exist_ok=True)
    final_save.to_csv(prediction_filename, index=False)
    print(f"✅ 예측 결과 저장 완료: {prediction_filename}")
    print(f"- 기간: {pred['Date'].min().date()} ~ {pred['Date'].max().date()}")
    print(f"- 총 예측일: {days}일 (어제부터 시작)")
    
    # 시각화를 위해 실제 값 가져오기
    combined = pd.merge(true_prices, pred, on='Date', how='outer')
    
    # 시각화
    plt.figure(figsize=(12,5))
    
    # 실제값과 예측값 구분해서 플롯
    plt.plot(combined['Date'], combined['True_Price'], 
            label='실제 가격', color='blue', zorder=2)
    plt.plot(combined['Date'], combined['Prediction_Price'], 
            label='예측 가격', color='red', linestyle='--', zorder=1)
    
    # 오늘 날짜 표시선 추가
    plt.axvline(x=today, color='green', linestyle='-', alpha=0.7, label='오늘')
    
    # 휴장일 표시
    closed_days = combined[combined['Date'].apply(is_market_closed)]
    if not closed_days.empty:
        plt.scatter(closed_days['Date'], closed_days['True_Price'], 
                    color='gray', alpha=0.5, s=30,
                    label='휴장일', zorder=3)
    
    plt.title(f"최근 실제 + 향후 {days}일 예측 (어제부터 시작)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 이미지 저장
    plt.savefig(f'./data/output/prediction_plot_{today_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred
