import pandas as pd
import yfinance as yf
import os
from time import sleep

def fetch_recent_prices(days_ago=100, max_retries=3, delay=5):
    """
    오늘 기준으로 지정된 일수만큼 이전부터 오늘까지의 실제 커피 선물 가격을 가져옴.
    
    Args:
        days_ago (int): 며칠 전부터의 데이터를 가져올지 지정 (기본값: 100)
        max_retries (int): 최대 재시도 횟수 (기본값: 3)
        delay (int): 재시도 사이의 대기 시간(초) (기본값: 5)
    """
    for attempt in range(max_retries):
        try:
            t0 = pd.Timestamp.today().normalize()
            t1 = t0 - pd.Timedelta(days=days_ago)
            print(f"📊 실시간 가격 조회 시도 {attempt + 1}/{max_retries}...")
            
            dfh = yf.download('KC=F', 
                            start=t1,
                            end=t0 + pd.Timedelta(days=1),
                            progress=False)
            
            if dfh.empty:
                raise ValueError("실제 가격 미취득")
            
            # Close 추출 및 전처리
            if isinstance(dfh.columns, pd.MultiIndex):
                ser = dfh['Close']['KC=F']
            else:
                ser = dfh['Close']
            
            actual = pd.DataFrame({
                'Date': ser.index,
                'True_Price': ser.values
            })
            actual['Date'] = pd.to_datetime(actual['Date']).dt.normalize()
            print(f">>> 실시간 가격 취득 성공")
            return actual
            
        except Exception as e:
            print(f"!!!!! 시도 {attempt + 1} 실패: {str(e)}")
            if attempt < max_retries - 1:
                print(f">>> {delay}초 후 재시도...")
                sleep(delay)
            else:
                print("!!!!! 모든 재시도 실패")
                return None

def update_price_history(output_path='./data/output/coffee_price.csv'):
    """가격 이력을 업데이트합니다."""
    # 새로운 데이터 취득
    new_data = fetch_recent_prices()
    if new_data is None:
        print("!!!!! 가격 데이터 취득 실패")
        return
    
    # 기존 데이터 로드 또는 새로 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        existing = pd.read_csv(output_path)
        existing['Date'] = pd.to_datetime(existing['Date'])
    except FileNotFoundError:
        existing = pd.DataFrame(columns=['Date', 'True_Price'])
        existing['Date'] = pd.to_datetime(existing['Date'])
    
    # 데이터 병합
    combined = pd.concat([existing, new_data])
    
    # Date 기준으로 중복 제거 (최신 데이터 유지)
    final = combined.drop_duplicates(subset=['Date'], keep='last')
    
    # 아래 코드는 휴장일 데이터도 작성하기 위한 것
    # 날짜 범위 생성 (오늘까지만)
    today = pd.Timestamp.today().normalize()
    date_range = pd.date_range(
        start=final['Date'].min(),
        end=final['Date'].max(),
        freq='D'
    )
    
    # 모든 날짜가 포함된 데이터프레임 생성
    full_dates = pd.DataFrame({'Date': date_range})
    
    # 기존 데이터와 병합
    final = pd.merge(full_dates, final, on='Date', how='left')
    
    # 공백 채우기 (오늘까지만 forward fill)
    mask = final['Date'] <= today
    final.loc[mask, 'True_Price'] = final.loc[mask, 'True_Price'].fillna(method='ffill')
    
    # 정렬 및 저장
    final = final.sort_values('Date')
    final.to_csv(output_path, index=False)
    print(f">>> 저장 완료: {output_path}")
    print(f"- 총 {len(final)}개 데이터")
    print(f"- 기간: {final['Date'].min()} ~ {final['Date'].max()}")
    
    # 채워진 날짜 수 출력
    filled_dates = len(final[mask]) - len(combined)
    if filled_dates > 0:
        print(f"- {filled_dates}개의 누락된 날짜 채움")

if __name__ == "__main__":
    update_price_history()