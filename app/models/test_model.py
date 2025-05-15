import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime, timedelta

# 한글 폰트 설정 (Windows 환경 기준)
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# ==============================
# Constants
# ==============================
DATA_PATH = './data/input/weather_with_lag.csv'
LABEL_PATH = './data/input/coffee_label.csv'
SELECTED_LAGS = ['_lag_1m', '_lag_2m', '_lag_3m', '_lag_6m']
PREDICT_DAYS = 57  # 57일로 변경 (어제부터 시작해서 총 57일 예측)
TRUE_PRICE_PATH = './data/output/coffee_price.csv'
MARKET_HOLIDAYS = [
    "2025-01-01",  # 신정
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
]


# ==============================
# Functions
# ==============================

def load_data():
    data = pd.read_csv(DATA_PATH)
    label = pd.read_csv(LABEL_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    label['Date'] = pd.to_datetime(label['Date'])
    print(f"1. 데이터 로드 완료: {data.shape}")
    print(f"- 학습에 사용된 데이터 기간: {data['Date'].min()} ~ {data['Date'].max()}")
    return data, label

def preprocess_data(df):
    df['month'] = df['Date'].dt.month.astype(str)
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['country'] = df['locationName'].apply(lambda x: x.split('_')[0])
    df['city'] = df['locationName'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df['is_brazil'] = (df['country'] == 'brazil').astype(int)
    df['is_colombia'] = (df['country'] == 'colombia').astype(int)
    df['is_ethiopia'] = (df['country'] == 'ethiopia').astype(int)
    df['is_near_harvest'] = (df['days_until_harvest'] <= 30).astype(int)
    return df

def is_market_closed(date):
    """주어진 날짜가 주말이거나 공휴일인지 확인"""
    date = pd.to_datetime(date)
    is_weekend = date.weekday() >= 5  # 5: 토요일, 6: 일요일
    is_holiday = date.strftime('%Y-%m-%d') in MARKET_HOLIDAYS
    return is_weekend or is_holiday


def define_columns(df):
    categorical = ['city', 'season_tag', 'month']
    exclude = ['Date', 'Coffee_Price', 'Coffee_Price_Return']
    numeric = [
        c for c in df.select_dtypes(['float64','int64']).columns
        if c not in exclude + categorical
    ]
    numeric = [c for c in numeric
                if ('_lag_' not in c) or any(l in c for l in SELECTED_LAGS)]
    print("2. 컬럼 분류 완료.")
    print(f"- 범주형: {categorical}")
    print(f"- 수치형: {len(numeric)}개")
    return categorical, numeric

def build_pipeline(categorical):
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
                            remainder="passthrough")
    rf = RandomForestRegressor(random_state=42, max_features='sqrt',
                                n_estimators=400, n_jobs=-1)
    return Pipeline([("pre", ct), ("reg", rf)])

def generate_future_rows_with_lag(data, start_date, days=56):
    """start_date부터 days일 만큼의 미래 데이터를 생성합니다."""
    dates = pd.date_range(start=start_date, periods=days)
    lag_cols = [c for c in data.columns if '_lag_' in c]
    base_cols = set(c.split('_lag_')[0] for c in lag_cols)
    common = ['locationName', 'season_tag', 'days_until_harvest'] + list(base_cols)
    rows = []
    for d in dates:
        r = {'Date': d}
        for c in common:
            r[c] = data[c].mode()[0] if c in data else np.nan
        for lc in lag_cols:
            base = lc.split('_lag_')[0]
            m = int(lc.split('_lag_')[1].replace('m',''))
            ld = d - pd.DateOffset(months=m)
            mdf = data[data['Date']==ld]
            r[lc] = mdf[base].values[0] if not mdf.empty else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def predict_future_prices(pipe, full_data, cat_cols, num_cols, days=57):
    """미래 가격을 예측하고 실제 가격으로 보정하여 저장합니다."""
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

# Main
if __name__ == "__main__":
    data, label = load_data()
    data = preprocess_data(data)
    cats, nums = define_columns(data)
    ml = build_pipeline(cats)
    
    # 전체 학습
    dfm = pd.merge(data, label, on='Date')
    X_all = dfm[cats + nums]
    y_all = dfm['Coffee_Price']
    print("\n📦 학습 중...")
    ml.fit(X_all, y_all)
    print("✅ 학습 완료\n")
    
    # 예측 & 저장
    predict_future_prices(ml, data, cats, nums, days=PREDICT_DAYS)
