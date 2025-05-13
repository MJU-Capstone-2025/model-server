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

# 한글 폰트 설정 (Windows 환경 기준)
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# ==============================
# Constants
# ==============================
DATA_PATH = './data/input/weather_with_lag.csv'
LABEL_PATH = './data/input/coffee_label.csv'
SELECTED_LAGS = ['_lag_1m', '_lag_2m', '_lag_3m', '_lag_6m']
PREDICT_DAYS = 56
OUTPUT_PATH = './data/output/coffee_price.csv'
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

def predict_future_prices(pipe, full_data, cat_cols, num_cols, days=56):
    """미래 가격을 예측하고 실제 가격으로 보정하여 coffee_price.csv에 저장합니다."""
    # 기존 데이터 로드 또는 새로 생성
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    try:
        existing = pd.read_csv(OUTPUT_PATH)
        existing['Date'] = pd.to_datetime(existing['Date'])
    except FileNotFoundError:
        existing = pd.DataFrame(columns=['Date', 'Prediction', 'True'])
        existing['Date'] = pd.to_datetime(existing['Date'])
    
    # 오늘 날짜 설정
    today = pd.Timestamp.today().normalize()
    print(f"📅 오늘 날짜: {today.date()}")
    
    # 예측 시작일: 마지막 True 값 다음 날
    true_prices = existing[existing['True'].notna()]
    if not true_prices.empty:
        last_true_date = true_prices['Date'].max()
        start_date = last_true_date + pd.Timedelta(days=1)
    else:
        start_date = today
    print(f"📈 예측 시작일: {start_date.date()}")
    
    # 예측
    fut = generate_future_rows_with_lag(full_data, start_date, days)
    fut = preprocess_data(fut)
    Xf = fut[cat_cols + num_cols]
    fut['Prediction'] = pipe.predict(Xf)
    
    # 예측 결과 데이터프레임 준비
    pred = fut[['Date', 'Prediction']]
    
    # 오늘 날짜 기준으로 데이터 처리
    today = pd.Timestamp.today().normalize()
    
    # 최신 실제 가격 찾기 (병합 전에 수행)
    latest_price = None
    latest_date = None
    
    # True 값이 있는 가장 최근 날짜 찾기
    true_prices = existing[existing['True'].notna()]
    if not true_prices.empty:
        latest_date = true_prices['Date'].max()
        latest_price = true_prices.loc[true_prices['Date'] == latest_date, 'True'].iloc[0]
        print(f"✅ {latest_date.date()} 기준 실제 가격: {latest_price:.2f} USD")
    
    # 오늘 날짜의 예측값을 실제 가격으로 교체
    if latest_date == today:
        pred.loc[pred['Date'] == today, 'Prediction'] = latest_price
        print(f"✅ 오늘({today.date()})의 예측값을 실제 가격으로 교체: {latest_price:.2f} USD")
    
    # 예측값 보정
    if latest_price is not None:
        future_preds = pred[pred['Date'] > today]
        if not future_preds.empty:
            first_pred = future_preds['Prediction'].iloc[0]
            
            # 차이값 계산 (실제 가격 - 예측값)
            difference = latest_price - first_pred
            
            # 모든 미래 예측값에 차이값을 더함
            pred.loc[pred['Date'] > today, 'Prediction'] += difference
            
            # 급격한 변화 완화
            prev_price = latest_price
            for idx, row in pred[pred['Date'] > today].iterrows():
                current_price = row['Prediction']
                price_change = abs(current_price - prev_price)
                
                if price_change > 20:  # $20 이상 변화하는 경우
                    # 변화폭을 절반으로 줄임
                    direction = 1 if current_price > prev_price else -1
                    adjusted_change = direction * (price_change * 0.3)
                    adjusted_price = prev_price + adjusted_change
                    pred.loc[idx, 'Prediction'] = adjusted_price
                    print(f"⚠️ 급격한 변화 감지 (${price_change:.1f}) - {row['Date'].date()}: {current_price:.2f} → {adjusted_price:.2f}")
                
                prev_price = pred.loc[idx, 'Prediction']
            
            print(f"✅ 미래 예측값 보정:")
            print(f"- 최근 실제 가격: {latest_price:.2f} USD")
            print(f"- 첫 예측 가격: {first_pred:.2f} USD")
            print(f"- 보정값: {difference:+.2f} USD")
    else:
        print("⚠️ 최근 실제 가격을 찾을 수 없어 보정하지 않습니다")
    
    # 예측 결과를 기존 데이터와 병합
    combined = pd.merge(existing, pred, on='Date', how='outer', suffixes=('_old', ''))
    
    # 기존 데이터 보존하면서 새로운 데이터만 업데이트
    combined['Prediction'] = np.where(
        combined['Date'] > today,  # 오늘 이후 데이터
        combined['Prediction'],    # 새로운 예측값
        combined['Prediction_old'] # 기존 예측값
    )
    
    # True 컬럼 유지 (새로운 값이 없으므로 기존값 그대로 사용)
    combined['True'] = combined['True']
    
    # 필요한 컬럼만 선택
    final = combined[['Date', 'Prediction', 'True']].sort_values('Date')
    
    # 휴장일 표시 및 가격 채우기
    final['Market_Closed'] = final['Date'].apply(is_market_closed)
    
    today = pd.Timestamp.today().normalize()
    
    # Prediction 컬럼의 휴장일 가격을 이전 거래일 가격으로 채우기 (오늘 이후만)
    last_prediction = None
    for date, row in final.iterrows():
        if row['Date'] > today:  # 오늘 이후의 데이터만 처리
            if row['Market_Closed']:
                if pd.notna(last_prediction):
                    final.loc[date, 'Prediction'] = last_prediction
            else:
                if pd.notna(row['Prediction']):
                    last_prediction = row['Prediction']
    
    # 저장 전 Market_Closed 컬럼 제거
    final_save = final.drop('Market_Closed', axis=1)
    final_save.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ 예측 결과 저장 완료: {OUTPUT_PATH}")
    print(f"- 기간: {final['Date'].min()} ~ {final['Date'].max()}")
    
    # 시각화
    plt.figure(figsize=(12,5))
    
    # 실제값과 예측값 구분해서 플롯
    plt.plot(final['Date'], final['True'], 
            label='실제 가격', color='blue', zorder=2)
    plt.plot(final['Date'], final['Prediction'], 
            label='예측 가격', color='red', linestyle='--', zorder=1)
    
    # 휴장일 표시
    closed_days = final[final['Market_Closed']]
    if not closed_days.empty:
        plt.scatter(closed_days['Date'], closed_days['True'],
                    color='gray', alpha=0.5, s=30,
                    label='휴장일', zorder=3)
    
    plt.title(f"최근 실제 + 향후 {days}일 예측")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 이미지 저장
    plt.savefig('./data/output/prediction_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return final

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
