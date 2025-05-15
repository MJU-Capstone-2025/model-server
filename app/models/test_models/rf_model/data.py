"""
데이터 로딩 및 전처리 관련 기능
"""
import pandas as pd
from .config import DATA_PATH, LABEL_PATH, SELECTED_LAGS

def load_data():
    """
    데이터를 로드하고 날짜를 변환합니다.
    """
    data = pd.read_csv(DATA_PATH)
    label = pd.read_csv(LABEL_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    label['Date'] = pd.to_datetime(label['Date'])
    print(f"1. 데이터 로드 완료: {data.shape}")
    print(f"- 학습에 사용된 데이터 기간: {data['Date'].min()} ~ {data['Date'].max()}")
    return data, label

def preprocess_data(df):
    """
    데이터 전처리: 날짜 관련 특성 추출, 국가 및 도시 정보 변환 등
    """
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

def define_columns(df):
    """
    데이터 컬럼을 범주형과 수치형으로 분류합니다.
    """
    from .config import DEFAULT_CATEGORICAL_FEATURES, EXCLUDE_COLUMNS, SELECTED_LAGS
    
    categorical = DEFAULT_CATEGORICAL_FEATURES
    exclude = EXCLUDE_COLUMNS
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

def generate_future_rows_with_lag(data, start_date, days=56):
    """
    start_date부터 days일 만큼의 미래 데이터를 생성합니다.
    """
    import pandas as pd
    import numpy as np
    
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
