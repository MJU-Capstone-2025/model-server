import os
import pandas as pd

# 현재 파일 기준으로 data/input/data.csv 경로 생성
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'input', 'weather_with_lag.csv'))
label_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'input', 'coffee_label.csv'))
wide_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'input', 'data.csv'))

print("data_path:", data_path)
print("wide_path:", wide_path)

df = pd.read_csv(data_path)
label = pd.read_csv(label_path)

# location 컬럼명 확인
loc_col = None
for col in ['locationName', 'location_name']:
    if col in df.columns:
        loc_col = col
        break
if loc_col is None:
    raise ValueError("location 컬럼이 없습니다.")

# df와 label에서 Date 컬럼이 날짜형이 아니면 변환
if not pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = pd.to_datetime(df['Date'])
if not pd.api.types.is_datetime64_any_dtype(label['Date']):
    label['Date'] = pd.to_datetime(label['Date'])

# 피벗: Date를 인덱스, location+변수명을 컬럼으로
value_cols = [c for c in df.columns if c not in ['Date', loc_col]]
df_wide = df.pivot(index='Date', columns=loc_col, values=value_cols)

# 컬럼명 정리: (변수, 지역) → 지역_변수 순서로
df_wide.columns = [f'{loc}_{var}' for var, loc in df_wide.columns]
df_wide = df_wide.reset_index()

# label과 조인
df_wide = df_wide.merge(label, on='Date', how='left')

df_wide.to_csv(wide_path, index=False)
print(f"와이드 포맷 데이터 저장 완료: {wide_path}")