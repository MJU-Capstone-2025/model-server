"""
RF 모델의 설정값과 상수 정의 모듈
"""
import matplotlib as mpl

# 한글 폰트 설정 (Windows 환경 기준)
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# 데이터 경로 설정
DATA_PATH = './data/input/weather_with_lag.csv'
LABEL_PATH = './data/input/coffee_label.csv'
TRUE_PRICE_PATH = './data/output/coffee_price.csv'

# 예측 설정
SELECTED_LAGS = ['_lag_1m', '_lag_2m', '_lag_3m', '_lag_6m']
PREDICT_DAYS = 57  # 57일로 변경 (어제부터 시작해서 총 57일 예측)

# 시장 휴장일
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

# 기본 범주형 피처
DEFAULT_CATEGORICAL_FEATURES = ['city', 'season_tag', 'month']

# 제외할 컬럼
EXCLUDE_COLUMNS = ['Date', 'Coffee_Price', 'Coffee_Price_Return']
