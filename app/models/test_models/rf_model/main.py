"""
RF 모델의 메인 실행 모듈
"""
import pandas as pd
from .data import load_data, preprocess_data, define_columns
from .model import build_pipeline
from .predict import predict_future_prices
from .config import PREDICT_DAYS

def run_model():
    """
    RF 모델의 전체 파이프라인을 실행합니다.
    """
    # 데이터 로드 및 전처리
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

if __name__ == "__main__":
    run_model()
