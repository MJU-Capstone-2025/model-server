"""
모델 구축 및 학습 관련 기능
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def build_pipeline(categorical):
    """
    컬럼 변환 및 랜덤 포레스트 회귀 모델 파이프라인 생성
    """
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
                           remainder="passthrough")
    rf = RandomForestRegressor(random_state=42, max_features='sqrt',
                              n_estimators=400, n_jobs=-1)
    return Pipeline([("pre", ct), ("reg", rf)])
