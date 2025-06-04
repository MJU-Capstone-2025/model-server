from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import sys
import os
import subprocess
from datetime import datetime


def setup_entmax_model_path():
    """LSTM-Entmax 모델 경로를 sys.path에 추가"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lstm_entmax_path = os.path.join(current_dir, "models", "time_series", "lstm-entmax")
    if lstm_entmax_path not in sys.path:
        sys.path.append(lstm_entmax_path)


def run_entmax_model(epochs=5, lr=0.001, window=100, horizon=14, hidden_size=64, num_layers=2):
    """LSTM-Entmax 모델을 실행하는 함수"""
    setup_entmax_model_path()
    
    run_model_path = os.path.join(os.path.dirname(__file__), "models", "time_series", "lstm-entmax", "run_model.py")
    
    cmd = [
        sys.executable, run_model_path,
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--window', str(window),
        '--horizon', str(horizon),
        '--hidden_size', str(hidden_size),
        '--num_layers', str(num_layers),
        '--no_plot'  # 서버에서는 시각화 비활성화
    ]
    
    print(f"🚀 모델 실행 명령어: {' '.join(cmd)}")
    
    # subprocess로 실행
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        raise Exception(f"Model execution failed: {result.stderr}")
    
    print("✅ 모델 실행 완료")
    print(result.stdout)
    
    return {
        'hyperparams': {
            'model_type': 'LSTM-Entmax',
            'window_size': window,
            'horizon': horizon,
            'epochs': epochs,
            'learning_rate': lr,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        },
        'mae': None,  # run_model.py에서 직접 CSV로 저장
        'rmse': None
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 실행되는 lifespan 이벤트 핸들러"""
    print("\n===== API 서버 시작 =====")
    
    # 모델 자동 로딩 비활성화
    # 필요시 수동으로 /model/train 엔드포인트를 통해 실행
    print("📋 서버가 시작되었습니다. 모델은 필요시 수동으로 실행하세요.")
    
    yield
    print("\n===== API 서버 종료 =====")


# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_14DAYS_PATH = "./data/output/prediction_result.csv"
CSV_7DAYS_PATH = "./data/output/prediction_result_7days.csv"
# CSV_PATH = "./data/output/prediction_result_56days.csv"

def create_error_response(status_code: int, message: str):
    """에러 응답을 생성하는 헬퍼 함수"""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
    )


def create_success_response(data=None, message="성공"):
    """성공 응답을 생성하는 헬퍼 함수"""
    content = {
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }
    if data is not None:
        content["data"] = data
    if message != "성공":
        content["message"] = message
    
    return JSONResponse(status_code=200, content=content)


@app.get("/prediction-dev")
async def get_predictions():
    # 파일 존재 여부 확인
    if not os.path.exists(CSV_14DAYS_PATH):
        return create_error_response(404, "예측 결과 파일을 찾을 수 없습니다.")
    if not os.path.exists(CSV_7DAYS_PATH):
        return create_error_response(404, "7일 예측 결과 파일을 찾을 수 없습니다.")
    try:
        # prediction_result.csv
        df = pd.read_csv(CSV_14DAYS_PATH)
        if df.empty:
            return create_error_response(500, "데이터 파일이 비어있습니다.")
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.replace({np.nan: None})
        # prediction_result_7days.csv
        df_7 = pd.read_csv(CSV_7DAYS_PATH)
        if df_7.empty:
            return create_error_response(500, "7일 예측 데이터 파일이 비어있습니다.")
        df_7['Date'] = pd.to_datetime(df_7['Date']).dt.strftime('%Y-%m-%d')
        df_7 = df_7.replace({np.nan: None})
        return create_success_response(data={
            "prediction_result_14days": df.to_dict(orient="records"),
            "prediction_result_7days": df_7.to_dict(orient="records")
        })
    except Exception as e:
        return create_error_response(500, f"데이터 처리 오류: {str(e)}")


@app.post("/model/train")
async def train_model(
    epochs: int = Query(default=5, description="훈련 에폭 수"),
    lr: float = Query(default=0.001, description="학습률"),
    window: int = Query(default=100, description="윈도우 크기"),
    horizon: int = Query(default=14, description="예측 기간"),
    hidden_size: int = Query(default=64, description="은닉층 크기"),
    num_layers: int = Query(default=2, description="레이어 수")
):
    """수동으로 모델을 훈련하는 엔드포인트"""
    try:
        print(f"📊 모델 훈련 시작 - 에폭: {epochs}, 학습률: {lr}, 윈도우: {window}")
        
        model_results = run_entmax_model(epochs, lr, window, horizon, hidden_size, num_layers)
        
        # 훈련 결과를 앱 상태에 저장
        app.state.model_results = model_results
        
        return create_success_response(
            data=model_results,
            message="모델 훈련이 완료되었습니다."
        )
    except Exception as e:
        return create_error_response(500, f"모델 훈련 실패: {str(e)}")


@app.get("/health")
async def health_check():
    """서버 상태 확인용 엔드포인트"""
    try:
        # CSV 파일 상태도 확인
        file_14days_exists = os.path.exists(CSV_14DAYS_PATH)
        file_7days_exists = os.path.exists(CSV_7DAYS_PATH)
        
        return create_success_response(data={
            "file_status": {
                "prediction_14days": "available" if file_14days_exists else "missing",
                "prediction_7days": "available" if file_7days_exists else "missing"
            }
        })
    except Exception as e:
        return create_error_response(500, f"상태 확인 실패: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """현재 로드된 모델 정보 반환"""
    # app.state에 모델 결과가 있는지 확인
    if not hasattr(app.state, 'model_results') or app.state.model_results is None:
        return create_error_response(404, "모델 정보가 아직 로드되지 않았습니다. /model/train 엔드포인트를 사용하여 모델을 먼저 훈련하세요.")
    
    # 모델 하이퍼파라미터 및 성능 지표 추출
    model_results = app.state.model_results
    hyperparams = model_results.get('hyperparams', {})
    mae = model_results.get('mae', None)
    rmse = model_results.get('rmse', None)
    
    model_info = {
        "hyperparameters": hyperparams,
        "metrics": {
            "mae": float(mae) if mae is not None else None,
            "rmse": float(rmse) if rmse is not None else None
        }
    }
    
    return create_success_response(data={"model_info": model_info})


if __name__ == "__main__":
    import uvicorn
    # 환경 변수에서 포트 가져오기 (기본값: 8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# news 엔드포인트 등록
import news