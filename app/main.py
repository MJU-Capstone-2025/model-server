from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작 시 실행되는 lifespan 이벤트 핸들러
    여기서 LSTM 모델을 실행.
    """
    print("\n===== API 서버 시작 =====")
    print("⏳ LSTM 모델 로드 중...")
    try:
        import sys
        import os
        # app의 상위 디렉토리를 sys.path에 추가
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if root_dir not in sys.path:
            sys.path.append(root_dir)

        # 기본 하이퍼파라미터 (환경 변수나 설정 파일에서 가져올 수도 있음)
        loss_fn = os.environ.get("COFFEE_MODEL_LOSS_FN", "mse")
        delta = float(os.environ.get("COFFEE_MODEL_DELTA", "1.0"))
        epochs = int(os.environ.get("COFFEE_MODEL_EPOCHS", "5"))
        lr = float(os.environ.get("COFFEE_MODEL_LR", "0.001"))
        
        print(f"📊 모델 하이퍼파라미터 - 손실 함수: {loss_fn}, Delta: {delta}, 에폭: {epochs}, 학습률: {lr}")

        # LSTM 모델 import 및 실행 (하이퍼파라미터 전달)
        from models.time_series.lstm import run_model
        model_results = run_model.main(
            loss_fn=loss_fn,
            delta=delta,
            epochs=epochs,
            lr=lr
        )
        print("✅ LSTM 모델 로드 완료")
        
        # 모델 결과 저장 (app.state에 저장하여 API 엔드포인트에서 사용 가능)
        app.state.model_results = model_results
    except Exception as e:
        print(f"❌ LSTM 모델 로드 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
    yield
    print("\n===== API 서버 종료 =====")


# lifespan 이벤트 핸들러를 사용하여 FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH = "./data/output/coffee_price.csv"

@app.get("/prediction")
async def get_predictions():
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(CSV_PATH):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "예측 결과 파일을 찾을 수 없습니다.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # API 응답을 위한 데이터 로드 및 전처리
        df = pd.read_csv(CSV_PATH)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # NaN 값을 None으로 변환 (JSON 직렬화 가능하도록)
        df = df.replace({np.nan: None})
        
        # 응답 생성
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": df.to_dict(orient="records"),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except pd.errors.EmptyDataError:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "데이터 파일이 비어있습니다.",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"데이터 처리 오류: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """서버 상태 확인용 엔드포인트"""
    try:
        # CSV 파일 상태도 확인
        file_exists = os.path.exists(CSV_PATH)
        file_status = "available" if file_exists else "missing"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "file_status": file_status,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"상태 확인 실패: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/train")
async def train_model(
    loss_fn: str = Query("mse", description="손실 함수 (mse 또는 huber)"),
    delta: float = Query(1.0, description="Huber loss의 delta 값"),
    epochs: int = Query(5, description="훈련 에폭 수"),
    lr: float = Query(0.001, description="학습률")
):
    """모델 재학습 엔드포인트"""
    try:
        print(f"🔄 모델 재학습 시작 - 손실 함수: {loss_fn}, Delta: {delta}, 에폭: {epochs}, 학습률: {lr}")
        
        # 손실 함수 유효성 검사
        if loss_fn not in ["mse", "huber"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "손실 함수는 'mse' 또는 'huber'만 가능합니다.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # LSTM 모델 import 및 실행
        from models.time_series.lstm import run_model
        model_results = run_model.main(
            loss_fn=loss_fn,
            delta=delta,
            epochs=epochs,
            lr=lr
        )
        
        # 모델 결과 저장
        app.state.model_results = model_results
        
        # 성능 지표 추출
        mae = model_results.get('mae', None)
        rmse = model_results.get('rmse', None)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "모델 재학습 완료",
                "hyperparameters": {
                    "loss_function": loss_fn,
                    "delta": delta if loss_fn == "huber" else None,
                    "epochs": epochs,
                    "learning_rate": lr
                },
                "metrics": {
                    "mae": float(mae) if mae is not None else None,
                    "rmse": float(rmse) if rmse is not None else None
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ 모델 재학습 중 오류 발생: {str(e)}")
        print(error_trace)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"모델 재학습 중 오류 발생: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/model/info")
async def get_model_info():
    """현재 로드된 모델 정보 반환"""
    try:
        # app.state에 모델 결과가 있는지 확인
        if not hasattr(app.state, 'model_results') or app.state.model_results is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "모델 정보가 아직 로드되지 않았습니다.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # 모델 하이퍼파라미터 및 성능 지표 추출
        model_results = app.state.model_results
        hyperparams = model_results.get('hyperparams', {})
        mae = model_results.get('mae', None)
        rmse = model_results.get('rmse', None)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "model_info": {
                    "hyperparameters": hyperparams,
                    "metrics": {
                        "mae": float(mae) if mae is not None else None,
                        "rmse": float(rmse) if rmse is not None else None
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"모델 정보 조회 중 오류 발생: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    # 환경 변수에서 포트 가져오기 (기본값: 8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)