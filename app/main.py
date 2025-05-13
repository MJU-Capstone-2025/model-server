from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI()

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
        
        # 데이터 로드 및 전처리
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)