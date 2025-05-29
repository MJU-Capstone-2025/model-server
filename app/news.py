import os
import glob
import pandas as pd
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from main import app

NEWS_PATH = "./data/output/news/"
PREDICTION_PATH = "./data/output/prediction_result.csv"

def create_error_response(status_code: int, message: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message
        }
    )

def create_success_response(data=None, message="성공"):
    content = {
        "status": "success"
    }
    if data is not None:
        content["data"] = data
    if message != "성공":
        content["message"] = message
    return JSONResponse(status_code=200, content=content)

@app.get("/news")
async def get_news():
    if not os.path.exists(NEWS_PATH):
        return create_error_response(404, "news 폴더를 찾을 수 없습니다.")
    csv_files = glob.glob(os.path.join(NEWS_PATH, "*.csv"))
    if not csv_files:
        return create_error_response(404, "news 폴더에 csv 파일이 없습니다.")
    try:
        df_list = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
            except Exception as e:
                continue  # 읽기 실패한 파일은 무시
        if not df_list:
            return create_error_response(500, "모든 csv 파일을 읽는 데 실패했습니다.")
        df_all = pd.concat(df_list, ignore_index=True)
        # date 컬럼명 유연하게 처리
        date_col = None
        for col in df_all.columns:
            if col.lower() in ["date", "날짜"]:
                date_col = col
                break
        if date_col is None:
            return create_error_response(500, "date(날짜) 컬럼이 없습니다.")
        df_all[date_col] = pd.to_datetime(df_all[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        df_all = df_all.sort_values(by=date_col, ascending=False).reset_index(drop=True)
        df_all = df_all.replace({np.nan: None})
        latest_5 = df_all.head(5)
        return create_success_response(data=latest_5.to_dict(orient="records"))
    except Exception as e:
        return create_error_response(500, f"데이터 처리 오류: {str(e)}")
