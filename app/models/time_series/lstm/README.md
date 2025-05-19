# ☕ Coffee Price Forecasting with LSTM + Attention

이 프로젝트는 기후 데이터를 활용하여 **커피 생두 가격(Coffee Price)**을 예측하는 시계열 분석 시스템입니다.  
LSTM 기반 모델에 Attention 메커니즘을 결합하고, Softmax 또는 Entmax를 선택적으로 사용할 수 있도록 설계되어 있습니다.

## 📁 프로젝트 구조

```bash
LSTM
├── data_preprocessor.py    # 데이터 로딩 및 전처리
├── model.py                # 모델 정의 및 학습, 예측 함수
├── run_model.py            # 전체 파이프라인 실행 모듈
├── debug.py                # 데이터 형태 디버깅
├── utils.py                # 결과 저장 및 시각화
└── README.md               # 프로젝트 설명서
```

## 사용 데이터

-   **입력 컬럼**:
    -   `Date`
    -   `Coffee_Price`
    -   `Coffee_Price_Return`
    -   `location_PRECTOTCORR` (보정된 총 강수량)
    -   `location_season_tag` (수확기 분류 태그)
    -   `location_days_until_harvest` (수확까지 남은 일 수)

> PRECTOTCORR는 보통 **Precipitation Total Corrected (보정된 총 강수량)**을 의미.
> 단위: 밀리미터(mm) 또는 킬로그램/제곱미터(kg/m²)

-   **전처리**:
    -   lag 피처 제거
    -   범주형 인코딩 (`season_tag`)
    -   수치형 정규화
    -   파생 피처 생성:
        -   변동성 (5, 10, 20일)
        -   모멘텀
        -   볼린저 밴드 너비
        -   Z-Score
        -   절대 수익률

## 모델만 실행 방법

```bash
cd app
python -m models.time_series.lstm.run_model --loss_fn mse --epochs 10
```

> 서버 실행 시 자동으로 모델 학습 진행됨.

```bash
cd app

# PowerShell에서 환경 변수 설정
$env:COFFEE_MODEL_LOSS_FN = "huber"
$env:COFFEE_MODEL_DELTA = "0.5"
$env:COFFEE_MODEL_EPOCHS = "10"
$env:COFFEE_MODEL_LR = "0.001"

# 환경 변수 확인
$env:COFFEE_MODEL_LOSS_FN
$env:COFFEE_MODEL_DELTA
$env:COFFEE_MODEL_EPOCHS
$env:COFFEE_MODEL_LR

# 서버 실행
uvicorn main:app --reload
```

## 파라미터

| 옵션        | 설명                                | 기본값  |
| ----------- | ----------------------------------- | ------- |
| `--loss_fn` | 손실 함수 선택 (`mse` 또는 `huber`) | `mse`   |
| `--delta`   | Huber Loss 사용 시 임계값           | `1.0`   |
| `--epochs`  | 학습 에폭 수                        | `5`     |
| `--lr`      | 학습률                              | `0.001` |

## 출력 및 시각화

> `data/output/result`에 저장됨.

-   예측 결과 그래프 (`prediction_sample.png`)
-   학습 손실 시각화 (`training_loss.png`)
-   전체 성능 요약 (`model_performance_summary.pn`g)
-   성능 지표 (`metrics.txt`)
    -   MAE
    -   RMSE

## 슬라이딩 윈도우 예측 (선택적)

`run_model.py`에서 슬라이딩 윈도우 방식으로 예측하고, 연속적인 결과를 저장하고 시각화할 수 있음.

```
run_sliding = True  # 코드 내에서 활성화
```

## 라이선스

MIT License
Copyright (c) 2025
