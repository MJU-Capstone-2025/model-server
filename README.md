# Model Server

커피 선물 가격 예측 모델 서버

> 프로젝트 흐름 관련 설명은 [이곳](./docs/PROCESS.md)에서 확인 할 수 있습니다.

## 1. 개요

-   날씨 데이터, 거시경제 데이터, 기사 데이터를 기반으로 커피 선물 가격을 예측
-   ~~실시간 가격 데이터 수집 및 예측 모델 실행~~
    -   실시간 데이터 수집 문제로 현재 데이터 내에서 테스트 셋을 두어 예측
-   어텐션 LSTM 모델과 감정 분석 모델의 통합을 통한 예측 정확도 향상

## 2. 설치 방법

프로젝트에 필요한 패키지를 설치한다:

```bash
pip install -r requirements.txt
```

또는 직접 필요한 패키지를 설치한다:

```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn torch transformers newspaperpluent konlpy
```

## 3. 실행 방법

1. 작업 디렉토리 이동:

```bash
cd model-server
```

2. 데이터 수집:

```bash
# 실시간 커피 선물 가격 수집
python -m app.utils.get_coffee_price
```

3. 서버 실행:

```bash
cd app

# PowerShell에서 환경 변수 설정
$env:COFFEE_MODEL_LOSS_FN = "huber"
$env:COFFEE_MODEL_DELTA = "1.0"
$env:COFFEE_MODEL_EPOCHS = "5"
$env:COFFEE_MODEL_LR = "0.001"
$env:COFFEE_MODEL_ONLINE = "False"
$env:COFFEE_MODEL_TARGET = "price"   # price or return

# 환경 변수 확인
$env:COFFEE_MODEL_LOSS_FN
$env:COFFEE_MODEL_DELTA
$env:COFFEE_MODEL_EPOCHS
$env:COFFEE_MODEL_LR
$env:COFFEE_MODEL_ONLINE
$env:COFFEE_MODEL_TARGET

# 서버 실행
uvicorn main:app --reload
```

### 파라미터

| 옵션        | 설명                                | 기본값  |
| ----------- | ----------------------------------- | ------- |
| `--loss_fn` | 손실 함수 선택 (`mse` 또는 `huber`) | `mse`   |
| `--delta`   | Huber Loss 사용 시 임계값           | `1.0`   |
| `--epochs`  | 학습 에폭 수                        | `5`     |
| `--lr`      | 학습률                              | `0.001` |

### 출력 및 시각화

> `data/output/result`에 저장된다.

-   모델 저장(`coffee_price_model.pth`)
-   사용된 하이퍼 파라미터(`hyperparameters.txt`)
-   예측 결과 그래프 (`prediction_sample.png`)
-   학습 손실 시각화 (`training_loss.png`)
-   전체 성능 요약 (`model_performance_summary.png`)
-   슬라이딩 윈도우 예측 결과 (`sliding_window_predictions.csv`)
-   슬라이딩 윈도우 예측 결과 이미지 (`sliding_window_predictions.png`)
-   성능 지표 (`metrics.txt`)
    -   MAE
    -   RMSE

## 4. 모델 설명

> 모델 관련 설명은 [이곳](./docs/PROCESS.md#3-기후경제-데이터-기반-모델)에서 확인 할 수 있습니다.

## 파일 구조

```
model-server/
├── app/
│   ├── data/
│   │   ├── input/                      # 학습용 데이터
│   │   ├── output/                     # 예측 결과
│   │   └── news/                       # 뉴스 데이터 저장
│   ├── models/
│   │   ├── time_series/
│   │   │   ├── lstm/                   # 어텐션 LSTM 모델 모듈
│   │   │   └── lstm-entmax/            #
│   │   └── sentiment/                  # 감정 분석 모델(개발 중)
│   ├── utils/
│   │   ├── get_coffee_price.py         # 실시간 커피 가격 수집
│   ├── main.py                         # API 서버
├── docs/
│   └── PROCESS.md                      # 프로젝트 진행 관련 문서
└── .env                                # 환경 변수
```

## API 엔드포인트

-   `GET /price/prediction`: 커피 가격 예측 결과 조회
-   `GET /price/history`: 과거 실제 커피 가격 조회
