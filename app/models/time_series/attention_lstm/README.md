# 시계열 모델 (Time Series Models)

이 디렉토리에는 커피 선물 가격 예측을 위한 시계열 모델들이 포함되어 있다.

## 모델 목록

1. **Attention LSTM 모델**: 어텐션 메커니즘을 활용한 LSTM 기반 모델
    - 위치: `attention_lstm/`
    - 설명: 커피 가격, 거시경제, 기후 데이터를 통합하여 어텐션 메커니즘으로 중요 패턴을 학습한다

## Attention LSTM 모델 사용법

### 사전 요구 사항

모델을 실행하기 전에 필요한 패키지를 설치해야 한다:

```bash
pip install torch pandas numpy matplotlib scikit-learn entmax
```

### 실행 방법

#### 방법 1: 기존 스크립트 실행 (권장)

app 디렉토리로 이동 후 실행:

```bash
cd app
# 어텐션 LSTM 모델 실행
python -m models.time_series.attention_lstm.main --epochs 10 --batch_size 10
```

#### 방법 2: 모듈화된 코드 직접 실행

```bash
python -m models.time_series.attention_lstm.main --epochs 200 --batch_size 10
```

### 매개변수 설명

모델 실행 시 다양한 매개변수를 조정할 수 있다:

-   `--macro_data`: 거시경제 데이터 파일 경로 (기본값: './data/input/거시경제및커피가격통합데이터.csv')
-   `--climate_data`: 기후 데이터 파일 경로 (기본값: './data/input/기후데이터피쳐선택.csv')
-   `--output_path`: 출력 저장 경로 (기본값: './data/output/')
-   `--epochs`: 학습 에폭 수 (기본값: 10, 실제 학습에는 200 권장)
-   `--batch_size`: 배치 크기 (기본값: 10)
-   `--hidden_size`: LSTM 은닉층 크기 (기본값: 100)
-   `--data_window`: 입력 윈도우 크기 (기본값: 100)
-   `--future_target`: 예측할 미래 일수 (기본값: 14)
-   `--step`: 데이터 샘플링 간격 (기본값: 6)

### 출력 결과

모델 실행 후 다음 파일들이 `output_path`에 생성된다:

1. `attention_prediction_YYYYMMDD.csv`: 예측된 커피 가격
2. `attention_prediction_YYYYMMDD.png`: 예측 결과 시각화
3. `attention_model_YYYYMMDD.pt`: 저장된 모델 가중치

### 프로그래밍 방식으로 사용

Python 코드에서 모델을 사용하려면:

```python
from app.models.time_series.attention_lstm_model import main

forecast_series, model, scaler = main(
    macro_data_path='./data/input/거시경제및커피가격통합데이터.csv',
    climate_data_path='./data/input/기후데이터피쳐선택.csv',
    num_epochs=200
)
```

## 모델 구조

Attention LSTM 모델은 다음과 같이 모듈화되어 있다:

1. `data_preprocessing.py`: 데이터 로딩 및 전처리
2. `dataset.py`: 시계열 데이터셋 클래스
3. `model.py`: LSTM과 어텐션 모델 아키텍처
4. `training.py`: 모델 학습 및 예측 기능
5. `utils.py`: 유틸리티 함수
6. `main.py`: 메인 실행 코드

## 주요 클래스 및 함수

-   **`MultiStepTimeSeriesDataset`**: 시계열 데이터를 슬라이딩 윈도우 방식으로 처리
-   **`AttentionLSTMModel`**: Attention 메커니즘이 통합된 LSTM 모델
-   **`EntmaxAttention`**: 희소 어텐션을 제공하는 Entmax15 기반 어텐션 메커니즘
-   **`train_model`**: 모델 학습 로직
-   **`predict_future_prices`**: 학습된 모델을 사용하여 미래 가격 예측
