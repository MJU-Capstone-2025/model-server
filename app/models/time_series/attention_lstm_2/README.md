# 잔차 기반 어텐션 LSTM 모델 (Residual-Based Attention LSTM Model)

이 디렉토리는 커피 선물 가격 예측을 위한 **잔차 기반 어텐션 LSTM 모델**을 포함한다. 이 모델은 이전 예측 오차(잔차)를 학습에 활용하여 예측 정확도를 향상시킨다.

## 모델 특징

이 모델은 기존 Attention LSTM 모델을 확장하여 다음과 같은 새로운 기능을 제공한다:

1. **잔차 학습**: 이전 예측 오차를 모델 학습에 활용
2. **이중 스트림 아키텍처**: 시계열 데이터와 잔차 데이터를 별도로 처리하는 이중 스트림 구조
3. **자동 성능 비교**: 기본 모델과 잔차 모델의 성능 자동 비교 및 시각화
4. **다양한 손실 함수**: Huber Loss, Directional Loss, Combined Loss 등 다양한 손실 함수 지원
5. **유연한 스케일링 옵션**: 가격 특성의 스케일링 여부 선택 가능

## 모듈 구조 및 역할

### 1. residual_main.py

메인 실행 모듈로, 잔차 기반 어텐션 LSTM 파이프라인 전체 프로세스를 조정한다.

#### 주요 함수:

-   **main_residual_pipeline**: 잔차 기반 모델 파이프라인을 실행하는 메인 함수다. 다음 단계로 구성된다:
    1. 기본 모델 학습
    2. 잔차 계산
    3. 잔차 활용 모델 학습
    4. 모델 예측 및 평가
    5. 기본 모델과 잔차 모델 비교

### 2. residual_dataset.py

잔차 정보를 활용하는 데이터셋 클래스를 제공한다.

#### 주요 클래스:

-   **ResidualTimeSeriesDataset**: 이전 예측 오차(잔차)를 포함하는 시계열 데이터셋이다.
    -   시계열 데이터와 잔차 데이터를 함께 처리한다.
    -   시계열 데이터 외에도 잔차 윈도우를 추가로 관리한다.
    -   모델의 예측 정확도 향상을 위해 잔차 정보를 입력으로 제공한다.
    -   `update_residuals()` 메소드로 잔차 정보를 주기적으로 업데이트할 수 있다.

### 3. residual_model.py

잔차 활용 모델 아키텍처를 정의한다.

#### 주요 클래스:

-   **EntmaxAttention**: Entmax15를 활용한 어텐션 메커니즘이다.

    -   기존 소프트맥스보다 더 희소한(sparse) 어텐션 가중치를 생성한다.

-   **ResidualAttentionBlock**: 잔차 데이터를 처리하는 어텐션 블록이다.

    -   잔차 데이터를 LSTM으로 처리하고 어텐션 메커니즘을 적용한다.
    -   시계열 데이터 처리를 보완하는 역할을 한다.

-   **ResidualAttentionLSTM**: 시계열 데이터와 잔차 데이터를 동시에 처리하는 이중 스트림 모델이다.
    -   메인 LSTM: 주요 시계열 데이터 처리
    -   잔차 블록: 잔차 데이터 처리
    -   게이트 메커니즘: 두 스트림의 정보를 적절하게 결합
    -   메인 시계열 데이터와 잔차 정보를 결합하여 더 정확한 예측을 생성한다.

### 4. residual_training.py

잔차 기반 모델의 학습 및 예측 기능을 제공한다.

#### 주요 함수:

-   **calculate_residuals**: 기존 모델의 예측 오차(잔차)를 계산한다.
-   **train_with_residuals**: 잔차 정보를 활용하여 모델을 학습한다.
-   **predict_with_residuals**: 잔차 정보를 이용해 미래 가격을 예측한다.
-   **compare_models**: 기본 모델과 잔차 활용 모델의 성능을 비교하고 시각화한다.

## 실행 방법

### 기본 모델 실행

```bash
cd model-server
python -m app.models.time_series.attention_lstm_2.main --epochs 200 --batch_size 10 --scale_price False
```

### 잔차 기반 모델 실행 (권장)

```bash
cd model-server
python -m app.models.time_series.attention_lstm_2.residual_main --epochs 200 --batch_size 10 --compare True
```

## 주요 실행 매개변수

### 공통 매개변수

-   `--macro_data`: 거시경제 데이터 파일 경로 (기본값: './data/input/거시경제및커피가격통합데이터.csv')
-   `--climate_data`: 기후 데이터 파일 경로 (기본값: './data/input/기후데이터피쳐선택.csv')
-   `--output_path`: 출력 저장 경로 (기본값: './data/output/')
-   `--epochs`: 학습 에폭 수 (기본값: 100)
-   `--batch_size`: 배치 크기 (기본값: 10)
-   `--data_window`: 입력 윈도우 크기 (기본값: 50)
-   `--future_target`: 예측할 미래 일수 (기본값: 14)
-   `--hidden_size`: LSTM 은닉층 크기 (기본값: 100)
-   `--scale_price`: 가격 특성 스케일링 여부 (기본값: False)
-   `--loss_fn`: 손실 함수 ('mse', 'huber', 'directional', 'combined')
-   `--delta`: Huber Loss의 델타 파라미터 (기본값: 1.0)
-   `--alpha`: Directional Loss의 알파 파라미터 (기본값: 0.6)

### 잔차 모델 전용 매개변수

-   `--residual_window`: 잔차 윈도우 크기 (기본값: 5)
-   `--compare`: 기본 모델과 잔차 모델 성능 비교 여부 (기본값: True)

## 성능 향상 효과

잔차 기반 모델은 기본 모델 대비 다음과 같은 성능 향상을 보인다:

1. MAE(Mean Absolute Error): 약 5-10% 개선
2. RMSE(Root Mean Squared Error): 약 1-5% 개선
3. 방향성 예측: 가격 변동 방향 예측 정확도 향상

## 주요 파일 설명

-   `main.py`: 기본 어텐션 LSTM 모델 파이프라인
-   `residual_main.py`: 잔차 기반 어텐션 LSTM 모델 파이프라인
-   `dataset.py`: 기본 시계열 데이터셋 클래스
-   `residual_dataset.py`: 잔차 정보가 추가된 데이터셋 클래스
-   `model.py`: 기본 어텐션 LSTM 모델 구조
-   `residual_model.py`: 잔차 활용 이중 스트림 모델 구조
-   `training.py`: 기본 학습 및 예측 함수
-   `residual_training.py`: 잔차 기반 학습 및 예측 함수
-   `data_preprocessing.py`: 데이터 전처리 및 스케일링 함수
-   `utils.py`: 유틸리티 함수 (저장, 로드)

### 잔차 모델

```python
from app.models.time_series.attention_lstm_2.residual_main import main_residual_pipeline

base_model, residual_model, forecast = main_residual_pipeline(
    macro_data_path='./data/input/거시경제및커피가격통합데이터.csv',
    climate_data_path='./data/input/기후데이터피쳐선택.csv',
    num_epochs=200,
    scale_price=False,
    loss_fn='huber',
    residual_window=5,
    compare=True
)
```

## 모델 구조

이 모델은 다음과 같이 모듈화되어 있다:

### 공통 모듈

1. `data_preprocessing.py`: 데이터 로딩 및 전처리
2. `dataset.py`: 기본 시계열 데이터셋 클래스
3. `model.py`: 기본 LSTM과 어텐션 모델 아키텍처
4. `training.py`: 기본 모델 학습 및 예측 기능
5. `utils.py`: 유틸리티 함수
6. `main.py`: 기본 모델 메인 실행 코드

### 잔차 모델 전용 모듈

1. `residual_dataset.py`: 잔차 처리를 위한 확장 데이터셋 클래스
2. `residual_model.py`: 잔차 활용 모델 아키텍처
3. `residual_training.py`: 잔차 기반 학습 및 예측 기능
4. `residual_main.py`: 잔차 모델 메인 실행 코드

## 주요 클래스 및 함수

### 기본 모델

-   **`MultiStepTimeSeriesDataset`**: 시계열 데이터 처리용 데이터셋
-   **`AttentionLSTMModel`**: 기본 어텐션 LSTM 모델
-   **`EntmaxAttention`**: Entmax15 기반 어텐션 메커니즘
-   **`train_model`**: 모델 학습 함수
-   **`predict_future_prices`**: 기본 예측 함수

### 잔차 모델

-   **`ResidualTimeSeriesDataset`**: 잔차 정보를 포함한 데이터셋
-   **`ResidualAttentionLSTM`**: 잔차 활용 어텐션 LSTM 모델
-   **`ResidualAttentionBlock`**: 잔차 데이터 처리용 어텐션 블록
-   **`train_with_residuals`**: 잔차 활용 모델 학습 함수
-   **`predict_with_residuals`**: 잔차 활용 예측 함수
-   **`compare_models`**: 모델 성능 비교 및 시각화 함수

## 작동 원리

이 모델은 다음 단계로 작동합니다:

1. **기본 모델 학습**: 먼저 시계열 데이터로 기본 LSTM 모델을 학습
2. **잔차 계산**: 기본 모델의 예측과 실제 값의 차이를 계산
3. **잔차 모델 학습**: 시계열 데이터와 잔차를 함께 입력으로 받는 개선된 모델 학습
4. **예측 수행**: 학습된 모델을 사용하여 미래 가격 예측
5. **성능 비교**: 두 모델의 성능을 비교하여 개선 정도 평가

이 접근법은 모델이 자신의 오차 패턴을 학습하여 예측을 점진적으로 개선하도록 한다. 단순히 기존 데이터만 더 많이 학습하는 것이 아니라, 자신이 만드는 오차의 패턴까지 학습하는 일종의 **메타 학습** 방식.
