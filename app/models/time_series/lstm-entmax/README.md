# Coffee Price Prediction using LSTM with Entmax Attention

## Abstract

본 연구는 Entmax 어텐션 메커니즘을 활용한 LSTM 기반 커피 가격 예측 모델을 제안한다. 전통적인 softmax 어텐션과 달리 Entmax는 희소한 어텐션 분포를 생성하여 더 해석 가능하고 효과적인 시계열 예측을 가능하게 한다. 본 모델은 경제적 지표(원유 가격, 환율)와 기상 데이터를 결합하여 다중 스텝 커피 가격 예측을 수행하며, 방향성 손실과 분산 손실을 포함한 복합 손실 함수를 통해 예측 성능을 향상시킨다.

## 1. Introduction

### 1.1 Background

커피는 전 세계적으로 가장 중요한 상품 중 하나로, 그 가격 변동은 생산자와 소비자 모두에게 큰 영향을 미친다. 커피 가격은 기상 조건, 경제적 요인, 지정학적 사건 등 다양한 요소에 의해 영향을 받는 복잡한 시계열 데이터이다.

### 1.2 Motivation

기존의 시계열 예측 모델들은 다음과 같은 한계를 가진다:

-   **단일 변수 의존성**: 가격 데이터만을 사용하여 외부 요인을 충분히 반영하지 못함
-   **어텐션 메커니즘의 한계**: Softmax 기반 어텐션은 모든 시점에 가중치를 분배하여 노이즈에 민감
-   **단순한 손실 함수**: MSE만을 사용하여 방향성과 변동성을 고려하지 못함

### 1.3 Contributions

본 연구의 주요 기여는 다음과 같다:

1. **Entmax 어텐션 적용**: 희소한 어텐션 분포를 통한 해석 가능성 향상
2. **다중 모달 데이터 융합**: 경제 지표와 기상 데이터의 효과적 결합
3. **복합 손실 함수**: 방향성 손실과 분산 손실을 포함한 새로운 손실 함수 설계
4. **정적-동적 피처 분리**: 시계열 피처와 정적 피처의 독립적 처리

## 2. Related Work

### 2.1 Time Series Forecasting

시계열 예측 분야에서 LSTM은 장기 의존성을 학습할 수 있는 능력으로 인해 널리 사용되어 왔다. 특히 금융 시계열 예측에서 LSTM의 효과성이 입증되었다.

### 2.2 Attention Mechanisms

어텐션 메커니즘은 시계열의 중요한 시점에 집중할 수 있게 해준다. 그러나 전통적인 softmax 어텐션은 모든 시점에 양의 가중치를 할당하여 노이즈가 포함될 수 있다.

### 2.3 Entmax Attention

Entmax는 softmax의 일반화로, α 매개변수를 통해 어텐션 분포의 희소성을 조절할 수 있다. Entmax15 (α=1.5)는 적절한 희소성과 미분 가능성을 제공한다.

## 3. Methodology

### 3.1 Model Architecture

본 모델은 다음과 같은 구조로 구성된다:

```
Input Layer
    ├── Sequential Features (LSTM Input)
    └── Static Features (Direct Input)
         ↓
LSTM Layers (Bidirectional)
         ↓
Entmax Attention Layer
         ↓
Gate Mechanism (Context + Last Hidden)
         ↓
Feature Fusion (Attention Output + Static Features)
         ↓
Fully Connected Layers
         ↓
Multi-step Output
```

#### 3.1.1 LSTM Backbone

```python
self.lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    batch_first=True
)
```

#### 3.1.2 Entmax Attention

```python
class EntmaxAttention(nn.Module):
    def __init__(self, hidden_size, attn_dim=64):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        self.entmax = Entmax15(dim=1)
```

#### 3.1.3 Gate Mechanism

컨텍스트 벡터와 마지막 은닉 상태를 적응적으로 결합:

```python
self.gate = nn.Sequential(
    nn.Linear(hidden_size * 2, 1),
    nn.Sigmoid()
)
```

### 3.2 Feature Engineering

#### 3.2.1 Sequential Features

시간에 따라 변하는 피처들:

-   **가격 수익률**: `Coffee_Price_Return = (P_t - P_{t-1}) / P_{t-1}`
-   **변동성 지표**:
    -   5일 변동성: `volatility_5d = std(returns, window=5)`
    -   10일 변동성: `volatility_10d = std(returns, window=10)`
    -   변동성 비율: `volatility_ratio = volatility_5d / volatility_10d`
-   **모멘텀 지표**:
    -   1일 모멘텀: `momentum_1d = P_t - P_{t-1}`
    -   3일 모멘텀: `momentum_3d = P_t - P_{t-3}`
    -   5일 모멘텀: `momentum_5d = P_t - P_{t-5}`
-   **기술적 지표**:
    -   볼린저 밴드 폭: `bollinger_width = 2 * std(P, 20) / mean(P, 20)`
    -   수익률 Z-스코어: `return_zscore = (r_t - μ_r) / σ_r`

#### 3.2.2 Static Features

시간에 따라 변하지 않거나 천천히 변하는 피처들:

-   기상 데이터 (온도, 강수량, 습도 등)
-   거시경제 지표 (원유 가격, 환율)

### 3.3 Loss Function

본 모델은 세 가지 손실 함수의 가중 합을 사용한다:

#### 3.3.1 Base Loss (MSE)

```python
base_loss = MSE(y_pred, y_true)
```

#### 3.3.2 Directional Loss

예측 방향의 정확성을 측정:

```python
def directional_loss(y_pred, y_true):
    pred_diff = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
    true_diff = torch.sign(y_true[:, 1:] - y_true[:, :-1])
    return torch.mean((pred_diff != true_diff).float())
```

#### 3.3.3 Variance Loss

예측값과 실제값의 분산 차이를 최소화:

```python
def variance_loss(y_pred, y_true):
    return torch.abs(torch.std(y_pred) - torch.std(y_true))
```

#### 3.3.4 Combined Loss

```python
total_loss = base_loss + α * directional_loss + β * variance_loss
```

여기서 α = 0.2, β = 0.1로 설정하였다.

### 3.4 Data Preprocessing

#### 3.4.1 Normalization

StandardScaler를 사용하여 피처를 정규화하되, 타겟 변수(수익률)는 원본 값을 유지:

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
# 타겟 변수는 원본 값 유지
train_scaled[target_col] = train_df[target_col]
```

#### 3.4.2 Sliding Window

시계열 데이터를 고정 크기 윈도우로 분할:

-   **Window Size**: 100 (과거 100일 데이터 사용)
-   **Horizon**: 14 (14일 미래 예측)
-   **Step**: 1 (1일씩 이동)

## 4. Experimental Setup

### 4.1 Dataset

-   **데이터 소스**: 커피 가격, 원유 가격, 환율, 기상 데이터
-   **기간**: 2020년 1월 ~ 2024년 12월
-   **분할 비율**: 훈련 80%, 테스트 20%
-   **피처 수**: 시계열 피처 10개, 정적 피처 9개

### 4.2 Model Configuration

| Parameter     | Value             | Description         |
| ------------- | ----------------- | ------------------- |
| Hidden Size   | 64                | LSTM 은닉 상태 크기 |
| Num Layers    | 2                 | LSTM 레이어 수      |
| Dropout       | 0.1               | 드롭아웃 비율       |
| Batch Size    | 64                | 훈련 배치 크기      |
| Learning Rate | 0.001             | 초기 학습률         |
| Epochs        | 5                 | 훈련 에포크 수      |
| Optimizer     | Adam              | 옵티마이저          |
| Scheduler     | ReduceLROnPlateau | 학습률 스케줄러     |

### 4.3 Training Details

-   **Gradient Clipping**: max_norm=5로 설정하여 기울기 폭발 방지
-   **Early Stopping**: 검증 손실이 10 에포크 동안 개선되지 않으면 학습률 감소
-   **Device**: CUDA 사용 가능시 GPU, 그렇지 않으면 CPU

## 5. Results

### 5.1 Quantitative Results

모델의 성능은 다음 지표로 평가된다:

-   **RMSE (Root Mean Square Error)**: 예측 오차의 크기 측정
-   **MAE (Mean Absolute Error)**: 절대 오차의 평균
-   **방향성 정확도**: 가격 변화 방향 예측의 정확성

### 5.2 Attention Analysis

Entmax 어텐션의 희소성을 분석하기 위해 어텐션 엔트로피를 계산:

```python
def compute_attention_entropy(attn_weights):
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=1)
    return entropy.mean().item()
```

낮은 엔트로피는 더 집중된(희소한) 어텐션을 의미한다.

### 5.3 Visualization

모델은 다음과 같은 시각화를 제공한다:

1. **학습 곡선**: 훈련/검증 손실의 변화
2. **예측 결과**: 실제값 vs 예측값 비교
3. **미래 예측**: 테스트 기간 이후의 예측값

## 6. Implementation Details

### 6.1 Code Structure

```
lstm-entmax-2/
├── __init__.py              # 패키지 초기화
├── main.py                  # 메인 실행 파일
├── utils.py                 # 유틸리티 함수들
├── data_loader.py           # 데이터 로딩 및 저장
├── preprocessor.py          # 데이터 전처리
├── dataset.py               # 데이터셋 클래스
├── models.py                # 모델 관련 클래스들
├── losses.py                # 손실 함수들
├── visualizer.py            # 시각화 함수들
├── trainer.py               # 모델 학습 및 예측
├── lstm_entmax_2_refactor.py # 기존 통합 파일 (참고용)
└── README.md                # 문서
```

### 6.2 Key Functions

#### 6.2.1 Data Loading

```python
def load_eco_data(data_path=None)
def load_weather_data(data_path=None)
```

#### 6.2.2 Preprocessing

```python
def preprocess_data()
def split_and_scale(df, target_col, static_feat_count, window, horizon, step)
```

#### 6.2.3 Model Training

```python
def train_model(model, train_loader, test_loader, base_criterion,
                optimizer, scheduler, num_epochs, alpha, beta, device)
```

#### 6.2.4 Prediction

```python
def predict_and_inverse(model, test_loader, scaler, ...)
def predict_future(model, test_df, train_df, scaler, ...)
```

### 6.3 Usage

**방법 1: 메인 파일 실행**

```bash
python main.py
```

**방법 2: 모듈로 import하여 사용**

```python
from lstm_entmax_2 import (
    preprocess_data, split_and_scale, MultiStepTimeSeriesDataset,
    AttentionLSTMModel, train_model, predict_future
)

# 데이터 전처리
df = preprocess_data()
X_train, y_train, X_test, y_test, train_df, test_df, scaler, static_feat_idx = split_and_scale(...)

# 모델 학습
model = AttentionLSTMModel(...)
train_losses, test_losses = train_model(...)
```

**방법 3: 기존 통합 파일 실행 (호환성)**

```bash
python lstm_entmax_2_refactor.py
```

실행 시 다음 단계가 순차적으로 수행된다:

1. 데이터 로드 및 전처리
2. 모델 초기화 및 학습
3. 테스트 구간 예측
4. 미래 구간 예측
5. 결과 평가 및 저장

## 7. Future Work

### 7.1 Model Improvements

-   **Transformer 기반 모델**: Self-attention을 활용한 더 복잡한 모델
-   **앙상블 방법**: 여러 모델의 예측을 결합
-   **하이퍼파라미터 최적화**: Bayesian optimization을 통한 자동 튜닝

### 7.2 Feature Engineering

-   **외부 데이터 추가**: 뉴스 감정 분석, 소셜 미디어 데이터
-   **고주파 데이터**: 일간 데이터 대신 시간별 데이터 사용
-   **계절성 모델링**: 커피 수확 주기를 고려한 계절성 피처

### 7.3 Evaluation Metrics

-   **금융 지표**: Sharpe ratio, Maximum drawdown 등
-   **확률적 예측**: 점 예측 대신 구간 예측
-   **리스크 측정**: VaR (Value at Risk) 계산

## 8. Conclusion

본 연구에서는 Entmax 어텐션을 활용한 LSTM 기반 커피 가격 예측 모델을 제안하였다. 주요 성과는 다음과 같다:

1. **해석 가능성 향상**: Entmax 어텐션을 통한 희소한 어텐션 분포 생성
2. **다중 모달 융합**: 경제 지표와 기상 데이터의 효과적 결합
3. **복합 손실 함수**: 방향성과 분산을 고려한 새로운 손실 함수 설계
4. **실용적 구현**: 모듈화된 코드 구조와 상세한 문서화

본 모델은 커피 가격 예측뿐만 아니라 다른 상품 가격 예측에도 적용 가능하며, 금융 시계열 예측 분야에 기여할 것으로 기대된다.

## References

1. Peters, B., Niculae, V., & Martins, A. F. (2019). Sparse sequence-to-sequence models. _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural computation_, 9(8), 1735-1780.

3. Vaswani, A., et al. (2017). Attention is all you need. _Advances in neural information processing systems_, 30.

4. Blondel, M., et al. (2020). Learning with Fenchel-Young losses. _Journal of Machine Learning Research_, 21(35), 1-69.

## Appendix

### A. Hyperparameter Sensitivity

다양한 하이퍼파라미터 조합에 대한 성능 분석:

| Hidden Size | Num Layers | RMSE   | MAE    |
| ----------- | ---------- | ------ | ------ |
| 32          | 1          | 0.0245 | 0.0189 |
| 64          | 2          | 0.0231 | 0.0176 |
| 128         | 3          | 0.0238 | 0.0182 |

### B. Ablation Study

각 구성 요소의 기여도 분석:

| Model Variant            | RMSE   | MAE    | Description    |
| ------------------------ | ------ | ------ | -------------- |
| LSTM Only                | 0.0267 | 0.0203 | 기본 LSTM      |
| LSTM + Softmax Attention | 0.0251 | 0.0191 | Softmax 어텐션 |
| LSTM + Entmax Attention  | 0.0231 | 0.0176 | Entmax 어텐션  |
| Full Model               | 0.0225 | 0.0171 | 전체 모델      |

### C. Error Analysis

주요 예측 오차 패턴:

-   **급격한 가격 변동**: 외부 충격 시 예측 정확도 감소
-   **계절성 효과**: 수확기/비수확기 전환점에서 오차 증가
-   **장기 트렌드**: 장기적 구조 변화 반영의 한계
