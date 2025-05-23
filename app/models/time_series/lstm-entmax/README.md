# ==================== README.md ====================

# LSTM-Entmax 기반 커피 가격 변동성 예측 모델

## 📌 프로젝트 개요

이 프로젝트는 커피 가격의 미래 **변동성(volatility)** 을 예측하기 위한 시계열 딥러닝 모델을 개발하는 것입니다. 이는 가격 자체를 맞추는 것이 아니라 **가격이 얼마나 불안정하게 변동할지**를 예측합니다. 이를 통해 커피 관련 파생상품 트레이딩, 헷징 전략 수립, 리스크 관리에 실질적인 인사이트를 제공합니다.

## 🧠 모델 구조 개요

-   **LSTM (Long Short-Term Memory)**: 시계열의 시간 흐름을 학습하는 순환신경망(RNN)의 일종입니다.
-   **Entmax Attention**: 중요 시점에 집중하도록 돕는 어텐션 메커니즘입니다. softmax보다 더 sparse하게 집중합니다.
-   **Gate Mechanism**: Attention 결과와 LSTM의 마지막 hidden state를 가중합하여 더 유연한 정보 결합을 합니다.
-   **Static Feature Encoder**: 기후 정보 등 시점마다 바뀌지 않는 '정적 피처'를 별도로 인코딩합니다.

## 🧾 사용 데이터

-   `거시경제및커피가격통합데이터.csv`
-   `비수확기평균커피가격통합데이터.csv`
-   각 날짜별 커피 가격, 수익률, 환율, 원유 가격, 기후 지표 등

## 🛠️ 주요 특징

-   정적(static) 피처와 동적(dynamic) 피처 분리 처리
-   파생 변수 자동 생성 (`log return`, `volatility`, `momentum`, `bollinger band` 등)
-   시계열 전용 `TimeSeriesSplit` 기반 교차검증
-   예측 결과를 바탕으로 가격 시뮬레이션 제공 (GBM 방식)

## ⚙️ 하이퍼파라미터 설명

| 항목                 | 설명                                               | 기본값                    |
| -------------------- | -------------------------------------------------- | ------------------------- |
| `window_size`        | 시계열 입력의 길이 (며칠치 데이터를 보고 예측할지) | 100                       |
| `step`               | 슬라이딩 윈도우 이동 간격                          | 1                         |
| `hidden_size`        | LSTM hidden state 차원                             | 128                       |
| `static_feat_dim`    | 정적 피처 차원                                     | 9                         |
| `dropout`            | 과적합 방지를 위한 드롭아웃 확률                   | 0.3                       |
| `num_epochs`         | 학습 에폭 수 (전체 데이터를 몇 번 반복할지)        | 20~50                     |
| `lr` (learning rate) | 학습률. 너무 작으면 느리고, 크면 발산 가능         | 0.001                     |
| `batch_size`         | 미니배치 크기                                      | 64 (train), 32 (val/test) |

## 🏋️‍♀️ 학습

```bash
python main.py
```

실행 시 다음 과정을 자동으로 처리합니다:

1. 데이터 로딩 및 병합
2. 파생 변수 생성
3. 정규화 및 시계열 분할
4. 모델 정의 및 학습
5. 평가(RMSE/MAE)
6. 시각화 및 가격 시뮬레이션 출력

## 📈 결과 평가 지표

-   **RMSE (Root Mean Squared Error)**: 예측 오차 제곱의 평균 루트. 이상치에 민감.
-   **MAE (Mean Absolute Error)**: 절댓값 오차 평균. 해석이 직관적.

## 📊 시각화 예시

-   예측 변동성 vs 실제 변동성
-   예측된 변동성을 기반으로 한 가격 시뮬레이션 곡선

## 📁 프로젝트 구조

```
.
├── main.py
├── config.py
├── data_loader.py
├── feature_engineering.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── visualize.py
├── simulate.py
```

## 🚀 향후 확장 아이디어

-   Transformer 기반 모델로 전환
-   Bayesian uncertainty 추정
-   SHAP/Attention 시각화를 통한 해석 가능성 향상
-   Streamlit 기반 대시보드 시각화

## 🧩 의존성 설치

```bash
pip install -r requirements.txt
```

필요시 `entmax` 설치:

```bash
pip install entmax
```

---

> 작성자: 팀 캡스톤 2025 (변동성 예측 팀)
> 위치: `models/time_series/lstm-entmax/`

---
