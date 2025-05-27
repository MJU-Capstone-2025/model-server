# Coffee Price Prediction using LSTM with Entmax Attention

## 개요

Entmax 어텐션 메커니즘을 활용한 LSTM 기반 커피 가격 예측 모델. 경제적 지표(원유 가격, 환율)와 기상 데이터를 결합하여 다중 스텝 커피 가격 예측을 수행.

## 설치 요구사항

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn matplotlib
pip install entmax
```

## 실행 방법

### 1. 기본 실행

```bash
cd app
python models/time_series/lstm-entmax/run_model.py
```

기본 설정으로 모델을 실행:

-   윈도우 크기: 100
-   예측 구간: 14일
-   에폭 수: 5
-   배치 크기: 64
-   학습률: 0.001

### 2. 사용자 정의 파라미터로 실행

#### 데이터 관련 파라미터

```bash
# 윈도우 크기와 예측 구간 변경
python models/time_series/lstm-entmax/run_model.py --window 50 --horizon 7

# 슬라이딩 윈도우 스텝 크기 변경
python models/time_series/lstm-entmax/run_model.py --step 2
```

#### 모델 관련 파라미터

```bash
# LSTM 구조 변경
python models/time_series/lstm-entmax/run_model.py --hidden_size 128 --num_layers 3 --dropout 0.2

# 더 큰 모델로 실행
python models/time_series/lstm-entmax/run_model.py --hidden_size 256 --num_layers 4
```

#### 학습 관련 파라미터

```bash
# 에폭 수와 배치 크기 변경
python models/time_series/lstm-entmax/run_model.py --epochs 10 --batch_size 32

# 학습률 조정
python models/time_series/lstm-entmax/run_model.py --lr 0.0001

# 손실 함수 가중치 조정
python models/time_series/lstm-entmax/run_model.py --alpha 0.3 --beta 0.15
```

### 3. 고급 사용법

#### 시각화 없이 실행 (서버 환경)

```bash
python models/time_series/lstm-entmax/run_model.py --no_plot
```

#### CPU 강제 사용

```bash
python models/time_series/lstm-entmax/run_model.py --device cpu
```

#### 종합 예시

```bash
# 고성능 설정으로 실행
python models/time_series/lstm-entmax/run_model.py --epochs 20 --window 200 --hidden_size 256 --num_layers 3 --batch_size 32 --lr 0.0005 --dropout 0.15
```

## 명령행 옵션

### 데이터 관련

| 옵션        | 타입 | 기본값 | 설명                                      |
| ----------- | ---- | ------ | ----------------------------------------- |
| `--window`  | int  | 100    | 입력 시퀀스 길이 (과거 몇 일 데이터 사용) |
| `--horizon` | int  | 14     | 예측 구간 길이 (미래 몇 일 예측)          |
| `--step`    | int  | 1      | 슬라이딩 윈도우 스텝 크기                 |

### 모델 관련

| 옵션            | 타입  | 기본값 | 설명                |
| --------------- | ----- | ------ | ------------------- |
| `--hidden_size` | int   | 64     | LSTM 은닉 상태 크기 |
| `--num_layers`  | int   | 2      | LSTM 레이어 수      |
| `--dropout`     | float | 0.1    | 드롭아웃 비율       |

### 학습 관련

| 옵션                | 타입  | 기본값 | 설명               |
| ------------------- | ----- | ------ | ------------------ |
| `--epochs`          | int   | 5      | 학습 에폭 수       |
| `--batch_size`      | int   | 64     | 훈련 배치 크기     |
| `--test_batch_size` | int   | 32     | 테스트 배치 크기   |
| `--lr`              | float | 0.001  | 학습률             |
| `--alpha`           | float | 0.2    | 방향성 손실 가중치 |
| `--beta`            | float | 0.1    | 분산 손실 가중치   |

### 기타

| 옵션         | 타입 | 기본값 | 설명                        |
| ------------ | ---- | ------ | --------------------------- |
| `--no_plot`  | flag | False  | 시각화 생략                 |
| `--device`   | str  | auto   | 사용할 장치 (auto/cpu/cuda) |
| `--examples` | flag | False  | 사용 예시 출력 후 종료      |
| `--help`     | flag | False  | 도움말 출력                 |

## 사용 예시

### 빠른 테스트

```bash
# 빠른 테스트 (작은 윈도우, 적은 에폭)
python models/time_series/lstm-entmax/run_model.py --window 30 --epochs 2 --batch_size 32
```

### 정확도 중심 설정

```bash
# 높은 정확도를 위한 설정
python models/time_series/lstm-entmax/run_model.py --epochs 15 --window 150 --hidden_size 128 --num_layers 3 --lr 0.0005
```

### 메모리 절약 설정

```bash
# 메모리가 부족한 환경
python models/time_series/lstm-entmax/run_model.py --batch_size 16 --test_batch_size 8 --hidden_size 32 --device cpu
```

### 실험적 설정

```bash
# 다양한 손실 함수 가중치 실험
python models/time_series/lstm-entmax/run_model.py --alpha 0.5 --beta 0.2 --epochs 10
```

## 출력 결과

실행 시 다음과 같은 결과를 얻을 수 있다:

1. **콘솔 출력**: 학습 진행 상황, 성능 지표 (RMSE, MAE)
2. **시각화**: 학습 곡선, 예측 결과 그래프 (--no_plot 옵션 사용 시 생략)
3. **CSV 파일**: `app/data/output/prediction_result.csv`에 예측 결과 저장

## 도움말 및 예시

```bash
# 전체 옵션 확인
python models/time_series/lstm-entmax/run_model.py --help

# 사용 예시 확인
python models/time_series/lstm-entmax/run_model.py --examples
```

## 파일 구조

```
lstm-entmax/
├── run_model.py          # 메인 실행 파일
├── utils.py              # 유틸리티 함수
├── data_loader.py        # 데이터 로딩
├── preprocessor.py       # 데이터 전처리
├── dataset.py            # 데이터셋 클래스
├── models.py             # 모델 정의
├── losses.py             # 손실 함수
├── visualizer.py         # 시각화
├── trainer.py            # 학습 및 예측
└── README.md             # 이 파일
```

## 문제 해결

### 메모리 부족 오류

```bash
# 배치 크기 줄이기
python models/time_series/lstm-entmax/run_model.py --batch_size 16 --test_batch_size 8

# CPU 사용
python models/time_series/lstm-entmax/run_model.py --device cpu
```

### CUDA 오류

```bash
# CPU 강제 사용
python models/time_series/lstm-entmax/run_model.py --device cpu
```

### 시각화 오류 (서버 환경)

```bash
# 시각화 비활성화
python models/time_series/lstm-entmax/run_model.py --no_plot
```

## 성능 튜닝 가이드

### 정확도 향상

1. **윈도우 크기 증가**: `--window 200`
2. **모델 크기 증가**: `--hidden_size 128 --num_layers 3`
3. **에폭 수 증가**: `--epochs 15`
4. **학습률 조정**: `--lr 0.0005`

### 학습 속도 향상

1. **배치 크기 증가**: `--batch_size 128`
2. **윈도우 크기 감소**: `--window 50`
3. **모델 크기 감소**: `--hidden_size 32 --num_layers 1`

### 메모리 사용량 감소

1. **배치 크기 감소**: `--batch_size 16`
2. **모델 크기 감소**: `--hidden_size 32`
3. **CPU 사용**: `--device cpu`
