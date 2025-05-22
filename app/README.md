# 생두 가격 예측 모델 서버

## 실행 방법

### 터미널에서 실행

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

### url로 실행

```
http://localhost:8000/train?loss_fn=huber&delta=0.5&epochs=10&lr=0.001
```

## 파이프라인

![pipeline](../docs/pipeline.drawio.svg)

파이프라인 다이어그램은 시계열 분석 기반의 커피 생두 가격 예측 모델의 전체 작동 과정을 보여주는 그림이다. 다이어그램에는 다음과 같은 주요 단계들이 포함되어 있다:

1. 데이터 로드 및 전처리: 날씨 데이터를 불러와서 lag 변수를 제거하고, 강수량(PRECTOTCORR) 컬럼만 남기는 등의 전처리를 수행하는 과정이다.

2. 특성 엔지니어링: 범주형 변수를 인코딩하고 변동성 관련 파생변수를 추가하는 등의 과정을 통해 데이터를 모델링에 적합한 형태로 가공하는 단계다.

3. 모델 설정 및 훈련: LSTM과 Attention 메커니즘을 결합한 모델을 구성하고, 전처리된 데이터로 학습을 진행하는 과정이다.

4. 예측 및 평가: 기본 예측 방식과 온라인 업데이트 예측 방식을 활용해 모델의 성능을 평가하는 단계다.

5. 결과 시각화 및 저장: 예측 결과를 다양한 형태로 시각화하고 CSV 파일로 저장하는 과정이다.

6. 슬라이딩 윈도우 예측: 연속적인 시간 구간에 대해 예측을 수행하고 결과를 시각화하는 단계다.

> 참고: [모델 링크](./models/time_series/lstm/)
