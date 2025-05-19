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
