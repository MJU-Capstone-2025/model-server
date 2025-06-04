FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 불필요한 파일들 제거 (선택사항)
RUN rm -rf \
    models/ \
    notebooks/ \
    tests/ \
    *.md \
    .git/ \
    __pycache__/ || true

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "_main:app", "--host", "0.0.0.0", "--port", "8000"]
