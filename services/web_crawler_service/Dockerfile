FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/services/web_crawler_service

COPY services/web_crawler_service/src ./src

ENV PYTHONPATH=/app/services/web_crawler_service/src

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--app-dir", "src"]
