# Build from repository root: docker build -f services/chat_service/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app/services/chat_service

# Copy dependency files first for better Docker cache reuse.
COPY services/chat_service/pyproject.toml services/chat_service/poetry.lock ./
COPY libs/shared /app/libs/shared

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Copy source after dependencies are installed.
COPY services/chat_service/src ./src

ENV PYTHONPATH=/app/services/chat_service/src

EXPOSE 8005

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--app-dir", "src"]
