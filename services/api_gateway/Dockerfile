# Build from repository root: docker build -f services/api_gateway/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app/services/api_gateway

# Copy dependency files first for better Docker cache reuse.
COPY services/api_gateway/pyproject.toml services/api_gateway/poetry.lock ./
# Monorepo dependency for this service.
COPY libs/shared /app/libs/shared

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Copy source after dependencies are installed.
COPY services/api_gateway/src ./src

ENV PYTHONPATH=/app/services/api_gateway/src

EXPOSE 8000

# Use uvicorn so the app binds to 0.0.0.0 in containers (main.py defaults to 127.0.0.1)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
