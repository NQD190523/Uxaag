FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry
ENV POETRY_INSTALLER_MAX_WORKERS=1

WORKDIR /app/services/document_service

# Copy dependency files first for better Docker cache reuse.
COPY services/document_service/pyproject.toml services/document_service/poetry.lock ./
COPY libs/shared /app/libs/shared
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Copy source after dependencies are installed.
COPY services/document_service/src ./src

ENV PYTHONPATH=/app/services/document_service/src

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]
