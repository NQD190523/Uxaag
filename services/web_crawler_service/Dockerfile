FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry
ENV POETRY_INSTALLER_MAX_WORKERS=1

WORKDIR /app/services/web_crawler_service

# Copy dependency files first for better Docker cache reuse.
# NOTE: poetry.lock is optional until it is generated (`poetry lock` in this folder).
COPY services/web_crawler_service/pyproject.toml services/web_crawler_service/poetry.lock* ./
COPY libs/shared /app/libs/shared
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Copy source after dependencies are installed.
COPY services/web_crawler_service/src ./src

ENV PYTHONPATH=/app/services/web_crawler_service/src

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--app-dir", "src"]
