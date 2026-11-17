# auth-service

Authentication microservice for RAG Chat Room Web.

This service is a standalone Poetry project. It depends on:

- `shared` (path dependency at `../../libs/shared`) for:
  - `shared.logger` (common logging utilities)
  - `shared.models` (shared SQLModel `Base`)

Run with:

```bash
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8006
```

