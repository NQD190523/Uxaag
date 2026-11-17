# api-gateway

API Gateway microservice for RAG Chat Room Web.

This service:

- Proxies requests to downstream microservices
- Applies auth, rate limiting, and routing
- Uses the shared library for common utilities:
  - `shared.logger` for logging
  - `shared.models` for shared SQLModel base (if needed)

Run with:

```bash
poetry install
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

