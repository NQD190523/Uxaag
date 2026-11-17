# Using the shared package in services (no sys.path)

The `shared` folder is an installable package. Each service declares it as a path dependency and imports `shared` directly.

## 1. Package layout

- `services/pyproject.toml` defines the **rag-chat-room-web-shared** package and exposes the `shared` package (the `shared/` directory).
- Installing from `services/` (e.g. `pip install -e services/` or Poetry path dep `".."`) makes `from shared.xxx` work.

## 2. In each service that uses shared

**In the service’s `pyproject.toml` (e.g. `services/auth_service/pyproject.toml`):**

```toml
[tool.poetry.dependencies]
rag-chat-room-web-shared = { path = "..", develop = true }
```

`path = ".."` is relative to the service directory, so it points at `services/`.

**In the service code:**

```python
from shared.logger import setup_logging, get_logger
from shared.models.base import Base
```

Do **not** add the project root or `shared` to `sys.path` in the service.

## 3. Install

From the service directory (e.g. `services/auth_service/`):

```bash
poetry lock
poetry install
```

Or from repo root:

```bash
poetry -C services/auth_service lock
poetry -C services/auth_service install
```

## 4. Adding shared to another service

1. Add to that service’s `pyproject.toml`:
   ```toml
   [tool.poetry.dependencies]
   rag-chat-room-web-shared = { path = "..", develop = true }
   ```
2. Use `from shared.xxx` in code.
3. Run `poetry lock` and `poetry install` in that service.

No `sys.path` changes are needed in any service.
