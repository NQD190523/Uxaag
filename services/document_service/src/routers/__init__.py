"""Legacy-path shim package: re-exports ``router`` from ``api.endpoints``.

NOTE: the real router moved to ``src/api/endpoints.py``; nothing imports this package.
Kept only so old import paths keep working (deletion candidate)."""

from api.endpoints import router

__all__ = ["router"]
