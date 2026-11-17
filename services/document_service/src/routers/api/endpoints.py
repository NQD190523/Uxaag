"""Backwards-compatible shim (router layer): re-exports ``api.endpoints.router``.

NOTE: Nothing in ``main.py`` imports this module; it only keeps old import paths working.
"""

from api.endpoints import router

__all__ = ["router"]
