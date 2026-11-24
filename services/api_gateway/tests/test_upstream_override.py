"""Tests for the GATEWAY_UPSTREAM_HOST override in ``core/config.py``.

Run from ``services/api_gateway``: ``.venv\\Scripts\\python.exe -m pytest tests``.
No network, DB or upstream service is touched.
"""

import os
import sys
from pathlib import Path

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# core.config refuses to import without a JWT secret; a dummy value is enough here.
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret")

from core.config import apply_upstream_host_override  # noqa: E402


def _table():
    """Return a small services table shaped like services.yaml."""
    return {
        "auth": {"url": "http://auth-service:8006/api/v1/auth", "protected": False},
        "chat": {"url": "http://chat-service:8005/api/v1/chat", "protected": True},
        "noport": {"url": "http://somehost/api", "protected": True},
        "nourl": {"protected": True},
    }


def test_empty_host_leaves_urls_unchanged():
    """Unset override (Docker) keeps the docker-compose hostnames."""
    services = apply_upstream_host_override(_table(), "")
    assert services["auth"]["url"] == "http://auth-service:8006/api/v1/auth"


def test_host_replaced_port_path_and_flags_kept():
    """Only the hostname changes; port, path and protected flag stay."""
    services = apply_upstream_host_override(_table(), "127.0.0.1")
    assert services["auth"]["url"] == "http://127.0.0.1:8006/api/v1/auth"
    assert services["chat"]["url"] == "http://127.0.0.1:8005/api/v1/chat"
    assert services["chat"]["protected"] is True
    assert services["auth"]["protected"] is False


def test_url_without_port_and_entry_without_url():
    """A URL without a port gets just the new host; entries without url are skipped."""
    services = apply_upstream_host_override(_table(), "127.0.0.1")
    assert services["noport"]["url"] == "http://127.0.0.1/api"
    assert "url" not in services["nourl"]
