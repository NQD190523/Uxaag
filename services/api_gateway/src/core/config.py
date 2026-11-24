"""Gateway configuration (config layer).

Loads ``.env`` for the gateway, reads the upstream service table from
``core/services.yaml`` (next to this module) into ``SERVICES`` and exposes the JWT settings used by
``core/security.py``.

Optional env ``GATEWAY_UPSTREAM_HOST`` (e.g. ``127.0.0.1``) replaces the hostname of
every upstream URL at load time, so a local non-Docker run can reach services that
``services.yaml`` names by their docker-compose hostnames. Unset = URLs used as-is.
"""

import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit
from dotenv import load_dotenv
import yaml

from shared.logger import get_logger

logger = get_logger("api-gateway")


# Directories
# __file__ -> .../services/api_gateway/src/core/config.py
SRC_DIR = Path(__file__).resolve().parents[2]   # .../services/api_gateway (the service root, despite the name)
SERVICE_ROOT = Path(__file__).resolve().parents[3]  # .../services (parent of the service root)
# services.yaml sits next to this module (src/core/services.yaml).
SERVICES_YAML_PATH = Path(__file__).resolve().parent / "services.yaml"

# Load .env from the *service root* (where services/api_gateway/.env lives)
env_path = SRC_DIR / ".env"
load_dotenv(dotenv_path=env_path)



def load_service_config() -> Dict[str, Any]:
    """Load the upstream service table from ``src/core/services.yaml``.

    Returns:
        Mapping of service name to its settings (``url``, ``prefix``, ``strip_prefix``,
        ``timeout``, ``protected``). Empty if the file has no ``services`` key.

    Raises:
        FileNotFoundError: If ``services.yaml`` is missing.
    """
    config_path = SERVICES_YAML_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    services = data.get("services",{})
    return apply_upstream_host_override(services, os.getenv("GATEWAY_UPSTREAM_HOST", "").strip())


def apply_upstream_host_override(services: Dict[str, Any], host: str) -> Dict[str, Any]:
    """Replace the hostname of each service ``url`` with ``host``, keeping scheme, port and path.

    Why: ``services.yaml`` uses docker-compose hostnames (``auth-service`` ...), which
    do not resolve when every service runs locally on Windows. Overriding by env keeps
    the YAML Docker-ready while letting a local launcher point the gateway at
    ``127.0.0.1``. Only the host changes, so routes and the ``protected`` flags are
    untouched.

    Args:
        services: Service table loaded from ``services.yaml`` (modified in place).
        host: Replacement hostname; empty string means no override.

    Returns:
        The same ``services`` mapping, for chaining.
    """
    if not host:
        return services
    for name, cfg in services.items():
        url = cfg.get("url") if isinstance(cfg, dict) else None
        if not url:
            continue
        parts = urlsplit(url)
        netloc = f"{host}:{parts.port}" if parts.port else host
        cfg["url"] = urlunsplit(parts._replace(netloc=netloc))
        # Upstream URLs carry no credentials, so logging them is safe.
        logger.info("Upstream override service=%s url=%s", name, cfg["url"])
    return services

# SECURITY: JWT_SECRET_KEY must match the auth service's secret. Fail fast at import
# time instead of silently verifying tokens against an empty key (value is never printed).
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is not set or empty; set it in the gateway environment "
        "(it must match the auth service's secret)."
    )
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30)
JWT_REFRESH_TOKEN_EXPIRE_DAYS = os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", 7)
# Loaded once at import time; restart the gateway after editing services.yaml.
SERVICES: Dict[str, Any] = load_service_config()
