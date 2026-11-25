"""Web crawler settings (config layer), read from the environment once at import.

``main.py`` loads the service-local ``.env`` before importing anything that reads these
values. Every limit has a safe default, and a malformed value falls back to that default
instead of crashing the service.
"""

import os


def _int_env(name: str, default: int) -> int:
    """Return ``int(os.environ[name])``, or ``default`` when unset or not an integer."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# Gateway base URL used for the membership check and for saving files.
# Local runs: http://127.0.0.1:8000; Docker: http://api-gateway:8000.
API_GATEWAY_URL: str = (os.getenv("API_GATEWAY_URL") or "http://127.0.0.1:8000").rstrip("/")

# Most URLs accepted in one request. The gateway timeout (120 s) must cover them all.
CRAWL_MAX_URLS: int = _int_env("CRAWL_MAX_URLS", 5)
# Per page: total seconds for the whole fetch (every redirect hop, DNS and body),
# largest body read, redirects followed.
CRAWL_TIMEOUT_SECONDS: int = _int_env("CRAWL_TIMEOUT_SECONDS", 15)
CRAWL_MAX_BYTES: int = _int_env("CRAWL_MAX_BYTES", 2 * 1024 * 1024)
CRAWL_MAX_REDIRECTS: int = _int_env("CRAWL_MAX_REDIRECTS", 5)

# Room folder (relative to the room root) that crawled pages are saved into.
CRAWL_SAVE_FOLDER: str = os.getenv("CRAWL_SAVE_FOLDER", "web").strip("/") or "web"

# Timeout for calls to the gateway (membership check and saves).
GATEWAY_TIMEOUT_SECONDS: int = _int_env("GATEWAY_TIMEOUT_SECONDS", 15)

# Sent on every page fetch so site owners can identify the traffic.
CRAWLER_USER_AGENT: str = "UxaagCrawler/1.0 (+room document import)"
