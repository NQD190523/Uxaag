"""Pytest setup for web-crawler-service: make ``src/`` importable like the running app.

Tests never touch the network: DNS is replaced by a fake resolver and HTTP by
``httpx.MockTransport``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
