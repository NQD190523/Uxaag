"""Logging setup shared by every microservice (shared library).

Call ``setup_logging("<service-name>")`` once in each service's ``main.py``; every
other module just does ``logger = get_logger("<service-name>")`` (or any name).

What it configures (on the root logger, so library and uvicorn logs are included):

- Console (stdout) handler.
- ``<LOG_DIR>/<service>/<service>.log``: INFO and above (or ``LOG_LEVEL``).
- ``<LOG_DIR>/<service>/<service>.error.log``: WARNING and above, with tracebacks.
- Both files rotate at 10 MB and keep 5 old files (about 100 MB per service at most).

Line format (text, default)::

    2026-11-18 10:21:33.123 | INFO  | chat-service | req=7f3a9c01b2d4 | user=42 | endpoints.create_chat_room:120 | Room created room_id=5

Environment variables:

- ``LOG_LEVEL``: DEBUG / INFO (default) / WARNING / ERROR.
- ``LOG_FORMAT``: ``text`` (default) or ``json`` (one JSON object per line).
- ``LOG_DIR``: base folder for log files (default ``uxaag/logs``; ``/app/logs`` in Docker).
- ``LOG_TO_FILE``: ``false`` to log to the console only.

Low overhead: records are put on an in-memory queue and written by one background
thread (``QueueListener``), so file I/O never blocks the event loop. Secrets
(tokens, passwords, Authorization values) are masked before anything is written.
"""

from __future__ import annotations

import atexit
from datetime import datetime
import json
import logging
import logging.handlers
import os
from pathlib import Path
import queue
import re
import shutil
import sys
from typing import Optional

from shared.log_context import request_id_var, user_id_var

# Rotation policy chosen by the author: 10 MB x 5 files per log file.
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

# Third-party loggers that are chatty at INFO; capped so they only report problems.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "sqlalchemy",
    "sqlalchemy.engine",
    "asyncio",
    "multipart",
    "watchfiles",
    "uvicorn.access",  # replaced by RequestLoggingMiddleware's one line per request
)

# libs/shared/shared/logger.py -> parents[3] is ``uxaag/`` locally and ``/app`` in Docker.
_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"

_listener: Optional[logging.handlers.QueueListener] = None
_atexit_registered = False
_service_name = "unknown"


class ContextFilter(logging.Filter):
    """Stamp each record with the service name and the current request/user id.

    Runs in the thread that logs (before the record is queued), which is where the
    request ``ContextVar``s hold the right values.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``service``, ``request_id`` and ``user_id`` attributes; never drops records."""
        record.service = _service_name
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True


# SECURITY: masks values that follow secret-looking keys, Bearer tokens and URL passwords.
_SECRET_PATTERNS = (
    # Bearer first, so "Authorization: Bearer <jwt>" masks the token, not the word.
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer ***"),
    (
        re.compile(
            r"(?i)\b(authorization|access_token|refresh_token|token|password|passwd|"
            r"secret|api[_-]?key|jwt_secret_key)(\"?'?\s*[:=]\s*\"?'?)([^\s\"',&}]+)"
        ),
        r"\1\2***",
    ),
    (re.compile(r"(://[^:/\s]+:)[^@/\s]+@"), r"\1***@"),
)


def redact(text: str) -> str:
    """Return ``text`` with secret values replaced by ``***``."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _QueueHandler(logging.handlers.QueueHandler):
    """Queue handler that freezes the message and traceback before queueing.

    The default implementation formats the whole line here; we only render the
    message (with redaction) and the traceback text, so the listener's formatters
    can still lay out the fields.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Return a picklable copy with ``msg`` rendered and ``exc_text`` filled in."""
        record = logging.makeLogRecord(record.__dict__)
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return record


class TextFormatter(logging.Formatter):
    """Readable one-line format: time | level | service | req | user | where | message."""

    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(service)s | "
                "req=%(request_id)s | user=%(user_id)s | "
                "%(module)s.%(funcName)s:%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class JSONFormatter(logging.Formatter):
    """One JSON object per line, for log tools (ELK, Loki, Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record's standard and context fields as JSON."""
        data = {
            # Local time with UTC offset, matching the text format's clock.
            "time": datetime.fromtimestamp(record.created).astimezone().isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": getattr(record, "service", _service_name),
            "request_id": getattr(record, "request_id", "-"),
            "user_id": getattr(record, "user_id", "-"),
            "logger": record.name,
            "where": f"{record.module}.{record.funcName}:{record.lineno}",
            "message": record.getMessage(),
        }
        if record.exc_text:
            data["traceback"] = record.exc_text
        return json.dumps(data, ensure_ascii=False)


_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _level(name: Optional[str]) -> int:
    """Map a level name to its number; unknown or empty names fall back to INFO."""
    return _LEVELS.get((name or "INFO").strip().upper(), logging.INFO)


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """``RotatingFileHandler`` that still rotates when Windows locks the file.

    On Windows a log file open in another program (``view-logs.bat`` uses
    ``Get-Content -Wait``, editors) cannot be renamed, which makes the standard handler
    fail and drop lines on every write. If the rename is refused, copy the file to the
    backup name and truncate the original instead.
    """

    def rotate(self, source: str, dest: str) -> None:
        """Rename ``source`` to ``dest``; fall back to copy + truncate if it is locked."""
        try:
            super().rotate(source, dest)
        except PermissionError:
            shutil.copyfile(source, dest)
            with open(source, "w", encoding="utf-8"):
                pass  # truncate in place; the open handle keeps working


class _DropDuplicateAsgiTraceback(logging.Filter):
    """Drop uvicorn's "Exception in ASGI application" record.

    ``RequestLoggingMiddleware`` already logs the same exception with the request id
    and user; uvicorn's copy has neither, so it only doubles the traceback.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for uvicorn's duplicate unhandled-exception record."""
        return not (
            record.name == "uvicorn.error"
            and str(record.msg).startswith("Exception in ASGI application")
        )


def _stop_listener() -> None:
    """Flush and stop the current queue listener (registered once with ``atexit``)."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def setup_logging(
    service_name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the whole process of one service (call once in ``main.py``).

    Calling it again (e.g. on uvicorn reload) replaces the previous setup cleanly.

    Args:
        service_name: Service name used in every line and for the log folder,
            e.g. ``"chat-service"``.
        log_level: Level name; ``LOG_LEVEL`` env var wins when set. Default INFO.
        log_file: Deprecated, ignored. Files are always
            ``<LOG_DIR>/<service>/<service>.log`` and ``.error.log``.

    Returns:
        The service's named logger.
    """
    global _listener, _service_name, _atexit_registered
    _service_name = service_name
    level = _level(os.getenv("LOG_LEVEL") or log_level)
    formatter: logging.Formatter = (
        JSONFormatter() if os.getenv("LOG_FORMAT", "text").lower() == "json" else TextFormatter()
    )

    # Stop a previous listener (reload) so its thread and files are released.
    _stop_listener()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    sinks: list[logging.Handler] = [console]

    if os.getenv("LOG_TO_FILE", "true").lower() != "false":
        log_dir = Path(os.getenv("LOG_DIR") or _DEFAULT_LOG_DIR) / service_name
        log_dir.mkdir(parents=True, exist_ok=True)
        main_file = SafeRotatingFileHandler(
            log_dir / f"{service_name}.log", maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT, encoding="utf-8", delay=True,
        )
        main_file.setLevel(level)
        main_file.setFormatter(formatter)
        error_file = SafeRotatingFileHandler(
            log_dir / f"{service_name}.error.log", maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT, encoding="utf-8", delay=True,
        )
        error_file.setLevel(logging.WARNING)
        error_file.setFormatter(formatter)
        sinks += [main_file, error_file]

    # Every logger propagates to root -> one queue -> one background writer thread.
    log_queue: queue.SimpleQueue = queue.SimpleQueue()
    queue_handler = _QueueHandler(log_queue)
    queue_handler.addFilter(ContextFilter())
    queue_handler.addFilter(_DropDuplicateAsgiTraceback())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(level)

    _listener = logging.handlers.QueueListener(log_queue, *sinks, respect_handler_level=True)
    _listener.start()
    # Flush queued lines on shutdown. Registered once: a second stop() on an already
    # stopped listener raises on Python 3.11 (used in Docker).
    if not _atexit_registered:
        atexit.register(_stop_listener)
        _atexit_registered = True

    # uvicorn installs its own handlers; route its loggers through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    logging.captureWarnings(True)

    service_logger = logging.getLogger(service_name)
    service_logger.handlers.clear()  # drop handlers left by older get_logger versions
    service_logger.setLevel(logging.NOTSET)  # inherit root level
    service_logger.propagate = True
    service_logger.info(
        "Logging configured level=%s format=%s files=%s",
        logging.getLevelName(level),
        "json" if isinstance(formatter, JSONFormatter) else "text",
        "on" if len(sinks) > 1 else "off",
    )
    return service_logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger; all output goes through the handlers set by ``setup_logging``.

    Safe to call at import time anywhere. No handlers are added here: until
    ``setup_logging`` runs, Python's fallback prints WARNING and above to stderr.

    Args:
        name: Logger name, usually the service name (``"auth-service"``) or ``__name__``.
    """
    return logging.getLogger(name)


def enable_console_logging_for_all(log_level: str = "INFO") -> None:
    """Deprecated no-op kept for old imports: ``setup_logging`` already covers all loggers."""
    return None
