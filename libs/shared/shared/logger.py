"""Shared logging helpers for all microservices (shared library).

Provides JSON and plain formatters plus ``setup_logging`` / ``get_logger`` /
``enable_console_logging_for_all``. Format is chosen by the ``ENVIRONMENT`` env var:
``production`` gives JSON lines, anything else gives readable text.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Optional
import os


class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format
    Perfect for log aggregation tools like ELK, Datadog, CloudWatch
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record as one JSON line.

        NOTE: only ``service_name``, ``request_id``, ``user_id`` and ``extra`` record
        attributes are read. Passing ``extra={...}`` to a logger call sets each key as its
        own attribute (it does not create ``record.extra``), so other keys are dropped.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service_name", "unknown"),
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add request_id if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add user_id if available
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }

        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    """Simple human-readable formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format as ``[LEVEL] logger - message`` (extras and tracebacks are omitted)."""
        return f"[{record.levelname}] {record.name} - {record.getMessage()}"


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for a microservice.

    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger instance

    Raises:
        AttributeError: If ``log_level`` is not a valid level name.
    """

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # Create formatter
    # Use JSON in production, plain text in development
    env_value = os.getenv("ENVIRONMENT", "development")
    is_production = env_value.lower() == "production"
    formatter = JSONFormatter() if is_production else PlainFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Startup banner goes through the logger itself (after the handler exists), not print().
    logger.info(
        "ENVIRONMENT=%s, using %s formatter for %s",
        env_value,
        "JSON" if is_production else "Plain",
        service_name,
    )

    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(service_name: str) -> logging.Logger:
    """Get a logger for a service, adding a stdout handler only if it has none.

    Safe to call at import time in every module. A logger already set up with
    ``setup_logging`` is returned unchanged.

    Args:
        service_name: Logger name, e.g. ``"auth-service"``.

    Returns:
        The named ``logging.Logger`` (level INFO when auto-configured).
    """

    logger = logging.getLogger(service_name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Use JSON in production, plain text in development (same logic as setup_logging)
        # Load environment variable, defaulting to development
        env_value = os.getenv("ENVIRONMENT", "development")
        is_production = env_value.lower() == "production"
        formatter = JSONFormatter() if is_production else PlainFormatter()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def enable_console_logging_for_all(log_level: str = "INFO") -> None:
    """
    Enable console logging for all loggers (root logger), so that logs
    from every module/service propagate to stdout.

    Args:
        log_level: Level name for the root logger and its handler (default INFO).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # Use JSON in production, plain text in development (same logic as setup_logging)
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        formatter = JSONFormatter() if is_production else PlainFormatter()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Route warnings.warn() output through logging (this does not change propagation;
    # named loggers propagate to root by default unless propagate is turned off).
    logging.captureWarnings(True)   

