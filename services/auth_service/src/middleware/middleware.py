"""Superseded: request logging now lives in the shared library.

The former ``LoggingMiddleware`` (``X-Correlation-ID``) was replaced by
``shared.request_logging.RequestLoggingMiddleware`` (registered in ``main.py``),
which handles the ``X-Request-ID`` header, request timing and log context.
This module is intentionally empty.
"""
