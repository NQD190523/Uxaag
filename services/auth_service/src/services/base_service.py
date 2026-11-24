"""
Base Service Class - Provides common functionality for all services

Layer: service (shared base). Gives subclasses a logger and small logging helpers.
"""

import logging
from abc import ABC
from typing import Generic, TypeVar

from shared.logger import get_logger

T = TypeVar("T")


class BaseService(ABC, Generic[T]):
    """
    Abstract base service class that provides common functionality
    for all business logic services.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the base service.
        
        Args:
            logger: Logger instance for service logging
        """
        self.logger = logger or get_logger("auth-service")

    def log_info(self, message: str, *args) -> None:
        """Log an info message (``%s``-style ``args`` are optional)."""
        self.logger.info(message, *args)

    def log_error(self, message: str, exception: Exception = None) -> None:
        """Log an error message with a traceback.

        Pass ``exception`` or call from inside an ``except`` block so the traceback is
        recorded, as required for ERROR lines by CODING_CONVENTIONS 4a.
        """
        if exception is not None:
            self.logger.error(message, exc_info=exception)
        else:
            # exc_info=True is a no-op outside an except block.
            self.logger.error(message, exc_info=True)

    def log_debug(self, message: str, *args) -> None:
        """Log a debug message (``%s``-style ``args`` are optional)."""
        self.logger.debug(message, *args)

    def log_warning(self, message: str, *args) -> None:
        """Log a warning message (``%s``-style ``args`` are optional)."""
        self.logger.warning(message, *args)
