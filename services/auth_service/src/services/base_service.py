"""
Base Service Class - Provides common functionality for all services

Layer: service (shared base). Gives subclasses a logger and small logging helpers.
"""

import logging
from abc import ABC
from typing import Generic, TypeVar

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
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_error(self, message: str, exception: Exception = None) -> None:
        """Log error message."""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
