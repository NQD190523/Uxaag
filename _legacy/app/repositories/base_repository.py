"""
Base repository classes for database operations.
"""

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base repository class with common database operations."""
    
    def __init__(self, session: Session):
        self.session = session


class AsyncBaseRepository:
    """Base async repository class with common database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
