"""
Database Manager for microservices.
"""

import asyncio
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
import logging

from .config import config
from app.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Centralized database manager for microservices."""
    
    def __init__(self, service_name: str):
        """
        Initialize database manager for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'auth', 'document', 'rag')
        """
        self.service_name = service_name
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    def initialize(self, database_url: Optional[str] = None, async_mode: bool = True):
        """
        Initialize database connections.
        
        Args:
            database_url: Database URL (uses config default if not provided)
            async_mode: Whether to use async connections
        """
        try:
            # Dispose of existing connections if re-initializing
            # This ensures old connections with potentially cached statements are closed
            if self.async_engine:
                try:
                    # Store reference for disposal
                    old_engine = self.async_engine
                    self.async_engine = None
                    # Dispose will be called when the old_engine is garbage collected
                    # or we can call it in close() method
                    logger.debug(f"Re-initializing database for {self.service_name}, old engine will be disposed")
                except Exception as e:
                    logger.warning(f"Error preparing to dispose old async engine: {e}")
            
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing old sync engine: {e}")
                self.engine = None
            
            # Get database URL
            db_url = database_url or config.get_database_url()
            
            # Add service-specific database name if using SQLite
            if "sqlite" in db_url and not db_url.endswith(".db"):
                db_url = db_url.replace("rag_chat_room_web.db", f"rag_chat_room_web_{self.service_name}.db")
            
            # For PostgreSQL, use the same database but different schemas per service
            if "postgresql" in db_url:
                # Keep the same database URL for PostgreSQL
                pass
            
            logger.info(f"Initializing database for {self.service_name}: {db_url}")
            
            if async_mode:
                # Create async engine
                if "postgresql" in db_url:
                    # Use asyncpg for PostgreSQL
                    if "postgresql+asyncpg://" not in db_url:
                        async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
                    else:
                        async_db_url = db_url
                else:
                    # Use aiosqlite for SQLite
                    if "sqlite+aiosqlite://" not in db_url:
                        async_db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://")
                    else:
                        async_db_url = db_url
                
                # Configure connect_args for asyncpg (pgbouncer compatibility)
                # Disable prepared statement cache to work with pgbouncer transaction/statement pooling
                connect_args = {
                    "ssl": False,
                    "command_timeout": 60,
                    "server_settings": {
                        "application_name": f"rag_chat_room_web_{self.service_name}",
                        "jit": "off",
                    },
                }

                self.async_engine = create_async_engine(
                async_db_url,
                echo=config.get("database.echo", False),
                pool_pre_ping=True,
                pool_recycle=300,
                connect_args=connect_args,
                )
                
                # Create async session factory
                self.async_session_factory = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
            else:
                # Create sync engine
                self.engine = create_engine(
                    db_url,
                    echo=config.get("database.echo", False),
                    pool_pre_ping=True,
                    pool_recycle=300,
                    connect_args={
                        "sslmode": "require" if "supabase" in db_url else None,
                        "connect_timeout": 60,
                        "application_name": f"rag_chat_room_web_{self.service_name}",
                    }
                )
                
                # Create session factory
                self.session_factory = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
            
            self._initialized = True
            logger.info(f"Database initialized successfully for {self.service_name}")
            
        except Exception as e:
            logger.exception(f"Failed to initialize database for {self.service_name}")
            raise
    
    async def create_tables(self):
        """Create all tables for the service."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")

        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            else:
                Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"Tables created successfully for {self.service_name}")
            
        except Exception as e:
            logger.exception(f"Failed to create tables for {self.service_name}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session."""
        if not self._initialized or not self.async_session_factory:
            raise RuntimeError("Async database not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    def get_sync_session(self):
        """Get sync database session."""
        if not self._initialized or not self.session_factory:
            raise RuntimeError("Sync database not initialized")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    result.fetchone()
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
            
            return {
                "status": "healthy",
                "service": self.service_name,
                "database": "connected"
            }
            
        except Exception as e:
            logger.exception(f"Database health check failed for {self.service_name}")
            return {
                "status": "unhealthy",
                "service": self.service_name,
                "database": "disconnected",
                "error": str(e)
            }
    
    async def close(self):
        """Close database connections."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            if self.engine:
                self.engine.dispose()
            
            logger.info(f"Database connections closed for {self.service_name}")
            
        except Exception as e:
            logger.exception(f"Error closing database connections for {self.service_name}")


# Global database managers for each service
_db_managers: Dict[str, DatabaseManager] = {}


def get_database_manager(service_name: str) -> DatabaseManager:
    """
    Get or create database manager for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        DatabaseManager instance
    """
    if service_name not in _db_managers:
        _db_managers[service_name] = DatabaseManager(service_name)
    
    return _db_managers[service_name]


async def initialize_service_database(service_name: str, database_url: Optional[str] = None):
    """
    Initialize database for a specific service.
    
    Args:
        service_name: Name of the service
        database_url: Optional custom database URL
    """
    db_manager = get_database_manager(service_name)
    db_manager.initialize(database_url)
    await db_manager.create_tables()


async def close_all_databases():
    """Close all database connections."""
    for db_manager in _db_managers.values():
        await db_manager.close()
    _db_managers.clear()
