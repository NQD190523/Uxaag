"""Database layer: async SQLAlchemy engine and session helpers for chat-service.

Connects to the shared Supabase Postgres (URL from ``config``). Use
``get_async_session`` for HTTP requests and ``get_background_session`` for
WebSocket / ``asyncio.create_task`` work that outlives a request.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from core.config import config
from shared.logger import get_logger

# Get configured logger
logger = get_logger("chat-service")
# Create an asynchronous SQLAlchemy engine
engine: AsyncEngine = create_async_engine (
        config.get_database_url(),
        echo=False,  # Set to True to log every SQL statement (noisy)
        future=True,
        # SECURITY: keep bound values (message text, ids) out of DB error messages,
        # which end up in logs/chat-service/*.log via logger.exception.
        hide_parameters=True,
        # Validate pooled connections before use and recycle them every 5 minutes
        # so stale connections dropped by the DB/pooler do not cause request errors.
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            # asyncpg options: TLS disabled and a 60 s per-query timeout.
            "ssl": False,
            "command_timeout": 60,
        },
)

# Session factory, create new session every request
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Dependency to get DB session
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding one request-scoped async session.

    The session is closed when the request finishes.

    Yields:
        An ``AsyncSession`` (``expire_on_commit=False``, ``autoflush=False``).
    """
    async with async_session_factory() as session:
        yield session
    
# For websocket/background flows where explicit context management is needed.
@asynccontextmanager
async def get_background_session() -> AsyncIterator[AsyncSession]:
    """Provide an explicit async context manager for non-request DB work.

    Why: request-scoped sessions are closed when the HTTP/WebSocket handler
    returns, so fire-and-forget tasks must open their own session.

    Yields:
        A fresh ``AsyncSession`` that is always closed on exit.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
