"""Async database engine and per-request session dependency (config/infrastructure layer).

Builds the SQLAlchemy async engine for the shared Supabase Postgres and exposes
``get_async_session`` for FastAPI ``Depends``.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from core.config import setting

# Create an asynchronous SQLAlchemy
engine: AsyncEngine = create_async_engine (
        setting.DATABASE_URL,
        echo=False,  # Set to False in production
        future=True,
        # SECURITY: keep bound values (tokens, emails, password hashes) out of DB error
        # messages, which end up in logs/auth-service/*.log via logger.exception.
        hide_parameters=True,
        pool_pre_ping=True,
        pool_recycle=300,
        # TODO(security): ssl is disabled here, which is only acceptable for local/dev connections;
        # revisit if the DB host requires TLS. command_timeout is in seconds.
        connect_args={
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
    """FastAPI dependency yielding one ``AsyncSession`` per request.

    The session is closed when the request finishes. Commits and rollbacks are
    done by the repositories, not here.

    Yields:
        An open async SQLModel session bound to the shared engine.
    """
    async with async_session_factory() as session:
        yield session
