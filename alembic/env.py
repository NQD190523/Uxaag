from logging.config import fileConfig
import os
from pathlib import Path
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add project root to path
# File is at: alembic/env.py
# parents[0] = env.py, [1] = alembic/, [2] = project root
base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

# Load environment variables FIRST (before importing models)
from dotenv import load_dotenv
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL from environment variable BEFORE importing models
# Convert async URL to sync URL for Alembic autogenerate
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Alembic autogenerate needs sync connection, not async
    sync_url = database_url
    if "postgresql+asyncpg://" in sync_url:
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql://")
    elif "sqlite+aiosqlite://" in sync_url:
        sync_url = sync_url.replace("sqlite+aiosqlite://", "sqlite://")
    
    config.set_main_option("sqlalchemy.url", sync_url)
else:
    raise ValueError("DATABASE_URL is not set")

# Import models AFTER setting up the URL
# This prevents async operations during model import
from sqlmodel import SQLModel
from shared.models.base import Base
from services.auth_service.src.models.user import User, RefreshToken
from services.chat_service.src.models.chat import chatRoom, chatMessage, chatParticipant

# add your model's MetaData object here
# for 'autogenerate' support
# SQLModel uses SQLModel.metadata, not Base.metadata
target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    
    # Ensure sync URL for offline mode
    if "postgresql+asyncpg://" in url:
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    elif "sqlite+aiosqlite://" in url:
        url = url.replace("sqlite+aiosqlite://", "sqlite://")
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    Uses sync connections - Alembic doesn't need async for migrations.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=True,  # needed for SQLite, good for others too
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
