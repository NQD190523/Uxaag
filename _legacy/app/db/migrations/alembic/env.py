import asyncio
from logging.config import fileConfig
import sys
from pathlib import Path
import os


from sqlalchemy import engine_from_config
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import pool

from alembic import context



# Add project root to path
# File is at: app/db/migrations/alembic/env.py
# parents[0] = alembic/, [1] = migrations/, [2] = db/, [3] = app/, [4] = project root
base_dir = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(base_dir))

# Import models
from db.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
    
# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
    
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set the database URL from environment variable
database_url = os.getenv("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)
else:
    raise ValueError("DATABASE_URL is not set")

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.
def do_run_migrations(connection):
    """
    This function is called by Alembic inside a synchronous context.
    It receives a synchronous-like connection (actually a proxy).
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Optional common settings
        compare_type=True,      # detect column type changes
        compare_server_default=True,  # detect default changes
        render_as_batch=True,   # needed for SQLite, good for others too
    )

    with context.begin_transaction():
        context.run_migrations()
# ------------------------------------------------------------------
# Async runner – the heart of async support
# ------------------------------------------------------------------
async def run_async_migrations(connectable: AsyncEngine):
    """
    Open an async connection, then run the sync migration function
    using connection.run_sync().
    """
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    # Clean up the engine (optional but good practice)
    await connectable.dispose()

def run_migrations_offline():
    """
    Run migrations in 'offline' mode – generates SQL scripts.
    """
    url = config.get_main_option("sqlalchemy.url")

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
    """
    Run migrations in 'online' mode.
    This is called when you run alembic commands from the CLI.
    """
    # Get a pre-created engine from the config if someone passed it
    # (rarely used, but supported)
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # Otherwise create the async engine from the sqlalchemy.url in alembic.ini
        # You can also read other options like echo, future, etc.
        sqlalchemy_url = config.get_main_option("sqlalchemy.url")

        connectable = create_async_engine(
            sqlalchemy_url,
            poolclass=pool.NullPool,  # recommended for migrations
            echo=True, # for debugging
            # future=True, already default in SA 2.0
        )

    # Run the async migration function
    asyncio.run(run_async_migrations(connectable))


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
