"""Auth service configuration (config layer).

Loads settings (database URL, JWT secret/algorithm, token lifetimes) from the
environment / the service's own ``.env`` into a single ``setting`` object.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Setting(BaseSettings):
    """Typed environment settings for the auth service.

    Required keys (no default): ``DATABASE_URL``, ``JWT_SECRET_KEY``,
    ``JWT_ALGORITHM``. The JWT secret must match the one used by the API gateway,
    otherwise the gateway rejects tokens issued here.
    """

    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    # Token lifetimes. Plain ints (no trailing commas, which would make them
    # tuples); pydantic parses the env-var strings with int().
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Reads auth_service/.env (src/core/config.py -> parents[2] is the service root);
    # unknown keys are ignored.
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
# Module-level singleton, built at import time (fails fast if a required key is missing).
setting  = Setting()