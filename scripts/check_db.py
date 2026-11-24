"""Database connectivity preflight for the local startup script.

Reads ``DATABASE_URL`` from a service's ``.env`` file (default:
``services/chat_service/.env``) and checks, step by step, that the shared
Supabase Postgres is reachable: URL parse -> DNS -> TCP -> Postgres login ->
expected tables. On failure it prints which step failed and the likely cause.

SECURITY: the URL, user name and password are never printed. Only scheme,
host, port, database name and password length are shown.

Usage (run with a service venv python that has ``asyncpg`` installed)::

    python scripts/check_db.py [path\\to\\.env]

Exit codes: 0 = connected, 1 = failed (details printed), 2 = asyncpg missing.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

# uxaag/ folder (this file lives in uxaag/scripts/).
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT_DIR / "services" / "chat_service" / ".env"
# Tables created by Alembic (auth + chat models); missing ones mean an old schema.
EXPECTED_TABLES = {"user", "refreshtoken", "chatroom", "chatmessage", "chatparticipant"}
TIMEOUT_S = 15


def fail(step: str, reason: str, hints: list[str]) -> None:
    """Print a formatted failure block and exit with code 1.

    Args:
        step: Name of the step that failed (e.g. ``"DNS"``).
        reason: Short description of the error (must not contain secrets).
        hints: Likely causes / fixes, printed as a bullet list.
    """
    print(f"  [FAIL] {step}: {reason}")
    for hint in hints:
        print(f"         - {hint}")
    sys.exit(1)


def read_database_url(env_path: Path) -> str:
    """Return ``DATABASE_URL`` from a dotenv file, without surrounding quotes.

    Why not python-dotenv: this script must also run with a bare interpreter, and
    only one key is needed. Later duplicates win, matching python-dotenv.

    Args:
        env_path: Path to the ``.env`` file.

    Returns:
        The raw URL string, or exits via ``fail`` if the file/key is missing.
    """
    if not env_path.exists():
        fail("Config", f"env file not found: {env_path}", ["Create it with DATABASE_URL=..."])
    url = ""
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("DATABASE_URL") and "=" in stripped:
            key, value = stripped.split("=", 1)
            if key.strip() == "DATABASE_URL":
                url = value.strip().strip('"').strip("'")
    if not url:
        fail("Config", f"DATABASE_URL missing or empty in {env_path}", [
            "Each line must look like DATABASE_URL=postgresql+asyncpg://...",
        ])
    return url


def classify_login_error(exc: BaseException) -> tuple[str, list[str]]:
    """Map an asyncpg/OS exception to a readable reason and hints.

    Args:
        exc: The exception raised by ``asyncpg.connect``.

    Returns:
        ``(reason, hints)``; the reason contains the error class and server
        message but never the connection URL.
    """
    import asyncpg  # local import: only reached when asyncpg is installed

    name = type(exc).__name__
    msg = str(exc).strip() or "(no message)"
    reason = f"{name}: {msg}"
    low = msg.lower()
    if isinstance(exc, asyncpg.InvalidPasswordError) or "password authentication" in low:
        return reason, [
            "Wrong password, or the password changed in Supabase (Settings > Database).",
            "Special characters in the password must be URL-encoded (@ -> %40, # -> %23).",
        ]
    if "tenant or user not found" in low:
        return reason, [
            "Pooler user must be 'postgres.<project-ref>' (with the project ref suffix).",
            "Check the pooler region/host matches your project (Connect > Session pooler).",
        ]
    if isinstance(exc, asyncpg.InvalidCatalogNameError):
        return reason, ["The database name after the last '/' does not exist (normally 'postgres')."]
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return reason, [
            "Server accepted TCP but did not finish the handshake in time.",
            "Project may be paused/restoring in Supabase, or the network blocks Postgres.",
        ]
    if "ssl" in low:
        return reason, ["Services connect with ssl=False; the pooler may now require SSL."]
    if isinstance(exc, ConnectionError) or "connection" in low:
        return reason, [
            "Connection dropped during login: project paused, pooler overloaded,",
            "or max client connections reached (check Supabase dashboard).",
        ]
    return reason, ["Unexpected error; see the message above."]


async def try_login(dsn: str) -> None:
    """Log in with the same asyncpg options chat-service uses and inspect the schema.

    Args:
        dsn: Plain ``postgresql://`` DSN (the ``+asyncpg`` driver suffix removed).
    """
    import asyncpg

    started = time.perf_counter()
    try:
        # Mirror chat_service/src/db/database.py connect_args (ssl=False) so a pass
        # here means the services will connect too. statement_cache_size=0 keeps the
        # check safe on the transaction pooler (port 6543), which has no prepared stmts.
        conn = await asyncpg.connect(
            dsn, timeout=TIMEOUT_S, ssl=False, statement_cache_size=0
        )
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, not raised
        if isinstance(exc, KeyboardInterrupt):
            raise
        reason, hints = classify_login_error(exc)
        fail("Postgres login", reason, hints)
        return
    try:
        elapsed = time.perf_counter() - started
        version = (await conn.fetchval("select version()")) or ""
        print(f"  [ OK ] Postgres login in {elapsed:.2f}s ({' '.join(version.split()[:2])})")
        rows = await conn.fetch(
            "select table_name from information_schema.tables where table_schema='public'"
        )
        tables = {r[0] for r in rows}
        missing = EXPECTED_TABLES - tables
        if missing:
            print(f"  [WARN] Missing tables: {', '.join(sorted(missing))}")
            print("         Run migrations from uxaag\\: alembic upgrade head")
        else:
            print(f"  [ OK ] All {len(EXPECTED_TABLES)} app tables present")
        if "alembic_version" in tables:
            head = await conn.fetchval("select version_num from alembic_version")
            print(f"  [ OK ] Alembic revision: {head}")
    finally:
        await conn.close()


def main() -> None:
    """Run the DNS -> TCP -> login checks and print a step-by-step report."""
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ENV
    url = read_database_url(env_path)
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 5432
    except ValueError as exc:
        fail("URL parse", str(exc), ["Check the DATABASE_URL format (port must be a number)."])
        return
    if not host:
        fail("URL parse", "no host found", [
            "Expected postgresql+asyncpg://USER:PASSWORD@HOST:PORT/DBNAME",
            "An unencoded '@' or '/' in the password breaks parsing; URL-encode it.",
        ])
    db_name = parts.path.lstrip("/") or "(none)"
    pw_len = len(unquote(parts.password or ""))
    mode = {5432: "session pooler / direct", 6543: "transaction pooler"}.get(port, "custom port")
    print(f"  Source : {env_path}")
    print(f"  Target : scheme={parts.scheme} host={host} port={port} ({mode}) db={db_name} password_len={pw_len}")
    if parts.scheme != "postgresql+asyncpg":
        print("  [WARN] Services expect the scheme 'postgresql+asyncpg://'.")

    try:
        addrs = sorted({info[4][0] for info in socket.getaddrinfo(host, port)})
    except socket.gaierror as exc:
        hints = ["Check the host spelling and your internet/DNS."]
        if host.startswith("db.") and host.endswith(".supabase.co"):
            hints.append("Supabase direct hosts (db.<ref>.supabase.co) are IPv6-only;")
            hints.append("use the Session pooler host (aws-...pooler.supabase.com) instead.")
        fail("DNS", f"cannot resolve {host} ({exc})", hints)
        return
    print(f"  [ OK ] DNS: {host} -> {', '.join(addrs)}")

    started = time.perf_counter()
    try:
        socket.create_connection((host, port), timeout=8).close()
    except OSError as exc:
        fail("TCP", f"cannot open {host}:{port} ({exc})", [
            "Firewall/VPN/corporate network may block outbound Postgres ports.",
            "Supabase project may be paused (free tier pauses after inactivity).",
        ])
    print(f"  [ OK ] TCP: {host}:{port} in {time.perf_counter() - started:.2f}s")

    try:
        import asyncpg  # noqa: F401
    except ImportError:
        print("  [SKIP] asyncpg not installed in this Python; login not tested.")
        sys.exit(2)
    asyncio.run(try_login(url.replace("postgresql+asyncpg://", "postgresql://", 1)))
    print("  Database OK.")


if __name__ == "__main__":
    main()
