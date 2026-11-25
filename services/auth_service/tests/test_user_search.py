"""Tests for user search hardening (security review, 2026-11-26).

``UserRepository.get_matching_users`` must treat ``%`` and ``_`` literally and return at
most ``MATCHING_USERS_LIMIT`` rows. Runs the real query on in-memory SQLite (never
Supabase) through a tiny async adapter.
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Imports in this service are relative to src/ (see .claude/rules/backend.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.user import User  # noqa: E402
from repositories import user_repository as repo_mod  # noqa: E402


class _AsyncAdapter:
    """Async ``execute`` over a sync SQLAlchemy session."""

    def __init__(self, session):
        self._s = session

    async def execute(self, *args, **kwargs):
        return self._s.execute(*args, **kwargs)


def _search(q, names):
    engine = create_engine("sqlite://")
    User.__table__.create(engine)
    with Session(engine) as session:
        for i, name in enumerate(names, start=1):
            session.add(User(
                user_id=i, username=name, email=f"{name}@example.com",
                full_name=name, hashed_password="x",
            ))
        session.commit()
        users = asyncio.run(repo_mod.UserRepository(_AsyncAdapter(session)).get_matching_users(q))
        return [u.username for u in users]


def test_percent_does_not_match_everyone():
    assert _search("%", ["alice", "bob", "50%off"]) == ["50%off"]


def test_underscore_is_literal():
    assert _search("_", ["a_b", "ab", "abc"]) == ["a_b"]


def test_results_are_capped():
    names = [f"user{i}" for i in range(repo_mod.MATCHING_USERS_LIMIT + 5)]
    assert len(_search("user", names)) == repo_mod.MATCHING_USERS_LIMIT


def test_normal_search_still_works():
    assert _search("Ali", ["alice", "bob"]) == ["alice"]
