"""Shared SQLModel base class for all service tables (shared library, model layer)."""

from sqlmodel import SQLModel


class Base(SQLModel):
    """
    Common base class for all SQLModel tables.
    """

    pass

