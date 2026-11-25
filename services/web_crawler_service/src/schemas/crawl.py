"""Crawl request/response models (schemas layer) for ``POST /crawl/rooms/{id}/pages``."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints

from core import config


class CrawlPagesRequest(BaseModel):
    """URLs to import into the room, one Markdown file per URL.

    Duplicates are removed by the service. The count is capped so one request stays
    within the gateway timeout.
    """

    # Each URL is capped at 2048 chars (the usual browser limit) so one item cannot be
    # megabytes long.
    urls: list[Annotated[str, StringConstraints(max_length=2048)]] = Field(
        ...,
        min_length=1,
        max_length=config.CRAWL_MAX_URLS,
        description="Absolute http(s) URLs",
    )


class CrawlPageResult(BaseModel):
    """Outcome for one URL. ``relative_path`` and ``title`` only when ``status == "ok"``."""

    url: str
    status: Literal["ok", "error"]
    relative_path: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None


class CrawlPagesResponse(BaseModel):
    """Per-URL results. ``success`` is True when at least one page was saved."""

    success: bool
    chat_room_id: int
    saved: int
    results: list[CrawlPageResult]
