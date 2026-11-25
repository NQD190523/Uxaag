"""Crawl orchestration (services layer): import a list of URLs into a room.

Flow for ``POST /crawl/rooms/{id}/pages``:
    1. check room membership once (fail fast, so non-members cannot use the crawler
       to make our server fetch pages);
    2. for each unique URL, concurrently: fetch (SSRF-guarded) -> convert -> save into
       ``<CRAWL_SAVE_FOLDER>/`` in the room via document-service.

One URL failing does not stop the others; each gets its own result. Auth problems
during a save (token expired, membership revoked) are reported per URL too, because
some pages may already be saved by then.
"""

from __future__ import annotations

import asyncio

from shared.logger import get_logger

from clients.gateway_client import (
    DocumentSaveError,
    GatewayAuthError,
    GatewayUnavailable,
    RoomAccessDenied,
    check_room_membership,
    save_markdown_file,
)
from core import config
from core.url_guard import UrlNotAllowedError, safe_url_for_log
from schemas.crawl import CrawlPageResult
from services.html_to_markdown import build_document
from services.page_fetcher import FetchError, fetch_page

logger = get_logger("web-crawler-service")

# Pages fetched at the same time within one request.
_CONCURRENCY = 3


def unique_urls(urls: list[str]) -> list[str]:
    """Strip blanks and drop duplicates, keeping the first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        url = raw.strip()
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


async def import_pages(*, chat_room_id: int, urls: list[str], authorization: str) -> list[CrawlPageResult]:
    """Fetch every URL and save it as Markdown in the room.

    Args:
        chat_room_id: Target room.
        urls: URLs from the request (validated count, not yet de-duplicated).
        authorization: Caller's ``Bearer`` header, forwarded to the gateway.

    Returns:
        One result per unique URL, in request order.

    Raises:
        GatewayAuthError, RoomAccessDenied, GatewayUnavailable: From the up-front
            membership check; nothing was fetched.
    """
    await check_room_membership(chat_room_id=chat_room_id, authorization=authorization)
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def one(url: str) -> CrawlPageResult:
        async with semaphore:
            return await _import_one(chat_room_id=chat_room_id, url=url, authorization=authorization)

    return list(await asyncio.gather(*(one(u) for u in unique_urls(urls))))


async def _import_one(*, chat_room_id: int, url: str, authorization: str) -> CrawlPageResult:
    """Fetch, convert and save one URL; every expected failure becomes an error result."""
    log_url = safe_url_for_log(url)
    try:
        page = await fetch_page(url)
        title, stem, markdown = build_document(page)
        path = await save_markdown_file(
            chat_room_id=chat_room_id,
            folder=config.CRAWL_SAVE_FOLDER,
            stem=stem,
            content=markdown,
            authorization=authorization,
        )
    except (UrlNotAllowedError, FetchError, DocumentSaveError) as e:
        logger.info("Page not imported room_id=%s url=%s reason=%s", chat_room_id, log_url, e)
        return CrawlPageResult(url=url, status="error", error=str(e))
    except GatewayAuthError:
        return CrawlPageResult(url=url, status="error", error="Your session expired, sign in again")
    except RoomAccessDenied:
        return CrawlPageResult(url=url, status="error", error="You are no longer a member of this room")
    except GatewayUnavailable:
        return CrawlPageResult(url=url, status="error", error="Document service unavailable, try again later")
    except Exception:
        # Converter bugs on odd HTML must not fail the whole request.
        logger.exception("Unexpected error importing room_id=%s url=%s", chat_room_id, log_url)
        return CrawlPageResult(url=url, status="error", error="Unexpected error while importing this page")
    logger.info("Page imported room_id=%s url=%s path=%s", chat_room_id, log_url, path)
    return CrawlPageResult(url=url, status="ok", relative_path=path, title=title)
