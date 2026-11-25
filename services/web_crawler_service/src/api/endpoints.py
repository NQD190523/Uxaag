"""Web crawler HTTP endpoints (router layer), prefix ``/crawl`` (mounted under ``/api/v1``).

Gateway route: ``/crawl/<rest>`` -> ``/api/v1/crawl/<rest>`` (protected), so callers
reach this with ``x-user-id`` set by the gateway and their own ``Authorization`` header,
which is forwarded on to chat- and document-service.
"""

from fastapi import APIRouter, Header, HTTPException, status
from shared.logger import get_logger

from clients.gateway_client import GatewayAuthError, GatewayUnavailable, RoomAccessDenied
from schemas.crawl import CrawlPagesRequest, CrawlPagesResponse
from services.crawl_service import import_pages

logger = get_logger("web-crawler-service")

router = APIRouter(prefix="/crawl", tags=["crawl"])


@router.post("/rooms/{chat_room_id}/pages", response_model=CrawlPagesResponse)
async def crawl_pages(
    chat_room_id: int,
    request: CrawlPagesRequest,
    x_user_id: str = Header(..., alias="x-user-id"),
    authorization: str | None = Header(default=None),
) -> CrawlPagesResponse:
    """POST /crawl/rooms/{chat_room_id}/pages - import web pages as Markdown files.

    Each URL is fetched (public http(s) addresses only), converted to Markdown and saved
    as ``web/<title-slug>.md`` in the room's document workspace.

    Auth: ``x-user-id`` + Bearer token; caller must be a room member.
    Body: ``CrawlPagesRequest`` (1 to ``CRAWL_MAX_URLS`` URLs).
    Returns: ``CrawlPagesResponse``. Status 200 even when some or all URLs failed; read
    each result's ``status`` and ``error``.
    Errors: 401 missing/rejected token, 403 not a member, 422 bad body,
    503 membership could not be checked.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    try:
        results = await import_pages(
            chat_room_id=chat_room_id, urls=request.urls, authorization=authorization
        )
    except GatewayAuthError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token rejected")
    except RoomAccessDenied:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not a member of this room")
    except GatewayUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Room membership could not be verified, try again later",
        )
    saved = sum(1 for r in results if r.status == "ok")
    logger.info("Crawl done room_id=%s user_id=%s saved=%s of=%s", chat_room_id, x_user_id, saved, len(results))
    return CrawlPagesResponse(success=saved > 0, chat_room_id=chat_room_id, saved=saved, results=results)
