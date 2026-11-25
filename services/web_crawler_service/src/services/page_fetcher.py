"""Page fetcher (services layer): download one web page safely.

Redirects are followed by hand rather than by httpx, so that the SSRF guard
(``core/url_guard.py``) runs on every hop. The body is streamed and reading stops at
``CRAWL_MAX_BYTES``, so a huge or endless response cannot exhaust memory.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from core import config
from core.url_guard import assert_url_allowed

# Content types the converter understands. Anything else (PDF, images, JSON) is refused.
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = frozenset({"text/plain", "text/markdown"})


class FetchError(Exception):
    """The page could not be fetched. The message is safe to show to the user."""


@dataclass(frozen=True)
class FetchedPage:
    """A downloaded page.

    Attributes:
        url: Final URL after redirects.
        content_type: Lower-cased media type without parameters, e.g. ``text/html``.
        body: Raw bytes. HTML stays bytes so BeautifulSoup can read the page's own
            ``<meta charset>``.
        encoding: Charset from the ``Content-Type`` header, if any.
    """

    url: str
    content_type: str
    body: bytes
    encoding: str | None


def _media_type(header: str | None) -> str:
    """Return ``"text/html"`` from ``"text/html; charset=utf-8"`` (lower-cased)."""
    return (header or "").split(";", 1)[0].strip().lower()


async def fetch_page(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> FetchedPage:
    """Fetch ``url``, following at most ``CRAWL_MAX_REDIRECTS`` redirects.

    Args:
        url: Absolute http(s) URL given by the user.
        transport: Test hook; production uses httpx's default network transport.

    Returns:
        The final page.

    Raises:
        UrlNotAllowedError: The URL or a redirect target fails the SSRF guard.
        FetchError: Network error, the total ``CRAWL_TIMEOUT_SECONDS`` deadline passed,
            non-200 status, unsupported content type, body over the size limit or too
            many redirects.
    """
    # One deadline for the whole fetch (all hops, DNS and body). httpx's own timeout
    # applies to each connect/read separately, so a server sending one byte every few
    # seconds could otherwise hold a fetch open for hours.
    try:
        async with asyncio.timeout(config.CRAWL_TIMEOUT_SECONDS):
            return await _fetch_with_redirects(url.strip(), transport)
    except TimeoutError as e:
        raise FetchError("The site took too long to answer") from e


async def _fetch_with_redirects(
    current: str, transport: httpx.AsyncBaseTransport | None
) -> FetchedPage:
    """Follow redirects by hand, running the SSRF guard before every hop.

    Raises:
        UrlNotAllowedError, FetchError: As described in ``fetch_page``.
    """
    # trust_env=False: an HTTP(S)_PROXY from the environment would make the proxy, not
    # us, connect to the target, so the address check would no longer protect anything.
    async with httpx.AsyncClient(
        timeout=config.CRAWL_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
        headers={
            "User-Agent": config.CRAWLER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        },
    ) as client:
        for _hop in range(config.CRAWL_MAX_REDIRECTS + 1):
            await assert_url_allowed(current)
            try:
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError("Redirect without a target")
                        # Location may be relative to the URL that sent it.
                        current = urljoin(current, location)
                        continue
                    return await _read_page(current, response)
            except httpx.TimeoutException as e:
                raise FetchError("The site took too long to answer") from e
            except httpx.HTTPError as e:
                raise FetchError("The site could not be reached") from e
    raise FetchError("Too many redirects")


async def _read_page(url: str, response: httpx.Response) -> FetchedPage:
    """Check status and type, then read the body up to the size limit.

    Raises:
        FetchError: Non-200 status, unsupported type or body too large.
    """
    if response.status_code != 200:
        raise FetchError(f"The site answered with HTTP {response.status_code}")
    media_type = _media_type(response.headers.get("content-type"))
    if media_type not in HTML_TYPES | TEXT_TYPES:
        raise FetchError(f"Unsupported content type: {media_type or 'unknown'}")

    limit = config.CRAWL_MAX_BYTES
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise FetchError("The page is too large to import")
    chunks: list[bytes] = []
    size = 0
    # Count bytes as they arrive: Content-Length can be missing or wrong.
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > limit:
            raise FetchError("The page is too large to import")
        chunks.append(chunk)
    return FetchedPage(
        url=url,
        content_type=media_type,
        body=b"".join(chunks),
        encoding=response.charset_encoding,
    )
