"""HTML -> Markdown conversion (services layer).

Turns a fetched page into the Markdown document saved in the room. The page is cleaned
first (scripts, styles, navigation, forms, images removed), then the main content
(``<article>``, ``<main>`` or ``<body>``) is converted with markdownify. A short header
records the title, the source URL and when it was fetched.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from markdownify import markdownify

from services.page_fetcher import HTML_TYPES, FetchedPage

# Removed before conversion: not readable content, or page chrome repeated on every page.
_DROP_TAGS = (
    "script", "style", "noscript", "template", "iframe", "svg", "canvas",
    "form", "button", "input", "select", "textarea",
    "nav", "header", "footer", "aside",
)
_MAX_SLUG_LENGTH = 60


def page_title(soup: BeautifulSoup) -> str | None:
    """Return the ``<title>``, else ``og:title``, else the first ``<h1>`` (one line)."""
    candidates = []
    if soup.title and soup.title.string:
        candidates.append(soup.title.string)
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        candidates.append(og["content"])
    h1 = soup.find("h1")
    if h1:
        candidates.append(h1.get_text(" "))
    for text in candidates:
        cleaned = " ".join(str(text).split())
        if cleaned:
            return cleaned[:200]
    return None


def html_to_markdown(html: bytes | str, base_url: str, encoding: str | None = None) -> tuple[str | None, str]:
    """Convert a page to Markdown.

    Args:
        html: Page source. Bytes let BeautifulSoup honour ``<meta charset>``.
        base_url: Final page URL, used to make relative links absolute (a relative
            link would be meaningless inside the room's document workspace).
        encoding: Charset from the HTTP header, tried first when ``html`` is bytes.

    Returns:
        ``(title or None, markdown body)``.
    """
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None)
    title = page_title(soup)
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if href.lower().startswith(("javascript:", "data:", "vbscript:")):
            # Keep the text, drop a link that would run code when clicked in the viewer.
            link.unwrap()
        else:
            link["href"] = urljoin(base_url, href)
    root = soup.find("article") or soup.find("main") or soup.body or soup
    body = markdownify(str(root), heading_style="ATX", strip=["img"])
    return title, _tidy(body)


def text_to_markdown(text: bytes, encoding: str | None) -> str:
    """Decode a ``text/plain`` page; it is kept as-is (plain text is valid Markdown)."""
    return _tidy(text.decode(encoding or "utf-8", errors="replace"))


def _tidy(markdown: str) -> str:
    """Strip trailing spaces and collapse runs of 3+ blank lines left by removed tags."""
    lines = [line.rstrip() for line in markdown.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def slugify(title: str | None, fallback: str = "page") -> str:
    """Make a file-name stem: lower-case ASCII letters, digits and dashes, max 60 chars.

    Accents are removed (``Tiếng Việt`` -> ``tieng-viet``); ``đ`` has no decomposed form,
    so it is mapped by hand.
    """
    text = (title or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:_MAX_SLUG_LENGTH].rstrip("-")
    return slug or fallback


def build_document(page: FetchedPage, fetched_at: datetime | None = None) -> tuple[str, str, str]:
    """Build the saved document for a fetched page.

    Args:
        page: Result of ``fetch_page``.
        fetched_at: Timestamp to record (defaults to now, UTC).

    Returns:
        ``(title, file name stem, full Markdown text)``. The title falls back to the host.
    """
    host = urlsplit(page.url).hostname or "page"
    if page.content_type in HTML_TYPES:
        title, body = html_to_markdown(page.body, page.url, page.encoding)
    else:
        title, body = None, text_to_markdown(page.body, page.encoding)
    title = title or host
    when = (fetched_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    # Angle brackets make the URL an autolink and stop Markdown from parsing it; a
    # space, "<" or ">" inside the URL would end the autolink, so they are percent-encoded.
    source = page.url.replace(" ", "%20").replace("<", "%3C").replace(">", "%3E")
    header = f"# {title}\n\n> Source: <{source}>  \n> Fetched: {when}\n\n---\n\n"
    text = header + (body or "_The page has no readable text._") + "\n"
    return title, slugify(title, fallback=slugify(host)), text
