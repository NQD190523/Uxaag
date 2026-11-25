"""Tests for HTML -> Markdown conversion and file naming (services/html_to_markdown.py)."""

from datetime import datetime, timezone

from services.html_to_markdown import build_document, html_to_markdown, slugify
from services.page_fetcher import FetchedPage

PAGE = b"""<html><head><title>  Hello   World </title><style>.x{}</style>
<script>alert(1)</script></head><body>
<nav><a href="/home">Home</a></nav>
<main><h1>Heading</h1><p>See <a href="/docs/a">docs</a> and
<a href="javascript:alert(1)">bad</a>.</p><img src="/i.png" alt="pic"></main>
<footer>footer text</footer></body></html>"""


def test_main_content_is_converted_and_chrome_removed():
    title, md = html_to_markdown(PAGE, "https://site.test/x/y")
    assert title == "Hello World"
    assert "# Heading" in md
    assert "alert" not in md and ".x{}" not in md
    assert "Home" not in md and "footer text" not in md
    assert "![" not in md  # images stripped


def test_links_are_absolute_and_script_links_dropped():
    _, md = html_to_markdown(PAGE, "https://site.test/x/y")
    assert "(https://site.test/docs/a)" in md
    assert "javascript:" not in md
    assert "bad" in md  # the text stays


def test_slugify():
    assert slugify("Tiếng Việt: Đà Nẵng!") == "tieng-viet-da-nang"
    assert slugify("   ") == "page"
    assert slugify("!!!", fallback="example-com") == "example-com"
    assert len(slugify("a" * 200)) == 60


def test_build_document_header_and_fallback_title():
    page = FetchedPage(url="https://example.com/notes.txt", content_type="text/plain", body=b"line one\n", encoding=None)
    title, stem, text = build_document(page, fetched_at=datetime(2026, 11, 26, 9, 30, tzinfo=timezone.utc))
    assert title == "example.com"
    assert stem == "example-com"
    assert text.startswith("# example.com\n\n> Source: <https://example.com/notes.txt>")
    assert "Fetched: 2026-11-26 09:30 UTC" in text
    assert text.rstrip().endswith("line one")


def test_source_url_cannot_break_the_autolink():
    page = FetchedPage(url="https://e.test/a b>c", content_type="text/plain", body=b"x", encoding=None)
    _, _, text = build_document(page)
    assert "> Source: <https://e.test/a%20b%3Ec>" in text


def test_meta_charset_is_honoured():
    html = '<html><head><meta charset="windows-1258"><title>x</title></head><body><p>caf\xe9</p></body></html>'
    _, md = html_to_markdown(html.encode("cp1258"), "https://site.test/")
    assert "café" in md
