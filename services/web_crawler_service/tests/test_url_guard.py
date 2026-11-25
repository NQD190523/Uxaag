"""Tests for the SSRF guard (core/url_guard.py) and guarded fetching (page_fetcher.py)."""

import asyncio

import httpx
import pytest

from core import config, url_guard
from core.url_guard import UrlNotAllowedError, assert_url_allowed, is_public_address
from services import page_fetcher
from services.page_fetcher import FetchError, fetch_page


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254",
        "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fe80::1", "fc00::1",
        "::ffff:127.0.0.1", "::ffff:10.0.0.1", "not-an-ip",
        # IPv6 forms embedding IPv4 (reviewer finding): IPv4-compatible, SIIT, NAT64,
        # 6to4, Teredo. Pinned so Python 3.11 (Docker) and 3.13 (venv) agree.
        "::7f00:1", "::ffff:0:7f00:1", "64:ff9b::a00:1", "64:ff9b::a9fe:a9fe",
        "64:ff9b:1::1", "2002:7f00:1::1", "2001:0:4136:e378::1",
    ],
)
def test_internal_addresses_are_refused(address):
    """Loopback, private, metadata, CGNAT, multicast and mapped-IPv4 forms are not public."""
    assert not is_public_address(address)


@pytest.mark.parametrize("address", ["93.184.215.14", "1.1.1.1", "2606:4700::1111"])
def test_public_addresses_are_allowed(address):
    assert is_public_address(address)


@pytest.mark.parametrize(
    "url, message",
    [
        ("ftp://example.com/x", "Only http and https"),
        ("file:///etc/passwd", "Only http and https"),
        ("http:///nohost", "no host"),
        ("http://user:pw@example.com/", "user name or password"),
        ("http://example.com:99999/", "Invalid URL"),
    ],
)
def test_bad_url_forms(url, message):
    with pytest.raises(UrlNotAllowedError, match=message):
        asyncio.run(assert_url_allowed(url))


def _fake_dns(table):
    """Resolver stand-in: host -> list of addresses (missing host = does not resolve)."""

    async def resolve(host, port):
        if host not in table:
            raise UrlNotAllowedError("The host name could not be resolved")
        return table[host]

    return resolve


def test_name_with_any_private_address_is_refused(monkeypatch):
    """One private A record among public ones is enough to refuse."""
    monkeypatch.setattr(url_guard, "resolve_host", _fake_dns({"mixed.test": ["1.1.1.1", "10.0.0.5"]}))
    with pytest.raises(UrlNotAllowedError, match="Private or internal"):
        asyncio.run(assert_url_allowed("https://mixed.test/"))


def test_host_is_resolved_as_httpx_encodes_it(monkeypatch):
    """IDNA: the name checked is the punycode httpx connects to, not socket's IDNA 2003 form."""
    asked = []

    async def resolve(host, port):
        asked.append(host)
        return ["1.1.1.1"]

    monkeypatch.setattr(url_guard, "resolve_host", resolve)
    asyncio.run(assert_url_allowed("https://faß.de/"))
    assert asked == ["xn--fa-hia.de"]  # IDNA 2008; IDNA 2003 would give "fass.de"


def test_literal_loopback_is_refused():
    """Real resolver path: an IP literal resolves to itself."""
    with pytest.raises(UrlNotAllowedError, match="Private or internal"):
        asyncio.run(assert_url_allowed("http://127.0.0.1:8006/api/v1/auth/me"))


# --- fetch_page with a mock transport --------------------------------------------------


@pytest.fixture
def dns(monkeypatch):
    """public.test is public; internal.test resolves to a private address."""
    monkeypatch.setattr(
        url_guard, "resolve_host",
        _fake_dns({"public.test": ["1.1.1.1"], "internal.test": ["10.0.0.7"]}),
    )


def _transport(handler):
    return httpx.MockTransport(handler)


def test_redirect_to_internal_host_is_blocked(dns):
    """A public page redirecting to an internal address must not be followed."""
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://internal.test/secret"})

    with pytest.raises(UrlNotAllowedError):
        asyncio.run(fetch_page("https://public.test/start", transport=_transport(handler)))
    assert seen == ["public.test"]  # the internal host was never contacted


def test_relative_redirect_is_followed(dns):
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>hi</p>")

    page = asyncio.run(fetch_page("https://public.test/old", transport=_transport(handler)))
    assert page.url == "https://public.test/new"


def test_redirect_loop_stops(dns, monkeypatch):
    monkeypatch.setattr(config, "CRAWL_MAX_REDIRECTS", 2)

    def handler(request):
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchError, match="Too many redirects"):
        asyncio.run(fetch_page("https://public.test/", transport=_transport(handler)))


def test_body_over_limit_is_refused(dns, monkeypatch):
    """The size cap holds even when Content-Length is absent."""
    monkeypatch.setattr(config, "CRAWL_MAX_BYTES", 100)

    async def chunks():
        # Streamed in two parts with no Content-Length header.
        yield b"x" * 80
        yield b"y" * 80

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=chunks())

    with pytest.raises(FetchError, match="too large"):
        asyncio.run(fetch_page("https://public.test/", transport=_transport(handler)))


def test_slow_drip_hits_total_deadline(dns, monkeypatch):
    """A server sending a byte now and then is cut off by the total deadline (reviewer finding)."""
    monkeypatch.setattr(config, "CRAWL_TIMEOUT_SECONDS", 0.3)

    async def drip():
        for _ in range(10):
            yield b"x"
            await asyncio.sleep(0.1)  # each read is fast; only the total is too long

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=drip())

    with pytest.raises(FetchError, match="too long"):
        asyncio.run(fetch_page("https://public.test/", transport=_transport(handler)))


@pytest.mark.parametrize(
    "status, ctype, message",
    [(404, "text/html", "HTTP 404"), (200, "application/pdf", "Unsupported content type")],
)
def test_bad_status_or_type(dns, status, ctype, message):
    def handler(request):
        return httpx.Response(status, headers={"content-type": ctype}, content=b"x")

    with pytest.raises(FetchError, match=message):
        asyncio.run(fetch_page("https://public.test/", transport=_transport(handler)))


def test_network_error_becomes_fetch_error(dns):
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(FetchError, match="could not be reached"):
        asyncio.run(fetch_page("https://public.test/", transport=_transport(handler)))


def test_proxy_env_is_ignored(dns, monkeypatch):
    """trust_env=False: a proxy setting must not route the fetch around the guard."""
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:3128")

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"ok")

    page = asyncio.run(page_fetcher.fetch_page("https://public.test/", transport=_transport(handler)))
    assert page.body == b"ok"
