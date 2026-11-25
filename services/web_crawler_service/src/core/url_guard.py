"""SSRF guard (core layer): decide whether the crawler may fetch a URL.

The crawler fetches URLs that users type, from inside our network. Without this guard a
user could make it read internal services (``http://127.0.0.1:8006``, the gateway,
Redis, Docker hostnames) or cloud metadata (``http://169.254.169.254``). The guard is run
on the first URL **and on every redirect target**, because a public page can redirect to
an internal address.

Rules:
    * scheme ``http`` or ``https`` only, a hostname is required, no ``user:pass@``;
    * every address the hostname resolves to must be globally routable. Loopback,
      private, link-local, CGNAT, multicast, reserved and unspecified addresses are
      refused, IPv4-mapped IPv6 (``::ffff:10.0.0.1``) is checked as its IPv4 address,
      and other IPv6 forms that embed IPv4 (NAT64, 6to4, Teredo, IPv4-compatible,
      SIIT) are refused.

Known gap (DNS rebinding): the name is resolved here and again by httpx when it connects,
so a hostile DNS server could answer differently the second time. Pinning the connection
to the checked IP needs a custom transport and is not done yet.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

import httpx

ALLOWED_SCHEMES = frozenset({"http", "https"})

# IPv6 ranges that embed an IPv4 address (besides ::ffff:0:0/96, handled by ipv4_mapped).
_IPV6_EMBEDDING_IPV4 = tuple(
    ipaddress.IPv6Network(net)
    for net in (
        "::/96",            # IPv4-compatible (deprecated), e.g. ::7f00:1
        "::ffff:0:0:0/96",  # SIIT
        "64:ff9b::/96",     # NAT64 well-known prefix
        "64:ff9b:1::/48",   # NAT64 local-use prefix
        "2002::/16",        # 6to4
        "2001::/32",        # Teredo
    )
)


class UrlNotAllowedError(Exception):
    """The URL must not be fetched. The message is safe to show to the user."""


def safe_url_for_log(url: str) -> str:
    """Return ``scheme://host/path`` only.

    Userinfo and the query string can carry credentials or tokens, so they never go to
    the logs.
    """
    try:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"
    except ValueError:
        return "<unparseable url>"


def check_url_syntax(url: str) -> tuple[str, int]:
    """Validate the URL's form without any network access.

    Args:
        url: Absolute URL from the user or from a ``Location`` header.

    Returns:
        ``(hostname, port)`` with the scheme's default port filled in.

    Raises:
        UrlNotAllowedError: Bad scheme, no host, credentials in the URL or bad port.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as e:
        raise UrlNotAllowedError("Invalid URL") from e
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlNotAllowedError("Only http and https URLs can be imported")
    if not parts.hostname:
        raise UrlNotAllowedError("The URL has no host name")
    if parts.username is not None or parts.password is not None:
        raise UrlNotAllowedError("URLs with a user name or password are not allowed")
    if port is None:
        port = 443 if parts.scheme.lower() == "https" else 80
    try:
        # Resolve the host exactly as httpx will connect to it (IDNA 2008 punycode).
        # socket's own IDNA 2003 encoding can name a different host (e.g. "faß.de").
        # raw_host is the encoded name httpx connects to (.host is the decoded form).
        host = httpx.URL(url.strip()).raw_host.decode("ascii")
    except (httpx.InvalidURL, UnicodeError) as e:
        raise UrlNotAllowedError("Invalid URL") from e
    if not host:
        raise UrlNotAllowedError("The URL has no host name")
    return host, port


def is_public_address(address: str) -> bool:
    """Return True if ``address`` (an IP literal) is globally routable.

    ``is_global`` is False for loopback, RFC 1918 private ranges, link-local (cloud
    metadata), CGNAT 100.64/10, unspecified, reserved and documentation ranges.
    Multicast is excluded explicitly because some multicast ranges count as global.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            # ::ffff:127.0.0.1 reaches 127.0.0.1, so judge the embedded IPv4 address.
            ip = ip.ipv4_mapped
        elif any(ip in net for net in _IPV6_EMBEDDING_IPV4):
            # These forms carry an IPv4 address that a NAT64/6to4/Teredo gateway may
            # route to (e.g. 64:ff9b::a9fe:a9fe -> 169.254.169.254). Real sites do not
            # publish them, so refuse them outright.
            return False
    # is_reserved is checked explicitly: some Python versions count parts of reserved
    # ranges as global (Docker runs 3.11, the dev venv 3.13).
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


async def resolve_host(host: str, port: int) -> list[str]:
    """Resolve ``host`` to every IP address it maps to (A and AAAA).

    Raises:
        UrlNotAllowedError: The name does not resolve.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise UrlNotAllowedError("The host name could not be resolved") from e
    return sorted({info[4][0] for info in infos})


async def assert_url_allowed(url: str) -> None:
    """Raise unless ``url`` is a public http(s) URL.

    Every resolved address must be public. Accepting a name with one public and one
    private address would let the connection pick the private one.

    Raises:
        UrlNotAllowedError: Any rule in the module docstring is broken.
    """
    host, port = check_url_syntax(url)
    addresses = await resolve_host(host, port)
    if not addresses:
        raise UrlNotAllowedError("The host name could not be resolved")
    if not all(is_public_address(a) for a in addresses):
        raise UrlNotAllowedError("Private or internal addresses cannot be imported")
