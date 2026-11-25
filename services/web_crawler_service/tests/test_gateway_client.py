"""Tests for the gateway client (clients/gateway_client.py) using httpx.MockTransport."""

import asyncio

import httpx
import pytest

from clients.gateway_client import (
    DocumentSaveError,
    GatewayAuthError,
    RoomAccessDenied,
    check_room_membership,
    save_markdown_file,
)


def _save(handler, stem="page"):
    return asyncio.run(save_markdown_file(
        chat_room_id=5, folder="web", stem=stem, content="# x",
        authorization="Bearer t", transport=httpx.MockTransport(handler),
    ))


def test_taken_name_gets_a_suffix():
    """409 on page.md and page-2.md, then page-3.md is saved."""
    tried = []

    def handler(request):
        import json
        name = json.loads(request.content)["filename"]
        tried.append(name)
        assert request.headers["authorization"] == "Bearer t"
        if name in ("page.md", "page-2.md"):
            return httpx.Response(409, json={"detail": "exists"})
        return httpx.Response(200, json={"relative_path": f"web/{name}"})

    assert _save(handler) == "web/page-3.md"
    assert tried == ["page.md", "page-2.md", "page-3.md"]


def test_all_names_taken():
    with pytest.raises(DocumentSaveError, match="No free file name"):
        _save(lambda r: httpx.Response(409, json={}))


@pytest.mark.parametrize("status, exc", [(401, GatewayAuthError), (403, RoomAccessDenied), (400, DocumentSaveError)])
def test_save_errors(status, exc):
    with pytest.raises(exc):
        _save(lambda r: httpx.Response(status, json={}))


def test_membership_false_is_denied():
    def handler(request):
        assert request.url.path == "/chat/rooms/5/membership"
        return httpx.Response(200, json={"is_member": False})

    with pytest.raises(RoomAccessDenied):
        asyncio.run(check_room_membership(
            chat_room_id=5, authorization="Bearer t", transport=httpx.MockTransport(handler)
        ))
