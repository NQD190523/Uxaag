"""Endpoint tests for POST /api/v1/crawl/rooms/{id}/pages with all I/O mocked.

The membership check, page fetching and saving are replaced at the crawl_service level,
so no network, gateway or document-service is needed.
"""

import pytest
from fastapi.testclient import TestClient

import main
from clients.gateway_client import GatewayUnavailable, RoomAccessDenied
from core.url_guard import UrlNotAllowedError
from services import crawl_service
from services.page_fetcher import FetchedPage

AUTH = {"x-user-id": "7", "Authorization": "Bearer test"}
URL = "/api/v1/crawl/rooms/42/pages"


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def io(monkeypatch):
    """Default fakes: member, every fetch works, saves record their arguments."""
    state = {"saved": [], "fetched": [], "member": None}

    async def membership(*, chat_room_id, authorization):
        if state["member"] is not None:
            raise state["member"]

    async def fetch(url):
        state["fetched"].append(url)
        if "internal" in url:
            raise UrlNotAllowedError("Private or internal addresses cannot be imported")
        body = f"<title>Page {len(state['fetched'])}</title><p>text</p>".encode()
        return FetchedPage(url=url, content_type="text/html", body=body, encoding="utf-8")

    async def save(*, chat_room_id, folder, stem, content, authorization):
        state["saved"].append((chat_room_id, folder, stem, authorization))
        return f"{folder}/{stem}.md"

    monkeypatch.setattr(crawl_service, "check_room_membership", membership)
    monkeypatch.setattr(crawl_service, "fetch_page", fetch)
    monkeypatch.setattr(crawl_service, "save_markdown_file", save)
    return state


def test_saves_each_url_into_web_folder(client, io):
    r = client.post(URL, json={"urls": ["https://a.test/", "https://b.test/", "https://a.test/"]}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["saved"] == 2  # duplicate removed
    assert [x["status"] for x in body["results"]] == ["ok", "ok"]
    assert all(s[1] == "web" and s[0] == 42 and s[3] == "Bearer test" for s in io["saved"])


def test_one_bad_url_does_not_fail_the_rest(client, io):
    r = client.post(URL, json={"urls": ["http://internal.test/", "https://ok.test/"]}, headers=AUTH)
    results = r.json()["results"]
    assert results[0]["status"] == "error" and "internal" in results[0]["error"]
    assert results[1]["status"] == "ok"
    assert r.json()["saved"] == 1


def test_non_member_gets_403_and_nothing_is_fetched(client, io):
    io["member"] = RoomAccessDenied()
    r = client.post(URL, json={"urls": ["https://a.test/"]}, headers=AUTH)
    assert r.status_code == 403
    assert io["fetched"] == []


def test_membership_service_down_gives_503(client, io):
    io["member"] = GatewayUnavailable("status 502")
    r = client.post(URL, json={"urls": ["https://a.test/"]}, headers=AUTH)
    assert r.status_code == 503


def test_missing_token_is_401(client, io):
    r = client.post(URL, json={"urls": ["https://a.test/"]}, headers={"x-user-id": "7"})
    assert r.status_code == 401


@pytest.mark.parametrize("urls", [[], [f"https://{i}.test/" for i in range(6)]])
def test_url_count_is_bounded(client, io, urls):
    r = client.post(URL, json={"urls": urls}, headers=AUTH)
    assert r.status_code == 422


def test_overlong_url_is_rejected(client, io):
    r = client.post(URL, json={"urls": ["https://a.test/" + "x" * 2100]}, headers=AUTH)
    assert r.status_code == 422


def test_save_refused_mid_run_is_reported_per_url(client, io, monkeypatch):
    async def save(**kwargs):
        raise RoomAccessDenied()

    monkeypatch.setattr(crawl_service, "save_markdown_file", save)
    r = client.post(URL, json={"urls": ["https://a.test/"]}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["success"] is False
    assert "no longer a member" in r.json()["results"][0]["error"]
