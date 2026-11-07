"""Web API tests: TestClient + dependency_overrides, a tiny in-memory index, no index file."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import create_app, get_index, highlight
from findex.web.settings import Settings


@pytest.fixture
def client(ix: Index) -> Iterator[TestClient]:
    app = create_app(Settings(index_path=Path("/nonexistent")))
    app.dependency_overrides[get_index] = lambda: ix
    yield TestClient(app)  # no `with`: lifespan (file loading) does not run


def test_search_ok_and_paginated(client: TestClient) -> None:
    r = client.get("/search", params={"q": "python", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert (body["total"], body["pages"], len(body["results"])) == (3, 2, 2)
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    page2 = client.get("/search", params={"q": "python", "k": 2, "page": 2}).json()
    ids = {h["doc_id"] for h in body["results"]} | {h["doc_id"] for h in page2["results"]}
    assert len(ids) == 3 and len(page2["results"]) == 1
    assert "X-Request-ID" in r.headers


@pytest.mark.parametrize(
    "params",
    [
        {"q": "x", "k": "abc"},
        {"q": "x", "k": 0},
        {"q": "x", "k": 51},
        {"q": ""},
        {},
        {"q": "x", "scorer": "foo"},
        {"q": "x", "page": 0},
    ],
)
def test_validation_422(client: TestClient, params: dict[str, str | int]) -> None:
    r = client.get("/search", params=params)
    assert r.status_code == 422 and "detail" in r.json()


def test_bad_query_400(client: TestClient) -> None:
    r = client.get("/search", params={"q": "(python"})
    assert r.status_code == 400 and "missing ')'" in r.json()["detail"]


def test_document_and_404(client: TestClient) -> None:
    r = client.get("/docs/2")
    assert r.status_code == 200 and r.json()["title"] == "C"
    r = client.get("/docs/999")
    assert r.status_code == 404 and r.json() == {"detail": "document 999 not found"}
    assert client.get("/docs/abc").status_code == 422


def test_stats(client: TestClient) -> None:
    assert client.get("/stats").json()["documents"] == 4


def test_health_503_when_index_not_loaded() -> None:
    app = create_app(Settings(index_path=Path("/nonexistent")))
    r = TestClient(app).get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"


def test_lifespan_loads_index_once_and_health_ok(index_file: Path) -> None:
    app = create_app(Settings(index_path=index_file))
    with TestClient(app) as c:  # `with` runs the lifespan
        assert c.get("/health").json() == {"status": "ok", "documents": 4}
        loaded = app.state.index
        c.get("/search", params={"q": "python"})
        assert app.state.index is loaded
    assert app.state.index is None  # shutdown ran


def test_missing_index_fails_at_startup() -> None:
    app = create_app(Settings(index_path=Path("/no/such/index.bin")))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_unexpected_error_is_clean_500_with_request_id(
    ix: Index, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(Settings(index_path=Path("/nonexistent")))

    def broken() -> Index:
        raise RuntimeError("boom")

    app.dependency_overrides[get_index] = broken
    r = TestClient(app, raise_server_exceptions=False).get(
        "/search", params={"q": "x"}, headers={"X-Request-ID": "req-42"}
    )
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error", "request_id": "req-42"}
    assert "Traceback" not in r.text and "boom" not in r.text
    assert "req-42" in caplog.text and "boom" in caplog.text  # full error in the log only


def test_html_ui(client: TestClient) -> None:
    r = client.get("/", params={"q": "python", "k": 2})
    assert r.status_code == 200 and "next →" in r.text and "3 documents" in r.text
    assert "documents" in client.get("/").text
    assert "bad query" in client.get("/", params={"q": "(x"}).text
    doc = client.get("/doc/0", params={"q": "python"})
    assert doc.status_code == 200 and "<h1>A</h1>" in doc.text
    assert client.get("/static/style.css").status_code == 200


def test_highlight_escapes_then_marks() -> None:
    assert str(highlight("<b>x</b> **loop** y")) == "&lt;b&gt;x&lt;/b&gt; <mark>loop</mark> y"


def test_openapi_docs_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(schema["paths"])
    assert client.get("/docs").status_code == 200


def test_semantic_mode_without_embeddings_is_400(client: TestClient) -> None:
    r = client.get("/search", params={"q": "python", "mode": "semantic"})
    assert r.status_code == 400 and "findex embed" in r.json()["detail"]
    assert client.get("/search", params={"q": "x", "mode": "fuzzy"}).status_code == 422
