from conftest import FakeEmbedder, FakeStore, make_hit
from fastapi.testclient import TestClient

from kb.answer.service import AskService
from kb.api.app import create_app


def _client(settings, meta=None):
    store = FakeStore([make_hit(i) for i in range(3)])
    store.alias = "kb_x"
    store.meta = meta
    service = AskService(settings, store, FakeEmbedder(), None)
    return TestClient(create_app(lambda: service))


def test_ask_validates_input(settings):
    with _client(settings) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "ok?", "mode": "magic"}).status_code == 422
        assert (
            client.post("/ask", json={"question": "ok?", "model": "gpt; rm -rf"}).status_code == 422
        )


def test_ask_returns_fragments_without_llm(settings):
    with _client(settings) as client:
        data = client.post("/ask", json={"question": "Ile kosztuje Hot Desk?"}).json()
    assert data["status"] == "fragments_only"
    assert len(data["fragments"]) == 3


def test_page_is_served_with_strict_csp(settings):
    with _client(settings) as client:
        resp = client.get("/")
        assert resp.status_code == 200 and "SpaceHub" in resp.text
        assert "default-src 'self'" in resp.headers["content-security-policy"]
        assert client.get("/static/app.js").status_code == 200
        assert "content-security-policy" not in client.get("/docs").headers


def test_health_requires_matching_embedding_model(settings):
    with _client(settings, meta={"embed_model": "fake/model", "embed_revision": "rev"}) as client:
        assert client.get("/health").status_code == 200
    with _client(settings, meta={"embed_model": "other", "embed_revision": "rev"}) as client:
        resp = client.get("/health")
        assert resp.status_code == 503 and "różni się" in resp.json()["detail"]
