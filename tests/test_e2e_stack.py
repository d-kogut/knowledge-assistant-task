"""Test działającego stacku (docker compose): stan indeksu i aktualizacja plików na żywo.

Uruchomienie: `make up`, potem `make test-e2e`. Bez `KB_E2E_API_URL` test jest pomijany.
Test tworzy na chwilę plik `knowledge/zz-e2e-*.md` i usuwa go na końcu.
Jakość wyszukiwania na pytaniach testowych sprawdza ewaluacja (`make eval`).
"""

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

pytestmark = pytest.mark.e2e

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"
WORD = "kwazikrokodyl"  # słowo, którego nie ma w dokumentacji


@pytest.fixture(scope="module")
def api():
    url = os.environ.get("KB_E2E_API_URL")
    if not url:
        pytest.skip("ustaw KB_E2E_API_URL, np. http://localhost:8000 (stack: make up)")
    client = httpx2.Client(base_url=url, timeout=30)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    return client


def _ask(api: httpx2.Client, question: str, **params: Any) -> dict[str, Any]:
    resp = api.post("/ask", json={"question": question, "llm": False, **params})
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def _wait_for(
    api: httpx2.Client, predicate: Callable[[dict[str, Any]], bool], timeout_s: float = 60
) -> float:
    """Pyta, aż warunek będzie spełniony; zwraca czas oczekiwania w sekundach."""
    started = time.monotonic()
    while True:
        data = _ask(api, f"{WORD} cena", mode="bm25", top_k=5)
        if predicate(data):
            return time.monotonic() - started
        if time.monotonic() - started > timeout_s:
            raise AssertionError(f"Brak oczekiwanej zmiany po {timeout_s} s: {data}")
        time.sleep(0.25)


def test_index_covers_all_documents(api):
    status = api.get("/status").json()
    files = [p for p in KNOWLEDGE.glob("*.md") if not p.name.startswith(".")]
    assert status["files"] == len(files)
    assert status["chunks"] >= len(files)
    assert status["errors"] == {}


def test_question_without_llm_returns_fragments_with_sources(api):
    data = _ask(api, "Ile kosztuje Hot Desk Student i jakie ma ograniczenia?")
    assert data["status"] == "fragments_only"
    top = data["fragments"][0]
    assert top["file"] == "02-membership-plans.md"
    assert top["line_start"] <= top["line_end"]
    assert "Hot Desk Student" in top["body"]


def test_new_edited_and_deleted_file_is_visible_live(api):
    path = KNOWLEDGE / f"zz-e2e-{uuid.uuid4().hex[:8]}.md"

    def bodies(data: dict[str, Any]) -> list[str]:
        return [f["body"] for f in data.get("fragments", []) if f["file"] == path.name]

    try:
        path.write_text(f"# Test e2e\n\n## Cena\n{WORD} kosztuje 111 PLN.\n", encoding="utf-8")
        added = _wait_for(api, lambda d: any("111 PLN" in b for b in bodies(d)))
        path.write_text(f"# Test e2e\n\n## Cena\n{WORD} kosztuje 222 PLN.\n", encoding="utf-8")
        edited = _wait_for(
            api,
            lambda d: (
                any("222 PLN" in b for b in bodies(d)) and not any("111" in b for b in bodies(d))
            ),
        )
    finally:
        path.unlink(missing_ok=True)
    deleted = _wait_for(api, lambda d: not bodies(d))
    print(f"\nwidoczne po: dodanie {added:.1f} s, edycja {edited:.1f} s, usunięcie {deleted:.1f} s")
