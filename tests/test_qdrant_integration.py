"""Testy z prawdziwym Qdrantem: zapis, BM25, wyszukiwanie hybrydowe, alias, stan indeksu.

Uruchamiają tymczasowy kontener Qdranta (ten sam obraz co w docker-compose.yml) na losowym
porcie i usuwają go na końcu. Bez Dockera są pomijane; w CI brak Dockera to błąd
(`KB_INTEGRATION_REQUIRED=1`). Zamiast PolDense (0,6 GB) działa prosty, deterministyczny
embedder z haszowania lematów: testy sprawdzają Qdranta i naszą warstwę zapisu
i wyszukiwania, a nie jakość modelu (tę mierzy ewaluacja).
"""

import math
import os
import shutil
import subprocess
import time
import uuid
import zlib
from pathlib import Path

import pytest
import yaml
from qdrant_client import QdrantClient
from qdrant_client import models as m

from kb.answer.schemas import AskRequest
from kb.answer.service import AskService
from kb.ingest.chunker import chunk_markdown
from kb.ingest.sync import Indexer
from kb.store import QdrantStore
from kb.text import lemma_text, lemmas

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]

DOCS = {
    "01-lokalizacje.md": (
        "# Lokalizacje\n\n"
        "## Krowodrza\n"
        "Lokalizacja Krowodrza jest pet-friendly: psy są mile widziane w strefie coworkingu.\n\n"
        "## Zabłocie\n"
        "Na Zabłociu obowiązuje zakaz wprowadzania zwierząt.\n"
    ),
    "02-cennik.md": (
        "# Cennik\n\n"
        "## Hot Desk\n"
        "| Plan | Cena |\n|---|---|\n"
        "| Hot Desk Student | 149 PLN |\n"
        "| Hot Desk Basic | 299 PLN |\n\n"
        "## Sale konferencyjne\n"
        "Sala Wawel ma wideokonferencję i kosztuje 120 PLN za godzinę.\n"
    ),
}


class HashingEmbedder:
    """Wektor = znormalizowany licznik lematów (haszowanych do `dim` pozycji)."""

    def __init__(self, model_name: str, dim: int = 64) -> None:
        self.model_name = model_name
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for lemma in lemmas(text):
            v[zlib.crc32(lemma.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else [1.0] + [0.0] * (self.dim - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _qdrant_image() -> str:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return str(compose["services"]["qdrant"]["image"])


def _connect(url: str, timeout_s: float = 60) -> QdrantClient:
    client = QdrantClient(url=url, timeout=5, check_compatibility=False)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            client.get_collections()
            return client
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.3)


def _unavailable(reason: str) -> None:
    if os.environ.get("KB_INTEGRATION_REQUIRED") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def qdrant_url():
    url = os.environ.get("KB_TEST_QDRANT_URL")
    if url:
        if _connect(url).get_collections().collections:
            pytest.fail(f"Qdrant {url} nie jest pusty, a testy usuwają kolekcje — użyj osobnej")
        yield url
        return

    if shutil.which("docker") is None:
        _unavailable("brak Dockera (albo ustaw KB_TEST_QDRANT_URL)")
    name = f"kb-test-qdrant-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::6333", _qdrant_image()],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        _unavailable(f"nie udało się uruchomić Qdranta: {started.stderr.strip()[:300]}")
    try:
        mapping = subprocess.run(
            ["docker", "port", name, "6333/tcp"], capture_output=True, text=True, check=True
        ).stdout
        url = f"http://127.0.0.1:{mapping.splitlines()[0].rsplit(':', 1)[1]}"
        _connect(url)
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def store(qdrant_url) -> QdrantStore:
    """Pusty Qdrant przed każdym testem."""
    s = QdrantStore(qdrant_url, "kb", "kb_meta")
    aliases = s.client.get_aliases().aliases
    if aliases:
        s.client.update_collection_aliases(
            change_aliases_operations=[
                m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=a.alias_name))
                for a in aliases
            ]
        )
    for c in s.client.get_collections().collections:
        s.client.delete_collection(c.name)
    return s


@pytest.fixture
def corpus(settings):
    for name, text in DOCS.items():
        (settings.knowledge_dir / name).write_text(text, encoding="utf-8")
    return settings.knowledge_dir


@pytest.fixture
def embedder(settings) -> HashingEmbedder:
    return HashingEmbedder(settings.embed_model)  # ta sama nazwa co w ustawieniach api


def _bootstrap(settings, store, embedder) -> Indexer:
    indexer = Indexer(settings, store, embedder)
    indexer.bootstrap()
    return indexer


def _bm25(store: QdrantStore, question: str, limit: int = 5):
    return store.search("bm25", None, lemma_text(question), limit, 30)


def test_bootstrap_indexes_files_switches_alias_and_writes_meta(settings, store, corpus, embedder):
    assert store.read_meta() is None and store.alias_target() is None
    indexer = Indexer(settings, store, embedder)
    report = indexer.bootstrap()

    expected = sum(len(chunk_markdown(t, n, settings.max_chunk_words)) for n, t in DOCS.items())
    assert report.added == expected
    assert store.alias_target() == indexer.collection
    assert store.count() == expected
    meta = store.read_meta()
    assert meta is not None
    assert (meta["files"], meta["chunks"], meta["embed_model"]) == (2, expected, "fake/model")


def test_bm25_matches_inflected_polish_forms(settings, store, corpus, embedder):
    _bootstrap(settings, store, embedder)
    # „psem” w pytaniu i „psy” w dokumencie to po lematyzacji to samo słowo „pies”.
    hits = _bm25(store, "Czy mogę przyjść z psem?")
    assert [h.section for h in hits] == ["Krowodrza"]


def test_hybrid_and_dense_return_full_payload(settings, store, corpus, embedder):
    _bootstrap(settings, store, embedder)
    question = "Ile kosztuje Hot Desk Student?"
    vector = embedder.embed_query(question)
    expected = next(
        c
        for c in chunk_markdown(DOCS["02-cennik.md"], "02-cennik.md", 350)
        if c.section == "Hot Desk"
    )
    for mode in ("hybrid", "dense"):
        hits = store.search(mode, vector, lemma_text(question), 3, 30)
        top = hits[0]
        assert (top.file, top.section, top.breadcrumb) == (
            "02-cennik.md",
            "Hot Desk",
            "Cennik › Hot Desk",
        )
        assert (top.line_start, top.line_end) == (expected.line_start, expected.line_end)
        assert "| Hot Desk Student | 149 PLN |" in top.body
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_fuses_dense_and_bm25_lists(settings, store, corpus, embedder):
    _bootstrap(settings, store, embedder)
    # Wektor wskazuje salę konferencyjną, a słowa (BM25) — fragment o psach.
    vector = embedder.embed_query("Sala Wawel wideokonferencja")
    dense = store.search("dense", vector, "", 2, 30)
    assert "Krowodrza" not in {h.section for h in dense}  # sam wektor jej nie znajduje
    hybrid = store.search("hybrid", vector, lemma_text("psem"), 2, 30)
    assert {h.section for h in hybrid} == {"Krowodrza", "Sale konferencyjne"}


def test_hybrid_works_when_question_has_no_lexical_terms(settings, store, corpus, embedder):
    _bootstrap(settings, store, embedder)
    question = "Czy to jest?"
    assert lemma_text(question) == ""  # same słowa funkcyjne
    assert store.search("hybrid", embedder.embed_query(question), "", 5, 30)
    assert _bm25(store, question) == []


def test_edit_and_delete_are_visible_in_search(settings, store, corpus, embedder):
    indexer = _bootstrap(settings, store, embedder)
    before = store.count()

    prices = corpus / "02-cennik.md"
    prices.write_text(DOCS["02-cennik.md"].replace("149 PLN", "159 PLN"), encoding="utf-8")
    report = indexer.sync_paths([prices])
    assert (report.added, report.changed, report.removed) == (0, 1, 0)
    assert "159 PLN" in _bm25(store, "159")[0].body
    assert _bm25(store, "149") == []

    places = corpus / "01-lokalizacje.md"
    removed = len(store.indexed_chunks(indexer.collection, file=places.name))
    places.unlink()
    indexer.sync_paths([places])
    assert store.count() == before - removed
    assert _bm25(store, "Czy mogę przyjść z psem?") == []
    assert store.read_meta()["files"] == 1


def test_model_change_builds_new_collection_and_keeps_previous(settings, store, corpus):
    first = _bootstrap(settings, store, HashingEmbedder("model-a", dim=64))
    second = _bootstrap(settings, store, HashingEmbedder("model-b", dim=32))
    assert second.collection != first.collection
    assert store.alias_target() == second.collection
    assert store.collection_exists(first.collection)  # poprzednia zostaje do szybkiego powrotu
    # zapytanie przez alias używa już nowej kolekcji (inny wymiar wektora)
    vector = HashingEmbedder("model-b", dim=32).embed_query("Hot Desk Student")
    assert store.search("dense", vector, "", 1, 30)[0].section == "Hot Desk"

    third = _bootstrap(settings, store, HashingEmbedder("model-c", dim=16))
    assert store.alias_target() == third.collection
    assert not store.collection_exists(first.collection)  # starsze niż poprzednia są usuwane
    assert store.collection_exists(second.collection)
    assert store.read_meta()["embed_model"] == "model-c"


def test_ask_service_with_real_qdrant(settings, store, corpus, embedder):
    _bootstrap(settings, store, embedder)
    service = AskService(settings, store, embedder, llms={})

    resp = service.ask(AskRequest(question="Czy mogę przyjść z psem na Krowodrzę?", debug=True))
    assert resp.status == "fragments_only"
    assert resp.fragments[0].file == "01-lokalizacje.md"
    assert resp.fragments[0].breadcrumb == "Lokalizacje › Krowodrza"
    assert resp.debug is not None
    assert {"hybrid", "dense", "bm25"} <= set(resp.debug)
    assert service.health() == (True, "ok")

    hits = store.all_hits()
    assert len(hits) == store.count()
    assert [(h.file, h.line_start) for h in hits] == sorted((h.file, h.line_start) for h in hits)
