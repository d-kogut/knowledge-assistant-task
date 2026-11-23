"""Atrapy zależności zewnętrznych: Qdrant, model embeddingów, Claude."""

from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from kb.config import Settings
from kb.ingest.chunker import Chunk
from kb.store import Hit, IndexedChunk, point_id


class FakeEmbedder:
    model_name = "fake/model"
    dim = 3

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += texts
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeStore:
    """Pamięciowy odpowiednik QdrantStore (te same metody, bez sieci)."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.collections: dict[str, dict[str, Chunk]] = {}
        self.alias: str | None = None
        self.meta: dict[str, Any] | None = None
        self.hits = hits or []
        self.searches: list[tuple[str, str]] = []
        self.ops: list[str] = []

    # ingest
    def alias_target(self) -> str | None:
        return self.alias

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, dim: int) -> None:
        self.collections[name] = {}

    def switch_alias(self, name: str) -> None:
        self.alias = name

    def drop_collections_except(self, keep: Iterable[str]) -> None:
        keep_set = set(keep)
        self.collections = {k: v for k, v in self.collections.items() if k in keep_set}

    def indexed_chunks(self, collection: str, file: str | None = None) -> list[IndexedChunk]:
        return [
            IndexedChunk(pid, c.chunk_id, c.file, c.content_hash)
            for pid, c in self.collections[collection].items()
            if file is None or c.file == file
        ]

    def upsert_chunks(
        self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        self.ops.append("upsert")
        for c in chunks:
            self.collections[collection][point_id(c.chunk_id)] = c

    def delete_points(self, collection: str, ids: Sequence[str]) -> None:
        if ids:
            self.ops.append("delete")
        for pid in ids:
            self.collections[collection].pop(pid, None)

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta = meta

    def read_meta(self) -> dict[str, Any] | None:
        return self.meta

    # api
    def search(
        self, mode: str, vector: list[float] | None, lemmas: str, limit: int, prefetch: int
    ) -> list[Hit]:
        self.searches.append((mode, lemmas))
        return self.hits[:limit]

    def all_hits(self) -> list[Hit]:
        return list(self.hits)

    def count(self) -> int:
        return len(self.hits)


def make_hit(
    i: int, body: str = "Treść", file: str = "02-plans.md", section: str | None = None
) -> Hit:
    section = section or f"Sekcja {i}"
    return Hit(
        chunk_id=f"{file}#s{i}",
        file=file,
        headings=["Dokument", section],
        breadcrumb=f"Dokument › {section}",
        section=section,
        body=body,
        line_start=10 * i,
        line_end=10 * i + 5,
        score=1.0 / (i + 1),
    )


def citation(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="search_result_location",
        search_result_index=index,
        cited_text=text,
        source="x",
        title="t",
        start_block_index=0,
        end_block_index=1,
    )


def response(*blocks: SimpleNamespace, model: str = "claude-test") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        model=model,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )


def text_block(text: str, *cits: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=list(cits) or None)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KB_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KB_OPENAI_API_KEY", raising=False)
    return Settings(
        knowledge_dir=tmp_path,
        embed_model="fake/model",
        embed_revision="rev",
        _env_file=None,
    )
