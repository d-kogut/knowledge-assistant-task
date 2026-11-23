"""Warstwa Qdranta: kolekcje, alias, zapis fragmentów, wyszukiwanie, stan synchronizacji.

Schemat punktu:
- wektor `dense` (PolDense, cosinus),
- payload: chunk_id, file, headings, breadcrumb, section, body, line_start, line_end,
  content_hash, lemmas (tekst zlematyzowany z indeksem pełnotekstowym BM25).

Aplikacja zawsze używa aliasu (`kb`). Nazwa kolekcji zawiera hash modelu, jego rewizji
i wersji chunkera, więc zmiana któregokolwiek oznacza nową kolekcję i atomowe
przełączenie aliasu bez przerwy w działaniu.
"""

import hashlib
import logging
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client import models as m

from kb.ingest.chunker import CHUNKER_VERSION, Chunk
from kb.text import lemma_text

logger = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "dense", "bm25"]
_ID_NAMESPACE = uuid.UUID("6f6b2c1e-6a43-4f4e-9d0e-6b6e6f776c65")
_META_ID = 1
_PAYLOAD_FIELDS = [
    "chunk_id",
    "file",
    "headings",
    "breadcrumb",
    "section",
    "body",
    "line_start",
    "line_end",
    "content_hash",
]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def collection_name_for(model: str, revision: str, dim: int) -> str:
    key = f"{model}@{revision}|dim={dim}|chunker={CHUNKER_VERSION}|bm25=lemmas-v1"
    return "kb_" + hashlib.sha256(key.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    file: str
    headings: list[str]
    breadcrumb: str
    section: str
    body: str
    line_start: int
    line_end: int
    score: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> Hit:
        return cls(
            chunk_id=payload["chunk_id"],
            file=payload["file"],
            headings=list(payload["headings"]),
            breadcrumb=payload["breadcrumb"],
            section=payload["section"],
            body=payload["body"],
            line_start=int(payload["line_start"]),
            line_end=int(payload["line_end"]),
            score=score,
        )

    def citation_blocks(self) -> list[str]:
        return Chunk(
            self.chunk_id,
            self.file,
            tuple(self.headings),
            self.body,
            self.line_start,
            self.line_end,
        ).citation_blocks()


@dataclass(frozen=True)
class IndexedChunk:
    point_id: str
    chunk_id: str
    file: str
    content_hash: str


class QdrantStore:
    def __init__(self, url: str, alias: str, meta_collection: str, timeout: int = 30):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.alias = alias
        self.meta_collection = meta_collection

    # --- połączenie i schemat -------------------------------------------------

    def wait_until_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        delay = 0.5
        while True:
            try:
                self.client.get_collections()
                return
            except Exception as exc:  # połączenie odrzucone, DNS itd.
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Qdrant niedostępny: {exc}") from exc
                logger.info("Czekam na Qdranta (%s)…", type(exc).__name__)
                time.sleep(delay)
                delay = min(delay * 2, 5.0)

    def alias_target(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def collection_exists(self, name: str) -> bool:
        return bool(self.client.collection_exists(name))

    def create_collection(self, name: str, dim: int) -> None:
        self.client.create_collection(
            name,
            vectors_config={"dense": m.VectorParams(size=dim, distance=m.Distance.COSINE)},
        )
        # Indeksy pól zakładamy przed danymi, żeby HNSW uwzględniał filtry.
        self.client.create_payload_index(
            name, "file", field_schema=m.PayloadSchemaType.KEYWORD, wait=True
        )
        self.client.create_payload_index(
            name,
            "lemmas",
            field_schema=m.TextIndexParams(
                type=m.TextIndexType.TEXT,
                tokenizer=m.TokenizerType.WHITESPACE,  # tekst jest już zlematyzowany
                lowercase=True,
                scoring=True,  # BM25
            ),
            wait=True,
        )
        logger.info("Utworzono kolekcję %s (wymiar %d)", name, dim)

    def switch_alias(self, name: str) -> None:
        ops: list[m.CreateAliasOperation | m.DeleteAliasOperation] = []
        if self.alias_target() is not None:
            ops.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=self.alias)))
        ops.append(
            m.CreateAliasOperation(
                create_alias=m.CreateAlias(collection_name=name, alias_name=self.alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)  # atomowo
        logger.info("Alias %s → %s", self.alias, name)

    def drop_collections_except(self, keep: Iterable[str]) -> None:
        keep_set = set(keep)
        for c in self.client.get_collections().collections:
            if (
                c.name.startswith("kb_")
                and c.name != self.meta_collection
                and c.name not in keep_set
            ):
                self.client.delete_collection(c.name)
                logger.info("Usunięto starą kolekcję %s", c.name)

    # --- zapis ----------------------------------------------------------------

    def indexed_chunks(self, collection: str, file: str | None = None) -> list[IndexedChunk]:
        flt = (
            m.Filter(must=[m.FieldCondition(key="file", match=m.MatchValue(value=file))])
            if file
            else None
        )
        out: list[IndexedChunk] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection,
                scroll_filter=flt,
                limit=1000,
                offset=offset,
                with_payload=["chunk_id", "file", "content_hash"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                out.append(
                    IndexedChunk(
                        str(p.id), payload["chunk_id"], payload["file"], payload["content_hash"]
                    )
                )
            if offset is None:
                return out

    def upsert_chunks(
        self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        points = [
            m.PointStruct(
                id=point_id(c.chunk_id),
                vector={"dense": v},
                payload={
                    "chunk_id": c.chunk_id,
                    "file": c.file,
                    "headings": list(c.headings),
                    "breadcrumb": c.breadcrumb,
                    "section": c.section,
                    "body": c.body,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "content_hash": c.content_hash,
                    "lemmas": lemma_text(c.embed_text),
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(collection, points[i : i + 256], wait=True)

    def delete_points(self, collection: str, ids: Sequence[str]) -> None:
        if ids:
            self.client.delete(
                collection, points_selector=m.PointIdsList(points=list(ids)), wait=True
            )

    # --- odczyt ---------------------------------------------------------------

    def search(
        self, mode: SearchMode, vector: list[float] | None, lemmas: str, limit: int, prefetch: int
    ) -> list[Hit]:
        text_query = m.TextQuery(text=lemmas)
        if mode == "dense":
            res = self.client.query_points(
                self.alias, query=vector, using="dense", limit=limit, with_payload=_PAYLOAD_FIELDS
            )
        elif mode == "bm25":
            if not lemmas.strip():
                return []
            res = self.client.query_points(
                self.alias,
                query=text_query,
                using="lemmas",
                limit=limit,
                with_payload=_PAYLOAD_FIELDS,
            )
        else:
            prefetches = [m.Prefetch(query=vector, using="dense", limit=prefetch)]
            if lemmas.strip():
                prefetches.append(m.Prefetch(query=text_query, using="lemmas", limit=prefetch))
            res = self.client.query_points(
                self.alias,
                prefetch=prefetches,
                query=m.RrfQuery(rrf=m.Rrf(k=60)),
                limit=limit,
                with_payload=_PAYLOAD_FIELDS,
            )
        return [Hit.from_payload(p.payload or {}, float(p.score)) for p in res.points]

    def all_hits(self) -> list[Hit]:
        """Wszystkie fragmenty w kolejności plików i linii (tryb „cała dokumentacja”)."""
        hits: list[Hit] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.alias, limit=1000, offset=offset, with_payload=_PAYLOAD_FIELDS
            )
            hits += [Hit.from_payload(p.payload or {}, 0.0) for p in points]
            if offset is None:
                return sorted(hits, key=lambda h: (h.file, h.line_start))

    def count(self) -> int:
        return int(self.client.count(self.alias, exact=True).count)

    # --- stan synchronizacji --------------------------------------------------

    def write_meta(self, meta: dict[str, Any]) -> None:
        if not self.collection_exists(self.meta_collection):
            self.client.create_collection(self.meta_collection, vectors_config={})
        self.client.upsert(
            self.meta_collection, [m.PointStruct(id=_META_ID, vector={}, payload=meta)], wait=True
        )

    def read_meta(self) -> dict[str, Any] | None:
        if not self.collection_exists(self.meta_collection):
            return None
        points = self.client.retrieve(self.meta_collection, [_META_ID], with_payload=True)
        return dict(points[0].payload or {}) if points else None
