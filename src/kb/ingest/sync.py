"""Synchronizacja plików z Qdrantem: tylko zmienione fragmenty są przeliczane.

Kolejność zapisu: najpierw nowe i zmienione fragmenty, potem usunięcie nieaktualnych —
pytanie zadane w trakcie aktualizacji nigdy nie trafia na brakującą treść.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from kb.config import Settings
from kb.embedder import Embedder
from kb.ingest.chunker import CHUNKER_VERSION, Chunk, chunk_markdown
from kb.store import IndexedChunk, collection_name_for

logger = logging.getLogger(__name__)


class Store(Protocol):
    def alias_target(self) -> str | None: ...
    def collection_exists(self, name: str) -> bool: ...
    def create_collection(self, name: str, dim: int) -> None: ...
    def switch_alias(self, name: str) -> None: ...
    def drop_collections_except(self, keep: Iterable[str]) -> None: ...
    def indexed_chunks(self, collection: str, file: str | None = None) -> list[IndexedChunk]: ...
    def upsert_chunks(
        self, collection: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None: ...
    def delete_points(self, collection: str, ids: Sequence[str]) -> None: ...
    def write_meta(self, meta: dict[str, Any]) -> None: ...


def is_document(path: Path) -> bool:
    return path.suffix == ".md" and not path.name.startswith(".")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


EMBED_BATCH = 256  # fragmentów na jedno wywołanie modelu (pomiar: 1,55x szybciej niż plik po pliku)


@dataclass
class _Plan:
    upsert: list[Chunk] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)


@dataclass
class SyncReport:
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    files_changed: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "changed": self.changed,
            "removed": self.removed,
            "files": sorted(set(self.files_changed)),
            "seconds": round(self.seconds, 2),
            "at": _now(),
        }


class Indexer:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.collection = collection_name_for(
            embedder.model_name, settings.embed_revision, embedder.dim
        )
        self.errors: dict[str, str] = {}
        self.last_change: dict[str, Any] | None = None

    # --- start ------------------------------------------------------------------

    def bootstrap(self) -> SyncReport:
        """Pierwsza synchronizacja; przy zmianie modelu/chunkera buduje nową kolekcję."""
        previous = self.store.alias_target()
        if previous == self.collection:
            return self.reconcile_all()

        logger.info("Buduję kolekcję %s (alias wskazywał: %s)", self.collection, previous or "brak")
        if self.store.collection_exists(self.collection):  # przerwana wcześniejsza budowa
            self.store.drop_collections_except([c for c in (previous,) if c])
        self.store.create_collection(self.collection, self.embedder.dim)
        report = self.reconcile_all(write_meta=False)
        self.store.switch_alias(self.collection)
        # Poprzednia kolekcja zostaje (szybki powrót), starsze są usuwane.
        self.store.drop_collections_except([self.collection, *([previous] if previous else [])])
        self._write_meta(report)
        return report

    # --- synchronizacja -------------------------------------------------------------

    def reconcile_all(self, write_meta: bool = True) -> SyncReport:
        """Pełne porównanie katalogu z indeksem."""
        started = time.perf_counter()
        report = SyncReport()
        by_file: dict[str, list[IndexedChunk]] = defaultdict(list)
        for ic in self.store.indexed_chunks(self.collection):
            by_file[ic.file].append(ic)

        on_disk = {
            p.name: p for p in sorted(self.settings.knowledge_dir.glob("*.md")) if is_document(p)
        }
        plan = _Plan()
        for name, path in on_disk.items():
            self._plan_file(name, path, by_file.get(name, []), report, plan)
        for name in by_file.keys() - on_disk.keys():
            self._plan_removal(name, by_file[name], report, plan)
        for name in list(self.errors):
            if name not in on_disk:
                self.errors.pop(name)

        self._apply(plan)
        return self._finish(report, started, write_meta)

    def sync_paths(self, paths: Iterable[Path]) -> SyncReport:
        """Synchronizacja plików zgłoszonych przez obserwatora."""
        started = time.perf_counter()
        report = SyncReport()
        plan = _Plan()
        for path in sorted({p for p in paths if is_document(p)}):
            indexed = self.store.indexed_chunks(self.collection, file=path.name)
            if path.exists():
                self._plan_file(path.name, path, indexed, report, plan)
            else:
                self._plan_removal(path.name, indexed, report, plan)
                self.errors.pop(path.name, None)
        self._apply(plan)
        return self._finish(report, started, write_meta=True)

    def _apply(self, plan: _Plan) -> None:
        """Najpierw zapis nowych i zmienionych fragmentów, dopiero potem usuwanie starych."""
        for i in range(0, len(plan.upsert), EMBED_BATCH):
            batch = plan.upsert[i : i + EMBED_BATCH]
            vectors = self.embedder.embed_documents([c.embed_text for c in batch])
            self.store.upsert_chunks(self.collection, batch, vectors)
        self.store.delete_points(self.collection, plan.delete)

    def _plan_file(
        self,
        name: str,
        path: Path,
        indexed: list[IndexedChunk],
        report: SyncReport,
        plan: _Plan,
    ) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("plik jest pusty")
            chunks = chunk_markdown(text, name, self.settings.max_chunk_words)
            if not chunks:
                raise ValueError("brak treści do zindeksowania")
        except FileNotFoundError:  # usunięty w trakcie
            self._plan_removal(name, indexed, report, plan)
            return
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if self.errors.get(name) != msg:
                logger.warning("%s: %s — zostaje poprzednia wersja w indeksie", name, msg)
            self.errors[name] = msg
            return
        self.errors.pop(name, None)

        existing = {ic.chunk_id: ic for ic in indexed}
        new_ids = {c.chunk_id for c in chunks}
        to_upsert = [
            c
            for c in chunks
            if c.chunk_id not in existing or existing[c.chunk_id].content_hash != c.content_hash
        ]
        stale = [ic.point_id for ic in indexed if ic.chunk_id not in new_ids]

        plan.upsert += to_upsert
        plan.delete += stale

        added = sum(1 for c in to_upsert if c.chunk_id not in existing)
        report.added += added
        report.changed += len(to_upsert) - added
        report.removed += len(stale)
        report.unchanged += len(chunks) - len(to_upsert)
        if to_upsert or stale:
            report.files_changed.append(name)
            logger.info(
                "%s: %d nowe, %d zmienione, %d usunięte",
                name,
                added,
                len(to_upsert) - added,
                len(stale),
            )

    def _plan_removal(
        self, name: str, indexed: list[IndexedChunk], report: SyncReport, plan: _Plan
    ) -> None:
        if not indexed:
            return
        plan.delete += [ic.point_id for ic in indexed]
        report.removed += len(indexed)
        report.files_changed.append(name)
        logger.info("%s: plik usunięty, usunięto %d fragmentów", name, len(indexed))

    def _finish(self, report: SyncReport, started: float, write_meta: bool) -> SyncReport:
        report.seconds = time.perf_counter() - started
        if report.has_changes:
            self.last_change = report.as_dict()
        if write_meta:
            self._write_meta(report)
        return report

    def _write_meta(self, report: SyncReport) -> None:
        indexed = self.store.indexed_chunks(self.collection)
        self.store.write_meta(
            {
                "collection": self.collection,
                "embed_model": self.embedder.model_name,
                "embed_revision": self.settings.embed_revision,
                "chunker_version": CHUNKER_VERSION,
                "files": len({ic.file for ic in indexed}),
                "chunks": len(indexed),
                "last_check": _now(),
                "last_change": self.last_change,
                "errors": dict(self.errors),
            }
        )
