"""Usługa ingest: pierwsza synchronizacja, potem obserwacja katalogu na żywo.

Uruchomienie: `python -m kb.ingest`.
"""

import logging
import signal
import threading
from pathlib import Path
from types import FrameType

from watchfiles import Change, DefaultFilter, watch

from kb.config import get_settings
from kb.embedder import SentenceTransformerEmbedder
from kb.ingest.sync import Indexer, is_document
from kb.logsetup import setup_logging
from kb.store import QdrantStore
from kb.text import lemma_text

logger = logging.getLogger("kb.ingest")


class MarkdownFilter(DefaultFilter):
    """Tylko pliki *.md; pomija pliki ukryte i tymczasowe pliki edytorów."""

    def __call__(self, change: Change, path: str) -> bool:
        return bool(super().__call__(change, path)) and is_document(Path(path))


def main() -> None:
    setup_logging()
    settings = get_settings()
    if not settings.knowledge_dir.is_dir():
        raise SystemExit(f"Brak katalogu z dokumentami: {settings.knowledge_dir}")

    stop = threading.Event()

    def _stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("Zatrzymuję (sygnał %d)", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    store = QdrantStore(settings.qdrant_url, settings.collection_alias, settings.meta_collection)
    store.wait_until_ready()
    embedder = SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_revision,
        settings.embed_query_prefix,
        settings.embed_batch_size,
    )
    # Pierwsze wywołanie modelu (ok. 1 s) i wczytanie słownika lematyzatora (ok. 1,8 s) są
    # wolne; bez rozgrzewki płaci za to pierwsza zmiana pliku po starcie.
    embedder.embed_documents(["rozgrzewka"])
    lemma_text("rozgrzewka")
    indexer = Indexer(settings, store, embedder)

    report = indexer.bootstrap()
    logger.info(
        "Gotowe: %d nowe, %d zmienione, %d usunięte, %d bez zmian (%.1f s)",
        report.added,
        report.changed,
        report.removed,
        report.unchanged,
        report.seconds,
    )
    settings.ready_file.touch()
    logger.info(
        "Obserwuję %s (kontrolne porównanie co %d s)",
        settings.knowledge_dir,
        settings.reconcile_interval_s,
    )

    for changes in watch(
        settings.knowledge_dir,
        watch_filter=MarkdownFilter(),
        debounce=settings.debounce_ms,
        rust_timeout=settings.reconcile_interval_s * 1000,
        yield_on_timeout=True,
        stop_event=stop,
        raise_interrupt=False,
    ):
        try:
            if changes:
                indexer.sync_paths({Path(path) for _, path in changes})
            else:
                indexer.reconcile_all()
        except Exception:  # Qdrant chwilowo niedostępny itp. — spróbujemy przy kolejnej zmianie
            logger.exception("Synchronizacja nieudana, ponowię przy następnym cyklu")
    settings.ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
