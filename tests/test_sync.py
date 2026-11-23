from conftest import FakeEmbedder, FakeStore

from kb.ingest.sync import Indexer

DOC = "# Cennik\n\n## Hot Desk\nHot Desk Basic: 299 PLN.\n\n## Biuro\nBiuro 4-osobowe: 3200 PLN.\n"


def _indexer(settings, store=None, embedder=None):
    return Indexer(settings, store or FakeStore(), embedder or FakeEmbedder())


def test_bootstrap_builds_collection_and_switches_alias(settings):
    (settings.knowledge_dir / "a.md").write_text(DOC, encoding="utf-8")
    store = FakeStore()
    indexer = _indexer(settings, store)
    report = indexer.bootstrap()
    assert report.added == 2
    assert store.alias == indexer.collection
    assert store.meta["chunks"] == 2 and store.meta["files"] == 1
    assert store.meta["embed_model"] == "fake/model"


def test_only_changed_chunks_are_reembedded(settings):
    path = settings.knowledge_dir / "a.md"
    path.write_text(DOC, encoding="utf-8")
    embedder = FakeEmbedder()
    indexer = _indexer(settings, embedder=embedder)
    indexer.bootstrap()
    embedder.embedded.clear()

    path.write_text(DOC.replace("3200", "3400"), encoding="utf-8")
    report = indexer.sync_paths([path])
    assert (report.added, report.changed, report.removed, report.unchanged) == (0, 1, 0, 1)
    assert len(embedder.embedded) == 1 and "3400" in embedder.embedded[0]


def test_removed_section_and_deleted_file(settings):
    path = settings.knowledge_dir / "a.md"
    path.write_text(DOC, encoding="utf-8")
    store = FakeStore()
    indexer = _indexer(settings, store)
    indexer.bootstrap()

    path.write_text(DOC.split("## Biuro")[0], encoding="utf-8")
    assert indexer.sync_paths([path]).removed == 1

    path.unlink()
    assert indexer.sync_paths([path]).removed == 1
    assert store.meta["chunks"] == 0


def test_reconcile_detects_changes_made_while_stopped(settings):
    (settings.knowledge_dir / "a.md").write_text(DOC, encoding="utf-8")
    store = FakeStore()
    _indexer(settings, store).bootstrap()

    (settings.knowledge_dir / "a.md").unlink()
    (settings.knowledge_dir / "b.md").write_text(DOC, encoding="utf-8")
    report = _indexer(settings, store).bootstrap()  # restart usługi
    assert (report.added, report.removed) == (2, 2)


def test_empty_file_keeps_previous_version_and_reports_error(settings):
    path = settings.knowledge_dir / "a.md"
    path.write_text(DOC, encoding="utf-8")
    store = FakeStore()
    indexer = _indexer(settings, store)
    indexer.bootstrap()

    path.write_text("   \n", encoding="utf-8")
    report = indexer.sync_paths([path])
    assert report.removed == 0
    assert store.meta["chunks"] == 2
    assert "a.md" in store.meta["errors"]

    path.write_text(DOC, encoding="utf-8")
    indexer.sync_paths([path])
    assert store.meta["errors"] == {}


def test_hidden_and_non_markdown_files_are_ignored(settings):
    (settings.knowledge_dir / "a.md").write_text(DOC, encoding="utf-8")
    (settings.knowledge_dir / ".a.md.swp").write_text(DOC, encoding="utf-8")
    (settings.knowledge_dir / "notes.txt").write_text(DOC, encoding="utf-8")
    store = FakeStore()
    _indexer(settings, store).bootstrap()
    assert store.meta["files"] == 1


def test_model_change_builds_new_collection_and_keeps_previous(settings):
    (settings.knowledge_dir / "a.md").write_text(DOC, encoding="utf-8")
    store = FakeStore()
    first = _indexer(settings, store)
    first.bootstrap()

    other = FakeEmbedder()
    other.model_name = "fake/other-model"
    second = _indexer(settings, store, other)
    second.bootstrap()
    assert second.collection != first.collection
    assert store.alias == second.collection
    assert set(store.collections) == {first.collection, second.collection}


def test_new_content_is_written_before_old_is_deleted(settings):
    """Pytanie w trakcie aktualizacji nie może trafić na brakującą treść."""
    path = settings.knowledge_dir / "a.md"
    path.write_text(DOC, encoding="utf-8")
    store = FakeStore()
    indexer = _indexer(settings, store)
    indexer.bootstrap()
    store.ops.clear()

    path.write_text(DOC.replace("## Biuro", "## Biura dedykowane"), encoding="utf-8")
    report = indexer.sync_paths([path])
    assert (report.added, report.removed) == (1, 1)
    assert store.ops == ["upsert", "delete"]


def test_all_files_are_embedded_in_one_batch(settings):
    for name in ("a.md", "b.md", "c.md"):
        (settings.knowledge_dir / name).write_text(DOC, encoding="utf-8")
    store = FakeStore()
    _indexer(settings, store).bootstrap()
    assert store.ops == ["upsert"]
    assert store.meta["chunks"] == 6
