from pathlib import Path

import pytest

from kb.ingest.chunker import _TABLE_SEPARATOR, chunk_markdown

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


def _corpus() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(KNOWLEDGE.glob("*.md"))}


@pytest.fixture(scope="module")
def corpus_chunks():
    return {name: chunk_markdown(text, name) for name, text in _corpus().items()}


def test_every_content_line_is_in_some_chunk(corpus_chunks):
    """Żadna linia treści (poza nagłówkami) nie może zginąć przy dzieleniu."""
    for name, text in _corpus().items():
        bodies = "\n".join(c.body for c in corpus_chunks[name])
        for line in text.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                assert line.strip() in bodies, f"{name}: zgubiona linia: {line!r}"


def test_chunk_ids_are_unique_and_stable(corpus_chunks):
    ids = [c.chunk_id for chunks in corpus_chunks.values() for c in chunks]
    assert len(ids) == len(set(ids))
    again = [c.chunk_id for name, text in _corpus().items() for c in chunk_markdown(text, name)]
    assert ids == again


def test_tables_are_never_split(corpus_chunks):
    for chunks in corpus_chunks.values():
        for c in chunks:
            lines = c.body.splitlines()
            rows = [i for i, line in enumerate(lines) if line.lstrip().startswith("|")]
            if rows:
                assert any(_TABLE_SEPARATOR.match(line) for line in lines), c.chunk_id


def test_text_under_heading_with_subsections_gets_own_chunk(corpus_chunks):
    locations = [c for c in corpus_chunks["01-company-overview.md"] if c.section == "Lokalizacje"]
    assert len(locations) == 1
    assert "SpaceHub Czyżyny" in locations[0].body
    perks = [
        c
        for c in corpus_chunks["08-community-and-events.md"]
        if c.section == "Program SpaceHub Perks"
    ]
    assert perks and "Fitness Platinium" in perks[0].body


def test_breadcrumb_and_lines():
    text = "# Tytuł\n\n## Sekcja\nTekst sekcji.\n\n### Pod\n- punkt\n"
    chunks = chunk_markdown(text, "a.md")
    assert [c.breadcrumb for c in chunks] == ["Tytuł › Sekcja", "Tytuł › Sekcja › Pod"]
    assert (chunks[0].line_start, chunks[0].line_end) == (3, 4)
    assert (chunks[1].line_start, chunks[1].line_end) == (6, 7)
    assert chunks[1].embed_text.startswith("Tytuł › Sekcja › Pod\n")


def test_hash_in_code_block_is_not_a_heading():
    text = "# T\n\n## Kod\n```bash\n# komentarz, nie nagłówek\necho 1\n```\n"
    chunks = chunk_markdown(text, "a.md")
    assert len(chunks) == 1
    assert "# komentarz" in chunks[0].body


def test_long_section_is_split_on_block_boundaries_keeping_table():
    paragraph = " ".join(["słowo"] * 120)
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i} |" for i in range(40))
    text = f"# T\n\n## Długa\n{paragraph}\n\n{paragraph}\n\n{table}\n\n{paragraph}\n"
    chunks = chunk_markdown(text, "a.md", max_words=200)
    assert len(chunks) >= 3
    with_table = [c for c in chunks if "| 0 | 0 |" in c.body]
    assert len(with_table) == 1 and "| 39 | 39 |" in with_table[0].body
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_content_hash_changes_with_content():
    a = chunk_markdown("# T\n\n## S\nA\n", "a.md")[0]
    b = chunk_markdown("# T\n\n## S\nB\n", "a.md")[0]
    assert a.chunk_id == b.chunk_id
    assert a.content_hash != b.content_hash


def test_citation_blocks_skip_table_separator():
    chunk = chunk_markdown("# T\n\n## S\n| a | b |\n|---|---|\n| 1 | 2 |\n", "a.md")[0]
    assert chunk.citation_blocks() == ["| a | b |", "| 1 | 2 |"]
