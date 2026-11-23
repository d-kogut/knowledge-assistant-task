"""Dzielenie dokumentu Markdown na fragmenty według nagłówków.

Zasady:
- fragment = sekcja H1/H2/H3 z własną treścią (tekst pod nagłówkiem, który ma podsekcje,
  też jest fragmentem — inaczej by zginął),
- tabela nigdy nie jest cięta,
- sekcja dłuższa niż `max_words` jest dzielona na granicach bloków Markdown
  (akapit, lista, tabela, blok kodu),
- nagłówki rozpoznaje parser (markdown-it-py), więc `#` w bloku kodu nie jest nagłówkiem.
"""

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property

from markdown_it import MarkdownIt
from markdown_it.token import Token

from kb.text import fold

CHUNKER_VERSION = "1"
MAX_HEADING_LEVEL = 3
BREADCRUMB_SEP = " › "

_md = MarkdownIt("commonmark").enable("table")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file: str
    headings: tuple[str, ...]
    body: str
    line_start: int  # 1-based, linia nagłówka sekcji
    line_end: int  # 1-based, ostatnia niepusta linia treści

    @property
    def breadcrumb(self) -> str:
        return BREADCRUMB_SEP.join(self.headings)

    @property
    def section(self) -> str:
        return self.headings[-1] if self.headings else self.file

    @property
    def embed_text(self) -> str:
        """Tekst dla modelu embeddingów i BM25: ścieżka nagłówków daje kontekst sekcji."""
        return f"{self.breadcrumb}\n{self.body}"

    @cached_property
    def content_hash(self) -> str:
        data = f"{CHUNKER_VERSION}\n{self.chunk_id}\n{self.embed_text}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def citation_blocks(self) -> list[str]:
        """Treść podzielona na linie (bez separatorów tabel) — najmniejsza jednostka cytatu."""
        return [
            line.strip()
            for line in self.body.splitlines()
            if line.strip() and not _TABLE_SEPARATOR.match(line)
        ]


@dataclass
class _Section:
    headings: tuple[str, ...]
    heading_line: int  # 0-based
    body_start: int  # 0-based, włącznie
    body_end: int  # 0-based, wyłącznie


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text).lower()).strip("-")


def _word_count(text: str) -> int:
    return len(text.split())


def _sections(tokens: list[Token], n_lines: int, file: str) -> list[_Section]:
    heads: list[tuple[int, int, int, str]] = []  # (level, start, end, title)
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.level == 0 and tok.map:
            level = int(tok.tag[1])
            if level <= MAX_HEADING_LEVEL:
                heads.append((level, tok.map[0], tok.map[1], tokens[i + 1].content.strip()))

    sections: list[_Section] = []
    first_start = heads[0][1] if heads else n_lines
    if first_start > 0:  # tekst przed pierwszym nagłówkiem
        sections.append(_Section((file,), 0, 0, first_start))

    path: list[str] = []
    for idx, (level, start, end, title) in enumerate(heads):
        path = [*path[: level - 1], title]
        next_start = heads[idx + 1][1] if idx + 1 < len(heads) else n_lines
        sections.append(_Section(tuple(path), start, end, next_start))
    return sections


def _blocks(tokens: list[Token], start: int, end: int) -> list[tuple[int, int]]:
    """Zakresy linii bloków najwyższego poziomu w [start, end)."""
    blocks = []
    for tok in tokens:
        if tok.level == 0 and tok.nesting in (0, 1) and tok.map and tok.type != "heading_open":
            b_start, b_end = tok.map
            if start <= b_start and b_end <= end:
                blocks.append((b_start, b_end))
    return blocks


def _split_ranges(
    tokens: list[Token], lines: list[str], start: int, end: int, max_words: int
) -> list[tuple[int, int]]:
    text = "\n".join(lines[start:end])
    if _word_count(text) <= max_words:
        return [(start, end)]
    blocks = _blocks(tokens, start, end)
    if not blocks:
        return [(start, end)]
    ranges: list[tuple[int, int]] = []
    cur_start, cur_words = blocks[0][0], 0
    for b_start, b_end in blocks:
        words = _word_count("\n".join(lines[b_start:b_end]))
        if cur_words and cur_words + words > max_words:
            ranges.append((cur_start, b_start))
            cur_start, cur_words = b_start, 0
        cur_words += words
    ranges.append((cur_start, end))
    return ranges


def chunk_markdown(text: str, file: str, max_words: int = 350) -> list[Chunk]:
    lines = text.splitlines()
    tokens = _md.parse(text)
    title = next(
        (
            tokens[i + 1].content.strip()
            for i, t in enumerate(tokens)
            if t.type == "heading_open" and t.tag == "h1"
        ),
        file,
    )

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for sec in _sections(tokens, len(lines), file):
        ranges = _split_ranges(tokens, lines, sec.body_start, sec.body_end, max_words)
        parts = [(s, e) for s, e in ranges if "\n".join(lines[s:e]).strip()]
        for part_no, (start, end) in enumerate(parts, start=1):
            body_lines = lines[start:end]
            last = max(i for i, line in enumerate(body_lines) if line.strip())
            body = "\n".join(body_lines[: last + 1]).strip("\n")
            if sec.headings == (file,):  # tekst przed pierwszym nagłówkiem
                headings: tuple[str, ...] = (title,)
            elif sec.headings[0] == title:
                headings = sec.headings
            else:  # plik bez H1
                headings = (title, *sec.headings)
            base_id = f"{file}#{_slug('/'.join(headings[1:])) or 'intro'}"
            chunk_id = base_id if len(parts) == 1 else f"{base_id}~{part_no}"
            suffix = 2
            while chunk_id in seen:  # zduplikowana ścieżka nagłówków w jednym pliku
                chunk_id = f"{base_id}-{suffix}"
                suffix += 1
            seen.add(chunk_id)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file=file,
                    headings=headings,
                    body=body,
                    line_start=(sec.heading_line if part_no == 1 else start) + 1,
                    line_end=start + last + 1,
                )
            )
    return chunks
