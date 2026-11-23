"""Modele danych API (żądanie i odpowiedź)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["hybrid", "dense", "bm25", "full"]
ProviderName = Literal["anthropic", "openai"]
Status = Literal["answered", "not_found", "fragments_only"]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2, max_length=1000, examples=["Ile kosztuje Hot Desk Student?"]
    )
    mode: Mode = Field(
        default="hybrid",
        description="hybrid = wektory + BM25 (domyślnie), dense / bm25 = jedna metoda, "
        "full = cała dokumentacja w prompcie (punkt odniesienia w testach)",
    )
    llm: bool = Field(default=True, description="false = tylko najlepsze fragmenty, bez modelu")
    provider: ProviderName | None = Field(
        default=None, description="anthropic / openai; brak = dostawca domyślny serwisu"
    )
    debug: bool = Field(default=False, description="dołącz wyniki każdego etapu wyszukiwania")
    top_k: int | None = Field(default=None, ge=1, le=50)
    model: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9.\-]{1,63}$",
        description="nadpisanie modelu dostawcy, np. claude-opus-5-5 albo gpt-6.2-astra",
    )


class Segment(BaseModel):
    text: str
    sources: list[int] = Field(default_factory=list, description="numery źródeł z listy sources")


class Source(BaseModel):
    n: int
    chunk_id: str
    file: str
    breadcrumb: str
    line_start: int
    line_end: int
    cited_texts: list[str]


class Fragment(BaseModel):
    chunk_id: str
    file: str
    breadcrumb: str
    line_start: int
    line_end: int
    body: str
    score: float


class AskResponse(BaseModel):
    status: Status
    note: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    fragments: list[Fragment] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None

    @property
    def answer_text(self) -> str:
        return "".join(s.text for s in self.segments)
