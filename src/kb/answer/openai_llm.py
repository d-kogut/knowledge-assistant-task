"""Odpowiedź przez OpenAI (Responses API + Structured Outputs) z cytatami sprawdzanymi w kodzie.

OpenAI nie ma odpowiednika cytatów `search_result` z Anthropic dla własnych dokumentów
(automatyczne cytaty dają tylko narzędzia hostowane: file search i web search, które
zastąpiłyby nasze wyszukiwanie). Dlatego:

1. każdy fragment dostaje id (`f1`, `f2`, …), a każda jego linia numer (`L1`, `L2`, …),
2. model zwraca JSON wymuszony schematem (Structured Outputs): segmenty odpowiedzi
   i przy każdym id fragmentu z numerami linii,
3. kod rozwiązuje odwołania do tekstu linii — cytat jest więc dosłowny z definicji —
   a odwołania do nieistniejących fragmentów lub linii odrzuca.

Odpowiedź bez ani jednego poprawnego odwołania jest odrzucana tak samo jak u Claude'a.
"""

from collections.abc import Sequence
from typing import Any

import openai
from pydantic import BaseModel, Field

from kb.answer.llm import LlmAnswer, LlmSegment, system_prompt
from kb.store import Hit

SYSTEM_PROMPT = system_prompt(
    [
        'Fragmenty dostajesz w znacznikach <fragment id="f…">; każda linia ma numer L…. '
        "Odpowiedź zwracasz jako JSON: listę segmentów, gdzie segment to jedno zdanie albo "
        "jeden punkt listy (punkt listy zaczyna się od „- ”).",
        "Każdy segment z informacją z dokumentacji musi mieć co najmniej jedno cytowanie: "
        "id fragmentu i numery linii, które bezpośrednio potwierdzają ten segment. "
        "Nie cytuj fragmentów, z których nie korzystasz.",
        "Jeśli fragmenty nie zawierają odpowiedzi na pytanie, ustaw found=false i zwróć "
        "pustą listę segmentów.",
    ]
)


class Citation(BaseModel):
    fragment: str = Field(description="id fragmentu, np. f3")
    lines: list[int] = Field(description="numery linii fragmentu (L…) potwierdzające segment")


class AnswerSegment(BaseModel):
    text: str = Field(description="jedno zdanie albo jeden punkt listy zaczynający się od '- '")
    citations: list[Citation]


class StructuredAnswer(BaseModel):
    found: bool = Field(description="false, jeśli fragmenty nie zawierają odpowiedzi")
    segments: list[AnswerSegment]


def fragment_id(index: int) -> str:
    return f"f{index + 1}"


def build_input(question: str, hits: Sequence[Hit]) -> str:
    """Fragmenty z id i numerami linii, na końcu pytanie (stały prefiks sprzyja cache)."""
    parts = []
    for i, hit in enumerate(hits):
        lines = "\n".join(f"L{n}: {text}" for n, text in enumerate(hit.citation_blocks(), 1))
        parts.append(
            f'<fragment id="{fragment_id(i)}" plik="{hit.file}" sekcja="{hit.breadcrumb}">\n'
            f"{lines}\n</fragment>"
        )
    return "\n\n".join(parts) + f"\n\nPytanie: {question}"


def _join(segments: list[str]) -> list[str]:
    """Skleja zdania spacją, a punkty listy umieszcza w nowych liniach."""
    out: list[str] = []
    for i, raw in enumerate(segments):
        text = raw.strip()
        if i == 0:
            out.append(text)
        elif text.startswith(("- ", "* ", "• ")) or out[-1].rstrip().endswith(":"):
            out.append("\n" + text)
        else:
            out.append(" " + text)
    return out


def _usage(usage_obj: Any) -> dict[str, int]:
    if usage_obj is None:
        return {}
    cached = int(getattr(getattr(usage_obj, "input_tokens_details", None), "cached_tokens", 0) or 0)
    reasoning = int(
        getattr(getattr(usage_obj, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    )
    # Ten sam kształt co u Anthropic: input_tokens = wejście bez cache.
    # output_tokens zawiera tokeny rozumowania (tak liczy je cennik OpenAI).
    return {
        "input_tokens": int(usage_obj.input_tokens or 0) - cached,
        "cache_read_input_tokens": cached,
        "output_tokens": int(usage_obj.output_tokens or 0),
        "reasoning_tokens": reasoning,
    }


def parse_response(response: Any, hits: Sequence[Hit]) -> LlmAnswer:
    parsed: StructuredAnswer | None = getattr(response, "output_parsed", None)
    if parsed is None:  # odmowa modelu albo ucięta odpowiedź (limit tokenów)
        status = getattr(response, "status", "?")
        raise ValueError(f"brak poprawnej odpowiedzi JSON (status: {status})")

    ids = {fragment_id(i): i for i in range(len(hits))}
    blocks = [h.citation_blocks() for h in hits]
    cited: dict[int, list[str]] = {}
    texts = _join([s.text for s in parsed.segments])
    segments: list[LlmSegment] = []
    for seg, text in zip(parsed.segments, texts, strict=True):
        indexes: list[int] = []
        for c in seg.citations:
            idx = ids.get(c.fragment.strip())
            if idx is None:
                continue  # nieistniejący fragment — odrzucamy
            if idx not in indexes:
                indexes.append(idx)
            quotes = cited.setdefault(idx, [])
            for n in c.lines:
                if 1 <= n <= len(blocks[idx]) and blocks[idx][n - 1] not in quotes:
                    quotes.append(blocks[idx][n - 1])  # dosłowna linia źródła
        segments.append(LlmSegment(text, indexes))

    not_found = not parsed.found or not "".join(texts).strip()
    return LlmAnswer(
        segments=[] if not_found else segments,
        cited={} if not_found else cited,
        not_found=not_found,
        model=str(getattr(response, "model", "")),
        usage=_usage(getattr(response, "usage", None)),
    )


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int,
        timeout_s: float,
        reasoning_effort: str | None = None,
        client: Any | None = None,
    ):
        self.default_model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self._client: Any = client or openai.OpenAI(
            api_key=api_key, timeout=timeout_s, max_retries=2
        )

    def answer(
        self, question: str, hits: Sequence[Hit], model: str | None = None, cache: bool = False
    ) -> LlmAnswer:
        # cache: OpenAI cache'uje wspólny prefiks automatycznie (fragmenty są przed pytaniem).
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self._client.responses.parse(
            model=model or self.default_model,
            instructions=SYSTEM_PROMPT,
            input=build_input(question, hits),
            text_format=StructuredAnswer,
            max_output_tokens=self.max_output_tokens,
            store=False,  # nie przechowuj treści dokumentów po stronie OpenAI
            **kwargs,
        )
        return parse_response(response, hits)
