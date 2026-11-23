"""Odpowiedź przez Claude'a (Anthropic) z cytatami API.

Fragmenty trafiają do modelu jako bloki `search_result` z włączonymi cytatami.
Każdy cytat w odpowiedzi zawiera `cited_text` wyciągnięty przez API dosłownie ze źródła,
więc model nie może podać nieistniejącego cytatu. Fragment jest podzielony na bloki-linie
(wiersz tabeli = blok), dzięki czemu cytaty są krótkie i konkretne.

Cytatów nie da się połączyć z wymuszonym JSON-em (API zwraca 400), więc odpowiedź to
zwykły tekst z blokami cytatów, a JSON dla klienta składa nasz kod.
"""

from collections.abc import Sequence
from typing import Any

import anthropic

from kb.answer.llm import LlmAnswer, LlmSegment, system_prompt
from kb.store import Hit

NOT_FOUND_MARKER = "BRAK_W_DOKUMENTACJI"

SYSTEM_PROMPT = system_prompt(
    [
        "Fragmenty dostajesz jako bloki search_result. Każde twierdzenie opieraj na "
        "konkretnym fragmencie — cytaty są włączone.",
        "Jeśli fragmenty nie zawierają odpowiedzi na pytanie, odpowiedz dokładnie: "
        f"{NOT_FOUND_MARKER} — i nic więcej.",
    ]
)


def search_result_block(hit: Hit, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "search_result",
        "source": hit.chunk_id,
        "title": f"{hit.file} › {hit.breadcrumb}",
        "content": [{"type": "text", "text": line} for line in hit.citation_blocks()],
        "citations": {"enabled": True},
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def build_messages(question: str, hits: Sequence[Hit], cache: bool = False) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        search_result_block(h, cache=cache and i == len(hits) - 1) for i, h in enumerate(hits)
    ]
    content.append({"type": "text", "text": f"Pytanie: {question}"})
    return [{"role": "user", "content": content}]


def parse_response(response: Any) -> LlmAnswer:
    """Zamienia odpowiedź Messages API na segmenty tekstu z indeksami źródeł."""
    segments: list[LlmSegment] = []
    cited: dict[int, list[str]] = {}
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue  # np. bloki thinking
        indexes: list[int] = []
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", None) != "search_result_location":
                continue
            idx = int(c.search_result_index)
            if idx not in indexes:
                indexes.append(idx)
            texts = cited.setdefault(idx, [])
            if c.cited_text and c.cited_text not in texts:
                texts.append(c.cited_text)
        segments.append(LlmSegment(block.text, indexes))

    full = "".join(s.text for s in segments).strip()
    not_found = full.startswith(NOT_FOUND_MARKER) or not full
    if not not_found and NOT_FOUND_MARKER in full:  # znacznik przy częściowej odpowiedzi
        for s in segments:
            s.text = s.text.replace(NOT_FOUND_MARKER, "").replace("  ", " ")

    usage_obj = getattr(response, "usage", None)
    usage = {
        key: int(getattr(usage_obj, key) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        if usage_obj is not None and getattr(usage_obj, key, None) is not None
    }
    return LlmAnswer(segments, cited, not_found, str(getattr(response, "model", "")), usage)


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        client: Any | None = None,
    ):
        self.default_model = model
        self.max_tokens = max_tokens
        self._client: Any = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=2
        )

    def answer(
        self, question: str, hits: Sequence[Hit], model: str | None = None, cache: bool = False
    ) -> LlmAnswer:
        response = self._client.messages.create(
            model=model or self.default_model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=build_messages(question, hits, cache=cache),
        )
        return parse_response(response)
