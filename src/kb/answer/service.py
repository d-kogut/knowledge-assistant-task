"""Obsługa pytania: wyszukiwanie → (opcjonalnie) model językowy → odpowiedź ze źródłami."""

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

from kb.answer.llm import LlmAnswer, LlmClient
from kb.answer.schemas import AskRequest, AskResponse, Fragment, Segment, Source
from kb.config import PROVIDERS, Settings
from kb.embedder import Embedder
from kb.store import Hit, SearchMode
from kb.text import lemma_text

logger = logging.getLogger(__name__)

NOT_FOUND_NOTE = "Nie ma tego w dokumentacji."
PROVIDER_LABELS = {"anthropic": "Claude (Anthropic)", "openai": "OpenAI"}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class SearchStore(Protocol):
    def search(
        self, mode: SearchMode, vector: list[float] | None, lemmas: str, limit: int, prefetch: int
    ) -> list[Hit]: ...
    def all_hits(self) -> list[Hit]: ...
    def alias_target(self) -> str | None: ...
    def read_meta(self) -> dict[str, Any] | None: ...
    def count(self) -> int: ...


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _fragment(hit: Hit) -> Fragment:
    return Fragment(
        chunk_id=hit.chunk_id,
        file=hit.file,
        breadcrumb=hit.breadcrumb,
        line_start=hit.line_start,
        line_end=hit.line_end,
        body=hit.body,
        score=round(hit.score, 4),
    )


def _ranking(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    return [
        {
            "rank": i,
            "chunk_id": h.chunk_id,
            "file": h.file,
            "section": h.section,
            "score": round(h.score, 4),
        }
        for i, h in enumerate(hits, start=1)
    ]


class AskService:
    def __init__(
        self,
        settings: Settings,
        store: SearchStore,
        embedder: Embedder,
        llms: dict[str, LlmClient] | None = None,
        default_provider: str | None = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llms = llms or {}
        if default_provider not in self.llms:
            default_provider = next(iter(self.llms), None)
        self.default_provider = default_provider

    # --- wyszukiwanie -------------------------------------------------------------

    def retrieve(
        self, question: str, mode: str, k: int, debug: bool
    ) -> tuple[list[Hit], dict[str, Any]]:
        if mode == "full":
            return self.store.all_hits(), {}
        lemmas = lemma_text(question)
        vector = self.embedder.embed_query(question) if mode != "bm25" else None
        search_mode: SearchMode = (
            "dense" if mode == "dense" else "bm25" if mode == "bm25" else "hybrid"
        )
        hits = self.store.search(search_mode, vector, lemmas, k, self.settings.prefetch_k)
        info: dict[str, Any] = {}
        if debug:
            info = {"query_lemmas": lemmas, mode: _ranking(hits)}
            if mode == "hybrid":
                pk = self.settings.prefetch_k
                info["dense"] = _ranking(self.store.search("dense", vector, lemmas, pk, pk)[:k])
                info["bm25"] = _ranking(self.store.search("bm25", None, lemmas, pk, pk)[:k])
        return hits, info

    # --- odpowiedź ------------------------------------------------------------------

    def ask(self, req: AskRequest) -> AskResponse:
        started = time.perf_counter()
        question = req.question.strip()
        k = req.top_k or self.settings.top_k

        t = time.perf_counter()
        hits, debug_info = self.retrieve(question, req.mode, k, req.debug)
        timings = {"retrieval": _ms(t)}
        debug = {"mode": req.mode, **debug_info} if req.debug else None

        if not hits:
            timings["total"] = _ms(started)
            return AskResponse(
                status="not_found", note=NOT_FOUND_NOTE, timings_ms=timings, debug=debug
            )

        provider = req.provider or self.default_provider
        llm = self.llms.get(provider) if provider else None
        if not req.llm or llm is None:
            if not req.llm:
                reason = "Odpowiedź bez modelu językowego (wyłączony w zapytaniu)."
            elif not self.llms:
                reason = (
                    "Odpowiedź bez modelu językowego: brak klucza API "
                    "(ANTHROPIC_API_KEY albo OPENAI_API_KEY)."
                )
            else:
                key = KEY_NAMES.get(provider or "", "klucza API")
                reason = f"Odpowiedź bez modelu językowego: brak klucza {key}."
            return self._fragments_only(hits, reason, timings, started, debug)

        t = time.perf_counter()
        try:
            answer = llm.answer(question, hits, model=req.model, cache=req.mode == "full")
        except Exception as exc:  # błąd API, limit, sieć — pokazujemy chociaż fragmenty
            logger.warning("%s niedostępny: %s: %s", provider, type(exc).__name__, exc)
            reason = (
                f"Odpowiedź bez modelu językowego: {PROVIDER_LABELS.get(llm.provider, provider)} "
                f"niedostępny ({type(exc).__name__})."
            )
            return self._fragments_only(hits, reason, timings, started, debug)
        timings["llm"] = _ms(t)
        timings["total"] = _ms(started)
        return self._from_llm(answer, hits, timings, debug, llm.provider)

    def _fragments_only(
        self,
        hits: Sequence[Hit],
        reason: str,
        timings: dict[str, float],
        started: float,
        debug: dict[str, Any] | None,
    ) -> AskResponse:
        timings["total"] = _ms(started)
        top = hits[: self.settings.fragments_without_llm]
        return AskResponse(
            status="fragments_only",
            note=reason,
            fragments=[_fragment(h) for h in top],
            timings_ms=timings,
            debug=debug,
        )

    def _from_llm(
        self,
        answer: LlmAnswer,
        hits: Sequence[Hit],
        timings: dict[str, float],
        debug: dict[str, Any] | None,
        provider: str,
    ) -> AskResponse:
        if debug is not None:
            debug["llm_raw_text"] = answer.text
            debug["context"] = [h.chunk_id for h in hits]

        # Numeracja źródeł w kolejności pierwszego cytowania.
        numbers: dict[int, int] = {}
        for seg in answer.segments:
            for idx in seg.result_indexes:
                if 0 <= idx < len(hits) and idx not in numbers:
                    numbers[idx] = len(numbers) + 1

        common = {
            "provider": provider,
            "model": answer.model,
            "usage": answer.usage,
            "timings_ms": timings,
            "debug": debug,
        }
        if answer.not_found or not numbers:
            note = NOT_FOUND_NOTE
            if not answer.not_found:  # odpowiedź bez żadnego cytatu nie jest pokazywana
                note += " (Model odpowiedział bez cytatów ze źródeł, więc odpowiedź odrzucono.)"
            return AskResponse(status="not_found", note=note, **common)

        segments = [
            Segment(text=s.text, sources=[numbers[i] for i in s.result_indexes if i in numbers])
            for s in answer.segments
        ]
        sources = [
            Source(
                n=n,
                chunk_id=hits[idx].chunk_id,
                file=hits[idx].file,
                breadcrumb=hits[idx].breadcrumb,
                line_start=hits[idx].line_start,
                line_end=hits[idx].line_end,
                cited_texts=answer.cited.get(idx, []),
            )
            for idx, n in sorted(numbers.items(), key=lambda kv: kv[1])
        ]
        return AskResponse(status="answered", segments=segments, sources=sources, **common)

    # --- stan -------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        meta = self.store.read_meta() or {}
        return {
            "collection": self.store.alias_target(),
            "files": meta.get("files"),
            "chunks": meta.get("chunks"),
            "last_check": meta.get("last_check"),
            "last_change": meta.get("last_change"),
            "errors": meta.get("errors", {}),
            "embed_model": self.settings.embed_model,
            "llm_enabled": bool(self.llms),
            "default_provider": self.default_provider,
            "llm_model": (
                self.llms[self.default_provider].default_model if self.default_provider else None
            ),
            "providers": [
                {
                    "name": name,
                    "label": PROVIDER_LABELS[name],
                    "enabled": name in self.llms,
                    "model": (
                        self.llms[name].default_model
                        if name in self.llms
                        else self.settings.model_for(name)
                    ),
                    "key": KEY_NAMES[name],
                }
                for name in PROVIDERS
            ],
        }

    def health(self) -> tuple[bool, str]:
        try:
            target = self.store.alias_target()
            meta = self.store.read_meta() or {}
        except Exception as exc:
            return False, f"Qdrant niedostępny: {type(exc).__name__}"
        if target is None:
            return False, "Indeks jeszcze nie istnieje (ingest w toku)"
        if (meta.get("embed_model"), meta.get("embed_revision")) != (
            self.settings.embed_model,
            self.settings.embed_revision,
        ):
            return False, "Model embeddingów api różni się od modelu, którym zbudowano indeks"
        return True, "ok"
