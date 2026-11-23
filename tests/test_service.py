from conftest import FakeEmbedder, FakeStore, citation, make_hit, response, text_block

from kb.answer.anthropic_llm import NOT_FOUND_MARKER, build_messages, parse_response
from kb.answer.schemas import AskRequest
from kb.answer.service import AskService


class FakeLlm:
    default_model = "claude-test"

    def __init__(self, resp=None, error: Exception | None = None, provider: str = "anthropic"):
        self.provider = provider
        self.resp = resp
        self.error = error
        self.calls: list[tuple[str, int, bool]] = []

    def answer(self, question, hits, model=None, cache=False):
        self.calls.append((question, len(hits), cache))
        if self.error:
            raise self.error
        return parse_response(self.resp)


def _service(settings, llm=None, hits=None):
    hits = hits if hits is not None else [make_hit(i) for i in range(5)]
    llms = {llm.provider: llm} if llm else {}
    return AskService(settings, FakeStore(hits), FakeEmbedder(), llms)


def test_search_results_are_split_into_line_blocks_with_citations():
    hit = make_hit(0, body="| Plan | Cena |\n|---|---|\n| Student | 149 PLN |")
    msg = build_messages("Ile?", [hit])[0]
    block = msg["content"][0]
    assert block["type"] == "search_result" and block["citations"] == {"enabled": True}
    assert [b["text"] for b in block["content"]] == ["| Plan | Cena |", "| Student | 149 PLN |"]
    assert msg["content"][-1] == {"type": "text", "text": "Pytanie: Ile?"}


def test_answer_with_citations_numbers_sources_in_order_of_use(settings):
    resp = response(
        text_block("Kosztuje 149 PLN", citation(2, "| Student | 149 PLN |")),
        text_block(" i wymaga legitymacji", citation(0, "Wymaga legitymacji.")),
        text_block("."),
    )
    out = _service(settings, FakeLlm(resp)).ask(AskRequest(question="Ile kosztuje Student?"))
    assert out.status == "answered"
    assert [s.sources for s in out.segments] == [[1], [2], []]
    assert [s.chunk_id for s in out.sources] == ["02-plans.md#s2", "02-plans.md#s0"]
    assert out.sources[0].cited_texts == ["| Student | 149 PLN |"]
    assert out.answer_text == "Kosztuje 149 PLN i wymaga legitymacji."


def test_marker_means_not_found(settings):
    out = _service(settings, FakeLlm(response(text_block(NOT_FOUND_MARKER)))).ask(
        AskRequest(question="Jaka jest stolica Francji?")
    )
    assert out.status == "not_found" and out.segments == []


def test_answer_without_any_citation_is_rejected(settings):
    out = _service(settings, FakeLlm(response(text_block("Paryż.")))).ask(
        AskRequest(question="Jaka jest stolica Francji?")
    )
    assert out.status == "not_found"
    assert "bez cytatów" in out.note


def test_without_api_key_returns_top_fragments(settings):
    out = _service(settings, llm=None).ask(AskRequest(question="Ile kosztuje Hot Desk?"))
    assert out.status == "fragments_only"
    assert len(out.fragments) == settings.fragments_without_llm
    assert "ANTHROPIC_API_KEY" in out.note and "OPENAI_API_KEY" in out.note


def test_api_error_falls_back_to_fragments(settings):
    out = _service(settings, FakeLlm(error=TimeoutError("timeout"))).ask(
        AskRequest(question="Ile kosztuje Hot Desk?")
    )
    assert out.status == "fragments_only"
    assert "TimeoutError" in out.note and "Claude" in out.note


def test_no_hits_is_not_found_without_calling_llm(settings):
    llm = FakeLlm(response(text_block("x")))
    out = _service(settings, llm, hits=[]).ask(AskRequest(question="???"))
    assert out.status == "not_found" and llm.calls == []


def test_full_mode_sends_all_chunks_with_cache(settings):
    llm = FakeLlm(response(text_block("Tak", citation(0, "Treść"))))
    _service(settings, llm, hits=[make_hit(i) for i in range(20)]).ask(
        AskRequest(question="Pytanie", mode="full")
    )
    assert llm.calls == [("Pytanie", 20, True)]


def test_debug_includes_each_retrieval_stage(settings):
    out = _service(settings, None).ask(AskRequest(question="Czy mogę przyjść z psem?", debug=True))
    assert out.debug is not None
    assert {"hybrid", "dense", "bm25", "query_lemmas"} <= set(out.debug)
    assert "pies" in out.debug["query_lemmas"]


def _two_providers(settings, default="anthropic"):
    claude = FakeLlm(response(text_block("Claude", citation(0, "Treść"))), provider="anthropic")
    gpt = FakeLlm(
        response(text_block("GPT", citation(1, "Treść")), model="gpt-test"), provider="openai"
    )
    hits = [make_hit(i) for i in range(5)]
    service = AskService(
        settings, FakeStore(hits), FakeEmbedder(), {"anthropic": claude, "openai": gpt}, default
    )
    return service, claude, gpt


def test_default_provider_is_used_when_request_has_none(settings):
    service, claude, gpt = _two_providers(settings, default="openai")
    out = service.ask(AskRequest(question="Pytanie?"))
    assert out.provider == "openai" and out.answer_text == "GPT"
    assert gpt.calls and not claude.calls


def test_request_can_choose_provider(settings):
    service, claude, gpt = _two_providers(settings)
    out = service.ask(AskRequest(question="Pytanie?", provider="openai"))
    assert out.provider == "openai" and out.model == "gpt-test"
    out = service.ask(AskRequest(question="Pytanie?", provider="anthropic"))
    assert out.provider == "anthropic" and out.answer_text == "Claude"
    assert len(claude.calls) == 1 and len(gpt.calls) == 1


def test_provider_without_key_returns_fragments_with_hint(settings):
    out = _service(settings, FakeLlm(response(text_block("x")))).ask(
        AskRequest(question="Pytanie?", provider="openai")
    )
    assert out.status == "fragments_only"
    assert "OPENAI_API_KEY" in out.note


def test_status_lists_both_providers(settings):
    service, _, _ = _two_providers(settings, default="openai")
    status = service.status()
    assert status["default_provider"] == "openai"
    assert [(p["name"], p["enabled"]) for p in status["providers"]] == [
        ("anthropic", True),
        ("openai", True),
    ]
    single = _service(settings, FakeLlm(response(text_block("x")))).status()
    openai_entry = next(p for p in single["providers"] if p["name"] == "openai")
    assert openai_entry == {
        "name": "openai",
        "label": "OpenAI",
        "enabled": False,
        "model": settings.openai_model,
        "key": "OPENAI_API_KEY",
    }
