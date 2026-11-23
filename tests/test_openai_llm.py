from types import SimpleNamespace

import pytest
from conftest import make_hit

from kb.answer.openai_llm import (
    SYSTEM_PROMPT,
    AnswerSegment,
    Citation,
    OpenAIClient,
    StructuredAnswer,
    build_input,
    parse_response,
)

TABLE = "| Plan | Cena |\n|---|---|\n| Student | 149 PLN |\n| Basic | 299 PLN |"


def _hits():
    return [make_hit(0, body=TABLE), make_hit(1, body="Wymaga legitymacji.\nLimit: 100 miejsc.")]


def _resp(parsed, model="gpt-6.2-sol-2026-11-17", status="completed"):
    usage = SimpleNamespace(
        input_tokens=1000,
        input_tokens_details=SimpleNamespace(cached_tokens=200),
        output_tokens=300,
        output_tokens_details=SimpleNamespace(reasoning_tokens=120),
    )
    return SimpleNamespace(output_parsed=parsed, model=model, usage=usage, status=status)


def _seg(text, *cits):
    return AnswerSegment(
        text=text, citations=[Citation(fragment=f, lines=lines) for f, lines in cits]
    )


def test_input_numbers_fragments_and_lines():
    text = build_input("Ile?", _hits())
    assert '<fragment id="f1" plik="02-plans.md"' in text
    assert "L1: | Plan | Cena |\nL2: | Student | 149 PLN |" in text  # bez separatora tabeli
    assert '<fragment id="f2"' in text and "L2: Limit: 100 miejsc." in text
    assert text.endswith("Pytanie: Ile?")


def test_citations_resolve_to_verbatim_lines():
    parsed = StructuredAnswer(
        found=True,
        segments=[
            _seg("Hot Desk Student kosztuje 149 PLN.", ("f1", [2])),
            _seg("Wymaga legitymacji.", ("f2", [1]), ("f1", [2])),
        ],
    )
    out = parse_response(_resp(parsed), _hits())
    assert not out.not_found
    assert out.text == "Hot Desk Student kosztuje 149 PLN. Wymaga legitymacji."
    assert [s.result_indexes for s in out.segments] == [[0], [1, 0]]
    assert out.cited == {0: ["| Student | 149 PLN |"], 1: ["Wymaga legitymacji."]}


def test_invalid_fragment_ids_and_lines_are_dropped():
    parsed = StructuredAnswer(
        found=True,
        segments=[_seg("Coś.", ("f9", [1]), ("f1", [0, 99, 3]))],
    )
    out = parse_response(_resp(parsed), _hits())
    assert out.segments[0].result_indexes == [0]
    assert out.cited == {0: ["| Basic | 299 PLN |"]}


def test_segment_without_valid_citation_has_no_sources():
    parsed = StructuredAnswer(found=True, segments=[_seg("Paryż.", ("f7", [1]))])
    out = parse_response(_resp(parsed), _hits())
    assert out.segments[0].result_indexes == []  # serwis odrzuci odpowiedź bez cytatów


def test_found_false_means_not_found():
    out = parse_response(_resp(StructuredAnswer(found=False, segments=[])), _hits())
    assert out.not_found and out.segments == [] and out.cited == {}


def test_list_items_go_to_new_lines():
    parsed = StructuredAnswer(
        found=True,
        segments=[
            _seg("Etapy windykacji:", ("f2", [1])),
            _seg("- przypomnienie po 1 dniu,", ("f2", [1])),
            _seg("- blokada po 21 dniach.", ("f2", [2])),
            _seg("Dane ze stanu na marzec 2026.", ("f2", [2])),
        ],
    )
    out = parse_response(_resp(parsed), _hits())
    assert out.text == (
        "Etapy windykacji:\n- przypomnienie po 1 dniu,\n- blokada po 21 dniach. "
        "Dane ze stanu na marzec 2026."
    )


def test_usage_matches_anthropic_shape():
    out = parse_response(_resp(StructuredAnswer(found=False, segments=[])), _hits())
    assert out.usage == {
        "input_tokens": 800,
        "cache_read_input_tokens": 200,
        "output_tokens": 300,
        "reasoning_tokens": 120,
    }
    assert out.model.startswith("gpt-6.2-sol")


def test_refusal_or_truncated_output_raises():
    with pytest.raises(ValueError, match="incomplete"):
        parse_response(_resp(None, status="incomplete"), _hits())


class _FakeResponses:
    def __init__(self, resp):
        self.resp = resp
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.resp


def test_client_sends_structured_request():
    fake = _FakeResponses(_resp(StructuredAnswer(found=False, segments=[])))
    client = OpenAIClient("k", "gpt-6.2-sol", 8000, 30, client=SimpleNamespace(responses=fake))
    client.answer("Ile?", _hits())
    kw = fake.kwargs
    assert kw["model"] == "gpt-6.2-sol"
    assert kw["instructions"] == SYSTEM_PROMPT
    assert kw["text_format"] is StructuredAnswer
    assert kw["store"] is False and kw["max_output_tokens"] == 8000
    assert "reasoning" not in kw

    effort = OpenAIClient(
        "k", "gpt-6.2-sol", 8000, 30, "low", client=SimpleNamespace(responses=fake)
    )
    effort.answer("Ile?", _hits(), model="gpt-6.2-astra")
    assert fake.kwargs["reasoning"] == {"effort": "low"}
    assert fake.kwargs["model"] == "gpt-6.2-astra"


def test_schema_is_strict_compatible():
    """Structured Outputs (strict) wymaga: wszystkie pola wymagane, bez dodatkowych pól."""
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(StructuredAnswer)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"found", "segments"}
    seg = schema["$defs"]["AnswerSegment"]
    assert seg["additionalProperties"] is False and set(seg["required"]) == {"text", "citations"}
