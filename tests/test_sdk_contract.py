"""Kontrakt z prawdziwymi SDK (anthropic, openai) bez sieci i bez kluczy.

Transport HTTP jest podmieniony: zapisuje wysłane żądanie i zwraca odpowiedź w formacie
z dokumentacji API. Sprawdza serializację zapytania (bloki search_result, Structured Outputs)
i odczyt odpowiedzi przez typy SDK — czyli to, czego atrapy obiektów nie pokrywają.
"""

import json

import anthropic
import httpx2
import openai
from conftest import make_hit

from kb.answer.anthropic_llm import AnthropicClient
from kb.answer.openai_llm import OpenAIClient, StructuredAnswer

HITS = [
    make_hit(0, body="| Plan | Cena |\n|---|---|\n| Hot Desk Student | 149 PLN |"),
    make_hit(1, body="Wymaga ważnej legitymacji studenckiej."),
]


def _transport(payload: dict, sent: list[dict]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx2.Response(200, json=payload)

    return httpx2.MockTransport(handler)


def test_anthropic_sdk_request_and_citations():
    sent: list[dict] = []
    payload = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {
                "type": "text",
                "text": "Hot Desk Student kosztuje 149 PLN",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "| Hot Desk Student | 149 PLN |",
                        "search_result_index": 0,
                        "source": HITS[0].chunk_id,
                        "title": "t",
                        "start_block_index": 1,
                        "end_block_index": 2,
                    }
                ],
            },
            {"type": "text", "text": "."},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1200, "output_tokens": 40, "cache_read_input_tokens": 0},
    }
    sdk = anthropic.Anthropic(
        api_key="test", http_client=httpx2.Client(transport=_transport(payload, sent))
    )
    answer = AnthropicClient("test", "claude-sonnet-5-5", 1500, 10, client=sdk).answer(
        "Ile kosztuje Hot Desk Student?", HITS
    )

    body = sent[0]["body"]
    assert sent[0]["path"] == "/v1/messages"
    first = body["messages"][0]["content"][0]
    assert first["type"] == "search_result" and first["citations"] == {"enabled": True}
    assert [b["text"] for b in first["content"]] == [
        "| Plan | Cena |",
        "| Hot Desk Student | 149 PLN |",
    ]
    assert "output_config" not in body  # cytaty nie łączą się z wymuszonym JSON-em

    assert answer.text == "Hot Desk Student kosztuje 149 PLN."
    assert answer.segments[0].result_indexes == [0]
    assert answer.cited == {0: ["| Hot Desk Student | 149 PLN |"]}
    assert answer.usage["input_tokens"] == 1200


def test_openai_sdk_structured_request_and_parsing():
    sent: list[dict] = []
    structured = {
        "found": True,
        "segments": [
            {
                "text": "Hot Desk Student kosztuje 149 PLN.",
                "citations": [{"fragment": "f1", "lines": [2]}],
            },
            {
                "text": "Wymaga ważnej legitymacji.",
                "citations": [{"fragment": "f2", "lines": [1]}],
            },
        ],
    }
    payload = {
        "id": "resp_01",
        "object": "response",
        "created_at": 1763900000,
        "status": "completed",
        "model": "gpt-6.2-sol-2026-11-17",
        "output": [
            {"type": "reasoning", "id": "rs_01", "summary": []},
            {
                "type": "message",
                "id": "msg_01",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": json.dumps(structured), "annotations": []}
                ],
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 220,
            "output_tokens_details": {"reasoning_tokens": 150},
            "total_tokens": 1720,
        },
    }
    sdk = openai.OpenAI(
        api_key="test", http_client=httpx2.Client(transport=_transport(payload, sent))
    )
    answer = OpenAIClient("test", "gpt-6.2-sol", 8000, 10, client=sdk).answer(
        "Ile kosztuje Hot Desk Student?", HITS
    )

    body = sent[0]["body"]
    assert sent[0]["path"] == "/v1/responses"
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["required"]) == {"found", "segments"}
    assert body["store"] is False and body["max_output_tokens"] == 8000
    assert '<fragment id="f1"' in body["input"] and body["instructions"]

    assert isinstance(answer, object) and not answer.not_found
    assert answer.text == "Hot Desk Student kosztuje 149 PLN. Wymaga ważnej legitymacji."
    assert answer.cited == {0: ["| Hot Desk Student | 149 PLN |"], 1: [HITS[1].body]}
    assert answer.usage["reasoning_tokens"] == 150
    assert StructuredAnswer.model_validate(structured).found
