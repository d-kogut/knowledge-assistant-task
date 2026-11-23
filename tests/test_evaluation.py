"""Metryki ewaluacji: na ich wynikach opierają się decyzje (np. top_k), więc też są testowane."""

import json

import httpx2
import pytest

from kb.evaluation.__main__ import (
    Result,
    Variant,
    check_minimums,
    cost,
    evaluate,
    fact_found,
    normalize,
    rescore,
    summarize,
)


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("6", "W SpaceHub pracuje 6 Community Managerów."),
        ("6", "Community Managerów: (6)."),
        ("2240", "Cena po rabacie: 2 240 PLN."),
        ("2240", "Cena po rabacie: 2,240 PLN."),
        ("9.7|9,7", "Marża wynosi 9,7%."),
        ("Kruk", "Klastrem zarządza Tomasz Kruka."),  # końcówka odmiany
        ("Zabłocie", "klaster ZABLOCIE"),  # bez polskich znaków, wielkość liter
        ("30 s|30s|30 sekund", "przełączenie trwa do 30 sekund"),
        ("6:00|6.00", "dostęp od 6:00 do 23:00"),
    ],
)
def test_fact_found(fact, text):
    assert fact_found(fact, text)


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("6", "Stan na marzec 2026."),  # dawniej zaliczane: „6” w „2026”
        ("6", "16 osób w zespole"),
        ("6", "6,5 etatu"),  # inna liczba
        ("30", "300 PLN"),
        ("18", "od 2018 roku"),
        ("149", "1149 PLN"),
        ("PIN", "spinacz"),  # fakt musi zaczynać słowo
        ("6:00", "16:00"),
    ],
)
def test_fact_not_found(fact, text):
    assert not fact_found(fact, text)


def test_normalize_removes_diacritics_and_thousand_separators():
    assert normalize("Zabłocie: 12 500 zł") == "zablocie: 12500 zl"
    assert normalize("99,9%") == "99,9%"  # ułamek to nie separator tysięcy


def test_variant_parse():
    v = Variant.parse("hybrid+openai:gpt-6.2-astra")
    assert (v.mode, v.provider, v.model, v.llm) == ("hybrid", "openai", "gpt-6.2-astra", True)
    assert not Variant.parse("bm25").llm
    with pytest.raises(ValueError, match="Nieznany"):
        Variant.parse("hybrid+mistral")
    with pytest.raises(ValueError, match="full wymaga"):
        Variant.parse("full")


def test_cost_uses_longest_model_prefix_and_cache_prices():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0}
    assert cost("claude-sonnet-5-5-20261001", usage) == pytest.approx(2.0 + 1.0)
    cached = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000}
    assert cost("claude-sonnet-5-5", cached) == pytest.approx(0.10)
    assert cost("nieznany-model", usage) == 0.0
    assert cost(None, usage) == 0.0


def _res(qid, status, expect="answered", rank=None, facts_ok=True, **kw):
    return Result(
        qid=qid,
        tags=[],
        status=status,
        expect=expect,
        rank=rank,
        coverage=1.0 if rank else 0.0,
        facts_total=1,
        facts_found=["x"] if facts_ok else [],
        facts_missing=[] if facts_ok else ["x"],
        **kw,
    )


def test_summarize_llm_variant():
    results = [
        _res("a", "answered", rank=1, cited_expected=True),
        _res("b", "answered", rank=4, facts_ok=False, cited_expected=False),
        _res("c", "not_found", rank=2),  # fałszywa odmowa
        _res("o", "not_found", expect="not_found"),
    ]
    s = summarize(Variant.parse("hybrid+anthropic"), results)
    assert s["top1"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["top3"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["mrr"] == pytest.approx((1 + 1 / 4 + 1 / 2) / 3, abs=1e-3)
    assert s["facts_all"] == pytest.approx(1 / 3, abs=1e-3)  # odmowa nie zalicza faktów
    assert s["false_refusals"] == 1
    assert s["refusal_ok"] == 1.0
    assert s["cited_expected"] == pytest.approx(1 / 3, abs=1e-3)


def _client(payload):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    return httpx2.Client(base_url="http://api", transport=httpx2.MockTransport(handler))


QUESTION = {
    "id": "r08",
    "question": "Ile osób pracuje w dziale operacji i kto zarządza klastrem Zabłocie?",
    "sources": [
        {"file": "09-team.md", "section": "Operacje (18 osób)"},
        {"file": "09-team.md", "section": "Klaster Zabłocie"},
    ],
    "facts": ["18", "Kruk"],
}


def test_evaluate_counts_rank_coverage_facts_and_citations():
    payload = {
        "status": "answered",
        "segments": [{"text": "W operacjach pracuje 18 osób, klastrem zarządza Ewa Kruk."}],
        "sources": [{"file": "09-team.md", "breadcrumb": "Zespół › Operacje (18 osób)"}],
        "model": "claude-sonnet-5-5",
        "usage": {"input_tokens": 1000, "output_tokens": 100},
        "debug": {
            "hybrid": [
                {"rank": 1, "file": "01-about.md", "section": "O firmie"},
                {"rank": 2, "file": "09-team.md", "section": "Operacje (18 osób)"},
                {"rank": 3, "file": "08-x.md", "section": "Inna"},
            ]
        },
    }
    res = evaluate(_client(payload), Variant.parse("hybrid+anthropic"), QUESTION)
    assert res.rank == 2
    assert res.coverage == 0.5  # druga oczekiwana sekcja poza kontekstem
    assert res.context_size == 3
    assert res.facts_ok and res.cited_expected
    assert res.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1_000_000)


def test_rescore_recomputes_facts_from_saved_answers(tmp_path):
    # Zapis ze starą (zbyt luźną) metryką: „6” zaliczone, bo w odpowiedzi jest „2026”.
    row = _res("c01", "answered", rank=1).__dict__ | {
        "answer": "Stan na marzec 2026: zespół liczy 8 Community Managerów.",
        "facts_found": ["6"],
        "facts_missing": [],
    }
    run = tmp_path / "run-1.json"
    run.write_text(
        json.dumps({"models": "claude-sonnet-5-5", "details": {"hybrid+anthropic": [row]}})
    )
    golden = [{"id": "c01", "facts": ["6"]}]
    summaries, details, models, notes = rescore([run], golden)
    assert details["hybrid+anthropic"][0].facts_missing == ["6"]
    assert summaries[0]["facts_all"] == 0.0
    assert models == "claude-sonnet-5-5"
    assert any("run-1.json" in n for n in notes)


def test_rescore_later_file_overrides_same_variant(tmp_path):
    golden = [{"id": "q", "facts": ["7"]}]
    for i, answer in enumerate(["brak", "jest 7"]):
        row = _res("q", "answered", rank=1).__dict__ | {"answer": answer}
        (tmp_path / f"run-{i}.json").write_text(json.dumps({"details": {"hybrid": [row]}}))
    summaries, *_ = rescore([tmp_path / "run-0.json", tmp_path / "run-1.json"], golden)
    assert summaries[0]["facts_all"] == 1.0


def test_check_minimums():
    summaries = [{"variant": "hybrid", "coverage": 1.0, "top10": 0.95}]
    assert check_minimums(summaries, ["hybrid.coverage=1"]) == []
    assert check_minimums(summaries, ["hybrid.top10=1"]) == ["hybrid.top10: 0.95 < 1"]
    assert check_minimums(summaries, ["dense.top10=0.9"]) == ["dense.top10: None < 0.9"]
