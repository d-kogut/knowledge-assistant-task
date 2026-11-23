"""Testy jakości przez HTTP: `python -m kb.evaluation --api http://localhost:8000`.

Dla każdego wariantu (metoda wyszukiwania × dostawca modelu albo bez modelu) zadaje
pytania z eval/golden.yaml i liczy:
- wyszukiwanie: czy właściwa sekcja jest w top 1/3/10, MRR, pokrycie oczekiwanych sekcji,
- odpowiedź: kluczowe fakty, poprawna odmowa, cytowanie właściwej sekcji, czas, koszt.

Odpowiedzi są zapisywane w eval/results/run-*.json, więc metryki można przeliczyć
bez ponownych zapytań: `--rescore eval/results/run-....json`.
`--min hybrid.coverage=1` kończy się błędem, gdy wynik spadnie poniżej progu (CI).
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
import yaml

from kb.text import fold

# Cennik USD za 1 mln tokenów (11.2026): (wejście, wyjście, odczyt z cache, zapis do cache).
# Dopasowanie po najdłuższym prefiksie nazwy modelu (API zwraca też nazwy z datą).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.50),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
    "gpt-6.2-sol": (2.0, 10.0, 0.10, 0.0),
    "gpt-6.2-astra": (8.0, 40.0, 0.40, 0.0),
    "gpt-6.1-terra": (1.0, 5.0, 0.05, 0.0),
    "gpt-6-luna": (0.10, 0.50, 0.01, 0.0),
}

MODES = ("bm25", "dense", "hybrid", "full")
DEFAULT_VARIANTS = "bm25,dense,hybrid,hybrid+anthropic,hybrid+openai,full+anthropic,full+openai"


@dataclass(frozen=True)
class Variant:
    """`tryb` albo `tryb+dostawca[:model]`.

    Przykłady: hybrid, hybrid+openai, full+anthropic:claude-opus-5-5.
    """

    name: str
    mode: str
    provider: str | None = None
    model: str | None = None

    @property
    def llm(self) -> bool:
        return self.provider is not None

    @classmethod
    def parse(cls, name: str) -> Variant:
        mode, _, rest = name.partition("+")
        provider, _, model = rest.partition(":")
        if mode not in MODES or (rest and provider not in ("anthropic", "openai")):
            raise ValueError(f"Nieznany wariant: {name}")
        if mode == "full" and not provider:
            raise ValueError("Wariant full wymaga dostawcy, np. full+anthropic")
        return cls(name, mode, provider or None, model or None)


def normalize(text: str) -> str:
    text = fold(text.lower()).replace("\u00a0", " ")
    return re.sub(r"(?<=\d)[ ,.](?=\d{3}(?!\d))", "", text)  # 2 240 / 2,240 -> 2240


# Fakt musi stać jako osobne słowo albo liczba: „6” nie pasuje do „2026” ani „16”,
# „30” nie pasuje do „300” ani „30,5”. Fakt kończący się literą może mieć końcówkę
# odmiany („Kruk” pasuje do „Kruka”), bo polskie odpowiedzi odmieniają nazwy.
_LEFT = r"(?<!\w)(?<!\d[.,])"
_RIGHT_AFTER_DIGIT = r"(?!\d)(?![.,]\d)"


def _fact_pattern(alternative: str) -> re.Pattern[str]:
    norm = normalize(alternative.strip())
    right = _RIGHT_AFTER_DIGIT if norm[-1:].isdigit() else ""
    return re.compile(_LEFT + re.escape(norm) + right)


def fact_found(fact: str, text: str) -> bool:
    """Czy którakolwiek z alternatyw faktu (`a|b`) występuje w tekście."""
    norm = normalize(text)
    return any(_fact_pattern(alt).search(norm) for alt in fact.split("|") if alt.strip())


def _matches(item: dict[str, Any], expected: list[dict[str, str]]) -> bool:
    return any(
        item["file"] == e["file"] and item["section"].lower() == e["section"].lower()
        for e in expected
    )


@dataclass
class Result:
    qid: str
    tags: list[str]
    status: str
    expect: str
    rank: int | None = None
    coverage: float | None = None
    facts_total: int = 0
    facts_found: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)
    cited_expected: bool | None = None
    seconds: float = 0.0
    cost_usd: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    context_size: int | None = None
    answer: str = ""

    @property
    def refusal_ok(self) -> bool:
        return self.status == "not_found"

    @property
    def facts_ok(self) -> bool:
        return self.facts_total > 0 and not self.facts_missing and self.status != "not_found"


def cost(model: str | None, usage: dict[str, int] | None) -> float:
    if not usage or not model:
        return 0.0
    price = next(
        (PRICES[key] for key in sorted(PRICES, key=len, reverse=True) if model.startswith(key)),
        None,
    )
    if price is None:
        return 0.0
    p_in, p_out, p_cache_read, p_cache_write = price
    return (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + usage.get("cache_read_input_tokens", 0) * p_cache_read
        + usage.get("cache_creation_input_tokens", 0) * p_cache_write
    ) / 1_000_000


def evaluate(client: httpx2.Client, variant: Variant, q: dict[str, Any]) -> Result:
    expect = q.get("expect", "answered")
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "question": q["question"],
        "mode": variant.mode,
        "llm": variant.llm,
        "provider": variant.provider,
        "debug": True,
    }
    if variant.model:
        payload["model"] = variant.model
    resp = client.post("/ask", json=payload)
    resp.raise_for_status()
    data = resp.json()
    res = Result(
        qid=q["id"],
        tags=q.get("tags", []),
        status=data["status"],
        expect=expect,
        seconds=round(time.perf_counter() - started, 2),
        cost_usd=cost(data.get("model"), data.get("usage")),
        usage=data.get("usage") or {},
    )
    expected = q.get("sources", [])

    ranking = (data.get("debug") or {}).get(variant.mode)
    if isinstance(ranking, list):
        res.context_size = len(ranking)
    if expected and isinstance(ranking, list):
        res.rank = next((r["rank"] for r in ranking if _matches(r, expected)), None)
        found = {
            (e["file"], e["section"].lower()) for e in expected for r in ranking if _matches(r, [e])
        }
        res.coverage = len(found) / len(expected)

    if data["status"] == "answered":
        res.answer = "".join(s["text"] for s in data["segments"])
        cited = [
            {"file": s["file"], "section": s["breadcrumb"].split(" › ")[-1]}
            for s in data["sources"]
        ]
        res.cited_expected = any(_matches(c, expected) for c in cited) if expected else None
    elif data["status"] == "fragments_only":
        res.answer = "\n\n".join(f["body"] for f in data["fragments"])

    score_facts(res, q.get("facts", []))
    return res


def summarize(variant: Variant, results: list[Result]) -> dict[str, Any]:
    answerable = [r for r in results if r.expect == "answered"]
    ranked = [r for r in answerable if r.coverage is not None]

    def share(items: list[Any], pred: Any) -> float | None:
        return round(sum(1 for i in items if pred(i)) / len(items), 3) if items else None

    out: dict[str, Any] = {"variant": variant.name, "questions": len(results)}
    if ranked:
        out |= {
            "top1": share(ranked, lambda r: r.rank == 1),
            "top3": share(ranked, lambda r: r.rank is not None and r.rank <= 3),
            "top10": share(ranked, lambda r: r.rank is not None and r.rank <= 10),
            "mrr": round(statistics.mean(1 / r.rank if r.rank else 0 for r in ranked), 3),
            "coverage": round(statistics.mean(r.coverage or 0 for r in ranked), 3),
        }
    out["facts_all"] = share(answerable, lambda r: r.facts_ok)
    out["facts_avg"] = round(
        statistics.mean(len(r.facts_found) / r.facts_total for r in answerable if r.facts_total), 3
    )
    if variant.llm:
        unanswerable = [r for r in results if r.expect == "not_found"]
        out["refusal_ok"] = share(unanswerable, lambda r: r.refusal_ok)
        out["false_refusals"] = sum(1 for r in answerable if r.status == "not_found")
        out["cited_expected"] = share(answerable, lambda r: r.cited_expected is True)
        out["cost_usd"] = round(sum(r.cost_usd for r in results), 4)
        with_usage = [r for r in results if r.usage]
        if with_usage:
            out["avg_input_tokens"] = round(
                statistics.mean(
                    r.usage.get("input_tokens", 0) + r.usage.get("cache_read_input_tokens", 0)
                    for r in with_usage
                )
            )
            out["avg_output_tokens"] = round(
                statistics.mean(r.usage.get("output_tokens", 0) for r in with_usage)
            )
    sizes = [r.context_size for r in results if r.context_size]
    if sizes:
        out["context_size"] = max(sizes)
    out["avg_seconds"] = round(statistics.mean(r.seconds for r in results), 2)
    return out


def _tokens(s: dict[str, Any]) -> str:
    if "avg_input_tokens" not in s:
        return "-"
    return f"{s['avg_input_tokens']} / {s['avg_output_tokens']}"


def _context(s: dict[str, Any]) -> str:
    if s["variant"].startswith("full"):
        return "całość"
    return str(s.get("context_size", "-"))  # starsze przebiegi nie zapisywały rozmiaru


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def markdown(
    summaries: list[dict[str, Any]],
    details: dict[str, list[Result]],
    models: str,
    when: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    when = when or datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Wyniki ewaluacji ({when}, modele: {models})",
        "",
        *[f"{n}  " for n in notes],
        *([""] if notes else []),
        "| Wariant | Kontekst | Top 1 | Top 3 | Top 10 | MRR | Pokrycie sekcji | Wszystkie fakty "
        "| Śr. fakty | Odmowa poprawna | Fałszywe odmowy | Cytuje właściwą sekcję "
        "| Śr. czas | Tokeny we/wy | Koszt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['variant']} | {_context(s)} "
            f"| {_pct(s.get('top1'))} | {_pct(s.get('top3'))} "
            f"| {_pct(s.get('top10'))} | {s.get('mrr', '-')} | {_pct(s.get('coverage'))} "
            f"| {_pct(s.get('facts_all'))} | {_pct(s.get('facts_avg'))} "
            f"| {_pct(s.get('refusal_ok'))} | {s.get('false_refusals', '-')} "
            f"| {_pct(s.get('cited_expected'))} | {s['avg_seconds']} s "
            f"| {_tokens(s)} "
            f"| {'$' + format(s['cost_usd'], '.3f') if 'cost_usd' in s else '-'} |"
        )
    lines += [
        "",
        "Kontekst: liczba fragmentów przekazywanych do modelu (top_k). Top N / MRR: pozycja "
        "pierwszej oczekiwanej sekcji. Pokrycie: odsetek oczekiwanych sekcji w kontekście.",
        "Fakty: dla wariantów bez LLM sprawdzane w 3 pokazanych fragmentach; liczby i słowa "
        "dopasowywane jako całość (6 ≠ 2026).",
        "",
        "## Szczegóły (pytanie × wariant): status, fakty, pozycja sekcji",
        "",
        "| Pytanie | " + " | ".join(details) + " |",
        "|---|" + "---|" * len(details),
    ]
    by_q: dict[str, dict[str, Result]] = {}
    for name, results in details.items():
        for r in results:
            by_q.setdefault(r.qid, {})[name] = r
    for qid, per_variant in by_q.items():
        cells = []
        for name in details:
            r = per_variant[name]
            if r.expect == "not_found":
                cell = "odmowa ✓" if r.refusal_ok else f"{r.status} ✗"
            else:
                facts = f"{len(r.facts_found)}/{r.facts_total}"
                rank = f", #{r.rank}" if r.rank else (", poza top 10" if r.coverage == 0 else "")
                mark = "✓" if r.facts_ok else "✗"
                cell = f"{facts} {mark}{rank}" if r.status != "not_found" else "odmowa ✗"
            cells.append(cell)
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def score_facts(res: Result, facts: list[str]) -> None:
    res.facts_total = len(facts)
    res.facts_found, res.facts_missing = [], []
    for fact in facts:
        (res.facts_found if fact_found(fact, res.answer) else res.facts_missing).append(fact)


def rescore(
    run_files: Sequence[Path], golden: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[Result]], str, list[str]]:
    """Przelicza metryki z zapisanych odpowiedzi (bez zapytań do API).

    Przy kilku plikach wariant o tej samej nazwie bierzemy z pliku podanego później.
    """
    facts = {q["id"]: q.get("facts", []) for q in golden}
    known = {f.name for f in fields(Result)}
    details: dict[str, list[Result]] = {}
    origin: dict[str, str] = {}
    models: set[str] = set()
    for path in run_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("models") and raw["models"] != "brak":
            models.update(m.strip() for m in raw["models"].split(","))
        for name, rows in raw["details"].items():
            results = []
            for row in rows:
                res = Result(**{k: v for k, v in row.items() if k in known})
                score_facts(res, facts.get(res.qid, []))
                results.append(res)
            details[name] = results
            origin[name] = path.name
    summaries = [summarize(Variant.parse(name), results) for name, results in details.items()]
    by_file: dict[str, list[str]] = {}
    for name, file in origin.items():
        by_file.setdefault(file, []).append(name)
    notes = [
        f"Przeliczone {datetime.now().strftime('%Y-%m-%d %H:%M')} z zapisanych odpowiedzi "
        "(bez nowych zapytań).",
        *[f"Źródło: {file}: {', '.join(names)}" for file, names in by_file.items()],
    ]
    return summaries, details, ", ".join(sorted(models)) or "brak", notes


def check_minimums(summaries: list[dict[str, Any]], minimums: Sequence[str]) -> list[str]:
    """Progi w formacie `wariant.metryka=wartość`, np. `hybrid.coverage=1`."""
    by_name = {s["variant"]: s for s in summaries}
    failures = []
    for item in minimums:
        target, _, value = item.partition("=")
        variant, _, metric = target.rpartition(".")
        actual = by_name.get(variant, {}).get(metric)
        if actual is None or actual < float(value):
            failures.append(f"{target}: {actual} < {value}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--golden", type=Path, default=Path("eval/golden.yaml"))
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help="po przecinku: tryb albo tryb+dostawca[:model], np. hybrid+openai:gpt-6.2-astra",
    )
    parser.add_argument("--only", default=None, help="tylko pytania o tych id (po przecinku)")
    parser.add_argument(
        "--rescore",
        type=Path,
        action="append",
        help="przelicz metryki z zapisanego run-*.json zamiast pytać API (można podać kilka)",
    )
    parser.add_argument(
        "--summary", type=Path, default=None, help="plik podsumowania (domyślnie OUT/summary.md)"
    )
    parser.add_argument(
        "--min",
        action="append",
        default=[],
        metavar="WARIANT.METRYKA=WARTOŚĆ",
        help="próg, np. hybrid.coverage=1 (kod wyjścia 1, gdy niespełniony)",
    )
    args = parser.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    summary_path = args.summary or args.out / "summary.md"
    if args.rescore:
        summaries, details, models, notes = rescore(args.rescore, golden)
        _write_report(summary_path, markdown(summaries, details, models, notes=notes))
        _exit_on_failures(check_minimums(summaries, args.min))
        return

    if args.only:
        wanted = set(args.only.split(","))
        golden = [q for q in golden if q["id"] in wanted]
    client = httpx2.Client(base_url=args.api, timeout=180)
    status = client.get("/status").json()
    providers = {p["name"]: p for p in status.get("providers", [])}

    variants = []
    for name in args.variants.split(","):
        v = Variant.parse(name.strip())
        if v.provider and not providers.get(v.provider, {}).get("enabled"):
            key = providers.get(v.provider, {}).get("key", "klucza API")
            print(f"Pomijam {name}: API nie ma {key}", file=sys.stderr)
            continue
        variants.append(v)
    models = (
        ", ".join(
            sorted({v.model or providers[v.provider]["model"] for v in variants if v.provider})
        )
        or "brak"
    )

    details = {}
    summaries = []
    for v in variants:
        print(f"== {v.name} ({len(golden)} pytań)", file=sys.stderr)
        results = []
        for q in golden:
            r = evaluate(client, v, q)
            mark = "✓" if (r.refusal_ok if r.expect == "not_found" else r.facts_ok) else "✗"
            print(f"  {q['id']} {mark} {r.status} rank={r.rank} {r.seconds}s", file=sys.stderr)
            results.append(r)
        details[v.name] = results
        summaries.append(summarize(v, results))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw = {
        "models": models,
        "status": status,
        "summaries": summaries,
        "details": {k: [r.__dict__ for r in v] for k, v in details.items()},
    }
    (args.out / f"run-{stamp}.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(summary_path, markdown(summaries, details, models))
    _exit_on_failures(check_minimums(summaries, args.min))


def _write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)


def _exit_on_failures(failures: list[str]) -> None:
    if failures:
        print("Progi niespełnione:\n  " + "\n  ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
