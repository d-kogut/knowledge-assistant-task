"""Pytanie z terminala do działającego API: `python -m kb.ask "pytanie"` (make ask Q=...)."""

import argparse
import os
import sys
import textwrap

import httpx2


def main() -> None:
    parser = argparse.ArgumentParser(description="Zadaj pytanie Knowledge Assistantowi.")
    parser.add_argument("question")
    parser.add_argument("--api", default=os.environ.get("KB_API_URL", "http://localhost:8000"))
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25", "full"])
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--no-llm", action="store_true", help="same fragmenty, bez modelu")
    args = parser.parse_args()

    resp = httpx2.post(
        f"{args.api}/ask",
        json={
            "question": args.question,
            "mode": args.mode,
            "llm": not args.no_llm,
            "provider": args.provider,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        sys.exit(f"Błąd API {resp.status_code}: {resp.text}")
    data = resp.json()

    print()
    if data["status"] == "answered":
        text = "".join(
            s["text"] + "".join(f"[{n}]" for n in s.get("sources", [])) for s in data["segments"]
        )
        print(text.strip())
        print("\nŹródła:")
        for src in data["sources"]:
            where = f"{src['file']}, linie {src['line_start']}-{src['line_end']}"
            print(f"  [{src['n']}] {src['breadcrumb']}  ({where})")
            for quote in src["cited_texts"][:3]:
                print(textwrap.indent(textwrap.shorten(quote, 160), "        „") + "”")
    else:
        print(data.get("note", ""))
        for i, frag in enumerate(data.get("fragments", []), start=1):
            where = f"{frag['file']}, linie {frag['line_start']}-{frag['line_end']}"
            print(f"\n[{i}] {frag['breadcrumb']}  ({where})")
            print(textwrap.indent(frag["body"], "    "))
    timings = data.get("timings_ms", {})
    print(f"\n({data.get('model') or 'bez LLM'}, {timings.get('total', 0) / 1000:.1f} s)")


if __name__ == "__main__":
    main()
