# Knowledge Assistant: SpaceHub

Serwis odpowiada po polsku na pytania o firmę SpaceHub wyłącznie na podstawie dokumentów
z `knowledge/`. Każda odpowiedź podaje źródła (plik, sekcja, linie) i dosłowne cytaty.
Jeśli odpowiedzi w dokumentach nie ma, serwis odpowiada „Nie ma tego w dokumentacji”.

Treść zadania: [docs/TASK.md](docs/TASK.md).

## Uruchomienie

Potrzebny jest Docker z Compose, ok. 3 GB wolnego dysku i ok. 3 GB RAM dla Dockera.
Działa na arm64 (Mac) i amd64.

```bash
git clone <repo> && cd knowledge-assistant-task
echo 'ANTHROPIC_API_KEY=twój-klucz' > .env # albo OPENAI_API_KEY; bez klucza pomiń
make up # docker compose up -d --build --wait
```

Strona: http://localhost:8000. API z formularzem: http://localhost:8000/docs.

Pierwszy start trwa kilka minut (budowa obrazu, pobranie modelu wyszukiwania 0,6 GB,
indeksowanie ok. 20 s). Kolejne starty zajmują kilka sekund.

Wystarczy jeden klucz: Claude (Anthropic) albo OpenAI. Z dwoma kluczami model wybiera się
przy każdym pytaniu. Bez klucza serwis zamiast odpowiedzi pokazuje 3 najlepiej pasujące
fragmenty.

## Jak to działa

1. Przygotowanie. Program dzieli 15 plików na 139 fragmentów (jeden fragment to jedna
   sekcja z nagłówkiem) i zapisuje je w bazie Qdrant. Dla każdego fragmentu zapisuje wektor
   znaczenia z lokalnego modelu PolDense i słowa w formie podstawowej (`psem` → `pies`).
2. Wyszukiwanie. Pytanie jest porównywane z fragmentami na dwa sposoby: po znaczeniu
   („zalegam z opłatami” trafia na windykację) i po słowach (nazwy i liczby, np. „Virtual
   Premium”). Oba rankingi są łączone, a do modelu trafia 20 najlepszych fragmentów.
   Ten krok działa lokalnie, trwa ok. 0,1 s i nic nie kosztuje.
3. Odpowiedź. Model językowy (Claude albo OpenAI) dostaje pytanie i te 20 fragmentów
   z poleceniem, żeby odpowiadał tylko na ich podstawie. Trwa to ok. 5 s i kosztuje ok.
   2,7 centa.
4. Sprawdzenie. Każde zdanie musi wskazywać fragment źródłowy. Tekst cytatu pochodzi
   dosłownie z dokumentu (wycina go API Claude'a albo nasz kod), więc nie da się go
   zmyślić. Odpowiedź bez cytatu jest odrzucana.
5. Aktualizacja. Dodany, zmieniony lub usunięty plik w `knowledge/` jest widoczny
   w odpowiedziach po mniej niż sekundzie.

## Dlaczego tak

- Najpierw wyszukiwanie, potem odpowiedź, bo tego wymaga zadanie i bo tylko tak da się
  obsłużyć dużą dokumentację. Cała dokumentacja w prompcie to dziś 61 tys. tokenów na
  pytanie, przy 500 plikach byłoby to ok. 2 mln.
- Dwa sposoby wyszukiwania naraz. Same słowa gubią parafrazy, samo znaczenie gubi liczby
  i nazwy. Razem wszystkie potrzebne sekcje trafiają do modelu w 21 z 21 pytań (osobno
  85% i 98% sekcji).
- Formy podstawowe słów, bo polska odmiana psuje zwykłe dopasowanie. Właściwa sekcja była
  w top 10 w 14 z 15 pytań, bez tego w 8.
- Lokalny model wyszukiwania nie wymaga drugiego klucza, a indeksowanie nic nie kosztuje.
- Qdrant i osobna usługa wczytywania pozwalają aktualizować indeks na żywo i przebudować
  go bez przerwy w działaniu.
- Bez frameworka RAG (LangChain itp.). Niezależny benchmark z 10.2026 pokazał tę samą
  jakość z frameworkiem i bez niego, a kodu jest mniej.

## Wyniki

`make eval` zadaje 24 pytania z [eval/golden.yaml](eval/golden.yaml): 10 z zadania,
parafrazy, pytania bez polskich znaków i z literówką, pytania o sprzeczne dane, obliczenia
oraz 3 pytania spoza dokumentacji, przy których serwis powinien odmówić. Fakt liczy się
tylko jako całe słowo lub liczba („6” nie zalicza się w „2026”).

Claude Sonnet 5.5, 23.11.2026:

| Wariant | Model czyta | Wszystkie fakty | Poprawne odmowy | Fałszywe odmowy | Cytuje właściwą sekcję | Czas | Koszt pytania |
|---|---|---|---|---|---|---|---|
| wyszukiwanie + Claude (domyślnie) | 20 fragmentów | 100% | 3/3 | 0 | 100% | 5,0 s | 2,7 ¢ |
| wyszukiwanie + Claude | 10 fragmentów | 90% | 3/3 | 1 | 95% | 4,4 s | 1,8 ¢ |
| cała dokumentacja + Claude | 139 fragmentów | 100% | 3/3 | 0 | 100% | 4,6 s | 1,8 ¢ (z cache) |
| samo wyszukiwanie, bez modelu | 3 fragmenty | 76% | - | - | - | 0,07 s | 0 |

Samo wyszukiwanie, 21 pytań z odpowiedzią w dokumentach:

| Metoda | Sekcja na 1. miejscu | w top 3 | w top 10 | Potrzebne sekcje w 20 fragmentach |
|---|---|---|---|---|
| po słowach (BM25) | 76% | 90% | 95% | 85% |
| po znaczeniu (PolDense) | 71% | 95% | 100% | 98% |
| oba razem | 76% | 90% | 100% | 100% |

Wnioski:

- Przy 10 fragmentach błędy wynikały z wyszukiwania: potrzebne sekcje były na pozycjach
  19 i 20. Model nie zmyślał, tylko pisał, czego brakuje. Dlatego model dostaje 20
  fragmentów.
- Model językowy podnosi wynik z 76% do 100%, odmawia przy pytaniach spoza dokumentacji,
  liczy (3 200 × 0,7 = 2 240 PLN) i pokazuje sprzeczności.
- Przy 15 plikach cała dokumentacja w prompcie działa równie dobrze i z cache jest tańsza
  (bez cache ok. 13 ¢ za pytanie). Wyszukiwanie zostaje domyślne, bo tylko ono się skaluje.

Pełne tabele `make eval` zapisuje w `eval/results/`. OpenAI nie zostało jeszcze zmierzone;
po wpisaniu `OPENAI_API_KEY` wystarczy `make eval`.

Pomiary na MacBooku M3 Pro (CPU): indeksowanie 139 fragmentów 19 s, zmiana pliku widoczna
po 0,3-0,7 s, wyszukiwanie 0,1 s, odpowiedź Claude'a 5 s (ok. 10,7 tys. tokenów wejścia,
590 wyjścia). RAM: api 0,9 GB, ingest 1,0 GB, Qdrant 0,1 GB. Obraz aplikacji ok. 2 GB.

## Użycie

- Strona http://localhost:8000: pytanie, wybór modelu, 10 przykładowych pytań, numery
  źródeł przy zdaniach (kliknięcie pokazuje cytat), podgląd rankingów wyszukiwania.
- Terminal: `make ask Q="Czy mogę przyjść z psem na Krowodrzę?"`, opcjonalnie `P=openai`.
- API: `POST /ask`, np.
  `curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Ile kosztuje Hot Desk Student?"}'`.
  Odpowiedź zawiera `status` (`answered`, `not_found` albo `fragments_only`), zdania
  z numerami źródeł (`segments`), źródła z cytatami (`sources`), model, zużycie tokenów
  i czasy. Parametry opcjonalne: `provider`, `model`, `llm` (false = same fragmenty),
  `top_k`, `debug` (rankingi wyszukiwania), `mode` (`hybrid`, `dense`, `bm25`, `full`).
- `GET /status`: stan indeksu i błędy plików. `GET /health`: gotowość.
- Podgląd bazy: http://localhost:6333/dashboard.

## Testy

| Polecenie | Co sprawdza |
|---|---|
| `make test` | 82 testy jednostkowe i 8 testów z prawdziwym Qdrantem w tymczasowym kontenerze (bez Dockera te 8 jest pomijanych) |
| `make test-e2e` | działający serwis po `make up`: stan indeksu, pytanie bez modelu, aktualizacja pliku na żywo |
| `make eval` | jakość odpowiedzi (sekcja Wyniki) |
| `make lint` | ruff i mypy (strict) |

Pokrycie kodu przez `make test`: 78%. CI (GitHub Actions) uruchamia lint i testy, a potem
cały serwis bez klucza API: ewaluację wyszukiwania z progami i test aktualizacji na żywo.

## Konfiguracja

Wszystkie zmienne w `.env` są opcjonalne.

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | brak | klucze; bez obu serwis pokazuje same fragmenty |
| `KB_LLM_PROVIDER` | `auto` | domyślny dostawca: `auto` (Anthropic, jeśli jest klucz), `anthropic`, `openai` |
| `KB_ANTHROPIC_MODEL` | `claude-sonnet-5-5` | model Claude |
| `KB_OPENAI_MODEL` | `gpt-6.2-sol` | model OpenAI |
| `KB_OPENAI_REASONING_EFFORT` | domyślny modelu | `low`, `medium` albo `high` |
| `KB_TOP_K` | `20` | ile fragmentów dostaje model |
| `KB_PREFETCH_K` | `30` | kandydaci z każdej metody wyszukiwania przed połączeniem |
| `KB_EMBED_MODEL`, `KB_EMBED_REVISION` | PolDense-150M | zmiana buduje nowy indeks bez przerwy w działaniu |
| `KB_MAX_CHUNK_WORDS` | `350` | dłuższe sekcje są dzielone |
| `KB_RECONCILE_INTERVAL_S` | `30` | co ile sekund pełne porównanie plików |

## Praca bez Dockera

```bash
uv sync
docker compose up -d qdrant
uv run python -m kb.ingest
uv run uvicorn kb.api.app:app --reload
```

Dwa ostatnie polecenia uruchom w osobnych terminalach. Projekt wymaga Pythona 3.15
(uv pobierze go sam).

## Struktura

- `src/kb/ingest/`: podział plików na fragmenty i synchronizacja z bazą
- `src/kb/answer/`: wyszukiwanie i odpowiedź (Claude, OpenAI)
- `src/kb/api/`: API i strona (HTML, JS, CSS bez budowania)
- `src/kb/evaluation/`: ewaluacja jakości
- `src/kb/store.py`, `embedder.py`, `text.py`, `config.py`: baza, model wyszukiwania,
  formy podstawowe słów, konfiguracja
- `eval/`: pytania testowe i wyniki
- `tests/`: testy jednostkowe, z Qdrantem i całego serwisu
- `docs/`: treść zadania

## Licencje

Biblioteki: MIT, BSD albo Apache 2.0. Qdrant: Apache 2.0. Model PolDense: Gemma Terms of Use
(darmowy, także komercyjnie, z listą zakazanych zastosowań).
