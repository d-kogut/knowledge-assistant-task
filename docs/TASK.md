# Zadanie rekrutacyjne — Knowledge Assistant

## Kontekst

W katalogu `knowledge/` znajduje się 15 plików z dokumentacją biznesową fikcyjnej firmy **SpaceHub** — sieci coworkingów w Krakowie. Dokumenty opisują procesy, cenniki, zespół, systemy IT, regulaminy i plany rozwoju.

## Zadanie

Zbuduj serwis (CLI, API lub prosty UI — dowolna forma), który **odpowiada na pytania użytkownika na podstawie tych dokumentów**.

Serwis powinien:
1. Wczytać dokumenty z katalogu `knowledge/`
2. Na pytanie użytkownika — znaleźć najrelewantniejsze fragmenty dokumentacji
3. Na podstawie znalezionych fragmentów — wygenerować odpowiedź

### Wymagania

- **Odpowiedzi muszą być oparte na dokumentacji**, nie na wiedzy ogólnej modelu AI
- Dowolny język programowania, framework, model AI
- Kod musi być czytelny i możliwy do uruchomienia po sklonowaniu repo
- Dołącz instrukcję uruchomienia

### Czego NIE wymagamy

- UI (wystarczy CLI lub endpoint API)
- Deployment (wystarczy że działa lokalnie)
- Obsługa wielu użytkowników jednocześnie

## Przykładowe pytania do przetestowania

Poniższe pytania pomogą Ci zweryfikować czy serwis działa poprawnie:

1. „Ile kosztuje Hot Desk Student i jakie ma ograniczenia?"
2. „Jakie są etapy windykacji gdy członek nie płaci?"
3. „Czy mogę przyjść z psem do lokalizacji na Krowodrzy?"
4. „Jakie sale konferencyjne mają wideokonferencję i ile kosztują?"
5. „Co się dzieje gdy zgubię kartę dostępu?"
6. „Jaka jest marża operacyjna SpaceHub i z czego się składają koszty?"
7. „Jakie są plany ekspansji na 2026-2027?"
8. „Ile osób pracuje w dziale operacji i kto zarządza klastrem Zabłocie?"
9. „Jak działa failover internetu i jaki jest SLA?"
10. „Czym różni się plan Virtual Basic od Virtual Premium?"

## Ocena

Na rozmowie po zadaniu porozmawiamy o:
- Jak działa Twoje rozwiązanie i dlaczego wybrałeś takie podejście?
- Co by się zmieniło gdyby dokumentów było 500 zamiast 15?
- Jakie widzisz ograniczenia swojego rozwiązania?

## Czas

Zadanie powinno zająć **3-6 godzin**. Możesz korzystać z dowolnych narzędzi AI (Copilot, ChatGPT, Claude, Codex, itp.) — jest to wręcz rekomendowane.

## Oddanie

Wyślij link do repozytorium (GitHub/GitLab) z rozwiązaniem.
