.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up down logs status ask eval reset dev-sync test test-e2e lint

help: ## Lista poleceń
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-10s %s\n", $$1, $$2}'

.env:
	@printf '%s\n' \
		'# Klucz API modelu językowego: wystarczy jeden. Bez klucza serwis pokazuje same fragmenty.' \
		'# Pozostałe zmienne (opcjonalne): README, sekcja Konfiguracja.' \
		'ANTHROPIC_API_KEY=' \
		'OPENAI_API_KEY=' > .env
	@echo "Utworzono .env. Wpisz ANTHROPIC_API_KEY lub OPENAI_API_KEY (bez klucza: same fragmenty)."

up: .env ## Zbuduj i uruchom (czeka, aż wszystko będzie gotowe); strona: http://localhost:8000
	$(COMPOSE) up -d --build --wait --wait-timeout 1200
	@echo "Gotowe: http://localhost:8000"

down: ## Zatrzymaj (indeks i modele zostają w wolumenach)
	$(COMPOSE) down

logs: ## Logi ingest i api na żywo
	$(COMPOSE) logs -f ingest api

status: ## Stan indeksu
	@$(COMPOSE) exec -T api python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/status')), indent=2, ensure_ascii=False))"

ask: ## Pytanie z terminala: make ask Q="Czy mogę przyjść z psem?" [P=openai]
	@test -n "$(Q)" || (echo 'Użycie: make ask Q="pytanie" [P=anthropic|openai]' && exit 1)
	@$(COMPOSE) exec -T api python -m kb.ask "$(Q)" $(if $(P),--provider $(P))

eval: ## Testy jakości (wyniki w eval/results/)
	$(COMPOSE) --profile eval run --rm eval

reset: ## Usuń indeks i modele (wolumeny); następny "make up" odbuduje je z plików
	$(COMPOSE) down -v

dev-sync: ## Środowisko lokalne bez Dockera (uv)
	uv sync

test: ## Testy jednostkowe + z prawdziwym Qdrantem (tymczasowy kontener; bez Dockera pomijane)
	uv run pytest

test-e2e: ## Test działającego stacku (po make up): stan indeksu, aktualizacja plików na żywo
	KB_E2E_API_URL=http://localhost:8000 uv run pytest -m e2e -rA

lint: ## ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
