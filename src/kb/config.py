"""Konfiguracja z zmiennych środowiskowych (prefiks KB_) i pliku .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai"]
PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai")


def _has_value(secret: SecretStr | None) -> bool:
    return bool(secret and secret.get_secret_value().strip())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KB_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Dokumenty i indeks
    knowledge_dir: Path = Path("knowledge")
    qdrant_url: str = "http://localhost:6333"
    collection_alias: str = "kb"
    meta_collection: str = "kb_meta"

    # Model embeddingów (rewizja przypięta: ta sama wersja w ingest i api)
    embed_model: str = "OPI-PIB/PolDense-150M"
    embed_revision: str = "9539d6511f45f72ec900552b8a6a419814228e51"
    # Prefiks z karty modelu PolDense; bez niego jakość spada o ok. 30 pkt NDCG
    embed_query_prefix: str = "[query]: "
    embed_batch_size: int = 16

    # Dzielenie na fragmenty
    max_chunk_words: int = 350

    # Wyszukiwanie. 20 = najmniejsza wartość, przy której w ewaluacji wszystkie potrzebne
    # sekcje trafiają do kontekstu (przy 10 i 15: 18 z 21 pytań)
    top_k: int = 20
    prefetch_k: int = 30
    fragments_without_llm: int = 3

    # Modele językowe: wystarczy jeden klucz; z dwoma można wybierać dostawcę per pytanie.
    # auto = Anthropic, jeśli jest klucz, w przeciwnym razie OpenAI.
    llm_provider: Literal["auto", "anthropic", "openai"] = "auto"
    llm_timeout_s: float = 120.0

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "KB_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = Field(
        default="claude-sonnet-5-5",
        validation_alias=AliasChoices("KB_ANTHROPIC_MODEL", "KB_LLM_MODEL", "anthropic_model"),
    )
    anthropic_max_tokens: int = 1500

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "KB_OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-6.2-sol"
    # Modele GPT-6 rozumują przed odpowiedzią; tokeny rozumowania wliczają się do limitu wyjścia.
    openai_max_output_tokens: int = 8000
    openai_reasoning_effort: Literal["low", "medium", "high"] | None = None  # None = domyślny

    # Ingest
    reconcile_interval_s: int = 30
    debounce_ms: int = 800
    ready_file: Path = Path("/tmp/kb-ingest-ready")  # noqa: S108 - znacznik dla healthchecka

    def provider_enabled(self, provider: Provider) -> bool:
        key = self.anthropic_api_key if provider == "anthropic" else self.openai_api_key
        return _has_value(key)

    @property
    def enabled_providers(self) -> list[Provider]:
        return [p for p in PROVIDERS if self.provider_enabled(p)]

    @property
    def default_provider(self) -> Provider | None:
        enabled = self.enabled_providers
        if self.llm_provider != "auto" and self.llm_provider in enabled:
            return self.llm_provider
        return enabled[0] if enabled else None

    @property
    def llm_enabled(self) -> bool:
        return bool(self.enabled_providers)

    def model_for(self, provider: Provider) -> str:
        return self.anthropic_model if provider == "anthropic" else self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
