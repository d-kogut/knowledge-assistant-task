"""API i strona: POST /ask, GET /status, GET /health, strona pod /."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kb import __doc__ as _pkg_doc
from kb.answer.llm import LlmClient
from kb.answer.schemas import AskRequest, AskResponse
from kb.answer.service import AskService
from kb.logsetup import setup_logging

if TYPE_CHECKING:
    from kb.config import Settings

logger = logging.getLogger("kb.api")
STATIC_DIR = Path(__file__).parent / "static"

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def build_llms(settings: Settings) -> dict[str, LlmClient]:
    """Klienci modeli dla dostawców, którzy mają klucz API."""
    from kb.answer.anthropic_llm import AnthropicClient
    from kb.answer.openai_llm import OpenAIClient

    llms: dict[str, LlmClient] = {}
    if settings.provider_enabled("anthropic") and settings.anthropic_api_key:
        llms["anthropic"] = AnthropicClient(
            settings.anthropic_api_key.get_secret_value().strip(),
            settings.anthropic_model,
            settings.anthropic_max_tokens,
            settings.llm_timeout_s,
        )
    if settings.provider_enabled("openai") and settings.openai_api_key:
        llms["openai"] = OpenAIClient(
            settings.openai_api_key.get_secret_value().strip(),
            settings.openai_model,
            settings.openai_max_output_tokens,
            settings.llm_timeout_s,
            settings.openai_reasoning_effort,
        )
    return llms


def build_service() -> AskService:
    """Składa serwis z prawdziwych zależności (Qdrant, PolDense, Claude / OpenAI)."""
    from kb.config import get_settings
    from kb.embedder import SentenceTransformerEmbedder
    from kb.store import QdrantStore
    from kb.text import lemma_text

    settings = get_settings()
    store = QdrantStore(settings.qdrant_url, settings.collection_alias, settings.meta_collection)
    store.wait_until_ready(timeout_s=300)
    embedder = SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_revision,
        settings.embed_query_prefix,
        settings.embed_batch_size,
    )
    embedder.embed_query("rozgrzewka")  # pierwsze wywołanie modelu jest wolne
    lemma_text("rozgrzewka")  # słownik lematyzatora ładuje się przy pierwszym użyciu (~1,8 s)

    llms = build_llms(settings)
    if llms:
        logger.info(
            "Modele językowe: %s (domyślnie: %s)",
            ", ".join(f"{name}={llm.default_model}" for name, llm in llms.items()),
            settings.default_provider,
        )
        if settings.llm_provider != "auto" and settings.llm_provider not in llms:
            logger.warning(
                "KB_LLM_PROVIDER=%s, ale brak klucza — domyślnie: %s",
                settings.llm_provider,
                settings.default_provider,
            )
    else:
        logger.warning(
            "Brak ANTHROPIC_API_KEY i OPENAI_API_KEY — tryb bez modelu językowego (same fragmenty)"
        )
    return AskService(settings, store, embedder, llms, settings.default_provider)


def create_app(service_factory: Callable[[], AskService] = build_service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        app.state.service = service_factory()
        yield

    app = FastAPI(
        title="Knowledge Assistant — SpaceHub",
        description=(_pkg_doc or "").strip(),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        # /docs ładuje Swagger UI z CDN, więc ścisłe CSP tylko dla strony i jej plików.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers.update(_SECURITY_HEADERS)
        return response

    def service(request: Request) -> AskService:
        svc: AskService = request.app.state.service
        return svc

    @app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """Odpowiedź na pytanie na podstawie dokumentacji, ze źródłami i cytatami."""
        try:
            return service(request).ask(body)
        except Exception as exc:
            logger.exception("Błąd obsługi pytania")
            raise HTTPException(503, f"Wyszukiwanie niedostępne: {type(exc).__name__}") from exc

    @app.get("/status")
    def status(request: Request) -> dict[str, Any]:
        """Stan indeksu: liczba plików i fragmentów, ostatnia zmiana, błędy plików."""
        return service(request).status()

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        ok, reason = service(request).health()
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "detail": reason}, 200 if ok else 503
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
