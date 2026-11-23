# syntax=docker/dockerfile:1

# Etap 1: zależności (uv + uv.lock), bez kompilatorów w obrazie końcowym.
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.15.2-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Najpierw same zależności — warstwa cache'owana, dopóki nie zmieni się uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Etap 2: obraz uruchomieniowy.
FROM python:3.15.2-slim-trixie
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /models /app/knowledge /app/eval \
 && chown -R app:app /models /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    KB_KNOWLEDGE_DIR=/app/knowledge
WORKDIR /app
USER app
EXPOSE 8000
CMD ["uvicorn", "kb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
