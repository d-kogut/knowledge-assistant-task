"""Model embeddingów (PolDense) — wspólny dla ingest i api."""

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lokalny model przez sentence-transformers, na CPU.

    Dokumenty kodujemy bez prefiksu, pytania z prefiksem z karty modelu (`[query]: `).
    """

    def __init__(self, model_name: str, revision: str, query_prefix: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer  # ciężki import tylko tu

        started = time.perf_counter()
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, revision=revision, device="cpu")
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if not dim:
            raise RuntimeError(f"Model {model_name} nie podaje wymiaru wektora")
        self.dim = int(dim)
        logger.info(
            "Model embeddingów %s (%s) załadowany w %.1f s, wymiar %d",
            model_name,
            revision[:8],
            time.perf_counter() - started,
            self.dim,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self.query_prefix + text, normalize_embeddings=True, show_progress_bar=False
        )
        return list(map(float, vector))
