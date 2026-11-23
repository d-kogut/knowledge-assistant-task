"""Wspólna konfiguracja logów."""

import logging
import os


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("KB_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
