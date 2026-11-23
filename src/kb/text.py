"""Normalizacja tekstu dla wyszukiwania słownikowego (BM25).

Polska odmiana sprawia, że "psem", "psy" i "pies" to dla BM25 różne słowa. Sprowadzamy
każde słowo do formy podstawowej (lemat, simplemma), usuwamy polskie znaki (pytania
bywają pisane bez ogonków) i odrzucamy słowa funkcyjne. Ta sama funkcja przetwarza
dokumenty i pytania, więc obie strony są porównywalne.
"""

import re
import unicodedata
from functools import lru_cache

import simplemma

_TOKEN = re.compile(r"[^\W_]+(?:[.,][^\W_]+)*")
_FOLD = str.maketrans({"ł": "l", "Ł": "L"})

# Słowa funkcyjne po lematyzacji i usunięciu polskich znaków.
STOPWORDS = frozenset(
    """
    a aby albo ale ani az bo by byc chyba co czy czyli dla do gdy gdyz gdzie i ich ile im
    ja jak jaki jakis jako je jego jej jesli jezeli juz kiedy ktory kto lub mi mnie moj my
    na nad nie niz no o od oraz po pod przez przy sie siebie sobie ta tak takze te ten tez
    to tu ty u w we wiec wszystko z za ze zas
    """.split()  # noqa: SIM905 - lista słów czytelniejsza jako tekst
)


def fold(text: str) -> str:
    """Usuwa polskie znaki: 'Zabłociu' -> 'Zablociu'."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLD))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=65536)
def _lemma(token: str) -> str:
    return fold(simplemma.lemmatize(token, lang="pl").lower())


def lemmas(text: str) -> list[str]:
    """Lista lematów bez polskich znaków i słów funkcyjnych."""
    out = []
    for token in _TOKEN.findall(text):
        lemma = _lemma(token)
        if lemma and lemma not in STOPWORDS:
            out.append(lemma)
    return out


def lemma_text(text: str) -> str:
    return " ".join(lemmas(text))
