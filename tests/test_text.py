from kb.text import fold, lemma_text, lemmas


def test_inflection_is_reduced_to_lemma():
    assert lemmas("psem") == lemmas("psy") == lemmas("pies") == ["pies"]
    assert lemmas("Krowodrzy") == lemmas("Krowodrza") == ["krowodrza"]
    assert lemmas("windykacji") == ["windykacja"]


def test_polish_characters_are_folded():
    assert fold("Zabłociu żółć") == "Zablociu zolc"
    assert lemmas("Zabłociu") == lemmas("zablocie")


def test_stopwords_are_removed():
    assert (
        lemma_text("Czy mogę przyjść z psem do lokalizacji na Krowodrzy?").split()[-1]
        == "krowodrza"
    )
    assert "czy" not in lemmas("Czy to jest w dokumentacji?")


def test_query_without_diacritics_matches_document_terms():
    doc = set(lemmas("Psy dozwolone w lokalizacjach Kazimierz i Zabłocie"))
    assert {"pies", "kazimierz"} <= doc & set(lemmas("czy moge przyjsc z psem na kazimierz"))
    assert {"pies", "zablocie"} <= doc & set(lemmas("Czy mogę przyjść z psem do Zabłocia?"))
    # Ograniczenie: forma odmieniona bez polskich znaków („zablocia”) nie jest w słowniku
    # lematyzatora — taki przypadek pokrywa wyszukiwanie wektorowe (PolDense).
    assert "zablocie" not in lemmas("do zablocia")
