from kb.config import Settings


def _settings(monkeypatch, **env):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "KB_LLM_PROVIDER", "KB_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_no_keys_means_no_llm(monkeypatch):
    s = _settings(monkeypatch)
    assert not s.llm_enabled and s.default_provider is None


def test_auto_prefers_anthropic_then_openai(monkeypatch):
    assert _settings(monkeypatch, OPENAI_API_KEY="sk-o").default_provider == "openai"
    both = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a", OPENAI_API_KEY="sk-o")
    assert both.default_provider == "anthropic"
    assert both.enabled_providers == ["anthropic", "openai"]


def test_explicit_provider_wins_when_key_present(monkeypatch):
    s = _settings(
        monkeypatch, ANTHROPIC_API_KEY="sk-a", OPENAI_API_KEY="sk-o", KB_LLM_PROVIDER="openai"
    )
    assert s.default_provider == "openai"


def test_explicit_provider_without_key_falls_back(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="sk-a", KB_LLM_PROVIDER="openai")
    assert s.default_provider == "anthropic"


def test_blank_key_is_ignored(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="  ", OPENAI_API_KEY="")
    assert not s.llm_enabled


def test_old_model_variable_still_works(monkeypatch):
    s = _settings(monkeypatch, KB_LLM_MODEL="claude-opus-5-5")
    assert s.anthropic_model == "claude-opus-5-5"
    assert s.model_for("openai") == "gpt-6.2-sol"
