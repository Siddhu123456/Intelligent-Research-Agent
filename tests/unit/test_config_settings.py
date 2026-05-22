import importlib

import config.settings as settings_module


def test_settings_class_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "langchain-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "test-project")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://example.com")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    importlib.reload(settings_module)

    settings = settings_module.Settings()

    assert settings.groq_api_key == "groq-key"
    assert settings.google_api_key == "google-key"
    assert settings.openrouter_api_key == "openrouter-key"
    assert settings.tavily_api_key == "tavily-key"
    assert settings.langchain_api_key == "langchain-key"
    assert settings.langchain_project == "test-project"
    assert settings.langchain_endpoint == "https://example.com"
    assert settings.langchain_tracing_v2 == "false"
