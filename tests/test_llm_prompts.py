import prompts
from llm import _degrade, _extract_json


def test_extract_json_handles_fences_and_prose():
    assert _extract_json('{"name": "A"}') == {"name": "A"}
    assert _extract_json('```json\n{"name": "A"}\n```') == {"name": "A"}
    assert _extract_json("Here it is: {\"name\": \"A\"} hope it helps") == {"name": "A"}
    assert _extract_json("no json") is None


def test_json_mode_degrades_only_on_format_errors():
    class ApiError(Exception):
        def __init__(self, msg, status):
            super().__init__(msg)
            self.status_code = status

    def wrapped(msg, status=None):
        err = RuntimeError("chat failed")
        err.__cause__ = ApiError(msg, status) if status else Exception(msg)
        return err

    assert _degrade("schema", wrapped("response_format json_schema is not supported", 400)) == "json_object"
    assert _degrade("json_object", wrapped("'json_object' response_format unsupported", 400)) == "prompt"
    assert _degrade("prompt", wrapped("response_format not supported", 400)) is None  # ultimo livello
    # errori che non riguardano il formato non devono cambiare modalità
    assert _degrade("schema", wrapped("Rate limit: limit 40000 tokens per minute", 429)) is None
    assert _degrade("schema", wrapped("context_length_exceeded: invalid_request_error", 400)) is None
    assert _degrade("schema", wrapped("Connection refused")) is None


def test_prompt_override_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "OVERRIDES_PATH", tmp_path / "prompts.json")
    default = prompts.get_prompt("cluster_naming")
    assert not default.customized
    prompts.save_prompt("cluster_naming", "custom $language", default.user)
    custom = prompts.get_prompt("cluster_naming")
    assert custom.customized and custom.version != default.version
    assert custom.render(language="Italian")[0] == "custom Italian"
    assert "keywords" not in prompts.missing_variables("cluster_naming", custom.system, custom.user)
    assert "language" in prompts.missing_variables("cluster_naming", "no vars", "")
    prompts.reset_prompt("cluster_naming")
    assert not prompts.get_prompt("cluster_naming").customized


def test_provider_config_never_leaks_shared_key(monkeypatch):
    """La chiave condivisa (es. OpenRouter) non deve finire a un provider diverso salvato dalla GUI."""
    import providers

    env = {"BASE_URL": "https://openrouter.ai/api/v1", "API_KEY": "sk-or-shared",
           "LLM_PROVIDER": "openai", "LLM_BASE_URL": "http://localhost:1234/v1",
           "EMBEDDING_PROVIDER": "openrouter"}
    monkeypatch.setattr(providers, "get_env", lambda k, d="": env.get(k, d))
    chat = providers.config_from_env("chat")
    assert chat.provider == "openai" and chat.api_key == ""  # provider salvato rispettato, nessuna fuga di chiave
    emb = providers.config_from_env("embedding")
    assert emb.provider == "openrouter" and emb.api_key == "sk-or-shared"  # stesso provider: chiave condivisa ok
