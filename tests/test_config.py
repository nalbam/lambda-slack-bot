import pytest


@pytest.fixture
def reload_config():
    """Build a fresh Settings from the current os.environ (no module reload)."""

    def _reload():
        from src.config import Settings

        return Settings.from_env()

    return _reload


def _clear_env(monkeypatch):
    for key in [
        "SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "LLM_PROVIDER", "LLM_MODEL",
        "IMAGE_PROVIDER", "IMAGE_MODEL", "OPENAI_API_KEY", "RESPONSE_LANGUAGE",
        "AGENT_MAX_STEPS", "DYNAMODB_TABLE_NAME", "AWS_REGION", "ALLOWED_CHANNEL_IDS",
        "ALLOWED_CHANNEL_MESSAGE", "MAX_LEN_SLACK", "MAX_THROTTLE_COUNT",
        "MAX_HISTORY_CHARS", "BOT_CURSOR", "SYSTEM_MESSAGE", "TAVILY_API_KEY", "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_defaults(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    s = reload_config()
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.image_provider == "openai"
    assert s.image_model == "gpt-image-1"
    assert s.response_language == "ko"
    assert s.agent_max_steps == 3
    assert s.max_len_slack == 3000
    assert s.allowed_channel_ids == []
    assert s.tavily_api_key is None


def test_invalid_enum_falls_back(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("RESPONSE_LANGUAGE", "jp")
    monkeypatch.setenv("LLM_PROVIDER", "mystery")
    s = reload_config()
    assert s.response_language == "ko"
    assert s.llm_provider == "openai"


def test_invalid_int_falls_back_to_default(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_STEPS", "not-an-int")
    s = reload_config()
    assert s.agent_max_steps == 3


def test_int_below_minimum_clamped(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MAX_LEN_SLACK", "10")
    s = reload_config()
    assert s.max_len_slack == 500


def test_list_env_splits_commas(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_CHANNEL_IDS", "C1,C2, C3 ")
    s = reload_config()
    assert s.allowed_channel_ids == ["C1", "C2", "C3"]


def test_list_env_none_sentinel(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_CHANNEL_IDS", "None")
    s = reload_config()
    assert s.allowed_channel_ids == []


def test_require_slack_credentials_raises_when_missing(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    s = reload_config()
    with pytest.raises(RuntimeError):
        s.require_slack_credentials()


def test_require_slack_credentials_ok(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-x")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")
    s = reload_config()
    s.require_slack_credentials()  # no raise
