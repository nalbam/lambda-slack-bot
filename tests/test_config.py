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
        "MAX_HISTORY_CHARS", "BOT_CURSOR", "SYSTEM_MESSAGE", "TAVILY_API_KEY", "XAI_API_KEY", "LOG_LEVEL",
        "DEFAULT_TIMEZONE", "MAX_DOC_CHARS", "MAX_DOC_PAGES", "MAX_DOC_BYTES",
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
    assert s.max_len_slack == 2000
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


def test_xai_provider_is_a_valid_enum_value(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("IMAGE_PROVIDER", "xai")
    s = reload_config()
    assert s.llm_provider == "xai"
    assert s.image_provider == "xai"


def test_xai_api_key_default_none_and_override(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    s = reload_config()
    assert s.xai_api_key is None

    monkeypatch.setenv("XAI_API_KEY", "xai-abc")
    s2 = reload_config()
    assert s2.xai_api_key == "xai-abc"


def test_doc_limits_defaults(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    s = reload_config()
    assert s.default_timezone == "Asia/Seoul"
    assert s.max_doc_chars == 20_000
    assert s.max_doc_pages == 50
    assert s.max_doc_bytes == 25 * 1024 * 1024


def test_default_timezone_fallback_on_invalid_env(monkeypatch, reload_config, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEFAULT_TIMEZONE", "Narnia/Center")
    with caplog.at_level("WARNING"):
        s = reload_config()
    assert s.default_timezone == "Asia/Seoul"
    assert any("DEFAULT_TIMEZONE" in rec.message for rec in caplog.records)


def test_doc_limits_honor_env_and_clamp(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MAX_DOC_CHARS", "5000")
    monkeypatch.setenv("MAX_DOC_PAGES", "0")  # below minimum → clamps to 1
    monkeypatch.setenv("MAX_DOC_BYTES", "100")  # below minimum → clamps to 65536
    s = reload_config()
    assert s.max_doc_chars == 5000
    assert s.max_doc_pages == 1
    assert s.max_doc_bytes == 65_536


def test_default_timezone_custom_value(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEFAULT_TIMEZONE", "America/New_York")
    s = reload_config()
    assert s.default_timezone == "America/New_York"
