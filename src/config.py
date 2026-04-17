import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env_local() -> None:
    """Load .env.local from the project root when running locally."""
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent.parent / ".env.local"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


_load_env_local()


_VALID_LANGUAGES = {"ko", "en"}
_VALID_PROVIDERS = {"openai", "bedrock"}


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid int for %s=%r, using default=%d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%d below minimum %d, using minimum", name, value, minimum)
        return minimum
    return value


def _list_env(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw or raw.lower() == "none":
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _enum_env(name: str, default: str, valid: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in valid:
        logger.warning("invalid %s=%r, falling back to %s", name, value, default)
        return default
    return value


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str
    slack_signing_secret: str
    llm_provider: str
    llm_model: str
    image_provider: str
    image_model: str
    agent_max_steps: int
    response_language: str
    dynamodb_table_name: str
    aws_region: str
    allowed_channel_ids: list[str] = field(default_factory=list)
    allowed_channel_message: str = ""
    max_len_slack: int = 3000
    max_throttle_count: int = 100
    max_history_chars: int = 4000
    max_output_tokens: int = 4096
    bot_cursor: str = ":robot_face:"
    system_message: str | None = None
    tavily_api_key: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        llm_provider = _enum_env("LLM_PROVIDER", "openai", _VALID_PROVIDERS)
        image_provider = _enum_env(
            "IMAGE_PROVIDER",
            os.getenv("LLM_PROVIDER", "openai").strip().lower() or "openai",
            _VALID_PROVIDERS,
        )
        response_language = _enum_env("RESPONSE_LANGUAGE", "ko", _VALID_LANGUAGES)
        system_message = os.getenv("SYSTEM_MESSAGE", "").strip() or None
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip() or None
        return cls(
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN", "").strip(),
            slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", "").strip(),
            llm_provider=llm_provider,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            image_provider=image_provider,
            image_model=os.getenv("IMAGE_MODEL", "gpt-image-1").strip(),
            agent_max_steps=_int_env("AGENT_MAX_STEPS", 3, minimum=1),
            response_language=response_language,
            dynamodb_table_name=os.getenv("DYNAMODB_TABLE_NAME", "lambda-slack-bot-dev").strip(),
            aws_region=os.getenv("AWS_REGION", "us-east-1").strip(),
            allowed_channel_ids=_list_env("ALLOWED_CHANNEL_IDS"),
            allowed_channel_message=os.getenv("ALLOWED_CHANNEL_MESSAGE", "").strip(),
            max_len_slack=_int_env("MAX_LEN_SLACK", 2000, minimum=500),
            max_throttle_count=_int_env("MAX_THROTTLE_COUNT", 100, minimum=1),
            max_history_chars=_int_env("MAX_HISTORY_CHARS", 4000, minimum=500),
            max_output_tokens=_int_env("MAX_OUTPUT_TOKENS", 4096, minimum=256),
            bot_cursor=os.getenv("BOT_CURSOR", ":robot_face:").strip() or ":robot_face:",
            system_message=system_message,
            tavily_api_key=tavily_key,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )

    def require_slack_credentials(self) -> None:
        """Lazy validation — call from request handlers, not at import time."""
        if not self.slack_bot_token or not self.slack_signing_secret:
            raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required")
