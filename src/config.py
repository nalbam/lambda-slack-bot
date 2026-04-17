import os
from dataclasses import dataclass
from pathlib import Path


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

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
            slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
            llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            image_provider=os.getenv("IMAGE_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower(),
            image_model=os.getenv("IMAGE_MODEL", "gpt-image-1"),
            agent_max_steps=int(os.getenv("AGENT_MAX_STEPS", "3")),
            response_language=os.getenv("RESPONSE_LANGUAGE", "ko"),
        )
