import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str
    slack_signing_secret: str
    llm_provider: str
    llm_model: str
    image_provider: str
    image_model: str
    agent_max_steps: int

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
        )
