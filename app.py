import os
import re

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

from src.agent import SlackMentionAgent
from src.config import Settings
from src.llm import LLMClient
from src.tools import ToolContext

settings = Settings.from_env()

if not settings.slack_bot_token or not settings.slack_signing_secret:
    raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required")

app = App(token=settings.slack_bot_token, signing_secret=settings.slack_signing_secret, process_before_response=True)

LABELS = {
    "ko": {"generated_image": "생성된 이미지", "error_prefix": "요청 처리 중 오류가 발생했습니다"},
    "en": {"generated_image": "Generated image", "error_prefix": "An error occurred while processing your request"},
}


@app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    text = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        image_provider=settings.image_provider,
        image_model=settings.image_model,
    )
    context = ToolContext(
        slack_client=client,
        channel=channel,
        thread_ts=thread_ts,
        event=event,
        settings=settings,
        llm=llm,
    )
    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
    )
    labels = LABELS.get(settings.response_language, LABELS["en"])

    try:
        result = agent.run(text)
        message = result.text
        if result.image_url:
            message = f"{message}\n\n{labels['generated_image']}: {result.image_url}"
        say(text=message, thread_ts=thread_ts)
    except (RuntimeError, ValueError) as exc:
        logger.exception("Failed to handle mention")
        say(text=f"{labels['error_prefix']}: {exc}", thread_ts=thread_ts)
    except Exception:
        logger.exception("Unexpected failure while handling mention")
        raise


def lambda_handler(event, context):
    os.environ.setdefault("SLACK_BOT_TOKEN", settings.slack_bot_token)
    os.environ.setdefault("SLACK_SIGNING_SECRET", settings.slack_signing_secret)
    return SlackRequestHandler(app).handle(event, context)
