import os
import re

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

from lambda_slack_bot.agent import SlackMentionAgent
from lambda_slack_bot.config import Settings
from lambda_slack_bot.llm import LLMClient
from lambda_slack_bot.tools import ToolContext

settings = Settings.from_env()

if not settings.slack_bot_token or not settings.slack_signing_secret:
    raise RuntimeError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required")

app = App(token=settings.slack_bot_token, signing_secret=settings.slack_signing_secret, process_before_response=True)


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
    agent = SlackMentionAgent(llm=llm, context=context, max_steps=settings.agent_max_steps)

    try:
        result = agent.run(text)
        message = result.text
        if result.image_url:
            message = f"{message}\n\n생성된 이미지: {result.image_url}"
        say(text=message, thread_ts=thread_ts)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to handle mention")
        say(text=f"요청 처리 중 오류가 발생했습니다: {exc}", thread_ts=thread_ts)


def lambda_handler(event, context):
    os.environ.setdefault("SLACK_BOT_TOKEN", settings.slack_bot_token)
    os.environ.setdefault("SLACK_SIGNING_SECRET", settings.slack_signing_secret)
    return SlackRequestHandler(app).handle(event, context)
