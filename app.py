"""AWS Lambda entrypoint for the Slack mention bot.

Flow per request:
  1. lambda_handler short-circuits Slack retries (X-Slack-Retry-Num header).
  2. Bolt dispatches event to app_mention / message handler.
  3. Handler acks immediately, then:
     - Deduplicates on client_msg_id via DynamoDB conditional put.
     - Checks channel allowlist + per-user throttle.
     - Sets typing status + sends a placeholder message.
     - Loads thread history from DynamoDB and runs the native-tool-calling agent.
     - Streams the final answer into `chat_update` (first chunk) + `chat_postMessage` (rest).
     - Persists updated conversation back to DynamoDB.
"""
from __future__ import annotations

import re
import uuid

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

from src.agent import SlackMentionAgent
from src.config import Settings
from src.dedup import ConversationStore, DedupStore
from src.llm import get_llm
from src.logging_utils import get_logger, log_event, set_request_id
from src.slack_helpers import (
    channel_allowed,
    sanitize_error,
    send_long_message,
    set_thread_status,
    throttled,
    user_name_cache,
)
from src.tools import ToolContext, ToolExecutor, default_registry

settings = Settings.from_env()
logger = get_logger("app")

_llm = None
_dedup: DedupStore | None = None
_conversations: ConversationStore | None = None
_bolt_app: App | None = None


LABELS = {
    "ko": {
        "generated_image": "생성된 이미지",
        "error_prefix": "요청 처리 중 오류가 발생했습니다",
        "throttled": "잠시 후 다시 시도해주세요. 처리 중인 요청이 많습니다.",
        "thinking": "생각 중... ",
        "max_steps": "답변을 정리 중... ",
    },
    "en": {
        "generated_image": "Generated image",
        "error_prefix": "An error occurred while processing your request",
        "throttled": "Too many in-flight requests. Please try again shortly.",
        "thinking": "Thinking... ",
        "max_steps": "Finalizing... ",
    },
}


def _labels() -> dict[str, str]:
    return LABELS.get(settings.response_language, LABELS["en"])


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(
            provider=settings.llm_provider,
            model=settings.llm_model,
            image_provider=settings.image_provider,
            image_model=settings.image_model,
            region=settings.aws_region,
        )
    return _llm


def _get_dedup() -> DedupStore:
    global _dedup
    if _dedup is None:
        _dedup = DedupStore(table_name=settings.dynamodb_table_name, region=settings.aws_region)
    return _dedup


def _get_conversations() -> ConversationStore:
    global _conversations
    if _conversations is None:
        _conversations = ConversationStore(table_name=settings.dynamodb_table_name, region=settings.aws_region)
    return _conversations


def _get_bolt_app() -> App:
    global _bolt_app
    if _bolt_app is not None:
        return _bolt_app
    settings.require_slack_credentials()
    app = App(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
        process_before_response=True,
    )

    @app.event("app_mention")
    def _on_mention(event, client, say, ack):  # noqa: ANN001
        ack()
        _process(event, client, say, is_dm=False)

    @app.event("message")
    def _on_message(event, client, say, ack):  # noqa: ANN001
        ack()
        if event.get("channel_type") != "im":
            return
        if event.get("bot_id") or event.get("subtype"):
            return
        _process(event, client, say, is_dm=True)

    _bolt_app = app
    return _bolt_app


MENTION_RE = re.compile(r"<@[^>]+>")


def _process(event: dict, client, say, is_dm: bool) -> None:  # noqa: ANN001
    set_request_id(str(uuid.uuid4()))
    labels = _labels()
    text = MENTION_RE.sub("", event.get("text", "")).strip()
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    user = event.get("user", "")

    dedup = _get_dedup()
    dedup_key = event.get("client_msg_id") or f"{channel}:{event.get('ts')}"
    try:
        if not dedup.reserve(f"dedup:{dedup_key}", user=user or "system"):
            log_event(logger, "dedup.skip", key=dedup_key)
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("dedup unavailable, proceeding without it: %s", exc)

    if not channel_allowed(channel, settings.allowed_channel_ids):
        msg = settings.allowed_channel_message or ""
        if msg:
            say(text=msg, thread_ts=thread_ts)
        log_event(logger, "channel.blocked", channel=channel)
        return

    try:
        active = dedup.count_user_active(user)
    except Exception as exc:  # noqa: BLE001
        logger.warning("throttle count unavailable: %s", exc)
        active = 0
    if active >= settings.max_throttle_count:
        say(text=labels["throttled"], thread_ts=thread_ts)
        log_event(logger, "throttle.limit", user=user, active=active)
        return

    if not text:
        return

    set_thread_status(client, channel, thread_ts, labels["thinking"] + settings.bot_cursor)

    try:
        placeholder = say(text=settings.bot_cursor, thread_ts=thread_ts)
        placeholder_ts = placeholder.get("ts") if isinstance(placeholder, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("placeholder say failed: %s", exc)
        placeholder_ts = None

    history_store = _get_conversations()
    history = history_store.get(thread_ts)

    llm = _get_llm()
    context = ToolContext(
        slack_client=client,
        channel=channel,
        thread_ts=thread_ts,
        event=event,
        settings=settings,
        llm=llm,
    )

    stream_ts = placeholder_ts

    def _stream_update(partial: str) -> None:
        if not stream_ts:
            return
        try:
            client.chat_update(channel=channel, ts=stream_ts, text=partial + " " + settings.bot_cursor)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat_update stream failed: %s", exc)

    on_stream = throttled(_stream_update, min_interval=0.6)

    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        registry=default_registry,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
        system_message=settings.system_message,
        history=history,
        on_stream=on_stream,
    )

    user_name = user_name_cache.get(client, user) if user else ""
    log_event(logger, "agent.start", user=user_name or user, channel=channel, is_dm=is_dm)

    try:
        result = agent.run(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failure")
        error_text = f"{labels['error_prefix']}: {sanitize_error(exc)}"
        if placeholder_ts:
            try:
                client.chat_update(channel=channel, ts=placeholder_ts, text=error_text)
            except Exception:  # noqa: BLE001
                say(text=error_text, thread_ts=thread_ts)
        else:
            say(text=error_text, thread_ts=thread_ts)
        return

    final_text = result.text or "(응답을 생성하지 못했습니다)"
    send_long_message(
        client=client,
        channel=channel,
        thread_ts=thread_ts,
        text=final_text,
        first_ts=placeholder_ts,
        max_len=settings.max_len_slack,
    )
    if result.image_url:
        say(text=f"{labels['generated_image']}: {result.image_url}", thread_ts=thread_ts)

    new_history = [
        *history,
        {"role": "user", "content": text},
        {"role": "assistant", "content": final_text},
    ]
    try:
        history_store.put(
            thread_ts,
            user=user or "unknown",
            messages=new_history,
            max_chars=settings.max_history_chars,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("conversation persist failed: %s", exc)

    log_event(
        logger,
        "agent.done",
        steps=result.steps,
        tool_calls=result.tool_calls_count,
        tokens_in=result.token_usage.get("input", 0),
        tokens_out=result.token_usage.get("output", 0),
    )


def lambda_handler(event, context):  # noqa: ANN001
    # Short-circuit Slack retries without re-running the agent.
    headers = event.get("headers") or {}
    normalized = {k.lower(): v for k, v in headers.items()}
    if normalized.get("x-slack-retry-num"):
        return {"statusCode": 200, "body": ""}
    return SlackRequestHandler(_get_bolt_app()).handle(event, context)
