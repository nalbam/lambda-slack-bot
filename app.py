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
    MessageFormatter,
    StreamingMessage,
    channel_allowed,
    sanitize_error,
    set_thread_status,
    user_name_cache,
)


def _split_for_slack(text: str, max_len: int) -> list[str]:
    return MessageFormatter.split_message(text, max_len=max_len)
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
        "max_steps": "답변 정리 중... ",
        "using_tools": "도구 사용 중: {tools}",
        "tool_ok": "도구 완료: {tool}",
        "tool_failed": "도구 실패: {tool}",
        "composing": "답변 작성 중...",
    },
    "en": {
        "generated_image": "Generated image",
        "error_prefix": "An error occurred while processing your request",
        "throttled": "Too many in-flight requests. Please try again shortly.",
        "thinking": "Thinking... ",
        "max_steps": "Finalizing... ",
        "using_tools": "Running tools: {tools}",
        "tool_ok": "Finished: {tool}",
        "tool_failed": "Failed: {tool}",
        "composing": "Composing the answer...",
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

    stream_msg = StreamingMessage(
        client=client,
        channel=channel,
        thread_ts=thread_ts,
        placeholder=settings.bot_cursor,
        min_interval=0.6,
    )
    try:
        stream_msg.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("streaming message start failed: %s", exc)

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

    def _on_step(step_num: int, phase: str, detail: dict) -> None:
        if phase == "tool_use":
            tools = ", ".join(detail.get("tools") or [])
            status = labels["using_tools"].format(tools=tools)
        elif phase == "tool_result":
            key = "tool_ok" if detail.get("ok") else "tool_failed"
            status = labels[key].format(tool=detail.get("tool") or "")
        elif phase == "compose":
            status = labels["max_steps"] if detail.get("max_steps_hit") else labels["composing"]
        else:
            return
        set_thread_status(client, channel, thread_ts, status)

    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        registry=default_registry,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
        system_message=settings.system_message,
        history=history,
        on_stream=stream_msg.append,
        on_step=_on_step,
    )

    user_name = user_name_cache.get(client, user) if user else ""
    log_event(logger, "agent.start", user=user_name or user, channel=channel, is_dm=is_dm)

    try:
        result = agent.run(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failure")
        error_text = f"{labels['error_prefix']}: {sanitize_error(exc)}"
        stream_msg.stop(error_text)
        if stream_msg.ts is None:
            say(text=error_text, thread_ts=thread_ts)
        return

    final_text = result.text or "(응답을 생성하지 못했습니다)"
    # If the final text fits into one Slack message, finalize the stream with it.
    # Otherwise, close the stream with the first chunk and post remaining chunks
    # as new thread messages (send_long_message pattern).
    chunks = _split_for_slack(final_text, settings.max_len_slack)
    stream_msg.stop(chunks[0])
    for extra in chunks[1:]:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=extra)
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
