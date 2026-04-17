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


from src.tools import ToolContext, default_registry

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
            api_keys={"xai": settings.xai_api_key},
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

    # Show a typing-style status indicator while the bot is "working" with
    # nothing to reply yet. We intentionally do NOT post a placeholder
    # chat.postMessage up front: that would render as a separate UI element
    # alongside the status line (a duplicate-message look on AI workspaces).
    # The placeholder is posted lazily in _on_stream_wrapped once the first
    # real content delta arrives. Slack auto-clears the status when the bot
    # posts in the thread; we also explicitly clear it after we finalize.
    set_thread_status(client, channel, thread_ts, labels["thinking"] + settings.bot_cursor)

    stream_msg = StreamingMessage(
        client=client,
        channel=channel,
        thread_ts=thread_ts,
        placeholder=settings.bot_cursor,
        min_interval=0.6,
        max_len=settings.max_len_slack,
    )

    def _on_stream_wrapped(delta: str) -> None:
        """Defer placeholder posting until the first real content arrives."""
        if not delta:
            return
        if stream_msg.ts is None:
            try:
                stream_msg.start()
            except Exception as exc:  # noqa: BLE001
                logger.warning("deferred streaming message start failed: %s", exc)
                return
        stream_msg.append(delta)

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
        # While no message is posted yet, use assistant_threads.setStatus.
        # Once the stream has started (stream_msg.ts is set), the bot message
        # is already visible — skip status updates to avoid re-triggering the
        # duplicate-UI problem.
        if stream_msg.ts is not None:
            return
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
        set_thread_status(client, channel, thread_ts, status + " " + settings.bot_cursor)

    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        registry=default_registry,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
        system_message=settings.system_message,
        history=history,
        on_stream=_on_stream_wrapped,
        on_step=_on_step,
        max_output_tokens=settings.max_output_tokens,
    )

    user_name = user_name_cache.get(client, user) if user else ""
    log_event(logger, "agent.start", user=user_name or user, channel=channel, is_dm=is_dm)

    try:
        result = agent.run(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent failure")
        error_text = f"{labels['error_prefix']}: {sanitize_error(exc)}"
        if stream_msg.ts is not None:
            stream_msg.stop(error_text)
        else:
            say(text=error_text, thread_ts=thread_ts)
        set_thread_status(client, channel, thread_ts, "")
        return

    final_text = result.text or "(응답을 생성하지 못했습니다)"
    # Split the answer by Slack's per-message limit. StreamingMessage.stop()
    # handles split internally when a placeholder exists; when it doesn't
    # (no stream deltas ever arrived — e.g. a provider that returned content
    # all at once), we post the chunks as fresh thread messages instead.
    chunks = MessageFormatter.split_message(final_text, max_len=settings.max_len_slack)
    if stream_msg.ts is not None:
        stream_msg.stop(chunks[0])
    else:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=chunks[0])
    for extra in chunks[1:]:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=extra)
    # Explicitly clear the typing-style status indicator. Slack usually
    # auto-clears it when the bot posts a reply, but an explicit clear
    # ensures there's no stale line left over from the last on_step update.
    set_thread_status(client, channel, thread_ts, "")
    # NOTE: do not post `result.image_url` as a separate text message —
    # the image is already uploaded inline to the thread by the
    # generate_image tool, and the LLM's reply is instructed to omit
    # the permalink. A trailing "생성된 이미지: <url>" line would just
    # duplicate what the user already sees.

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
