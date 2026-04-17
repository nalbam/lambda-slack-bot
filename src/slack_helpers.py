"""Slack-facing helpers: message splitting, status indicator, user name cache, allowlist."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


CODE_FENCE = "```"
PARAGRAPH_SEP = "\n\n"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class MessageFormatter:
    """Split a long message into Slack-safe chunks.

    Strategy (hierarchical):
      1. Prefer splitting on code fences so multi-line code blocks stay intact.
      2. Otherwise split on paragraph boundaries (\\n\\n).
      3. Otherwise split on sentence boundaries.
      4. Final fallback: hard slice at max_len.
    """

    @staticmethod
    def split_message(text: str, max_len: int = 3000) -> list[str]:
        if not text:
            return [""]
        if len(text) <= max_len:
            return [text]

        if CODE_FENCE in text:
            parts = text.split(CODE_FENCE)
            chunks: list[str] = []
            for idx, part in enumerate(parts):
                wrapped = f"{CODE_FENCE}{part}{CODE_FENCE}" if idx % 2 == 1 else part
                if not wrapped:
                    continue
                chunks.extend(MessageFormatter._split_text(wrapped, max_len))
            return MessageFormatter._merge_small(chunks, max_len)

        return MessageFormatter._split_text(text, max_len)

    @staticmethod
    def _split_text(text: str, max_len: int) -> list[str]:
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        for paragraph in text.split(PARAGRAPH_SEP):
            if len(paragraph) <= max_len:
                chunks.append(paragraph)
                continue
            # paragraph too long: split by sentence
            buf = ""
            for sentence in SENTENCE_SPLIT_RE.split(paragraph):
                if not sentence:
                    continue
                candidate = f"{buf} {sentence}".strip() if buf else sentence
                if len(candidate) > max_len:
                    if buf:
                        chunks.append(buf)
                    # sentence itself too long -> hard slice
                    while len(sentence) > max_len:
                        chunks.append(sentence[:max_len])
                        sentence = sentence[max_len:]
                    buf = sentence
                else:
                    buf = candidate
            if buf:
                chunks.append(buf)
        return MessageFormatter._merge_small(chunks, max_len)

    @staticmethod
    def _merge_small(chunks: Iterable[str], max_len: int) -> list[str]:
        out: list[str] = []
        buf = ""
        for chunk in chunks:
            if not chunk:
                continue
            candidate = f"{buf}{PARAGRAPH_SEP}{chunk}" if buf else chunk
            if len(candidate) <= max_len:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = chunk
        if buf:
            out.append(buf)
        return out or [""]


def send_long_message(
    *,
    client: Any,
    channel: str,
    thread_ts: str,
    text: str,
    first_ts: str | None = None,
    max_len: int = 3000,
    interval: float = 0.0,
) -> None:
    """Send a possibly-long message to Slack, updating the placeholder first if provided."""
    chunks = MessageFormatter.split_message(text, max_len=max_len)
    for idx, chunk in enumerate(chunks):
        if idx == 0 and first_ts:
            try:
                client.chat_update(channel=channel, ts=first_ts, text=chunk)
                continue
            except SlackApiError as exc:
                logger.warning("chat_update failed, falling back to postMessage: %s", exc)
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=chunk)
        if interval > 0 and idx < len(chunks) - 1:
            time.sleep(interval)


def set_thread_status(client: Any, channel: str, thread_ts: str, status: str) -> None:
    """Best-effort typing indicator. Swallows API errors (feature may not be enabled)."""
    try:
        client.assistant_threads_setStatus(channel_id=channel, thread_ts=thread_ts, status=status)
    except (SlackApiError, AttributeError, TypeError) as exc:
        logger.debug("assistant_threads_setStatus failed: %s", exc)


@dataclass
class UserNameCache:
    """Module-level cache keyed by user_id. Survives warm starts."""

    _cache: dict[str, str]

    @classmethod
    def _default(cls) -> "UserNameCache":
        return cls(_cache={})

    def get(self, client: Any, user_id: str) -> str:
        if not user_id:
            return ""
        if user_id in self._cache:
            return self._cache[user_id]
        try:
            info = client.users_info(user=user_id)
            profile = (info.get("user") or {}).get("profile") or {}
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or (info.get("user") or {}).get("real_name")
                or user_id
            )
        except SlackApiError as exc:
            logger.debug("users_info failed for %s: %s", user_id, exc)
            name = user_id
        self._cache[user_id] = name
        return name


user_name_cache = UserNameCache._default()


def channel_allowed(channel: str, allowed_ids: list[str]) -> bool:
    """Return True if no allowlist configured or channel is listed."""
    if not allowed_ids:
        return True
    return channel in allowed_ids


def sanitize_error(exc: BaseException) -> str:
    """User-facing error text. Strips internal paths/tokens."""
    msg = str(exc) or exc.__class__.__name__
    # Redact anything that looks like a Slack/OpenAI token.
    msg = re.sub(r"xox[abprs]-[A-Za-z0-9-]+", "[redacted-slack-token]", msg)
    msg = re.sub(r"sk-[A-Za-z0-9\-_]{10,}", "[redacted-openai-key]", msg)
    # Truncate stack-like paths.
    msg = re.sub(r"(/[\w./-]+\.py)", "[path]", msg)
    if len(msg) > 300:
        msg = msg[:297] + "..."
    return msg


def throttled(fn: Callable[[str], None], min_interval: float) -> Callable[[str], None]:
    """Wrap a callback so it fires at most once per min_interval seconds."""
    state = {"last": 0.0, "buf": ""}

    def emit(delta: str) -> None:
        state["buf"] += delta
        now = time.monotonic()
        if now - state["last"] >= min_interval:
            fn(state["buf"])
            state["last"] = now

    return emit
