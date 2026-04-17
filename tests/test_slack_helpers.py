from unittest.mock import MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from src.slack_helpers import (
    MessageFormatter,
    UserNameCache,
    channel_allowed,
    sanitize_error,
    send_long_message,
)


def test_split_short_message_is_single_chunk():
    chunks = MessageFormatter.split_message("hello", max_len=100)
    assert chunks == ["hello"]


def test_split_by_paragraph():
    para1 = "A" * 1500
    para2 = "B" * 1500
    text = f"{para1}\n\n{para2}"
    chunks = MessageFormatter.split_message(text, max_len=2000)
    assert len(chunks) == 2
    assert all(len(c) <= 2000 for c in chunks)


def test_split_by_sentence_when_paragraph_too_long():
    sent = "Sentence with some length. " * 100  # long paragraph without \n\n
    chunks = MessageFormatter.split_message(sent, max_len=300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_split_keeps_small_code_blocks_intact():
    """Code blocks that fit within max_len should not be split."""
    body = "text before\n\n```\nprint('x')\n```\n\ntext after"
    chunks = MessageFormatter.split_message(body, max_len=2000)
    # Entire content fits; must be a single chunk.
    assert len(chunks) == 1
    assert chunks[0].count("```") == 2


def test_split_code_block_longer_than_max_len_still_respects_limit():
    """When a code block exceeds max_len, fences may not balance per chunk,
    but no chunk may exceed max_len."""
    code = "```\n" + ("def x():\n    return 1\n" * 100) + "```"
    chunks = MessageFormatter.split_message(code, max_len=500)
    assert all(len(c) <= 500 for c in chunks)
    # Total fence count preserved across all chunks.
    total_fences = sum(c.count("```") for c in chunks)
    assert total_fences == 2


def test_split_empty_string():
    assert MessageFormatter.split_message("", max_len=100) == [""]


def test_send_long_message_first_chunk_uses_chat_update():
    client = MagicMock()
    send_long_message(
        client=client,
        channel="C1",
        thread_ts="ts1",
        text="short",
        first_ts="ts0",
        max_len=1000,
    )
    client.chat_update.assert_called_once_with(channel="C1", ts="ts0", text="short")
    client.chat_postMessage.assert_not_called()


def test_send_long_message_multi_chunk():
    client = MagicMock()
    text = "A" * 1200 + "\n\n" + "B" * 1200
    send_long_message(
        client=client,
        channel="C1",
        thread_ts="ts1",
        text=text,
        first_ts="ts0",
        max_len=1500,
    )
    assert client.chat_update.call_count == 1
    assert client.chat_postMessage.call_count >= 1


def test_send_long_message_falls_back_when_chat_update_fails():
    client = MagicMock()
    client.chat_update.side_effect = SlackApiError("fail", {"error": "msg_too_long"})
    send_long_message(
        client=client,
        channel="C1",
        thread_ts="ts1",
        text="hi",
        first_ts="ts0",
        max_len=1000,
    )
    client.chat_postMessage.assert_called_once()


def test_user_name_cache_uses_display_name():
    cache = UserNameCache._default()
    client = MagicMock()
    client.users_info.return_value = {"user": {"profile": {"display_name": "Alice"}}}
    assert cache.get(client, "U1") == "Alice"
    # second call is cached
    assert cache.get(client, "U1") == "Alice"
    client.users_info.assert_called_once()


def test_user_name_cache_falls_back_to_user_id_on_error():
    cache = UserNameCache._default()
    client = MagicMock()
    client.users_info.side_effect = SlackApiError("fail", {})
    assert cache.get(client, "U2") == "U2"


def test_channel_allowed_no_allowlist():
    assert channel_allowed("C1", []) is True


def test_channel_allowed_allowlist_match():
    assert channel_allowed("C1", ["C1", "C2"]) is True


def test_channel_allowed_allowlist_miss():
    assert channel_allowed("C9", ["C1", "C2"]) is False


def test_sanitize_error_redacts_tokens():
    class FakeErr(Exception):
        pass

    exc = FakeErr("failed with token xoxb-12345-67890 for /path/to/file.py boom")
    out = sanitize_error(exc)
    assert "xoxb-12345" not in out
    assert "redacted-slack-token" in out
    assert "[path]" in out


def test_sanitize_error_redacts_openai_key():
    exc = ValueError("Bad request using sk-proj-abcdefghij1234567890xyz")
    out = sanitize_error(exc)
    assert "sk-proj" not in out


def test_sanitize_error_truncates_long():
    exc = ValueError("x" * 1000)
    out = sanitize_error(exc)
    assert len(out) <= 300


# --------------------------------------------------------------------------- #
# StreamingMessage
# --------------------------------------------------------------------------- #

from src.slack_helpers import StreamingMessage


def _slack_client_native_stream():
    """Client whose api_call responds to chat.startStream/appendStream/stopStream."""
    client = MagicMock()

    def api_call(method, params=None, **_):
        if method == "chat.startStream":
            return {"ok": True, "channel": params["channel"], "ts": "1234.5678"}
        if method == "chat.appendStream":
            return {"ok": True, "channel": params["channel"], "ts": params["ts"]}
        if method == "chat.stopStream":
            return {"ok": True, "channel": params["channel"], "ts": params["ts"]}
        raise AssertionError(f"unexpected api_call: {method}")

    client.api_call.side_effect = api_call
    return client


def test_streaming_message_native_start_uses_api_call():
    client = _slack_client_native_stream()
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1", placeholder=":robot:")
    sm.start()
    assert sm.ts == "1234.5678"
    assert sm._native is True
    client.api_call.assert_called_once()
    assert client.api_call.call_args.args[0] == "chat.startStream"


def test_streaming_message_fallback_when_native_fails():
    client = MagicMock()
    client.api_call.side_effect = SlackApiError("no", {"error": "method_deprecated"})
    client.chat_postMessage.return_value = {"ok": True, "ts": "fallback-ts"}
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1")
    sm.start()
    assert sm.ts == "fallback-ts"
    assert sm._native is False
    client.chat_postMessage.assert_called_once()


def test_streaming_message_append_throttles():
    client = _slack_client_native_stream()
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1", min_interval=10.0)
    sm.start()
    # First append should flush (last_flush=0 -> elapsed > interval)
    sm.append("hello ")
    # Second append within interval: should buffer, not flush
    sm.append("world")
    # Count of appendStream calls should be <= 1 within this tight window
    append_calls = [c for c in client.api_call.call_args_list if c.args[0] == "chat.appendStream"]
    assert len(append_calls) <= 1


def test_streaming_message_stop_finalizes_native():
    client = _slack_client_native_stream()
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1")
    sm.start()
    sm.stop("final answer")
    stop_calls = [c for c in client.api_call.call_args_list if c.args[0] == "chat.stopStream"]
    assert len(stop_calls) == 1
    assert stop_calls[0].kwargs["params"]["markdown_text"] == "final answer"


def test_streaming_message_stop_fallback_uses_chat_update():
    client = MagicMock()
    client.api_call.side_effect = SlackApiError("no", {"error": "method_deprecated"})
    client.chat_postMessage.return_value = {"ok": True, "ts": "fallback-ts"}
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1")
    sm.start()
    sm.stop("done")
    client.chat_update.assert_called_with(channel="C1", ts="fallback-ts", text="done")


def test_streaming_message_stop_is_idempotent():
    client = _slack_client_native_stream()
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1")
    sm.start()
    sm.stop("a")
    sm.stop("b")
    stop_calls = [c for c in client.api_call.call_args_list if c.args[0] == "chat.stopStream"]
    assert len(stop_calls) == 1  # only first stop fires


def test_streaming_message_append_noop_before_start():
    client = MagicMock()
    sm = StreamingMessage(client=client, channel="C1", thread_ts="ts1")
    sm.append("hi")  # should not explode, ts is None
    client.api_call.assert_not_called()
