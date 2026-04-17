import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.llm import ToolCall
from src.tools import (
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_registry,
    fetch_thread_history,
    generate_image,
    read_attached_images,
    search_web,
)


def _settings(**overrides) -> Settings:
    base = {
        "slack_bot_token": "xoxb-test",
        "slack_signing_secret": "sig",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "image_provider": "openai",
        "image_model": "gpt-image-1",
        "agent_max_steps": 3,
        "response_language": "ko",
        "dynamodb_table_name": "t",
        "aws_region": "us-east-1",
    }
    base.update(overrides)
    return Settings(**base)


def _ctx(event=None, slack_client=None, llm=None):
    return ToolContext(
        slack_client=slack_client or MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event or {},
        settings=_settings(),
        llm=llm or MagicMock(),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_default_registry_has_expected_tools():
    names = set(default_registry.names())
    assert {"read_attached_images", "fetch_thread_history", "search_web", "generate_image"}.issubset(names)
    assert "search_slack_messages" not in names  # removed — user-token only, tied to installer


def test_registry_specs_match_llm_shape():
    for spec in default_registry.specs():
        assert set(spec.keys()) == {"name", "description", "parameters"}
        assert spec["parameters"]["type"] == "object"


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


def test_executor_unknown_tool():
    registry = ToolRegistry()
    executor = ToolExecutor(_ctx(), registry)
    result = executor.execute(ToolCall(id="1", name="nope", arguments={}))
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_executor_timeout_guards_slow_tools():
    import time

    registry = ToolRegistry()

    def slow(ctx):
        time.sleep(1.0)

    from src.tools import ToolDef

    registry.register(ToolDef(name="slow", description="", parameters={"type": "object", "properties": {}}, fn=slow))
    executor = ToolExecutor(_ctx(), registry, timeout=0.1)
    result = executor.execute(ToolCall(id="1", name="slow", arguments={}))
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_executor_wraps_boto_client_error():
    """Bedrock invoke failures (botocore ClientError) must be returned as
    {ok: False, error: ...} so the LLM can plan around the failure instead
    of the exception bubbling out of the agent loop."""
    from botocore.exceptions import ClientError

    registry = ToolRegistry()

    def failing_bedrock(ctx):
        raise ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Legacy model"}},
            "InvokeModel",
        )

    from src.tools import ToolDef

    registry.register(
        ToolDef(
            name="bedrock_thing",
            description="",
            parameters={"type": "object", "properties": {}},
            fn=failing_bedrock,
        )
    )
    executor = ToolExecutor(_ctx(), registry)
    result = executor.execute(ToolCall(id="1", name="bedrock_thing", arguments={}))
    assert result["ok"] is False
    assert "ResourceNotFoundException" in result["error"] or "Legacy" in result["error"]


def test_executor_captures_tool_error():
    registry = ToolRegistry()

    def boom(ctx):
        raise ValueError("nope")

    from src.tools import ToolDef

    registry.register(ToolDef(name="boom", description="", parameters={"type": "object", "properties": {}}, fn=boom))
    executor = ToolExecutor(_ctx(), registry)
    result = executor.execute(ToolCall(id="1", name="boom", arguments={}))
    assert result["ok"] is False
    assert "nope" in result["error"]


def test_executor_per_tool_timeout_override():
    """A tool registered with its own timeout overrides the executor default."""
    import time

    registry = ToolRegistry()

    def moderately_slow(ctx):
        time.sleep(0.3)
        return "done"

    from src.tools import ToolDef

    registry.register(
        ToolDef(
            name="slowish",
            description="",
            parameters={"type": "object", "properties": {}},
            fn=moderately_slow,
            timeout=1.0,
        )
    )
    # Default timeout short enough to kill a naïve tool; per-tool override lets
    # this one finish.
    executor = ToolExecutor(_ctx(), registry, timeout=0.1)
    result = executor.execute(ToolCall(id="1", name="slowish", arguments={}))
    assert result["ok"] is True
    assert result["result"] == "done"


def test_generate_image_tool_has_extended_timeout():
    """Image generation is slow; its registered timeout must be > default."""
    td = default_registry.get("generate_image")
    assert td is not None
    assert td.timeout is not None
    assert td.timeout >= 60.0


# --------------------------------------------------------------------------- #
# read_attached_images SSRF guard
# --------------------------------------------------------------------------- #


def test_read_attached_images_rejects_non_slack_host():
    event = {"files": [{"mimetype": "image/png", "url_private_download": "https://evil.example.com/x.png"}]}
    with pytest.raises(ValueError):
        read_attached_images(_ctx(event=event), limit=1)


def test_read_attached_images_rejects_http_scheme():
    event = {"files": [{"mimetype": "image/png", "url_private_download": "http://files.slack.com/x.png"}]}
    with pytest.raises(ValueError):
        read_attached_images(_ctx(event=event), limit=1)


def test_read_attached_images_accepts_slack_host_variants():
    event = {
        "files": [
            {"mimetype": "image/png", "url_private_download": "https://files-pri.slack.com/x.png", "name": "a"},
        ]
    }
    llm = MagicMock()
    llm.describe_image.return_value = "a cat"
    ctx = _ctx(event=event, llm=llm)
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"fake"
        result = read_attached_images(ctx, limit=1)
    assert result == [{"name": "a", "summary": "a cat"}]


def test_read_attached_images_skips_non_image_mimetypes():
    event = {"files": [{"mimetype": "application/pdf", "url_private_download": "https://files.slack.com/x.pdf"}]}
    assert read_attached_images(_ctx(event=event), limit=1) == []


# --------------------------------------------------------------------------- #
# fetch_thread_history
# --------------------------------------------------------------------------- #


def test_fetch_thread_history_resolves_user_files_and_reactions():
    """History should carry display names, file metadata, and reactions so the
    LLM can answer things like "누가 좋아요 눌렀어?" or "아까 그 이미지 분석해줘"."""
    from src.slack_helpers import user_name_cache

    # Reset the module-level cache so prior tests don't leak.
    user_name_cache._cache.clear()

    client = MagicMock()
    client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U1",
                "text": "look at this",
                "ts": "1713.1",
                "files": [
                    {
                        "name": "cat.png",
                        "mimetype": "image/png",
                        "url_private_download": "https://files.slack.com/x/cat.png",
                        "permalink": "https://slack/p1",
                        "title": "cute",
                    }
                ],
            },
            {
                "user": "U2",
                "text": "nice!",
                "ts": "1713.2",
                "reactions": [
                    {"name": "thumbsup", "count": 2, "users": ["U1", "U3"]},
                ],
            },
        ]
    }

    def _users_info(user):
        return {"user": {"profile": {"display_name": f"name-{user}"}}}

    client.users_info.side_effect = _users_info

    out = fetch_thread_history(_ctx(slack_client=client), limit=5)
    assert len(out) == 2
    first, second = out
    assert first["user"] == "name-U1"
    assert first["text"] == "look at this"
    assert first["ts"] == "1713.1"
    assert first["files"] == [
        {
            "name": "cat.png",
            "mimetype": "image/png",
            "url_private_download": "https://files.slack.com/x/cat.png",
            "permalink": "https://slack/p1",
            "title": "cute",
        }
    ]
    assert first["reactions"] == []

    assert second["user"] == "name-U2"
    assert second["files"] == []
    assert second["reactions"] == [
        {"emoji": "thumbsup", "count": 2, "users": ["name-U1", "name-U3"]}
    ]


def test_read_attached_images_accepts_extra_urls():
    """Images referenced from fetch_thread_history (url_private_download) must
    be loadable via read_attached_images(urls=[...])."""
    ctx = _ctx()
    ctx.llm.describe_image.return_value = "a cat history"
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"fake-bytes"
        out = read_attached_images(
            ctx,
            limit=5,
            urls=["https://files.slack.com/x/cat.png"],
        )
    assert out == [{"name": "cat.png", "summary": "a cat history"}]


def test_read_attached_images_urls_reject_non_slack_host():
    ctx = _ctx()
    with pytest.raises(ValueError):
        read_attached_images(ctx, urls=["https://evil.example.com/cat.png"])


def test_read_attached_images_respects_total_limit_across_event_and_urls():
    event = {
        "files": [
            {
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/e1.png",
                "name": "e1.png",
            }
        ]
    }
    ctx = _ctx(event=event)
    ctx.llm.describe_image.return_value = "desc"
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"x"
        out = read_attached_images(
            ctx,
            limit=2,
            urls=[
                "https://files.slack.com/u1.png",
                "https://files.slack.com/u2.png",  # should be skipped (limit=2)
            ],
        )
    assert len(out) == 2
    assert {item["name"] for item in out} == {"e1.png", "u1.png"}


# --------------------------------------------------------------------------- #
# search_web
# --------------------------------------------------------------------------- #


def test_search_web_ddg_parses_results():
    ctx = _ctx()
    payload = {
        "AbstractURL": "https://example.com/a",
        "AbstractText": "abstract",
        "RelatedTopics": [{"Text": "t1", "FirstURL": "https://example.com/1"}],
    }
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        results = search_web(ctx, query="q", limit=5)
    assert results[0]["url"] == "https://example.com/a"
    assert results[1]["url"] == "https://example.com/1"


def test_search_web_uses_tavily_when_key_set():
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event={},
        settings=_settings(tavily_api_key="tvly-xyz"),
        llm=MagicMock(),
    )
    payload = {"results": [{"title": "t", "url": "https://x", "content": "c"}]}
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        out = search_web(ctx, query="q", limit=5)
    assert out == [{"title": "t", "url": "https://x", "content": "c"}]


# --------------------------------------------------------------------------- #
# generate_image
# --------------------------------------------------------------------------- #


def test_generate_image_returns_permalink():
    llm = MagicMock()
    llm.generate_image.return_value = b"imgbytes"
    client = MagicMock()
    client.files_upload_v2.return_value = {"file": {"permalink": "https://slack/abc", "title": "t"}}
    ctx = _ctx(slack_client=client, llm=llm)
    out = generate_image(ctx, prompt="cat")
    assert out == {"permalink": "https://slack/abc", "title": "t"}
    llm.generate_image.assert_called_once_with("cat")


# --------------------------------------------------------------------------- #
# get_current_time
# --------------------------------------------------------------------------- #


def test_get_current_time_uses_default_timezone():
    from src.tools import get_current_time

    ctx = _ctx()  # _settings() default_timezone defaults to Asia/Seoul
    out = get_current_time(ctx)
    assert out["timezone"] == "Asia/Seoul"
    assert out["iso"].endswith("+09:00")
    # Weekday is a full English day name (Monday..Sunday)
    assert out["weekday"] in {
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    }
    assert isinstance(out["unix"], int)


def test_get_current_time_respects_custom_timezone():
    from src.tools import get_current_time

    ctx = _ctx()
    out = get_current_time(ctx, timezone="UTC")
    assert out["timezone"] == "UTC"
    assert out["iso"].endswith("+00:00")


def test_get_current_time_invalid_tz_via_executor():
    """Invalid timezone should surface as {ok: False, error: ...} via the
    executor so the LLM can recover."""
    from src.tools import default_registry

    executor = ToolExecutor(_ctx(), default_registry)
    result = executor.execute(
        ToolCall(id="t1", name="get_current_time", arguments={"timezone": "Narnia/Center"})
    )
    assert result["ok"] is False
    assert "unknown timezone" in result["error"]


# --------------------------------------------------------------------------- #
# read_attached_document
# --------------------------------------------------------------------------- #


def test_read_attached_document_text_file():
    from src.tools import read_attached_document

    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/notes.txt",
                "name": "notes.txt",
            }
        ]
    }
    ctx = _ctx(event=event)
    body = b"Hello\n  world.\nLine 3."
    with patch("src.tools.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.read.return_value = body
        resp.headers = {"Content-Length": str(len(body))}
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "notes.txt"
    assert entry["mimetype"] == "text/plain"
    assert entry["truncated"] is False
    assert "Hello" in entry["text"]
    assert entry["chars"] == len(entry["text"])
    assert entry["pages"] == 0  # text files report 0 pages
