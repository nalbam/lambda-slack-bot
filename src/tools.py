"""Tool registry + 5 built-in tools with JSON Schema specs.

Tools are declared once via the `@tool(...)` decorator. The same registry
produces JSON Schemas for LLM function calling AND the executor's dispatch
table. A per-call timeout guards against slow network I/O from one tool
blocking the whole agent loop.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from slack_sdk.errors import SlackApiError

from src.config import Settings
from src.llm import LLMProvider, ToolCall

logger = logging.getLogger(__name__)


SLACK_FILE_HOSTS = {"files.slack.com", "files-edge.slack.com", "files-pri.slack.com"}
DUCKDUCKGO_HOST = "api.duckduckgo.com"
TAVILY_HOST = "api.tavily.com"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Any]
    timeout: float | None = None  # None -> use executor default


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolDef] = field(default_factory=dict)

    def register(self, td: ToolDef) -> None:
        self._tools[td.name] = td

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]


def tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    parameters: dict[str, Any],
    timeout: float | None = None,
):
    def decorator(fn: Callable[..., Any]):
        registry.register(
            ToolDef(name=name, description=description, parameters=parameters, fn=fn, timeout=timeout)
        )
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class ToolContext:
    slack_client: Any
    channel: str
    thread_ts: str
    event: dict[str, Any]
    settings: Settings
    llm: LLMProvider


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class ToolExecutor:
    def __init__(self, context: ToolContext, registry: ToolRegistry, timeout: float = 20.0):
        self.context = context
        self.registry = registry
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=2)

    def execute(self, call: ToolCall) -> dict[str, Any]:
        td = self.registry.get(call.name)
        started = time.monotonic()
        if td is None:
            return {"ok": False, "error": f"unknown tool: {call.name}"}
        effective_timeout = td.timeout if td.timeout is not None else self.timeout
        try:
            future = self._pool.submit(td.fn, self.context, **(call.arguments or {}))
            result = future.result(timeout=effective_timeout)
            return {"ok": True, "result": result, "duration_ms": int((time.monotonic() - started) * 1000)}
        except FuturesTimeout:
            logger.warning("tool %s timed out after %.1fs", call.name, effective_timeout)
            return {"ok": False, "error": f"tool '{call.name}' timed out after {effective_timeout}s"}
        except (TypeError, ValueError, KeyError, urllib.error.URLError, json.JSONDecodeError, SlackApiError) as exc:
            logger.exception("tool %s failed", call.name)
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Built-in tools
# --------------------------------------------------------------------------- #

default_registry = ToolRegistry()


@tool(
    default_registry,
    name="read_attached_images",
    description="Read images attached to the current Slack mention and return textual descriptions.",
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3}},
        "required": [],
    },
)
def read_attached_images(ctx: ToolContext, limit: int = 3) -> list[dict[str, str]]:
    files = ctx.event.get("files") or []
    token = ctx.settings.slack_bot_token
    out: list[dict[str, str]] = []
    for file_info in files[:limit]:
        mime = str(file_info.get("mimetype", ""))
        if not mime.startswith("image/"):
            continue
        download_url = file_info.get("url_private_download") or file_info.get("url_private")
        if not download_url:
            continue
        parsed = urllib.parse.urlparse(download_url)
        if parsed.scheme != "https" or parsed.hostname not in SLACK_FILE_HOSTS:
            raise ValueError("invalid Slack file download URL")
        req = urllib.request.Request(download_url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310 (host allowlisted)
            data = response.read()
        out.append({"name": file_info.get("name", "image"), "summary": ctx.llm.describe_image(data, mime)})
    return out


@tool(
    default_registry,
    name="fetch_thread_history",
    description="Fetch recent messages from the current Slack thread for context.",
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20}},
        "required": [],
    },
)
def fetch_thread_history(ctx: ToolContext, limit: int = 20) -> list[dict[str, str]]:
    return _with_slack_retry(
        lambda: ctx.slack_client.conversations_replies(channel=ctx.channel, ts=ctx.thread_ts, limit=limit),
        lambda res: [
            {"user": item.get("user", ""), "text": item.get("text", "")}
            for item in res.get("messages", [])
        ],
        label="conversations_replies",
    )


@tool(
    default_registry,
    name="search_slack_messages",
    description="Search messages across Slack workspace by query string.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
        },
        "required": ["query"],
    },
)
def search_slack_messages(ctx: ToolContext, query: str, limit: int = 10) -> list[dict[str, str]]:
    return _with_slack_retry(
        lambda: ctx.slack_client.search_messages(query=query, count=limit),
        lambda res: [
            {"channel": m.get("channel", {}).get("name", ""), "text": m.get("text", "")}
            for m in res.get("messages", {}).get("matches", [])
        ],
        label="search_messages",
    )


@tool(
    default_registry,
    name="search_web",
    description="Search the public web for up-to-date information. Uses Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo Instant Answer.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
    },
)
def search_web(ctx: ToolContext, query: str, limit: int = 5) -> list[dict[str, str]]:
    if ctx.settings.tavily_api_key:
        return _tavily_search(ctx.settings.tavily_api_key, query, limit)
    return _ddg_search(query, limit)


@tool(
    default_registry,
    name="generate_image",
    description="Generate an image from a prompt and upload it to the Slack thread. Returns the permalink.",
    parameters={
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
    timeout=75.0,  # gpt-image-1 / titan / stability can take 30–60s
)
def generate_image(ctx: ToolContext, prompt: str) -> dict[str, str]:
    image_bytes = ctx.llm.generate_image(prompt)
    upload = ctx.slack_client.files_upload_v2(
        channel=ctx.channel,
        thread_ts=ctx.thread_ts,
        title="Generated image",
        filename="generated.png",
        file=image_bytes,
    )
    file_info = upload.get("file", {})
    return {"permalink": file_info.get("permalink", ""), "title": file_info.get("title", "generated.png")}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _with_slack_retry(call: Callable[[], Any], map_result: Callable[[Any], Any], label: str, attempts: int = 3) -> Any:
    delay = 1.0
    last: SlackApiError | None = None
    for attempt in range(attempts):
        try:
            return map_result(call())
        except SlackApiError as exc:
            error = (exc.response or {}).get("error") if hasattr(exc, "response") else None
            if error == "ratelimited" and attempt < attempts - 1:
                retry_after = int((exc.response.headers or {}).get("Retry-After", delay)) if hasattr(exc, "response") else delay
                logger.warning("%s rate limited, sleeping %ds", label, retry_after)
                time.sleep(retry_after)
                delay *= 2
                last = exc
                continue
            raise
    if last:
        raise last
    return []


def _ddg_search(query: str, limit: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": 1, "no_html": 1})
    url = f"https://{DUCKDUCKGO_HOST}/?{params}"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != DUCKDUCKGO_HOST:
        raise ValueError("invalid web search URL")
    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    results: list[dict[str, str]] = []
    if payload.get("AbstractURL"):
        results.append({"title": payload.get("AbstractText", ""), "url": payload["AbstractURL"]})
    for item in payload.get("RelatedTopics", []):
        if "Text" in item and "FirstURL" in item:
            results.append({"title": item["Text"], "url": item["FirstURL"]})
            if len(results) >= limit:
                break
    return results[:limit]


def _tavily_search(api_key: str, query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://{TAVILY_HOST}/search"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != TAVILY_HOST:
        raise ValueError("invalid Tavily URL")
    body = json.dumps({"api_key": api_key, "query": query, "max_results": limit}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in payload.get("results", [])[:limit]
    ]


