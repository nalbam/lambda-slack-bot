"""Tool registry + 4 built-in tools with JSON Schema specs.

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

from botocore.exceptions import BotoCoreError, ClientError
from slack_sdk.errors import SlackApiError

from src.config import Settings
from src.llm import LLMProvider, ToolCall
from src.slack_helpers import user_name_cache

logger = logging.getLogger(__name__)


SLACK_FILE_HOSTS = {"files.slack.com", "files-edge.slack.com", "files-pri.slack.com"}
DUCKDUCKGO_HOST = "api.duckduckgo.com"
TAVILY_HOST = "api.tavily.com"
DOC_TEXT_PREFIX = "text/"
DOC_PDF_MIME = "application/pdf"


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
        except (
            TypeError,
            ValueError,
            KeyError,
            urllib.error.URLError,
            json.JSONDecodeError,
            SlackApiError,
            BotoCoreError,
            ClientError,
        ) as exc:
            logger.exception("tool %s failed", call.name)
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Built-in tools
# --------------------------------------------------------------------------- #

default_registry = ToolRegistry()


@tool(
    default_registry,
    name="read_attached_images",
    description=(
        "Read image files and return textual descriptions. By default reads "
        "images attached to the current Slack mention. Pass `urls` to also "
        "read images referenced from thread history (e.g. url_private_download "
        "returned by fetch_thread_history)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional Slack file URLs to describe (must be on files*.slack.com).",
            },
        },
        "required": [],
    },
)
def read_attached_images(
    ctx: ToolContext,
    limit: int = 3,
    urls: list[str] | None = None,
) -> list[dict[str, str]]:
    token = ctx.settings.slack_bot_token
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _fetch(url: str, mime_hint: str, name: str) -> None:
        if url in seen:
            return
        seen.add(url)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in SLACK_FILE_HOSTS:
            raise ValueError("invalid Slack file download URL")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310 (host allowlisted)
            data = response.read()
        mime = mime_hint if mime_hint.startswith("image/") else _guess_image_mime(url)
        if not mime.startswith("image/"):
            return
        out.append({"name": name, "summary": ctx.llm.describe_image(data, mime)})

    # 1) Images from the current mention event
    for file_info in (ctx.event.get("files") or [])[:limit]:
        if len(out) >= limit:
            break
        mime = str(file_info.get("mimetype", ""))
        if not mime.startswith("image/"):
            continue
        dl = file_info.get("url_private_download") or file_info.get("url_private")
        if not dl:
            continue
        _fetch(dl, mime, file_info.get("name", "image"))

    # 2) Extra URLs provided by the caller (typically from fetch_thread_history)
    for extra in (urls or []):
        if len(out) >= limit:
            break
        _fetch(extra, "", _filename_from_url(extra))

    return out


def _guess_image_mime(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for ext, mime in (
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
        (".bmp", "image/bmp"),
        (".heic", "image/heic"),
    ):
        if path.endswith(ext):
            return mime
    return "image/png"  # conservative default; describe_image will still attempt


def _filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = path.rsplit("/", 1)[-1] if path else "image"
    return name or "image"


def _fetch_slack_file(url: str, token: str, max_bytes: int) -> tuple[bytes, str]:
    """Fetch a Slack file with size guard. Returns (body, mimetype_from_header).

    Raises:
      ValueError: on disallowed host, oversize via Content-Length, or
                  oversize discovered while reading the body.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in SLACK_FILE_HOSTS:
        raise ValueError("invalid Slack file download URL")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
        content_length = response.headers.get("Content-Length") if response.headers else None
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError(f"document exceeds MAX_DOC_BYTES={max_bytes}")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"document exceeds MAX_DOC_BYTES={max_bytes}")
        mime = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower() if response.headers else ""
    return body, mime


def _parse_text(data: bytes, max_chars: int) -> tuple[str, bool]:
    text = data.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated


@tool(
    default_registry,
    name="read_attached_document",
    description=(
        "Read PDF or text/* files attached to the current Slack mention "
        "(and optionally extra URLs on files*.slack.com) and return the "
        "extracted text. Images are skipped — use read_attached_images "
        "for those. Returns one entry per document; if a document fails "
        "(encrypted, oversize, corrupt) the entry carries an 'error' key."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra Slack file URLs (must be on files*.slack.com).",
            },
        },
        "required": [],
    },
    timeout=30.0,
)
def read_attached_document(
    ctx: ToolContext,
    limit: int = 2,
    urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    token = ctx.settings.slack_bot_token
    max_bytes = ctx.settings.max_doc_bytes
    max_chars = ctx.settings.max_doc_chars
    # max_pages consumed in PDF branch (Task 5)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _is_doc_mime(mime: str) -> bool:
        mime = (mime or "").lower()
        return mime == DOC_PDF_MIME or mime.startswith(DOC_TEXT_PREFIX)

    def _process(url: str, file_mime_hint: str, name: str) -> None:
        if url in seen or len(out) >= limit:
            return
        seen.add(url)
        try:
            body, header_mime = _fetch_slack_file(url, token, max_bytes)
        except ValueError as exc:
            out.append({"name": name, "error": str(exc)})
            return
        except urllib.error.HTTPError as exc:
            out.append({"name": name, "error": f"HTTPError: {exc.code}"})
            return
        mime = (header_mime or file_mime_hint or "").lower()
        if mime == DOC_PDF_MIME:
            # Implemented in Task 5
            out.append({"name": name, "error": "pdf parsing not implemented yet"})
            return
        if mime.startswith(DOC_TEXT_PREFIX):
            text, truncated = _parse_text(body, max_chars)
            out.append(
                {
                    "name": name,
                    "mimetype": mime,
                    "pages": 0,
                    "chars": len(text),
                    "truncated": truncated,
                    "text": text,
                }
            )
            return
        # non-doc mime: silently skip (images handled by read_attached_images)

    for file_info in (ctx.event.get("files") or [])[:limit]:
        if len(out) >= limit:
            break
        mime = str(file_info.get("mimetype", ""))
        if not _is_doc_mime(mime):
            continue
        dl = file_info.get("url_private_download") or file_info.get("url_private")
        if not dl:
            continue
        _process(dl, mime, file_info.get("name", "document"))

    for extra in (urls or []):
        if len(out) >= limit:
            break
        _process(extra, "", _filename_from_url(extra))

    return out


@tool(
    default_registry,
    name="fetch_thread_history",
    description=(
        "Fetch recent messages from the current Slack thread for context. "
        "Returns each message's user display name, text, file metadata "
        "(for images include url_private_download so read_attached_images "
        "can describe them), reactions with emoji names and reacting users, "
        "and timestamp."
    ),
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20}},
        "required": [],
    },
)
def fetch_thread_history(ctx: ToolContext, limit: int = 20) -> list[dict[str, Any]]:
    def _map(res: dict[str, Any]) -> list[dict[str, Any]]:
        client = ctx.slack_client
        out: list[dict[str, Any]] = []
        for item in res.get("messages", []):
            user_id = item.get("user") or item.get("bot_id") or ""
            files = []
            for f in item.get("files") or []:
                files.append(
                    {
                        "name": f.get("name", ""),
                        "mimetype": f.get("mimetype", ""),
                        "url_private_download": f.get("url_private_download", ""),
                        "permalink": f.get("permalink", ""),
                        "title": f.get("title", ""),
                    }
                )
            reactions = []
            for r in item.get("reactions") or []:
                reacting_users = [user_name_cache.get(client, u) for u in (r.get("users") or [])]
                reactions.append(
                    {
                        "emoji": r.get("name", ""),
                        "count": r.get("count", 0),
                        "users": reacting_users,
                    }
                )
            out.append(
                {
                    "user": user_name_cache.get(client, user_id) if user_id else "",
                    "text": item.get("text", ""),
                    "ts": item.get("ts", ""),
                    "files": files,
                    "reactions": reactions,
                }
            )
        return out

    return _with_slack_retry(
        lambda: ctx.slack_client.conversations_replies(
            channel=ctx.channel, ts=ctx.thread_ts, limit=limit
        ),
        _map,
        label="conversations_replies",
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


@tool(
    default_registry,
    name="get_current_time",
    description=(
        "Return the current wall-clock time. Uses the server default "
        "timezone (DEFAULT_TIMEZONE env) unless 'timezone' is provided. "
        "Useful for 'today', 'now', 'this week', or weekday questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "Optional IANA timezone (e.g. 'Asia/Seoul', 'UTC', "
                    "'America/New_York'). Omit to use the server default."
                ),
            }
        },
        "required": [],
    },
)
def get_current_time(ctx: ToolContext, timezone: str | None = None) -> dict[str, Any]:
    from datetime import datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    tz_name = timezone or ctx.settings.default_timezone
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {tz_name}") from exc
    now = datetime.now(tz)
    return {
        "iso": now.isoformat(timespec="seconds"),
        "timezone": tz_name,
        "weekday": now.strftime("%A"),
        "unix": int(now.timestamp()),
    }


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


