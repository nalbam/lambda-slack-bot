# Agent Tools: `get_current_time` + `read_attached_document` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two agent tools to `src/tools.py` — `get_current_time` (TZ-aware now) and `read_attached_document` (Slack-attached PDF/text extraction) — following the existing `@tool`-decorator + `ToolRegistry` pattern, with 12 new tests and config-driven limits.

**Architecture:** Two additive functions decorated with `@tool(default_registry, ...)`. Configuration extends `Settings` with four env-backed fields (`DEFAULT_TIMEZONE`, `MAX_DOC_CHARS`, `MAX_DOC_PAGES`, `MAX_DOC_BYTES`). `read_attached_document` reuses `read_attached_images`'s Slack host allowlist + bearer-token fetch pattern and returns a list of per-document entries (success or per-item error) so partial batches yield useful results.

**Tech Stack:** Python 3.12, `zoneinfo` (stdlib), `pypdf>=4.0` (new dep, pure Python, deferred-import to protect cold-start), `pytest` + `unittest.mock` for testing.

**Spec:** `docs/superpowers/specs/2026-04-17-agent-tools-time-document-design.md`

---

## File Structure

Files modified / created in this plan:

| Path | Purpose | Action |
|------|---------|--------|
| `src/config.py` | `Settings` dataclass + `_int_env` / `_tz_env` helpers | Modify — add 4 fields + new `_tz_env` helper |
| `src/tools.py` | Tool registry + built-in tools | Modify — add 2 tools + 2-3 helpers (~150 lines added) |
| `requirements.txt` | Runtime deps | Modify — add `pypdf>=4.0` |
| `.env.example` | Env var documentation | Modify — add 4 new vars |
| `README.md` | Public docs | Modify — list two new tools under "Tools" |
| `tests/test_config.py` | Settings tests | Modify — add 3 tests (TZ default/fallback, doc limits) |
| `tests/test_tools.py` | Tool tests | Modify — add 12 tests |
| `tests/conftest.py` | Pytest shared fixtures | Create only if helper for PDF fixtures is not inlined (see Task 5 note) |

**Note on `src/tools.py` size:** adding ~150 lines takes the file from ~400 → ~550. Still well under the 800-line hard cap; no structural split in this plan. A follow-up PR can split into a `tools/` package if further tools are added.

**Error policy (binding rule for implementer):** per-document failures during `read_attached_document` (SSRF reject, size cap, page cap, encrypted, corrupt PDF, per-URL HTTP 4xx) → emit `{"name": ..., "error": ...}` entry in the returned list; tool-call still returns `ok: true` via the executor. Only unexpected exceptions (network-wide DNS failure, TypeError from bad args) propagate to the executor, which wraps them as `{"ok": false, "error": ...}`. This intentionally overrides the older "re-raise PdfReadError as ValueError" wording in the spec — catching per-item gives the LLM better partial results.

---

## Task 1: Extend `Settings` with timezone and document-limit fields

**Files:**
- Modify: `src/config.py:28-40` (add `_tz_env` helper next to `_int_env`)
- Modify: `src/config.py:58-116` (add 4 fields on `Settings`, populate in `from_env`)
- Modify: `tests/test_config.py:17-24` (extend `_clear_env` list)
- Modify: `tests/test_config.py` (append 3 new tests)

- [ ] **Step 1: Write failing tests for new config fields**

Append to `tests/test_config.py`:

```python
def test_doc_limits_defaults(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    s = reload_config()
    assert s.default_timezone == "Asia/Seoul"
    assert s.max_doc_chars == 20_000
    assert s.max_doc_pages == 50
    assert s.max_doc_bytes == 25 * 1024 * 1024


def test_default_timezone_fallback_on_invalid_env(monkeypatch, reload_config, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEFAULT_TIMEZONE", "Narnia/Center")
    with caplog.at_level("WARNING"):
        s = reload_config()
    assert s.default_timezone == "Asia/Seoul"
    assert any("DEFAULT_TIMEZONE" in rec.message for rec in caplog.records)


def test_doc_limits_honor_env_and_clamp(monkeypatch, reload_config):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MAX_DOC_CHARS", "5000")
    monkeypatch.setenv("MAX_DOC_PAGES", "0")  # below minimum → clamps to 1
    monkeypatch.setenv("MAX_DOC_BYTES", "100")  # below minimum → clamps to 65536
    s = reload_config()
    assert s.max_doc_chars == 5000
    assert s.max_doc_pages == 1
    assert s.max_doc_bytes == 65_536
```

Also extend the `_clear_env` helper list at `tests/test_config.py:17-24` to include `DEFAULT_TIMEZONE`, `MAX_DOC_CHARS`, `MAX_DOC_PAGES`, `MAX_DOC_BYTES`.

- [ ] **Step 2: Run the new tests — expect all 3 to FAIL**

Run: `python -m pytest tests/test_config.py::test_doc_limits_defaults tests/test_config.py::test_default_timezone_fallback_on_invalid_env tests/test_config.py::test_doc_limits_honor_env_and_clamp -v`

Expected: 3 failures with `AttributeError: 'Settings' object has no attribute 'default_timezone'` (etc.).

- [ ] **Step 3: Implement `_tz_env` helper + 4 new `Settings` fields**

In `src/config.py`, add below `_int_env`:

```python
def _tz_env(name: str, default: str) -> str:
    """Return a validated IANA timezone name, warning + falling back on bad input."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        ZoneInfo(raw)
    except ZoneInfoNotFoundError:
        logger.warning("invalid %s=%r, falling back to %s", name, raw, default)
        return default
    return raw
```

In the `Settings` dataclass, add (matching the existing field style — these all have defaults so they come after the defaulted block, before the last default field):

```python
    default_timezone: str = "Asia/Seoul"
    max_doc_chars: int = 20_000
    max_doc_pages: int = 50
    max_doc_bytes: int = 25 * 1024 * 1024
```

In `Settings.from_env`, wire up the fields:

```python
            default_timezone=_tz_env("DEFAULT_TIMEZONE", "Asia/Seoul"),
            max_doc_chars=_int_env("MAX_DOC_CHARS", 20_000, minimum=1000),
            max_doc_pages=_int_env("MAX_DOC_PAGES", 50, minimum=1),
            max_doc_bytes=_int_env("MAX_DOC_BYTES", 25 * 1024 * 1024, minimum=64 * 1024),
```

- [ ] **Step 4: Run the new tests — all 3 PASS**

Run: `python -m pytest tests/test_config.py -v`

Expected: all config tests (old + new) pass.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): add DEFAULT_TIMEZONE and MAX_DOC_* settings"
```

Commit message body (via heredoc if preferred):
> Adds config scaffolding for the upcoming get_current_time and
> read_attached_document tools. Invalid DEFAULT_TIMEZONE falls back
> to Asia/Seoul with a warning, matching the existing enum-fallback
> pattern.

---

## Task 2: `get_current_time` — default timezone path

**Files:**
- Modify: `src/tools.py` (append new tool near the bottom, before `# Helpers` section divider at line ~341)
- Modify: `tests/test_tools.py` (extend imports, add tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools.py`:

```python
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
```

`_settings()` does NOT need updating — `default_timezone` has a dataclass default, so `Settings(**base)` still works unchanged.

- [ ] **Step 2: Run test — expect FAIL**

Run: `python -m pytest tests/test_tools.py::test_get_current_time_uses_default_timezone -v`

Expected: `ImportError: cannot import name 'get_current_time' from 'src.tools'`.

- [ ] **Step 3: Implement minimal `get_current_time`**

In `src/tools.py`, after the `generate_image` tool and before the `# Helpers` section divider, add:

```python
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
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/test_tools.py::test_get_current_time_uses_default_timezone -v`

- [ ] **Step 5: Commit**

```bash
git add src/tools.py tests/test_tools.py
git commit -m "feat(tools): add get_current_time with server default timezone"
```

---

## Task 3: `get_current_time` — custom timezone + invalid-TZ error

**Files:**
- Modify: `tests/test_tools.py` (append 2 tests)

(No code change needed — Task 2 already accepts the `timezone` parameter.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tools.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect PASS immediately**

Task 2's implementation already accepts `timezone` and re-raises `ZoneInfoNotFoundError` as `ValueError`. These tests exist as regression coverage — the TDD fail-first principle is intentionally bent here because the behavior is already in place. If either fails, it means Task 2 was implemented incorrectly.

Run: `python -m pytest tests/test_tools.py::test_get_current_time_respects_custom_timezone tests/test_tools.py::test_get_current_time_invalid_tz_via_executor -v`

If either fails, inspect Task 2 implementation; common gotcha is forgetting the `ZoneInfoNotFoundError` → `ValueError` re-raise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools.py
git commit -m "test(tools): cover custom and invalid timezone for get_current_time"
```

---

## Task 4: Add `pypdf` dependency + `read_attached_document` text-file happy path

**Files:**
- Modify: `requirements.txt` (add `pypdf>=4.0`)
- Modify: `src/tools.py` (add new tool + `_fetch_slack_file` + `_parse_text` helpers)
- Modify: `tests/test_tools.py` (text-file test)

- [ ] **Step 1: Add `pypdf` to requirements.txt**

Append to `requirements.txt`:

```
pypdf>=4.0,<6.0
```

Install locally: `pip install pypdf`

- [ ] **Step 2: Write the failing test for a text/plain attachment**

Append to `tests/test_tools.py`:

```python
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
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `python -m pytest tests/test_tools.py::test_read_attached_document_text_file -v`

Expected: `ImportError`.

- [ ] **Step 4: Implement `read_attached_document` (text path only for now)**

In `src/tools.py`, at the top with other constants add:

```python
DOC_TEXT_PREFIX = "text/"
DOC_PDF_MIME = "application/pdf"
```

After `_filename_from_url` (~line 232) but before the `fetch_thread_history` tool, add helpers:

```python
def _fetch_slack_file(url: str, token: str, max_bytes: int) -> tuple[bytes, str]:
    """Fetch a Slack file with size guard. Returns (body, mimetype_from_header).

    Raises:
      ValueError: on disallowed host, oversize via Content-Length, or
                  oversize discovered while streaming the body.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in SLACK_FILE_HOSTS:
        raise ValueError("invalid Slack file download URL")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as response:  # noqa: S310
        content_length = response.headers.get("Content-Length") if response.headers else None
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError(f"document exceeds MAX_DOC_BYTES={max_bytes}")
        # Stream-read with running counter (covers missing-Content-Length case)
        chunks: list[bytes] = []
        read = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            read += len(chunk)
            if read > max_bytes:
                raise ValueError(f"document exceeds MAX_DOC_BYTES={max_bytes}")
            chunks.append(chunk)
        mime = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower() if response.headers else ""
    return b"".join(chunks), mime


def _parse_text(data: bytes, max_chars: int) -> tuple[str, bool]:
    text = data.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated
```

Add the tool (before the existing `# Helpers` divider — keep it grouped with the other built-ins):

```python
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
```

- [ ] **Step 5: Run test — expect PASS**

Run: `python -m pytest tests/test_tools.py::test_read_attached_document_text_file -v`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/tools.py tests/test_tools.py
git commit -m "feat(tools): add read_attached_document with text-file path"
```

---

## Task 5: PDF happy path + page cap + char cap

**Files:**
- Modify: `src/tools.py` (implement `_parse_pdf` helper + wire into tool)
- Modify: `tests/test_tools.py` (3 tests + reusable fixture helper)

- [ ] **Step 1: Add `reportlab` to `requirements-dev.txt` and install**

Append to `requirements-dev.txt`:

```
reportlab>=4.0,<5.0
```

Install: `pip install reportlab`

Rationale: `pypdf` has no public text-drawing API (only raw page/stream manipulation via private methods, which is fragile across `pypdf` 4.x/5.x minor versions). `reportlab` is the standard Python PDF generator, test-only, and its output is extractable by `pypdf.PdfReader.extract_text()` reliably.

- [ ] **Step 2: Add PDF-builder helper in `tests/test_tools.py`**

Below the existing imports add:

```python
def _build_pdf_bytes(pages_text: list[str]) -> bytes:
    """Build a minimal PDF (one page per string) using reportlab. Test-only."""
    from io import BytesIO
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import letter

    buf = BytesIO()
    canvas = Canvas(buf, pagesize=letter)
    for text in pages_text:
        canvas.drawString(72, 720, text)
        canvas.showPage()
    canvas.save()
    return buf.getvalue()
```

- [ ] **Step 3: Write 3 failing tests**

Append to `tests/test_tools.py`:

```python
def _mock_pdf_response(opener, body: bytes, headers=None):
    """Wire the urlopen mock to stream `body` in chunks through `_fetch_slack_file`."""
    resp = opener.return_value.__enter__.return_value
    buf = {"pos": 0}

    def _chunked(n=-1):
        if n == -1:
            remaining = body[buf["pos"]:]
            buf["pos"] = len(body)
            return remaining
        chunk = body[buf["pos"]:buf["pos"] + n]
        buf["pos"] += len(chunk)
        return chunk

    resp.read.side_effect = _chunked
    resp.headers = dict(headers or {"Content-Length": str(len(body)), "Content-Type": "application/pdf"})


def test_read_attached_document_pdf_happy_path():
    from src.tools import read_attached_document

    pdf = _build_pdf_bytes(["Hello PDF page one.", "Page two here."])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "name": "report.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "report.pdf"
    assert entry["pages"] == 2
    assert entry["truncated"] is False
    assert entry["chars"] > 0


def test_read_attached_document_pdf_truncation():
    from src.tools import read_attached_document

    # Long text → forced truncation
    pdf = _build_pdf_bytes(["A" * 500])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/big.pdf",
                "name": "big.pdf",
            }
        ]
    }
    ctx = _ctx(
        event=event,
    )
    ctx = ToolContext(
        slack_client=ctx.slack_client,
        channel=ctx.channel,
        thread_ts=ctx.thread_ts,
        event=ctx.event,
        settings=_settings(max_doc_chars=50),
        llm=ctx.llm,
    )
    with patch("src.tools.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert out[0]["truncated"] is True
    assert out[0]["chars"] == 50


def test_read_attached_document_page_cap():
    from src.tools import read_attached_document

    pdf = _build_pdf_bytes(["p1", "p2", "p3"])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/pages.pdf",
                "name": "pages.pdf",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_pages=2),
        llm=MagicMock(),
    )
    with patch("src.tools.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_PAGES" in out[0]["error"]
```

Note on `_settings()`: the four new `Settings` fields (`default_timezone`, `max_doc_*`) all have dataclass defaults, so `_settings()` does NOT need to list them in its `base` dict — `**overrides` already forwards them when specified. Do not touch `_settings` unless a test actually breaks.

- [ ] **Step 4: Run tests — expect FAIL** (PDF branch returns "not implemented yet")

Run: `python -m pytest tests/test_tools.py -k pdf -v`

- [ ] **Step 5: Implement `_parse_pdf` helper**

In `src/tools.py`, after `_parse_text`:

```python
def _parse_pdf(
    data: bytes,
    max_pages: int,
    max_chars: int,
) -> tuple[str, int, bool]:
    """Extract text from a PDF. Raises ValueError for recoverable issues so the
    caller can emit a per-document error entry."""
    from io import BytesIO

    # Deferred import keeps pypdf out of cold-start for requests that never
    # touch this tool.
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError, DependencyError

    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise ValueError(f"PdfReadError: {exc}") from exc
    if reader.is_encrypted:
        raise ValueError("encrypted PDF not supported")
    page_count = len(reader.pages)
    if page_count > max_pages:
        raise ValueError(f"document exceeds MAX_DOC_PAGES={max_pages}")
    pieces: list[str] = []
    total = 0
    truncated = False
    for page in reader.pages:
        try:
            piece = page.extract_text() or ""
        except (PdfReadError, DependencyError) as exc:
            raise ValueError(f"PdfReadError: {exc}") from exc
        pieces.append(piece)
        total += len(piece)
        if total >= max_chars:
            truncated = True
            break
    text = "\n".join(pieces)
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, page_count, truncated
```

Replace the `if mime == DOC_PDF_MIME:` branch in `read_attached_document` (currently "not implemented yet") with:

```python
        if mime == DOC_PDF_MIME:
            try:
                text, pages, truncated = _parse_pdf(
                    body, ctx.settings.max_doc_pages, max_chars
                )
            except ValueError as exc:
                out.append({"name": name, "error": str(exc)})
                return
            out.append(
                {
                    "name": name,
                    "mimetype": DOC_PDF_MIME,
                    "pages": pages,
                    "chars": len(text),
                    "truncated": truncated,
                    "text": text,
                }
            )
            return
```

- [ ] **Step 6: Run tests — expect all 3 PASS**

Run: `python -m pytest tests/test_tools.py -k "read_attached_document" -v`

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt src/tools.py tests/test_tools.py
git commit -m "feat(tools): parse PDFs in read_attached_document with page/char caps"
```

---

## Task 6: Size cap — Content-Length and streamed-read paths

**Files:**
- Modify: `tests/test_tools.py` (2 tests)

(No code change — Task 4's `_fetch_slack_file` already implements both guards; these tests verify them.)

- [ ] **Step 1: Write 2 failing-or-passing tests**

Append to `tests/test_tools.py`:

```python
def test_read_attached_document_size_cap_via_content_length():
    from src.tools import read_attached_document

    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/huge.txt",
                "name": "huge.txt",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_bytes=100),  # tiny cap
        llm=MagicMock(),
    )
    with patch("src.tools.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.headers = {"Content-Length": "200"}  # > cap
        resp.read.return_value = b"x" * 10  # should never be read past cap
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_BYTES" in out[0]["error"]


def test_read_attached_document_size_cap_via_streamed_read():
    from src.tools import read_attached_document

    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/nohead.txt",
                "name": "nohead.txt",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_bytes=100),
        llm=MagicMock(),
    )
    body = b"y" * 200
    with patch("src.tools.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.headers = {}  # no Content-Length
        buf = {"pos": 0}

        def _chunked(n=-1):
            if n == -1:
                remaining = body[buf["pos"]:]
                buf["pos"] = len(body)
                return remaining
            chunk = body[buf["pos"]:buf["pos"] + n]
            buf["pos"] += len(chunk)
            return chunk

        resp.read.side_effect = _chunked
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_BYTES" in out[0]["error"]
```

- [ ] **Step 2: Run tests — both PASS**

Run: `python -m pytest tests/test_tools.py -k size_cap -v`

If either fails, the issue is in `_fetch_slack_file` — check both the Content-Length pre-check and the running-counter in the chunked read loop.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools.py
git commit -m "test(tools): cover size cap paths in read_attached_document"
```

---

## Task 7: SSRF, encrypted PDF, non-doc MIME, and HTTP error tests

**Files:**
- Modify: `tests/test_tools.py` (4 tests)

Code in Tasks 4 and 5 already handles these; this task is pure test coverage + verification.

- [ ] **Step 1: Write failing (or immediately-passing) tests**

Append to `tests/test_tools.py`:

```python
def test_read_attached_document_rejects_non_slack_host():
    from src.tools import read_attached_document

    ctx = _ctx()
    out = read_attached_document(
        ctx, urls=["https://evil.example.com/foo.pdf"], limit=1
    )
    assert len(out) == 1
    assert "error" in out[0]
    assert "invalid" in out[0]["error"].lower()


def test_read_attached_document_skips_encrypted_pdf():
    from src.tools import read_attached_document
    from io import BytesIO
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # NOTE: pypdf>=4.0 uses keyword-only user_password. If requirements.txt's
    # upper pin is ever relaxed past 6.0, verify this signature still holds.
    writer.encrypt(user_password="secret")
    buf = BytesIO()
    writer.write(buf)
    encrypted_pdf = buf.getvalue()

    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/enc.pdf",
                "name": "enc.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, encrypted_pdf)
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "encrypted" in out[0]["error"]


def test_read_attached_document_skips_image_mime():
    from src.tools import read_attached_document

    event = {
        "files": [
            {
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/a.png",
                "name": "a.png",
            }
        ]
    }
    ctx = _ctx(event=event)
    # urlopen should NOT be called — image MIMEs are filtered before fetch
    with patch("src.tools.urllib.request.urlopen") as opener:
        out = read_attached_document(ctx, limit=1)
    opener.assert_not_called()
    assert out == []


def test_read_attached_document_http_error_returns_per_item():
    from src.tools import read_attached_document
    import urllib.error

    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/missing.pdf",
                "name": "missing.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.urllib.request.urlopen") as opener:
        opener.side_effect = urllib.error.HTTPError(
            url="https://files.slack.com/missing.pdf",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    assert "error" in out[0]
    assert "404" in out[0]["error"]
```

- [ ] **Step 2: Run all 4 tests**

Run: `python -m pytest tests/test_tools.py -k "rejects_non_slack_host or encrypted or image_mime or http_error" -v`

All should pass. If `test_read_attached_document_rejects_non_slack_host` fails because `_fetch_slack_file` raises `ValueError` before we wrap it into per-item, verify the `except ValueError as exc` block in `_process` (Task 4 code) converts it to a per-item entry.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools.py
git commit -m "test(tools): cover SSRF/encrypted/HTTP-error paths for documents"
```

---

## Task 8: Registry discoverability test

**Files:**
- Modify: `tests/test_tools.py:53-57` (extend existing `test_default_registry_has_expected_tools`)

- [ ] **Step 1: Update the registry assertion**

Change:

```python
def test_default_registry_has_expected_tools():
    names = set(default_registry.names())
    assert {"read_attached_images", "fetch_thread_history", "search_web", "generate_image"}.issubset(names)
```

to:

```python
def test_default_registry_has_expected_tools():
    names = set(default_registry.names())
    assert {
        "read_attached_images",
        "fetch_thread_history",
        "search_web",
        "generate_image",
        "get_current_time",
        "read_attached_document",
    }.issubset(names)
```

- [ ] **Step 2: Run test — expect PASS**

Run: `python -m pytest tests/test_tools.py::test_default_registry_has_expected_tools -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools.py
git commit -m "test(tools): assert new tools are registered"
```

---

## Task 9: Documentation — `.env.example` and `README.md`

**Files:**
- Modify: `.env.example` (add 4 new vars in appropriate sections)
- Modify: `README.md:26-29` (add 2 tool bullets); also `README.md:39-65` env-var table (add 4 rows)

- [ ] **Step 1: Update `.env.example`**

Add a new section or inline with existing `Agent behavior`:

```
# --- Document / time ---
DEFAULT_TIMEZONE="Asia/Seoul"    # IANA TZ for get_current_time default
MAX_DOC_CHARS=20000              # per-document extracted-text cap (>=1000)
MAX_DOC_PAGES=50                 # per-document PDF page cap (>=1)
MAX_DOC_BYTES=26214400           # per-document download cap, 25MB (>=65536)
```

- [ ] **Step 2: Update `README.md` tool bullets (lines 26-29)**

Add after `generate_image`:

```
  - `get_current_time` — 서버 기본 TZ(또는 `timezone` 인자) 로 현재 시각/요일 반환
  - `read_attached_document` — 첨부 PDF/텍스트 파일 추출 (페이지·바이트·문자 상한 적용)
```

- [ ] **Step 3: Update `README.md` env-var table**

Add 4 rows (slot them between `MAX_HISTORY_CHARS` and `BOT_CURSOR` so the "behavior" vars stay grouped):

```
| `DEFAULT_TIMEZONE` | | `Asia/Seoul` | `get_current_time` 기본 TZ (IANA). 잘못된 이름이면 기본값으로 폴백 + 경고 |
| `MAX_DOC_CHARS` | | `20000` | `read_attached_document` 추출 텍스트 최대 문자수 (≥1000) |
| `MAX_DOC_PAGES` | | `50` | `read_attached_document` PDF 최대 페이지수 (≥1) |
| `MAX_DOC_BYTES` | | `26214400` | `read_attached_document` 다운로드 최대 바이트 (기본 25MB, ≥65536) |
```

- [ ] **Step 4: Check for hard-coded tool counts**

Use the Grep tool: search `README.md` and `CLAUDE.md` for patterns `four tools|네 개의 tool|4 tool|four built-in`. If any match, bump to 6 / 여섯. If none, skip. (The current repo has none — this step is a safety net in case the spec doc got out of sync with the source.)

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md
git commit -m "docs: describe get_current_time and read_attached_document"
```

---

## Task 10: Full validation + coverage

**Files:** none (pure verification)

- [ ] **Step 1: Run the full test suite with coverage**

Run: `python -m pytest --cov=src --cov-report=term-missing`

Expected: all tests pass; `src/tools.py` coverage should remain ≥80% (add tests if it drops).

- [ ] **Step 2: Run `localtest.py` sanity checks**

```bash
python localtest.py "지금 몇 시야?"
# expect a response that invokes get_current_time and answers with Asia/Seoul time

python localtest.py --quiet-steps "오늘 무슨 요일이야?"
# expect weekday in Korean or English depending on RESPONSE_LANGUAGE
```

A Slack bot token is not required for these checks — the agent loop runs against the configured LLM provider, and `get_current_time` makes no Slack API call. You DO still need a working `LLM_PROVIDER` + API key (e.g. `OPENAI_API_KEY` or `XAI_API_KEY`) in `.env.local`.

- [ ] **Step 3: Commit (only if anything changed)**

```bash
git status
# if nothing changed, skip
# otherwise:
git add .
git commit -m "chore: final polish after validation run"
```

- [ ] **Step 4: Verify branch is clean**

```bash
git status  # expect "nothing to commit, working tree clean"
git log --oneline -10  # expect 7-9 new commits from this plan
```

---

## Acceptance Criteria (re-stated from spec)

- [ ] `python -m pytest` passes with every new test (≥12 added here) green.
- [ ] `pytest --cov=src` shows ≥80% coverage (project-wide), no regression on `src/tools.py`.
- [ ] `python localtest.py "지금 몇 시"` exercises `get_current_time` end-to-end.
- [ ] `.env.example`, `README.md` reflect the new tools and env vars.
- [ ] No changes to `app.py`, `src/agent.py`, `src/llm.py` — the two tools integrate purely via the existing `@tool` registry plumbing.
- [ ] Deferred imports (`pypdf`, `zoneinfo`) inside tool bodies — never at module top of `tools.py` — so cold-start for non-document/non-time requests is unchanged.

## Rollback Plan

If something regresses after deploy:
1. `git revert <commit-range>` for the commits produced by Tasks 4-7 (tool impl) — keeps config scaffolding, drops behavior.
2. Alternatively, remove `@tool(...)` decorator from the two new functions to unregister without reverting code; agent falls back to the four original tools.
