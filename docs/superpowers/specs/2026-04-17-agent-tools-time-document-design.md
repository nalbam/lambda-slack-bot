# Design: Add `get_current_time` and `read_attached_document` Agent Tools

Date: 2026-04-17
Status: Draft (awaiting review)
Author: Claude Code (brainstorm session with @nalbam)

## Problem

The agent's tool registry currently exposes four tools: `read_attached_images`, `fetch_thread_history`, `search_web`, `generate_image`. Two common question classes fail without native tool support:

1. **Time-sensitive queries** — "오늘 몇 일?", "지금 뉴욕 시간", "무슨 요일" — the LLM answers from training data and is often wrong.
2. **Document attachments** — users drop PDFs / text files into a Slack thread expecting the bot to summarize or answer. Today those attachments are ignored (image-only path).

## Goals

- Add `get_current_time` and `read_attached_document` tools following the existing `@tool`-decorator + `ToolRegistry` pattern.
- Preserve every agent-pipeline invariant documented in `CLAUDE.md` (native function calling, no keyword heuristics, host allowlist for downloads).
- Keep Lambda cold-start and bundle impact minimal (pure-Python deps only).
- Maintain 80%+ test coverage; add unit tests for every decision point.

## Non-Goals

- `.docx` / `.xlsx` support (deferred; path B/C from brainstorm).
- OCR for scanned PDFs.
- Arbitrary-URL document fetch (Slack host allowlist only, consistent with `read_attached_images`).
- Per-page chunked return (rejected in favor of single truncated blob — simpler for the current 4-tool agent loop).

## Decisions (from brainstorm, 2026-04-17)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | PDF + text/* only; dependency `pypdf>=4.0` | Pure Python, <1MB, smallest bundle/coldstart impact. Matches project's x86_64 pure-pip philosophy. |
| 2 | Hard page cap + text length cap + `truncated` flag | Partial extraction is useful for Q&A; simpler than per-page chunking; prevents parse/token runaway. |
| 3 | `DEFAULT_TIMEZONE` env + optional `timezone` param (IANA) | Fastest path for the majority-Korean user base; `zoneinfo` stdlib keeps implementation a one-liner. |

## Architecture

### Components Touched

- `src/tools.py` — two new `@tool`-decorated functions + small helpers. Estimated +150 lines (file goes from ~400 to ~550; under the 800-line hard cap but nearing the 500-line refactor-suggest threshold — future-PR candidate to split into a `tools/` package, out of scope here).
- `src/config.py` — four new env-backed fields on `Settings`:
  - `default_timezone: str = "Asia/Seoul"` (env `DEFAULT_TIMEZONE`)
  - `max_doc_chars: int = 20_000` (env `MAX_DOC_CHARS`, clamps `>= 1000`). Note: `MAX_OUTPUT_TOKENS=4096` (default) gives the LLM roughly 12–16 KB of response budget; the tool's 20 KB text cap is intentionally slightly larger than that so the LLM sees more context than it can quote verbatim, forcing it to summarize. Operators lowering `MAX_OUTPUT_TOKENS` should lower `MAX_DOC_CHARS` proportionally.
  - `max_doc_pages: int = 50` (env `MAX_DOC_PAGES`, clamps `>= 1`)
  - `max_doc_bytes: int = 25 * 1024 * 1024` (env `MAX_DOC_BYTES`, clamps `>= 64 * 1024`)
  - Invalid values fall back to defaults with a warning (matches existing `LLM_PROVIDER` / `AGENT_MAX_STEPS` pattern). Invalid `DEFAULT_TIMEZONE` also falls back to `Asia/Seoul` with a warning.
- `requirements.txt` — add `pypdf>=4.0`.
- `.env.example` — document the four new vars.
- `README.md` — update tool list (four → six) and the "Built-in tools" section.
- `CLAUDE.md` — planner should `grep` for "4 tool" / "four tools" first; only edit if a matching reference exists (current file mentions the registry but not a fixed tool count).
- `tests/test_tools.py` — 11 new tests.
- `tests/test_config.py` — 1 new test for TZ fallback.

### Tool #1: `get_current_time`

**LLM schema**

```json
{
  "name": "get_current_time",
  "description": "Return the current wall-clock time. Uses the server default timezone unless 'timezone' is provided. Useful for 'today', 'now', 'this week', weekday questions.",
  "parameters": {
    "type": "object",
    "properties": {
      "timezone": {
        "type": "string",
        "description": "Optional IANA timezone name (e.g. 'Asia/Seoul', 'UTC', 'America/New_York'). Omit to use the server default."
      }
    },
    "required": []
  }
}
```

**Implementation sketch**

```python
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

def get_current_time(ctx: ToolContext, timezone: str | None = None) -> dict:
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

**Return shape** — `{iso, timezone, weekday, unix}`. Example:
```json
{"iso": "2026-04-17T22:45:10+09:00", "timezone": "Asia/Seoul", "weekday": "Friday", "unix": 1744903510}
```

**Error handling** — invalid TZ → `ValueError` (already in executor's catch list → `{"ok": false, "error": "ValueError: unknown timezone: X"}`).

**Timeout** — default executor timeout (20s) is far more than enough; no custom timeout needed.

### Tool #2: `read_attached_document`

**LLM schema**

```json
{
  "name": "read_attached_document",
  "description": "Read PDF or text files attached to the current Slack mention (and optionally extra URLs on files*.slack.com) and return extracted text. Images are skipped; use read_attached_images for those.",
  "parameters": {
    "type": "object",
    "properties": {
      "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
      "urls": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Additional Slack file URLs to read (must be on files*.slack.com)."
      }
    },
    "required": []
  }
}
```

**Flow**

```
ctx.event.files → filter (pdf | text/*) → dedupe → enforce limit
  + additional urls (Slack host allowlist)
  → for each file:
      1. HEAD-less size check via streamed read with running byte counter
      2. Content-Length pre-check if header present
      3. If bytes > MAX_DOC_BYTES: abort, emit error entry
      4. MIME == application/pdf:
         - PdfReader(BytesIO)
         - if reader.is_encrypted: skip with error entry
         - if len(reader.pages) > MAX_DOC_PAGES: skip with error entry
         - concat page.extract_text() across pages
      5. MIME startswith text/:
         - data.decode("utf-8", errors="replace")
      6. Apply MAX_DOC_CHARS cap; set truncated flag
  → list of {name, mimetype, pages, chars, truncated, text} OR {name, error}
```

**Return shape (per document)**

```json
{
  "name": "report.pdf",
  "mimetype": "application/pdf",
  "pages": 34,
  "chars": 18420,
  "truncated": false,
  "text": "..."
}
```

Error entry (corrupted, encrypted, oversize, etc.):

```json
{"name": "huge.pdf", "error": "document exceeds MAX_DOC_PAGES=50"}
```

The tool returns a list so multiple attachments in one message all get processed in a single tool-call (mirrors `read_attached_images`).

**SSRF defense** — reuse `SLACK_FILE_HOSTS`; validate scheme + hostname before every fetch; `Authorization: Bearer <SLACK_BOT_TOKEN>` header.

**Size guard** — two layers:
1. `Content-Length` header (if present) checked before reading the body.
2. Streamed read with a running counter; if it exceeds `MAX_DOC_BYTES`, close the response and raise `ValueError("document exceeds MAX_DOC_BYTES")`.

**Page guard** — after constructing `PdfReader`, check `len(reader.pages) <= MAX_DOC_PAGES` before extracting text.

**Char cap** — accumulate extracted text per page; when the running length hits `MAX_DOC_CHARS`, slice and set `truncated=True`; stop parsing further pages.

**pypdf exceptions** — `pypdf.errors.PdfReadError` and `pypdf.errors.DependencyError` are not in `ToolExecutor`'s catch list. Catch them inside the tool function and re-raise as `ValueError` (which the executor already handles uniformly). This avoids widening the executor's exception net.

**Timeout** — `timeout=30.0` on the `@tool` decorator. pypdf parsing of a 50-page PDF fits comfortably under 30s in the 5120MB Lambda; adds headroom for network download.

### Data Flow (end-to-end)

```
user mention (files: [report.pdf])
  ↓
app.lambda_handler → _process → ToolContext built
  ↓
Agent.run → LLMProvider.chat(tools=registry.specs())
  ↓
LLM returns tool_calls=[{name:"read_attached_document", arguments:{}}]
  ↓
ToolExecutor.execute (30s timeout, ThreadPoolExecutor)
  ↓
read_attached_document:
  - scans ctx.event.files → finds report.pdf
  - urlopen(files.slack.com URL, Bearer token)
  - pypdf extract → char cap → truncated=false
  ↓
tool result JSON → appended to messages
  ↓
Agent next hop → LLMProvider.chat (no tools this time)
  ↓
LLM composes final answer → Slack via streaming
```

### Error Policy (rule for planner)

**Single rule:** per-document errors (size cap, page cap, encrypted, non-Slack host, corrupt PDF) → produce a `{"name": ..., "error": ...}` entry in the returned list; the tool call as a whole still succeeds (`ok: true`). Only tool-fatal failures (the entire HTTP fetch path throws `URLError` / `SlackApiError`, or `TypeError` on bad args) propagate to the executor, which wraps them as `{ok: false, error: ...}`.

This keeps the LLM's next hop informative ("one of your two PDFs was encrypted") rather than losing partial results.

### Error Handling Matrix

| Failure | Tool behaves how | User-visible impact |
|---|---|---|
| No attachments + empty `urls` | Returns `[]` | LLM sees empty result, replies accordingly |
| Non-Slack host in `urls` | Per-item `{"error": "invalid Slack file download URL"}` | LLM is told which URL was rejected, continues with other docs |
| HTTP 403/404 | Per-item `{"error": "HTTPError: 404"}` (caught inside `_fetch`) | LLM told which document failed, other docs still processed |
| Network error before any fetch starts (e.g., DNS failure on the whole batch) | `URLError` propagates → executor returns `{ok: false}` tool-wide | LLM apologizes / retries |
| Content-Length > `MAX_DOC_BYTES` | Per-item `{"error": "document exceeds MAX_DOC_BYTES=..."}` | LLM asks user to shorten or upload only excerpts |
| Streamed read > `MAX_DOC_BYTES` | Same | Same |
| `pages > MAX_DOC_PAGES` | Per-item `{"error": "document exceeds MAX_DOC_PAGES=..."}` | LLM asks user for specific pages |
| Encrypted PDF | Per-item `{"error": "encrypted PDF not supported"}` | LLM tells user to remove password |
| `PdfReadError` (corrupt) | Caught, re-raised as `ValueError` → executor returns `{"ok": false}` | LLM says file unreadable |
| Unknown TZ in `get_current_time` | `ValueError("unknown timezone: X")` → executor `{"ok": false}` | LLM retries with valid TZ or apologizes |
| Invalid `DEFAULT_TIMEZONE` env | Settings logs warning, falls back to `Asia/Seoul` | Lambda still starts |

### Testing Plan

Target: >= 80% coverage (project standard). 12 new tests total.

**`tests/test_tools.py`** (11 tests):

1. `test_get_current_time_default_tz` — `settings.default_timezone="Asia/Seoul"`, no arg → `timezone == "Asia/Seoul"`, `iso` ends with `+09:00`.
2. `test_get_current_time_custom_tz` — arg `"UTC"` → `timezone=="UTC"`, iso ends with `+00:00`.
3. `test_get_current_time_invalid_tz` — arg `"Narnia/Center"` → `ToolExecutor.execute` returns `{ok: false, error: "ValueError: unknown timezone: ..."}`.
4. `test_read_attached_document_pdf_happy_path` — build a 2-page PDF in memory with `pypdf.PdfWriter`, mock `urlopen` with `BytesIO`, fake `ctx.event.files`, assert result has `pages=2`, `truncated=False`, text contains expected fixture strings.
5. `test_read_attached_document_text_file` — MIME `text/plain`, fixture content, asserts decoded text returned.
6. `test_read_attached_document_truncation` — `MAX_DOC_CHARS=50`, fixture PDF with >50 chars, asserts `truncated=True`, `len(text)==50`.
7. `test_read_attached_document_page_cap` — `MAX_DOC_PAGES=1`, 2-page fixture PDF, asserts per-item `error` entry.
8a. `test_read_attached_document_size_cap_content_length` — `MAX_DOC_BYTES=100`, mock response with `Content-Length: 200` header (body may be shorter), asserts per-item `error` entry and that the body is not read past the cap.
8b. `test_read_attached_document_size_cap_streamed` — `MAX_DOC_BYTES=100`, mock response with no `Content-Length` header and a 200-byte body, asserts per-item `error` entry triggered by the running byte counter during streaming.
9. `test_read_attached_document_rejects_non_slack_host` — `urls=["https://evil.example/foo.pdf"]` → per-item `error` OR `ValueError` handled uniformly.
10. `test_read_attached_document_skips_encrypted_pdf` — generate encrypted PDF via pypdf, assert `error` entry.
11. `test_read_attached_document_skips_non_doc_mime` — `ctx.event.files` contains an `image/png` — tool returns `[]` (image handled by the other tool).

**`tests/test_config.py`** (1 test):

12. `test_default_timezone_fallback_on_invalid_env` — `DEFAULT_TIMEZONE="Narnia/Center"` → `settings.default_timezone == "Asia/Seoul"` + warning logged.

All tests use `unittest.mock.patch` on `src.tools.urllib.request.urlopen` (same pattern as existing image/web tool tests). No network, no real Slack API.

### Dependency & Bundle Impact

| Change | Size | Cold-start impact |
|---|---|---|
| `pypdf>=4.0` added to `requirements.txt` | ~1MB unzipped | Import cost ~30ms (deferred: only imported inside `read_attached_document`, not at module top) |

Deferred import pattern: `from pypdf import PdfReader` lives inside the function body, not at the top of `tools.py`. This keeps `app.lambda_handler` cold-start identical for requests that never trigger the document tool.

### Security Review Pre-check

- [x] No new secrets in code/logs.
- [x] Input validation: `timezone` via `zoneinfo.ZoneInfo` (raises on bad input); `urls` scheme+host checked before fetch; `limit` bounded by JSON schema.
- [x] SSRF prevention: Slack host allowlist reused.
- [x] Authorization: Slack bot token used only for downloads to allowlisted hosts.
- [x] Rate limiting: per-call via executor timeout + agent `max_steps` bound.
- [x] No sensitive data logged (filenames and sizes only, not text content).

### Invariants Maintained (CLAUDE.md)

- [x] Native function calling preserved — no JSON-in-prompt.
- [x] Tool registered via `@tool` decorator (single source of truth for dispatch + LLM schema). `ToolRegistry.specs()` auto-updates.
- [x] `app.py` untouched; orchestration stays in the agent loop.
- [x] Slack host allowlist reused, not widened.
- [x] Deferred `pypdf` import to keep cold-start on non-document requests untouched.

### Rollout

1. Single PR, squash-merged to `main`.
2. No feature flag — tools activate immediately after deploy.
3. Agent `max_steps=3` default means the LLM has room to call `read_attached_document` → compose without hitting the forced-compose fallback.
4. Observability: `log_event(logger, "tool.read_attached_document", files=N, chars=M, truncated=bool)` emits into existing structured-log stream; no schema change.

## Open Questions / Follow-ups (post-merge)

- If bundle growth becomes a concern over time, migrate to `tools/` package split (`time.py`, `documents.py`, `images.py`, `web.py`, `generation.py`) — noted but explicitly out of scope for this change.
- `fetch_url` (general-web content reader) is a strong complement to `search_web`; not part of this PR but the natural next tool per brainstorm.
- OCR fallback (Path C from brainstorm) becomes relevant if users start attaching scanned contracts; re-evaluate based on real usage logs.

## Acceptance Criteria

- [ ] `pytest` passes with >=80% coverage, all 12 new tests green.
- [ ] `python localtest.py "지금 몇 시야?"` returns current Korean time via the tool (verifiable in step logs).
- [ ] A PDF attached in a Slack thread is summarized by the bot (manual test after deploy).
- [ ] No regression in existing tool tests.
- [ ] `.env.example`, `README.md`, `CLAUDE.md` reflect the new tools and env vars.
