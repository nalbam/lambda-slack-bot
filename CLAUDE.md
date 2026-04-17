# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env.local   # fill in values

# Local CLI runner (no Slack connection needed)
python localtest.py "질문"
python localtest.py --stream "질문"
python localtest.py                  # interactive stdin

# Tests
python -m pytest
python -m pytest --cov=src --cov-report=term-missing
python -m pytest tests/test_agent.py::test_agent_runs_tool_then_returns_text -v

# Deploy (requires IAM OIDC role `lambda-slack-bot`)
npm i -g serverless@3
npm i serverless-python-requirements serverless-dotenv-plugin
serverless deploy --stage dev --region us-east-1
```

Lambda entrypoint: `app.lambda_handler`. Slack events land at `POST /slack/events` via API Gateway.

## Architecture — the non-obvious parts

### Agent loop uses NATIVE function calling, not JSON prompting

`src/agent.py` passes `registry.specs()` directly to `LLMProvider.chat(tools=...)`. The provider (`src/llm.py`) translates that to OpenAI `tools=[{type:"function",function:{...}}]` or Bedrock `tools=[{name, description, input_schema}]` (Claude) / `toolConfig` (Nova). There is **no JSON-in-prompt parsing** — tool calls arrive as structured objects. Loop terminates when `stop_reason != "tool_use"` or `max_steps` hit. On max_steps, a forced compose step (`_compose_without_tools`) runs with `tools=None`.

Duplicate tool-call suppression: `_call_signature` = `name + sha1(args_json)`. A repeated signature within the loop is short-circuited with `{"ok": False, "error": "duplicate call skipped"}` and handed back to the LLM so it can move on.

### Two LLM provider families, one Protocol

`LLMProvider` is a Protocol implemented by `OpenAIProvider` and `BedrockProvider`. Bedrock routes internally on model family prefix:
- `anthropic.claude*` → `invoke_model` with Messages API shape, `content[].type=="tool_use"` parsing.
- `amazon.nova*` → `converse` / `converse_stream` with `toolConfig` + `output.message.content[].toolUse`.
- Unknown → Claude path without tools.

`_to_anthropic_messages` / `_to_nova_messages` translate our canonical role/tool_calls/tool messages to each backend's shape. `tool` role becomes an Anthropic `tool_result` content block inside a user message; Nova becomes a `toolResult` content block.

Image generation is family-routed too: Titan/Nova-Canvas use `TEXT_IMAGE` task; Stability uses `text_prompts`. See `_build_image_body`.

`_CompositeProvider` wraps two providers when text and image providers differ (e.g., OpenAI text + Bedrock image).

### Slack retry → DynamoDB conditional put dedup

`lambda_handler` short-circuits when `X-Slack-Retry-Num` header is present (returns 200 OK). Even without the retry header, the first line of `_process()` is `DedupStore.reserve(f"dedup:{client_msg_id}")` which does `put_item(ConditionExpression="attribute_not_exists(id)")`. Duplicate key raises `ConditionalCheckFailedException` → False → silent return. This is the only race-safe dedup (get-then-put has a window). TTL 1h via `expire_at`.

### Single table, two key prefixes

`DYNAMODB_TABLE_NAME` stores both dedup reservations (`dedup:{msg_id}`) and thread conversation memory (`ctx:{thread_ts}`). GSI `user-index` (hash `user`, range `expire_at`) backs per-user throttle via `count_user_active(user)`. `ConversationStore.put` trims with `truncate_to_chars(messages, max_chars)` (drop oldest until serialized size fits).

### Message splitting is code-fence-aware

`MessageFormatter.split_message` (in `src/slack_helpers.py`) splits on `\`\`\`` first (so complete code blocks survive), then on `\n\n`, then on `.!?` sentence boundaries, then hard slice. `_merge_small` rejoins adjacent small chunks up to `max_len`. First chunk goes via `chat_update` on the placeholder message; the rest via `chat_postMessage(thread_ts=…)`. If `chat_update` fails (`msg_too_long` etc.), that chunk falls back to a new message.

### Config is lazy, not import-time

`Settings.from_env()` runs at module load but does NOT validate Slack credentials. `Settings.require_slack_credentials()` is called from `_get_bolt_app()` so the first request fails cleanly but tests and tooling can import `app` without `SLACK_BOT_TOKEN`. The old `RuntimeError` at module top is gone.

Enum/int validation quietly falls back to defaults with a warning: invalid `LLM_PROVIDER=mystery` → `openai`, `AGENT_MAX_STEPS=not-int` → `3`, below-minimum values clamp up.

### Streaming only on the final step

The agent streams only after all tool calls resolve (via `stream_chat` in `_compose_without_tools`). Mid-loop LLM calls are non-streaming to keep `tool_calls` parsing simple. `throttled(fn, 0.6)` in `app.py` coalesces stream deltas into ~0.6s `chat_update` intervals on the placeholder so Slack isn't hammered.

### Structured logging with request_id

`src/logging_utils.py` installs a JSON handler on root. `set_request_id(uuid)` is called at the start of each `_process`. `log_event(logger, "agent.done", steps=..., tokens_in=...)` emits records whose `extra_fields` dict survives into the JSON payload — useful for CloudWatch Insights queries. Because `logging.LoggerAdapter.process()` in Python 3.12 overwrites `extra=`, `log_event` dispatches via `logger.logger` (the underlying `Logger`) instead of the adapter.

## Deployment

`serverless.yml` provisions:
- Lambda: python3.12, arm64, 5120MB, 90s timeout.
- DynamoDB: hash `id`, GSI `user-index` (user + expire_at, KEYS_ONLY), TTL `expire_at`.
- IAM: `dynamodb:GetItem/PutItem/Query` on table + GSI, `bedrock:InvokeModel*`/`Converse*`.

`.github/workflows/push-main.yml` runs pytest (with coverage), then `configure-aws-credentials` OIDC → `serverless deploy`. Secrets and Variables split described in README.

## Testing

Coverage target 80%+, currently 83% overall. Key approach:
- `moto[dynamodb]` for `DedupStore` / `ConversationStore` integration tests.
- `responses` / `unittest.mock.patch("src.tools.urllib.request.urlopen")` for web tools.
- `ScriptedLLM` (see `tests/test_agent.py`) emits predefined `LLMResult` sequences to drive loop scenarios without any network.
- Provider tests use `MagicMock` clients (no real OpenAI / Bedrock calls).
- `tests/test_config.py` builds `Settings` from `monkeypatch`-controlled env without reloading the module.

## Things that are easy to break

- **Dropping the `_CompositeProvider` branch** in `get_llm` breaks mixed-provider setups (OpenAI text + Bedrock image).
- **Changing `DedupStore.reserve` to a read-then-write pattern** reintroduces the retry race.
- **Losing the `id` prefix scheme** (`dedup:` vs `ctx:`) collides the two store types.
- **Switching to `LoggerAdapter.info(extra=…)`** — in Python 3.12 the adapter's `process()` overwrites `extra`; keep going through `logger.logger` for `extra_fields`.
- **Removing the SSRF host allowlist** in `read_attached_images` opens up arbitrary URL fetch with the bot token.
- **Adding a tool without updating `ToolRegistry.specs()`** — the `@tool` decorator handles both dispatch and LLM schema from a single declaration; inline dict tricks will silently desync.

## Excluded (Phase 2+)

- Bedrock Knowledge Base (S3 Vectors + RAG) ingestion pipeline
- `reaction_added` event wiring + domain-specific handlers (refund masking, etc.)
- CloudWatch Alarms / X-Ray / multi-language prompts beyond ko/en
