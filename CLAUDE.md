# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Local env setup (loaded by src/config.py via python-dotenv)
cp .env.example .env.local   # fill in OPENAI_API_KEY at minimum

# Run the agent locally (no Slack connection required)
python localtest.py "질문 내용"   # one-shot
python localtest.py               # interactive (stdin, Ctrl+D to submit)

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_agent.py::AgentTests::test_runs_tool_then_returns_final_text -v
```

Lambda entrypoint: `app.lambda_handler` (Slack events routed via `slack_bolt.adapter.aws_lambda.SlackRequestHandler`).

## Architecture

Slack `app_mention` → `app.py` → `SlackMentionAgent.run()` → final `say()` reply to the thread.

The agent is a 3-phase LLM loop, not a single completion:

1. **Plan** — `LLMClient.chat_json` asked to emit strict JSON (`goal`, `plan`, `tool_calls`, `requires_image`, `image_prompt`). The tool catalog from `ToolExecutor.available_tools` is injected into the prompt.
2. **Tool loop** — up to `AGENT_MAX_STEPS` iterations. Each iteration executes every `tool_call` in the current state, appends `{step, tool_call, result}` to `observations`, then re-queries the LLM for the next JSON state (more tool calls or `final_answer`). If `generate_image` returns ok, its Slack permalink is captured as `image_url`.
3. **Compose** — `chat_text` produces the user-facing Slack reply from plan + observations + draft answer. An extra post-loop `generate_image` fires only if the LLM flagged `requires_image` but no image was produced.

Key invariants to preserve when editing:

- `ToolExecutor.execute` wraps every tool in `{ok, result}` / `{ok, error}` — the agent inspects `result.get("ok")` and `result.get("result")` on that exact shape. Changing it breaks the image-url capture path and the observation log.
- `LLMClient.chat_json` already tolerates non-JSON wrappers via `_extract_first_json_object` (scans for first `{...}`). Don't add another layer of JSON cleanup at call sites.
- `generate_image` is both a registered tool AND a post-loop fallback. If you move image generation, keep both paths.
- `read_attached_images` and `search_web` pin URL hosts (`SLACK_FILE_HOSTS`, `WEB_SEARCH_HOST`) as an SSRF guard — don't remove the `urlparse` checks.

## Provider switching

`LLMClient` is a single class with branches on `provider` (`openai`|`bedrock`) for text/vision and on `image_provider` for image generation. `boto3.client("bedrock-runtime")` is only created when either provider is `bedrock`. Bedrock payloads assume Anthropic (`anthropic_version: bedrock-2023-05-31`) for chat/vision and Titan-style `TEXT_IMAGE` for images — switching Bedrock model families requires editing those bodies, not just `LLM_MODEL`.

`IMAGE_PROVIDER` defaults to whatever `LLM_PROVIDER` is — set it explicitly when mixing (e.g. OpenAI text + Bedrock images).

## Local testing without Slack

`localtest.py` substitutes a `_StubSlackClient` when `SLACK_BOT_TOKEN` is missing or still the placeholder `xoxb-your...`. The stub returns structurally valid but empty responses, so Slack-dependent tools degrade gracefully instead of raising. Tests that need real Slack behavior must use a real token.

## Env vars

Required for Lambda: `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` (checked at import time in `app.py`). Required for OpenAI paths: `OPENAI_API_KEY`. Optional: `LLM_PROVIDER`, `LLM_MODEL`, `IMAGE_PROVIDER`, `IMAGE_MODEL`, `AGENT_MAX_STEPS` (default 3), `RESPONSE_LANGUAGE` (`ko`/`en`, default `ko`).

`src/config.py` auto-loads `.env.local` from the repo root via `python-dotenv` when present (non-overriding), so local runs and pytest pick it up automatically.
