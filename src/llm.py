"""LLM provider abstraction with native function calling.

Three providers:
- OpenAIProvider: OpenAI chat completions with `tools=`, vision, image generation.
- XAIProvider: xAI (Grok) — OpenAI-wire compatible at https://api.x.ai/v1.
  Shares `_OpenAICompatProvider` machinery with OpenAI; differs in image kwargs
  (omits `size`, forces `response_format=b64_json`) and token params
  (always `max_tokens` + `temperature`).
- BedrockProvider: family-routed. Anthropic Claude uses Messages API with tools;
  Amazon Nova uses Converse API with toolConfig; others fall back to plain text.
  Accepts both bare model IDs and `us./eu./apac./global.` inference-profile IDs.

All providers implement the LLMProvider protocol so the Agent loop is
provider-agnostic. `_CompositeProvider` wraps two providers when text and
image providers differ (e.g., OpenAI text + xAI image).
"""
from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import boto3

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

ToolSpec = dict[str, Any]  # {"name","description","parameters"(JSON Schema)}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"] = "end_turn"
    token_usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResult: ...

    def stream_chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        max_tokens: int = 1024,
    ) -> str: ...

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str: ...

    def generate_image(self, prompt: str) -> bytes: ...


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

_RETRYABLE_BEDROCK = {"ThrottlingException", "ServiceQuotaExceededException", "ModelTimeoutException"}


def _with_retry(fn: Callable[[], Any], label: str, attempts: int = 3) -> Any:
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            code = getattr(getattr(exc, "response", None), "get", lambda _k, _d=None: None)("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code in _RETRYABLE_BEDROCK and attempt < attempts - 1:
                logger.warning("%s retryable (%s), backoff %.1fs", label, code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    if last_exc:
        raise last_exc


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


_OPENAI_NEW_GENERATION_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_new_gen_openai(model: str) -> bool:
    """Newer OpenAI models (gpt-5, o1/o3/o4 reasoning) use `max_completion_tokens`
    and disallow `temperature` overrides."""
    return any(model.startswith(p) for p in _OPENAI_NEW_GENERATION_PREFIXES)


# --------------------------------------------------------------------------- #
# Module-level helpers shared between OpenAI-compatible providers (OpenAI, xAI)
# --------------------------------------------------------------------------- #


def _to_openai_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate canonical messages (our agent's shape) to OpenAI's wire shape."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                }
            )
        else:
            out.append(msg)
    return out


def _build_openai_tools_payload(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _map_openai_finish_reason(finish: str | None) -> Literal["end_turn", "tool_use", "max_tokens", "other"]:
    if finish == "tool_calls":
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    if finish in {"stop", None}:
        return "end_turn"
    return "other"


def _extract_openai_usage(usage_obj) -> dict[str, int]:
    if not usage_obj:
        return {}
    return {
        "input": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "output": getattr(usage_obj, "completion_tokens", 0) or 0,
    }


def _parse_openai_completion(completion) -> LLMResult:
    choice = completion.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for call in (msg.tool_calls or []):
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))

    return LLMResult(
        content=msg.content or "",
        tool_calls=tool_calls,
        stop_reason=_map_openai_finish_reason(choice.finish_reason),
        token_usage=_extract_openai_usage(getattr(completion, "usage", None)),
    )


def _consume_openai_stream(stream, on_delta: Callable[[str], None]) -> LLMResult:
    """Drain an OpenAI-compatible chat completion stream.

    Stops forwarding content to `on_delta` once a tool_calls delta arrives —
    any trailing commentary would otherwise leak into the final user reply.
    tool_calls chunks are accumulated by index and returned as ToolCall list.
    """
    content_parts: list[str] = []
    tool_calls_accum: dict[int, dict[str, Any]] = {}
    saw_tool_calls = False
    finish_reason: str | None = None
    usage_obj = None

    for chunk in stream:
        usage_obj = getattr(chunk, "usage", None) or usage_obj
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if getattr(delta, "tool_calls", None):
            saw_tool_calls = True
            for tc in delta.tool_calls:
                idx = tc.index
                slot = tool_calls_accum.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            if not saw_tool_calls:
                on_delta(delta.content)
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason

    tool_calls: list[ToolCall] = []
    for idx in sorted(tool_calls_accum):
        slot = tool_calls_accum[idx]
        try:
            args = json.loads(slot["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=slot["id"] or "", name=slot["name"], arguments=args))

    return LLMResult(
        content="".join(content_parts),
        tool_calls=tool_calls,
        stop_reason=_map_openai_finish_reason(finish_reason),
        token_usage=_extract_openai_usage(usage_obj),
    )


class _OpenAICompatProvider:
    """Shared machinery for any OpenAI-wire-compatible chat/vision/image API.

    Subclasses set BASE_URL / API_KEY_ENV_VAR and override small hooks
    (`_token_params`, `_image_generate_kwargs`). The heavy lifting —
    payload assembly, streaming, tool_calls parsing — lives on this base
    and on the module-level helpers above.
    """

    BASE_URL: str | None = None  # None = OpenAI default
    API_KEY_ENV_VAR: str = "OPENAI_API_KEY"

    def __init__(self, model: str, image_model: str, api_key: str | None = None):
        self.model = model
        self.image_model = image_model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if self.BASE_URL:
                kwargs["base_url"] = self.BASE_URL
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    # -- hooks -------------------------------------------------------------- #

    def _token_params(self, max_tokens: int) -> dict[str, Any]:
        """Default: OpenAI legacy models use max_tokens+temperature."""
        return {"max_tokens": max_tokens, "temperature": 0.2}

    def _image_generate_kwargs(self, prompt: str) -> dict[str, Any]:
        """Default OpenAI (dall-e / gpt-image-1) image call kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.image_model,
            "prompt": prompt,
            "size": "1024x1024",
        }
        # gpt-image-1 rejects `response_format` (b64 is the default); only legacy
        # DALL-E models need the explicit flag.
        if self.image_model.startswith("dall-e"):
            kwargs["response_format"] = "b64_json"
        return kwargs

    # -- LLMProvider surface ----------------------------------------------- #

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResult:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *_to_openai_wire_messages(messages)],
            **self._token_params(max_tokens),
        }
        if tools:
            payload["tools"] = _build_openai_tools_payload(tools)
            payload["tool_choice"] = "auto"

        if on_delta is None:
            completion = client.chat.completions.create(**payload)
            return _parse_openai_completion(completion)

        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        stream = client.chat.completions.create(**payload)
        return _consume_openai_stream(stream, on_delta)

    def stream_chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        max_tokens: int = 1024,
    ) -> str:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *_to_openai_wire_messages(messages)],
            stream=True,
            **self._token_params(max_tokens),
        )
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full += delta
                on_delta(delta)
        return full

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        client = self._get_client()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image for a Slack conversation."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content or ""

    def generate_image(self, prompt: str) -> bytes:
        client = self._get_client()
        response = client.images.generate(**self._image_generate_kwargs(prompt))
        return base64.b64decode(response.data[0].b64_json)


class OpenAIProvider(_OpenAICompatProvider):
    BASE_URL = None  # default OpenAI endpoint
    API_KEY_ENV_VAR = "OPENAI_API_KEY"

    def _token_params(self, max_tokens: int) -> dict[str, Any]:
        # Newer OpenAI reasoning models only accept max_completion_tokens and
        # reject `temperature`. Legacy chat models still use max_tokens.
        if _is_new_gen_openai(self.model):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens, "temperature": 0.2}


class XAIProvider(_OpenAICompatProvider):
    """xAI (Grok) — OpenAI-wire compatible, different base URL and image params.

    Models:
      text:  grok-4-1-fast-reasoning, grok-4.20-0309-reasoning, ...
      image: grok-imagine-image, grok-imagine-image-pro

    Differences from OpenAI that matter here:
      - `images.generate` rejects `size` (uses `aspect_ratio`/`resolution`).
        We omit `size` and request `response_format=b64_json` so we can
        decode bytes locally, matching the rest of the pipeline.
      - All current grok chat models accept `max_tokens` + `temperature`
        the classic way — no `max_completion_tokens` split.
    """

    BASE_URL = "https://api.x.ai/v1"
    API_KEY_ENV_VAR = "XAI_API_KEY"

    def _image_generate_kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.image_model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }


# --------------------------------------------------------------------------- #
# Bedrock
# --------------------------------------------------------------------------- #


_INFERENCE_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.")


def _strip_inference_profile_prefix(model_id: str) -> str:
    """Return the bare family id from a Bedrock model or inference-profile id.

    Inference profile IDs prefix the family with a region routing hint, e.g.
    `us.anthropic.claude-haiku-4-5-20251001-v1:0`. For family-level routing
    ("is this a Claude? a Nova? Titan?") we care about the bare portion.
    """
    for p in _INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(p):
            return model_id[len(p):]
    return model_id


class BedrockProvider:
    def __init__(self, model: str, image_model: str, region: str):
        self.model = model
        self.image_model = image_model
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    @property
    def _text_family(self) -> str:
        return _strip_inference_profile_prefix(self.model)

    @property
    def _image_family(self) -> str:
        return _strip_inference_profile_prefix(self.image_model)

    # -- text / tool use ---------------------------------------------------- #

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        on_delta: Callable[[str], None] | None = None,
    ) -> LLMResult:
        # Bedrock tool_use streaming is not yet implemented in this provider;
        # accept the on_delta parameter for API compatibility but use the
        # blocking path, then emit the final content as a single delta so
        # callers still receive *something* through the streaming channel.
        family = self._text_family
        if family.startswith("anthropic.claude"):
            result = self._claude_chat(system, messages, tools, max_tokens)
        elif family.startswith("amazon.nova"):
            result = self._nova_chat(system, messages, tools, max_tokens)
        else:
            result = self._claude_chat(system, messages, None, max_tokens)
        if on_delta is not None and result.content and not result.tool_calls:
            on_delta(result.content)
        return result

    def _claude_chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
        }
        if tools:
            body["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]

        client = self._get_client()
        response = _with_retry(
            lambda: client.invoke_model(modelId=self.model, body=json.dumps(body)),
            label="bedrock.invoke_model",
        )
        payload = json.loads(response["body"].read())

        content_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in payload.get("content", []):
            if block.get("type") == "text":
                content_text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", str(uuid.uuid4())),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )

        stop_reason_raw = payload.get("stop_reason", "end_turn")
        stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
        if stop_reason_raw == "tool_use":
            stop_reason = "tool_use"
        elif stop_reason_raw == "max_tokens":
            stop_reason = "max_tokens"
        elif stop_reason_raw == "end_turn":
            stop_reason = "end_turn"
        else:
            stop_reason = "other"

        usage = payload.get("usage") or {}
        return LLMResult(
            content="".join(content_text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            token_usage={"input": usage.get("input_tokens", 0) or 0, "output": usage.get("output_tokens", 0) or 0},
        )

    def _nova_chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        max_tokens: int,
    ) -> LLMResult:
        client = self._get_client()
        payload: dict[str, Any] = {
            "modelId": self.model,
            "system": [{"text": system}],
            "messages": self._to_nova_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2},
        }
        if tools:
            payload["toolConfig"] = {
                "tools": [
                    {"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": t["parameters"]}}}
                    for t in tools
                ]
            }

        response = _with_retry(lambda: client.converse(**payload), label="bedrock.converse")
        out_msg = response.get("output", {}).get("message", {})
        content_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in out_msg.get("content", []):
            if "text" in block:
                content_text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(id=tu.get("toolUseId") or str(uuid.uuid4()), name=tu.get("name", ""), arguments=tu.get("input", {}) or {})
                )

        stop_reason_raw = response.get("stopReason", "end_turn")
        stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
        if stop_reason_raw == "tool_use":
            stop_reason = "tool_use"
        elif stop_reason_raw == "max_tokens":
            stop_reason = "max_tokens"
        elif stop_reason_raw == "end_turn":
            stop_reason = "end_turn"
        else:
            stop_reason = "other"

        usage = response.get("usage") or {}
        return LLMResult(
            content="".join(content_text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            token_usage={"input": usage.get("inputTokens", 0) or 0, "output": usage.get("outputTokens", 0) or 0},
        )

    # -- streaming --------------------------------------------------------- #

    def stream_chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_delta: Callable[[str], None],
        max_tokens: int = 1024,
    ) -> str:
        # Bedrock streaming implementation: Claude Messages stream or Converse stream.
        client = self._get_client()
        full = ""
        family = self._text_family
        if family.startswith("anthropic.claude"):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system,
                "messages": self._to_anthropic_messages(messages),
            }
            response = client.invoke_model_with_response_stream(modelId=self.model, body=json.dumps(body))
            for event in response.get("body", []):
                chunk = event.get("chunk", {})
                if not chunk:
                    continue
                payload = json.loads(chunk.get("bytes", b"{}"))
                if payload.get("type") == "content_block_delta":
                    delta = (payload.get("delta") or {}).get("text") or ""
                    if delta:
                        full += delta
                        on_delta(delta)
            return full

        # Nova Converse stream
        response = client.converse_stream(
            modelId=self.model,
            system=[{"text": system}],
            messages=self._to_nova_messages(messages),
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
        )
        for event in response.get("stream", []):
            cbd = event.get("contentBlockDelta")
            if cbd:
                delta = (cbd.get("delta") or {}).get("text") or ""
                if delta:
                    full += delta
                    on_delta(delta)
        return full

    # -- vision / image ----------------------------------------------------- #

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image for a Slack conversation."},
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}},
                    ],
                }
            ],
        }
        client = self._get_client()
        response = client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        for block in payload.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    def generate_image(self, prompt: str) -> bytes:
        body = self._build_image_body(prompt)
        client = self._get_client()
        response = client.invoke_model(modelId=self.image_model, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        return self._extract_image_bytes(payload)

    def _build_image_body(self, prompt: str) -> dict[str, Any]:
        family = self._image_family
        if family.startswith("amazon.titan-image") or family.startswith("amazon.nova-canvas"):
            return {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {"numberOfImages": 1, "quality": "standard", "height": 1024, "width": 1024},
            }
        if family.startswith("stability."):
            return {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "steps": 30, "seed": 0}
        raise ValueError(f"unsupported Bedrock image model: {self.image_model}")

    def _extract_image_bytes(self, payload: dict[str, Any]) -> bytes:
        if "images" in payload and payload["images"]:
            return base64.b64decode(payload["images"][0])
        if "artifacts" in payload and payload["artifacts"]:
            return base64.b64decode(payload["artifacts"][0]["base64"])
        raise ValueError("no image returned from Bedrock")

    # -- format helpers ----------------------------------------------------- #

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate our canonical messages format into Anthropic Messages API shape.

        Our format mirrors OpenAI's: role=user/assistant/tool, content can be str
        or list. We map `tool` role to a user message with a tool_result block,
        and `assistant` messages with tool_calls into tool_use content blocks.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc.get("arguments", {})}
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role or "user", "content": msg.get("content", "")})
        return out

    @staticmethod
    def _to_nova_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": msg.get("tool_call_id", ""),
                                    "content": [{"text": str(msg.get("content", ""))}],
                                }
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"text": msg["content"]})
                for tc in msg["tool_calls"]:
                    blocks.append({"toolUse": {"toolUseId": tc["id"], "name": tc["name"], "input": tc.get("arguments", {})}})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role or "user", "content": [{"text": str(msg.get("content", ""))}]})
        return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_llm(
    provider: str,
    model: str,
    image_provider: str,
    image_model: str,
    region: str = "us-east-1",
    api_keys: dict[str, str | None] | None = None,
) -> LLMProvider:
    """Build an LLM client for the requested provider(s).

    `api_keys` carries per-provider keys that need explicit wiring (xAI today;
    OpenAI reads OPENAI_API_KEY from env directly, Bedrock uses the AWS SDK
    credential chain).
    """
    api_keys = api_keys or {}

    def build(p: str) -> LLMProvider:
        if p == "bedrock":
            return BedrockProvider(model=model, image_model=image_model, region=region)
        if p == "xai":
            return XAIProvider(model=model, image_model=image_model, api_key=api_keys.get("xai"))
        return OpenAIProvider(model=model, image_model=image_model)

    text = build(provider)
    if image_provider == provider:
        return text
    return _CompositeProvider(text=text, image=build(image_provider))


@dataclass
class _CompositeProvider:
    """Delegates generate_image to a different provider than chat/vision."""

    text: LLMProvider
    image: LLMProvider

    def chat(self, system, messages, tools=None, max_tokens=1024, on_delta=None):
        return self.text.chat(system, messages, tools=tools, max_tokens=max_tokens, on_delta=on_delta)

    def stream_chat(self, system, messages, on_delta, max_tokens=1024):
        return self.text.stream_chat(system, messages, on_delta, max_tokens=max_tokens)

    def describe_image(self, image_bytes, mime_type):
        return self.text.describe_image(image_bytes, mime_type)

    def generate_image(self, prompt):
        return self.image.generate_image(prompt)
