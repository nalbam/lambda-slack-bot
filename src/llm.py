"""LLM provider abstraction with native function calling.

Two providers:
- OpenAIProvider: chat completions with `tools=`, vision, image generation.
- BedrockProvider: family-routed. Claude 3/3.5 uses Messages API with tools;
  Amazon Nova uses Converse API with toolConfig; others fall back to plain text.

Both providers implement the LLMProvider protocol so the Agent loop is
provider-agnostic.
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


class OpenAIProvider:
    def __init__(self, model: str, image_model: str):
        self.model = model
        self.image_model = image_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _token_params(self, max_tokens: int) -> dict[str, Any]:
        if _is_new_gen_openai(self.model):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens, "temperature": 0.2}

    @staticmethod
    def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._to_openai_messages(messages)],
            **self._token_params(max_tokens),
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        completion = client.chat.completions.create(**payload)
        choice = completion.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))

        stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"] = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        elif choice.finish_reason not in {"stop", None}:
            stop_reason = "other"

        usage = getattr(completion, "usage", None)
        token_usage = {
            "input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0,
        } if usage else {}

        return LLMResult(content=msg.content or "", tool_calls=tool_calls, stop_reason=stop_reason, token_usage=token_usage)

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
            messages=[{"role": "system", "content": system}, *self._to_openai_messages(messages)],
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
        kwargs: dict[str, Any] = {
            "model": self.image_model,
            "prompt": prompt,
            "size": "1024x1024",
        }
        # gpt-image-1 rejects `response_format` (b64 is the default); only legacy
        # DALL-E models need the explicit flag.
        if self.image_model.startswith("dall-e"):
            kwargs["response_format"] = "b64_json"
        response = client.images.generate(**kwargs)
        return base64.b64decode(response.data[0].b64_json)


# --------------------------------------------------------------------------- #
# Bedrock
# --------------------------------------------------------------------------- #


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

    # -- text / tool use ---------------------------------------------------- #

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        if self.model.startswith("anthropic.claude"):
            return self._claude_chat(system, messages, tools, max_tokens)
        if self.model.startswith("amazon.nova"):
            return self._nova_chat(system, messages, tools, max_tokens)
        # Unknown family — plain text completion, no tools.
        return self._claude_chat(system, messages, None, max_tokens)

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
        if self.model.startswith("anthropic.claude"):
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
        if self.image_model.startswith("amazon.titan-image") or self.image_model.startswith("amazon.nova-canvas"):
            return {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {"numberOfImages": 1, "quality": "standard", "height": 1024, "width": 1024},
            }
        if self.image_model.startswith("stability."):
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
) -> LLMProvider:
    text: LLMProvider
    if provider == "bedrock":
        text = BedrockProvider(model=model, image_model=image_model, region=region)
    else:
        text = OpenAIProvider(model=model, image_model=image_model)

    if image_provider == provider:
        return text

    # Mixed: text and image come from different backends.
    image_llm: LLMProvider
    if image_provider == "bedrock":
        image_llm = BedrockProvider(model=model, image_model=image_model, region=region)
    else:
        image_llm = OpenAIProvider(model=model, image_model=image_model)

    return _CompositeProvider(text=text, image=image_llm)


@dataclass
class _CompositeProvider:
    """Delegates generate_image to a different provider than chat/vision."""

    text: LLMProvider
    image: LLMProvider

    def chat(self, system, messages, tools=None, max_tokens=1024):
        return self.text.chat(system, messages, tools=tools, max_tokens=max_tokens)

    def stream_chat(self, system, messages, on_delta, max_tokens=1024):
        return self.text.stream_chat(system, messages, on_delta, max_tokens=max_tokens)

    def describe_image(self, image_bytes, mime_type):
        return self.text.describe_image(image_bytes, mime_type)

    def generate_image(self, prompt):
        return self.image.generate_image(prompt)
