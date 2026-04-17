import json
from unittest.mock import MagicMock, patch

from src.llm import BedrockProvider, OpenAIProvider, ToolCall


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


def _openai_completion(content="", tool_calls=None, finish="stop"):
    choice = MagicMock()
    choice.finish_reason = finish
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 10
    completion.usage.completion_tokens = 20
    return completion


def _openai_tool_call(call_id, name, args_obj):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args_obj)
    return tc


def test_openai_chat_parses_text():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="hello")
    result = provider.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "hello"
    assert result.stop_reason == "end_turn"
    assert result.tool_calls == []
    assert result.token_usage == {"input": 10, "output": 20}


def test_openai_legacy_model_uses_max_tokens_and_temperature():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="x")
    provider.chat(system="s", messages=[])
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert "max_tokens" in kwargs
    assert "temperature" in kwargs
    assert "max_completion_tokens" not in kwargs


def test_openai_new_generation_uses_max_completion_tokens():
    provider = OpenAIProvider(model="gpt-5.4", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="x")
    provider.chat(system="s", messages=[])
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_openai_o1_model_uses_max_completion_tokens():
    provider = OpenAIProvider(model="o1-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="x")
    provider.chat(system="s", messages=[])
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in kwargs
    assert "temperature" not in kwargs


def test_openai_translates_canonical_tool_calls():
    """Canonical assistant tool_calls must be serialized to OpenAI's wire format."""
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="done")

    canonical = [
        {"role": "user", "content": "ask"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "search_web", "arguments": {"query": "q"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true}"},
    ]
    provider.chat(system="s", messages=canonical)
    sent = provider._client.chat.completions.create.call_args.kwargs["messages"]
    # system + 3 canonical = 4
    assert len(sent) == 4
    assistant = sent[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "search_web"
    # arguments must be a JSON string, not a dict
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)
    import json as _json

    assert _json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"query": "q"}


def test_openai_chat_parses_tool_calls():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    tc = _openai_tool_call("call_1", "search_web", {"query": "aws"})
    provider._client.chat.completions.create.return_value = _openai_completion(
        tool_calls=[tc], finish="tool_calls"
    )
    result = provider.chat(system="s", messages=[], tools=[{"name": "search_web", "description": "", "parameters": {}}])
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_web"
    assert result.tool_calls[0].arguments == {"query": "aws"}


def test_openai_chat_handles_bad_tool_arguments():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    tc = MagicMock()
    tc.id = "x"
    tc.function.name = "search_web"
    tc.function.arguments = "not json"
    provider._client.chat.completions.create.return_value = _openai_completion(
        tool_calls=[tc], finish="tool_calls"
    )
    result = provider.chat(system="s", messages=[], tools=[{"name": "search_web", "description": "", "parameters": {}}])
    assert result.tool_calls[0].arguments == {}


def test_openai_stream_chat_invokes_callback():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()

    def _chunk(text):
        ch = MagicMock()
        ch.choices[0].delta.content = text
        return ch

    provider._client.chat.completions.create.return_value = iter([_chunk("he"), _chunk("llo")])
    seen = []
    result = provider.stream_chat(system="s", messages=[], on_delta=seen.append)
    assert result == "hello"
    assert seen == ["he", "llo"]


# --------------------------------------------------------------------------- #
# Bedrock — Claude
# --------------------------------------------------------------------------- #


def _bedrock_response(payload: dict):
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    return {"body": body}


def test_bedrock_claude_chat_text():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {
            "content": [{"type": "text", "text": "안녕"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )
    result = provider.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "안녕"
    assert result.stop_reason == "end_turn"
    assert result.token_usage == {"input": 5, "output": 7}


def test_bedrock_claude_chat_with_tool_use():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {
            "content": [
                {"type": "text", "text": "I'll search."},
                {"type": "tool_use", "id": "tu_1", "name": "search_web", "input": {"query": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )
    tools = [{"name": "search_web", "description": "", "parameters": {"type": "object"}}]
    result = provider.chat(system="s", messages=[], tools=tools)
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "search_web"
    assert result.tool_calls[0].arguments == {"query": "x"}


def test_bedrock_message_translation_tool_role():
    messages = [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "foo", "arguments": {"a": 1}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{\"ok\":true}"},
    ]
    translated = BedrockProvider._to_anthropic_messages(messages)
    assert translated[0] == {"role": "user", "content": "ask"}
    assert translated[1]["role"] == "assistant"
    assert translated[1]["content"][0]["type"] == "tool_use"
    assert translated[1]["content"][0]["name"] == "foo"
    assert translated[2]["role"] == "user"
    assert translated[2]["content"][0]["type"] == "tool_result"


# --------------------------------------------------------------------------- #
# Bedrock — Nova
# --------------------------------------------------------------------------- #


def test_bedrock_nova_chat_text():
    provider = BedrockProvider(model="amazon.nova-pro-v1:0", image_model="amazon.nova-canvas-v1:0", region="us-east-1")
    provider._client = MagicMock()
    provider._client.converse.return_value = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 2},
    }
    result = provider.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert result.content == "hi"
    assert result.stop_reason == "end_turn"
    assert result.token_usage == {"input": 1, "output": 2}


def test_bedrock_nova_tool_use():
    provider = BedrockProvider(model="amazon.nova-pro-v1:0", image_model="amazon.nova-canvas-v1:0", region="us-east-1")
    provider._client = MagicMock()
    provider._client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "let me search"},
                    {"toolUse": {"toolUseId": "tu1", "name": "search_web", "input": {"query": "q"}}},
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 2, "outputTokens": 3},
    }
    result = provider.chat(
        system="s",
        messages=[],
        tools=[{"name": "search_web", "description": "", "parameters": {"type": "object"}}],
    )
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "search_web"


# --------------------------------------------------------------------------- #
# Image model adapter
# --------------------------------------------------------------------------- #


def test_build_image_body_titan():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    body = provider._build_image_body("a cat")
    assert body["taskType"] == "TEXT_IMAGE"


def test_build_image_body_stability():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="stability.stable-diffusion-xl-v1",
        region="us-east-1",
    )
    body = provider._build_image_body("a cat")
    assert body["text_prompts"][0]["text"] == "a cat"


def test_build_image_body_unknown_raises():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="mystery.v1",
        region="us-east-1",
    )
    import pytest

    with pytest.raises(ValueError):
        provider._build_image_body("x")


def test_bedrock_describe_image_returns_text():
    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"content": [{"type": "text", "text": "a cat"}]}
    )
    out = provider.describe_image(b"fake", "image/png")
    assert out == "a cat"


def test_bedrock_generate_image_titan_returns_bytes():
    import base64 as _b64

    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="amazon.titan-image-generator-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"images": [_b64.b64encode(b"imgdata").decode()]}
    )
    assert provider.generate_image("cat") == b"imgdata"


def test_bedrock_generate_image_stability_returns_bytes():
    import base64 as _b64

    provider = BedrockProvider(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_model="stability.stable-diffusion-xl-v1",
        region="us-east-1",
    )
    provider._client = MagicMock()
    provider._client.invoke_model.return_value = _bedrock_response(
        {"artifacts": [{"base64": _b64.b64encode(b"xyz").decode()}]}
    )
    assert provider.generate_image("cat") == b"xyz"


def test_openai_describe_image_uses_vision_format():
    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _openai_completion(content="it's a cat")
    out = provider.describe_image(b"\x89PNG", "image/png")
    assert out == "it's a cat"
    args = provider._client.chat.completions.create.call_args.kwargs
    assert args["messages"][0]["content"][1]["type"] == "image_url"


def test_openai_generate_image_decodes_b64():
    import base64 as _b64

    provider = OpenAIProvider(model="gpt-4o-mini", image_model="gpt-image-1")
    provider._client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(b64_json=_b64.b64encode(b"hello").decode())]
    provider._client.images.generate.return_value = response
    assert provider.generate_image("cat") == b"hello"
    kwargs = provider._client.images.generate.call_args.kwargs
    # gpt-image-1 must NOT send response_format (API rejects it)
    assert "response_format" not in kwargs


def test_openai_generate_image_dalle_sends_response_format():
    import base64 as _b64

    provider = OpenAIProvider(model="gpt-4o-mini", image_model="dall-e-3")
    provider._client = MagicMock()
    response = MagicMock()
    response.data = [MagicMock(b64_json=_b64.b64encode(b"ok").decode())]
    provider._client.images.generate.return_value = response
    provider.generate_image("cat")
    kwargs = provider._client.images.generate.call_args.kwargs
    assert kwargs["response_format"] == "b64_json"


def test_composite_provider_routes_image_to_image_llm():
    from src.llm import _CompositeProvider

    text = MagicMock()
    image = MagicMock()
    image.generate_image.return_value = b"img"
    composite = _CompositeProvider(text=text, image=image)
    composite.chat(system="s", messages=[])
    text.chat.assert_called_once()
    composite.generate_image("x")
    image.generate_image.assert_called_once_with("x")
