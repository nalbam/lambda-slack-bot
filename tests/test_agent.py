from unittest.mock import MagicMock

from src.agent import SlackMentionAgent
from src.llm import LLMResult, ToolCall
from src.tools import ToolContext, ToolRegistry, tool


def _ctx():
    return ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event={},
        settings=MagicMock(),
        llm=MagicMock(),
    )


class ScriptedLLM:
    def __init__(self, results: list[LLMResult]):
        self._results = list(results)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools=None, max_tokens=1024):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._results:
            return LLMResult(content="(empty)", tool_calls=[], stop_reason="end_turn")
        return self._results.pop(0)

    def stream_chat(self, system, messages, on_delta, max_tokens=1024):
        result = self.chat(system, messages)
        if on_delta and result.content:
            on_delta(result.content)
        return result.content

    def describe_image(self, b, m):
        return "desc"

    def generate_image(self, p):
        return b"img"


def _registry_with_search():
    reg = ToolRegistry()

    @tool(reg, name="search_web", description="", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
    def _search(ctx, query):
        return [{"title": "AWS", "url": "https://aws.amazon.com"}]

    return reg


def test_agent_terminates_when_no_tool_calls():
    reg = _registry_with_search()
    llm = ScriptedLLM([LLMResult(content="final", tool_calls=[], stop_reason="end_turn")])
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=3)
    result = agent.run("question")
    assert result.text == "final"
    assert result.steps == 1
    assert result.tool_calls_count == 0


def test_agent_runs_tool_then_returns_text():
    reg = _registry_with_search()
    llm = ScriptedLLM(
        [
            LLMResult(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_web", arguments={"query": "aws"})],
                stop_reason="tool_use",
            ),
            LLMResult(content="결과 기반 답변", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=3)
    result = agent.run("질문")
    assert result.text == "결과 기반 답변"
    assert result.tool_calls_count == 1
    assert result.steps == 2


def test_agent_duplicate_call_is_skipped():
    reg = _registry_with_search()
    called = {"count": 0}

    reg = ToolRegistry()

    @tool(reg, name="search_web", description="", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
    def _search(ctx, query):
        called["count"] += 1
        return [{"title": "X"}]

    # LLM calls the same tool twice with identical args, then finishes.
    llm = ScriptedLLM(
        [
            LLMResult(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_web", arguments={"query": "same"})],
                stop_reason="tool_use",
            ),
            LLMResult(
                content="",
                tool_calls=[ToolCall(id="c2", name="search_web", arguments={"query": "same"})],
                stop_reason="tool_use",
            ),
            LLMResult(content="done", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=5)
    agent.run("q")
    assert called["count"] == 1  # second call suppressed


def test_agent_captures_image_url():
    reg = ToolRegistry()

    @tool(reg, name="generate_image", description="", parameters={"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]})
    def _gen(ctx, prompt):
        return {"permalink": "https://slack/x"}

    llm = ScriptedLLM(
        [
            LLMResult(
                content="",
                tool_calls=[ToolCall(id="c1", name="generate_image", arguments={"prompt": "cat"})],
                stop_reason="tool_use",
            ),
            LLMResult(content="here is your image", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=3)
    result = agent.run("그려줘")
    assert result.image_url == "https://slack/x"


def test_agent_forces_final_compose_at_max_steps():
    reg = _registry_with_search()
    # Every step returns more tool calls — never ends.
    def infinite():
        while True:
            yield LLMResult(
                content="",
                tool_calls=[ToolCall(id="x", name="search_web", arguments={"query": "q"})],
                stop_reason="tool_use",
            )

    class EndlessLLM:
        def __init__(self):
            self._gen = infinite()
            self.end_called = False

        def chat(self, *a, **k):
            return next(self._gen)

        def stream_chat(self, system, messages, on_delta, max_tokens=1024):
            self.end_called = True
            if on_delta:
                on_delta("forced")
            return "forced"

        def describe_image(self, *a, **k):
            return ""

        def generate_image(self, *a, **k):
            return b""

    llm = EndlessLLM()
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=2, on_stream=lambda d: None)
    result = agent.run("q")
    assert result.text == "forced"
    assert llm.end_called is True
    assert result.steps == 2


def test_agent_aggregates_token_usage():
    reg = _registry_with_search()
    llm = ScriptedLLM(
        [
            LLMResult(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_web", arguments={"query": "x"})],
                stop_reason="tool_use",
                token_usage={"input": 10, "output": 20},
            ),
            LLMResult(
                content="done",
                tool_calls=[],
                stop_reason="end_turn",
                token_usage={"input": 5, "output": 7},
            ),
        ]
    )
    agent = SlackMentionAgent(llm=llm, context=_ctx(), registry=reg, max_steps=3)
    result = agent.run("q")
    assert result.token_usage == {"input": 15, "output": 27}
