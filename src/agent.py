"""Agent loop using native LLM function calling.

No more JSON-in-prompt tool orchestration: we pass tool specs directly to
the LLM and let the provider emit structured tool_calls. The loop ends
when the LLM stops requesting tools (stop_reason == "end_turn") or we hit
`max_steps`. Duplicate tool calls (same name+args) within the loop are
short-circuited to prevent runaway loops.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm import LLMProvider, ToolCall
from src.logging_utils import log_event
from src.tools import ToolContext, ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    text: str
    image_url: str | None = None
    steps: int = 0
    tool_calls_count: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)


class SlackMentionAgent:
    def __init__(
        self,
        llm: LLMProvider,
        context: ToolContext,
        registry: ToolRegistry,
        max_steps: int,
        tool_executor: ToolExecutor | None = None,
        response_language: str = "ko",
        system_message: str | None = None,
        history: list[dict[str, Any]] | None = None,
        on_stream: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.context = context
        self.registry = registry
        self.max_steps = max_steps
        self.executor = tool_executor or ToolExecutor(context, registry)
        self.response_language = response_language
        self.system_message = system_message
        self.history = history or []
        self.on_stream = on_stream

    def run(self, user_message: str) -> AgentResult:
        system = self._build_system_prompt()
        messages: list[dict[str, Any]] = [*self.history, {"role": "user", "content": user_message}]
        seen_calls: set[str] = set()
        image_url: str | None = None
        total_usage = {"input": 0, "output": 0}
        tool_calls_total = 0
        steps = 0

        for step in range(self.max_steps):
            steps = step + 1
            result = self.llm.chat(system, messages, tools=self.registry.specs())
            total_usage["input"] += result.token_usage.get("input", 0)
            total_usage["output"] += result.token_usage.get("output", 0)
            log_event(logger, "llm.hop", step=steps, stop=result.stop_reason, tool_calls=len(result.tool_calls))

            if not result.tool_calls:
                return self._finalize(result.content, messages, image_url, steps, tool_calls_total, total_usage)

            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in result.tool_calls
                    ],
                }
            )

            for call in result.tool_calls:
                tool_calls_total += 1
                signature = self._call_signature(call)
                if signature in seen_calls:
                    tool_result = {"ok": False, "error": "duplicate call skipped"}
                else:
                    seen_calls.add(signature)
                    tool_result = self.executor.execute(call)
                log_event(logger, "tool.result", step=steps, tool=call.name, ok=tool_result.get("ok", False))

                if call.name == "generate_image" and tool_result.get("ok"):
                    permalink = (tool_result.get("result") or {}).get("permalink")
                    if permalink:
                        image_url = permalink

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        # max_steps reached — force one final compose without tools.
        final_text = self._compose_without_tools(system, messages)
        return AgentResult(text=final_text, image_url=image_url, steps=steps, tool_calls_count=tool_calls_total, token_usage=total_usage)

    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        base = (
            self.system_message
            or "You are an assistant for Slack mention requests. Plan work, call tools when needed, and provide concise helpful answers."
        )
        return f"{base}\n\nRespond in language: {self.response_language}."

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        args_blob = json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False)
        return f"{call.name}:{hashlib.sha1(args_blob.encode()).hexdigest()[:12]}"

    def _finalize(
        self,
        text: str,
        messages: list[dict[str, Any]],  # noqa: ARG002
        image_url: str | None,
        steps: int,
        tool_calls_count: int,
        token_usage: dict[str, int],
    ) -> AgentResult:
        clean = (text or "").strip()
        if self.on_stream and clean:
            self.on_stream(clean)
        return AgentResult(
            text=clean,
            image_url=image_url,
            steps=steps,
            tool_calls_count=tool_calls_count,
            token_usage=token_usage,
        )

    def _compose_without_tools(self, system: str, messages: list[dict[str, Any]]) -> str:
        """Force a final answer when max_steps is reached — no tools permitted."""
        directive = (
            "Provide the final answer now. Do not request any more tools; summarize based on prior observations."
        )
        messages = [*messages, {"role": "user", "content": directive}]
        if self.on_stream:
            return self.llm.stream_chat(system, messages, on_delta=self.on_stream)
        result = self.llm.chat(system, messages, tools=None)
        return (result.content or "").strip()
