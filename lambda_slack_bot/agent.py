import json
from dataclasses import dataclass
from typing import Any

from lambda_slack_bot.llm import LLMClient
from lambda_slack_bot.tools import ToolContext, ToolExecutor


@dataclass
class AgentResult:
    text: str
    image_url: str | None = None


class SlackMentionAgent:
    def __init__(self, llm: LLMClient, context: ToolContext, max_steps: int, tool_executor: ToolExecutor | None = None):
        self.llm = llm
        self.context = context
        self.max_steps = max_steps
        self.tools = tool_executor or ToolExecutor(context)

    def run(self, user_message: str) -> AgentResult:
        system = (
            "You are an assistant for Slack mention requests. "
            "Plan work, use tools when needed, and provide concise helpful final answers in Korean."
        )
        planner_prompt = (
            "Return strict JSON with keys: goal, plan (array), tool_calls (array of {name, arguments}), "
            "requires_image (boolean), image_prompt (string). "
            f"Available tools: {json.dumps(self.tools.available_tools, ensure_ascii=False)}."
        )
        state = self.llm.chat_json(system, f"{planner_prompt}\n\nUser message:\n{user_message}")

        observations: list[dict[str, Any]] = []
        next_calls = state.get("tool_calls", []) or []
        image_url = None

        for _ in range(self.max_steps):
            if not next_calls:
                break
            for call in next_calls:
                result = self.tools.execute(call.get("name", ""), call.get("arguments", {}))
                observations.append({"tool_call": call, "result": result})
                if call.get("name") == "generate_image" and result.get("ok"):
                    image_url = result.get("result", {}).get("permalink")

            state = self.llm.chat_json(
                system,
                "Based on observations, either request additional tool calls or propose final answer. "
                "Return strict JSON with keys: tool_calls, final_answer, requires_image, image_prompt.\n\n"
                f"User message:\n{user_message}\n\n"
                f"Observations:\n{json.dumps(observations, ensure_ascii=False)}",
            )
            next_calls = state.get("tool_calls", []) or []

        if state.get("requires_image") and state.get("image_prompt") and not image_url:
            generated = self.tools.execute("generate_image", {"prompt": state["image_prompt"]})
            if generated.get("ok"):
                image_url = generated.get("result", {}).get("permalink")

        final_text = self.llm.chat_text(
            system,
            "Compose the final Slack response. Use Korean by default and include clear bullet points when helpful.\n\n"
            f"User message:\n{user_message}\n\n"
            f"Plan:\n{json.dumps(state.get('plan', []), ensure_ascii=False)}\n\n"
            f"Observations:\n{json.dumps(observations, ensure_ascii=False)}\n\n"
            f"Draft answer:\n{state.get('final_answer', '')}",
        )

        return AgentResult(text=final_text.strip(), image_url=image_url)
