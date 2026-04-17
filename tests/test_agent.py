import unittest

from lambda_slack_bot.agent import SlackMentionAgent
from lambda_slack_bot.tools import ToolContext


class FakeLLM:
    def __init__(self):
        self.json_calls = 0

    def chat_json(self, _system, _prompt):
        self.json_calls += 1
        if self.json_calls == 1:
            return {
                "goal": "find answer",
                "plan": ["search web"],
                "tool_calls": [{"name": "search_web", "arguments": {"query": "aws lambda slack bolt"}}],
                "requires_image": False,
                "image_prompt": "",
            }
        return {
            "tool_calls": [],
            "final_answer": "done",
            "requires_image": False,
            "image_prompt": "",
            "plan": ["search web"],
        }

    def chat_text(self, _system, _prompt):
        return "최종 응답"


class FakeTools:
    def __init__(self):
        self.called = []
        self.available_tools = [{"name": "search_web", "description": "..."}]

    def execute(self, name, arguments):
        self.called.append((name, arguments))
        return {"ok": True, "result": [{"title": "AWS", "url": "https://aws.amazon.com"}]}


class AgentTests(unittest.TestCase):
    def test_runs_tool_then_returns_final_text(self):
        llm = FakeLLM()
        tools = FakeTools()
        context = ToolContext(
            slack_client=None,
            channel="C1",
            thread_ts="123.456",
            event={},
            settings=None,
            llm=llm,
        )

        agent = SlackMentionAgent(llm=llm, context=context, max_steps=2, tool_executor=tools)
        result = agent.run("질문")

        self.assertEqual(result.text, "최종 응답")
        self.assertIsNone(result.image_url)
        self.assertEqual(len(tools.called), 1)
        self.assertEqual(tools.called[0][0], "search_web")


if __name__ == "__main__":
    unittest.main()
