"""
Local CLI test script – runs the SlackMentionAgent without an actual Slack connection.

Usage:
    python local_test.py "질문 내용"
    python local_test.py          # interactive mode (reads from stdin)

Environment:
    Copy .env.example to .env.local and fill in real values before running.
    At minimum, set OPENAI_API_KEY (and SLACK_BOT_TOKEN if you want real Slack tool calls).
"""

import sys
import types
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Stub Slack WebClient so Slack-dependent tools degrade gracefully when no
# real SLACK_BOT_TOKEN is provided (or the token is invalid).
# ---------------------------------------------------------------------------
class _StubSlackClient:
    """Returns empty but structurally correct responses for every Slack call."""

    def conversations_replies(self, **_):
        return {"messages": []}

    def search_messages(self, **_):
        return {"messages": {"matches": []}}

    def files_upload_v2(self, **_):
        return {"file": {"permalink": "", "title": "generated.png"}}


def _build_slack_client(token: str):
    if token and not token.startswith("xoxb-your"):
        try:
            from slack_sdk import WebClient
            return WebClient(token=token)
        except Exception:
            pass
    return _StubSlackClient()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # .env.local is loaded automatically inside src/config.py
    from src.config import Settings
    from src.llm import LLMClient
    from src.tools import ToolContext, ToolExecutor
    from src.agent import SlackMentionAgent

    settings = Settings.from_env()

    if not settings.slack_bot_token or settings.slack_bot_token.startswith("xoxb-your"):
        print("[경고] SLACK_BOT_TOKEN이 설정되지 않았습니다. Slack 관련 도구는 빈 결과를 반환합니다.\n")
    if not settings.llm_provider == "bedrock":
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("[오류] OPENAI_API_KEY가 설정되지 않았습니다. .env.local 을 확인하세요.")
            sys.exit(1)

    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        image_provider=settings.image_provider,
        image_model=settings.image_model,
    )

    slack_client = _build_slack_client(settings.slack_bot_token)

    context = ToolContext(
        slack_client=slack_client,
        channel="local",
        thread_ts="0",
        event={},
        settings=settings,
        llm=llm,
    )

    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
    )

    # ------------------------------------------------------------------
    # Get user message from CLI arg or stdin
    # ------------------------------------------------------------------
    if len(sys.argv) > 1:
        user_message = " ".join(sys.argv[1:])
    else:
        print("질문을 입력하세요 (Ctrl+D 로 종료):")
        try:
            user_message = sys.stdin.read().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

    if not user_message:
        print("[오류] 질문이 비어 있습니다.")
        sys.exit(1)

    print(f"\n▶ 질문: {user_message}\n")
    print("처리 중...\n")

    try:
        result = agent.run(user_message)
        print("─" * 60)
        print(result.text)
        if result.image_url:
            print(f"\n[이미지] {result.image_url}")
        print("─" * 60)
    except Exception as exc:
        print(f"[오류] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
