"""Local CLI test script — runs SlackMentionAgent without a real Slack connection.

Usage:
    python localtest.py "질문 내용"
    python localtest.py              # interactive mode (stdin, Ctrl+D to submit)
    python localtest.py --stream "질문 내용"   # show streaming deltas

Environment:
    Copy .env.example to .env.local and fill in values before running.
    Minimum required: OPENAI_API_KEY (for OpenAI provider).
"""
from __future__ import annotations

import argparse
import logging
import sys


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


class _StubSlackClient:
    """Returns empty but structurally correct responses for every Slack call."""

    def conversations_replies(self, **_):
        return {"messages": []}

    def search_messages(self, **_):
        return {"messages": {"matches": []}}

    def files_upload_v2(self, **_):
        return {"file": {"permalink": "", "title": "generated.png"}}

    def users_info(self, **_):
        return {"user": {"profile": {"display_name": "local-user"}}}


def _build_slack_client(token: str):
    if token and not token.startswith("xoxb-your"):
        try:
            from slack_sdk import WebClient

            return WebClient(token=token)
        except Exception:
            pass
    return _StubSlackClient()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local agent test runner.")
    parser.add_argument("question", nargs="*", help="Question text. If omitted, read from stdin.")
    parser.add_argument("--stream", action="store_true", help="Print streaming deltas as they arrive.")
    args = parser.parse_args()

    from src.agent import SlackMentionAgent
    from src.config import Settings
    from src.llm import get_llm
    from src.tools import ToolContext, default_registry

    settings = Settings.from_env()

    if not settings.slack_bot_token or settings.slack_bot_token.startswith("xoxb-your"):
        print("[경고] SLACK_BOT_TOKEN 미설정 — Slack 관련 도구는 빈 결과를 반환합니다.\n")
    if settings.llm_provider == "openai":
        import os

        if not os.getenv("OPENAI_API_KEY"):
            print("[오류] OPENAI_API_KEY가 설정되지 않았습니다. .env.local 을 확인하세요.")
            sys.exit(1)

    llm = get_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        image_provider=settings.image_provider,
        image_model=settings.image_model,
        region=settings.aws_region,
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

    if args.question:
        user_message = " ".join(args.question).strip()
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

    on_stream = None
    if args.stream:
        def on_stream(delta: str) -> None:  # noqa: RUF013
            sys.stdout.write(delta)
            sys.stdout.flush()

    agent = SlackMentionAgent(
        llm=llm,
        context=context,
        registry=default_registry,
        max_steps=settings.agent_max_steps,
        response_language=settings.response_language,
        system_message=settings.system_message,
        on_stream=on_stream,
    )

    try:
        result = agent.run(user_message)
    except Exception as exc:  # noqa: BLE001
        print(f"[오류] {exc}")
        sys.exit(1)

    if not args.stream:
        print("─" * 60)
        print(result.text)
        if result.image_url:
            print(f"\n[이미지] {result.image_url}")
        print("─" * 60)
    else:
        print()

    print(
        f"\nsteps={result.steps} tool_calls={result.tool_calls_count} "
        f"tokens_in={result.token_usage.get('input', 0)} tokens_out={result.token_usage.get('output', 0)}"
    )


if __name__ == "__main__":
    main()
