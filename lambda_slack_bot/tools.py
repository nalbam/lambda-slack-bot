import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from slack_sdk import WebClient

from lambda_slack_bot.config import Settings
from lambda_slack_bot.llm import LLMClient


@dataclass
class ToolContext:
    slack_client: WebClient
    channel: str
    thread_ts: str
    event: dict[str, Any]
    settings: Settings
    llm: LLMClient


class ToolExecutor:
    def __init__(self, context: ToolContext):
        self.context = context
        self._tools = {
            "read_attached_images": self.read_attached_images,
            "fetch_thread_history": self.fetch_thread_history,
            "search_slack_messages": self.search_slack_messages,
            "search_web": self.search_web,
            "generate_image": self.generate_image,
        }

    @property
    def available_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "read_attached_images", "description": "Read and summarize attached images in the mention."},
            {"name": "fetch_thread_history", "description": "Fetch Slack thread history for context."},
            {"name": "search_slack_messages", "description": "Search Slack messages by query."},
            {"name": "search_web", "description": "Search the web for latest information."},
            {"name": "generate_image", "description": "Generate an image and upload it to Slack."},
        ]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return {"ok": True, "result": self._tools[name](**(arguments or {}))}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"ok": False, "error": str(exc)}

    def read_attached_images(self, limit: int = 3) -> list[dict[str, str]]:
        files = self.context.event.get("files", [])
        token = self.context.settings.slack_bot_token
        out = []
        for file_info in files[:limit]:
            if not str(file_info.get("mimetype", "")).startswith("image/"):
                continue
            req = urllib.request.Request(
                file_info["url_private_download"],
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:  # nosec B310
                data = response.read()
            out.append(
                {
                    "name": file_info.get("name", "image"),
                    "summary": self.context.llm.describe_image(data, file_info.get("mimetype", "image/png")),
                }
            )
        return out

    def fetch_thread_history(self, limit: int = 20) -> list[dict[str, str]]:
        res = self.context.slack_client.conversations_replies(
            channel=self.context.channel,
            ts=self.context.thread_ts,
            limit=limit,
        )
        return [
            {"user": item.get("user", ""), "text": item.get("text", "")}
            for item in res.get("messages", [])
        ]

    def search_slack_messages(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        res = self.context.slack_client.search_messages(query=query, count=limit)
        matches = res.get("messages", {}).get("matches", [])
        return [{"channel": m.get("channel", {}).get("name", ""), "text": m.get("text", "")} for m in matches]

    def search_web(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": 1, "no_html": 1})
        url = f"https://api.duckduckgo.com/?{params}"
        with urllib.request.urlopen(url, timeout=15) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for item in payload.get("RelatedTopics", []):
            if "Text" in item and "FirstURL" in item:
                results.append({"title": item["Text"], "url": item["FirstURL"]})
                if len(results) >= limit:
                    break
        if payload.get("AbstractURL"):
            results.insert(0, {"title": payload.get("AbstractText", ""), "url": payload["AbstractURL"]})
        return results[:limit]

    def generate_image(self, prompt: str) -> dict[str, str]:
        image_bytes = self.context.llm.generate_image(prompt)
        upload = self.context.slack_client.files_upload_v2(
            channel=self.context.channel,
            thread_ts=self.context.thread_ts,
            title="Generated image",
            filename="generated.png",
            file=image_bytes,
        )
        file_info = upload.get("file", {})
        return {"permalink": file_info.get("permalink", ""), "title": file_info.get("title", "generated.png")}
