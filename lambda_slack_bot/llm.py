import base64
import json
from typing import Any

import boto3


class LLMClient:
    def __init__(self, provider: str, model: str, image_provider: str, image_model: str):
        self.provider = provider
        self.model = model
        self.image_provider = image_provider
        self.image_model = image_model
        self._bedrock = boto3.client("bedrock-runtime") if provider == "bedrock" or image_provider == "bedrock" else None

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "bedrock":
            return self._bedrock_chat(system_prompt, user_prompt)
        return self._openai_chat(system_prompt, user_prompt)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.chat_text(system_prompt, user_prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            return {}

    def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        if self.provider == "bedrock":
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
            response = self._bedrock.invoke_model(modelId=self.model, body=json.dumps(body))
            payload = json.loads(response["body"].read())
            return payload["content"][0]["text"]

        from openai import OpenAI

        client = OpenAI()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image for a Slack conversation."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ]}
            ],
        )
        return completion.choices[0].message.content or ""

    def generate_image(self, prompt: str) -> bytes:
        if self.image_provider == "bedrock":
            body = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {"numberOfImages": 1, "quality": "standard", "height": 1024, "width": 1024},
            }
            response = self._bedrock.invoke_model(modelId=self.image_model, body=json.dumps(body))
            payload = json.loads(response["body"].read())
            return base64.b64decode(payload["images"][0])

        from openai import OpenAI

        client = OpenAI()
        response = client.images.generate(model=self.image_model, prompt=prompt, size="1024x1024")
        return base64.b64decode(response.data[0].b64_json)

    def _openai_chat(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""

    def _bedrock_chat(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}],
                }
            ],
        }
        response = self._bedrock.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"]
