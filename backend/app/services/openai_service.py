from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.openai")


class OpenAIResponsesService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds, max_retries=0)
        self.system_prompt = self._load_system_prompt(settings.prompt_path)

    @staticmethod
    def _load_system_prompt(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def generate_json(self, user_prompt: str, conversation: list[dict[str, str]]) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        input_items: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_prompt}],
            }
        ]
        for message in conversation:
            input_items.append(
                {
                    "role": message["role"],
                    "content": [{"type": "input_text", "text": message["content"]}],
                }
            )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        )

        last_error: Exception | None = None
        for attempt in range(self.settings.openai_max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.settings.openai_model,
                    input=input_items,
                )
                text = response.output_text.strip()
                return self._parse_json(text)
            except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError) as exc:
                last_error = exc
                logger.warning("transient model error attempt=%s error=%s", attempt + 1, exc)
                time.sleep(min(2 ** attempt, 4))
            except Exception as exc:  # pragma: no cover
                logger.error("openai responses create failed error=%s", exc)
                raise

        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"answer": cleaned}
