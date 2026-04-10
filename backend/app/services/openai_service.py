from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.openai")


class OpenAIResponsesService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Ignore broken shell/system proxy env vars for OpenAI requests.
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=settings.openai_timeout_seconds),
        )
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
        prompt_chars = len(user_prompt)
        conversation_chars = sum(len(message.get("content", "")) for message in conversation)
        estimated_input_chars = sum(self._count_input_item_chars(item) for item in input_items)
        logger.info(
            "openai request start model=%s attempts=%s prompt_chars=%s conversation_messages=%s conversation_chars=%s estimated_input_chars=%s",
            self.settings.openai_model,
            self.settings.openai_max_retries + 1,
            prompt_chars,
            len(conversation),
            conversation_chars,
            estimated_input_chars,
        )

        last_error: Exception | None = None
        for attempt in range(self.settings.openai_max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.settings.openai_model,
                    input=input_items,
                )
                text = self._extract_response_text(response)
                logger.info(
                    "openai response received attempt=%s response_id=%s status=%s output_chars=%s",
                    attempt + 1,
                    getattr(response, "id", None),
                    getattr(response, "status", None),
                    len(text),
                )
                payload = self._parse_json(text)
                answer = str(payload.get("answer") or "").strip()
                if not answer:
                    logger.warning(
                        "openai response missing answer attempt=%s response_id=%s output_chars=%s",
                        attempt + 1,
                        getattr(response, "id", None),
                        len(text),
                    )
                    raise RuntimeError("OpenAI response did not contain a usable answer.")
                return payload
            except RateLimitError as exc:
                last_error = exc
                if self._is_quota_error(exc):
                    logger.error("openai quota exceeded error=%s", exc)
                    raise RuntimeError("OpenAI quota is exhausted for the configured API key.") from exc
                logger.warning("transient model error attempt=%s error=%s", attempt + 1, exc)
                time.sleep(min(2 ** attempt, 4))
            except BadRequestError as exc:
                last_error = exc
                logger.error(
                    "openai bad request attempt=%s prompt_chars=%s estimated_input_chars=%s error=%s",
                    attempt + 1,
                    prompt_chars,
                    estimated_input_chars,
                    exc,
                )
                if self._is_prompt_too_large_error(exc):
                    raise RuntimeError(
                        "OpenAI rejected the request because the grounded prompt was too large."
                    ) from exc
                raise RuntimeError("OpenAI rejected the request payload.") from exc
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
        if not cleaned:
            return {}
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        candidates = [cleaned]
        extracted = OpenAIResponsesService._extract_json_object_text(cleaned)
        if extracted and extracted not in candidates:
            candidates.append(extracted)
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
        return {"answer": cleaned}

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if output_text:
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            item_content = getattr(item, "content", None)
            if item_content is None and isinstance(item, dict):
                item_content = item.get("content")
            for content in item_content or []:
                text_value = getattr(content, "text", None)
                if isinstance(content, dict):
                    text_value = content.get("text") or content.get("value")
                if text_value:
                    chunks.append(str(text_value).strip())
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def _extract_json_object_text(text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    @staticmethod
    def _count_input_item_chars(item: dict[str, Any]) -> int:
        total = 0
        for content in item.get("content", []):
            total += len(str(content.get("text") or ""))
        return total

    @staticmethod
    def _is_quota_error(exc: RateLimitError) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code != 429:
            return False
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error") or {}
            if error.get("code") == "insufficient_quota":
                return True
            if "quota" in str(error.get("message", "")).lower():
                return True
        return "insufficient_quota" in str(exc).lower()

    @staticmethod
    def _is_prompt_too_large_error(exc: BadRequestError) -> bool:
        text = str(exc).lower()
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error") or {}
            text = f"{text} {error.get('message', '')}".lower()
        markers = [
            "maximum context length",
            "too many tokens",
            "context length",
            "prompt is too long",
            "input is too long",
        ]
        return any(marker in text for marker in markers)
