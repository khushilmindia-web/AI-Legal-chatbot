from __future__ import annotations

import importlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache

from backend.paths import INTENTS_PATH
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.intent_service")

INTENT_THRESHOLD = 0.7
ALLOWED_INTENTS = {"greeting", "thanks", "goodbye", "advice", "legal_help", "procedural"}
CASELAW_MARKERS = (
    "section ",
    "article ",
    "judgment",
    "judgement",
    "case law",
    "citation",
    "precedent",
    "supreme court",
    "high court",
    "ipc",
    "crpc",
    "bns",
    "bnss",
    "constitution",
    "act",
    "rule",
)
PROCEDURAL_MARKERS = (
    "what should i do",
    "what can i do",
    "what to do",
    "how do i",
    "how to",
    "process",
    "procedure",
    "steps",
    "complaint",
    "fir",
    "cyber fraud",
    "consumer complaint",
    "police complaint",
    "bank fraud",
    "chargeback",
    "notice",
    "document",
    "documents",
    "verification",
    "where to file",
    "where should i file",
    "insect",
    "contaminated",
    "unsafe food",
    "food poisoning",
    "spoiled",
    "chocolate",
    "expired",
    "seller",
    "refund",
    "replacement",
    "damaged",
    "defective",
    "mobile snatched",
    "phone stolen",
    "landlord",
    "eviction",
    "deposit",
    "lockout",
)
LEGAL_HELP_PRIORITY_MARKERS = (
    "upi",
    "fake link",
    "debited",
    "debit",
    "fraud",
    "bank fraud",
    "cyber fraud",
    "phishing",
    "scam",
    "otp",
    "wallet",
    "money got debited",
    "money deducted",
    "bank account",
    "complaint",
    "fir",
    "police",
    "landlord",
    "tenant",
    "threat",
    "evict",
    "notice",
    "insect",
    "contaminated",
    "unsafe food",
    "food poisoning",
    "spoiled",
    "expired",
    "seller",
    "refund",
    "replacement",
    "damaged",
    "defective",
    "snatched",
    "stolen",
    "robbed",
    "deposit",
    "lockout",
)
PURE_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|namaste|hii+|heyy+)\W*$"
)
ISSUE_STATEMENT_MARKERS = (
    "my ",
    "i am ",
    "i'm ",
    "i was ",
    "someone ",
    "landlord",
    "tenant",
    "harass",
    "threat",
    "evict",
    "fraud",
    "scam",
    "bank",
    "police",
    "salary",
    "property",
    "notice",
    "complaint",
    "insect",
    "contaminated",
    "unsafe food",
    "food poisoning",
    "spoiled",
    "expired",
    "seller",
    "refund",
    "replacement",
    "damaged",
    "defective",
    "snatched",
    "stolen",
    "robbed",
    "abuse",
    "assault",
    "cheat",
    "problem",
    "issue",
)
TECHNICAL_TOPIC_MARKERS = (
    "api",
    "sdk",
    "token",
    "auth",
    "authentication",
    "oauth",
    "http",
    "https",
    "json",
    "payload",
    "endpoint",
    "request body",
    "response body",
    "status code",
    "backend",
    "frontend",
    "database",
    "python",
    "javascript",
    "fastapi",
    "react",
    "openai",
    "groq",
    "debug",
    "bug",
    "error",
)


@dataclass
class IntentDecision:
    handled: bool
    answer: str = ""
    intent: str | None = None
    confidence: float = 0.0
    source: str = "none"


class IntentRoutingService:
    def route(
        self,
        message: str,
        *,
        conversation_started: bool = False,
        active_intent: str | None = None,
    ) -> IntentDecision:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        if not normalized:
            return IntentDecision(handled=False)

        catalog = self._intent_catalog()
        if self._looks_like_technical_topic(normalized):
            return IntentDecision(
                handled=True,
                answer="I can help with legal issues here. For API or technical topics, please ask the technical question directly and I will answer it without using India Kanoon.",
                intent="technical_redirect",
                confidence=1.0,
                source="heuristic",
            )
        if self._looks_like_caselaw_query(normalized):
            return IntentDecision(handled=False)
        if self._looks_like_priority_legal_help_issue(normalized):
            answer = self._response_for_tag("legal_help", catalog) or self._response_for_tag("procedural", catalog)
            if answer:
                return IntentDecision(True, answer, "legal_help", 0.99, "heuristic")
        if self._looks_like_issue_statement(normalized):
            answer = self._response_for_tag("legal_help", catalog) or self._response_for_tag("procedural", catalog)
            if answer:
                return IntentDecision(True, answer, "legal_help", 0.96, "heuristic")
        if self._looks_like_procedural_query(normalized) or (
            conversation_started and active_intent == "legal_help"
        ):
            answer = self._response_for_tag("legal_help", catalog) or self._response_for_tag("procedural", catalog)
            if answer:
                return IntentDecision(True, answer, "legal_help", 0.95, "heuristic")
        heuristic = self._match_catalog_patterns(
            normalized,
            catalog,
            conversation_started=conversation_started,
        )
        if heuristic is not None:
            return heuristic

        bot = self._get_bot()
        if bot is None:
            return IntentDecision(handled=False)

        try:
            intents = bot.predict_class(message)
        except Exception as exc:
            logger.warning("intent bot predict failed error=%s", exc)
            return IntentDecision(handled=False)

        if not intents:
            return IntentDecision(handled=False)

        top_intent = intents[0]
        tag = str(top_intent.get("intent") or "").strip().lower()
        confidence = float(top_intent.get("probability") or 0.0)
        logger.info("intent prediction tag=%s confidence=%.3f", tag, confidence)
        if tag in {"greeting", "thanks", "goodbye"} and self._contains_substantive_legal_issue(normalized):
            return IntentDecision(handled=False, intent=tag, confidence=confidence)
        if conversation_started and tag in {"greeting", "thanks", "goodbye"}:
            return IntentDecision(handled=False, intent=tag, confidence=confidence)
        if tag == "greeting" and not self._is_pure_greeting(normalized):
            return IntentDecision(handled=False, intent=tag, confidence=confidence)
        if tag not in ALLOWED_INTENTS or confidence < INTENT_THRESHOLD:
            return IntentDecision(handled=False, intent=tag or None, confidence=confidence)
        if tag in {"legal_help", "procedural"}:
            answer = self._response_for_tag(tag, catalog) or self._response_for_tag("legal_help", catalog)
            if answer:
                return IntentDecision(True, answer, tag, confidence, "intent_bot")
            return IntentDecision(True, "", tag, confidence, "intent_bot")

        answer = self._response_for_tag(tag, catalog)
        if not answer:
            try:
                answer = str(bot.get_response(message) or "").strip()
            except Exception as exc:
                logger.warning("intent bot response failed tag=%s error=%s", tag, exc)
                return IntentDecision(handled=False, intent=tag, confidence=confidence)
        if not answer or answer.lower().startswith("sorry, i didn't understand"):
            return IntentDecision(handled=False, intent=tag, confidence=confidence)
        return IntentDecision(
            handled=True,
            answer=answer,
            intent=tag,
            confidence=confidence,
            source="intent_bot",
        )

    def _match_catalog_patterns(
        self,
        normalized: str,
        catalog: dict[str, dict[str, list[str]]],
        *,
        conversation_started: bool,
    ) -> IntentDecision | None:
        for tag, payload in catalog.items():
            if tag not in ALLOWED_INTENTS:
                continue
            if conversation_started and tag in {"greeting", "thanks", "goodbye"}:
                continue
            if tag == "greeting" and self._contains_substantive_legal_issue(normalized):
                continue
            for pattern in payload.get("patterns", []):
                pattern_normalized = re.sub(r"\s+", " ", pattern.strip()).lower()
                if not pattern_normalized:
                    continue
                if normalized == pattern_normalized:
                    answer = self._response_for_tag(tag, catalog)
                    if answer:
                        return IntentDecision(True, answer, tag, 1.0, "pattern")
        return None

    @staticmethod
    def _looks_like_caselaw_query(normalized: str) -> bool:
        return any(marker in normalized for marker in CASELAW_MARKERS)

    @staticmethod
    def _looks_like_procedural_query(normalized: str) -> bool:
        return any(marker in normalized for marker in PROCEDURAL_MARKERS)

    @staticmethod
    def _looks_like_priority_legal_help_issue(normalized: str) -> bool:
        return any(marker in normalized for marker in LEGAL_HELP_PRIORITY_MARKERS)

    @staticmethod
    def _looks_like_issue_statement(normalized: str) -> bool:
        return len(normalized.split()) >= 3 and any(marker in normalized for marker in ISSUE_STATEMENT_MARKERS)

    @staticmethod
    def _contains_substantive_legal_issue(normalized: str) -> bool:
        return IntentRoutingService._looks_like_priority_legal_help_issue(normalized) or IntentRoutingService._looks_like_issue_statement(normalized) or IntentRoutingService._looks_like_procedural_query(normalized)

    @staticmethod
    def _is_pure_greeting(normalized: str) -> bool:
        return bool(PURE_GREETING_PATTERN.match(normalized))

    @staticmethod
    def _looks_like_technical_topic(normalized: str) -> bool:
        for marker in TECHNICAL_TOPIC_MARKERS:
            if " " in marker:
                if marker in normalized:
                    return True
                continue
            if re.search(rf"\b{re.escape(marker)}\b", normalized):
                return True
        return False

    @staticmethod
    def _response_for_tag(tag: str, catalog: dict[str, dict[str, list[str]]]) -> str:
        responses = catalog.get(tag, {}).get("responses", [])
        if not responses:
            return ""
        return random.choice(responses).strip()

    @staticmethod
    @lru_cache(maxsize=1)
    def _intent_catalog() -> dict[str, dict[str, list[str]]]:
        try:
            payload = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("intent catalog unavailable error=%s", exc)
            return {}

        catalog: dict[str, dict[str, list[str]]] = {}
        for item in payload.get("intents", []):
            tag = str(item.get("tag") or "").strip().lower()
            if not tag:
                continue
            catalog[tag] = {
                "patterns": [str(value).strip() for value in item.get("patterns", []) if str(value).strip()],
                "responses": [str(value).strip() for value in item.get("responses", []) if str(value).strip()],
            }
        return catalog

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_bot():
        try:
            module = importlib.import_module("backend.Intent_Bot.Intent_bot")
            return module.IntentBot()
        except Exception as exc:
            logger.warning("intent bot unavailable error=%s", exc)
            return None
