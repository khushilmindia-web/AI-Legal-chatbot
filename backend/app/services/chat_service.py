from __future__ import annotations

import re
from datetime import datetime

import requests

from backend.app.core.config import Settings
from backend.app.models.schemas import ChatRequest, ChatUploadResponse, InternalChatResult
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.session_store import SessionStore
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.chat_service")


class ChatService:
    def __init__(self, settings: Settings, store: SessionStore) -> None:
        self.settings = settings
        self.store = store
        self.extractor = FileExtractionService(settings)
        self.indiankanoon = IndianKanoonService(settings)

    async def handle_chat(
        self,
        request: ChatRequest,
        user_id: int,
        fallback_state: str | None = None,
        uploaded_texts: list[str] | None = None,
        extraction_warnings: list[str] | None = None,
    ) -> ChatUploadResponse:
        message = request.message.strip()
        if not message:
            raise ValueError("Message cannot be empty")

        chat_id = request.chat_id
        if chat_id is None:
            session = self.store.create_session(self._generate_title(message), user_id=user_id)
            chat_id = session["id"]
        elif self.store.get_session(chat_id, user_id=user_id) is None:
            raise ValueError("Chat session not found")

        domain = self._detect_domain(message)
        resolved_state = request.state or fallback_state or self.settings.default_state
        warnings = list(extraction_warnings or [])
        if uploaded_texts:
            warnings.append("Uploaded files were received but chat answers now use only India Kanoon API search results.")

        search_results = self._search_indiankanoon_results(
            query=message,
            state=resolved_state,
            domain=domain,
        )
        internal = self._build_indiankanoon_result(search_results, domain, warnings)

        self.store.add_message(chat_id, "user", message, metadata={"domain": domain}, user_id=user_id)
        self.store.add_message(
            chat_id,
            "assistant",
            internal.answer,
            metadata={
                "domain": internal.domain,
                "follow_up_question": internal.follow_up_question,
                "citations": internal.citations,
                "authorities": internal.authorities,
                "documents_to_keep": internal.documents_to_keep,
                "likely_forum": internal.likely_forum,
                "caution": internal.caution,
                "warnings": internal.warnings,
            },
            user_id=user_id,
        )

        session = self.store.get_session(chat_id, user_id=user_id)
        if session is None:
            raise ValueError("Chat session not found")
        return ChatUploadResponse(
            chat_id=chat_id,
            title=session["title"],
            answer=internal.answer,
            created_at=datetime.fromisoformat(session["updated_at"]),
            domain=internal.domain,
            follow_up_question=internal.follow_up_question,
            citations=internal.citations,
            warnings=internal.warnings,
            authorities=internal.authorities,
            documents_to_keep=internal.documents_to_keep,
            likely_forum=internal.likely_forum,
            caution=internal.caution,
        )

    def _generate_title(self, message: str) -> str:
        cleaned = re.sub(r"\s+", " ", message.strip()).rstrip("?.!,;:")
        words = cleaned.split()
        title = " ".join(words[:7])
        if len(words) > 7:
            title += "..."
        return title or "New Chat"

    def _detect_domain(self, message: str) -> str:
        normalized = message.lower()
        rules = {
            "cyber": ["upi", "otp", "phishing", "cyber", "debit", "wallet", "fake link", "bank account"],
            "criminal": ["fir", "police", "arrest", "bail", "crime", "theft", "fraud complaint"],
            "consumer": ["refund", "defective", "consumer", "service", "seller", "e-commerce", "warranty"],
            "property": ["landlord", "tenant", "rent", "property", "eviction", "sale deed", "mutation"],
            "constitutional": ["article", "constitution", "fundamental right", "writ", "article 21", "article 14"],
        }
        for domain, keywords in rules.items():
            if any(keyword in normalized for keyword in keywords):
                return domain
        return "general"

    def _search_indiankanoon_results(self, query: str, state: str | None, domain: str) -> list[dict[str, str]]:
        if not self.indiankanoon.configured:
            logger.warning("indiankanoon token missing for chat query=%r", query[:80])
            return []
        try:
            return self.indiankanoon.search_references(
                query=query,
                doctypes=self._resolve_doctypes(state=state, domain=domain),
                max_results=3,
            )
        except (requests.RequestException, ValueError) as exc:
            logger.warning("indiankanoon search failed query=%r error=%s", query[:80], exc)
            return []

    def _build_indiankanoon_result(
        self,
        search_results: list[dict[str, str]],
        domain: str,
        warnings: list[str],
    ) -> InternalChatResult:
        if not search_results:
            return InternalChatResult(
                answer="No relevant legal data found on India Kanoon",
                domain=domain,
                follow_up_question=None,
                citations=[],
                authorities=[],
                documents_to_keep=[],
                likely_forum=None,
                caution=None,
                warnings=warnings,
                raw_json={"results": []},
            )

        lines = ["India Kanoon search results:"]
        citations: list[str] = []
        authorities: list[str] = []
        for index, item in enumerate(search_results, start=1):
            title = item.get("title") or "Untitled"
            headline = self._clean_search_snippet(item.get("headline") or "")
            court = item.get("docsource") or "India Kanoon"
            url = item.get("url") or ""
            lines.append(f"{index}. {title}")
            lines.append(f"Court: {court}")
            if headline:
                lines.append(f"Snippet: {headline}")
            if url:
                lines.append(f"Link: {url}")
                citations.append(f"{title} | {court} | {url}")
            else:
                citations.append(f"{title} | {court}")
            if court not in authorities:
                authorities.append(court)

        top_source = search_results[0].get("docsource") or None
        return InternalChatResult(
            answer="\n".join(lines),
            domain=domain,
            follow_up_question=None,
            citations=citations[:5],
            authorities=authorities[:5],
            documents_to_keep=[],
            likely_forum=top_source,
            caution=None,
            warnings=warnings,
            raw_json={"results": search_results},
        )

    @staticmethod
    def _clean_search_snippet(text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:400]

    def _resolve_doctypes(self, state: str | None, domain: str | None) -> str:
        state_map = {
            "gujarat": "gujarat",
            "delhi": "delhi,delhidc",
            "maharashtra": "bombay",
            "karnataka": "karnataka",
            "kerala": "kerala",
            "tamil nadu": "chennai",
            "west bengal": "kolkata",
            "uttar pradesh": "allahabad,lucknow",
            "rajasthan": "rajasthan,jodhpur",
        }
        domain_defaults = {
            "consumer": "consumer,judgments",
            "constitutional": "judgments,laws",
            "criminal": "judgments,laws",
            "property": "judgments,laws",
            "cyber": "judgments,laws",
            "general": "judgments,laws",
        }
        state_key = (state or "").strip().lower()
        if state_key in state_map:
            return state_map[state_key]
        return domain_defaults.get(domain or "general", "judgments,laws")
