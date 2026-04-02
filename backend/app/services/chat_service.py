from __future__ import annotations

import json
import re
from datetime import datetime

from backend.app.core.config import Settings
from backend.app.models.schemas import ChatRequest, ChatUploadResponse, InternalChatResult
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.openai_service import OpenAIResponsesService
from backend.app.services.retrieval import RetrievalService
from backend.app.services.session_store import SessionStore
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.chat_service")


class ChatService:
    def __init__(self, settings: Settings, store: SessionStore) -> None:
        self.settings = settings
        self.store = store
        self.retrieval = RetrievalService(settings)
        self.extractor = FileExtractionService(settings)
        self.openai = OpenAIResponsesService(settings)

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

        history = self.store.get_messages(chat_id, user_id=user_id)
        domain = self._detect_domain(message)
        resolved_state = request.state or fallback_state or self.settings.default_state
        immediate_help = self._should_help_first(message, domain)
        follow_up_question = None if immediate_help else self._choose_follow_up(request, domain, message, resolved_state)
        retrieval_chunks = self.retrieval.get_context(
            query=message,
            uploaded_texts=uploaded_texts or [],
            state=resolved_state,
            domain=domain,
        )
        citations = [chunk.source for chunk in retrieval_chunks[:3]]
        grounded_context = "\n\n".join(
            [f"Source: {chunk.source}\n{chunk.text[:1200]}" for chunk in retrieval_chunks[:4]]
        )
        prompt = self._build_prompt(
            request=request,
            domain=domain,
            resolved_state=resolved_state,
            grounded_context=grounded_context,
            follow_up_question=follow_up_question,
            immediate_help=immediate_help,
            extraction_warnings=extraction_warnings or [],
        )

        conversation = [{"role": item["role"], "content": item["content"]} for item in history[-8:]]
        try:
            model_payload = self.openai.generate_json(prompt, conversation=conversation)
            internal = self._normalize_model_payload(model_payload, domain, citations, extraction_warnings or [])
        except Exception as exc:
            logger.error("model generation failed error=%s", exc)
            internal = InternalChatResult(
                answer="I could not generate a reliable legal guidance response just now. Please try again shortly. This is general legal information, not a substitute for professional legal representation.",
                domain=domain,
                follow_up_question=follow_up_question,
                citations=citations,
                authorities=self._default_authorities(domain),
                documents_to_keep=self._default_documents(domain),
                likely_forum=self._default_forum(domain, resolved_state, request.district),
                caution="The answer could not be completed due to a model or network issue.",
                warnings=["Temporary LLM failure handled safely."],
            )

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

    def _should_help_first(self, message: str, domain: str) -> bool:
        normalized = message.lower()
        if domain == "cyber" and any(term in normalized for term in ["otp", "upi", "debited", "fake link", "bank account"]):
            return True
        if len(message.split()) >= 10:
            return True
        return any(term in normalized for term in ["my ", "i ", "received", "got", "landlord", "police", "notice"])

    def _choose_follow_up(self, request: ChatRequest, domain: str, message: str, resolved_state: str | None) -> str | None:
        normalized = message.lower()
        if domain == "cyber" and not any(term in normalized for term in ["bank", "upi", "wallet", "app", "account"]):
            return "Which bank, app, or payment platform was involved?"
        if domain == "criminal" and not request.case_stage:
            return "Has a police complaint or FIR already been filed?"
        if domain == "property" and not request.district:
            return "Which district is the property located in?"
        if domain in {"property", "constitutional"} and not resolved_state:
            return "Please confirm the State where this issue happened or where you may need to take legal action."
        if domain == "consumer" and not resolved_state:
            return "Please confirm the State where the seller, service provider, or transaction is connected."
        if domain == "general" and len(message.split()) < 8:
            return "Please share one more key fact so I can guide you properly, such as the issue type or what exactly happened."
        return None

    def _build_prompt(
        self,
        request: ChatRequest,
        domain: str,
        resolved_state: str,
        grounded_context: str,
        follow_up_question: str | None,
        immediate_help: bool,
        extraction_warnings: list[str],
    ) -> str:
        guidance = {
            "message": request.message,
            "domain": domain,
            "state": resolved_state,
            "district": request.district,
            "case_stage": request.case_stage,
            "is_own_matter": request.is_own_matter,
            "immediate_help": immediate_help,
            "follow_up_question": follow_up_question,
            "grounded_context": grounded_context,
            "extraction_warnings": extraction_warnings,
            "required_output": {
                "answer": "string",
                "follow_up_question": "string or null",
                "authorities": ["string"],
                "documents_to_keep": ["string"],
                "likely_forum": "string",
                "caution": "string",
                "citations": ["string"],
            },
        }
        return (
            "Return valid JSON only.\n"
            "If the user already described a concrete incident, help first and only ask one targeted follow-up if truly needed.\n"
            "Keep the answer concise, practical, and simple.\n"
            "Use only grounded context when citing law or authorities. If uncertain, say so clearly.\n\n"
            + json.dumps(guidance, ensure_ascii=False, indent=2)
        )

    def _normalize_model_payload(
        self,
        payload: dict,
        domain: str,
        citations: list[str],
        warnings: list[str],
    ) -> InternalChatResult:
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            answer = "I do not have enough grounded information to give a reliable answer yet. Please share one more key fact. This is general legal information, not a substitute for professional legal representation."
        if "general legal information" not in answer.lower():
            answer = answer.rstrip() + "\n\nThis is general legal information, not a substitute for professional legal representation."
        return InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=payload.get("follow_up_question"),
            citations=[str(item) for item in (payload.get("citations") or citations or [])][:5],
            authorities=[str(item) for item in (payload.get("authorities") or self._default_authorities(domain))][:5],
            documents_to_keep=[str(item) for item in (payload.get("documents_to_keep") or self._default_documents(domain))][:5],
            likely_forum=str(payload.get("likely_forum") or self._default_forum(domain, None, None)),
            caution=str(payload.get("caution") or self._default_caution(domain)),
            warnings=warnings,
            raw_json=payload,
        )

    def _default_authorities(self, domain: str) -> list[str]:
        defaults = {
            "cyber": ["Information Technology Act, 2000", "Bank and payment complaint channels, where applicable"],
            "criminal": ["Bharatiya Nagarik Suraksha Sanhita, 2023", "Bharatiya Nyaya Sanhita, 2023"],
            "consumer": ["Consumer Protection Act, 2019"],
            "property": ["Transfer of Property Act, 1882", "State-specific rent or land records rules, where applicable"],
            "constitutional": ["Constitution of India"],
            "general": ["Applicable Indian law depends on the exact facts and State-specific context"],
        }
        return defaults.get(domain, defaults["general"])

    def _default_documents(self, domain: str) -> list[str]:
        defaults = {
            "cyber": ["transaction IDs", "screenshots", "bank complaint acknowledgements", "SMS or OTP trail"],
            "criminal": ["complaint copy", "FIR copy if any", "identity proof", "incident chronology"],
            "consumer": ["invoice", "payment proof", "emails or complaint screenshots"],
            "property": ["rent agreement or title papers", "notices", "payment or possession proof"],
            "constitutional": ["order, notice, or government action record if available"],
            "general": ["documents directly linked to the dispute", "dates, notices, and communication records"],
        }
        return defaults.get(domain, defaults["general"])

    def _default_forum(self, domain: str, state: str | None, district: str | None) -> str:
        place = ", ".join([item for item in [district, state] if item]) or "your local jurisdiction"
        defaults = {
            "cyber": f"Cyber cell, bank grievance mechanism, or local police station in {place}",
            "criminal": f"Local police station or Magistrate court in {place}",
            "consumer": f"District Consumer Commission in {place}",
            "property": f"Civil court, rent forum, or local revenue office in {place}",
            "constitutional": f"High Court with jurisdiction over {place}",
            "general": f"The right forum in {place} depends on the exact dispute and stage",
        }
        return defaults.get(domain, defaults["general"])

    def _default_caution(self, domain: str) -> str:
        cautions = {
            "cyber": "Act quickly. Delays can make fund tracing and complaint escalation harder.",
            "criminal": "Police and court procedure depends on the stage and exact allegations.",
            "property": "Property and tenancy remedies are often strongly affected by State-specific law.",
            "consumer": "Keep written proof and complaint records before taking the next procedural step.",
            "general": "State-specific law and facts may change the legal position.",
        }
        return cautions.get(domain, cautions["general"])
