from __future__ import annotations

import re
from datetime import datetime, timezone

import requests

from backend.app.core.config import Settings
from backend.app.models.schemas import ChatRequest, ChatUploadResponse, ConversationState, InternalChatResult
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.intent_service import IntentRoutingService
from backend.app.services.session_store import SessionStore
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.chat_service")


class ChatService:
    def __init__(self, settings: Settings, store: SessionStore) -> None:
        self.settings = settings
        self.store = store
        self.extractor = FileExtractionService(settings)
        self.indiankanoon = IndianKanoonService(settings)
        self.intent_router = IntentRoutingService()

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

        conversation_state = ConversationState.model_validate(
            self.store.get_conversation_state(chat_id, user_id=user_id)
        )
        domain = self._detect_domain(message)
        resolved_state = request.state or fallback_state or self.settings.default_state
        warnings = list(extraction_warnings or [])
        if uploaded_texts:
            warnings.append("Uploaded files were received but chat answers now use only India Kanoon API search results.")

        try:
            internal, next_state = self._generate_chat_result(
                message=message,
                domain=domain,
                resolved_state=resolved_state,
                warnings=warnings,
                conversation_state=conversation_state,
            )
        except Exception:
            logger.exception("chat response generation failed query=%r domain=%s", message[:120], domain)
            internal = self._build_fallback_result(domain=domain, warnings=warnings)
            next_state = conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "fallback",
                    "awaiting_details": False,
                    "legal_domain": domain,
                }
            )

        assistant_metadata = {
            "domain": internal.domain,
            "follow_up_question": internal.follow_up_question,
            "citations": internal.citations,
            "authorities": internal.authorities,
            "documents_to_keep": internal.documents_to_keep,
            "likely_forum": internal.likely_forum,
            "caution": internal.caution,
            "warnings": internal.warnings,
            "conversation_state": next_state.model_dump(),
        }

        try:
            self.store.add_message(
                chat_id,
                "user",
                message,
                metadata={
                    "domain": domain,
                    "conversation_state": conversation_state.model_dump(),
                },
                user_id=user_id,
            )
            self.store.add_message(chat_id, "assistant", internal.answer, metadata=assistant_metadata, user_id=user_id)
            self.store.update_conversation_state(chat_id, next_state.model_dump(), user_id=user_id)
        except Exception:
            logger.exception("chat persistence failed chat_id=%s user_id=%s", chat_id, user_id)

        session = self.store.get_session(chat_id, user_id=user_id)
        created_at = datetime.fromisoformat(session["updated_at"]) if session is not None else datetime.now(timezone.utc)
        title = session["title"] if session is not None else self._generate_title(message)
        return ChatUploadResponse(
            chat_id=chat_id,
            title=title,
            answer=internal.answer,
            created_at=created_at,
            domain=internal.domain,
            follow_up_question=internal.follow_up_question,
            citations=internal.citations,
            warnings=internal.warnings,
            authorities=internal.authorities,
            documents_to_keep=internal.documents_to_keep,
            likely_forum=internal.likely_forum,
            caution=internal.caution,
        )

    def _generate_chat_result(
        self,
        *,
        message: str,
        domain: str,
        resolved_state: str | None,
        warnings: list[str],
        conversation_state: ConversationState,
    ) -> tuple[InternalChatResult, ConversationState]:
        if conversation_state.active_intent == "legal_help" and conversation_state.conversation_started:
            enriched_state = self._enrich_legal_help_state(
                current_state=conversation_state,
                message=message,
                domain=conversation_state.legal_domain or domain,
            )
            internal = self._build_legal_help_result(
                query=self._merge_legal_help_context(enriched_state, message),
                domain=enriched_state.legal_domain or domain,
                warnings=warnings,
                intro=self._continuation_intro(message),
                conversation_state=enriched_state,
            )
            next_state = self._updated_conversation_state(
                current_state=enriched_state,
                message=message,
                internal=internal,
                domain=enriched_state.legal_domain or domain,
                forced_intent="legal_help",
            )
            return internal, next_state

        intent_result = self.intent_router.route(
            message,
            conversation_started=conversation_state.conversation_started,
            active_intent=conversation_state.active_intent,
        )
        if intent_result.handled:
            logger.info(
                "chat handled by intent route intent=%s confidence=%.3f source=%s",
                intent_result.intent,
                intent_result.confidence,
                intent_result.source,
            )
            if intent_result.intent in {"legal_help", "procedural"}:
                internal = self._build_legal_help_result(
                    query=message,
                    domain=domain,
                    warnings=warnings,
                    intro=intent_result.answer,
                    conversation_state=self._enrich_legal_help_state(
                        current_state=conversation_state,
                        message=message,
                        domain=domain,
                    ),
                )
                next_state = self._updated_conversation_state(
                    current_state=conversation_state,
                    message=message,
                    internal=internal,
                    domain=domain,
                    forced_intent="legal_help",
                )
            else:
                internal = self._build_intent_result(
                    answer=intent_result.answer,
                    domain=domain,
                    warnings=warnings,
                )
                next_state = self._updated_conversation_state(
                    current_state=conversation_state,
                    message=message,
                    internal=internal,
                    domain=domain,
                    forced_intent=intent_result.intent,
                )
            return internal, next_state

        if self._should_use_indiankanoon(message):
            search_results = self._search_indiankanoon_results(
                query=message,
                state=resolved_state,
                domain=domain,
            )
            if search_results:
                internal = self._build_indiankanoon_result(search_results, domain, warnings)
                next_state = self._updated_conversation_state(
                    current_state=conversation_state,
                    message=message,
                    internal=internal,
                    domain=domain,
                    forced_intent="indiankanoon",
                )
                return internal, next_state

        internal = self._build_legal_help_result(
            query=message,
            domain=domain,
            warnings=warnings,
            conversation_state=self._enrich_legal_help_state(
                current_state=conversation_state,
                message=message,
                domain=domain,
            ),
        )
        next_state = self._updated_conversation_state(
            current_state=conversation_state,
            message=message,
            internal=internal,
            domain=domain,
            forced_intent="legal_help",
        )
        return internal, next_state

    def _updated_conversation_state(
        self,
        *,
        current_state: ConversationState,
        message: str,
        internal: InternalChatResult,
        domain: str,
        forced_intent: str | None,
    ) -> ConversationState:
        active_intent = forced_intent or current_state.active_intent
        next_issue = message if active_intent == "legal_help" and not current_state.last_user_issue else current_state.last_user_issue
        if active_intent == "legal_help" and current_state.last_user_issue:
            next_issue = self._merge_text(current_state.last_user_issue, message)
        issue_type = current_state.issue_type
        city = current_state.city
        police_station = current_state.police_station
        bank_name = current_state.bank_name
        platform = current_state.platform
        notice_stage = current_state.notice_stage
        document_type = current_state.document_type
        guidance_key = current_state.last_guidance_key

        if active_intent == "legal_help":
            state_from_raw = self._enrich_legal_help_state(
                current_state=current_state,
                message=next_issue or message,
                domain=domain,
            )
            issue_type = state_from_raw.issue_type
            city = state_from_raw.city
            police_station = state_from_raw.police_station
            bank_name = state_from_raw.bank_name
            platform = state_from_raw.platform
            notice_stage = state_from_raw.notice_stage
            document_type = state_from_raw.document_type
            guidance_key = self._guidance_key_for_state(state_from_raw, internal.follow_up_question)
        return current_state.model_copy(
            update={
                "conversation_started": True,
                "active_intent": active_intent,
                "awaiting_details": bool(internal.follow_up_question) and active_intent == "legal_help",
                "last_user_issue": next_issue if active_intent == "legal_help" else message,
                "legal_domain": domain if active_intent == "legal_help" else current_state.legal_domain,
                "last_follow_up_question": internal.follow_up_question,
                "issue_type": issue_type,
                "city": city,
                "police_station": police_station,
                "bank_name": bank_name,
                "platform": platform,
                "notice_stage": notice_stage,
                "document_type": document_type,
                "last_guidance_key": guidance_key,
            }
        )

    @staticmethod
    def _merge_legal_help_context(conversation_state: ConversationState, message: str) -> str:
        if not conversation_state.last_user_issue:
            return message
        return ChatService._merge_text(conversation_state.last_user_issue, message)

    @staticmethod
    def _merge_text(previous: str, current: str) -> str:
        previous_clean = previous.strip()
        current_clean = current.strip()
        if not previous_clean:
            return current_clean
        if not current_clean:
            return previous_clean
        return f"{previous_clean}\n{current_clean}"

    @staticmethod
    def _continuation_intro(message: str) -> str:
        normalized = message.strip()
        if not normalized:
            return "Based on what you've shared so far, here are the next practical steps."
        return "Based on what you've shared so far, here are the next practical steps, rights, and action points."

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
            "criminal": [
                "fir",
                "police",
                "arrest",
                "bail",
                "crime",
                "theft",
                "fraud complaint",
                "ipc",
                "crpc",
                "bns",
                "bnss",
                "section ",
            ],
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
            query_variants = self._build_query_variants(query)
            doctypes_options = self._resolve_doctype_candidates(query=query, state=state, domain=domain)
            results = self.indiankanoon.search_references_multi(
                query_variants=query_variants,
                doctypes_options=doctypes_options,
                max_results=3,
            )
            logger.info(
                "indiankanoon search completed query=%r domain=%s state=%s doctypes=%s query_variants=%s results=%s",
                query[:80],
                domain,
                state,
                doctypes_options,
                len(query_variants),
                len(results),
            )
            return results
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
            return self._build_fallback_result(domain=domain, warnings=warnings)

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

    def _build_intent_result(self, answer: str, domain: str, warnings: list[str]) -> InternalChatResult:
        return InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=None,
            citations=[],
            authorities=[],
            documents_to_keep=[],
            likely_forum=None,
            caution=None,
            warnings=warnings,
            raw_json={"results": [], "source": "intent"},
        )

    def _build_legal_help_result(
        self,
        *,
        query: str,
        domain: str,
        warnings: list[str],
        intro: str | None = None,
        conversation_state: ConversationState | None = None,
    ) -> InternalChatResult:
        lines: list[str] = []
        intro_text = (intro or "").strip()
        if intro_text:
            lines.append(intro_text)
        effective_state = conversation_state or ConversationState()
        if not effective_state.issue_type:
            effective_state = self._enrich_legal_help_state(
                current_state=effective_state,
                message=query,
                domain=domain,
            )
        guidance = self._legal_help_guidance(query=query, domain=domain, conversation_state=effective_state)
        lines.extend(guidance["lines"])
        return InternalChatResult(
            answer="\n".join(lines).strip(),
            domain=domain,
            follow_up_question=guidance.get("follow_up_question"),
            citations=[],
            authorities=guidance.get("authorities", []),
            documents_to_keep=guidance.get("documents_to_keep", []),
            likely_forum=guidance.get("likely_forum"),
            caution=guidance.get("caution"),
            warnings=warnings,
            raw_json={"results": [], "source": "legal_help"},
        )

    def _build_fallback_result(self, domain: str, warnings: list[str]) -> InternalChatResult:
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

    @staticmethod
    def _clean_search_snippet(text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:400]

    def _should_use_indiankanoon(self, query: str) -> bool:
        normalized = query.lower()
        caselaw_signals = [
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
        ]
        return any(signal in normalized for signal in caselaw_signals)

    def _legal_help_guidance(
        self,
        *,
        query: str,
        domain: str,
        conversation_state: ConversationState,
    ) -> dict[str, str | list[str] | None]:
        normalized = query.lower()
        issue_type = conversation_state.issue_type or self._detect_issue_type(normalized, domain)
        if issue_type == "cyber_fraud":
            return self._cyber_fraud_guidance(conversation_state)
        if issue_type == "consumer":
            return self._consumer_guidance(conversation_state)
        if issue_type == "notice":
            return self._notice_guidance(conversation_state)
        if issue_type == "police_complaint":
            return self._police_guidance(conversation_state)
        if issue_type == "documents":
            return self._documents_guidance(conversation_state)
        if any(token in normalized for token in {"cyber fraud", "upi", "wallet", "bank fraud", "phishing", "fake link", "otp", "debit"}):
            return {
                "lines": [
                    "Immediate steps: block the payment channel, call the bank or wallet helpline, and secure the account credentials.",
                    "File a complaint on the National Cyber Crime Portal or call 1930 as early as possible.",
                    "Keep transaction IDs, screenshots, bank SMS alerts, account statements, and complaint acknowledgements ready.",
                ],
                "likely_forum": "Cyber Crime Portal / Police Station / Bank",
                "authorities": ["cyber cell", "bank", "police"],
                "documents_to_keep": ["transaction ID", "screenshots", "bank statement", "complaint acknowledgement"],
                "caution": "Act quickly because reversal and fraud-tracing options are time-sensitive.",
                "follow_up_question": "Which platform or bank account was involved in the fraud?",
            }
        if "consumer" in normalized or any(token in normalized for token in {"refund", "defective", "seller", "service", "warranty"}):
            return {
                "lines": [
                    "Start by sending a written complaint to the seller or service provider and keep proof of delivery.",
                    "Collect the invoice, payment proof, chats or emails, warranty terms, and photographs of the defect or deficiency.",
                    "If the issue is not resolved, prepare for a consumer complaint with the product details, loss suffered, and relief claimed.",
                ],
                "likely_forum": "Consumer Commission",
                "authorities": ["seller", "consumer commission"],
                "documents_to_keep": ["invoice", "payment proof", "complaint copy", "photos"],
                "caution": "Make sure the chronology and the exact refund or compensation amount are clearly documented.",
                "follow_up_question": "Was any written complaint already sent to the seller or service provider?",
            }
        if "notice" in normalized:
            return {
                "lines": [
                    "Before sending or replying to a legal notice, organize the agreement, messages, invoices, payment records, and timeline of events.",
                    "Check the exact demand, deadline, and the relief being claimed before drafting a reply.",
                    "A proper reply usually states the facts, records your position, and attaches the key supporting documents.",
                ],
                "likely_forum": "Pre-litigation / Advocate Notice",
                "authorities": ["advocate"],
                "documents_to_keep": ["notice copy", "agreement", "payment proof", "communications"],
                "caution": "Do not ignore the notice deadline if a response has been demanded.",
                "follow_up_question": "Have you received the notice already, or are you planning to send one?",
            }
        if any(token in normalized for token in {"fir", "police complaint", "police", "complaint"}):
            return {
                "lines": [
                    "Write a short complaint with dates, place, names, and the exact incident in chronological order.",
                    "Carry your ID proof and supporting documents such as screenshots, recordings, receipts, or witness details.",
                    "If the matter involves a cognizable offence, ask for FIR registration and keep the complaint acknowledgement or diary number.",
                ],
                "likely_forum": "Police Station",
                "authorities": ["police"],
                "documents_to_keep": ["written complaint", "ID proof", "screenshots", "supporting records"],
                "caution": "Keep a copy of every complaint submitted and note the date, officer, and acknowledgement number.",
                "follow_up_question": "Which city or police station is connected with the complaint?",
            }
        if any(token in normalized for token in {"document", "documents", "verification", "certificate"}):
            return {
                "lines": [
                    "First identify which office or platform requires the document and whether originals, self-attested copies, or notarized copies are needed.",
                    "Keep the ID proof, address proof, application form, and supporting records arranged in one folder.",
                    "If this is for a complaint or filing, preserve both the submitted set and the stamped acknowledgement copy.",
                ],
                "likely_forum": "Relevant Authority / Filing Office",
                "authorities": ["issuing authority"],
                "documents_to_keep": ["ID proof", "address proof", "application copy", "acknowledgement"],
                "caution": "Check whether the authority requires originals, notarization, or online upload in a specific format.",
                "follow_up_question": "Which document or filing process are you trying to complete?",
            }
        return {
            "lines": [
                "Start by writing down the facts in date order and collecting the key documents, messages, receipts, and IDs linked to the issue.",
                "Identify the immediate forum first, such as the police station, cyber portal, consumer forum, bank, or a legal notice step.",
                "Once the facts and documents are clear, the next step is to prepare the complaint or response in a concise written form.",
            ],
            "likely_forum": "Depends on the issue",
            "authorities": [],
            "documents_to_keep": ["ID proof", "supporting documents", "written complaint or reply draft"],
            "caution": "If there is a deadline, fraud risk, or threat of coercive action, move on the first complaint or response quickly.",
            "follow_up_question": "What is the exact issue and which authority or person is involved?",
        }

    def _enrich_legal_help_state(
        self,
        *,
        current_state: ConversationState,
        message: str,
        domain: str,
    ) -> ConversationState:
        normalized = message.lower()
        issue_type = current_state.issue_type or self._detect_issue_type(normalized, domain)
        city = current_state.city or self._extract_city(message)
        police_station = current_state.police_station or self._extract_police_station(message)
        bank_name = current_state.bank_name or self._extract_bank_name(message)
        platform = current_state.platform or self._extract_platform(message)
        notice_stage = current_state.notice_stage or self._extract_notice_stage(message)
        document_type = current_state.document_type or self._extract_document_type(message)
        return current_state.model_copy(
            update={
                "issue_type": issue_type,
                "legal_domain": current_state.legal_domain or domain,
                "city": city,
                "police_station": police_station,
                "bank_name": bank_name,
                "platform": platform,
                "notice_stage": notice_stage,
                "document_type": document_type,
            }
        )

    def _detect_issue_type(self, normalized: str, domain: str) -> str:
        if any(token in normalized for token in {"cyber fraud", "upi", "wallet", "bank fraud", "phishing", "fake link", "otp", "debit"}):
            return "cyber_fraud"
        if any(token in normalized for token in {"fir", "police complaint", "police", "complaint"}):
            return "police_complaint"
        if "consumer" in normalized or any(token in normalized for token in {"refund", "defective", "seller", "service", "warranty"}):
            return "consumer"
        if "notice" in normalized:
            return "notice"
        if any(token in normalized for token in {"document", "documents", "verification", "certificate"}):
            return "documents"
        if domain == "cyber":
            return "cyber_fraud"
        if domain == "consumer":
            return "consumer"
        if domain in {"criminal", "property"}:
            return "police_complaint"
        return "general"

    def _cyber_fraud_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        if not state.bank_name and not state.platform:
            return {
                "lines": [
                    "Immediate steps: block the payment channel, call the bank or wallet helpline, and secure the account credentials.",
                    "File a complaint on the National Cyber Crime Portal or call 1930 as early as possible.",
                    "Keep transaction IDs, screenshots, bank SMS alerts, account statements, and complaint acknowledgements ready.",
                ],
                "likely_forum": "Cyber Crime Portal / Police Station / Bank",
                "authorities": ["cyber cell", "bank", "police"],
                "documents_to_keep": ["transaction ID", "screenshots", "bank statement", "complaint acknowledgement"],
                "caution": "Act quickly because reversal and fraud-tracing options are time-sensitive.",
                "follow_up_question": "Which platform or bank account was involved in the fraud?",
            }

        bank_or_platform = state.bank_name or state.platform or "the affected account"
        return {
            "lines": [
                f"Because {bank_or_platform} was involved, contact that bank or platform immediately, ask them to block further transactions, and request a complaint or reference number.",
                "Submit the same facts on the National Cyber Crime Portal or by calling 1930, then attach the screenshots, transaction ID, debit message, and bank statement.",
                "If the money has already been debited, visit the nearest cyber cell or police station with your ID proof and ask them to record the complaint with the transaction trail.",
            ],
            "likely_forum": "Cyber Crime Portal / Bank / Local Police or Cyber Cell",
            "authorities": ["cyber cell", "bank", "police"],
            "documents_to_keep": ["transaction ID", "screenshots", "bank statement", "complaint reference number"],
            "caution": "Keep every complaint reference number because banks and cyber cells often ask for the earlier complaint acknowledgement.",
            "follow_up_question": None,
        }

    def _consumer_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        return {
            "lines": [
                "Start by sending a written complaint to the seller or service provider and keep proof of delivery.",
                "Collect the invoice, payment proof, chats or emails, warranty terms, and photographs of the defect or deficiency.",
                "If the issue is not resolved, prepare for a consumer complaint with the product details, loss suffered, and relief claimed.",
            ],
            "likely_forum": "Consumer Commission",
            "authorities": ["seller", "consumer commission"],
            "documents_to_keep": ["invoice", "payment proof", "complaint copy", "photos"],
            "caution": "Make sure the chronology and the exact refund or compensation amount are clearly documented.",
            "follow_up_question": "Was any written complaint already sent to the seller or service provider?",
        }

    def _notice_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        if state.notice_stage:
            return {
                "lines": [
                    f"Since this is at the {state.notice_stage} stage, collect the notice copy, agreement, payment records, and the full date-wise timeline before taking the next step.",
                    "Prepare a factual reply or draft notice that clearly answers each allegation or demand and attach only the key supporting documents.",
                    "Keep proof of dispatch or delivery and do not miss the deadline mentioned in the notice.",
                ],
                "likely_forum": "Pre-litigation / Advocate Notice",
                "authorities": ["advocate"],
                "documents_to_keep": ["notice copy", "agreement", "payment proof", "communications"],
                "caution": "A delayed or vague reply can weaken your position later.",
                "follow_up_question": None,
            }
        return {
            "lines": [
                "Before sending or replying to a legal notice, organize the agreement, messages, invoices, payment records, and timeline of events.",
                "Check the exact demand, deadline, and the relief being claimed before drafting a reply.",
                "A proper reply usually states the facts, records your position, and attaches the key supporting documents.",
            ],
            "likely_forum": "Pre-litigation / Advocate Notice",
            "authorities": ["advocate"],
            "documents_to_keep": ["notice copy", "agreement", "payment proof", "communications"],
            "caution": "Do not ignore the notice deadline if a response has been demanded.",
            "follow_up_question": "Have you received the notice already, or are you planning to send one?",
        }

    def _police_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        if state.police_station or state.city:
            location = state.police_station or f"{state.city} police station"
            city_text = state.city or "your city"
            return {
                "lines": [
                    f"Since the complaint is connected with {location}, prepare a short written complaint with dates, place, names, and the exact incident in chronological order before you visit or submit it there.",
                    f"Carry your ID proof, screenshots, receipts, recordings, and witness details, and ask the police in {city_text} for a diary number or FIR acknowledgement if the matter is cognizable.",
                    "If the local police refuse to register the complaint, keep a copy of the written complaint and escalate it to the senior officer while preserving the submission proof.",
                ],
                "likely_forum": "Police Station / Senior Police Officer",
                "authorities": ["police"],
                "documents_to_keep": ["written complaint", "ID proof", "screenshots", "supporting records", "acknowledgement"],
                "caution": "Do not leave without noting the officer name, date, and acknowledgement or diary number.",
                "follow_up_question": None,
            }
        return {
            "lines": [
                "Write a short complaint with dates, place, names, and the exact incident in chronological order.",
                "Carry your ID proof and supporting documents such as screenshots, recordings, receipts, or witness details.",
                "If the matter involves a cognizable offence, ask for FIR registration and keep the complaint acknowledgement or diary number.",
            ],
            "likely_forum": "Police Station",
            "authorities": ["police"],
            "documents_to_keep": ["written complaint", "ID proof", "screenshots", "supporting records"],
            "caution": "Keep a copy of every complaint submitted and note the date, officer, and acknowledgement number.",
            "follow_up_question": "Which city or police station is connected with the complaint?",
        }

    def _documents_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        if state.document_type:
            return {
                "lines": [
                    f"For the {state.document_type} process, first confirm whether the authority wants originals, self-attested copies, notarized copies, or an online upload.",
                    "Keep the ID proof, address proof, application form, and the supporting records arranged in one set for submission and one set for your own file.",
                    "If this document is being submitted for a complaint or filing, preserve the stamped acknowledgement or upload receipt after submission.",
                ],
                "likely_forum": "Relevant Authority / Filing Office",
                "authorities": ["issuing authority"],
                "documents_to_keep": ["ID proof", "address proof", "application copy", "acknowledgement"],
                "caution": "Check the authority format and size rules before uploading or submitting originals.",
                "follow_up_question": None,
            }
        return {
            "lines": [
                "First identify which office or platform requires the document and whether originals, self-attested copies, or notarized copies are needed.",
                "Keep the ID proof, address proof, application form, and supporting records arranged in one folder.",
                "If this is for a complaint or filing, preserve both the submitted set and the stamped acknowledgement copy.",
            ],
            "likely_forum": "Relevant Authority / Filing Office",
            "authorities": ["issuing authority"],
            "documents_to_keep": ["ID proof", "address proof", "application copy", "acknowledgement"],
            "caution": "Check whether the authority requires originals, notarization, or online upload in a specific format.",
            "follow_up_question": "Which document or filing process are you trying to complete?",
        }

    def _guidance_key_for_state(self, state: ConversationState, follow_up_question: str | None) -> str:
        parts = [
            state.issue_type or "general",
            state.city or "",
            state.police_station or "",
            state.bank_name or "",
            state.platform or "",
            state.notice_stage or "",
            state.document_type or "",
            follow_up_question or "",
        ]
        return "|".join(parts)

    @staticmethod
    def _extract_city(message: str) -> str | None:
        lowered = message.lower().strip()
        city_match = re.search(r"\b([a-z][a-z ]{1,40})\s+city\b", lowered)
        if city_match:
            return city_match.group(1).strip().title()
        known_cities = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad"]
        compact = re.sub(r"[^a-z ]", " ", lowered)
        compact = re.sub(r"\s+", " ", compact).strip()
        for city in known_cities:
            if city.lower() in compact:
                return city
        return None

    @staticmethod
    def _extract_police_station(message: str) -> str | None:
        match = re.search(r"\b([a-z0-9 .'-]{2,60}\s+police station)\b", message, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip().title()
        return None

    @staticmethod
    def _extract_bank_name(message: str) -> str | None:
        banks = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Bank of Baroda", "Kotak", "Canara", "Union Bank"]
        lowered = message.lower()
        for bank in banks:
            if bank.lower() in lowered:
                return bank
        return None

    @staticmethod
    def _extract_platform(message: str) -> str | None:
        platforms = ["PhonePe", "Google Pay", "Paytm", "WhatsApp", "Telegram", "Instagram", "Facebook", "UPI"]
        lowered = message.lower()
        for platform in platforms:
            if platform.lower() in lowered:
                return platform
        return None

    @staticmethod
    def _extract_notice_stage(message: str) -> str | None:
        lowered = message.lower()
        if "received" in lowered:
            return "reply"
        if "send" in lowered or "sending" in lowered:
            return "drafting"
        return None

    @staticmethod
    def _extract_document_type(message: str) -> str | None:
        match = re.search(r"\b([a-z][a-z ]{2,50})\s+(certificate|document|application|verification)\b", message, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip().title()
        return None

    def _resolve_doctype_candidates(self, query: str, state: str | None, domain: str | None) -> list[str | None]:
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
        state_doctypes = state_map.get(state_key)
        domain_doctypes = domain_defaults.get(domain or "general", "judgments,laws")
        candidates: list[str | None] = []

        if self._is_statute_query(query):
            candidates.extend([domain_doctypes, "laws", "judgments", None, state_doctypes])
        else:
            candidates.extend([state_doctypes, domain_doctypes, None])

        return self._dedupe_values(candidates)

    def _build_query_variants(self, query: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", query.strip())
        lowered = cleaned.lower()
        variants = [cleaned]

        replacements = {
            " ipc": " Indian Penal Code",
            " crpc": " Code of Criminal Procedure",
            " cpc": " Code of Civil Procedure",
            " bns": " Bharatiya Nyaya Sanhita",
            " bnss": " Bharatiya Nagarik Suraksha Sanhita",
            " ni act": " Negotiable Instruments Act",
        }
        expanded = lowered
        for needle, replacement in replacements.items():
            expanded = re.sub(rf"\b{re.escape(needle.strip())}\b", replacement, expanded, flags=re.IGNORECASE)
        expanded = re.sub(r"\s+", " ", expanded).strip()
        if expanded and expanded != lowered:
            variants.append(expanded.title())

        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", lowered, re.IGNORECASE)
        article_match = re.search(r"\barticle\s+([0-9]+[a-z]?)\b", lowered, re.IGNORECASE)
        if section_match and "ipc" in lowered:
            variants.append(f"Indian Penal Code section {section_match.group(1).upper()}")
        if article_match and "constitution" in lowered:
            variants.append(f"Constitution of India article {article_match.group(1).upper()}")

        return self._dedupe_values(variants)

    @staticmethod
    def _is_statute_query(query: str) -> bool:
        normalized = query.lower()
        statute_signals = [
            "section ",
            "article ",
            "ipc",
            "crpc",
            "cpc",
            "bns",
            "bnss",
            "constitution",
            "act",
            "rule",
        ]
        return any(signal in normalized for signal in statute_signals)

    @staticmethod
    def _dedupe_values(values: list[str | None]) -> list[str | None]:
        deduped: list[str | None] = []
        for value in values:
            if value in deduped:
                continue
            deduped.append(value)
        return deduped
