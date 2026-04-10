from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import requests

from backend.app.core.config import Settings
from backend.app.models.schemas import ChatRequest, ChatUploadResponse, ConversationState, InternalChatResult
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.indiankanoon_service import IndianKanoonService, SOURCE_AUTHORITY_SCORES
from backend.app.services.legal_hybrid_retrieval import LegalHybridRetrievalService
from backend.app.services.intent_service import IntentRoutingService
from backend.app.services.legal_domain_classifier import LegalDomainClassifier
from backend.app.services.openai_service import OpenAIResponsesService
from backend.app.services.session_store import SessionStore
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.chat_service")


class ChatService:
    def __init__(self, settings: Settings, store: SessionStore) -> None:
        self.settings = settings
        self.store = store
        self.extractor = FileExtractionService(settings)
        self.indiankanoon = IndianKanoonService(settings)
        self.hybrid_retrieval = LegalHybridRetrievalService(settings, indiankanoon_service=self.indiankanoon)
        self.intent_router = IntentRoutingService()
        self.domain_classifier = LegalDomainClassifier()
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
        is_new_chat = chat_id is None
        if is_new_chat:
            session = self.store.create_session(self._generate_title(message), user_id=user_id)
            chat_id = session["id"]
        elif self.store.get_session(chat_id, user_id=user_id) is None:
            raise ValueError("Chat session not found")

        if is_new_chat:
            conversation_state = ConversationState()
            previous_messages: list[dict[str, Any]] = []
        else:
            conversation_state = ConversationState.model_validate(
                self.store.get_conversation_state(chat_id, user_id=user_id)
            )
            previous_messages = self.store.get_messages(chat_id, user_id=user_id)
        effective_message = self._effective_query_for_turn(message=message, conversation_state=conversation_state)
        domain = self._classify_domain_for_turn(
            message=effective_message,
            conversation_state=conversation_state,
        )
        if not domain and conversation_state.legal_domain:
            domain = conversation_state.legal_domain
        resolved_state = request.state or fallback_state or self.settings.default_state
        warnings = list(extraction_warnings or [])

        try:
            internal, next_state = self._generate_chat_result(
                message=effective_message,
                user_message=message,
                domain=domain,
                resolved_state=resolved_state,
                warnings=warnings,
                previous_messages=previous_messages,
                conversation_state=conversation_state,
                uploaded_texts=uploaded_texts or [],
            )
        except Exception:
            logger.exception("chat response generation failed query=%r domain=%s", effective_message[:120], domain)
            internal = self._build_fallback_result(domain=domain, warnings=warnings)
            next_state = conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "indiankanoon_rag",
                    "awaiting_details": False,
                    "last_user_issue": effective_message,
                    "legal_domain": domain,
                }
            )

        output_strategy = self._classify_legal_query(
            message=next_state.last_user_issue or effective_message,
            domain=domain,
            conversation_state=next_state,
        )
        internal = internal.model_copy(
            update={
                "answer": self._validate_final_output(
                    answer=internal.answer,
                    query=effective_message,
                    strategy=output_strategy,
                ),
                "citations": self._filter_criminal_reference_strings(internal.citations, strategy=output_strategy),
                "authorities": self._filter_criminal_reference_strings(internal.authorities, strategy=output_strategy),
            }
        )

        assistant_metadata = {
            "domain": internal.domain,
            "legal_domain_label": domain,
            "follow_up_question": internal.follow_up_question,
            "citations": internal.citations,
            "authorities": internal.authorities,
            "documents_to_keep": internal.documents_to_keep,
            "likely_forum": internal.likely_forum,
            "caution": internal.caution,
            "warnings": internal.warnings,
            "conversation_state": next_state.model_dump(),
            "pipeline": next_state.active_intent or "indiankanoon_rag",
            "retrieval": internal.raw_json,
        }

        try:
            self.store.add_message(
                chat_id,
                "user",
                message,
                metadata={
                    "domain": domain,
                    "legal_domain_label": domain,
                    "effective_query": effective_message,
                    "conversation_state": conversation_state.model_dump(),
                    "pipeline": next_state.active_intent or "indiankanoon_rag",
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
        user_message: str,
        domain: str,
        resolved_state: str | None,
        warnings: list[str],
        previous_messages: list[dict[str, Any]],
        conversation_state: ConversationState,
        uploaded_texts: list[str],
    ) -> tuple[InternalChatResult, ConversationState]:
        strategy = self._classify_legal_query(
            message=message,
            domain=domain,
            conversation_state=conversation_state,
        )
        logger.info(
            "chat pipeline stage=start query=%r domain=%s state=%s previous_messages=%s strategy=%s",
            message[:160],
            domain,
            resolved_state,
            len(previous_messages),
            strategy["answer_mode"],
        )
        direct_result = self._route_direct_response(
            message=message,
            domain=domain,
            warnings=warnings,
            conversation_state=conversation_state,
        )
        if direct_result is not None:
            internal, next_state = direct_result
            logger.info(
                "chat pipeline stage=direct_response active_intent=%s follow_up=%s",
                next_state.active_intent,
                bool(internal.follow_up_question),
            )
            return internal, next_state

        query_variants = self._build_query_variants(message)
        doctypes_options = self._resolve_doctype_candidates(
            query=message,
            state=resolved_state,
            domain=domain,
            answer_mode=strategy["answer_mode"],
        )
        logger.info(
            "chat pipeline stage=query_normalization query=%r variants=%s doctypes=%s",
            message[:160],
            query_variants,
            doctypes_options,
        )
        documents = self._retrieve_grounded_documents(
            query=message,
            query_variants=query_variants,
            doctypes_options=doctypes_options,
            state=resolved_state,
            domain=domain,
            answer_mode=strategy["answer_mode"],
        )
        documents = self._filter_grounded_documents_for_query(
            documents=documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )
        documents = sorted(documents, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        documents = self._select_primary_authorities(
            documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )
        uploaded_documents = self._build_uploaded_documents(uploaded_texts)
        context_documents = [*documents, *uploaded_documents]
        retrieval_confidence = self._grounded_retrieval_confidence(
            documents=documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )
        has_live_documents = any(
            str(doc.get("source_kind") or "").strip().lower() != "internal" for doc in documents
        )
        if not context_documents:
            logger.warning("chat pipeline stage=retrieval no_documents query=%r", message[:160])
            if self._should_fallback_to_legal_intake(message=message, domain=domain):
                logger.info("chat pipeline stage=retrieval reroute_to_legal_intake query=%r", message[:160])
                return self._handle_legal_help_interview(
                    message=user_message,
                    domain=domain,
                    warnings=warnings,
                    conversation_state=conversation_state,
                    is_continuation=conversation_state.active_intent == "legal_help" and conversation_state.conversation_started,
                )
            return self._build_fallback_result(domain=domain, warnings=warnings), ConversationState(
                conversation_started=True,
                active_intent="indiankanoon_rag",
                legal_domain=domain,
                last_user_issue=message,
            )
        if documents and retrieval_confidence < 0.6 and not has_live_documents:
            logger.warning(
                "chat pipeline stage=retrieval low_confidence query=%r confidence=%.2f answer_mode=%s",
                message[:160],
                retrieval_confidence,
                strategy["answer_mode"],
            )
            if self._should_fallback_to_legal_intake(message=message, domain=domain):
                return self._handle_legal_help_interview(
                    message=user_message,
                    domain=domain,
                    warnings=warnings,
                    conversation_state=conversation_state,
                    is_continuation=conversation_state.active_intent == "legal_help" and conversation_state.conversation_started,
                )
            return self._build_fallback_result(domain=domain, warnings=warnings), ConversationState(
                conversation_started=True,
                active_intent="indiankanoon_rag",
                legal_domain=domain,
                last_user_issue=message,
            )

        citations = self._build_citations(context_documents)
        authorities = self._build_authorities(context_documents)
        grounded_context = self._build_grounded_context(context_documents)
        grounded_context = self._trim_grounded_context(grounded_context)
        logger.info(
            "chat pipeline stage=context docs=%s uploaded_docs=%s citations=%s authorities=%s context_chars=%s",
            len(documents),
            len(uploaded_documents),
            len(citations),
            authorities,
            len(grounded_context),
        )
        llm_payload = self._generate_grounded_answer(
            query=message,
            domain=domain,
            state=resolved_state,
            context=grounded_context,
            citations=citations,
            conversation=self._conversation_for_llm(previous_messages),
            documents=context_documents,
        )
        answer = str(llm_payload.get("answer") or "").strip()
        if not answer:
            logger.warning("chat pipeline stage=llm empty_answer query=%r", message[:160])
            llm_payload = self._build_structured_grounded_payload(
                query=message,
                domain=domain,
                documents=context_documents,
                citations=citations,
            )
            answer = str(llm_payload.get("answer") or "").strip()
        answer = self._normalize_final_answer(answer, citations=citations)
        answer = self._ensure_upload_context_reflected(answer, uploaded_documents)
        answer = self._validate_final_output(
            answer=answer,
            query=message,
            strategy=strategy,
        )
        llm_payload["answer"] = answer
        citations = self._filter_criminal_reference_strings(citations, strategy=strategy)
        authorities = self._filter_criminal_reference_strings(authorities, strategy=strategy)

        internal = InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=self._optional_string(llm_payload.get("follow_up_question")),
            citations=citations,
            authorities=authorities,
            documents_to_keep=self._coerce_string_list(llm_payload.get("documents_to_keep")),
            likely_forum=self._optional_string(llm_payload.get("likely_forum")) or (authorities[0] if authorities else None),
            caution=self._optional_string(llm_payload.get("caution")),
            warnings=warnings,
            raw_json={
                "pipeline": "indiankanoon_rag",
                "strategy": strategy,
                "retrieval_confidence": retrieval_confidence,
                "query_variants": query_variants,
                "doctypes_options": doctypes_options,
                "documents": documents,
                "uploaded_documents": uploaded_documents,
                "llm_payload": llm_payload,
            },
        )
        next_state = conversation_state.model_copy(
            update={
                "conversation_started": True,
                "active_intent": "indiankanoon_rag",
                "awaiting_details": bool(internal.follow_up_question),
                "interview_mode": False,
                "last_user_issue": message,
                "legal_domain": domain,
                "last_follow_up_question": internal.follow_up_question,
                "current_intake_key": None,
            }
        )
        return internal, next_state

    def _route_direct_response(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        if not normalized:
            return None

        if conversation_state.conversation_started and conversation_state.active_intent == "legal_help":
            return self._handle_legal_help_interview(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                is_continuation=True,
            )

        if conversation_state.conversation_started and conversation_state.active_intent == "indiankanoon_rag":
            return None

        if self._should_use_indiankanoon(message) or self._looks_like_grounded_authority_query(normalized):
            return None

        intent_decision = self.intent_router.route(
            message,
            conversation_started=conversation_state.conversation_started,
            active_intent=conversation_state.active_intent,
        )
        if intent_decision.intent == "technical_redirect" and intent_decision.handled:
            internal = self._build_intent_result(intent_decision.answer, domain, warnings)
            next_state = conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "technical_redirect",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "legal_domain": domain,
                }
            )
            return internal, next_state

        if self._is_procedural_or_practical_query(normalized) or intent_decision.intent == "legal_help":
            return self._handle_legal_help_interview(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                is_continuation=False,
            )

        if intent_decision.handled and intent_decision.intent in {"greeting", "thanks", "goodbye", "advice"}:
            if self._looks_like_grounded_authority_query(normalized):
                return None
            internal = self._build_intent_result(intent_decision.answer, domain, warnings)
            next_state = conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": intent_decision.intent,
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "legal_domain": domain,
                }
            )
            return internal, next_state
        return None

    def _retrieve_grounded_documents(
        self,
        *,
        query: str,
        query_variants: list[str],
        doctypes_options: list[str | None],
        state: str | None,
        domain: str | None,
        answer_mode: str,
    ) -> list[dict[str, Any]]:
        try:
            result = self.hybrid_retrieval.retrieve(
                query=query,
                query_variants=query_variants,
                state=state,
                domain=domain,
                answer_mode=answer_mode,
                doctypes_options=doctypes_options,
            )
            logger.info(
                "chat pipeline stage=retrieval internal=%s live=%s confidence=%.2f documents=%s",
                result.internal_count,
                result.live_count,
                result.internal_confidence,
                len(result.documents),
            )
            return result.documents
        except (requests.RequestException, ValueError) as exc:
            logger.warning("chat pipeline stage=retrieval error=%s", exc)
            return []

    def _generate_grounded_answer(
        self,
        *,
        query: str,
        domain: str,
        state: str | None,
        context: str,
        citations: list[str],
        conversation: list[dict[str, str]],
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self._build_grounded_prompt(
            query=query,
            domain=domain,
            state=state,
            context=context,
            citations=citations,
        )
        logger.info(
            "chat pipeline stage=llm_request query=%r citations=%s history=%s prompt_chars=%s",
            query[:160],
            len(citations),
            len(conversation),
            len(prompt),
        )
        try:
            payload = self.openai.generate_json(prompt, conversation)
            normalized_payload = self._normalize_llm_payload(
                payload=payload,
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
            )
            logger.info(
                "chat pipeline stage=llm_response keys=%s answer_chars=%s",
                sorted(normalized_payload.keys()),
                len(str(normalized_payload.get("answer") or "")),
            )
            return normalized_payload
        except Exception as exc:
            logger.warning("chat pipeline stage=llm_error query=%r error=%s", query[:160], exc)
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
            )

    def _effective_query_for_turn(self, *, message: str, conversation_state: ConversationState) -> str:
        cleaned = re.sub(r"\s+", " ", message.strip())
        if not cleaned:
            return ""
        if (
            conversation_state.conversation_started
            and conversation_state.active_intent == "indiankanoon_rag"
            and conversation_state.awaiting_details
            and conversation_state.last_user_issue
        ):
            return self._merge_text(conversation_state.last_user_issue, cleaned)
        return cleaned

    @staticmethod
    def _build_uploaded_documents(uploaded_texts: list[str]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for index, text in enumerate(uploaded_texts, start=1):
            cleaned = re.sub(r"\s+", " ", str(text or "").strip())
            if not cleaned:
                continue
            excerpt = cleaned[:1800]
            title = f"Uploaded Document {index}"
            documents.append(
                {
                    "doc_id": f"upload-{index}",
                    "title": title,
                    "headline": excerpt[:240],
                    "fragment_headline": excerpt[:240],
                    "fragment_excerpt": excerpt,
                    "doc_excerpt": excerpt,
                    "docsource": "user_upload",
                    "citations": [],
                    "publishdate": "",
                    "url": "",
                    "score": 100.0 - index,
                }
            )
        return documents

    def _build_grounded_prompt(
        self,
        *,
        query: str,
        domain: str,
        state: str | None,
        context: str,
        citations: list[str],
    ) -> str:
        citation_block = "\n".join(f"- {item}" for item in citations[:6])
        return (
            "Answer the user's Indian legal query using only the grounded context below.\n"
            "Use only the retrieved internal corpus and Indian Kanoon sources that are included in the context.\n"
            "Do not invent facts, legal rules, procedures, deadlines, or authorities.\n"
            "Do not provide direct legal advice or claim to act as a lawyer; give general legal information only.\n"
            "Use a clear, slightly robotic tone with brief empathy where appropriate.\n"
            "If the situation is urgent, use direct time-sensitive wording. For fraud or fast-moving loss, explain that reporting within 24 hours can improve recovery chances when the grounded context supports that urgency.\n"
            "If the context is weak, conflicting, or incomplete, say so explicitly and keep the answer cautious.\n"
            "Always mention the sources you relied on in a dedicated Sources section.\n"
            "If uploaded user documents are present, explicitly use their facts and mention them in the final answer.\n"
            "Prefer this answer shape where relevant: Summary, Legal position, Practical next steps, Sources, Disclaimer.\n"
            "Return strict JSON with keys: answer, follow_up_question, likely_forum, caution, documents_to_keep.\n\n"
            f"User query: {query}\n"
            f"Detected domain: {domain}\n"
            f"State context: {state or 'Unknown'}\n\n"
            f"Grounded citations:\n{citation_block or '- None'}\n\n"
            f"Grounded retrieved context:\n{context}\n"
        )

    @staticmethod
    def _conversation_for_llm(previous_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        conversation: list[dict[str, str]] = []
        for item in previous_messages[-6:]:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            conversation.append({"role": role, "content": content})
        return conversation

    def _build_grounded_context(self, documents: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, doc in enumerate(documents, start=1):
            source_kind = str(doc.get("source_kind") or ("internal" if str(doc.get("docsource") or "").startswith("internal:") else "indiankanoon"))
            lines = [
                f"Document {index}",
                f"Title: {doc.get('title') or 'Untitled'}",
                f"Authority: {doc.get('docsource') or 'Indian Kanoon'}",
                f"Source kind: {source_kind}",
            ]
            if doc.get("publishdate"):
                lines.append(f"Date: {doc.get('publishdate')}")
            if doc.get("headline"):
                lines.append(f"Search snippet: {self._clean_search_snippet(str(doc.get('headline') or ''))}")
            if doc.get("fragment_headline"):
                lines.append(f"Relevant fragment: {self._clean_search_snippet(str(doc.get('fragment_headline') or ''))}")
            excerpt = str(doc.get("doc_excerpt") or doc.get("fragment_excerpt") or "").strip()
            if excerpt:
                lines.append(f"Excerpt: {self._clean_search_snippet(excerpt)}")
            citations = doc.get("citations") or []
            if citations:
                lines.append("Citations: " + ", ".join(str(item) for item in citations[:3]))
            if doc.get("url"):
                lines.append(f"URL: {doc.get('url')}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _select_primary_authorities(
        self,
        documents: list[dict[str, Any]],
        *,
        query: str,
        answer_mode: str,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        filtered = [doc for doc in documents if self._document_is_usable(doc)]
        if not filtered:
            return []
        internal_docs = [doc for doc in filtered if str(doc.get("source_kind") or "").strip().lower() == "internal"]
        live_docs = [doc for doc in filtered if str(doc.get("source_kind") or "").strip().lower() != "internal"]

        def _dedupe_and_limit(docs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
            selected: list[dict[str, Any]] = []
            seen_sources: set[str] = set()
            for doc in docs:
                source = f"{str(doc.get('source_kind') or '').strip().lower()}:{str(doc.get('docsource') or '').strip().lower()}:{str(doc.get('title') or '').strip().lower()}"
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                selected.append(doc)
                if len(selected) >= limit:
                    break
            return selected

        ranked_internal = sorted(internal_docs, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        ranked_live = sorted(live_docs, key=lambda item: float(item.get("score") or 0.0), reverse=True)

        if answer_mode == "statute_first" and ranked_live and any(
            str(doc.get("docsource") or "").strip().lower() == "laws"
            and self._looks_like_exact_authority_match_for_response(query=query, document=doc)
            for doc in ranked_live
        ):
            selected_live = _dedupe_and_limit(ranked_live, 2)
            return selected_live[:4]

        selected_internal = _dedupe_and_limit(ranked_internal, 2)
        selected_live = _dedupe_and_limit(ranked_live, 2)

        selected = selected_internal + [doc for doc in selected_live if doc not in selected_internal]
        if selected:
            return selected[:4]

        return _dedupe_and_limit(filtered, 4)

    def _document_is_usable(self, doc: dict[str, Any]) -> bool:
        raw_text = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
            ]
        )
        noisy_markers = [
            "traceback",
            "debug",
            "internal server error",
            "error in evaluting the fragments",
            "error evaluating the fragments",
            "unrelated case dump",
            "request_id",
            "stack trace",
        ]
        if any(marker in raw_text.lower() for marker in noisy_markers):
            return False
        cleaned = self._clean_search_snippet(raw_text)
        return bool(cleaned) and not any(marker in cleaned.lower() for marker in noisy_markers)

    @staticmethod
    def _build_citations(documents: list[dict[str, Any]]) -> list[str]:
        live_items: list[str] = []
        internal_items: list[str] = []
        upload_items: list[str] = []
        for doc in documents:
            title = str(doc.get("title") or "Untitled")
            source = str(doc.get("docsource") or "Indian Kanoon")
            date = str(doc.get("publishdate") or "").strip()
            url = str(doc.get("url") or "")
            entry = f"{title} | {source}"
            if date:
                entry += f" | {date}"
            if url:
                entry += f" | {url}"
            source_kind = str(doc.get("source_kind") or "").strip().lower()
            if source == "user_upload":
                if entry not in upload_items:
                    upload_items.append(entry)
            elif source_kind == "internal" or source.startswith("internal:"):
                if entry not in internal_items:
                    internal_items.append(entry)
            else:
                if entry not in live_items:
                    live_items.append(entry)

        ordered = [*live_items, *internal_items, *upload_items]
        return ordered[:3]

    def _ensure_upload_context_reflected(self, answer: str, uploaded_documents: list[dict[str, Any]]) -> str:
        if not uploaded_documents:
            return answer

        first_upload_excerpt = str(uploaded_documents[0].get("doc_excerpt") or "").strip()
        snippet = self._clean_search_snippet(first_upload_excerpt)[:140]
        upload_note = "Uploaded document facts were considered"
        if snippet:
            upload_note += f" (for example: {snippet})."
        else:
            upload_note += "."

        pattern = re.compile(r"(Practical Next Steps:\s*)(.*?)(\nSources:)", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(answer)
        if not match:
            return answer
        practical_steps = match.group(2).strip()
        practical_lower = practical_steps.lower()
        if "uploaded document" in practical_lower or "uploaded file" in practical_lower or "user upload" in practical_lower:
            return answer
        updated_steps = f"{practical_steps} {upload_note}".strip()
        return f"{answer[:match.start(2)]}{updated_steps}{answer[match.end(2):]}"

    @staticmethod
    def _build_authorities(documents: list[dict[str, Any]]) -> list[str]:
        authorities: list[str] = []
        for doc in documents:
            authority = str(doc.get("docsource") or "").strip()
            if authority and authority not in authorities:
                authorities.append(authority)
        return authorities[:5]

    def _build_structured_grounded_payload(
        self,
        *,
        query: str,
        domain: str,
        documents: list[dict[str, Any]],
        citations: list[str],
    ) -> dict[str, Any]:
        answer = self._grounded_structured_answer(query=query, domain=domain, documents=documents, citations=citations)
        likely_forum = self._infer_likely_forum(documents)
        return {
            "answer": answer,
            "follow_up_question": None,
            "likely_forum": likely_forum,
            "caution": self._grounded_caution(domain),
            "documents_to_keep": self._grounded_documents_to_keep(domain),
        }

    def _normalize_llm_payload(
        self,
        *,
        payload: dict[str, Any],
        query: str,
        domain: str,
        documents: list[dict[str, Any]],
        citations: list[str],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
            )

        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
            )

        normalized = dict(payload)
        normalized["answer"] = self._normalize_final_answer(answer, citations=citations)
        normalized["follow_up_question"] = self._optional_string(payload.get("follow_up_question"))
        normalized["likely_forum"] = self._optional_string(payload.get("likely_forum")) or self._infer_likely_forum(documents)
        normalized["caution"] = self._optional_string(payload.get("caution")) or self._grounded_caution(domain)
        normalized["documents_to_keep"] = self._coerce_string_list(payload.get("documents_to_keep")) or self._grounded_documents_to_keep(domain)
        return normalized

    def _grounded_structured_answer(
        self,
        *,
        query: str,
        domain: str,
        documents: list[dict[str, Any]],
        citations: list[str],
    ) -> str:
        if not documents:
            return "No relevant legal data found on India Kanoon"

        top = documents[0]
        summary_bits = [self._clean_search_snippet(str(top.get("title") or "the strongest retrieved authority"))]
        source = str(top.get("docsource") or "").strip()
        if source:
            summary_bits.append(source)
        summary = "Most relevant authority: " + " | ".join(summary_bits) + "."

        legal_excerpt = str(
            top.get("fragment_excerpt")
            or top.get("doc_excerpt")
            or top.get("fragment_headline")
            or top.get("headline")
            or ""
        ).strip()
        legal_position = self._clean_search_snippet(legal_excerpt) if legal_excerpt else "The retrieved Indian Kanoon materials indicate the closest grounded legal position available for this query."

        next_steps = self._grounded_next_steps(query=query, domain=domain, documents=documents)
        sources_line = "; ".join(citations[:3]) if citations else "Indian Kanoon retrieved authorities."
        return self._format_final_answer(
            summary=summary,
            legal_position=legal_position,
            practical_next_steps=next_steps,
            sources=sources_line,
            disclaimer="This is general legal information based on retrieved Indian Kanoon material, not a substitute for professional legal advice.",
        )

    @staticmethod
    def _trim_grounded_context(context: str, limit: int = 12000) -> str:
        if len(context) <= limit:
            return context
        trimmed = context[:limit].rsplit("\n", 1)[0].strip()
        if not trimmed:
            trimmed = context[:limit].strip()
        return trimmed + "\n[Context truncated for model length.]"

    def _grounded_next_steps(self, *, query: str, domain: str, documents: list[dict[str, Any]]) -> str:
        normalized = query.lower()
        if domain == "cyber" or any(token in normalized for token in {"upi", "fake link", "phishing", "otp", "debit", "bank fraud"}):
            return (
                "Preserve screenshots, transaction IDs, debit messages, and complaint references; contact the bank or platform immediately; "
                "and file the cyber-fraud complaint promptly while checking the retrieved authority for the exact legal basis."
            )
        if self._is_statute_query(query):
            return (
                "Read the cited provision and the linked authority together, confirm whether the result is the bare provision or an interpreting judgment, "
                "and verify the latest applicable law before relying on it in any complaint, notice, or court filing."
            )
        top_authority = self._infer_likely_forum(documents) or "the cited authority"
        return (
            f"Read the strongest retrieved authority from {top_authority}, compare it to your facts and dates, "
            "and preserve the relevant documents before taking the next legal step."
        )

    @staticmethod
    def _grounded_documents_to_keep(domain: str) -> list[str]:
        if domain == "cyber":
            return ["transaction ID", "screenshots", "bank statement", "complaint acknowledgement"]
        if domain == "consumer":
            return ["invoice", "payment proof", "complaint copy", "communications"]
        if domain == "property":
            return ["agreement", "rent receipts", "messages", "ID proof"]
        if domain == "tax":
            return ["tax notice", "return records", "assessment order", "payment challans"]
        if domain == "corporate":
            return ["board records", "shareholding documents", "ROC filings", "relevant agreements"]
        if domain == "document_review":
            return ["document copy", "annexures", "related correspondence", "supporting records"]
        if domain == "procedure":
            return ["application copy", "supporting documents", "proof of filing", "authority notice"]
        if domain == "criminal":
            return ["written complaint", "ID proof", "supporting records", "acknowledgement"]
        return ["ID proof", "supporting documents", "copy of the relevant authority"]

    @staticmethod
    def _grounded_caution(domain: str) -> str:
        if domain == "cyber":
            return "Act quickly because cyber-fraud recovery and tracing steps are often time-sensitive."
        if domain == "criminal":
            return "Verify the exact facts, section, and procedural stage before acting on the retrieved authority."
        if domain in {"tax", "corporate"}:
            return "Check the latest statutory text, compliance timeline, and current forum-specific procedure before acting."
        return "Check the original source and current facts before taking legal action."

    def _filter_grounded_documents_for_query(
        self,
        *,
        documents: list[dict[str, Any]],
        query: str,
        answer_mode: str,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for doc in documents:
            if self._document_matches_query_strategy(doc=doc, query=query, answer_mode=answer_mode):
                filtered.append(doc)
        if filtered:
            return filtered
        fallback_filtered: list[dict[str, Any]] = []
        for doc in documents:
            if self._document_is_safe_authority_fallback(doc=doc, query=query, answer_mode=answer_mode):
                fallback_filtered.append(doc)
        return fallback_filtered

    def _document_matches_query_strategy(self, *, doc: dict[str, Any], query: str, answer_mode: str) -> bool:
        query_lower = query.lower()
        title = str(doc.get("title") or "").lower()
        source = str(doc.get("docsource") or "").strip().lower()
        source_kind = str(doc.get("source_kind") or "").strip().lower()
        document_kind = str(doc.get("document_kind") or "").strip().lower()
        text = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("headline") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
            ]
        ).lower()
        query_tokens = self._normalize_query_tokens(query_lower)
        text_tokens = self._normalize_query_tokens(text)
        overlap = len(query_tokens & text_tokens)
        meaningful_query_tokens = self._meaningful_query_tokens(query_lower)
        meaningful_text_tokens = self._meaningful_query_tokens(text)
        meaningful_overlap = len(meaningful_query_tokens & meaningful_text_tokens)

        if source_kind == "internal":
            if answer_mode == "statute_first":
                section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query_lower)
                if document_kind in {"statute", "rule"}:
                    if section_match:
                        return f"section {section_match.group(1)}" in text or "article" in text or meaningful_overlap >= 1
                    return meaningful_overlap >= 1 or "section" in text or "article" in text
                if document_kind in {"judgment", "order"} and ("section" in text or "article" in text or "rule" in text):
                    if section_match:
                        return f"section {section_match.group(1)}" in text and meaningful_overlap >= 1
                    return meaningful_overlap >= 2
                if section_match:
                    return f"section {section_match.group(1)}" in text and meaningful_overlap >= 1
                return meaningful_overlap >= 2 and any(token in text for token in {"section", "article", "rule"})
            if answer_mode == "case_first":
                if document_kind in {"judgment", "order"}:
                    return meaningful_overlap >= 1 or "judgment" in text or "case" in text or "court" in text
                if document_kind in {"statute", "rule"} and any(token in text for token in {"section", "article", "rule"}):
                    return meaningful_overlap >= 1
                return meaningful_overlap >= 2
            return meaningful_overlap >= 2

        if answer_mode == "statute_first":
            section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query_lower)
            negated_section = False
            if section_match:
                negation_pattern = rf"\b(?:no|without|not)\s+section\s+{re.escape(section_match.group(1))}\b"
                negated_section = bool(re.search(negation_pattern, text))
            if source == "laws":
                if negated_section:
                    return False
                if section_match and f"section {section_match.group(1)}" not in title and overlap < 2:
                    return False
                return True
            if source in SOURCE_AUTHORITY_SCORES:
                if negated_section:
                    return False
                if overlap >= 1 or any(marker in text for marker in {"judgment", "case", "court", "holding"}):
                    return True
                return False
            if negated_section:
                return False
            if section_match and f"section {section_match.group(1)}" in text and overlap >= 2:
                return True
            return False

        if answer_mode == "case_first":
            if source == "laws" and overlap < 3:
                return False
            return overlap >= 2

        return overlap >= 2 or source in {"laws", "supremecourt", "consumer"}

    def _document_is_safe_authority_fallback(self, *, doc: dict[str, Any], query: str, answer_mode: str) -> bool:
        if answer_mode != "statute_first":
            return False
        source = str(doc.get("docsource") or "").strip().lower()
        source_kind = str(doc.get("source_kind") or "").strip().lower()
        if source not in SOURCE_AUTHORITY_SCORES and source != "laws":
            if source_kind != "internal":
                return False
        text = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("headline") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
            ]
        ).lower()
        if not text.strip():
            return False
        if any(marker in text for marker in {"traceback", "internal server error", "debug:", "unrelated"}):
            return False
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query.lower())
        if section_match:
            negation_pattern = rf"\b(?:no|without|not)\s+section\s+{re.escape(section_match.group(1))}\b"
            if re.search(negation_pattern, text):
                return False
        if source_kind == "internal":
            kind = str(doc.get("document_kind") or "").strip().lower()
            section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query.lower())
            if kind in {"statute", "rule", "judgment", "order"}:
                if section_match:
                    section_value = section_match.group(1)
                    if section_value not in text and f"section {section_value}" not in text:
                        return False
                return True
        meaningful_overlap = len(self._meaningful_query_tokens(query.lower()) & self._meaningful_query_tokens(text))
        support_markers = {
            "judgment discusses",
            "relevant fragment",
            "legal position",
            "procedural obligations",
            "procedural steps",
            "applicable legal position",
        }
        if meaningful_overlap < 1 and not any(marker in text for marker in support_markers):
            return False
        if re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query.lower()):
            section_value = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query.lower()).group(1)
            if f"section {section_value}" not in text and section_value not in text:
                return False
        return True

    def _grounded_retrieval_confidence(
        self,
        *,
        documents: list[dict[str, Any]],
        query: str,
        answer_mode: str,
    ) -> float:
        if not documents:
            return 0.0
        top = documents[0]
        score = float(top.get("score") or 0.0)
        source = str(top.get("docsource") or "").strip().lower()
        normalized_query = query.lower()
        confidence = 0.35
        if answer_mode == "statute_first" and source == "laws":
            confidence += 0.35
        elif answer_mode == "case_first" and source != "laws":
            confidence += 0.25
        else:
            confidence += 0.15
        if self._looks_like_exact_authority_match_for_response(query=normalized_query, document=top):
            confidence += 0.2
        if score >= 25:
            confidence += 0.15
        elif score >= 18:
            confidence += 0.08
        return min(confidence, 0.95)

    def _validate_final_output(
        self,
        *,
        answer: str,
        query: str,
        strategy: dict[str, str | bool],
    ) -> str:
        cleaned = answer or ""
        filler_patterns = [
            r"\bit is important to note that\b",
            r"\bit may be noted that\b",
            r"\bplease note that\b",
            r"\bin the present case\b",
            r"\bkindly note that\b",
        ]
        cleaned_lines: list[str] = []
        for line in cleaned.splitlines():
            normalized_line = line
            for pattern in filler_patterns:
                normalized_line = re.sub(pattern, "", normalized_line, flags=re.IGNORECASE)
            normalized_line = re.sub(r"[ \t]{2,}", " ", normalized_line).strip()
            if normalized_line:
                cleaned_lines.append(normalized_line)
        cleaned = "\n".join(cleaned_lines)
        cleaned = self._strip_unsupported_criminal_references(
            answer=cleaned,
            query=query,
            allow_criminal_sections=bool(strategy.get("allow_criminal_sections")),
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _strip_unsupported_criminal_references(
        self,
        *,
        answer: str,
        query: str,
        allow_criminal_sections: bool,
    ) -> str:
        if allow_criminal_sections:
            return answer
        cleaned = re.sub(
            r"\b(?:BNS|BNSS)\s+section\s+\d+[A-Za-z0-9()/-]*\b",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        return cleaned

    def _filter_criminal_reference_strings(
        self,
        values: list[str],
        *,
        strategy: dict[str, str | bool],
    ) -> list[str]:
        if bool(strategy.get("allow_criminal_sections")):
            return values
        filtered: list[str] = []
        for value in values:
            if re.search(r"\b(?:BNS|BNSS)\s+section\b", value, flags=re.IGNORECASE):
                continue
            filtered.append(value)
        return filtered

    @staticmethod
    def _normalize_query_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _meaningful_query_tokens(text: str) -> set[str]:
        tokens = ChatService._normalize_query_tokens(text)
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "been",
            "but",
            "can",
            "could",
            "do",
            "does",
            "for",
            "from",
            "had",
            "has",
            "have",
            "how",
            "i",
            "if",
            "in",
            "into",
            "is",
            "it",
            "may",
            "might",
            "much",
            "my",
            "no",
            "not",
            "of",
            "on",
            "or",
            "our",
            "please",
            "query",
            "should",
            "that",
            "the",
            "their",
            "there",
            "these",
            "they",
            "this",
            "those",
            "to",
            "under",
            "was",
            "we",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "without",
            "would",
            "legal",
            "authority",
            "authorities",
            "case",
            "cases",
            "court",
            "courts",
            "judgment",
            "judgement",
            "law",
            "laws",
            "act",
            "acts",
            "statute",
            "statutes",
            "article",
            "articles",
            "rule",
            "rules",
            "section",
            "sections",
            "procedure",
            "procedures",
            "information",
            "advice",
            "issue",
            "issues",
            "matching",
            "extremely",
            "obscure",
        }
        return {token for token in tokens if token not in stopwords}

    def _facts_clearly_indicate_criminal_issue(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        criminal_markers = {
            "fraud",
            "cheat",
            "cheating",
            "threat",
            "threatening",
            "intimidation",
            "snatched",
            "snatching",
            "stolen",
            "theft",
            "robbed",
            "assault",
            "forgery",
            "extortion",
        }
        return any(marker in normalized for marker in criminal_markers)

    def _looks_like_exact_authority_match_for_response(self, *, query: str, document: dict[str, Any]) -> bool:
        title = str(document.get("title") or "")
        normalized_title = title.lower()
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query, re.IGNORECASE)
        if section_match and f"section {section_match.group(1)}" in normalized_title:
            return True
        if "ipc" in query and "indian penal code" in normalized_title:
            return True
        if "ni act" in query and "negotiable instruments act" in normalized_title:
            return True
        if "constitution" in query and "constitution" in normalized_title:
            return True
        return False

    def _normalize_final_answer(self, answer: str, citations: list[str] | None = None) -> str:
        cleaned = self._sanitize_text_block(answer)
        if not cleaned:
            return self._format_final_answer(
                summary="Relevant Indian Kanoon material was retrieved for this query.",
                legal_position="The retrieved authorities provide the closest grounded legal position available for this query.",
                practical_next_steps="Review the strongest cited authority and match it against your facts before taking further legal action.",
                sources="Indian Kanoon retrieved authorities.",
                disclaimer="This is general legal information, not a substitute for professional legal advice.",
            )

        sections = {
            "Summary": "",
            "Legal Position": "",
            "Practical Next Steps": "",
            "Sources": "",
            "Disclaimer": "",
        }
        current: str | None = None
        alias_map = {
            "summary": "Summary",
            "legal position": "Legal Position",
            "legal position ": "Legal Position",
            "legal_position": "Legal Position",
            "legal": "Legal Position",
            "practical next steps": "Practical Next Steps",
            "practical next step": "Practical Next Steps",
            "next steps": "Practical Next Steps",
            "practical": "Practical Next Steps",
            "sources": "Sources",
            "source": "Sources",
            "disclaimer": "Disclaimer",
        }

        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            matched = False
            for alias, canonical in alias_map.items():
                prefix = f"{alias}:"
                if line.lower().startswith(prefix):
                    value = line[len(prefix):].strip()
                    if value:
                        sections[canonical] = self._merge_section_value(sections[canonical], value)
                    current = canonical
                    matched = True
                    break
            if matched:
                continue
            if current:
                sections[current] = self._merge_section_value(sections[current], line)

        if not sections["Summary"]:
            sections["Summary"] = self._first_meaningful_sentence(cleaned) or "Relevant Indian Kanoon material was retrieved for this query."
        if not sections["Legal Position"]:
            sections["Legal Position"] = sections["Summary"]
        if not sections["Practical Next Steps"]:
            sections["Practical Next Steps"] = "Review the strongest cited authority and match it against your facts before taking further legal action."
        if citations:
            sections["Sources"] = "; ".join(citations[:3])
        elif not sections["Sources"]:
            sections["Sources"] = "Indian Kanoon retrieved authorities."
        if not sections["Disclaimer"]:
            sections["Disclaimer"] = "This is general legal information, not a substitute for professional legal advice."

        return self._format_final_answer(
            summary=sections["Summary"],
            legal_position=sections["Legal Position"],
            practical_next_steps=sections["Practical Next Steps"],
            sources=sections["Sources"],
            disclaimer=sections["Disclaimer"],
        )

    def _format_final_answer(
        self,
        *,
        summary: str,
        legal_position: str,
        practical_next_steps: str,
        sources: str,
        disclaimer: str,
    ) -> str:
        cleaned_summary = self._sanitize_section_value(summary, fallback="Relevant Indian Kanoon material was retrieved for this query.")
        cleaned_legal_position = self._sanitize_section_value(
            legal_position,
            fallback="The retrieved authorities provide the closest grounded legal position available for this query.",
        )
        cleaned_steps = self._sanitize_section_value(
            practical_next_steps,
            fallback="Review the strongest cited authority and match it against your facts before taking further legal action.",
        )
        cleaned_disclaimer = self._sanitize_section_value(
            disclaimer,
            fallback="This is general legal information, not a substitute for professional legal advice.",
        )
        lines = [
            f"Summary: {cleaned_summary}",
            f"Legal Position: {cleaned_legal_position}",
            f"Practical Next Steps: {cleaned_steps}",
            f"Sources: {self._sanitize_sources(sources)}",
            f"Disclaimer: {cleaned_disclaimer}",
        ]
        return "\n".join(lines)

    def _sanitize_sources(self, text: str) -> str:
        cleaned = self._sanitize_section_value(text)
        parts = [part.strip(" ;,") for part in re.split(r"[;\n]+", cleaned) if part.strip(" ;,")]
        deduped: list[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return "; ".join(deduped[:3]) if deduped else "Indian Kanoon retrieved authorities."

    def _sanitize_section_value(self, text: str, fallback: str = "Not available from the retrieved material.") -> str:
        cleaned = self._sanitize_text_block(text)
        cleaned = re.sub(r"\b(Document \d+|Search snippet|Relevant fragment|Excerpt|Title|Authority|Date|URL|Citations)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(errmsg|error|debug|traceback|stack trace|request_id)\b\s*:?\s*[^.;]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
        return cleaned or fallback

    def _sanitize_text_block(self, text: str) -> str:
        cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"`{3,}.*?`{3,}", " ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\{[^{}]*\"errmsg\"[^{}]*\}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[Context truncated for model length\.\]", " ", cleaned)
        raw_lines = cleaned.split("\n")
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw_lines if re.sub(r"\s+", " ", line).strip()]
        deduped_lines: list[str] = []
        for line in lines:
            if line not in deduped_lines:
                deduped_lines.append(line)
        return "\n".join(deduped_lines)

    @staticmethod
    def _merge_section_value(existing: str, new_value: str) -> str:
        if not existing:
            return new_value
        if new_value.lower() in existing.lower():
            return existing
        if existing.lower() in new_value.lower():
            return new_value
        return f"{existing} {new_value}".strip()

    @staticmethod
    def _first_meaningful_sentence(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        match = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
        return match[0].strip()

    @staticmethod
    def _infer_likely_forum(documents: list[dict[str, Any]]) -> str | None:
        for doc in documents:
            source_kind = str(doc.get("source_kind") or "").strip().lower()
            forum = str(doc.get("docsource") or "").strip()
            if source_kind != "internal" and forum:
                return forum
        for doc in documents:
            forum = str(doc.get("docsource") or "").strip()
            if forum:
                return forum
        return None

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

    def _classify_domain_for_turn(self, *, message: str, conversation_state: ConversationState) -> str:
        normalized = re.sub(r"\s+", " ", message.strip())
        if not normalized:
            return conversation_state.legal_domain or "civil"
        if (
            conversation_state.conversation_started
            and conversation_state.legal_domain
            and len(normalized.split()) <= 8
            and not self._looks_like_grounded_authority_query(normalized.lower())
        ):
            return conversation_state.legal_domain
        classified = self.domain_classifier.classify(normalized)
        if classified == "procedure" and conversation_state.legal_domain and conversation_state.legal_domain != "procedure":
            return conversation_state.legal_domain
        return classified

    def _classify_legal_query(
        self,
        *,
        message: str,
        domain: str,
        conversation_state: ConversationState,
    ) -> dict[str, str | bool]:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        if self._is_statute_query(normalized):
            answer_mode = "statute_first"
            intent = "authority_lookup"
        elif any(token in normalized for token in {"judgment", "judgement", "precedent", "citation", "supreme court", "high court", "case law"}):
            answer_mode = "case_first"
            intent = "authority_lookup"
        elif conversation_state.active_intent == "indiankanoon_rag":
            answer_mode = "grounded_general"
            intent = "authority_lookup"
        elif self._is_procedural_or_practical_query(normalized):
            answer_mode = "fact_guidance"
            intent = "practical_guidance"
        else:
            answer_mode = "grounded_general"
            intent = "authority_lookup"

        allow_criminal_sections = (
            self._user_requested_sections(normalized)
            or answer_mode in {"statute_first", "case_first"}
            or self._facts_clearly_indicate_criminal_issue(normalized)
            or (
                conversation_state.active_intent == "legal_help"
                and not conversation_state.awaiting_details
                and (conversation_state.issue_type in {"snatching_theft", "cyber_fraud", "fir_refusal", "police_complaint"})
            )
        )
        return {
            "intent": intent,
            "answer_mode": answer_mode,
            "allow_criminal_sections": allow_criminal_sections,
            "domain": domain,
        }

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
        formatted_answer = self._format_final_answer(
            summary=answer or "General assistance was requested.",
            legal_position="This response is general guidance and does not rely on India Kanoon authority retrieval.",
            practical_next_steps="Ask a specific legal question or share the practical issue if you need more targeted guidance.",
            sources="General guidance response.",
            disclaimer="This is general legal information, not a substitute for professional legal advice.",
        )
        return InternalChatResult(
            answer=formatted_answer,
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

    def _handle_legal_help_interview(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        is_continuation: bool,
    ) -> tuple[InternalChatResult, ConversationState]:
        merged_issue = self._merge_legal_help_context(conversation_state, message) if is_continuation else message
        section_requested = self._user_requested_sections(merged_issue)
        base_state = self._enrich_legal_help_state(
            current_state=conversation_state,
            message=merged_issue,
            domain=domain,
        )
        collected_facts = dict(base_state.collected_facts)
        collected_facts.update(self._seed_interview_facts(base_state, merged_issue))
        collected_facts.update(self._extract_incremental_issue_facts(state=base_state, message=message))

        if is_continuation and conversation_state.current_intake_key:
            collected_facts[conversation_state.current_intake_key] = self._extract_interview_answer(
                key=conversation_state.current_intake_key,
                message=message,
                state=base_state,
            )

        plan = self._interview_plan_for_issue(base_state.issue_type or "general")
        if self._has_enough_information_for_guidance(base_state.issue_type or "general", collected_facts):
            plan = []
        next_item = self._next_missing_interview_item(plan=plan, collected_facts=collected_facts)

        if next_item is None:
            internal = self._build_virtual_advocate_result(
                query=merged_issue,
                domain=domain,
                warnings=warnings,
                state=base_state,
                collected_facts=collected_facts,
            )
            next_state = base_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "legal_help",
                    "awaiting_details": False,
                    "interview_mode": False,
                    "intake_stage": len(plan),
                    "last_user_issue": merged_issue,
                    "legal_domain": domain,
                    "last_follow_up_question": None,
                    "current_intake_key": None,
                    "collected_facts": collected_facts,
                }
            )
            return internal, next_state

        question_text = self._build_virtual_advocate_question(
            state=base_state,
            issue_type=base_state.issue_type or "general",
            question=next_item["question"],
            is_first_question=not is_continuation,
            collected_facts=collected_facts,
            allow_section_mentions=section_requested,
        )
        legal_references = self._legal_references_for_issue(state=base_state, facts=collected_facts) if section_requested else []
        likely_forum = self._forum_for_issue(base_state.issue_type or "general")
        authority_items = legal_references or ([likely_forum] if likely_forum else [])
        internal = InternalChatResult(
            answer=question_text,
            domain=domain,
            follow_up_question=str(next_item["question"]),
            citations=legal_references,
            authorities=authority_items,
            documents_to_keep=self._documents_for_issue(base_state.issue_type or "general", collected_facts)[:3],
            likely_forum=likely_forum,
            caution=None,
            warnings=warnings,
            raw_json={"source": "legal_help_interview", "collected_facts": collected_facts, "legal_references": legal_references},
        )
        next_state = base_state.model_copy(
            update={
                "conversation_started": True,
                "active_intent": "legal_help",
                "awaiting_details": True,
                "interview_mode": True,
                "intake_stage": int(next_item["index"]),
                "last_user_issue": merged_issue,
                "legal_domain": domain,
                "last_follow_up_question": str(next_item["question"]),
                "current_intake_key": str(next_item["key"]),
                "collected_facts": collected_facts,
            }
        )
        return internal, next_state

    def _build_virtual_advocate_question(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        question: str,
        is_first_question: bool,
        collected_facts: dict[str, str],
        allow_section_mentions: bool,
    ) -> str:
        del is_first_question
        acknowledgement = self._acknowledgement_for_issue(issue_type)
        immediate = self._immediate_guidance_for_issue(state=state, issue_type=issue_type, collected_facts=collected_facts)
        urgent_action = self._urgent_action_line(state=state, issue_type=issue_type, collected_facts=collected_facts)
        legal_basis = (
            self._early_legal_basis_text(state=state, issue_type=issue_type, collected_facts=collected_facts)
            if allow_section_mentions
            else ""
        )
        parts = [acknowledgement, immediate]
        if urgent_action:
            parts.append(urgent_action)
        if legal_basis:
            parts.append(legal_basis)
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _user_requested_sections(message: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        if not normalized:
            return False
        section_markers = [
            "which section",
            "what section",
            "legal section",
            "applicable section",
            "under which section",
            "section ",
            "ipc",
            "bns",
            "bnss",
            "ni act",
            "negotiable instruments act",
            "provision",
            "what law applies",
        ]
        return any(marker in normalized for marker in section_markers)

    @staticmethod
    def _acknowledgement_for_issue(issue_type: str) -> str:
        mapping = {
            "cyber_fraud": "I understand this is stressful, but we can act quickly to reduce the damage.",
            "snatching_theft": "I understand this is upsetting, but we should act quickly and methodically.",
            "landlord_harassment": "I understand this is stressful, but we should act carefully and without delay.",
            "food_safety": "I understand why you're concerned, and we should secure the evidence quickly.",
            "consumer": "I understand the difficulty, and we should lock in the complaint record quickly.",
            "fir_refusal": "I understand the frustration, but we should preserve the record and escalate promptly.",
            "notice": "I understand the concern, and we should protect the deadline immediately.",
            "police_complaint": "I understand the concern, and we should secure the complaint record quickly.",
            "documents": "I understand the issue, and we should avoid a late or incomplete filing.",
            "general": "I understand the concern, and we should move carefully but without delay.",
        }
        return mapping.get(issue_type, "I understand the concern.")

    def _immediate_guidance_for_issue(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        collected_facts: dict[str, str],
    ) -> str:
        if issue_type == "cyber_fraud":
            bank_platform = (
                collected_facts.get("bank_platform")
                or state.bank_name
                or self._display_platform_name(state.platform)
                or "your bank or payment app"
            )
            return (
                f"Immediate action: contact {bank_platform} right now, ask them to block further misuse, keep the transaction ID, screenshots, messages, and bank alerts together, and report the matter through 1930 or the National Cyber Crime Portal. If you report within 24 hours, the recovery chances are usually better."
            )
        if issue_type == "snatching_theft":
            return (
                "Immediate action: block the SIM and secure your email, banking apps, and other linked accounts, keep the IMEI, invoice, or item details ready, and go to the nearest police station with a short written complaint."
            )
        if issue_type == "landlord_harassment":
            return (
                "Immediate action: keep the rent agreement, rent proof, messages, and any threat record together, do not vacate or hand over original papers under pressure, and if there is a direct threat, lockout, or force, preserve proof immediately."
            )
        if issue_type == "food_safety":
            return (
                "Immediate action: do not throw away the packet or product, keep the invoice, batch details, and clear photographs, and send a written complaint to the seller or brand."
            )
        if issue_type == "consumer":
            return (
                "Immediate action: keep the invoice, payment proof, and defect or service record together, put the complaint in writing to the seller, platform, or service provider, and preserve screenshots, emails, and complaint numbers."
            )
        if issue_type == "fir_refusal":
            return (
                "Immediate action: keep a copy of the written complaint you already gave, note the police station, officer, date, and any refusal or diary detail, and preserve proof that the complaint was submitted but not registered."
            )
        if issue_type == "notice":
            return (
                "Immediate action: check the notice deadline immediately, keep the notice, agreement, messages, and payment records together, and do not send a rushed reply before the record is organized."
            )
        if issue_type == "police_complaint":
            return (
                "Immediate action: write the incident in short date-wise order, keep the main proof ready, and submit a written complaint at the police station while asking for acknowledgement or diary details."
            )
        if issue_type == "documents":
            return (
                "Immediate action: confirm exactly what document or format the authority wants, keep one set for submission and one set for your own record, and keep the receipt or acknowledgement after submission."
            )
        return "Immediate action: write down the facts clearly, keep the main proof together, and do not delay the first complaint or response if there is urgency or a deadline."

    def _urgent_action_line(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        collected_facts: dict[str, str],
    ) -> str:
        if issue_type == "cyber_fraud":
            bank_platform = (
                collected_facts.get("bank_platform")
                or state.bank_name
                or self._display_platform_name(state.platform)
                or "your bank or payment app"
            )
            return f"Most urgent right now: stop any further misuse through {bank_platform} and report the matter without delay. The first 24 hours are especially important for tracing and recovery."
        if issue_type == "snatching_theft":
            return "Most urgent right now: secure the SIM and linked accounts before the device or item can be misused. Early action reduces the misuse risk."
        if issue_type == "landlord_harassment":
            return "Most urgent right now: preserve proof of threats or pressure and avoid rushed steps under intimidation, especially if lockout or threats may escalate today."
        if issue_type == "food_safety":
            return "Most urgent right now: preserve the packet and photographs before the product evidence is lost or the condition changes."
        if issue_type == "consumer":
            return "Most urgent right now: put the complaint in writing and preserve the payment and defect record while the transaction trail is still easy to prove."
        if issue_type == "fir_refusal":
            return "Most urgent right now: keep the written complaint copy and proof that the station did not register it, because delay makes escalation harder."
        if issue_type == "notice":
            return "Most urgent right now: check the deadline and do not miss it, because a late reply can weaken your position."
        if issue_type == "police_complaint":
            return "Most urgent right now: prepare the written incident summary and keep your main proof ready, because early reporting preserves chronology and proof."
        if issue_type == "documents":
            return "Most urgent right now: confirm the exact filing requirement before submitting anything incomplete, because a late or wrong filing can trigger rejection."
        return ""

    def _contextualize_follow_up_question(
        self,
        *,
        issue_type: str,
        question: str,
        state: ConversationState,
        collected_facts: dict[str, str],
    ) -> str:
        bank_platform = collected_facts.get("bank_platform") or state.bank_name or self._display_platform_name(state.platform)
        station = collected_facts.get("station_details") or state.police_station or state.city
        if issue_type == "cyber_fraud" and bank_platform:
            if "When did this happen" in question:
                return f"For {bank_platform}, when exactly did this happen, and was it one transaction or multiple debits?"
            if "Have you already reported" in question:
                return f"For {bank_platform}, have you already reported it to the bank, app, or 1930?"
        if issue_type in {"snatching_theft", "police_complaint", "fir_refusal"} and station:
            if "Which police station or city" in question or "Which city or police station" in question:
                return f"I already have {station} from you, so tell me the next missing point instead: {question}"
            if "Have you already filed" in question:
                return f"For the matter linked to {station}, have you already filed the complaint or FIR, or are you still at the first-report stage?"
        if issue_type == "landlord_harassment" and collected_facts.get("harassment_details"):
            if "What exactly is the landlord doing" in question:
                return "You already mentioned the harassment broadly, so tell me the most important part now: is it mainly eviction pressure, lockout, threats, rent demand, or deposit issue?"
        return question

    def _early_legal_basis_text(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        collected_facts: dict[str, str],
    ) -> str:
        references = self._legal_references_for_issue(state=state, facts=collected_facts)
        if not references:
            return ""
        if issue_type == "cyber_fraud":
            return "On these facts, this may broadly be treated as cheating or personation, and the process usually starts with the bank or payment app, 1930, and then the police route under BNSS procedure."
        if issue_type == "snatching_theft":
            return "On these facts, this may broadly be treated as theft or snatching, and the process usually starts at the police station under BNSS procedure."
        if issue_type in {"fir_refusal", "police_complaint"}:
            return "If the facts show a cognizable offence, the complaint and investigation route usually starts at the police station under BNSS procedure."
        if issue_type == "landlord_harassment":
            return "If the matter includes threats or intimidation, it may also have a criminal complaint angle apart from the tenancy dispute."
        return ""

    def _build_virtual_advocate_result(
        self,
        *,
        query: str,
        domain: str,
        warnings: list[str],
        state: ConversationState,
        collected_facts: dict[str, str],
    ) -> InternalChatResult:
        legal_references = self._legal_references_for_issue(state=state, facts=collected_facts)
        paragraph = self._compose_virtual_advocate_paragraph(query=query, state=state, facts=collected_facts)
        steps = self._compose_virtual_advocate_steps(state=state, facts=collected_facts)
        intro = self._completed_guidance_intro(state=state, facts=collected_facts)
        answer = "\n\n".join(part for part in [intro, paragraph] if part).strip()
        if steps:
            answer += (
                f"\n\n{self._steps_heading_for_issue(state.issue_type or 'general')}:\n"
                + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
            )
        return InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=None,
            citations=legal_references,
            authorities=legal_references,
            documents_to_keep=self._documents_for_issue(state.issue_type or "general", collected_facts),
            likely_forum=self._forum_for_issue(state.issue_type or "general"),
            caution=None,
            warnings=warnings,
            raw_json={"source": "virtual_advocate", "collected_facts": collected_facts, "legal_references": legal_references},
        )

    def _completed_guidance_intro(self, *, state: ConversationState, facts: dict[str, str]) -> str:
        issue_type = state.issue_type or "general"
        if issue_type == "cyber_fraud":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("reporting_status"), {"reported", "1930", "cyber crime portal", "complaint number", "bank complaint"}):
                progress = " You have already started the reporting trail, so the next step is to strengthen the record and push the follow-up."
            return (
                "I understand this is stressful, but we can act quickly to reduce the damage. "
                "Based on the facts shared, time matters here, and reporting within 24 hours usually improves the recovery chances."
                + progress
            )
        if issue_type == "snatching_theft":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("police_status"), {"filed", "fir", "complaint"}):
                progress = " You have already started the complaint side, so the next step is to strengthen the record and protect against misuse."
            return (
                "I understand this is upsetting, but we should act quickly and methodically. "
                "The earlier you secure accounts and strengthen the complaint record, the lower the misuse risk."
                + progress
            )
        if issue_type == "landlord_harassment":
            return (
                "I understand this is stressful, but we should act carefully and without delay. "
                "If the pressure is escalating, preserving proof early will matter."
            )
        if issue_type == "food_safety":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("seller_contact"), {"complained", "written complaint", "seller", "brand", "platform", "emailed"}):
                progress = " You have already started the seller-side complaint trail, so the next step is to preserve the evidence and prepare escalation if needed."
            return (
                "I understand why you're concerned, and we should secure the evidence quickly. "
                "The position is easier to prove while the packet, batch details, and photographs are still intact."
                + progress
            )
        if issue_type == "consumer":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("complaint_status"), {"complaint", "emailed", "written", "ticket", "support"}):
                progress = " You have already started the complaint record, so the next step is to tighten the proof and the exact relief sought."
            return (
                "I understand the difficulty, and we should lock in the complaint record quickly. "
                "Early written complaints usually make the transaction trail easier to prove."
                + progress
            )
        if issue_type == "fir_refusal":
            return (
                "I understand the frustration, but we should preserve the record and escalate promptly. "
                "Delay can make the refusal trail harder to prove."
            )
        if issue_type == "notice":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("notice_stage_detail"), {"received", "reply"}):
                progress = " You have already reached the notice stage, so the next step is to organise the reply record rather than restate that part."
            return (
                "I understand the concern, and we should protect the deadline immediately. "
                "A late or rushed reply can weaken your position."
                + progress
            )
        if issue_type == "police_complaint":
            return (
                "I understand the concern, and we should secure the complaint record quickly. "
                "Early reporting usually preserves the chronology better."
            )
        if issue_type == "documents":
            return (
                "I understand the issue, and we should avoid a late or incomplete filing. "
                "Checking the requirement now is safer than correcting a rejection later."
            )
        return "I understand the concern, and we should move carefully but without delay."

    @staticmethod
    def _steps_heading_for_issue(issue_type: str) -> str:
        headings = {
            "cyber_fraud": "Immediate next steps",
            "snatching_theft": "Priority actions",
            "landlord_harassment": "Recommended actions",
            "food_safety": "What to do next",
            "consumer": "Practical actions",
            "fir_refusal": "Escalation steps",
            "notice": "Response plan",
            "police_complaint": "Complaint steps",
            "documents": "Filing plan",
            "general": "Next steps",
        }
        return headings.get(issue_type, "Next steps")

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
        effective_state = self._enrich_legal_help_state(
            current_state=effective_state,
            message=query,
            domain=domain,
        )
        guidance = self._legal_help_guidance(query=query, domain=domain, conversation_state=effective_state)
        lines.extend(str(item).strip() for item in guidance.get("lines", []) if str(item).strip())
        summary = self._compose_legal_help_summary(
            query=query,
            state=effective_state,
            guidance=guidance,
            intro_text=intro_text,
        )
        legal_position = self._compose_legal_help_position(
            query=query,
            state=effective_state,
            guidance=guidance,
        )
        practical_next_steps = self._compose_legal_help_next_steps(
            guidance=guidance,
            lines=lines,
        )
        sources = self._build_guidance_sources(guidance)
        answer = self._format_final_answer(
            summary=summary,
            legal_position=legal_position,
            practical_next_steps=practical_next_steps,
            sources=sources,
            disclaimer="This is general legal information and procedural guidance, not a substitute for professional legal advice.",
        )
        return InternalChatResult(
            answer=answer,
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
            answer=self._format_final_answer(
                summary="No highly relevant India Kanoon authority was found for this query.",
                legal_position="A grounded statute or case-law answer could not be confirmed from the retrieved legal authorities.",
                practical_next_steps="Try a more specific section, statute, judgment, court, or citation query, or ask the practical legal issue directly for procedural guidance.",
                sources="India Kanoon search did not return a sufficiently relevant authority.",
                disclaimer="This is general legal information, not a substitute for professional legal advice.",
            ),
            domain=domain,
            follow_up_question=None,
            citations=[],
            authorities=[],
            documents_to_keep=[],
            likely_forum=None,
            caution=None,
            warnings=warnings,
            raw_json={"pipeline": "indiankanoon_rag", "documents": []},
        )

    @staticmethod
    def _clean_search_snippet(text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\{[^{}]*\"errmsg\"[^{}]*\}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(errmsg|debug|traceback|stack trace)\b\s*:?\s*[^.;]*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
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
            "cheque bounce",
            "negotiable instruments act",
            "ni act",
        ]
        return any(signal in normalized for signal in caselaw_signals)

    @staticmethod
    def _looks_like_grounded_authority_query(normalized: str) -> bool:
        authority_markers = [
            "legal position",
            "authority",
            "authorities",
            "case law",
            "judgment",
            "judgement",
            "precedent",
            "cheque bounce",
            "negotiable instruments act",
            "ni act",
        ]
        return any(marker in normalized for marker in authority_markers)

    def _is_procedural_or_practical_query(self, normalized: str) -> bool:
        procedural_markers = [
            "how do i",
            "how to",
            "what should i do",
            "what can i do",
            "process",
            "procedure",
            "steps",
            "money got debited",
            "money deducted",
            "consumer complaint",
            "police complaint",
            "fir",
            "fraud",
            "fake link",
            "upi",
            "cyber fraud",
            "bank fraud",
            "notice",
            "document",
            "documents",
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
            "snatched",
            "stolen",
            "landlord",
            "eviction",
            "deposit",
            "lockout",
        ]
        if self._should_use_indiankanoon(normalized):
            return False
        return any(marker in normalized for marker in procedural_markers)

    def _should_fallback_to_legal_intake(self, *, message: str, domain: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        if self._should_use_indiankanoon(normalized):
            return False
        if self._is_procedural_or_practical_query(normalized):
            return True
        return self._detect_issue_type(normalized, domain) != "general"

    @staticmethod
    def _build_guidance_sources(guidance: dict[str, str | list[str] | None]) -> str:
        likely_forum = str(guidance.get("likely_forum") or "").strip()
        authorities = [str(item).strip() for item in (guidance.get("authorities") or []) if str(item).strip()]
        parts: list[str] = []
        if likely_forum:
            parts.append(likely_forum)
        parts.extend(item for item in authorities if item not in parts)
        return "; ".join(parts) if parts else "General procedural legal guidance."

    def _compose_legal_help_summary(
        self,
        *,
        query: str,
        state: ConversationState,
        guidance: dict[str, str | list[str] | None],
        intro_text: str,
    ) -> str:
        guidance_summary = str(guidance.get("summary") or "").strip()
        if guidance_summary:
            return guidance_summary

        issue_sentence = self._issue_summary_sentence(query=query, state=state)
        if intro_text:
            return f"{intro_text} {issue_sentence}".strip()
        return issue_sentence

    def _compose_legal_help_position(
        self,
        *,
        query: str,
        state: ConversationState,
        guidance: dict[str, str | list[str] | None],
    ) -> str:
        guidance_position = str(guidance.get("legal_position") or "").strip()
        if guidance_position:
            return guidance_position

        issue_type = state.issue_type or "general"
        location = state.police_station or state.city
        if issue_type == "cyber_fraud":
            bank_or_platform = state.bank_name or state.platform or "the affected bank or payment platform"
            return (
                f"A fraudulent debit or fake-link incident involving {bank_or_platform} should be treated as an urgent"
                " complaint and record-preservation matter, so early reporting and a consistent transaction trail become important."
            )
        if issue_type == "food_safety":
            return (
                "A contamination or unsafe-food complaint becomes stronger when the packet details, invoice, defect photographs,"
                " and any illness records are preserved before the seller or authority is approached."
            )
        if issue_type == "snatching_theft":
            location_text = f" in {location}" if location else ""
            return (
                f"A snatching or theft complaint{location_text} usually needs prompt police reporting plus immediate protection"
                " of any linked phone, SIM, banking, or account access."
            )
        if issue_type == "fir_refusal":
            return (
                "When the police do not register the complaint, the practical legal route is to preserve the written complaint"
                " and escalate the same facts with proof of submission instead of relying only on oral follow-up."
            )
        if issue_type == "landlord_harassment":
            return (
                "Harassment, illegal pressure, or forced-eviction threats from a landlord should be documented carefully,"
                " with rent and communication records preserved before any next complaint, notice, or civil step."
            )
        if issue_type == "consumer":
            return (
                "Consumer matters usually turn on a clear written complaint, proof of payment, product or service records,"
                " and a precise statement of the refund, replacement, repair, or compensation being sought."
            )
        if issue_type == "notice":
            return (
                "Notice-stage disputes are handled best when the timeline, agreement, and supporting records are organized"
                " first, so the next reply or notice remains factual, timely, and consistent."
            )
        if issue_type == "police_complaint":
            return (
                "A police complaint becomes more workable when the incident is written in date order and supported by IDs,"
                " witness details, receipts, screenshots, or other primary records."
            )
        if issue_type == "documents":
            return (
                "Document-filing issues usually depend on submitting the right form of proof in the exact format required"
                " by the receiving authority, while keeping your own acknowledgement copy."
            )
        return (
            f"The next legal step for this issue depends on the facts already shared{self._with_query_reference(query)},"
            " the documents available, and the authority that needs to be approached first."
        )

    def _compose_legal_help_next_steps(
        self,
        *,
        guidance: dict[str, str | list[str] | None],
        lines: list[str],
    ) -> str:
        explicit_steps = [
            str(item).strip()
            for item in (guidance.get("practical_next_steps") or guidance.get("steps") or [])
            if str(item).strip()
        ]
        steps = explicit_steps or lines[2:] or lines
        if not steps:
            steps = ["Write down the facts in date order and collect the key documents before taking the next step."]

        step_parts: list[str] = []
        for index, step in enumerate(steps, start=1):
            cleaned = re.sub(r"^\d+\.\s*", "", step).strip()
            if not cleaned:
                continue
            step_parts.append(f"Step {index}: {cleaned}")

        documents = [str(item).strip() for item in (guidance.get("documents_to_keep") or []) if str(item).strip()]
        if documents:
            step_parts.append("Keep ready: " + ", ".join(documents[:6]) + ".")

        caution = str(guidance.get("caution") or "").strip()
        if caution:
            step_parts.append("Caution: " + caution)

        return " ".join(step_parts)

    def _issue_summary_sentence(self, *, query: str, state: ConversationState) -> str:
        issue_type = state.issue_type or "general"
        lowered = query.lower()
        if issue_type == "cyber_fraud":
            bank_or_platform = state.bank_name or state.platform or "the affected bank or payment app"
            return (
                f"This looks like a cyber-fraud or unauthorized debit situation involving {bank_or_platform},"
                " so the answer should focus on rapid reporting, transaction preservation, and account protection."
            )
        if issue_type == "food_safety":
            product = "the product"
            if "chocolate" in lowered:
                product = "the chocolate packet"
            return (
                f"This looks like a contamination or unsafe-food complaint concerning {product},"
                " so the immediate focus is evidence preservation and seller or authority escalation."
            )
        if issue_type == "snatching_theft":
            item = "the stolen item"
            if "mobile" in lowered or "phone" in lowered:
                item = "the mobile phone"
            return (
                f"This appears to be a snatching or theft issue involving {item},"
                " so the practical answer should prioritize police reporting and protection against misuse."
            )
        if issue_type == "fir_refusal":
            return (
                "This is a complaint-registration problem, so the next move is not to repeat the same oral request"
                " but to preserve the written complaint and escalate it properly."
            )
        if issue_type == "landlord_harassment":
            return (
                "This looks like a landlord-tenant pressure or eviction-threat issue, so the answer should focus on"
                " preserving tenancy records, written communication, and the safest escalation path."
            )
        if issue_type == "consumer":
            return (
                "This appears to be a consumer dispute, so the practical path is a written seller or service complaint"
                " backed by payment and defect records."
            )
        if issue_type == "notice":
            return (
                "This is a notice-stage matter, so the useful answer is to organize the record first and then take a"
                " timely, factual reply or drafting step."
            )
        if issue_type == "police_complaint":
            return (
                "This looks like a police-complaint issue, so the answer should focus on chronology, supporting proof,"
                " and acknowledgement of submission."
            )
        if issue_type == "documents":
            return (
                "This looks like a document-process issue, so the useful answer is to confirm the exact filing"
                " requirement and preserve the acknowledgement."
            )
        return "This issue needs a fact-based procedural answer built around the records, forum, and immediate next step."

    @staticmethod
    def _with_query_reference(query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query.strip())
        if not cleaned:
            return ""
        snippet = cleaned[:90].rstrip(" ,;:")
        return f" in '{snippet}'"

    def _interview_plan_for_issue(self, issue_type: str) -> list[dict[str, str | int]]:
        plans: dict[str, list[tuple[str, str]]] = {
            "cyber_fraud": [
                ("incident_timing", "When did this happen, and was it one transaction or multiple debits?"),
                ("reporting_status", "Have you already reported it to the bank, app, or 1930?"),
            ],
            "snatching_theft": [
                ("incident_timing", "When and where did the snatching or theft happen?"),
                ("police_status", "Have you already filed a police complaint or FIR, or are you still at the first-report stage?"),
            ],
            "landlord_harassment": [
                ("harassment_details", "What exactly is the landlord doing: eviction pressure, lockout, threats, rent demand, or deposit issue?"),
                ("record_status", "Do you have the rent agreement, rent-payment proof, and messages or call records?"),
            ],
            "food_safety": [
                ("product_details", "What product was involved, and do you still have the packet, batch details, and invoice?"),
                ("seller_contact", "Have you already complained to the seller, brand, or platform?"),
            ],
            "consumer": [
                ("purchase_details", "What product or service is involved, and when did you pay for it?"),
                ("complaint_status", "Have you already sent a written complaint to the seller, platform, or service provider?"),
            ],
            "fir_refusal": [
                ("incident_details", "What was the underlying incident for which you wanted the FIR registered?"),
                ("station_details", "Which police station or city is involved?"),
            ],
            "notice": [
                ("notice_stage_detail", "Did you receive a notice or are you planning to send one?"),
                ("notice_content", "What is the main demand or allegation in the notice, and what deadline has been mentioned?"),
            ],
            "police_complaint": [
                ("incident_details", "What exactly happened, and when did it happen?"),
                ("station_details", "Which city or police station is connected with the complaint?"),
            ],
            "documents": [
                ("document_purpose", "Which document or filing process is involved, and which office or platform is asking for it?"),
                ("blocker_details", "What exactly is the problem: missing document, rejection, mismatch, upload issue, or attestation issue?"),
            ],
            "general": [
                ("issue_details", "What exactly happened, and who is the other side or authority involved?"),
                ("timing_details", "When did it happen, and is there any urgent deadline, threat, or loss happening right now?"),
            ],
        }
        selected = plans.get(issue_type, plans["general"])
        return [{"index": index, "key": key, "question": question} for index, (key, question) in enumerate(selected, start=1)]

    @staticmethod
    def _has_enough_information_for_guidance(issue_type: str, collected_facts: dict[str, str]) -> bool:
        checks = {
            "cyber_fraud": lambda facts: bool(facts.get("incident_timing") and facts.get("reporting_status")),
            "snatching_theft": lambda facts: bool(facts.get("incident_timing") and facts.get("police_status")),
            "landlord_harassment": lambda facts: bool(facts.get("harassment_details")),
            "food_safety": lambda facts: bool(facts.get("product_details")),
            "consumer": lambda facts: bool(facts.get("purchase_details")),
            "fir_refusal": lambda facts: bool(facts.get("incident_details") and facts.get("station_details")),
            "notice": lambda facts: bool(facts.get("notice_stage_detail") and facts.get("notice_content")),
            "police_complaint": lambda facts: bool(facts.get("incident_details") and facts.get("station_details")),
            "documents": lambda facts: bool(facts.get("document_purpose") and facts.get("blocker_details")),
            "general": lambda facts: bool(facts.get("issue_details") and facts.get("timing_details")),
        }
        checker = checks.get(issue_type, checks["general"])
        return checker(collected_facts)

    @staticmethod
    def _next_missing_interview_item(
        *,
        plan: list[dict[str, str | int]],
        collected_facts: dict[str, str],
    ) -> dict[str, str | int] | None:
        for item in plan:
            key = str(item["key"])
            if not str(collected_facts.get(key) or "").strip():
                return item
        return None

    def _seed_interview_facts(self, state: ConversationState, message: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        if cleaned:
            facts["issue_summary"] = cleaned
        if state.issue_type == "food_safety":
            if any(token in lowered for token in {"packet", "sealed", "batch", "invoice", "bill"}):
                facts["product_details"] = cleaned
            if any(token in lowered for token in {"photo", "photos", "photograph", "video", "ill", "illness", "sick"}):
                facts["health_effect"] = cleaned
        if state.issue_type == "consumer":
            if any(token in lowered for token in {"invoice", "bill", "paid", "payment", "order", "service"}):
                facts["purchase_details"] = cleaned
            if any(token in lowered for token in {"complaint", "emailed", "wrote", "written", "ticket"}):
                facts["complaint_status"] = cleaned
        if state.issue_type == "cyber_fraud":
            if any(token in lowered for token in {"transaction id", "debited", "debit", "amount", "rs", "rupees"}):
                facts["amount_details"] = cleaned
        if state.city:
            facts["station_details"] = state.city
        if state.police_station:
            facts["station_details"] = state.police_station
        if state.bank_name:
            facts["bank_platform"] = state.bank_name
        if state.platform:
            platform_name = self._display_platform_name(state.platform)
            if platform_name:
                facts["bank_platform"] = " / ".join(item for item in [facts.get("bank_platform", "").strip(), platform_name] if item)
        return facts

    def _extract_incremental_issue_facts(self, *, state: ConversationState, message: str) -> dict[str, str]:
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        if not cleaned:
            return {}

        facts: dict[str, str] = {}
        issue_type = state.issue_type or "general"

        if issue_type == "cyber_fraud":
            if self._message_has_timing_signal(lowered):
                facts["incident_timing"] = cleaned
            if any(
                token in lowered
                for token in {
                    "reported",
                    "not reported",
                    "have not reported",
                    "haven't reported",
                    "did not report",
                    "called 1930",
                    "1930",
                    "cyber crime portal",
                    "complaint number",
                    "bank complaint",
                }
            ):
                facts["reporting_status"] = cleaned
            if any(token in lowered for token in {"transaction id", "utr", "rs", "rupees", "amount", "debited", "debit"}):
                facts["amount_details"] = cleaned
        elif issue_type == "snatching_theft":
            if self._message_has_timing_signal(lowered):
                facts["incident_timing"] = cleaned
            elif (state.city or state.police_station) and self._looks_like_location_only_answer(message):
                location = state.police_station or state.city
                if location:
                    facts["incident_timing"] = f"the incident in {location}"
            if any(
                token in lowered
                for token in {
                    "not filed",
                    "have not filed",
                    "haven't filed",
                    "did not file",
                    "filed",
                    "fir",
                    "complaint",
                    "first-report stage",
                }
            ):
                facts["police_status"] = cleaned
            if any(token in lowered for token in {"imei", "invoice", "bill", "box", "serial", "purchase proof", "proof", "receipt"}):
                facts["proof_details"] = cleaned
        elif issue_type == "landlord_harassment":
            if any(token in lowered for token in {"eviction", "lockout", "threat", "deposit", "rent demand", "harassing"}):
                facts["harassment_details"] = cleaned
            if any(token in lowered for token in {"agreement", "rent proof", "messages", "call record", "record", "receipt"}):
                facts["record_status"] = cleaned
            if any(token in lowered for token in {"urgent", "today", "tomorrow", "lockout", "threat", "immediately"}):
                facts["urgency_status"] = cleaned
        elif issue_type == "food_safety":
            if any(token in lowered for token in {"packet", "batch", "invoice", "bill", "sealed", "product"}):
                facts["product_details"] = cleaned
            if any(token in lowered for token in {"photo", "photos", "video", "ill", "illness", "sick", "vomit", "medical"}):
                facts["health_effect"] = cleaned
            if any(token in lowered for token in {"complained", "seller", "brand", "platform", "emailed", "written complaint"}):
                facts["seller_contact"] = cleaned
        elif issue_type == "consumer":
            if any(token in lowered for token in {"invoice", "bill", "payment", "paid", "order", "service"}):
                facts["purchase_details"] = cleaned
            if any(token in lowered for token in {"complaint", "emailed", "wrote", "written", "ticket", "support"}):
                facts["complaint_status"] = cleaned
            if any(token in lowered for token in {"refund", "replacement", "repair", "compensation"}):
                facts["relief_sought"] = cleaned
        elif issue_type == "fir_refusal":
            if any(token in lowered for token in {"incident", "assault", "fraud", "theft", "threat", "harassment", "happened"}):
                facts["incident_details"] = cleaned
            if any(token in lowered for token in {"written complaint", "submission", "receipt", "diary", "refused", "did not register"}):
                facts["submission_status"] = cleaned
            if state.city or state.police_station:
                facts.setdefault("station_details", state.police_station or state.city or cleaned)
        elif issue_type == "police_complaint":
            if any(token in lowered for token in {"incident", "happened", "assault", "fraud", "theft", "threat"}):
                facts["incident_details"] = cleaned
            if any(token in lowered for token in {"proof", "screenshot", "receipt", "recording", "witness", "cctv"}):
                facts["proof_details"] = cleaned
            if state.city or state.police_station:
                facts.setdefault("station_details", state.police_station or state.city or cleaned)
        elif issue_type == "notice":
            if any(token in lowered for token in {"received", "reply", "send", "sending", "draft"}):
                facts["notice_stage_detail"] = cleaned
            if any(token in lowered for token in {"demand", "deadline", "allegation", "payment", "reply"}):
                facts["notice_content"] = cleaned
            if any(token in lowered for token in {"agreement", "invoice", "record", "message", "payment proof"}):
                facts["record_status"] = cleaned
        elif issue_type == "documents":
            if any(token in lowered for token in {"certificate", "document", "application", "verification", "filing", "upload"}):
                facts["document_purpose"] = cleaned
            if any(token in lowered for token in {"rejection", "mismatch", "missing", "upload issue", "attestation", "blocked"}):
                facts["blocker_details"] = cleaned
            if any(token in lowered for token in {"deadline", "last date", "urgent", "tomorrow"}):
                facts["deadline_details"] = cleaned

        return facts

    def _extract_interview_answer(self, *, key: str, message: str, state: ConversationState) -> str:
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        if key == "station_details":
            return state.police_station or state.city or cleaned
        if key == "bank_platform":
            bank = state.bank_name or ""
            platform = state.platform or ""
            combined = " / ".join(item for item in [bank, platform] if item)
            return combined or cleaned
        if key == "incident_timing" and state.issue_type == "snatching_theft":
            if self._looks_like_location_only_answer(cleaned) and (state.police_station or state.city):
                location = state.police_station or state.city
                return f"the incident in {location}"
            if not self._message_has_timing_signal(lowered) and (state.police_station or state.city):
                location = state.police_station or state.city
                return cleaned if cleaned else f"the incident in {location}"
        return cleaned

    @staticmethod
    def _message_has_timing_signal(normalized: str) -> bool:
        return any(
            token in normalized
            for token in {
                "today",
                "yesterday",
                "morning",
                "afternoon",
                "evening",
                "night",
                "am",
                "pm",
                "ago",
                "at ",
                "around ",
                "when ",
                "date",
            }
        )

    @staticmethod
    def _extract_query_evidence_detail(query: str, tokens: set[str]) -> str | None:
        parts = [part.strip() for part in re.split(r"[\n.]", query) if part.strip()]
        for part in reversed(parts):
            lowered = part.lower()
            if any(token in lowered for token in tokens):
                return part
        return None

    @staticmethod
    def _looks_like_location_only_answer(message: str) -> bool:
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        if not cleaned:
            return False
        if ChatService._message_has_timing_signal(lowered):
            return False
        if "police station" in lowered or "city" in lowered:
            return True
        return bool(ChatService._extract_city(cleaned) or ChatService._extract_police_station(cleaned))

    def _compose_virtual_advocate_paragraph(
        self,
        *,
        query: str,
        state: ConversationState,
        facts: dict[str, str],
    ) -> str:
        issue_type = state.issue_type or "general"
        if issue_type == "cyber_fraud":
            bank_platform = (
                facts.get("bank_platform")
                or state.bank_name
                or self._display_platform_name(state.platform)
                or "your bank or payment app"
            )
            timing = facts.get("incident_timing") or "the recent incident"
            reporting = facts.get("reporting_status") or "the current reporting stage"
            amount = facts.get("amount_details") or "the affected transaction details"
            return (
                f"On the facts shared, this looks like a cheating or personation-style cyber-fraud complaint involving {bank_platform}; the core legal position is that, depending on the exact deception used, the facts may commonly be framed under BNS section 318 and BNS section 319, and the police complaint route normally starts through BNSS section 173 for information in a cognizable case. After {timing}, the priority is to preserve {amount}, keep the complaint trail consistent, and report quickly because the present reporting stage ({reporting}) will affect how easily the loss can be traced or escalated."
            )
        if issue_type == "snatching_theft":
            station = facts.get("station_details") or state.city or "the local police station"
            timing = facts.get("incident_timing") or "the incident"
            police_status = facts.get("police_status") or "the complaint stage"
            proof = (
                facts.get("proof_details")
                or self._extract_query_evidence_detail(query, {"imei", "invoice", "bill", "receipt", "serial", "box", "proof"})
                or "the device and evidence details"
            )
            return (
                f"On the facts shared, this appears to be a theft or snatching case; the core legal position is that BNS section 303 covers theft and BNS section 304 specifically addresses snatching, while the police-reporting route usually begins under BNSS section 173. After {timing}, the practical focus is to secure the phone or account side immediately and make sure {station} receives a clean complaint covering the incident, the present police stage ({police_status}), and the available proof such as {proof}."
            )
        if issue_type == "landlord_harassment":
            details = facts.get("harassment_details") or "the landlord pressure"
            records = facts.get("record_status") or "the tenancy records"
            urgency = facts.get("urgency_status") or "the present urgency"
            return (
                f"This looks like a landlord-tenant dispute centred on {details}; the core legal answer is that this is not always a pure BNS case, but where threats or intimidation are present the complaint may also involve BNS section 351 on criminal intimidation, while the non-criminal side depends on tenancy records and written proof. The practical legal position is to avoid reactive steps, preserve {records}, and assess the immediate risk level because {urgency} determines whether the next move should be a written response, complaint, or urgent protection against lockout or threats."
            )
        if issue_type == "food_safety":
            product = facts.get("product_details") or "the product and packet details"
            health = facts.get("health_effect") or "the health impact and photographs"
            seller_contact = facts.get("seller_contact") or "the seller or brand complaint stage"
            return (
                f"This appears to be a food-safety or contamination issue; the core legal answer is that it is usually handled first as a food-safety or consumer grievance rather than a standard BNS/BNSS prosecution matter unless deception, injury, or some separate criminal act is clearly made out. The matter becomes stronger if {product}, {health}, and {seller_contact} are kept aligned before you push for refund, replacement, or escalation to the food-safety or consumer forum."
            )
        if issue_type == "consumer":
            purchase = facts.get("purchase_details") or "the purchase details"
            complaint = facts.get("complaint_status") or "the complaint status"
            relief = facts.get("relief_sought") or "the relief you want"
            return (
                f"This reads like a consumer dispute; the core legal answer is that it is ordinarily pursued through seller, platform, and consumer-forum remedies, and it becomes a BNS matter only if the facts show a separate cheating or intimidation component. The best practical position is to keep {purchase} and {complaint} in one written record so your demand for {relief} is specific, supportable, and ready for the seller, platform, or consumer forum if needed."
            )
        if issue_type == "fir_refusal":
            incident = facts.get("incident_details") or "the underlying incident"
            station = facts.get("station_details") or "the concerned police station"
            submission = facts.get("submission_status") or "the present submission record"
            if not self._facts_clearly_indicate_criminal_issue(incident):
                return (
                    f"This is primarily a complaint-registration issue, so if {station} has not acted on {incident},"
                    f" the practical priority is to preserve {submission} and escalate the same written complaint"
                    " properly instead of relying on repeated oral requests."
                )
            return (
                f"This is primarily an FIR-registration issue; the core legal position is that information about a cognizable offence ordinarily enters the police process through BNSS section 173, so if {station} has not acted on {incident}, the important step is to preserve {submission} and escalate the same written complaint properly instead of relying on repeated oral requests."
            )
        if issue_type == "notice":
            stage = facts.get("notice_stage_detail") or "the notice stage"
            content = facts.get("notice_content") or "the demand and deadline"
            records = facts.get("record_status") or "the agreement and supporting record"
            return (
                f"This is a notice-stage matter; the core legal answer is that it is usually a pre-litigation civil or commercial response question rather than a BNS/BNSS issue unless the notice facts also disclose an independent criminal offence. The safest practical approach is to respond only after {stage}, {content}, and {records} are organized in one factual file, because that will shape whether the next move is a reply, rebuttal, settlement discussion, or further legal drafting."
            )
        if issue_type == "police_complaint":
            incident = facts.get("incident_details") or "the incident facts"
            station = facts.get("station_details") or "the relevant police station"
            proof = facts.get("proof_details") or "the available supporting proof"
            if not self._facts_clearly_indicate_criminal_issue(incident):
                return (
                    f"This looks like a police-complaint matter. The strongest practical course is to reduce"
                    f" {incident} into a short written chronology for {station} and support it with {proof},"
                    " so the complaint record is clear from the first submission."
                )
            return (
                f"This looks like a police-complaint matter; the core legal answer is that where the facts disclose a cognizable offence, the complaint route usually starts through BNSS section 173 and then investigation proceeds under the usual BNSS process. The strongest practical course is to reduce {incident} into a short written chronology for {station} and support it with {proof}, so the complaint record is clear from the first submission."
            )
        if issue_type == "documents":
            purpose = facts.get("document_purpose") or "the filing purpose"
            blocker = facts.get("blocker_details") or "the present blocker"
            deadline = facts.get("deadline_details") or "the deadline position"
            return (
                f"This is more of a filing or document-process problem than a criminal-law issue, so the core legal answer is to identify the exact authority requirement first rather than force it into a BNS/BNSS framework. The immediate focus should be on clarifying {purpose}, resolving {blocker}, and keeping proof of submission in view because {deadline} may affect the next step."
            )
        return (
            f"From what you have shared about {query.strip()}, the right advice depends on getting the facts, timeline, and supporting record into one clear version first, because that determines which authority or next procedural step actually makes sense."
        )

    def _compose_virtual_advocate_steps(self, *, state: ConversationState, facts: dict[str, str]) -> list[str]:
        issue_type = state.issue_type or "general"
        if issue_type == "cyber_fraud":
            steps = []
            if not self._fact_indicates_completed_action(facts.get("reporting_status"), {"bank complaint", "reported", "1930", "cyber crime portal"}):
                steps.append("Notify the bank or payment app immediately and keep the complaint or service reference number.")
                steps.append("Report the same transaction trail through 1930 or the National Cyber Crime Portal without changing the facts.")
            steps.append("Keep the transaction ID, screenshots, statement, and complaint acknowledgements ready for bank or police follow-up.")
            return steps
        if issue_type == "snatching_theft":
            steps = ["Secure the SIM, email, banking apps, and other linked access immediately if a phone was taken."]
            if not self._fact_indicates_completed_action(facts.get("police_status"), {"filed", "fir", "complaint"}):
                steps.append("File or supplement the police complaint with the time, place, item details, and any witness or CCTV clue.")
            steps.append("Keep the IMEI, invoice, complaint copy, and acknowledgement ready for follow-up or escalation.")
            return steps
        if issue_type == "landlord_harassment":
            return [
                "Keep the rent agreement, rent proof, and message trail in one dated file.",
                "Avoid oral-only arguments and respond in writing where possible.",
                "If there is lockout, force, or threat, preserve proof immediately and treat that part as urgent.",
            ]
        if issue_type == "food_safety":
            steps = ["Preserve the packet, batch details, invoice, and defect photographs."]
            if not self._fact_indicates_completed_action(facts.get("seller_contact"), {"complained", "seller", "brand", "platform", "emailed", "written complaint"}):
                steps.append("Send a written complaint to the seller, brand, or platform and keep the reply trail.")
            steps.append("If the product caused illness or appears unsafe, prepare the record for food-safety or consumer escalation.")
            return steps
        if issue_type == "consumer":
            steps = []
            if not self._fact_indicates_completed_action(facts.get("complaint_status"), {"complaint", "emailed", "written", "ticket", "support"}):
                steps.append("Put the complaint in writing to the seller, platform, or service provider.")
            steps.extend(
                [
                    "Collect the invoice, payment proof, chats, warranty terms, and defect or service evidence.",
                    "State the exact relief you want before escalating further.",
                ]
            )
            return steps
        if issue_type == "fir_refusal":
            return [
                "Keep a dated copy of the original complaint and any proof of refusal or non-registration.",
                "Escalate the same complaint in writing to the senior police officer with the supporting record.",
                "Preserve every submission receipt, diary note, or acknowledgement for the next step.",
            ]
        if issue_type == "notice":
            steps = []
            if not self._fact_indicates_completed_action(facts.get("notice_stage_detail"), {"received"}):
                steps.append("Collect the notice, agreement, payment records, and message trail in one file.")
            steps.extend(
                [
                    "Map the exact demand, deadline, and your factual reply position before sending anything.",
                    "Keep proof of dispatch or delivery once the reply or notice is sent.",
                ]
            )
            return steps
        if issue_type == "police_complaint":
            return [
                "Write the incident in chronological order before visiting or submitting to the station.",
                "Carry the main supporting records and ask for an acknowledgement or diary entry.",
                "If the complaint is not recorded properly, keep the written copy for escalation.",
            ]
        if issue_type == "documents":
            return [
                "Confirm the exact filing format and supporting document requirement.",
                "Prepare one clean submission set and one personal record set.",
                "Keep the receipt or acknowledgement immediately after submission.",
            ]
        return [
            "Write down the facts and timeline in one short note.",
            "Keep the key proof and records together before taking the next legal step.",
        ]

    @staticmethod
    def _fact_indicates_completed_action(text: str | None, keywords: set[str]) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized:
            return False
        negative_markers = {
            "not yet",
            "not reported",
            "have not",
            "haven't",
            "did not",
            "no complaint",
            "not filed",
            "didn't",
            "planning to send",
        }
        if any(marker in normalized for marker in negative_markers):
            return False
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _forum_for_issue(issue_type: str) -> str | None:
        mapping = {
            "cyber_fraud": "Bank / Cyber Crime Portal / Cyber Cell",
            "snatching_theft": "Police Station",
            "landlord_harassment": "Police / Civil or Rent Dispute Forum",
            "food_safety": "Seller / Food Safety Authority / Consumer Forum",
            "consumer": "Seller / Consumer Forum",
            "fir_refusal": "Senior Police Officer",
            "notice": "Advocate Notice / Pre-litigation",
            "police_complaint": "Police Station",
            "documents": "Relevant Filing Authority",
        }
        return mapping.get(issue_type, "Relevant Authority / Appropriate Forum")

    @staticmethod
    def _documents_for_issue(issue_type: str, facts: dict[str, str]) -> list[str]:
        mapping = {
            "cyber_fraud": ["transaction ID", "screenshots", "bank statement", "complaint acknowledgement"],
            "snatching_theft": ["complaint copy", "IMEI or item details", "invoice", "ID proof"],
            "landlord_harassment": ["rent agreement", "rent proof", "messages", "notice copy"],
            "food_safety": ["packet", "invoice", "photos", "medical record if any"],
            "consumer": ["invoice", "payment proof", "complaint copy", "photos or chats"],
            "fir_refusal": ["written complaint", "proof of refusal", "supporting evidence"],
            "notice": ["notice copy", "agreement", "payment records", "communications"],
            "police_complaint": ["written complaint", "ID proof", "supporting evidence"],
            "documents": ["application copy", "ID proof", "acknowledgement"],
        }
        documents = mapping.get(issue_type, ["supporting documents"])
        return [item for item in documents if item]

    @staticmethod
    def _legal_references_for_issue(*, state: ConversationState, facts: dict[str, str]) -> list[str]:
        issue_type = state.issue_type or "general"
        if issue_type == "cyber_fraud":
            return [
                "BNS section 318 (cheating)",
                "BNS section 319 (cheating by personation)",
                "BNSS section 173 (information in cognizable cases)",
            ]
        if issue_type == "snatching_theft":
            return [
                "BNS section 303 (theft)",
                "BNS section 304 (snatching)",
                "BNSS section 173 (information in cognizable cases)",
            ]
        if issue_type == "fir_refusal":
            incident = (facts.get("incident_details") or "").lower()
            if not any(token in incident for token in {"fraud", "cheat", "threat", "snatch", "stolen", "theft", "robbed", "assault"}):
                return []
            return [
                "BNSS section 173 (information in cognizable cases)",
                "BNSS section 175 (police officer's power to investigate cognizable case)",
            ]
        if issue_type == "landlord_harassment":
            harassment = (facts.get("harassment_details") or "").lower()
            refs = []
            if any(token in harassment for token in {"threat", "threatening", "intimidation"}):
                refs.append("BNS section 351 (criminal intimidation)")
            refs.append("BNSS section 173 (information in cognizable cases) where threats or other cognizable facts are disclosed")
            return refs
        if issue_type == "police_complaint":
            incident = (facts.get("incident_details") or "").lower()
            if not any(token in incident for token in {"fraud", "cheat", "threat", "snatch", "stolen", "theft", "robbed", "assault"}):
                return []
            return [
                "BNSS section 173 (information in cognizable cases)",
                "BNSS section 175 (police officer's power to investigate cognizable case)",
            ]
        return []

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
        if issue_type == "fir_refusal":
            return self._fir_refusal_guidance(conversation_state)
        if issue_type == "landlord_harassment":
            return self._landlord_harassment_guidance(conversation_state)
        if issue_type == "food_safety":
            return self._food_safety_guidance(query, conversation_state)
        if issue_type == "snatching_theft":
            return self._snatching_theft_guidance(query, conversation_state)
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
        if any(
            token in normalized
            for token in {
                "refuse to file fir",
                "refused to file fir",
                "police not filing fir",
                "police refused",
                "fir not registered",
                "fir refusal",
                "police not taking complaint",
            }
        ):
            return "fir_refusal"
        if any(
            token in normalized
            for token in {
                "landlord",
                "tenant",
                "evict",
                "eviction",
                "rent",
                "deposit not returned",
                "security deposit",
                "harassing me",
                "lock me out",
            }
        ):
            return "landlord_harassment"
        if any(
            token in normalized
            for token in {
                "snatching",
                "snatched",
                "mobile snatching",
                "phone snatched",
                "phone stolen",
                "mobile stolen",
                "stolen phone",
                "theft",
                "stolen",
                "robbery",
                "robbed",
            }
        ):
            return "snatching_theft"
        if any(
            token in normalized
            for token in {
                "insect",
                "expired",
                "contaminated",
                "contamination",
                "unsafe food",
                "food poisoning",
                "spoiled",
                "spoilt",
                "defect in food",
                "sealed packet",
                "packaged food",
                "chocolate",
            }
        ):
            return "food_safety"
        if any(token in normalized for token in {"fir", "police complaint", "police", "complaint"}):
            return "police_complaint"
        if "consumer" in normalized or any(
            token in normalized
            for token in {"refund", "defective", "seller", "service", "warranty", "broken", "damaged", "defect", "replacement"}
        ):
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
                "summary": "This appears to be a cyber-fraud or unauthorized debit complaint, so the immediate priority is to stop further loss and create an early complaint trail.",
                "legal_position": "In a fake-link, OTP, or unauthorized debit situation, the practical value comes from urgent reporting, preserving the transaction trail, and keeping every complaint reference consistent across the bank and the cyber-crime channel.",
                "practical_next_steps": [
                    "Immediately contact the bank, card, wallet, or payment app helpline and ask them to block further debit or freeze the affected channel, then note the complaint number.",
                    "Submit the same facts without delay through the National Cyber Crime Portal or by calling 1930, and keep the acknowledgement, screenshots, and debit alerts together. Reporting within 24 hours usually improves the chances of tracing and recovery.",
                    "Prepare a file with the transaction ID, account statement, SMS or email alerts, fake-link or chat screenshots, and your ID proof in case the bank, cyber cell, or police ask for supporting records.",
                ],
                "lines": [
                    "Immediate steps: block the payment channel, call the bank or wallet helpline, and secure the account credentials.",
                    "File a complaint on the National Cyber Crime Portal or call 1930 as early as possible.",
                    "Keep transaction IDs, screenshots, bank SMS alerts, account statements, and complaint acknowledgements ready.",
                ],
                "likely_forum": "Cyber Crime Portal / Police Station / Bank",
                "authorities": ["cyber cell", "bank", "police"],
                "documents_to_keep": ["transaction ID", "screenshots", "bank statement", "complaint acknowledgement"],
                "caution": "Act quickly because reversal and fraud-tracing options are time-sensitive, and reporting within 24 hours usually gives the best recovery chance.",
                "follow_up_question": "Which platform or bank account was involved in the fraud?",
            }

        bank_or_platform = state.bank_name or state.platform or "the affected account"
        return {
            "summary": f"This looks like a cyber-fraud complaint linked to {bank_or_platform}, so the answer should move straight to bank or platform reporting, cyber complaint registration, and record preservation.",
            "legal_position": f"Because {bank_or_platform} is already identified, the practical legal route is to use the same transaction facts with the bank or platform, the cyber-crime reporting channel, and the police or cyber cell if the debit has already gone through.",
            "practical_next_steps": [
                f"Contact {bank_or_platform} immediately, ask them to block further misuse, and record the complaint or service reference number.",
                "Register the incident on the National Cyber Crime Portal or by calling 1930 using the same transaction details, screenshots, debit alerts, and account statement. Reporting within 24 hours usually improves the chances of tracing and recovery.",
                "If the amount has already been debited, carry the complaint references, transaction trail, and ID proof to the cyber cell or local police station so the record is preserved consistently.",
            ],
            "lines": [
                f"Because {bank_or_platform} was involved, contact that bank or platform immediately, ask them to block further transactions, and request a complaint or reference number.",
                "Submit the same facts on the National Cyber Crime Portal or by calling 1930, then attach the screenshots, transaction ID, debit message, and bank statement.",
                "If the money has already been debited, visit the nearest cyber cell or police station with your ID proof and ask them to record the complaint with the transaction trail.",
            ],
            "likely_forum": "Cyber Crime Portal / Bank / Local Police or Cyber Cell",
            "authorities": ["cyber cell", "bank", "police"],
            "documents_to_keep": ["transaction ID", "screenshots", "bank statement", "complaint reference number"],
            "caution": "Keep every complaint reference number because banks and cyber cells often ask for the earlier complaint acknowledgement, and early reporting is important for recovery.",
            "follow_up_question": None,
        }

    def _consumer_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        return {
            "summary": "This appears to be a consumer dispute, so the first useful move is a written complaint backed by payment proof and clear defect or deficiency evidence.",
            "legal_position": "Consumer matters are easier to pursue when the product or service issue, the money paid, and the exact relief claimed are all stated clearly in writing before escalation.",
            "practical_next_steps": [
                "Send a written complaint to the seller, brand, or service provider and keep delivery proof, ticket number, email trail, or chat acknowledgment.",
                "Collect the invoice, payment proof, warranty terms, chats, and photographs or recordings that show the defect, delay, or deficiency in service.",
                "If the problem is not resolved, prepare a concise consumer complaint file stating the chronology, the loss suffered, and the exact refund, replacement, repair, or compensation requested.",
            ],
            "lines": [
                "Start by sending a written complaint to the seller or service provider and keep proof of delivery.",
                "Collect the invoice, payment proof, chats or emails, warranty terms, and photographs of the defect or deficiency.",
                "If the issue is not resolved, prepare a consumer complaint that clearly states the defect or service deficiency, the loss suffered, and the refund, replacement, repair, or compensation you are seeking.",
            ],
            "likely_forum": "Consumer Commission",
            "authorities": ["seller", "consumer commission"],
            "documents_to_keep": ["invoice", "payment proof", "complaint copy", "photos"],
            "caution": "Make sure the chronology and the exact refund or compensation amount are clearly documented.",
            "follow_up_question": "Was any written complaint already sent to the seller or service provider?",
        }

    def _fir_refusal_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        location = state.police_station or (f"{state.city} police station" if state.city else "the local police station")
        return {
            "summary": f"This is an FIR-registration refusal issue at {location}, so the response should focus on preserving the complaint record and escalating it properly.",
            "legal_position": "When the local police do not register the complaint, practical progress usually comes from re-submitting the same dated complaint with proof and escalating it to the senior officer rather than relying only on oral follow-up.",
            "practical_next_steps": [
                f"Keep a dated copy of the complaint given to {location}, note the officer details, and preserve any diary entry, refusal note, or submission proof already available.",
                "Escalate the same incident facts in writing to the senior police officer, attaching the earlier complaint copy and the evidence that the local station did not act on it.",
                "Keep all supporting material such as screenshots, medical papers, recordings, witness details, and ID proof together so the escalation remains fact-consistent.",
            ],
            "lines": [
                f"If the police at {location} are refusing to register the FIR or complaint, keep a dated copy of your written complaint, the officer details, and any refusal or diary reference that you were given.",
                "Escalate the same complaint in writing to the senior police officer with the incident facts, evidence, and proof that the local station did not register it, and keep proof of delivery or submission.",
                "Preserve the complaint copy, screenshots, recordings, medical papers, witness details, or other supporting records so you can show that you acted promptly and consistently.",
            ],
            "likely_forum": "Senior Police Officer / Police Station",
            "authorities": ["police", "senior police officer"],
            "documents_to_keep": ["written complaint", "proof of refusal or submission", "supporting records", "ID proof"],
            "caution": "Do not rely on only a verbal refusal because escalation works better when you preserve the written complaint and proof of submission.",
            "follow_up_question": "Which city or police station refused to register the complaint?",
        }

    def _landlord_harassment_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        location = state.city or "your area"
        return {
            "summary": f"This looks like a landlord-harassment or eviction-pressure situation in {location}, so the answer should focus on tenancy records, written proof, and safe escalation.",
            "legal_position": "Landlord disputes become more manageable when rent proof, possession records, messages, and any notice or threat are preserved before responding or escalating the matter.",
            "practical_next_steps": [
                "Arrange the rent agreement, rent receipts or transfer proof, messages, calls, and any notice or demand in date order so the pressure tactics are clearly documented.",
                "Respond in writing where possible, keep paying or recording rent properly if that is part of the dispute, and avoid vacating, handing over originals, or making admissions under pressure.",
                "If the conduct includes threats, lockout, forced entry, or seizure of belongings, preserve immediate proof and treat that part as an urgent complaint issue while preparing the next civil or notice step.",
            ],
            "lines": [
                f"If the landlord is harassing, threatening eviction, or interfering with possession in {location}, first preserve the rent agreement, rent-payment proof, messages, call recordings, and any notice or threat details in date order.",
                "Do not stop keeping records of rent or communication, and respond in writing where possible so there is a clear paper trail of the harassment, illegal demand, or eviction threat.",
                "If the conduct involves threats, forced entry, lockout, or seizure of belongings, treat that as an urgent complaint issue and preserve proof before approaching the police or taking the next housing-related legal step.",
            ],
            "likely_forum": "Police Station / Rent or Civil Dispute Forum / Advocate Notice",
            "authorities": ["landlord", "police", "advocate"],
            "documents_to_keep": ["rent agreement", "rent receipts", "messages", "notice copy", "ID proof"],
            "caution": "Do not hand over original documents or vacate under pressure without documenting the facts and getting proper written records.",
            "follow_up_question": "Is the problem about eviction, deposit, lockout, or threats from the landlord?",
        }

    def _food_safety_guidance(self, query: str, state: ConversationState) -> dict[str, str | list[str] | None]:
        product_hint = "the food product"
        if "chocolate" in query.lower():
            product_hint = "the chocolate packet"
        return {
            "summary": f"This appears to be a contamination or unsafe-food complaint involving {product_hint}, so the immediate priority is to preserve the packet and defect evidence before the matter becomes harder to prove.",
            "legal_position": "Food contamination complaints are much stronger when the packet, batch details, expiry, invoice, defect photographs, and any illness records are preserved before the seller or authority is approached.",
            "practical_next_steps": [
                f"Do not discard {product_hint}; keep the packet, batch number, expiry date, and clear photographs or video of the defect in the same condition as far as possible.",
                "Send a written complaint to the seller, brand, or platform asking for refund or replacement and keep the reply, complaint number, and invoice or payment proof.",
                "If the product appears unsafe or someone fell ill, prepare the packet details, invoice, photos, and any medical papers for escalation to the food safety authority or consumer forum.",
            ],
            "lines": [
                f"Because the issue appears to be contamination or an unsafe defect in {product_hint}, preserve the packet, batch details, expiry date, purchase bill, and clear photos or video of the defect before consuming or discarding it.",
                "Start with a written complaint to the seller or brand and ask for a refund, replacement, and written acknowledgment, then keep all chats, emails, and complaint reference numbers.",
                "If the product appears unsafe or harmful, escalate the matter to the food safety authority or consumer forum with the packet details, invoice, photos, and any medical records if illness occurred.",
            ],
            "likely_forum": "Seller / Brand / Food Safety Authority / Consumer Commission",
            "authorities": ["seller", "food safety authority", "consumer commission"],
            "documents_to_keep": ["product packet", "invoice", "photos or video", "written complaint", "medical records if any"],
            "caution": "Do not rely only on a verbal complaint because contamination or unsafe-food claims are stronger when the packet details and defect evidence are preserved.",
            "follow_up_question": "Do you still have the packet, invoice, and photographs of the defect?",
        }

    def _snatching_theft_guidance(self, query: str, state: ConversationState) -> dict[str, str | list[str] | None]:
        item_hint = "the stolen property"
        lowered = query.lower()
        if "mobile" in lowered or "phone" in lowered:
            item_hint = "the mobile phone"
        if state.police_station or state.city:
            location = state.police_station or f"{state.city} police station"
            return {
                "summary": f"This is a snatching or theft complaint involving {item_hint} linked to {location}, so the next step is to strengthen the police record and prevent misuse of the device or account.",
                "legal_position": "Once the place of complaint is known, the practical focus is to ensure the police record contains the incident details, item identifiers, and any account-risk material linked to the stolen phone or property.",
                "practical_next_steps": [
                    f"File or supplement the complaint at {location} with the exact date, time, route, item description, witness details, and any CCTV or location clues.",
                    "If a phone was taken, block the SIM, secure the connected email and banking apps, and keep the IMEI number, purchase invoice, and screenshots of linked accounts ready.",
                    "Ask for the complaint acknowledgement or FIR details, and if the record is not being taken properly, preserve the written complaint copy for escalation.",
                ],
                "lines": [
                    f"Since the matter concerns {item_hint} and is linked to {location}, file or supplement the police complaint there with the date, place, route, device details, and any witness or CCTV information.",
                    "If a phone was taken, block the SIM, secure the connected email and banking apps, keep the IMEI number, and preserve device-purchase records and screenshots showing the linked accounts.",
                    "Ask for an acknowledgment or FIR details, and if the police refuse to record the complaint, preserve the written complaint copy and escalate it to the senior officer with submission proof.",
                ],
                "likely_forum": "Police Station / Senior Police Officer",
                "authorities": ["police", "telecom provider"],
                "documents_to_keep": ["written complaint", "IMEI or device details", "purchase invoice", "ID proof", "acknowledgement"],
                "caution": "Act quickly after snatching or theft because SIM misuse, device access, and recovery leads are time-sensitive.",
                "follow_up_question": None,
            }
        return {
            "summary": f"This appears to be a snatching or theft issue involving {item_hint}, so the immediate priority is incident documentation, device or account protection, and prompt police reporting.",
            "legal_position": "Snatching and theft matters usually become more workable when the first complaint captures the place, time, item details, and linked phone or banking risk without delay.",
            "practical_next_steps": [
                "Write down the exact time, place, route, item description, and how the snatching or theft happened before those details start changing.",
                "If a mobile phone was taken, block the SIM immediately, secure linked email and banking access, and collect the IMEI number and purchase invoice or device-box details.",
                "Take a written complaint with the incident facts, item identifiers, witness details, and any CCTV leads to the police station and keep the diary number or acknowledgement.",
            ],
            "lines": [
                f"Because this appears to be a snatching or theft issue involving {item_hint}, write down the exact time, place, route, item details, and how the incident happened before filing the complaint.",
                "If a phone was taken, immediately block the SIM, secure linked email and banking apps, note the IMEI number, and keep the purchase invoice or device box details ready.",
                "File a police complaint or FIR with the incident facts, item details, witnesses, and any CCTV leads, and keep the acknowledgment or diary number for follow-up.",
            ],
            "likely_forum": "Police Station",
            "authorities": ["police", "telecom provider"],
            "documents_to_keep": ["written complaint", "IMEI or device details", "purchase invoice", "ID proof"],
            "caution": "Move quickly after the incident because snatched phones or stolen devices may be misused for OTP, wallet, or account access.",
            "follow_up_question": "Which city or police station is connected with the snatching or theft complaint?",
        }

    def _notice_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        if state.notice_stage:
            return {
                "summary": f"This is a notice matter at the {state.notice_stage} stage, so the next useful step is a document-backed, deadline-conscious reply or draft.",
                "legal_position": "Notice disputes are handled best when the allegations, timeline, and supporting records are organized first, so the next reply or notice stays factual and avoids avoidable admissions.",
                "practical_next_steps": [
                    "Collect the notice copy, agreement, payment records, and date-wise timeline before drafting anything further.",
                    "Prepare a point-by-point reply or notice draft that addresses the allegations or demand with only the necessary supporting documents attached.",
                    "Keep proof of dispatch, email delivery, or courier tracking and do not miss the stated deadline.",
                ],
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
            "summary": "This looks like a legal-notice matter, so the immediate need is to organize the record before sending or replying to anything.",
            "legal_position": "A useful notice response depends on the exact demand, deadline, and supporting record, so the facts should be sorted first before a reply is drafted.",
            "practical_next_steps": [
                "Collect the notice, agreement, invoices, payment records, and message trail in date order.",
                "Identify exactly what the other side is demanding, by when, and what part of it you agree or disagree with.",
                "Prepare a factual reply or draft that states your position clearly and keeps proof of dispatch or delivery.",
            ],
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
                "summary": f"This is a police-complaint issue linked to {location}, so the next useful step is a clean written complaint with evidence and acknowledgement.",
                "legal_position": "Police-complaint matters usually move better when the incident chronology, identities, and supporting records are presented in one written complaint and the submission is acknowledged.",
                "practical_next_steps": [
                    f"Prepare a short complaint for {location} with dates, place, names, and the exact incident in chronological order.",
                    f"Carry ID proof, screenshots, receipts, recordings, and witness details, and ask the police in {city_text} for a diary number or FIR acknowledgement if the offence is cognizable.",
                    "If the complaint is not being registered, keep a copy of the written submission and preserve proof so the matter can be escalated to the senior officer.",
                ],
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
            "summary": "This looks like a police-complaint issue, so the immediate need is a clear written incident note supported by the main records.",
            "legal_position": "Police complaints are usually more effective when the chronology, names, place, and supporting records are written down before you approach the station.",
            "practical_next_steps": [
                "Write a short complaint in date order covering the incident, people involved, place, and immediate evidence.",
                "Carry ID proof and the main supporting material such as screenshots, receipts, recordings, or witness details.",
                "Ask for FIR registration if the matter is cognizable, or at least keep the diary number or acknowledgement for follow-up.",
            ],
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
                "summary": f"This is a document-process issue for {state.document_type}, so the practical answer is to match the authority's filing format and preserve acknowledgement.",
                "legal_position": "Document submissions usually go smoothly only when the required copy type, attestation requirement, and acknowledgement process are confirmed in advance.",
                "practical_next_steps": [
                    "Confirm whether the authority needs originals, self-attested copies, notarized copies, or a specific online-upload format.",
                    "Keep the ID proof, address proof, application form, and supporting records arranged as one submission set plus one personal record set.",
                    "After submission, keep the stamped acknowledgement, receipt, or upload confirmation safely for later follow-up.",
                ],
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
            "summary": "This looks like a document or filing-process question, so the immediate need is to confirm the exact requirement and keep an acknowledgement trail.",
            "legal_position": "Document-process issues usually depend less on legal argument and more on filing the correct document set in the right format while preserving proof of submission.",
            "practical_next_steps": [
                "Identify the office or platform involved and confirm whether it needs originals, self-attested copies, notarized copies, or an online upload.",
                "Arrange the ID proof, address proof, application form, and supporting records in one folder before submission.",
                "Keep both the submitted set and the acknowledgement or receipt copy for follow-up.",
            ],
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
        platforms = ["PhonePe", "Google Pay", "Paytm", "WhatsApp", "Telegram", "Instagram", "Facebook"]
        lowered = message.lower()
        for platform in platforms:
            if platform.lower() in lowered:
                return platform
        return None

    @staticmethod
    def _display_platform_name(platform: str | None) -> str | None:
        text = (platform or "").strip()
        if not text or text.lower() == "upi":
            return None
        return text

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

    def _resolve_doctype_candidates(
        self,
        query: str,
        state: str | None,
        domain: str | None,
        answer_mode: str = "grounded_general",
    ) -> list[str | None]:
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
            "civil": "judgments,laws",
            "tax": "laws,judgments",
            "corporate": "laws,judgments",
            "document_review": "laws,judgments",
            "procedure": "judgments,laws",
            "cyber": "judgments,laws",
            "general": "judgments,laws",
        }
        state_key = (state or "").strip().lower()
        state_doctypes = state_map.get(state_key)
        domain_doctypes = domain_defaults.get(domain or "general", "judgments,laws")
        candidates: list[str | None] = []

        if answer_mode == "statute_first":
            candidates.extend(["laws", domain_doctypes, "judgments", None, state_doctypes])
        elif answer_mode == "case_first":
            candidates.extend(["judgments", domain_doctypes, state_doctypes, "laws", None])
        elif self._is_statute_query(query):
            candidates.extend([domain_doctypes, "laws", "judgments", None, state_doctypes])
        else:
            candidates.extend([domain_doctypes, state_doctypes, "judgments", "laws", None])

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

        if any(token in lowered for token in {"upi", "fake link", "phishing", "otp", "debit", "cyber fraud", "bank fraud"}):
            variants.extend(
                [
                    f"{cleaned} legal remedy India",
                    f"{cleaned} cyber fraud complaint India",
                    "unauthorized UPI debit cyber fraud India",
                ]
            )
        if "judgement" in lowered:
            variants.append(cleaned.replace("judgement", "judgment"))

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
            if value in ("",) or value in deduped:
                continue
            deduped.append(value)
        return deduped

    @staticmethod
    def _coerce_string_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()][:6]
        return []

    @staticmethod
    def _optional_string(value: object) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None
