from __future__ import annotations

import json
import re
import difflib
from datetime import datetime, timezone
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import requests

from backend.app.core.config import Settings
from backend.app.models.schemas import ChatRequest, ChatUploadResponse, ConversationState, InternalChatResult
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.domain_packs import LegalDomainPackService
from backend.app.services.google_custom_search_service import GoogleCustomSearchService

from backend.app.services.indiankanoon_service import IndianKanoonService, SOURCE_AUTHORITY_SCORES
from backend.app.services.legal_dataset_service import LegalProvisionMatch, LocalLegalDatasetService
from backend.app.services.legal_hybrid_retrieval import LegalHybridRetrievalService
from backend.app.services.intent_service import IntentRoutingService
from backend.app.services.legal_domain_classifier import LegalDomainClassifier

from backend.app.services.local_ml import LocalSemanticSupportChecker, LocalTextSimilarityService
from backend.app.services.openai_service import OpenAIResponsesService

from backend.app.services.storage import SessionStoreProtocol
from backend.app.utils.request_context import get_logger

logger = get_logger("lawyer_ai.chat_service")

DIRECT_ANSWER_CACHE_TTL_SECONDS = 300.0
DIRECT_ANSWER_CACHE_MAX_ENTRIES = 256

FAST_AUTHORITY_LOOKUPS: dict[str, dict[str, Any]] = {
    "constitution_article_14": {
        "title": "Article 14 of the Constitution of India",
        "summary": "Article 14 guarantees equality before the law and equal protection of the laws in India.",
        "legal_position": "It is the core equality clause. State action should not be arbitrary, and any legal classification generally has to rest on an intelligible differentia linked to a legitimate objective.",
        "next_steps": "Read the exact article text first, then match the alleged unequal treatment or arbitrariness to the facts, dates, and public authority involved before relying on it in any notice, writ, or challenge.",
        "source": "Constitution of India | Article 14",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["relevant order or notice", "communications", "supporting records", "ID proof"],
        "caution": "How Article 14 applies depends heavily on the facts, the public action challenged, and the latest court interpretation.",
        "disclaimer_mode": "medium_risk",
    },
    
    "constitution_article_19": {
        "title": "Article 19 of the Constitution of India",
        "summary": "Article 19 protects key freedoms such as speech and expression, assembly, association, movement, residence, and profession, subject to constitutionally permitted restrictions.",
        "legal_position": "It applies through its separate clauses, and each freedom can be limited only on the grounds recognized in the Constitution. The exact protection depends on which clause is involved and the nature of the restriction.",
        "next_steps": "Identify the exact freedom involved and the restriction imposed, then read the article text with the relevant clause before relying on it in any constitutional challenge or representation.",
        "source": "Constitution of India | Article 19",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["impugned order or restriction", "communications", "supporting records", "ID proof"],
        "caution": "Article 19 issues are clause-specific, so the exact sub-clause and restriction ground should be checked carefully.",
        "disclaimer_mode": "medium_risk",
    },
    
    "constitution_article_21": {
        "title": "Article 21 of the Constitution of India",
        "summary": "Article 21 protects life and personal liberty except according to procedure established by law.",
        "legal_position": "It is a foundational rights provision and has been interpreted broadly by courts. In simple terms, State action affecting life or personal liberty usually has to follow a lawful, fair, and non-arbitrary procedure.",
        "next_steps": "Read the article text first, then identify the State action, detention, restriction, or deprivation involved and preserve the relevant orders, notices, timeline, and supporting records before taking the next legal step.",
        "source": "Constitution of India | Article 21",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["order or notice copy", "timeline", "supporting records", "ID proof"],
        "caution": "The real effect of Article 21 depends on the exact facts and the latest constitutional case law on the issue.",
        "disclaimer_mode": "medium_risk",
    },
    
    "constitution_article_22": {
        "title": "Article 22 of the Constitution of India",
        "summary": "Article 22 provides protections in arrest and detention matters.",
        "legal_position": "In broad terms, it addresses safeguards such as being informed of the grounds of arrest, consulting a legal practitioner, and production before a magistrate within the constitutionally recognized timeframe, subject to the preventive-detention clauses.",
        "next_steps": "Check the exact arrest or detention stage, keep the arrest memo, remand papers, and timeline ready, and then match the facts against the specific safeguard you want to rely on.",
        "source": "Constitution of India | Article 22",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["arrest memo", "remand papers", "timeline", "ID proof"],
        "caution": "Article 22 questions are highly fact-sensitive and can differ between ordinary arrest and preventive detention.",
        "disclaimer_mode": "high_risk",
    },
    
    "constitution_article_32": {
        "title": "Article 32 of the Constitution of India",
        "summary": "Article 32 gives the right to move the Supreme Court for enforcement of fundamental rights.",
        "legal_position": "It is a constitutional remedy provision. In general terms, it supports approaching the Supreme Court where a fundamental-rights violation is alleged and the constitutional conditions for invoking that remedy are met.",
        "next_steps": "Identify the specific fundamental right said to be violated, preserve the challenged action and supporting timeline, and then check whether the constitutional remedy is really the appropriate forum for the facts.",
        "source": "Constitution of India | Article 32",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Supreme Court",
        "documents_to_keep": ["challenged order", "timeline", "supporting records", "ID proof"],
        "caution": "A constitutional remedy under Article 32 is forum-sensitive, so the exact maintainability position should be checked carefully.",
        "disclaimer_mode": "high_risk",
    },
    
    "constitution_article_226": {
        "title": "Article 226 of the Constitution of India",
        "summary": "Article 226 empowers High Courts to issue writs and directions in appropriate cases.",
        "legal_position": "In simple terms, it is a broad constitutional remedy provision used before High Courts. Its practical use depends on the nature of the right, the authority involved, territorial jurisdiction, and the maintainability of the writ remedy.",
        "next_steps": "Identify the State or public authority involved, preserve the challenged action and timeline, and then check the territorial High Court and the exact writ or relief that fits the facts.",
        "source": "Constitution of India | Article 226",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "High Court",
        "documents_to_keep": ["challenged order", "timeline", "supporting records", "ID proof"],
        "caution": "Article 226 remedies are forum- and fact-sensitive, so the territorial and maintainability position should be checked before acting.",
        "disclaimer_mode": "high_risk",
    },
    
    "constitution_article_300a": {
        "title": "Article 300A of the Constitution of India",
        "summary": "Article 300A protects a person from being deprived of property except by authority of law.",
        "legal_position": "In simple terms, property can be taken away only under a valid law and through a legally supported process. It is not framed like a Fundamental Right, but it remains an important constitutional protection against unlawful deprivation of property.",
        "next_steps": "Read the exact article text, then compare it with the acquisition notice, order, mutation action, demolition step, or other property-related action affecting you before taking the next legal step.",
        "source": "Constitution of India | Article 300A",
        "authority": "Constitution of India",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["order or notice copy", "property records", "title documents", "supporting records"],
        "caution": "Article 300A questions are fact-sensitive and often depend on the statutory process, notice, hearing, and remedy structure involved.",
        "disclaimer_mode": "medium_risk",
    },
    
    "ipc_section_420": {
        "title": "Section 420 of the Indian Penal Code, 1860",
        "summary": "Section 420 IPC deals with cheating and dishonestly inducing delivery of property.",
        "legal_position": "In broad terms, it is commonly invoked where deception is alleged to have dishonestly caused a person to deliver property, money, or a valuable security. Whether it applies depends on the dishonest inducement and the facts showing cheating from the outset.",
        "next_steps": "Read the exact section text first, then map the alleged deception, inducement, money or property movement, and supporting documents before relying on it in a complaint, FIR request, or legal response.",
        "source": "Indian Penal Code, 1860 | Section 420",
        "authority": "Indian Penal Code, 1860",
        "domain": "criminal",
        "likely_forum": "Indian Penal Code, 1860",
        "documents_to_keep": ["written complaint", "payment proof", "messages", "supporting records"],
        "caution": "Section 420 questions are fact-sensitive, and the precise criminal framing should be checked against the full incident record and current law.",
        "disclaimer_mode": "high_risk",
    },
    
    "ipc_section_406": {
        "title": "Section 406 of the Indian Penal Code, 1860",
        "summary": "Section 406 IPC concerns punishment for criminal breach of trust.",
        "legal_position": "In broad terms, it is linked to situations where property or dominion over property is entrusted and is then dishonestly misappropriated or dealt with contrary to the legal direction or trust attached to it.",
        "next_steps": "Read the exact section text first, then gather the entrustment record, communications, payment or property trail, and the later misuse allegations before relying on it for a complaint or defence step.",
        "source": "Indian Penal Code, 1860 | Section 406",
        "authority": "Indian Penal Code, 1860",
        "domain": "criminal",
        "likely_forum": "Indian Penal Code, 1860",
        "documents_to_keep": ["entrustment record", "messages", "payment or property trail", "written complaint"],
        "caution": "The distinction between a civil dispute and criminal breach of trust depends on the facts and the original entrustment record.",
        "disclaimer_mode": "high_risk",
    },
    
    "ipc_section_498a": {
        "title": "Section 498A of the Indian Penal Code, 1860",
        "summary": "Section 498A IPC addresses cruelty by the husband or his relatives toward a married woman.",
        "legal_position": "In simple terms, it is a criminal provision used in matrimonial cruelty allegations. The exact application depends on the specific conduct alleged, the chronology, and the supporting material.",
        "next_steps": "Read the exact section text first, then organize the incident chronology, messages, medical records if any, complaint history, and witness or family records before relying on it in a complaint or legal response.",
        "source": "Indian Penal Code, 1860 | Section 498A",
        "authority": "Indian Penal Code, 1860",
        "domain": "criminal",
        "likely_forum": "Indian Penal Code, 1860",
        "documents_to_keep": ["incident chronology", "messages", "medical records if any", "complaint copies"],
        "caution": "Matrimonial-cruelty matters are highly fact-sensitive and should be matched carefully to the complaint record and current legal position.",
        "disclaimer_mode": "high_risk",
    },
    
    "ni_act_section_138": {
        "title": "Section 138 of the Negotiable Instruments Act, 1881",
        "summary": "Section 138 NI Act concerns cheque dishonour for insufficiency of funds or related banking reasons, subject to the statutory conditions in the provision.",
        "legal_position": "It is the main cheque-dishonour provision. In practice, the exact result depends on the statutory ingredients, the bank return, notice compliance, limitation-sensitive steps, and the surrounding transaction facts.",
        "next_steps": "Read the exact section text first, then keep the cheque, return memo, notice record, dispatch proof, and transaction documents ready before relying on it for a complaint or reply.",
        "source": "Negotiable Instruments Act, 1881 | Section 138",
        "authority": "Negotiable Instruments Act, 1881",
        "domain": "criminal",
        "likely_forum": "Negotiable Instruments Act, 1881",
        "documents_to_keep": ["cheque copy", "bank return memo", "notice copy", "dispatch proof"],
        "caution": "Section 138 matters are very stage-sensitive, so the notice and filing record should be checked carefully against the current law.",
        "disclaimer_mode": "high_risk",
    },
}

CONSTITUTIONAL_EXPLAINER_LOOKUPS: dict[str, dict[str, Any]] = {
    "fundamental_rights": {
        "title": "Fundamental Rights under the Constitution of India",
        "summary": "Fundamental Rights are the core constitutional rights that protect individual liberty, equality, freedom, and legal safeguards against improper State action.",
        "legal_position": "In simple terms, they are enforceable constitutional protections. They broadly cover equality rights, freedom rights, protections in criminal-law situations, freedom of religion, cultural and educational rights, and constitutional remedies.",
        "next_steps": "If you only need the concept, start with the constitutional grouping of the rights and the article ranges. If you need to act on a violation, identify the exact right affected, the State action involved, and the relevant documents or timeline first.",
    
        "article_breakdown": [
            "Articles 12-13: define the State for Part III and make laws inconsistent with Fundamental Rights vulnerable to challenge.",
            "Articles 14-18: Right to Equality, including equality before law, non-discrimination, equality of opportunity, abolition of untouchability, and abolition of titles.",
            "Articles 19-22: Right to Freedom, including freedoms under Article 19 and criminal-law protections under Articles 20, 21, and 22.",
            "Articles 23-24: Right against Exploitation, including prohibition of trafficking, forced labour, and child labour in hazardous work.",
            "Articles 25-28: Freedom of Religion.",
            "Articles 29-30: Cultural and Educational Rights, especially for minorities.",
            "Article 32: Right to Constitutional Remedies before the Supreme Court.",
            "Articles 33-35: special provisions on modification, application, and implementation of certain Fundamental Rights.",
        ],
    
        "sources": "Constitution of India | Fundamental Rights overview",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["relevant order or notice", "timeline", "supporting records"],
        "caution": "The exact remedy depends on which specific right is involved and the facts of the alleged violation.",
        "disclaimer_mode": "medium_risk",
    },
    
    "fundamental_duties": {
        "title": "Fundamental Duties under the Constitution of India",
        "summary": "Fundamental Duties are constitutional expectations placed on citizens to uphold constitutional values, public spirit, and civic responsibility.",
        "legal_position": "In simple terms, they are not usually framed like ordinary personal claims against the State, but they remain an important constitutional guide to civic conduct and constitutional interpretation.",
        "next_steps": "If you need a basic explanation, focus on their role as citizen duties under the Constitution. If your question is tied to a dispute, identify the exact policy, restriction, or public issue involved before going further.",
    
        "article_breakdown": [
            "Article 51A(a): respect the Constitution, its ideals and institutions, the National Flag, and the National Anthem.",
            "Article 51A(b): cherish and follow the ideals of the freedom struggle.",
            "Article 51A(c): uphold and protect the sovereignty, unity, and integrity of India.",
            "Article 51A(d): defend the country and render national service when called upon.",
            "Article 51A(e): promote harmony and the spirit of common brotherhood, and renounce practices derogatory to the dignity of women.",
            "Article 51A(f): value and preserve the rich heritage of the country's composite culture.",
            "Article 51A(g): protect and improve the natural environment and have compassion for living creatures.",
            "Article 51A(h): develop scientific temper, humanism, and the spirit of inquiry and reform.",
            "Article 51A(i): safeguard public property and abjure violence.",
            "Article 51A(j): strive toward excellence in all spheres of individual and collective activity.",
            "Article 51A(k): parent or guardian duty to provide education opportunities to children between six and fourteen years.",
        ],
    
        "sources": "Constitution of India | Fundamental Duties overview",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["relevant notice or policy", "communications", "supporting records"],
        "caution": "A general explanation of fundamental duties is different from a case-specific constitutional remedy analysis.",
        "disclaimer_mode": "medium_risk",
    },
    
    "directive_principles": {
        "title": "Directive Principles of State Policy under the Constitution of India",
        "summary": "Directive Principles of State Policy are constitutional principles meant to guide governance and public policy in India.",
        "legal_position": "In simple terms, they are constitutional governance goals rather than ordinary directly enforceable personal rights. They often help explain the social-welfare and policy direction of the State under the Constitution.",
        "next_steps": "If you need only the concept, read them as constitutional policy principles. If your issue concerns a government action or challenge, identify the exact policy, scheme, or constitutional question involved first.",
    
        "article_breakdown": [
            "Articles 36-37: define the DPSP framework and clarify that these principles guide governance even though they are not directly enforceable in court like Fundamental Rights.",
            "Articles 38-39: promote social justice, welfare, adequate livelihood, equal pay, and prevention of concentration of wealth.",
            "Article 39A: equal justice and free legal aid.",
            "Article 40: organization of village panchayats.",
            "Article 41: right to work, education, and public assistance in certain cases.",
            "Article 42: just and humane conditions of work and maternity relief.",
            "Article 43: living wage and decent standard of life for workers.",
            "Article 43A: participation of workers in management of industries.",
            "Article 43B: promotion of cooperative societies.",
            "Article 44: Uniform Civil Code.",
            "Article 45: early childhood care and education for children.",
            "Article 46: promotion of educational and economic interests of Scheduled Castes, Scheduled Tribes, and other weaker sections.",
            "Article 47: nutrition, public health, and prohibition of intoxicating drinks and drugs harmful to health.",
            "Article 48: agriculture and animal husbandry.",
            "Article 48A: protection and improvement of the environment and safeguarding forests and wildlife.",
            "Article 49: protection of monuments and places of national importance.",
            "Article 50: separation of judiciary from executive in public services.",
            "Article 51: promotion of international peace and security.",
        ],
    
        "sources": "Constitution of India | Directive Principles of State Policy overview",
        "domain": "constitutional",
        "likely_forum": "Constitution of India",
        "documents_to_keep": ["relevant policy or order", "scheme documents", "supporting records"],
        "caution": "A DPSP explanation does not by itself answer whether a specific legal challenge or remedy is maintainable on your facts.",
        "disclaimer_mode": "medium_risk",
    },
}

GENERAL_LEGAL_EXPLAINER_LOOKUPS: dict[str, dict[str, Any]] = {
    "arbitration": {
        "title": "Arbitration",
        "summary": "a private dispute-resolution process where the parties agree to have their dispute decided by an arbitrator instead of going through a full court trial.",
        "legal_position": "In practice, it usually depends on an arbitration clause or a later agreement between the parties. The arbitrator hears both sides and gives an award, which can be binding subject to the limited court challenge framework under arbitration law.",
        "next_steps": "If you are checking whether arbitration applies to your dispute, start by reading the contract for an arbitration clause and identifying the seat, forum, and procedure mentioned there.",
        "domain": "civil",
        "disclaimer_mode": "low_risk",
    },
    
    "fir": {
        "title": "First Information Report (FIR)",
        "summary": "the formal police record of information about a cognizable offence that sets the criminal process in motion.",
        "legal_position": "In practice, it is the starting point for police investigation in cognizable criminal matters. Its exact significance depends on the offence category, the facts disclosed, and the later investigation record.",
        "next_steps": "If you are dealing with a real incident, first organize the date, place, people involved, and any supporting evidence before approaching the police or reviewing the FIR text.",
        "domain": "criminal",
        "disclaimer_mode": "low_risk",
    },
    
    "bail": {
        "title": "Bail",
        "summary": "the legal release of an accused person from custody subject to the conditions imposed by the court or the law.",
        "legal_position": "In broad terms, bail is about liberty during the criminal process, not a final decision on guilt. Whether it is granted depends on the offence, the stage of the case, statutory limits, and the facts placed before the court.",
        "next_steps": "If your question is practical, first identify the offence sections, the arrest or notice stage, and the court handling the matter before deciding the next bail step.",
        "domain": "criminal",
        "disclaimer_mode": "medium_risk",
    },
    
    "anticipatory_bail": {
        "title": "Anticipatory Bail",
        "summary": "the pre-arrest bail protection a court may grant where a person reasonably expects arrest in a non-bailable matter.",
        "legal_position": "In practice, it is a preventive liberty remedy. The result depends on the offence, the facts alleged, the need for custodial interrogation, and the court's view of the case at that stage.",
        "next_steps": "If this relates to a real dispute, first identify the likely offence sections, the police station or complaint stage, and the documents you would rely on before taking the next step.",
        "domain": "criminal",
        "disclaimer_mode": "medium_risk",
    },
    
    "legal_notice": {
        "title": "Legal Notice",
        "summary": "a formal written communication used to state a legal demand, allegation, or proposed action before the dispute moves further.",
        "legal_position": "In practice, it helps set out the claim clearly, preserve the sender's position, and give the other side a chance to respond before litigation or another formal step.",
        "next_steps": "If you are dealing with an actual notice, read the demand, timeline, and supporting documents carefully before replying or sending one.",
        "domain": "civil",
        "disclaimer_mode": "low_risk",
    },
}

class ChatService:
    def __init__(self, settings: Settings, store: SessionStoreProtocol) -> None:
        self.settings = settings
        self.store = store
        self.extractor = FileExtractionService(settings)
        self.intent_router = IntentRoutingService()
        self.domain_classifier = LegalDomainClassifier()
        self.domain_packs = LegalDomainPackService(settings)
        self._indiankanoon: IndianKanoonService | None = None
        self._hybrid_retrieval: LegalHybridRetrievalService | None = None
        self._google_search: GoogleCustomSearchService | None = None
        self._legal_dataset: LocalLegalDatasetService | None = None
        self._local_similarity: LocalTextSimilarityService | None = None
        self._semantic_support_checker: LocalSemanticSupportChecker | None = None
        self._openai: OpenAIResponsesService | None = None
        self._direct_answer_cache: dict[str, dict[str, Any]] = {}

    @property
    def indiankanoon(self) -> IndianKanoonService:

        if self._indiankanoon is None:
            self._indiankanoon = IndianKanoonService(self.settings)
        return self._indiankanoon

    @property
    def hybrid_retrieval(self) -> LegalHybridRetrievalService:

        if self._hybrid_retrieval is None:
            self._hybrid_retrieval = LegalHybridRetrievalService(self.settings, indiankanoon_service=self.indiankanoon)
        return self._hybrid_retrieval

    @property
    def google_search(self) -> GoogleCustomSearchService:

        if self._google_search is None:
            self._google_search = GoogleCustomSearchService(self.settings)
        return self._google_search

    @property
    def legal_dataset(self) -> LocalLegalDatasetService:

        if self._legal_dataset is None:
            self._legal_dataset = LocalLegalDatasetService()
        return self._legal_dataset

    @property
    def local_similarity(self) -> LocalTextSimilarityService:

        if self._local_similarity is None:
            self._local_similarity = LocalTextSimilarityService(self.settings)
        return self._local_similarity

    @property
    def semantic_support_checker(self) -> LocalSemanticSupportChecker:

        if self._semantic_support_checker is None:
            self._semantic_support_checker = LocalSemanticSupportChecker(
                self.settings,
                similarity_service=self.local_similarity,
            )

        return self._semantic_support_checker

    @property
    def openai(self) -> OpenAIResponsesService:

        if self._openai is None:
            self._openai = OpenAIResponsesService(self.settings)
        return self._openai

    def _build_direct_answer_cache_key(self, *, kind: str, normalized_query: str) -> str | None:
        clean_kind = str(kind or "").strip().lower()
        clean_query = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())

        if not clean_kind or not clean_query:
            return None
        return f"{clean_kind}::{clean_query}"

    def _get_direct_answer_cache_entry(self, cache_key: str | None) -> InternalChatResult | None:

        if not cache_key:
            return None
        cached = self._direct_answer_cache.get(cache_key)

        if not cached:
            return None

        if float(cached.get("expires_at") or 0.0) <= monotonic():
            self._direct_answer_cache.pop(cache_key, None)
            return None

        # Refresh recency for bounded eviction.
        self._direct_answer_cache.pop(cache_key, None)
        self._direct_answer_cache[cache_key] = cached
        return cached["internal"].model_copy(deep=True)

    def _set_direct_answer_cache_entry(self, cache_key: str | None, internal: InternalChatResult) -> None:

        if not cache_key:
            return

        cached_raw_json = dict(internal.raw_json or {})
        cached_raw_json.pop("cache_hit", None)
        cached_internal = internal.model_copy(
            deep=True,

            update={
                "warnings": [],
                "raw_json": cached_raw_json,
            },
        )

        self._direct_answer_cache[cache_key] = {
            "expires_at": monotonic() + DIRECT_ANSWER_CACHE_TTL_SECONDS,
            "internal": cached_internal,
        }

        while len(self._direct_answer_cache) > DIRECT_ANSWER_CACHE_MAX_ENTRIES:
            oldest_key = next(iter(self._direct_answer_cache))
            self._direct_answer_cache.pop(oldest_key, None)

    @staticmethod
    def _with_direct_cache_metadata(internal: InternalChatResult, *, cache_hit: bool, warnings: list[str]) -> InternalChatResult:
        raw_json = dict(internal.raw_json or {})
        raw_json["cache_hit"] = cache_hit
        return internal.model_copy(
            deep=True,

            update={
                "warnings": list(warnings),
                "raw_json": raw_json,
            },
        )

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
        conversation_state = self._merge_case_details_into_state(
            current_state=conversation_state,
            request=request,
            fallback_state=fallback_state,
            uploaded_texts=uploaded_texts or [],
        )

        retrieval_query = self._retrieval_query_for_turn(message=message, conversation_state=conversation_state)
        domain = self._classify_domain_for_turn(
            message=retrieval_query,
            conversation_state=conversation_state,
        )

        if not domain and conversation_state.legal_domain:
            domain = conversation_state.legal_domain
        resolved_state = conversation_state.case_state or self.settings.default_state
        warnings = list(extraction_warnings or [])

        try:
            internal, next_state = self._generate_chat_result(
                message=retrieval_query,
                user_message=message,
                domain=domain,
                resolved_state=resolved_state,
                warnings=warnings,
                previous_messages=previous_messages,
                conversation_state=conversation_state,
                uploaded_texts=uploaded_texts or [],
            )

        except Exception:
            logger.exception("chat response generation failed query=%r domain=%s", retrieval_query[:120], domain)
            internal = self._build_safe_fallback_result(
                kind="technical_failure",
                domain=domain,
                warnings=warnings,
                query=message,
            )

            next_state = conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "indiankanoon_rag",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "last_grounded_query": retrieval_query,
                    "legal_domain": domain,
                }
            )

        output_strategy = self._classify_legal_query(
            message=next_state.last_user_issue or message,
            domain=domain,
            conversation_state=next_state,
            uploaded_texts=uploaded_texts or [],
        )

        route_classification = self._classify_pipeline_path(
            message=message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts or [],
        )

        existing_validation_flags = [str(flag) for flag in (internal.raw_json.get("validation_flags") or []) if str(flag)]

        if self._should_use_strict_grounded_validation(strategy=output_strategy, raw_json=internal.raw_json):
            validated_answer, validation_flags = self._validate_final_output(
                answer=internal.answer,
                query=message,
                strategy=output_strategy,
                citations=internal.citations,
                authorities=internal.authorities,
            )

        else:
            validated_answer, validation_flags = internal.answer, []
        merged_validation_flags = list(dict.fromkeys([*existing_validation_flags, *validation_flags]))
        response_mode = str((internal.raw_json or {}).get("response_mode") or output_strategy.get("response_mode") or "").strip().lower()
        internal = internal.model_copy(

            update={
                "answer": validated_answer
                if response_mode == "authority"
                else self._apply_disclaimer_mode(
                    answer=validated_answer,
                    mode=str((internal.raw_json or {}).get("disclaimer_mode") or "medium_risk"),
                ),

                "citations": self._filter_criminal_reference_strings(internal.citations, strategy=output_strategy),
                "authorities": self._filter_criminal_reference_strings(internal.authorities, strategy=output_strategy),
                "raw_json": {
                    **internal.raw_json,
                    "validation_flags": merged_validation_flags,
                },
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
            "validation_flags": merged_validation_flags,
            "query_profile": (internal.raw_json or {}).get("query_profile"),
            "source_sufficiency": (internal.raw_json or {}).get("source_sufficiency"),
            "disclaimer_mode": (internal.raw_json or {}).get("disclaimer_mode"),
            "playbook_id": (internal.raw_json or {}).get("playbook_id"),
            "route_classification": route_classification,
        }

        try:
            self.store.add_message(
                chat_id,
                "user",
                message,

                metadata={
                    "domain": domain,
                    "legal_domain_label": domain,
                    "raw_user_message": message,
                    "retrieval_query": retrieval_query,
                    "conversation_state": conversation_state.model_dump(),
                    "pipeline": next_state.active_intent or "indiankanoon_rag",
                    "route_classification": route_classification,
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

    @staticmethod
    def _summarize_uploaded_texts(uploaded_texts: list[str]) -> list[str]:
        summaries: list[str] = []

        for text in uploaded_texts:
            cleaned = re.sub(r"\s+", " ", str(text or "").strip())

            if not cleaned:
                continue
            summary = cleaned[:160].strip()

            if len(cleaned) > 160:
                summary = f"{summary}..."

            if summary not in summaries:
                summaries.append(summary)
        return summaries[:5]

    def _merge_case_details_into_state(
        self,
        *,
        current_state: ConversationState,
        request: ChatRequest,
        fallback_state: str | None,
        uploaded_texts: list[str],
    ) -> ConversationState:
        request_state = (request.state or "").strip() or None
        request_district = (request.district or "").strip() or None
        request_case_stage = (request.case_stage or "").strip() or None
        case_state = request_state or current_state.case_state or fallback_state or self.settings.default_state
        uploaded_summaries = list(current_state.uploaded_document_summaries or [])

        for summary in self._summarize_uploaded_texts(uploaded_texts):

            if summary not in uploaded_summaries:
                uploaded_summaries.append(summary)

        return current_state.model_copy(

            update={
                "case_state": case_state,
                "district": request_district or current_state.district,
                "case_stage": request_case_stage or current_state.case_stage,
                "is_own_matter": request.is_own_matter if request.is_own_matter is not None else current_state.is_own_matter,
                "uploaded_document_summaries": uploaded_summaries,
            }
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
        understanding_profile = self._understand_legal_query(user_message)
        normalized_user_message = str(understanding_profile.get("normalized_query") or user_message).strip() or user_message
        query_flow = self._classify_query_flow(
            message=user_message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts,
        )

        local_match = self.legal_dataset.lookup_query(normalized_user_message)
        direct_authority_key = self._fast_authority_lookup_key(normalized_user_message)

        if query_flow["flow_type"] == "provision_lookup" and self._should_use_local_legal_dataset_fast_path(
            normalized_query=normalized_user_message,
            authority_key=direct_authority_key,
            local_match=local_match,
        ):

            logger.info(
                "chat pipeline stage=start query=%r domain=%s state=%s previous_messages=%s route_class=%s route_reason=%s",
                message[:160],
                domain,
                resolved_state,
                len(previous_messages),
                "fast",
                "local_legal_dataset_fast_path",
            )

            local_fast_path_result = self._route_fast_authority_lookup(
                message=user_message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                query_profile={
                    "flow_type": str(query_flow["flow_type"]),
                    "response_mode": self._response_mode_for_flow(str(query_flow["flow_type"])),
                },

                strategy={},
                understanding_profile=understanding_profile,
            )

            if local_fast_path_result is not None:
                internal, next_state = local_fast_path_result
                logger.info("chat pipeline stage=authority_fast_path authority_key=%s", (internal.raw_json or {}).get("authority_key"))
                return internal, next_state

        route_classification = self._classify_pipeline_path(
            message=normalized_user_message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts,
            understanding_profile=understanding_profile,
        )

        if route_classification["path"] == "fast":
            logger.info(
                "chat pipeline stage=start query=%r domain=%s state=%s previous_messages=%s route_class=%s route_reason=%s",
                message[:160],
                domain,
                resolved_state,
                len(previous_messages),
                route_classification["path"],
                route_classification["reason"],
            )

            fast_path_result = self._route_fast_authority_lookup(
                message=user_message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                query_profile={
                    "flow_type": str(query_flow["flow_type"]),
                    "response_mode": self._response_mode_for_flow(str(query_flow["flow_type"])),
                },

                strategy={},
                understanding_profile=understanding_profile,
            )

            if fast_path_result is not None:
                internal, next_state = fast_path_result
                logger.info(
                    "chat pipeline stage=authority_fast_path authority_key=%s",
                    (internal.raw_json or {}).get("authority_key"),
                )

                return internal, next_state

        strategy = self._classify_legal_query(
            message=user_message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts,
        )

        query_profile = self._build_query_profile(
            message=user_message,
            domain=domain,
            conversation_state=conversation_state,
            strategy=strategy,
            uploaded_texts=uploaded_texts,
        )

        logger.info(
            "chat pipeline stage=start query=%r domain=%s state=%s previous_messages=%s strategy=%s query_type=%s route=%s urgency=%s route_class=%s route_reason=%s",
            message[:160],
            domain,
            resolved_state,
            len(previous_messages),
            strategy["answer_mode"],
            query_profile["query_type"],
            query_profile["route_target"],
            query_profile["urgency"],
            route_classification["path"],
            route_classification["reason"],
        )

        direct_result = self._route_direct_response(
            message=user_message,
            domain=domain,
            warnings=warnings,
            conversation_state=conversation_state,
            query_profile=query_profile,
            understanding_profile=understanding_profile,
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
            query_type=str(query_profile.get("flow_type") or "general_legal_research"),
            understanding_profile=understanding_profile,
        )

        documents = self._filter_grounded_documents_for_query(
            documents=documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )

        documents = self._annotate_grounded_documents(
            documents=documents,
            query=message,
            domain=domain,
            state=resolved_state,
        )

        documents = sorted(documents, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        documents = self._select_primary_authorities(
            documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )

        documents = self._rerank_grounded_documents(
            documents=documents,
            query=message,
            domain=domain,
            state=resolved_state,
            answer_mode=strategy["answer_mode"],
            query_profile=query_profile,
        )

        uploaded_documents = self._build_uploaded_documents(uploaded_texts)
        context_documents = [*documents, *uploaded_documents]
        retrieval_confidence = self._grounded_retrieval_confidence(
            documents=documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )

        retrieval_confidence_level = self._classify_retrieval_confidence_level(
            confidence=retrieval_confidence,
            documents=documents,
            answer_mode=strategy["answer_mode"],
        )

        source_sufficiency = self._assess_source_sufficiency(
            documents=documents,
            retrieval_confidence=retrieval_confidence,
            retrieval_confidence_level=retrieval_confidence_level,
            query_profile=query_profile,
            state=resolved_state,
        )

        trusted_curated_google_authority_context = self._has_trusted_curated_google_authority_context(
            documents=documents,
            query=message,
            answer_mode=strategy["answer_mode"],
        )

        google_retrieval_context = any(
            str(doc.get("source_kind") or "").strip().lower() == "google_custom_search"
            for doc in documents
        )

        has_live_documents = any(
            str(doc.get("source_kind") or "").strip().lower() != "internal" for doc in documents
        )

        if not context_documents:
            logger.warning("chat pipeline stage=retrieval no_documents query=%r", message[:160])

            if self._should_prefer_practical_guidance(
                query_profile=query_profile,
                strategy=strategy,
                message=message,
                domain=domain,
            ):

                logger.info("chat pipeline stage=retrieval reroute_to_legal_intake query=%r", message[:160])
                return self._handle_legal_help_interview(
                    message=user_message,
                    domain=domain,
                    warnings=warnings,
                    conversation_state=conversation_state,
                    is_continuation=conversation_state.active_intent == "legal_help" and conversation_state.conversation_started,
                )

            return self._build_safe_fallback_result(
                kind="no_relevant_authority",
                domain=domain,
                warnings=warnings,
                query=user_message,
            ), conversation_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "indiankanoon_rag",
                    "legal_domain": domain,
                    "last_user_issue": user_message,
                    "last_grounded_query": message,
                }
            )

        if documents and source_sufficiency["label"] == "weak" and not trusted_curated_google_authority_context:

            logger.warning(
                "chat pipeline stage=retrieval low_confidence query=%r confidence=%.2f answer_mode=%s has_live=%s sufficiency=%s",
                message[:160],
                retrieval_confidence,
                strategy["answer_mode"],
                has_live_documents,
                source_sufficiency["label"],
            )

            if self._should_prefer_practical_guidance(
                query_profile=query_profile,
                strategy=strategy,
                message=message,
                domain=domain,
            ):

                return self._handle_legal_help_interview(
                    message=user_message,
                    domain=domain,
                    warnings=warnings,
                    conversation_state=conversation_state,
                    is_continuation=conversation_state.active_intent == "legal_help" and conversation_state.conversation_started,
                )

            low_confidence_result = self._build_safe_fallback_result(
                kind="low_confidence",
                query=user_message,
                domain=domain,
                warnings=warnings,
            )

            return low_confidence_result, ConversationState(
                conversation_started=True,
                active_intent="indiankanoon_rag",
                legal_domain=domain,
                last_user_issue=user_message,
                last_grounded_query=message,
                awaiting_details=bool(low_confidence_result.follow_up_question),
                last_follow_up_question=low_confidence_result.follow_up_question,
            )

        citations = self._build_citations(context_documents)
        authorities = self._build_authorities(context_documents)
        evidence_packet = self._build_evidence_packet(
            query=user_message,
            retrieval_query=message,
            domain=domain,
            state=resolved_state,
            documents=context_documents,
            citations=citations,
            authorities=authorities,
            query_profile=query_profile,
            source_sufficiency=source_sufficiency,
        )

        grounded_context = self._build_grounded_context(context_documents)
        grounded_context = self._trim_grounded_context(grounded_context)

        logger.info(
            "chat pipeline stage=context docs=%s uploaded_docs=%s citations=%s authorities=%s context_chars=%s disclaimer_mode=%s sufficiency=%s",
            len(documents),
            len(uploaded_documents),
            len(citations),
            authorities,
            len(grounded_context),
            evidence_packet["disclaimer_mode"],
            source_sufficiency["label"],
        )

        llm_payload = self._generate_grounded_answer(
            query=message,
            domain=domain,
            state=resolved_state,
            context=grounded_context,
            citations=citations,
            conversation=self._conversation_for_llm(previous_messages),
            documents=context_documents,
            evidence_packet=evidence_packet,
            response_mode=str(strategy.get("response_mode") or "research"),
        )

        answer = str(llm_payload.get("answer") or "").strip()

        if not answer:
            logger.warning("chat pipeline stage=llm empty_answer query=%r", message[:160])
        response_mode = str(strategy.get("response_mode") or "research").strip().lower()

        if not answer and response_mode == "authority":
            fallback = self._build_safe_fallback_result(
                kind="no_relevant_authority",
                domain=domain,
                warnings=warnings,
                query=user_message,
            )

            return fallback, ConversationState(
                conversation_started=True,
                active_intent="indiankanoon_rag",
                legal_domain=domain,
                last_user_issue=user_message,
                last_grounded_query=message,
                awaiting_details=bool(fallback.follow_up_question),
                last_follow_up_question=fallback.follow_up_question,
            )

        if not answer:
            llm_payload = self._build_structured_grounded_payload(
                query=message,
                domain=domain,
                documents=context_documents,
                citations=citations,
                response_mode=response_mode,
            )

            answer = str(llm_payload.get("answer") or "").strip()

        if response_mode != "authority":
            answer = self._normalize_final_answer(answer, citations=citations)
            answer = self._ensure_upload_context_reflected(answer, uploaded_documents)
            answer = self._ensure_scope_notes_reflected(answer, evidence_packet)
        strict_grounded_validation = self._should_use_strict_grounded_validation(
            strategy=strategy,
            raw_json={"query_profile": query_profile},
        )

        if trusted_curated_google_authority_context:
            strict_grounded_validation = False

        if self._is_deterministic_grounded_payload(llm_payload):
            strict_grounded_validation = False

        if strict_grounded_validation:
            answer, validation_flags = self._validate_final_output(
                answer=answer,
                query=user_message,
                strategy=strategy,
                citations=citations,
                authorities=authorities,
            )

            semantic_support = self._run_semantic_support_check(
                answer=answer,
                query=user_message,
                documents=context_documents,
                source_sufficiency=source_sufficiency,
            )

            if semantic_support["status"] == "unsupported" and source_sufficiency["label"] != "strong":
                validation_flags = list(dict.fromkeys([*validation_flags, "semantic_support_failed", "unsupported_output_fallback"]))
                fallback = self._build_safe_fallback_result(
                    kind="unsupported_output",
                    domain=domain,
                    warnings=warnings,
                    query=user_message,
                )

                answer = fallback.answer
                llm_payload["follow_up_question"] = fallback.follow_up_question
                llm_payload["likely_forum"] = fallback.likely_forum
                llm_payload["caution"] = fallback.caution
                llm_payload["documents_to_keep"] = fallback.documents_to_keep

        else:
            validation_flags = []
            semantic_support = {
                "status": "skipped_trusted_curated_google_authority" if trusted_curated_google_authority_context else "skipped_non_grounded_practical",
                "score": None,
                "backend": "trusted_curated_google_authority_bypass" if trusted_curated_google_authority_context else "bypass",
                "details": [],
            }

        llm_payload["answer"] = answer
        llm_payload["validation_flags"] = validation_flags
        llm_payload["semantic_support"] = semantic_support
        citations = self._filter_criminal_reference_strings(citations, strategy=strategy)
        authorities = self._filter_criminal_reference_strings(authorities, strategy=strategy)

        internal = InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=self._optional_string(llm_payload.get("follow_up_question")),
            citations=citations,
            authorities=authorities,
            documents_to_keep=self._coerce_string_list(llm_payload.get("documents_to_keep")),
            likely_forum=None

            if response_mode == "authority"

            else (self._optional_string(llm_payload.get("likely_forum")) or (authorities[0] if authorities else None)),
            caution=self._optional_string(llm_payload.get("caution")),
            warnings=warnings,

            raw_json={
                "pipeline": "indiankanoon_rag",
                "response_mode": response_mode,
                "strategy": strategy,
                "query_profile": query_profile,
                "retrieval_confidence": retrieval_confidence,
                "retrieval_confidence_level": retrieval_confidence_level,
                "source_sufficiency": source_sufficiency,
                "raw_user_message": user_message,
                "retrieval_query": message,
                "query_variants": query_variants,
                "doctypes_options": doctypes_options,
                "documents": documents,
                "uploaded_documents": uploaded_documents,
                "evidence_packet": evidence_packet,
                "llm_payload": llm_payload,
                "validation_flags": validation_flags,
                "disclaimer_mode": evidence_packet["disclaimer_mode"],
                "semantic_support": semantic_support,
                "source": str(llm_payload.get("source") or ("curated_google_authority_lookup" if google_retrieval_context else "indiankanoon_rag")),
                "retrieval_source": "curated_google_authority_lookup" if google_retrieval_context else "indiankanoon_rag",
            },
        )

        next_state = conversation_state.model_copy(
            update={
                "conversation_started": True,
                "active_intent": "indiankanoon_rag",
                "awaiting_details": bool(internal.follow_up_question),
                "interview_mode": False,
                "last_user_issue": user_message,
                "last_grounded_query": message,
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
        query_profile: dict[str, Any],
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized = str(understanding_profile.get("normalized_query") or "").strip().lower()
        answer_intent = str(understanding_profile.get("answer_intent") or "").strip().lower()
        response_mode = str(query_profile.get("response_mode") or self._response_mode_for_flow(query_profile.get("flow_type") or "")).strip().lower()

        if not normalized:
            return None

        if answer_intent == "mixed_direct":
            mixed_constitutional_result = self._route_mixed_constitutional_simple_query(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                understanding_profile=understanding_profile,
            )

            if mixed_constitutional_result is not None:
                return mixed_constitutional_result

        if answer_intent == "general_explainer":
            general_explainer_result = self._route_general_legal_explainer(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                understanding_profile=understanding_profile,
            )

            if general_explainer_result is not None:
                return general_explainer_result

        if answer_intent in {"explainer", "authority_explainer", "mixed_direct", ""}:
            constitutional_explainer_result = self._route_constitutional_explainer(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                understanding_profile=understanding_profile,
            )

            if constitutional_explainer_result is not None:
                return constitutional_explainer_result
        clarification_hint = understanding_profile.get("clarification_hint")

        if self._should_try_curated_google_before_clarification(
            normalized_query=normalized,
            domain=domain,
            clarification_hint=clarification_hint,
        ):
            return None

        if clarification_hint:
            direct_clarification_result = self._build_direct_answer_clarification_result(
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                raw_message=message,
                clarification_hint=clarification_hint,
            )

            if direct_clarification_result is not None:
                return direct_clarification_result
        routing = self._classify_routing_precedence(normalized, domain=domain)

        if (
            response_mode == "scenario"
            and conversation_state.conversation_started
            and conversation_state.active_intent == "legal_help"
        ):

            return self._handle_legal_help_interview(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                is_continuation=True,
            )

        if conversation_state.conversation_started and conversation_state.active_intent == "indiankanoon_rag":
            return None

        if routing["prefer_grounded_authority"]:
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

        if (
            response_mode == "scenario"

            and (
                routing["prefer_playbook"]
                or query_profile.get("route_target") == "playbook"
                or intent_decision.intent == "legal_help"
            )
        ):

            return self._handle_legal_help_interview(
                message=message,
                domain=domain,
                warnings=warnings,
                conversation_state=conversation_state,
                is_continuation=False,
            )

        if intent_decision.handled and intent_decision.intent in {"greeting", "thanks", "goodbye", "advice"}:

            if routing["prefer_grounded_authority"]:
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

    def _classify_pipeline_path(
        self,
        *,
        message: str,
        domain: str,
        conversation_state: ConversationState,
        uploaded_texts: list[str] | None = None,
        understanding_profile: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized = re.sub(r"\s+", " ", str(message or "").strip())
        normalized_lower = normalized.lower()
        flow = self._classify_query_flow(
            message=message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts or [],
        )

        answer_intent = str(understanding_profile.get("answer_intent") or "").strip().lower()

        if flow["flow_type"] == "scenario_practical_legal_issue":
            return {"path": "medium", "reason": "practical_guidance"}

        if flow["flow_type"] == "uploaded_document_query":
            return {"path": "heavy", "reason": "uploaded_document_grounded"}

        if flow["flow_type"] == "provision_lookup" and answer_intent == "authority_explainer":
            return {"path": "fast", "reason": "authority_fast_path"}

        if self._should_prefer_curated_google_authority_route(
            normalized_query=normalized_lower,
            understanding_profile=understanding_profile,
        ):
            return {"path": "heavy", "reason": "confidence_based_authority_retrieval"}

        if answer_intent == "mixed_direct" and understanding_profile.get("all_components_direct"):
            return {"path": "medium", "reason": "constitutional_mixed_direct"}

        if answer_intent == "explainer":
            return {"path": "medium", "reason": "constitutional_explainer"}

        if answer_intent == "general_explainer":
            return {"path": "medium", "reason": "general_legal_explainer"}

        if understanding_profile.get("clarification_hint"):

            if self._should_try_curated_google_before_clarification(
                normalized_query=normalized_lower,
                domain=domain,
                clarification_hint=understanding_profile.get("clarification_hint"),
            ):

                return {"path": "heavy", "reason": "confidence_based_authority_retrieval"}
            return {"path": "medium", "reason": "direct_answer_clarification"}

        routing = self._classify_routing_precedence(normalized_lower, domain=domain)

        if conversation_state.active_intent == "legal_help" or routing["prefer_playbook"]:
            return {"path": "medium", "reason": "practical_guidance"}

        if flow["flow_type"] in {"provision_lookup", "general_legal_research"}:
            return {"path": "heavy", "reason": "grounded_authority"}

        if routing["prefer_grounded_authority"] or self._is_statute_query(normalized_lower) or self._looks_like_grounded_authority_query(normalized_lower):
            return {"path": "heavy", "reason": "grounded_authority"}

        return {"path": "medium", "reason": "default_direct_or_playbook"}

    def _route_constitutional_explainer(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized_query = str(understanding_profile.get("normalized_query") or message)
        explainer_key = self._constitutional_explainer_key(normalized_query)

        if not explainer_key:
            return None

        cache_key = self._build_direct_answer_cache_key(
            kind="constitutional_explainer",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "constitutional_explainer",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )
            return internal, next_state
        payload = CONSTITUTIONAL_EXPLAINER_LOOKUPS.get(explainer_key)

        if not payload:
            return None

        format_profile = dict(understanding_profile.get("direct_answer_format") or {})
        preferences = dict(format_profile.get("detail_instructions") or {})
        curated_google_documents = self._lookup_curated_google_direct_documents(normalized_query)

        answer = self._format_constitutional_explainer_answer(
            title=str(payload.get("title") or "Constitutional explainer"),
            summary=str(payload.get("summary") or ""),
            legal_position=str(payload.get("legal_position") or ""),
            next_steps=str(payload.get("next_steps") or ""),
            caution=str(payload.get("caution") or ""),
            article_breakdown=[str(item).strip() for item in (payload.get("article_breakdown") or []) if str(item).strip()],
            format_profile=format_profile,
        )

        citation_values = self._preferred_direct_citations(
            google_documents=curated_google_documents,
            fallback_citations=[str(payload.get("sources") or "").strip()],
        )

        likely_forum = str(payload.get("likely_forum") or "Constitution of India") if preferences["practical_context"] else None
        documents_to_keep = (
            [str(item) for item in (payload.get("documents_to_keep") or []) if str(item).strip()]

            if preferences["practical_context"]
            else []
        )

        internal = InternalChatResult(
            answer=answer,
            domain=str(payload.get("domain") or domain or conversation_state.legal_domain or "constitutional"),
            follow_up_question=None,
            citations=citation_values,
            authorities=["Constitution of India"],
            documents_to_keep=documents_to_keep,
            likely_forum=likely_forum,
            caution=str(payload.get("caution") or "").strip() or None,
            warnings=warnings,

            raw_json={
                "source": "constitutional_explainer",
                "pipeline": "constitutional_explainer",
                "explainer_key": explainer_key,
                "curated_google_used": bool(curated_google_documents),
                "curated_google_count": len(curated_google_documents),
                "direct_answer_format": format_profile,
                "disclaimer_mode": str(payload.get("disclaimer_mode") or "medium_risk"),
            },
        )

        internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
        self._set_direct_answer_cache_entry(cache_key, internal)
        next_state = conversation_state.model_copy(

            update={
                "conversation_started": True,
                "active_intent": "constitutional_explainer",
                "awaiting_details": False,
                "last_user_issue": message,
                "legal_domain": internal.domain,
                "last_follow_up_question": None,
            }
        )
        return internal, next_state

    def _route_general_legal_explainer(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized_query = str(understanding_profile.get("normalized_query") or message)
        explainer_key = self._general_legal_explainer_key(normalized_query)

        if not explainer_key:
            return None
        cache_key = self._build_direct_answer_cache_key(
            kind="general_legal_explainer",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "general_legal_explainer",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )

            return internal, next_state
        payload = GENERAL_LEGAL_EXPLAINER_LOOKUPS.get(explainer_key)

        if not payload:
            return None

        format_profile = dict(understanding_profile.get("direct_answer_format") or {})
        answer = self._format_general_legal_explainer_answer(
            title=str(payload.get("title") or "Legal explainer"),
            summary=str(payload.get("summary") or ""),
            legal_position=str(payload.get("legal_position") or ""),
            next_steps=str(payload.get("next_steps") or ""),
            format_profile=format_profile,
        )

        internal = InternalChatResult(
            answer=answer,
            domain=str(payload.get("domain") or domain or conversation_state.legal_domain or "general"),
            follow_up_question=None,
            citations=[],
            authorities=[],
            documents_to_keep=[],
            likely_forum=None,
            caution=None,
            warnings=warnings,

            raw_json={
                "source": "general_legal_explainer",
                "pipeline": "general_legal_explainer",
                "explainer_key": explainer_key,
                "direct_answer_format": format_profile,
                "disclaimer_mode": str(payload.get("disclaimer_mode") or "low_risk"),
            },
        )

        internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
        self._set_direct_answer_cache_entry(cache_key, internal)
        next_state = conversation_state.model_copy(

            update={
                "conversation_started": True,
                "active_intent": "general_legal_explainer",
                "awaiting_details": False,
                "last_user_issue": message,
                "legal_domain": internal.domain,
                "last_follow_up_question": None,
            }
        )

        return internal, next_state

    def _route_curated_google_authority_variant(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        variant = understanding_profile.get("authority_lookup_variant")

        if not isinstance(variant, dict):
            return None
        normalized_query = str(understanding_profile.get("normalized_query") or message)
        google_query = str(variant.get("google_query") or "").strip()

        if not google_query:
            return None
        variant_domain = str(variant.get("domain") or domain or conversation_state.legal_domain or "civil")
        cache_key = self._build_direct_answer_cache_key(
            kind="curated_google_authority_variant",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "curated_google_authority_lookup",
                    "awaiting_details": bool(internal.follow_up_question),
                    "last_user_issue": message,
                    "last_grounded_query": google_query,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": internal.follow_up_question,
                }
            )
            return internal, next_state

        google_documents = self._lookup_curated_google_authority_variant_documents(
            google_query=google_query,
            authority_query=normalized_query,
        )

        if google_documents:
            citations = self._build_citations(google_documents)
            authorities = self._build_authorities(google_documents)
            source_sufficiency = {
                "label": "strong",
                "jurisdiction_note": "",
                "recency_note": "",
            }

            evidence_packet = self._build_evidence_packet(
                query=message,
                retrieval_query=google_query,
                domain=variant_domain,
                state=None,
                documents=google_documents,
                citations=citations,
                authorities=authorities,
                query_profile={
                    "query_type": "statute_lookup",
                    "route_target": "grounded",
                    "urgency": "low",
                    "issue_type": "",
                },

                source_sufficiency=source_sufficiency,
            )

            grounded_context = self._trim_grounded_context(self._build_grounded_context(google_documents))

            logger.info(
                "curated google authority route context authority_query=%r google_query=%r docs=%s citations=%s authorities=%s context_chars=%s evidence_top_sources=%s",
                normalized_query[:160],
                google_query[:160],
                len(google_documents),
                len(citations),
                authorities,
                len(grounded_context),
                evidence_packet.get("top_sources"),
            )

            payload = self._generate_grounded_answer(
                query=message,
                domain=variant_domain,
                state=None,
                context=grounded_context,
                citations=citations,
                conversation=[],
                documents=google_documents,
                evidence_packet=evidence_packet,
                response_mode="authority",
            )
            answer = str(payload.get("answer") or "").strip()
            internal = InternalChatResult(
                answer=answer,
                domain=variant_domain,
                follow_up_question=None,
                citations=citations,
                authorities=authorities,
                documents_to_keep=self._coerce_string_list(payload.get("documents_to_keep")),
                likely_forum=None,
                caution=self._optional_string(payload.get("caution")),
                warnings=warnings,

                raw_json={
                    "source": "curated_google_authority_lookup",
                    "pipeline": "curated_google_authority_lookup",
                    "google_query": google_query,
                    "variant_kind": str(variant.get("kind") or "authority_variant"),
                    "documents": google_documents,
                    "disclaimer_mode": "none",
                    "response_mode": "authority",
                },
            )

            internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
            self._set_direct_answer_cache_entry(cache_key, internal)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "curated_google_authority_lookup",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "last_grounded_query": google_query,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )
            return internal, next_state

        logger.info(
            "curated google authority route fallback authority_query=%r google_query=%r reason=no_documents_after_filter action=defer_to_grounded_retrieval",
            normalized_query[:160],
            google_query[:160],
        )
        return None

    def _route_mixed_constitutional_simple_query(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized_query = str(understanding_profile.get("normalized_query") or message)
        components = understanding_profile.get("mixed_components") or self._mixed_constitutional_components(normalized_query)

        if not components:
            return None

        curated_google_documents = self._lookup_curated_google_direct_documents(normalized_query)
        cache_key = self._build_direct_answer_cache_key(
            kind="constitutional_mixed_direct",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "constitutional_explainer",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )
            return internal, next_state

        format_profile = dict(understanding_profile.get("direct_answer_format") or {})
        preferences = dict(format_profile.get("detail_instructions") or {})
        answer_parts: list[str] = []
        citations: list[str] = []
        authorities: list[str] = []
        component_domains: list[str] = []
        authority_only = True
        constitutional_only = True
        for component in components:

            if component["kind"] == "authority":
                payload = FAST_AUTHORITY_LOOKUPS.get(component["key"])

                if not payload:
                    return None

                citations.append(str(payload.get("source") or "").strip())
                authority = str(payload.get("authority") or "").strip()

                if authority:
                    authorities.append(authority)

                    if authority != "Constitution of India":
                        constitutional_only = False
                component_domain = str(payload.get("domain") or "").strip()

                if component_domain:
                    component_domains.append(component_domain)
                answer_parts.append(

                    self._format_constitutional_authority_explainer_answer(
                        title=str(payload.get("title") or ""),
                        summary=str(payload.get("summary") or ""),
                        legal_position=str(payload.get("legal_position") or ""),
                        format_profile=format_profile,
                    )
                )

                continue
            payload = CONSTITUTIONAL_EXPLAINER_LOOKUPS.get(component["key"])

            if not payload:
                return None
            authority_only = False
            citations.append(str(payload.get("sources") or "").strip())
            authorities.append("Constitution of India")
            component_domain = str(payload.get("domain") or "").strip()

            if component_domain:
                component_domains.append(component_domain)

            answer_parts.append(
                self._format_constitutional_explainer_answer(
                    title=str(payload.get("title") or "Constitutional explainer"),
                    summary=str(payload.get("summary") or ""),
                    legal_position=str(payload.get("legal_position") or ""),
                    next_steps=str(payload.get("next_steps") or ""),
                    caution=str(payload.get("caution") or ""),
                    article_breakdown=[str(item).strip() for item in (payload.get("article_breakdown") or []) if str(item).strip()],
                    format_profile=format_profile,
                )
            )

        unique_authorities = [item for item in dict.fromkeys(authorities) if item]
        derived_domain = component_domains[0] if component_domains else str(domain or conversation_state.legal_domain or "constitutional")
        retrieval_source = "constitutional_mixed_direct"

        if authority_only and not constitutional_only:
            retrieval_source = "authority_mixed_direct"

        internal = InternalChatResult(
            answer="\n\n".join(part for part in answer_parts if part.strip()),
            domain=derived_domain,
            follow_up_question=None,
            citations=self._preferred_direct_citations(
                google_documents=curated_google_documents,
                fallback_citations=[citation for citation in dict.fromkeys(citations) if citation],
            ),

            authorities=unique_authorities,
            documents_to_keep=["relevant order or notice", "timeline", "supporting records"] if preferences["practical_context"] else [],
            likely_forum=unique_authorities[0] if preferences["practical_context"] and unique_authorities else None,
            caution=None,
            warnings=warnings,

            raw_json={
                "source": retrieval_source,
                "pipeline": "constitutional_explainer" if retrieval_source == "constitutional_mixed_direct" else "authority_fast_path",
                "component_keys": [component["key"] for component in components],
                "curated_google_used": bool(curated_google_documents),
                "curated_google_count": len(curated_google_documents),
                "direct_answer_format": format_profile,
                "disclaimer_mode": "medium_risk",
            },
        )

        internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
        self._set_direct_answer_cache_entry(cache_key, internal)
        next_state = conversation_state.model_copy(

            update={
                "conversation_started": True,
                "active_intent": "constitutional_explainer",
                "awaiting_details": False,
                "last_user_issue": message,
                "legal_domain": internal.domain,
                "last_follow_up_question": None,
            }
        )
        return internal, next_state

    @staticmethod
    def _format_constitutional_explainer_answer(
        *,
        title: str,
        summary: str,
        legal_position: str,
        next_steps: str,
        caution: str,
        article_breakdown: list[str],
        format_profile: dict[str, Any] | None = None,
    ) -> str:
        format_config = format_profile or {}
        layout = str(format_config.get("layout") or "conversational").strip().lower()
        preferences = {
            "include_articles": False,
            "concise": False,
            "step_by_step": False,
            "in_points": False,
            "practical_context": False,
            **(format_config.get("detail_instructions") or {}),
        }

        lead = ChatService._format_conversational_lead(title=title, summary=summary, conversational=layout != "definition")

        if preferences["include_articles"] and article_breakdown:
            formatter = ChatService._format_numbered_lines if preferences["step_by_step"] else ChatService._format_bulleted_lines
            article_lines = formatter(article_breakdown)
            practical_line = f"If you later need the practical side, {next_steps.strip()}" if next_steps.strip() else ""
            return "\n\n".join(

                part

                for part in [
                    lead,
                    ("Here is the article-wise breakdown:\n" if not preferences["step_by_step"] else "Here it is step by step with the article-wise breakdown:\n") + article_lines,
                    ChatService._format_human_legal_position(legal_position),
                    practical_line,
                ]

                if part
            )

        if preferences["in_points"]:
            point_lines = ChatService._build_structured_explainer_point_lines(
                title=title,
                summary=summary,
                legal_position=legal_position,
                next_steps=next_steps,
                article_breakdown=article_breakdown,
                practical_context=preferences["practical_context"],
            )
          
            return "\n\n".join(
                part for part in [lead, "Here are the key points:\n" + ChatService._format_bulleted_lines(point_lines)] if part
            )
        
        if preferences["step_by_step"] and article_breakdown:
            return "\n\n".join(
                part
        
                for part in [
                    lead,
                    "Here it is step by step:\n" + ChatService._format_numbered_lines(article_breakdown),
                    ChatService._format_human_legal_position(legal_position),
                ]
        
                if part
            )
        
        if preferences["concise"]:
            parts = [
                lead,
                ChatService._format_human_legal_position(legal_position),
            ]
        
            return "\n\n".join(part for part in parts if part)
        
        explanation = ChatService._format_human_legal_position(legal_position)
        closing_parts = [next_steps.strip(), caution.strip()]
        closing = " ".join(part for part in closing_parts if part)
        parts = [part for part in [lead, explanation, closing] if part]
        return "\n\n".join(parts)

    @staticmethod
    def _format_conversational_lead(*, title: str, summary: str, conversational: bool) -> str:
        clean_title = title.strip()
        clean_summary = ChatService._polish_direct_answer_text(summary)
        
        if not clean_title:
            return clean_summary
        
        if not clean_summary:
            return clean_title
        
        if not conversational:
            return ChatService._format_sentence_style_authority_lead(title=clean_title, summary=clean_summary)
        
        summary_lower = clean_summary.lower()
        title_lower = clean_title.lower()
        repeated_subject_patterns = {
            "fundamental duties": ("fundamental duties are ", "fundamental duties "),
            "fundamental rights": ("fundamental rights are ", "fundamental rights "),
            "directive principles of state policy": (
                "directive principles of state policy are ",
                "directive principles of state policy ",
            ),
        }
        
        for subject, prefixes in repeated_subject_patterns.items():
            if subject in title_lower:
        
                for prefix in prefixes:
                    if summary_lower.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):].strip()
                        summary_lower = clean_summary.lower()
                        break
        
        if clean_summary[:1].isupper():
            clean_summary = clean_summary[:1].lower() + clean_summary[1:]
        plural_markers = ("fundamental duties", "fundamental rights", "directive principles")
        verb = "are" if any(marker in title_lower for marker in plural_markers) else "is"
        
        return f"{clean_title} {verb} {clean_summary}"

    @staticmethod
    def _format_sentence_style_authority_lead(*, title: str, summary: str) -> str:
        clean_title = title.strip()
        clean_summary = ChatService._polish_direct_answer_text(summary)
        
        if not clean_title:
            return clean_summary
        
        if not clean_summary:
            return clean_title
        
        summary_lower = clean_summary.lower()
        duplicate_prefix_patterns = [
            r"^article\s+\d+[a-z]?\s+",
            r"^section\s+\d+[a-z]?\s+(?:ipc|bns|bnss|ni act)?\s*",
        ]
        
        for pattern in duplicate_prefix_patterns:
        
            if re.match(pattern, summary_lower):
                clean_summary = re.sub(pattern, "", clean_summary, flags=re.IGNORECASE).strip()
                summary_lower = clean_summary.lower()
                break
        
        if clean_summary[:1].isupper():
            clean_summary = clean_summary[:1].lower() + clean_summary[1:]
        
        return f"{clean_title} {clean_summary}".strip()

    @staticmethod
    def _format_human_legal_position(text: str, *, prefix: str = "In practice, ") -> str:
        clean_text = ChatService._polish_direct_answer_text(text)
        
        if not clean_text:
            return ""
        lower_text = clean_text.lower()
        
        if lower_text.startswith(("in practice,", "practically speaking,", "put simply,", "this means")):
            return clean_text
        
        if lower_text.startswith(("courts ", "the courts ", "they ", "it ", "this ", "these ")):
        
            if clean_text[:1].isupper():
                clean_text = clean_text[:1].lower() + clean_text[1:]
            return f"{prefix}{clean_text}".strip()
        
        return clean_text

    @staticmethod
    def _format_general_legal_explainer_answer(
        *,
        title: str,
        summary: str,
        legal_position: str,
        next_steps: str,
        format_profile: dict[str, Any] | None = None,
    ) -> str:
        format_config = format_profile or {}
        layout = str(format_config.get("layout") or "conversational").strip().lower()
        preferences = {
            "concise": False,
            "step_by_step": False,
            "in_points": False,
            "practical_context": False,
            **(format_config.get("detail_instructions") or {}),
        }
        
        lead = ChatService._format_conversational_lead(title=title, summary=summary, conversational=layout != "definition")
        explanation = ChatService._format_human_legal_position(legal_position)
        clean_next_steps = next_steps.strip()
        
        if layout == "points" or preferences["in_points"]:
            point_lines = ChatService._build_structured_general_explainer_point_lines(
                summary=summary,
                legal_position=legal_position,
                next_steps=next_steps,
                practical_context=preferences["practical_context"],
            )
        
            return "\n\n".join(
                part for part in [title.strip(), "Here are the key points:\n" + ChatService._format_bulleted_lines(point_lines)] if part
            )
        
        if layout == "step_by_step" or preferences["step_by_step"]:
            numbered_lines = [line for line in [ChatService._polish_direct_answer_text(summary), explanation] if line]
        
            if preferences["practical_context"] and clean_next_steps:
                numbered_lines.append(clean_next_steps)
        
            return "\n\n".join(
                part for part in [title.strip(), "Here it is step by step:\n" + ChatService._format_numbered_lines(numbered_lines)] if part
            )
        
        if layout == "concise" or preferences["concise"]:
            return lead
        
        closing = clean_next_steps if preferences["practical_context"] and clean_next_steps else ""
        
        return "\n\n".join(part for part in [lead, explanation, closing] if part)

    @staticmethod
    def _looks_like_ambiguous_article_prompt(normalized_query: str) -> bool:
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return False
        
        if re.search(r"\b(?:article|art)\s+\d+[a-z]?\b", compact):
            return False
        
        ambiguous_patterns = (
            r"^(?:article|art)$",
            r"^(?:article|art)\s+in\s+(?:the\s+)?constitution$",
            r"^(?:constitution|constitutional)\s+article$",
            r"^(?:which|what)\s+(?:article|art)$",
            r"^(?:which|what)\s+(?:constitution|constitutional)\s+article$",
            r"^(?:tell me about|explain)\s+(?:article|art)$",
        )
        
        return any(re.fullmatch(pattern, compact) for pattern in ambiguous_patterns)

    @staticmethod
    def _looks_like_ambiguous_section_prompt(normalized_query: str) -> bool:
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return False
        
        if re.search(r"\bsection\s+\d+[a-z]?\b", compact):
            return False
        
        ambiguous_patterns = (
            r"^section$",
            r"^section\s+in\s+(?:the\s+)?(?:act|code|ipc|bns|bnss|constitution)$",
            r"^(?:which|what)\s+section$",
            r"^(?:tell me about|explain)\s+section$",
        )
        
        return any(re.fullmatch(pattern, compact) for pattern in ambiguous_patterns)

    @staticmethod
    def _looks_like_complete_authority_reference_prompt(normalized_query: str) -> bool:
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return False
        
        if re.search(r"\b(?:article|art)\s+\d+[a-z]?\b", compact):
            return True
        section_match = re.search(r"\bsection\s+\d+[a-z]?\b", compact)
        
        if not section_match:
            return False
        
        statute_markers = {
            "bns",
            "bnss",
            "ipc",
            "indian penal code",
            "bharatiya nyaya sanhita",
            "bharatiya nagarik suraksha sanhita",
            "negotiable instruments act",
            "ni act",
            "act",
            "code",
            "constitution",
        }
        
        return any(marker in compact for marker in statute_markers)

    @staticmethod
    def _polish_direct_answer_text(text: str) -> str:
        clean_text = str(text or "").strip()
        
        if not clean_text:
            return ""
        clean_text = re.sub(r"\bIn simple terms,\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\bIn broad terms,\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\bIn general terms,\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\bBroadly speaking,\s*", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    def _lookup_curated_google_direct_documents(self, normalized_query: str) -> list[dict[str, Any]]:
        query = re.sub(r"\s+", " ", str(normalized_query or "").strip())
        
        if not query or not self.google_search.configured:
            return []
        
        try:
            result = self.google_search.search(
                query=query,
                max_results=min(self.settings.google_search_max_results, 2),
                site_restrict=LegalHybridRetrievalService._google_site_restrict(query),
                trusted_only=True,
            )
        
        except Exception:
            logger.exception("curated google direct lookup failed query=%r", query[:160])
            return []
        
        if not LegalHybridRetrievalService._curated_google_documents_strong(result.documents):
            return []
        
        return list(result.documents)

    def _lookup_curated_google_authority_variant_documents(
        self,
        *,
        google_query: str,
        authority_query: str,
    ) -> list[dict[str, Any]]:
        query = re.sub(r"\s+", " ", str(google_query or "").strip())
        
        if not query or not self.google_search.configured:
            return []
        logger.info(
            "curated google authority lookup rewrite authority_query=%r google_query=%r",
            authority_query[:160],
            query[:160],
        )
        
        try:
            result = self.google_search.search(
                query=query,
                max_results=min(self.settings.google_search_max_results, 2),
                site_restrict=LegalHybridRetrievalService._google_site_restrict(query),
                trusted_only=True,
            )
        
        except Exception:
            logger.exception(
                "curated google authority lookup failed authority_query=%r google_query=%r",
                authority_query[:160],
                query[:160],
            )
            return []
        
        documents = list(result.documents)
        logger.info(
            "curated google authority lookup post_search authority_query=%r google_query=%r trusted_result_count=%s documents=%s",
            authority_query[:160],
            query[:160],
            int(result.trusted_result_count),
            [
        
                {
                    "title": str(doc.get("title") or "")[:140],
                    "docsource": str(doc.get("docsource") or ""),
                    "authority_type": str(doc.get("authority_type") or ""),
                    "score": float(doc.get("score") or 0.0),
                }
        
                for doc in documents[:3]
            ],
        )
        
        if not documents:
            logger.info(
                "curated google authority lookup outcome authority_query=%r google_query=%r decision=no_documents",
                authority_query[:160],
                query[:160],
            )
        
            return []
        
        if LegalHybridRetrievalService._curated_google_documents_strong(documents):
            logger.info(
                "curated google authority lookup outcome authority_query=%r google_query=%r decision=strong_documents",
                authority_query[:160],
                query[:160],
            )
            return documents
        
        trusted_documents = [
            doc
            for doc in documents
            if self._document_matches_trusted_google_domain(document=doc)
        ]
        
        authoritative_documents = [
            doc
            for doc in documents
            if self._curated_google_authority_variant_document_authoritative(document=doc)
        ]
        
        relevant_documents = [
            doc
            for doc in documents
        
            if self._curated_google_authority_variant_document_relevant(
                authority_query=authority_query,
                document=doc,
            )
        ]
        
        filtered_document_logs = [
            {
                "title": str(doc.get("title") or "")[:140],
                "domain": self._google_document_domain(document=doc),
                "rejected_from_authoritative": self._curated_google_authority_variant_authoritative_reject_reasons(document=doc),
                "rejected_from_relevant": self._curated_google_authority_variant_relevance_reject_reasons(
                    authority_query=authority_query,
                    document=doc,
                ),
            }
        
            for doc in documents[:3]
        ]
        
        logger.info(
            "curated google authority lookup filtering authority_query=%r trusted_hits=%s authoritative_hits=%s relevant_hits=%s authoritative_titles=%s relevant_titles=%s filtered=%s",
            authority_query[:160],
            len(trusted_documents),
            len(authoritative_documents),
            len(relevant_documents),
            [str(doc.get('title') or '')[:140] for doc in authoritative_documents[:3]],
            [str(doc.get('title') or '')[:140] for doc in relevant_documents[:3]],
            filtered_document_logs,
        )
        
        if relevant_documents:
            return relevant_documents[:2]
        return authoritative_documents[:2]

    def _curated_google_authority_variant_document_relevant(
        self,
        *,
        authority_query: str,
        document: dict[str, Any],
    ) -> bool:
        
        if self._curated_google_authority_variant_authoritative_reject_reasons(document=document):
            return False
        
        return not self._curated_google_authority_variant_relevance_reject_reasons(
            authority_query=authority_query,
            document=document,
        )

    def _curated_google_authority_variant_document_authoritative(self, *, document: dict[str, Any]) -> bool:
        
        return not self._curated_google_authority_variant_authoritative_reject_reasons(document=document)

    def _curated_google_authority_variant_authoritative_reject_reasons(
        self,
        *,
        document: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        source_kind = str(document.get("source_kind") or "").strip().lower()
        authority_type = str(document.get("authority_type") or "").strip().lower()
        
        if source_kind != "google_custom_search":
            reasons.append("not_google_custom_search")
        
        if authority_type not in {"government_portal", "regulator", "court_portal"} and not self._document_matches_trusted_google_domain(document=document):
            reasons.append("not_authoritative_or_trusted_domain")
        
        return reasons

    def _curated_google_authority_variant_relevance_reject_reasons(
        self,
        *,
        authority_query: str,
        document: dict[str, Any],
    ) -> list[str]:
        reasons = self._curated_google_authority_variant_authoritative_reject_reasons(document=document)
        
        if reasons:
            return reasons
        
        if self._looks_like_exact_authority_match_for_response(query=authority_query, document=document):
            return []
        
        if self._looks_like_relaxed_authority_match_for_response(query=authority_query, document=document):
            return []
        
        return ["no_exact_or_relaxed_authority_match"]

    def _document_matches_trusted_google_domain(self, *, document: dict[str, Any]) -> bool:
        explicit_flag = document.get("trusted_domain_match")
        
        if isinstance(explicit_flag, bool):
            return explicit_flag
        domain = self._google_document_domain(document=document)
        
        if not domain:
            return False
        trusted_domains = self.settings.google_search_trusted_domains
        return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in trusted_domains)

    @staticmethod
    def _google_document_domain(*, document: dict[str, Any]) -> str:
        source_domain = str(document.get("source_domain") or "").strip().lower()
        
        if source_domain:
            return source_domain
        docsource = str(document.get("docsource") or "").strip().lower()
        
        if docsource.startswith("google:"):
            candidate = docsource.split(":", 1)[1].strip()
        
            if candidate:
                return candidate
        
        url = str(document.get("url") or "").strip()
        host = urlparse(url).netloc.lower().strip()
        
        if host.startswith("www."):
            host = host[4:]
        return host

    def _looks_like_relaxed_authority_match_for_response(self, *, query: str, document: dict[str, Any]) -> bool:
        normalized_query = str(query or "").lower()
        text = " ".join(
        
            [
                str(document.get("title") or ""),
                str(document.get("headline") or ""),
                str(document.get("fragment_headline") or ""),
                str(document.get("fragment_excerpt") or ""),
                str(document.get("doc_excerpt") or ""),
            ]
        ).lower()
        
        article_match = re.search(r"\barticle\s+([0-9]+[a-z]?)\b", normalized_query, re.IGNORECASE)
        
        if article_match and "constitution" in normalized_query:
            article_value = article_match.group(1).lower()
            return "constitution" in text and f"article {article_value}" in text
        
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", normalized_query, re.IGNORECASE)
        
        if not section_match:
            return False
        
        section_value = section_match.group(1).lower()
        
        if "bns" in normalized_query or "bharatiya nyaya sanhita" in normalized_query:
            return "bharatiya nyaya sanhita" in text and f"section {section_value}" in text
        
        if "bnss" in normalized_query or "bharatiya nagarik suraksha sanhita" in normalized_query:
            return "bharatiya nagarik suraksha sanhita" in text and f"section {section_value}" in text
        
        if "ipc" in normalized_query or "indian penal code" in normalized_query:
            return "indian penal code" in text and f"section {section_value}" in text
        
        return f"section {section_value}" in text

    def _should_prefer_curated_google_authority_route(
        self,
        *,
        normalized_query: str,
        understanding_profile: dict[str, Any] | None,
    ) -> bool:
        
        if not self.google_search.configured:
            return False
        profile = understanding_profile or {}
        
        if isinstance(profile.get("authority_lookup_variant"), dict):
            return True
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return False
        return self._looks_like_complete_authority_reference_prompt(compact)

    @staticmethod
    def _preferred_direct_citations(*, google_documents: list[dict[str, Any]], fallback_citations: list[str]) -> list[str]:
        
        google_citations = [
            str(doc.get("title") or doc.get("source_domain") or "").strip()
            for doc in google_documents
            if str(doc.get("title") or doc.get("source_domain") or "").strip()
        ]
        
        if google_citations:
            return [citation for citation in dict.fromkeys(google_citations) if citation][:3]
        return [citation for citation in dict.fromkeys(fallback_citations) if citation]

    def _should_try_curated_google_before_clarification(
        self,
        *,
        normalized_query: str,
        domain: str,
        clarification_hint: dict[str, str] | Any,
    ) -> bool:
        
        if not isinstance(clarification_hint, dict):
            return False
        
        if not self.google_search.configured:
            return False
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return False
        kind = str(clarification_hint.get("kind") or "").strip().lower()
        
        if kind == "authority_article":
            return bool(re.search(r"\b(?:article|art)\s+\d+[a-z]?\b", compact))
        
        if kind == "authority_section":
            has_section_identifier = bool(re.search(r"\bsection\s+\d+[a-z]?\b", compact))
            has_statute_hint = any(
                token in compact
                for token in {" act", " ipc", " bns", " bnss", " ni act", " constitution", " code"}
            )
            return has_section_identifier and has_statute_hint
        
        if kind == "constitutional_topic":
            return False
        return False

    @staticmethod
    def _understand_legal_query(message: str) -> dict[str, Any]:
        raw_query = str(message or "").strip()
        normalized_query, corrected_terms = ChatService._normalize_legal_query_text(raw_query)
        detail_instructions = ChatService._extract_query_detail_instructions(normalized_query)
        mixed_components = ChatService._decompose_direct_answer_components(normalized_query)
        authority_lookup_variant = ChatService._detect_curated_google_authority_target(
            raw_query=raw_query,
            normalized_query=normalized_query,
            corrected_terms=corrected_terms,
        )
        
        answer_intent = ChatService._classify_direct_answer_intent(
            normalized_query=normalized_query,
            mixed_components=mixed_components,
        )
        
        direct_answer_format = ChatService._build_direct_answer_format_profile(
            answer_intent=answer_intent,
            detail_instructions=detail_instructions,
            mixed_components=mixed_components,
        )
        
        clarification_hint = ChatService._detect_direct_answer_clarification_hint(
            normalized_query=normalized_query,
            answer_intent=answer_intent,
        )
        
        return {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "corrected_terms": corrected_terms,
            "detail_instructions": detail_instructions,
            "answer_intent": answer_intent,
            "mixed_components": mixed_components,
            "all_components_direct": bool(mixed_components),
            "authority_lookup_variant": authority_lookup_variant,
            "direct_answer_format": direct_answer_format,
            "clarification_hint": clarification_hint,
        }

    @staticmethod
    def _classify_direct_answer_intent(
        *,
        normalized_query: str,
        mixed_components: list[dict[str, str]] | None = None,
    ) -> str:
        components = mixed_components or []
        
        if components:
            return "mixed_direct"
        
        if ChatService._fast_authority_lookup_key(normalized_query):
            return "authority_explainer"
        
        if ChatService._looks_like_complete_authority_reference_prompt(normalized_query):
            return "authority_explainer"
        
        if ChatService._constitutional_explainer_key(normalized_query):
            return "explainer"
        
        if ChatService._general_legal_explainer_key(normalized_query):
            return "general_explainer"
        
        return "other"

    @staticmethod
    def _decompose_direct_answer_components(normalized_query: str) -> list[dict[str, str]]:
        components: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        
        if not normalized_query:
            return components

        for key in ChatService._extract_grouped_authority_keys(normalized_query):
            marker = ("authority", key)
        
            if marker not in seen:
                seen.add(marker)
                components.append({"kind": "authority", "key": key})

        for explainer_key, phrases in {
            "fundamental_duties": {"fundamental duties", "what are duties in constitution", "duties under constitution"},
            "directive_principles": {"directive principles", "directive principles of state policy", "dpsp"},
            "fundamental_rights": {"fundamental rights", "basic rights in constitution", "rights under constitution"},
        }.items():
        
            if any(phrase in normalized_query for phrase in phrases):
                marker = ("explainer", explainer_key)
        
                if marker not in seen:
                    seen.add(marker)
                    components.append({"kind": "explainer", "key": explainer_key})
        
        return components if len(components) >= 2 else []

    @staticmethod
    def _build_direct_answer_format_profile(
        *,
        answer_intent: str,
        detail_instructions: dict[str, bool] | None = None,
        mixed_components: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        
        preferences = {
            "include_articles": False,
            "concise": False,
            "step_by_step": False,
            "in_points": False,
            "practical_context": False,
            **(detail_instructions or {}),
        }
        
        explainer_layout = "conversational"
        
        if preferences["include_articles"]:
            explainer_layout = "article_breakdown"
        
        elif preferences["in_points"]:
            explainer_layout = "points"
        
        elif preferences["step_by_step"]:
            explainer_layout = "step_by_step"
        
        elif preferences["concise"]:
            explainer_layout = "concise"
        authority_layout = "definition"
        
        if preferences["in_points"]:
            authority_layout = "points"
        
        elif preferences["step_by_step"]:
            authority_layout = "step_by_step"

        component_kinds = [str(component.get("kind") or "").strip().lower() for component in (mixed_components or [])]
        
        if answer_intent == "authority_explainer":
        
            return {
                "family": "authority",
                "layout": authority_layout,
                "detail_instructions": preferences,
                "component_kinds": [],
            }
        
        if answer_intent == "mixed_direct":
        
            return {
                "family": "mixed",
                "layout": "mixed_direct",
                "detail_instructions": preferences,
                "component_kinds": component_kinds,
                "authority_layout": authority_layout,
                "explainer_layout": explainer_layout,
            }
        
        if answer_intent == "explainer":
        
            return {
                "family": "explainer",
                "layout": explainer_layout,
                "detail_instructions": preferences,
                "component_kinds": [],
            }
        
        if answer_intent == "general_explainer":
            general_layout = "conversational"
        
            if preferences["in_points"]:
                general_layout = "points"
        
            elif preferences["step_by_step"]:
                general_layout = "step_by_step"
        
            elif preferences["concise"]:
                general_layout = "concise"
        
            return {
                "family": "general_explainer",
                "layout": general_layout,
                "detail_instructions": preferences,
                "component_kinds": [],
            }
        
        return {
            "family": "other",
            "layout": "none",
            "detail_instructions": preferences,
            "component_kinds": component_kinds,
        }

    @staticmethod
    def _detect_direct_answer_clarification_hint(
        
        *,
        normalized_query: str,
        answer_intent: str,
    ) -> dict[str, str] | None:
        
        if answer_intent != "other":
            return None
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        
        if not compact:
            return None

        heavy_authority_markers = {
            "judgment",
            "judgement",
            "case law",
            "citation",
            "precedent",
            "latest",
            "recent",
            "current",
            "punishment",
            "penalty",
            "sentence",
            "legal position",
            "interpretation",
            "ruling",
        }
        
        if any(marker in compact for marker in heavy_authority_markers):
            return None

        if ChatService._resolve_direct_constitutional_authority_alias(compact):
            return None
        
        if ChatService._looks_like_complete_authority_reference_prompt(compact):
            return None

        if ChatService._looks_like_ambiguous_article_prompt(compact):
        
            return {
                "kind": "authority_article",
                "question": "Which Constitution article do you want explained?",
                "answer": "I can help with that, but I need one small clarification first.\nWhich Constitution article do you want explained?",
            }

        if ChatService._looks_like_ambiguous_section_prompt(compact):
            return {
                "kind": "authority_section",
                "question": "Which exact section and statute do you want explained?",
                "answer": "I can help with that, but I need one small clarification first.\nWhich exact section and statute do you want explained?",
            }

        constitutional_topic_markers = {
            "constitutional rights",
            "constitution rights",
            "rights in constitution",
            "duties in constitution",
            "constitutional duties",
            "constitution duties",
            "directive principles",
            "constitutional principles",
        }
        
        if any(marker in compact for marker in constitutional_topic_markers) or (
            "constitution" in compact and any(token in compact for token in {"rights", "duties", "principles"})
        ):
        
            return {
                "kind": "constitutional_topic",
                "question": "Do you want Fundamental Rights, Fundamental Duties, or DPSP?",
                "answer": "I can help with that. I just need one quick clarification first.\nDo you want Fundamental Rights, Fundamental Duties, or DPSP?",
            }

        return None

    @staticmethod

    def _normalize_legal_query_text(message: str) -> tuple[str, list[str]]:
        normalized = re.sub(r"[+/]", " and ", str(message or ""))
        normalized = re.sub(r"[^a-zA-Z0-9() \-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()

        if not normalized:
            return "", []

        corrections = {
            "fundamantal": "fundamental",
            "fundemental": "fundamental",
            "fndamentl": "fundamental",
            "consitution": "constitution",
            "constution": "constitution",
            "konstitusion":"constitution",
            "artcle": "article",
            "articel": "article",
            "artikal":"article",
            "rite": "right",
            "rites": "rights",
            "raits":"rights",
            "diretive": "directive",
            "directve": "directive",
            "principels": "principles",
            "priciples": "principles",
            "art ": "article "}

        corrected_terms: list[str] = []
        normalized, alias_corrections = ChatService._normalize_authority_alias_phrases(normalized)
        corrected_terms.extend(alias_corrections)
        for wrong, right in corrections.items():

            if wrong in normalized:
                normalized = normalized.replace(wrong, right)
                corrected_terms.append(f"{wrong}->{right}")

        normalized, authority_corrections = ChatService._normalize_authority_lookup_tokens(normalized)
        corrected_terms.extend(authority_corrections)
        normalized, direct_answer_corrections = ChatService._normalize_direct_answer_legal_terms(normalized)
        corrected_terms.extend(direct_answer_corrections)
        normalized, order_corrections = ChatService._normalize_authority_word_order(normalized)
        corrected_terms.extend(order_corrections)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized, corrected_terms

    @staticmethod

    def _normalize_authority_alias_phrases(message: str) -> tuple[str, list[str]]:
        normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())

        if not normalized:
            return "", []

        replacements = [
            (r"\bbandharan\b", "constitution", "bandharan->constitution"),
            (r"\bbandhran\b", "constitution", "bandhran->constitution"),
            (r"\bsamvidhan\b", "constitution", "samvidhan->constitution"),
            (r"\bkalam\b", "article", "kalam->article"),
            (r"\bartical\b", "article", "artical->article"),
            (r"\barticale\b", "article", "articale->article"),
            (r"\barticl\b", "article", "articl->article"),
            (r"\barticel\b", "article", "articel->article"),
            (r"\batricle\b", "article", "atricle->article"),
            (r"\bartikal\b", "article", "artikal->article"),
            (r"\bdhara\b", "section", "dhara->section"),
            (r"\bsekshan\b", "section", "sekshan->section"),
            (r"\bsektion\b", "section", "sektion->section"),
            (r"\bsectin\b", "section", "sectin->section"),
            (r"\bsectoin\b", "section", "sectoin->section"),
            (r"\bsecn\b", "section", "secn->section"),
            (r"\bsec\b", "section", "sec->section"),
            (r"\bipcc\b", "ipc", "ipcc->ipc"),
            (r"\bbharatiya nyaya sanhita\b", "bns", "bharatiya nyaya sanhita->bns"),
            (r"\bbharatiya nyaaya sanhita\b", "bns", "bharatiya nyaaya sanhita->bns"),
            (r"\bbharatiya nagarik suraksha sanhita\b", "bnss", "bharatiya nagarik suraksha sanhita->bnss"),
            (r"\bbharatiya nagrik suraksha sanhita\b", "bnss", "bharatiya nagrik suraksha sanhita->bnss"),
        ]

        corrected_terms: list[str] = []

        for pattern, replacement, label in replacements:
            updated = re.sub(pattern, replacement, normalized)

            if updated != normalized:
                normalized = updated
                corrected_terms.append(label)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized, corrected_terms

    @staticmethod
    def _normalize_authority_word_order(message: str) -> tuple[str, list[str]]:
        normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
        if not normalized:
            return "", []

        corrected_terms: list[str] = []
        patterns = [
            (
                r"\b([0-9]+[a-z]?)\s+(ipc|indian penal code|bns|bnss|ni act|negotiable instruments act)\s+section\b",
                r"section \1 \2",
            ),
            (
                r"\b([0-9]+[a-z]?)\s+(constitution)\s+article\b",
                r"article \1 \2",
            ),
        ]

        for pattern, replacement in patterns:
            updated = re.sub(pattern, replacement, normalized)
            if updated != normalized:
                corrected_terms.append(f"{normalized}->{updated}")
                normalized = updated
        return normalized, corrected_terms

    @staticmethod

    def _normalize_authority_lookup_tokens(message: str) -> tuple[str, list[str]]:
        tokens = str(message or "").split()

        if not tokens:
            return "", []

        controlled_vocabulary = (
            "article",
            "art",
            "section",
            "constitution",
            "bns",
            "bnss",
            "ipc",
            "code",
            "notice",
            "legal",
            "bail",
            "fir",
            "arbitration",
        )

        corrected_terms: list[str] = []
        normalized_tokens: list[str] = []

        for token in tokens:

            if len(token) < 4 or token.isdigit() or token in controlled_vocabulary:
                normalized_tokens.append(token)
                continue

            matches = difflib.get_close_matches(token, controlled_vocabulary, n=1, cutoff=0.6)
            replacement = matches[0] if matches else token

            if replacement != token:
                corrected_terms.append(f"{token}->{replacement}")
            normalized_tokens.append(replacement)

        return " ".join(normalized_tokens), corrected_terms

    @staticmethod
    def _normalize_direct_answer_legal_terms(message: str) -> tuple[str, list[str]]:
        tokens = str(message or "").split()

        if not tokens:
            return "", []

        controlled_vocabulary = (
            "article",
            "section",
            "constitution",
            "constitutional",
            "fundamental",
            "rights",
            "right",
            "duties",
            "duty",
            "directive",
            "principles",
            "policy",
            "state",
            "dpsp",
            "ipc",
            "code",
        )

        corrected_terms: list[str] = []
        normalized_tokens: list[str] = []

        for token in tokens:

            if len(token) < 4 or not token.isalpha() or token in controlled_vocabulary:
                normalized_tokens.append(token)
                continue

            matches = difflib.get_close_matches(token, controlled_vocabulary, n=1, cutoff=0.78)
            replacement = matches[0] if matches else token

            if replacement != token:
                corrected_terms.append(f"{token}->{replacement}")
            normalized_tokens.append(replacement)

        return " ".join(normalized_tokens), corrected_terms

    @staticmethod
    def _detect_curated_google_authority_target(
        *,
        raw_query: str,
        normalized_query: str,
        corrected_terms: list[str],
    ) -> dict[str, str] | None:
        raw_lower = re.sub(r"\s+", " ", str(raw_query or "").strip().lower())
        normalized = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())

        if not normalized:
            return None

        variant_signal = any(
            marker in raw_lower

            for marker in {
                "bandharan",
                "bandhran",
                "samvidhan",
                "kalam",
                "dhara",
                "artical",
                "articale",
                "articl",
                "articel",
                "atricle",
                "sekshan",
                "sektion",
                " sec ",
            }
        ) or bool(re.search(r"\b(?:bns|bnss)\b", raw_lower)) or any(
            "->" in correction

            and any(token in correction for token in {"article", "section", "constitution", "bns", "bnss"})
            for correction in (corrected_terms or [])
        )

        article_match = re.search(
            r"\b(?:constitution\s+)?(?:article|art)\s+([0-9]+[a-z]?)\b|\b([0-9]+[a-z]?)\s+(?:article|art)\s+(?:constitution)\b",
            normalized,
        )

        if article_match and ("constitution" in normalized or variant_signal):
            article_value = article_match.group(1) or article_match.group(2)
            article_value = str(article_value or "").upper()

            if article_value:
                return {
                    "kind": "constitutional_article_authority",
                    "google_query": f"Article {article_value} Constitution of India explanation",
                    "domain": "constitutional",
                }

        if article_match and ChatService._looks_like_complete_authority_reference_prompt(normalized):
            article_value = article_match.group(1) or article_match.group(2)
            article_value = str(article_value or "").upper()

            if article_value:
                return {
                    "kind": "constitutional_article_authority",
                    "google_query": f"Article {article_value} Constitution of India explanation",
                    "domain": "constitutional",
                }

        section_patterns = [
            re.search(r"\bbnss\s+section\s+([0-9]+[a-z]?)\b", normalized),
            re.search(r"\bsection\s+([0-9]+[a-z]?)\s+bnss\b", normalized),
            re.search(r"\bbns\s+section\s+([0-9]+[a-z]?)\b", normalized),
            re.search(r"\bsection\s+([0-9]+[a-z]?)\s+bns\b", normalized),
        ]

        bnss_match = section_patterns[0] or section_patterns[1]

        if bnss_match:
            section_value = str(bnss_match.group(1) or "").upper()

            return {
                "kind": "bnss_section_variant",
                "google_query": f"Section {section_value} Bharatiya Nagarik Suraksha Sanhita explanation",
                "domain": "criminal",
            }

        bns_match = section_patterns[2] or section_patterns[3]

        if bns_match:
            section_value = str(bns_match.group(1) or "").upper()

            return {
                "kind": "bns_section_variant",
                "google_query": f"Section {section_value} Bharatiya Nyaya Sanhita explanation",
                "domain": "criminal",
            }

        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", normalized)

        if section_match and ChatService._looks_like_complete_authority_reference_prompt(normalized):
            section_value = str(section_match.group(1) or "").upper()

            if "ipc" in normalized or "indian penal code" in normalized:

                return {
                    "kind": "ipc_section_authority",
                    "google_query": f"Section {section_value} Indian Penal Code explanation",
                    "domain": "criminal",
                }

            if "ni act" in normalized or "negotiable instruments act" in normalized:

                return {
                    "kind": "ni_act_section_authority",
                    "google_query": f"Section {section_value} Negotiable Instruments Act explanation",
                    "domain": "civil",
                }

            act_match = re.search(r"\bsection\s+[0-9]+[a-z]?\s+((?:[a-z]+\s+){0,5}act)\b", normalized)

            if act_match:
                act_name = " ".join(word.capitalize() for word in act_match.group(1).split())

                return {
                    "kind": "generic_act_section_authority",
                    "google_query": f"Section {section_value} {act_name} explanation",
                    "domain": "civil",
                }

        return None

    @staticmethod
    def _extract_query_detail_instructions(message: str) -> dict[str, bool]:
        lowered = re.sub(r"\s+", " ", str(message or "").strip().lower())

        if not lowered:

            return {
                "include_articles": False,
                "concise": False,
                "step_by_step": False,
                "in_points": False,
                "practical_context": False,
            }

        return {
            "include_articles": any(
                phrase in lowered
                for phrase in {
                    "along with articles",
                    "with articles",
                    "with article numbers",
                    "article numbers",
                    "article wise",
                    "article-wise",
                    "which articles",
                    "what articles",

                }
            ),

            "concise": any(
                phrase in lowered
                for phrase in {
                    "in short",
                    "briefly",
                    "brief",
                    "concise",
                    "short version",
                    "short answer",
                    "simple summary",
                }
            ),

            "step_by_step": any(
                phrase in lowered
                for phrase in {
                    "step by step",
                    "step-by-step",
                }
            ),

            "in_points": any(
                phrase in lowered
                for phrase in {
                    "point wise",
                    "pointwise",
                    "in points",
                    "bullet points",
                    "as points",
                }
            ),

            "practical_context": any(
                phrase in lowered
                for phrase in {
                    "violation",
                    "remedy",
                    "what should i do",
                    "how to use",
                    "court",
                    "writ",
                    "petition",
                    "challenge",
                    "file",
                    "case",
                    "dispute",
                }
            ),
        }

    @staticmethod
    def _format_constitutional_authority_explainer_answer(
        *,
        title: str,
        summary: str,
        legal_position: str,
        format_profile: dict[str, Any] | None = None,
    ) -> str:
        layout = str((format_profile or {}).get("authority_layout") or (format_profile or {}).get("layout") or "definition").strip().lower()

        if layout == "points":

            return "\n\n".join(
                part

                for part in [
                    title.strip(),
                    ChatService._format_bulleted_lines(
                        ChatService._build_structured_authority_point_lines(
                            summary=summary,
                            legal_position=legal_position,
                        )
                    ),
                ]

                if part
            )

        if layout == "step_by_step":
            return "\n\n".join(

                part

                for part in [
                    title.strip(),

                    ChatService._format_numbered_lines(
                        [
                            line

                            for line in [
                                summary.strip(),
                                ChatService._format_human_legal_position(legal_position),
                            ]

                            if line
                        ]
                    ),
                ]

                if part
            )

        lead = (
            ChatService._format_sentence_style_authority_lead(title=title, summary=summary)

            if layout == "definition"
            else " ".join(part for part in [title.strip(), summary.strip()] if part)
        )

        parts = [lead, ChatService._format_human_legal_position(legal_position)]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _constitutional_response_preferences(message: str) -> dict[str, bool]:
        return ChatService._extract_query_detail_instructions(message)

    @staticmethod
    def _format_bulleted_lines(lines: list[str]) -> str:
        return "\n".join(f"- {line}" for line in lines)

    @staticmethod
    def _format_numbered_lines(lines: list[str]) -> str:
        return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))

    @staticmethod
    def _build_structured_explainer_point_lines(
        *,
        title: str,
        summary: str,
        legal_position: str,
        next_steps: str,
        article_breakdown: list[str],
        practical_context: bool,
    ) -> list[str]:
        point_lines: list[str] = []
        clean_summary = ChatService._polish_direct_answer_text(summary)
        clean_legal_position = ChatService._polish_direct_answer_text(legal_position)
        clean_next_steps = next_steps.strip()

        if clean_summary:
            point_lines.append(f"Overview: {clean_summary}")

        if article_breakdown:
            point_lines.append("Article-wise breakdown:")
            point_lines.extend(article_breakdown)

        if clean_legal_position:
            point_lines.append(f"Legal position: {clean_legal_position}")

        if practical_context and clean_next_steps:
            point_lines.append(f"Practical use: {clean_next_steps}")
        return point_lines

    @staticmethod
    def _build_structured_authority_point_lines(*, summary: str, legal_position: str) -> list[str]:
        point_lines: list[str] = []
        clean_summary = ChatService._polish_direct_answer_text(summary)
        clean_legal_position = ChatService._polish_direct_answer_text(legal_position)

        if clean_summary:
            point_lines.append(f"Overview: {clean_summary}")

        if clean_legal_position:
            point_lines.append(f"Legal position: {clean_legal_position}")

        return point_lines

    @staticmethod
    def _build_structured_general_explainer_point_lines(
        *,
        summary: str,
        legal_position: str,
        next_steps: str,
        practical_context: bool,
    ) -> list[str]:
        point_lines: list[str] = []
        clean_summary = ChatService._polish_direct_answer_text(summary)
        clean_legal_position = ChatService._polish_direct_answer_text(legal_position)
        clean_next_steps = next_steps.strip()

        if clean_summary:
            point_lines.append(f"Overview: {clean_summary}")

        if clean_legal_position:
            point_lines.append(f"Legal position: {clean_legal_position}")

        if practical_context and clean_next_steps:
            point_lines.append(f"Practical use: {clean_next_steps}")
        return point_lines

    def _route_fast_authority_lookup(
        self,
        *,
        message: str,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        query_profile: dict[str, Any],
        strategy: dict[str, str | bool],
        understanding_profile: dict[str, Any] | None = None,
    ) -> tuple[InternalChatResult, ConversationState] | None:

        del domain, strategy
        understanding_profile = understanding_profile or self._understand_legal_query(message)
        normalized_query = str(understanding_profile.get("normalized_query") or message)
        local_match = self.legal_dataset.lookup_query(normalized_query)
        authority_key = self._fast_authority_lookup_key(normalized_query)
        flow_type = str(query_profile.get("flow_type") or "")
        use_local_dataset = self._should_use_local_legal_dataset_fast_path(
            normalized_query=normalized_query,
            authority_key=authority_key,
            local_match=local_match,
        )

        if local_match is None and (not authority_key or flow_type == "provision_lookup"):
            return None

        cache_key = self._build_direct_answer_cache_key(
            kind="local_legal_dataset_fast_path" if use_local_dataset else "authority_fast_path",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "authority_fast_path",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "last_grounded_query": message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )
            return internal, next_state

        if use_local_dataset and local_match is not None:
            response_mode = str(query_profile.get("response_mode") or "authority")
            internal = InternalChatResult(
                answer=self._format_local_legal_dataset_answer(local_match),
                domain=local_match.domain,
                follow_up_question=None,
                citations=[local_match.source],
                authorities=[local_match.answer_title],
                documents_to_keep=[],
                likely_forum=None,
                caution=None,
                warnings=warnings,

                raw_json={
                    "source": "local_legal_dataset",
                    "pipeline": "local_legal_dataset_fast_path",
                    "authority_key": None,
                    "local_dataset_match": {
                        "dataset": local_match.dataset,
                        "provision_number": local_match.provision_number,
                        "title": local_match.title,
                        "source": local_match.source,
                    },

                    "query_profile": {
                        **query_profile,
                        "response_mode": response_mode,
                    },

                    "response_mode": response_mode,
                    "direct_answer_format": {"family": "authority", "layout": "structured"},
                    "disclaimer_mode": "none",
                },
            )

            internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
            self._set_direct_answer_cache_entry(cache_key, internal)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "local_legal_dataset_fast_path",
                    "awaiting_details": False,
                    "last_user_issue": message,
                    "last_grounded_query": message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": None,
                }
            )

            return internal, next_state
        curated_google_documents = self._lookup_curated_google_direct_documents(normalized_query)
        payload = FAST_AUTHORITY_LOOKUPS.get(authority_key)

        if not payload:
            return None
        authority = str(payload.get("authority") or payload.get("title") or "").strip()
        fast_format_profile = self._resolve_fast_authority_format_profile(
            dict(understanding_profile.get("direct_answer_format") or {})
        )

        answer = self._format_fast_authority_answer(
            title=str(payload.get("title") or ""),
            summary=str(payload.get("summary") or ""),
            legal_position=str(payload.get("legal_position") or ""),
            format_profile=fast_format_profile,
        )

        internal = InternalChatResult(
            answer=answer,
            domain=str(payload.get("domain") or conversation_state.legal_domain or "civil"),
            follow_up_question=None,
            citations=self._preferred_direct_citations(
                google_documents=curated_google_documents,
                fallback_citations=[str(payload.get("source") or payload.get("title") or "").strip()],
            ),

            authorities=[authority] if authority else [],
            documents_to_keep=[],
            likely_forum=None,
            caution=None,
            warnings=warnings,

            raw_json={
                "source": "authority_fast_path",
                "pipeline": "authority_fast_path",
                "authority_key": authority_key,
                "curated_google_used": bool(curated_google_documents),
                "curated_google_count": len(curated_google_documents),
                "direct_answer_format": fast_format_profile,
                "disclaimer_mode": str(payload.get("disclaimer_mode") or "medium_risk"),
            },
        )

        internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
        self._set_direct_answer_cache_entry(cache_key, internal)
        next_state = conversation_state.model_copy(

            update={
                "conversation_started": True,
                "active_intent": "authority_fast_path",
                "awaiting_details": False,
                "last_user_issue": message,
                "last_grounded_query": message,
                "legal_domain": internal.domain,
                "last_follow_up_question": None,
            }
        )

        return internal, next_state

    @staticmethod
    def _format_fast_authority_answer(
        *,
        title: str,
        summary: str,
        legal_position: str,
        format_profile: dict[str, Any] | None = None,
    ) -> str:
        layout = str((format_profile or {}).get("authority_layout") or (format_profile or {}).get("layout") or "definition").strip().lower()

        if layout == "points":
            return "\n\n".join(
                part

                for part in [
                    title.strip(),
                    ChatService._format_bulleted_lines(
                        ChatService._build_structured_authority_point_lines(
                            summary=summary,
                            legal_position=legal_position,
                        )
                    ),
                ]

                if part
            )

        if layout == "step_by_step":
            return "\n\n".join(
                part

                for part in [
                    title.strip(),
                    ChatService._format_numbered_lines(
                        [
                            line

                            for line in [
                                summary.strip(),
                                ChatService._format_human_legal_position(legal_position),
                            ]

                            if line
                        ]
                    ),
                ]

                if part
            )

        if layout == "concise":
            return ChatService._format_sentence_style_authority_lead(title=title, summary=summary)

        lead = (
            ChatService._format_sentence_style_authority_lead(title=title, summary=summary)

            if layout == "definition"

            else " ".join(part for part in [title.strip(), summary.strip()] if part)
        )

        parts = [lead, ChatService._format_human_legal_position(legal_position)]

        return "\n\n".join(part for part in parts if part)

    @staticmethod

    def _format_local_legal_dataset_answer(match: LegalProvisionMatch) -> str:
        return ChatService._format_authority_structured_answer(
            matched_query=match.provision_number,
            title=match.title,
            text=match.text,
            explanation=match.explanation,
            source=match.source,
        )

    @staticmethod

    def _response_mode_for_flow(flow_type: str) -> str:
        normalized = str(flow_type or "").strip().lower()

        if normalized == "provision_lookup":
            return "authority"

        if normalized == "scenario_practical_legal_issue":
            return "scenario"

        return "research"

    @staticmethod
    def _format_authority_structured_answer(
        *,
        matched_query: str,
        title: str,
        text: str,
        explanation: str,
        source: str,
    ) -> str:
        cleaned_match = ChatService._sanitize_section_value(matched_query, fallback="The requested provision")
        cleaned_title = ChatService._sanitize_section_value(title, fallback="the relevant legal topic")
        cleaned_text = ChatService._sanitize_section_value(text, fallback="The exact text was not fully available in the retrieved material.")
        cleaned_explanation = ChatService._sanitize_section_value(
            explanation,
            fallback="This is the closest grounded explanation available for the provision you asked about.",
        )
        cleaned_source = ChatService._sanitize_sources(source)
        act_name = ChatService._authority_act_name(source=cleaned_source, title=cleaned_title, matched_query=cleaned_match)
        what_it_is = ChatService._build_authority_opening(
            matched_query=cleaned_match,
            act_name=act_name,
            title=cleaned_title,
        )
        meaning = ChatService._polish_authority_explanation(
            explanation=cleaned_explanation,
            title=cleaned_title,
        )
        key_points = ChatService._derive_key_points(cleaned_title, meaning)
        punishment = ChatService._derive_punishment_text(cleaned_text, meaning, cleaned_title)
        practical_use = ChatService._authority_practical_use(
            matched_query=cleaned_match,
            act_name=act_name,
            title=cleaned_title,
        )
        return ChatService._format_numbered_legal_answer(
            what_it_is=what_it_is,
            meaning=meaning,
            key_points=key_points,
            punishment=punishment,
            practical_use=practical_use,
            source=cleaned_source,
            disclaimer=ChatService._default_brief_disclaimer(),
        )

    @staticmethod
    def _build_authority_opening(*, matched_query: str, act_name: str, title: str) -> str:
        cleaned_match = ChatService._sanitize_section_value(matched_query, fallback="This provision")
        cleaned_act = ChatService._sanitize_section_value(act_name, fallback="the relevant law")
        cleaned_title = ChatService._sanitize_section_value(title, fallback="the issue in question")
        return f"{cleaned_match} of {cleaned_act} deals with {cleaned_title.lower()}."

    @staticmethod
    def _polish_authority_explanation(*, explanation: str, title: str) -> str:
        cleaned = ChatService._sanitize_section_value(
            explanation,
            fallback="In plain terms, this provision should be read in light of its wording, conditions, and practical use.",
        )
        normalized_title = ChatService._sanitize_section_value(title, fallback="the issue in question").lower()
        substitutions = [
            (
                r"^this\s+is\s+the\s+constitutional\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it sets out the constitutional position on {normalized_title}.",
            ),
            (
                r"^this\s+is\s+the\s+ipc\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it addresses {normalized_title}.",
            ),
            (
                r"^this\s+is\s+the\s+bns\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it covers {normalized_title}.",
            ),
            (
                r"^this\s+is\s+the\s+bnss\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it sets out the procedural rule on {normalized_title}.",
            ),
            (
                r"^this\s+is\s+the\s+statutory\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it explains the legal rule on {normalized_title}.",
            ),
            (
                r"^(?:article|section|rule)\s+[0-9a-z()/-]+\s+is\s+the\s+constitutional\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it sets out the constitutional position on {normalized_title}.",
            ),
            (
                r"^(?:article|section|rule)\s+[0-9a-z()/-]+\s+is\s+the\s+ipc\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it addresses {normalized_title}.",
            ),
            (
                r"^(?:article|section|rule)\s+[0-9a-z()/-]+\s+is\s+the\s+bns\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it covers {normalized_title}.",
            ),
            (
                r"^(?:article|section|rule)\s+[0-9a-z()/-]+\s+is\s+the\s+bnss\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it sets out the procedural rule on {normalized_title}.",
            ),
            (
                r"^(?:article|section|rule)\s+[0-9a-z()/-]+\s+is\s+the\s+statutory\s+provision\s+dealing\s+with\s+.+",
                f"In plain terms, it explains the legal rule on {normalized_title}.",
            ),
        ]
        lowered = cleaned.lower()
        for pattern, replacement in substitutions:
            if re.match(pattern, lowered, flags=re.IGNORECASE):
                remainder = ""
                sentence_parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
                if len(sentence_parts) > 1:
                    remainder = sentence_parts[1].strip()
                cleaned = replacement if not remainder else f"{replacement} {remainder}"
                break
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    @staticmethod
    def _derive_key_points(*values: str, limit: int = 3) -> list[str]:
        points: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = ChatService._sanitize_text_block(value)
            if not cleaned:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
                point = ChatService._sanitize_section_value(sentence, fallback="").strip(" .")
                point = re.sub(r"^(?:the relevant text says|in plain terms|practically)\s*:\s*", "", point, flags=re.IGNORECASE)
                if re.search(r"\b(?:punish(?:ed|ment)?|imprisonment|fine|penalty|liable|sentence)\b", point, flags=re.IGNORECASE):
                    continue
                if len(point) < 18:
                    continue
                point = re.sub(r"^it\s+", "", point, flags=re.IGNORECASE)
                point = re.sub(r"\s+", " ", point).strip(" .")
                if len(point) > 140:
                    point = point[:140].rsplit(" ", 1)[0].strip(" .")
                lowered = point.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                points.append(point)
                if len(points) >= limit:
                    return points
        return points or ["Read the exact provision with the facts and stage of the matter before relying on it."]

    @staticmethod
    def _derive_punishment_text(*values: str) -> str:
        punishment_markers = r"\b(?:punish(?:ed|ment)?|imprisonment|fine|penalty|liable|sentence)\b"
        for value in values:
            cleaned = ChatService._sanitize_text_block(value)
            if not cleaned:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
                if not re.search(punishment_markers, sentence, flags=re.IGNORECASE):
                    continue
                snippet_match = re.search(
                    r"(shall be punished[^.]*|punishment[^.]*|imprisonment[^.]*|fine[^.]*|liable to fine[^.]*)",
                    sentence,
                    flags=re.IGNORECASE,
                )
                snippet = snippet_match.group(1) if snippet_match else sentence
                point = ChatService._sanitize_section_value(snippet, fallback="").strip(" .")
                point = re.sub(r"^(?:the relevant text says)\s*:\s*", "", point, flags=re.IGNORECASE)
                if point:
                    return point
        return "No specific punishment is stated in the material I relied on."

    @staticmethod
    def _authority_practical_use(*, matched_query: str, act_name: str, title: str) -> str:
        combined = f"{matched_query} {act_name} {title}".lower()
        if "constitution" in combined:
            domain = "constitutional"
        elif "nagarik suraksha sanhita" in combined or "bnss" in combined:
            domain = "procedure"
        elif any(token in combined for token in {"penal code", "nyaya sanhita", "ni act", "negotiable instruments"}):
            domain = "criminal"
        else:
            domain = "general"
        provision_label = f"{matched_query} of {act_name}"
        return ChatService._grounded_statute_next_steps(domain=domain, title=provision_label)

    @staticmethod
    def _format_numbered_legal_answer(
        *,
        what_it_is: str,
        meaning: str,
        key_points: list[str] | str | None,
        punishment: str,
        practical_use: str,
        source: str,
        disclaimer: str,
    ) -> str:
        cleaned_what = ChatService._sanitize_section_value(what_it_is, fallback="Relevant legal material was identified.")
        cleaned_meaning = ChatService._sanitize_section_value(
            meaning,
            fallback="This is the closest grounded explanation available from the material I relied on.",
        )
        if isinstance(key_points, str):
            key_point_items = ChatService._derive_key_points(key_points)
        else:
            key_point_items = [ChatService._sanitize_section_value(item, fallback="").strip(" .") for item in (key_points or []) if str(item).strip()]
        key_point_items = [item for item in key_point_items if item][:3] or ["Read the exact provision with the facts and stage of the matter before relying on it."]
        cleaned_punishment = ChatService._sanitize_section_value(
            punishment,
            fallback="No specific punishment is stated in the material I relied on.",
        )
        cleaned_practical = ChatService._sanitize_section_value(
            practical_use,
            fallback="Use the provision only after matching it with the actual facts and stage of the matter.",
        )
        cleaned_source = ChatService._sanitize_sources(source)
        cleaned_disclaimer = ChatService._sanitize_section_value(
            disclaimer,
            fallback=ChatService._default_brief_disclaimer(),
        )
        lines = [
            f"1. What it is: {cleaned_what}",
            f"2. Meaning: {cleaned_meaning}",
            "3. Key points / elements:",
            *[f"- {item}" for item in key_point_items],
            f"4. Punishment: {cleaned_punishment}",
            f"5. Practical use: {cleaned_practical}",
            f"6. Source: {cleaned_source}",
            f"Note: {cleaned_disclaimer}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _authority_act_name(*, source: str, title: str, matched_query: str) -> str:
        cleaned_source = ChatService._sanitize_section_value(source, fallback="the relevant law")
        cleaned_source = re.sub(r"\s*\(.*?(local dataset|dataset).*?\)\s*", "", cleaned_source, flags=re.IGNORECASE).strip(" .")
        combined = f"{cleaned_source} {title}".strip()
        known_laws = [
            "Constitution of India",
            "Indian Penal Code, 1860",
            "Indian Penal Code",
            "Bharatiya Nyaya Sanhita",
            "Bharatiya Nagarik Suraksha Sanhita",
            "Negotiable Instruments Act, 1881",
        ]
        for law in known_laws:
            if law.lower() in combined.lower():
                return law
        if cleaned_source and ":" not in cleaned_source and "|" not in cleaned_source:
            return cleaned_source
        title_text = f"{matched_query} in {title}"
        match = re.search(r"\b(?:Article|Section|Rule)\s+[0-9A-Z()/-]+\s+(?:of|in)\s+(.+)$", title_text, flags=re.IGNORECASE)
        if match:
            candidate = ChatService._sanitize_section_value(match.group(1), fallback="the relevant law").strip(" .")
            if candidate:
                return candidate
        return "the relevant law"

    def _build_grounded_authority_payload(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
        citations: list[str],
    ) -> dict[str, Any] | None:
        match = self._select_best_authority_document_match(query=query, documents=documents)

        if match is None:
            return None
        primary_document = match["document"]
        matched_query = str(match["matched_query"] or self._derive_authority_provision_number(query=query, document=primary_document))
        title = self._derive_authority_title(document=primary_document, matched_query=matched_query)
        text = self._derive_authority_text(document=primary_document)
        explanation = self._derive_authority_explanation(
            query=query,
            document=primary_document,
            matched_query=matched_query,
            title=title,
            match_level=str(match["match_level"]),
        )

        source = self._derive_authority_source(document=primary_document, citations=citations)
        answer = self._format_authority_structured_answer(
            matched_query=matched_query,
            title=title,
            text=text,
            explanation=explanation,
            source=source,
        )

        return {
            "answer": answer,
            "follow_up_question": None,
            "likely_forum": None,
            "caution": None,
            "documents_to_keep": [],
            "source": "deterministic_grounded_authority",
            "authority_match_level": str(match["match_level"]),
            "llm_failed": False,
        }

    def _select_best_authority_document_match(self, *, query: str, documents: list[dict[str, Any]]) -> dict[str, Any] | None:

        if not documents:
            return None
        assessments = [
            self._assess_authority_document_relevance(doc=doc, query=query, rank=index)

            for index, doc in enumerate(documents)
        ]
        exact_matches = [item for item in assessments if item["match_level"] == "exact"]

        if exact_matches:
            exact_matches.sort(
                key=lambda item: (
                    int(item["source_rank"]),
                    -float(item["score"]),
                    int(item["rank"]),
                )
            )
            return exact_matches[0]

        partial_matches = [item for item in assessments if item["match_level"] == "partial"]

        if partial_matches:
            partial_matches.sort(
                key=lambda item: (
                    int(item["source_rank"]),
                    -float(item["score"]),
                    int(item["rank"]),
                )
            )
            return partial_matches[0]

        return None

    def _assess_authority_document_relevance(self, *, doc: dict[str, Any], query: str, rank: int) -> dict[str, Any]:
        reference = self._parse_authority_reference(query)
        text = self._document_authority_text(doc)
        title = str(doc.get("title") or "").strip()
        source_rank = self._authority_source_rank(doc)
        exact = self._document_supports_requested_authority(doc=doc, query=query)
        partial = False

        score = float(doc.get("score") or 0.0)
        matched_query = reference["label"]

        if not exact:
            normalized_text = text.lower()
            statute_aliases = reference["statute_aliases"]
            statute_hit = any(alias in normalized_text for alias in statute_aliases) if statute_aliases else False
            identifier = str(reference["identifier"] or "").lower()
            negated_label = bool(identifier) and (
                re.search(rf"\b(?:without|not|no)\b[^.:\n]{{0,40}}\bsection\s+{re.escape(identifier)}\b", normalized_text) is not None
                or re.search(rf"\b(?:without|not|no)\b[^.:\n]{{0,40}}\barticle\s+{re.escape(identifier)}\b", normalized_text) is not None
                or re.search(rf"\b(?:without|not|no)\b[^.:\n]{{0,40}}\brule\s+{re.escape(identifier)}\b", normalized_text) is not None
            )

            identifier_hit = bool(identifier) and (
                f"section {identifier}" in normalized_text
                or f"article {identifier}" in normalized_text
                or f"rule {identifier}" in normalized_text
                or re.search(rf"\b{re.escape(identifier)}\b", normalized_text) is not None
            )

            if negated_label:
                identifier_hit = False
            meaningful_overlap = len(self._meaningful_query_tokens(query.lower()) & self._meaningful_query_tokens(normalized_text))

            partial = (
                (statute_hit and meaningful_overlap >= 1)
                or (identifier_hit and meaningful_overlap >= 1)
                or (reference["kind"] == "query" and meaningful_overlap >= 2)
            )

            if reference["kind"] in {"section", "article", "rule"} and statute_aliases and not statute_hit and not identifier_hit:
                partial = False

            if negated_label and reference["kind"] in {"section", "article", "rule"}:
                partial = False
        match_level = "exact" if exact else ("partial" if partial else "none")

        if match_level == "partial" and reference["label"]:
            matched_query = reference["label"]

        elif not matched_query:
            matched_query = self._clean_search_snippet(title) or "Requested provision/query"

        return {
            "document": doc,
            "match_level": match_level,
            "matched_query": matched_query,
            "rank": rank,
            "score": score,
            "source_rank": source_rank,
        }

    @staticmethod
    def _authority_source_rank(doc: dict[str, Any]) -> int:
        source_kind = str(doc.get("source_kind") or "").strip().lower()
        source = str(doc.get("docsource") or "").strip().lower()

        if source_kind == "local_legal_dataset":
            return 0

        if source in {"laws", "constitution", "supremecourt"} or source_kind == "indiankanoon":
            return 1

        if source_kind == "internal":
            return 2

        if source == "user_upload" or source_kind == "user_upload":
            return 3

        if source_kind == "google_custom_search" or source.startswith("google:"):
            return 4

        return 5

    def _parse_authority_reference(self, query: str) -> dict[str, Any]:
        normalized = re.sub(r"\s+", " ", str(query or "").strip().lower())
        label = re.sub(r"\s+", " ", str(query or "").strip()) or "Requested provision/query"

        statute_alias_map = {
            "constitution": ("constitution", "constitution of india"),
            "ipc": ("ipc", "indian penal code"),
            "bns": ("bns", "bharatiya nyaya sanhita"),
            "bnss": ("bnss", "bharatiya nagarik suraksha sanhita"),
            "crpc": ("crpc", "code of criminal procedure"),
            "cpc": ("cpc", "code of civil procedure"),
            "ni act": ("ni act", "negotiable instruments act"),
        }

        for kind in ("article", "section", "rule"):
            match = re.search(rf"\b{kind}\s+([0-9]+[a-z]?)\b", normalized)

            if match:
                identifier = match.group(1).upper()
                statute_key = ""
                statute_aliases: tuple[str, ...] = ()

                for candidate_key, aliases in statute_alias_map.items():

                    if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
                        statute_key = candidate_key
                        statute_aliases = aliases
                        break

                label = f"{kind.title()} {identifier}" + (f" {statute_key.upper()}" if statute_key in {"ipc", "bns", "bnss", "crpc", "cpc"} else "")

                return {
                    "kind": kind,
                    "identifier": identifier,
                    "statute_key": statute_key,
                    "statute_aliases": statute_aliases,
                    "label": label,
                }

        for statute_key, aliases in statute_alias_map.items():
            match = re.search(
                rf"\b(?:{'|'.join(re.escape(alias) for alias in aliases)})\s+(?:section\s+)?([0-9]+[a-z]?)\b",
                normalized,
            )

            if match:
                identifier = match.group(1).upper()
                return {
                    "kind": "section",
                    "identifier": identifier,
                    "statute_key": statute_key,
                    "statute_aliases": aliases,
                    "label": f"Section {identifier} {statute_key.upper()}" if statute_key in {"ipc", "bns", "bnss", "crpc", "cpc"} else f"Section {identifier} {statute_key.title()}",
                }

        return {
            "kind": "query",
            "identifier": "",
            "statute_key": "",
            "statute_aliases": (),
            "label": label,
        }

    def _derive_authority_provision_number(self, *, query: str, document: dict[str, Any] | None) -> str:
        normalized_query = re.sub(r"\s+", " ", str(query or "").strip().lower())

        for label in ("article", "section", "rule"):
            match = re.search(rf"\b{label}\s+([0-9]+[a-z]?)\b", normalized_query)

            if match:
                return f"{label.title()} {match.group(1).upper()}"
        raw_text = self._document_authority_text(document or {})

        for label in ("article", "section", "rule"):
            match = re.search(rf"\b{label}\s+([0-9]+[a-z]?)\b", raw_text)

            if match:
                return f"{label.title()} {match.group(1).upper()}"
        title = str((document or {}).get("title") or "Requested provision").strip()

        return self._clean_search_snippet(title) or "Requested provision"

    def _derive_authority_title(self, *, document: dict[str, Any] | None, matched_query: str) -> str:

        candidates = [
            str((document or {}).get("headline") or "").strip(),
            str((document or {}).get("fragment_headline") or "").strip(),
            str((document or {}).get("fragment_title") or "").strip(),
            str((document or {}).get("title") or "").strip(),
        ]

        for candidate in candidates:
            cleaned = self._clean_search_snippet(candidate).strip(" .")

            if not cleaned:
                continue

            cleaned = re.sub(
                rf"^{re.escape(matched_query)}\s+(?:in|of)\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" .")
            cleaned = re.sub(
                r"^(?:Article|Section|Rule)\s+[0-9A-Z()/-]+\s+(?:of|in)\s+.+?\s+(?:explains|contains|provides|covers|sets out)\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" .")

            if cleaned:
                return cleaned

        return "Statutory text"

    def _derive_authority_text(self, *, document: dict[str, Any] | None) -> str:

        candidates = [
            str((document or {}).get("doc_excerpt") or "").strip(),
            str((document or {}).get("fragment_excerpt") or "").strip(),
            str((document or {}).get("headline") or "").strip(),
            str((document or {}).get("fragment_headline") or "").strip(),
            str((document or {}).get("title") or "").strip(),
        ]

        for candidate in candidates:
            cleaned = self._sanitize_text_block(candidate)
            cleaned = re.sub(r"\{[^{}]*\"errmsg\"[^{}]*\}", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(errmsg|debug|traceback|stack trace)\b\s*:?\s*[^.;]*", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")

            if cleaned:
                return cleaned[:1800]

        return "No grounded statutory text excerpt was available in the retrieved authority material."

    def _derive_authority_explanation(
        self,
        *,
        query: str,
        document: dict[str, Any] | None,
        matched_query: str,
        title: str,
        match_level: str,
    ) -> str:
        normalized_query = str(query or "").strip().lower()
        normalized_title = title.strip().rstrip(".").lower() or "the requested legal subject"
        authority_type = str((document or {}).get("authority_type") or "").strip().lower()

        if match_level == "partial":
            return (
                f"I found material on a related point, but not a clean exact match for the provision you named. "
                f"The closest grounded result discusses {normalized_title}, so it should be treated cautiously until the exact section or article is checked."
            )

        if matched_query.lower().startswith("article"):
            return (
                f"In plain terms, it sets out the constitutional position on {normalized_title}. "
                "How far it helps in a real matter depends on the facts, the context in which it is invoked, and the way courts have interpreted it."
            )

        if "ipc" in normalized_query or "indian penal code" in normalized_query:
            return (
                f"In plain terms, it addresses {normalized_title}. "
                "It is usually invoked when the legal ingredients of the offence are present, and it does not automatically apply just because the dispute sounds similar."
            )

        if "bns" in normalized_query or "bharatiya nyaya sanhita" in normalized_query:
            return (
                f"In plain terms, it covers {normalized_title}. "
                "In practice, it applies only when the facts fit the statutory ingredients and the allegation is properly supported in the complaint or case record."
            )

        if "bnss" in normalized_query or "bharatiya nagarik suraksha sanhita" in normalized_query:
            return (
                f"In plain terms, it sets out the procedural rule for {normalized_title}. "
                "Its effect usually depends on the stage of the case, the forum involved, and whether the procedural conditions have been met."
            )

        if authority_type == "statute":
            return (
                f"In plain terms, it explains the legal rule on {normalized_title}. "
                "The practical effect depends on the exact wording, any built-in conditions or exceptions, and any later judicial interpretation."
            )

        return (
            f"In plain terms, the retrieved material addresses {normalized_title}. "
            "Its weight depends on the source, the wording used, and how closely it matches the facts you are dealing with."
        )

    def _derive_authority_source(self, *, document: dict[str, Any] | None, citations: list[str]) -> str:
        title = str((document or {}).get("title") or "Retrieved authority").strip()
        source = str((document or {}).get("docsource") or "authority_source").strip()
        source_kind = str((document or {}).get("source_kind") or "").strip().lower()
        if citations and not str(citations[0]).strip().lower().startswith("internal"):
            return self._clean_visible_authority_source(citations[0])
        if source_kind == "internal" or source.startswith("internal:"):
            cleaned_title = self._clean_search_snippet(title).strip(" .")
            return cleaned_title or "Internal legal dataset"
        url = str((document or {}).get("url") or "").strip()
        if "indiankanoon.org" in url.lower():
            cleaned_title = self._clean_search_snippet(title).strip(" .")
            return cleaned_title or "India Kanoon"
        cleaned_source = self._clean_visible_authority_source(source)
        if cleaned_source and cleaned_source != "authority_source":
            return cleaned_source
        cleaned_title = self._clean_search_snippet(title).strip(" .")
        return cleaned_title or "Retrieved legal source"

    @staticmethod
    def _clean_visible_authority_source(text: str) -> str:
        cleaned = ChatService._sanitize_section_value(text, fallback="Retrieved legal source")
        lowered = cleaned.lower()
        if lowered.startswith("internal:"):
            return "Internal legal dataset"
        cleaned = re.sub(r"\s*\|\s*internal:[^|]+$", "", cleaned, flags=re.IGNORECASE).strip(" .")
        cleaned = re.sub(r"\s*\(local dataset:\s*[^)]+\)", "", cleaned, flags=re.IGNORECASE).strip(" .")
        cleaned = re.sub(r"\bAIR\s+\d{4}[^;,.|]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\d+\s*SCC\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\d+\s*CriLJ\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\.pdf\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("google:indiacode.nic.in", "India Code").replace("google:indiankanoon.org", "India Kanoon")
        if lowered in {"authority_source", "internal"}:
            return "Internal legal dataset"
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;,-|")
        return cleaned or "Retrieved legal source"

    @staticmethod
    def _should_use_local_legal_dataset_fast_path(
        *,
        normalized_query: str,
        authority_key: str | None,
        local_match: LegalProvisionMatch | None,
    ) -> bool:

        if local_match is None:
            return False
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())

        if not compact:
            return False

        formatting_tail = r"(?:\s+(?:in short|briefly|short|step by step|in points|pointwise))?"

        if re.fullmatch(rf"article\s+[0-9]+[a-z]?{formatting_tail}", compact):
            return True

        if re.fullmatch(rf"section\s+[0-9]+[a-z]?{formatting_tail}", compact):
            return True

        if re.fullmatch(rf"(?:ipc|indian penal code)\s+[0-9]+[a-z]?{formatting_tail}", compact):
            return True

        if re.fullmatch(rf"section\s+[0-9]+[a-z]?\s+(?:ipc|indian penal code){formatting_tail}", compact):
            return True

        if re.search(r"\b(?:explain|what is|tell me about)\s+article\s+[0-9]+[a-z]?\b", compact):
            return True

        return bool(
            re.search(r"\b(?:explain|what is|tell me about)\s+section\s+[0-9]+[a-z]?\s+(?:ipc|indian penal code)\b", compact)
        )

    @staticmethod
    def _resolve_fast_authority_format_profile(format_profile: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(format_profile or {})
        detail_instructions = dict(resolved.get("detail_instructions") or {})
        authority_layout = str(resolved.get("authority_layout") or "").strip().lower()
        if not authority_layout:

            if detail_instructions.get("in_points"):
                authority_layout = "points"

            elif detail_instructions.get("step_by_step"):
                authority_layout = "step_by_step"

            elif detail_instructions.get("concise"):
                authority_layout = "concise"

            else:
                authority_layout = str(resolved.get("layout") or "definition").strip().lower()

        resolved["authority_layout"] = authority_layout or "definition"
        resolved["layout"] = resolved["authority_layout"]

        return resolved

    def _build_direct_answer_clarification_result(
        self,
        *,
        domain: str,
        warnings: list[str],
        conversation_state: ConversationState,
        raw_message: str,
        clarification_hint: dict[str, str] | Any,
    ) -> tuple[InternalChatResult, ConversationState] | None:

        if not isinstance(clarification_hint, dict):
            return None

        normalized_query, _ = self._normalize_legal_query_text(raw_message)
        cache_key = self._build_direct_answer_cache_key(
            kind="direct_answer_clarification",
            normalized_query=normalized_query,
        )

        cached_internal = self._get_direct_answer_cache_entry(cache_key)

        if cached_internal is not None:
            internal = self._with_direct_cache_metadata(cached_internal, cache_hit=True, warnings=warnings)
            next_state = conversation_state.model_copy(

                update={
                    "conversation_started": True,
                    "active_intent": "direct_answer_clarification",
                    "awaiting_details": True,
                    "last_user_issue": raw_message,
                    "legal_domain": internal.domain,
                    "last_follow_up_question": internal.follow_up_question,
                }
            )
            return internal, next_state

        answer = str(clarification_hint.get("answer") or "").strip()

        question = str(clarification_hint.get("question") or "").strip()
        clarification_kind = str(clarification_hint.get("kind") or "direct_answer").strip()

        if not answer or not question:
            return None

        internal = InternalChatResult(
            answer=answer,
            domain=str(domain or conversation_state.legal_domain or "constitutional"),
            follow_up_question=question,
            citations=[],
            authorities=[],
            documents_to_keep=[],
            likely_forum=None,
            caution=None,
            warnings=warnings,

            raw_json={
                "source": "direct_answer_clarification",
                "pipeline": "direct_answer_clarification",
                "clarification_kind": clarification_kind,
                "disclaimer_mode": "medium_risk",
            },
        )

        internal = self._with_direct_cache_metadata(internal, cache_hit=False, warnings=warnings)
        self._set_direct_answer_cache_entry(cache_key, internal)
        next_state = conversation_state.model_copy(

            update={
                "conversation_started": True,
                "active_intent": "direct_answer_clarification",
                "awaiting_details": True,
                "last_user_issue": raw_message,
                "legal_domain": internal.domain,
                "last_follow_up_question": question,
            }
        )
        return internal, next_state

    def _retrieve_grounded_documents(
        self,
        *,
        query: str,
        query_variants: list[str],
        doctypes_options: list[str | None],
        state: str | None,
        domain: str | None,
        answer_mode: str,
        query_type: str,
        understanding_profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        authority_lookup_target = (understanding_profile or {}).get("authority_lookup_variant")

        try:
            result = self.hybrid_retrieval.retrieve(
                query=query,
                query_variants=query_variants,
                state=state,
                domain=domain,
                answer_mode=answer_mode,
                query_type=query_type,
                doctypes_options=doctypes_options,
                curated_google_query=str(authority_lookup_target.get("google_query") or "").strip()

                if isinstance(authority_lookup_target, dict)

                else None,
            )

            logger.info(
                "chat pipeline stage=retrieval local=%s internal=%s live=%s google=%s confidence=%.2f documents=%s",
                result.local_dataset_count,
                result.internal_count,
                result.live_count,
                result.google_count,
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
        evidence_packet: dict[str, Any],
        response_mode: str,
    ) -> dict[str, Any]:

        if str(response_mode).strip().lower() == "authority":
            payload = self._build_grounded_authority_payload(
                query=query,
                documents=documents,
                citations=citations,
            )

            if payload is None:
                return {
                    "answer": "",
                    "follow_up_question": None,
                    "likely_forum": None,
                    "caution": None,
                    "documents_to_keep": [],
                    "source": "deterministic_grounded_authority",
                    "authority_match_level": "none",
                    "llm_failed": False,
                }

            payload["source"] = "deterministic_grounded_authority"
            return payload

        prompt = self._build_grounded_prompt(
            query=query,
            domain=domain,
            state=state,
            context=context,
            citations=citations,
            evidence_packet=evidence_packet,
        )

        logger.info(
            "chat pipeline stage=llm_request query=%r citations=%s history=%s prompt_chars=%s context_chars=%s context_non_empty=%s docs=%s top_doc_titles=%s",
            query[:160],
            len(citations),
            len(conversation),
            len(prompt),
            len(context),
            bool(str(context or "").strip()),
            len(documents),
            [str(doc.get("title") or "")[:120] for doc in documents[:3]],
        )

        try:
            payload = self.openai.generate_json(prompt, conversation)
            normalized_payload = self._normalize_llm_payload(
                payload=payload,
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
                response_mode=response_mode,
            )

            logger.info(
                "chat pipeline stage=llm_response keys=%s answer_chars=%s",
                sorted(normalized_payload.keys()),
                len(str(normalized_payload.get("answer") or "")),
            )
            return normalized_payload

        except Exception as exc:
            logger.warning("chat pipeline stage=llm_error query=%r error=%s", query[:160], exc)
            payload = self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
                response_mode=response_mode,
            )

            payload["source"] = "deterministic_grounded_fallback"
            payload["llm_error"] = str(exc)
            payload["llm_failed"] = True
            return payload

    def _retrieval_query_for_turn(self, *, message: str, conversation_state: ConversationState) -> str:
        cleaned = re.sub(r"\s+", " ", message.strip())

        if not cleaned:
            return ""
        prior_grounded_query = conversation_state.last_grounded_query or conversation_state.last_user_issue

        if (
            conversation_state.conversation_started
            and conversation_state.active_intent == "indiankanoon_rag"
            and conversation_state.awaiting_details
            and prior_grounded_query
        ):
            return self._merge_text(prior_grounded_query, cleaned)

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
                    "source_kind": "user_upload",
                    "authority_type": "user_upload",
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
        evidence_packet: dict[str, Any],
    ) -> str:
        citation_block = "\n".join(f"- {item}" for item in citations[:6])
        packet_json = json.dumps(evidence_packet, ensure_ascii=True, indent=2)

        return (
            "Answer the user's Indian legal query using only the grounded context below.\n"
            "Use only the grounded sources included in the context, such as the local legal dataset, India Kanoon, and trusted Google authority material.\n"
            "Do not invent facts, legal rules, procedures, deadlines, or authorities.\n"
            "Do not provide direct legal advice or claim to act as a lawyer; give general legal information only.\n"
            "Use a clear, human, lawyer-like Indian tone. Do not sound robotic or textbook-like.\n"
            "Acknowledge the user's latest message specifically before giving guidance.\n"
            "Do not reuse the same opening, empathy line, or action wording across consecutive replies when the facts have changed.\n"
            "Keep the structure consistent, but make the wording sound natural rather than template-like.\n"
            "If the situation is urgent, use direct time-sensitive wording. For fraud or fast-moving loss, explain that reporting within 24 hours can improve recovery chances when the grounded context supports that urgency.\n"
            "If the context is weak, conflicting, or incomplete, say so explicitly and keep the answer cautious.\n"
            "Never dump raw or noisy text. Remove AIR, SCC, PDF fragments, debug text, and broken snippets from the answer.\n"
            "Always mention the source you relied on in a clean Source line.\n"
            "If uploaded user documents are present, explicitly use their facts and mention them in the final answer.\n"
            "For law or provision queries, prefer this answer shape in the answer text: 1. What it is, 2. Meaning, 3. Key points / elements, 4. Punishment, 5. Practical use, 6. Source.\n"
            "For scenario queries, give practical steps, relevant sections if grounded, where to complain, documents to keep, and source.\n"
            "Use the evidence packet to understand source sufficiency, jurisdiction notes, recency notes, disclaimer mode, and urgency.\n"
            "If the evidence packet says the sources are partial or dated, say that clearly in the answer.\n"
            "Return strict JSON with keys: answer, follow_up_question, likely_forum, caution, documents_to_keep.\n\n"
            f"User query: {query}\n"
            f"Detected domain: {domain}\n"
            f"State context: {state or 'Unknown'}\n\n"
            f"Grounded citations:\n{citation_block or '- None'}\n\n"
            f"Structured evidence packet:\n{packet_json}\n\n"
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

            if doc.get("authority_type"):
                lines.append(f"Authority type: {doc.get('authority_type')}")

            if doc.get("jurisdiction"):
                lines.append(f"Jurisdiction: {doc.get('jurisdiction')}")

            if doc.get("recency_bucket"):
                lines.append(f"Recency bucket: {doc.get('recency_bucket')}")

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
        curated_google_docs = [
            doc

            for doc in live_docs
            if str(doc.get("source_kind") or "").strip().lower() == "google_custom_search"
            and str(doc.get("authority_type") or "").strip().lower() in {"government_portal", "regulator", "court_portal"}
        ]

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

        if curated_google_docs and self._query_prefers_curated_google_primary_authority(
            query=query,
            answer_mode=answer_mode,
        ):
            ranked_google = sorted(curated_google_docs, key=lambda item: float(item.get("score") or 0.0), reverse=True)
            selected_google = _dedupe_and_limit(ranked_google, 2)

            if selected_google:
                return selected_google[:2]

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

    @staticmethod
    def _query_prefers_curated_google_primary_authority(*, query: str, answer_mode: str) -> bool:
        normalized = query.lower()
        case_law_markers = {
            "judgment",
            "judgement",
            "case law",
            "precedent",
            "citation",
            "ratio",
            "ruling",
            "supreme court",
            "high court",
            "latest judgment",
            "latest case law",
            "interpretation",
            "legal position",
        }
        if any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in case_law_markers):
            return False
        if bool(re.search(r"\barticle\s+\d+[a-z]?\b", normalized)):
            return True
        if answer_mode in {"statute_first", "case_first", "grounded"} and bool(
            re.search(r"\bsection\s+\d+[a-z]?\b", normalized)
        ):
            return True
        return any(
            marker in normalized
            for marker in {
                "constitution",
                "fundamental rights",
                "fundamental duties",
                "directive principles",
                "dpsp",
            }
        )

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
            if upload_note.lower() in answer.lower():
                return answer
            numbered_source = re.search(r"(\n\s*6\.\s*Source:)", answer, flags=re.IGNORECASE)
            if numbered_source:
                return f"{answer[:numbered_source.start()]} {upload_note}{answer[numbered_source.start():]}"
            source_line = re.search(r"(\n\s*Source:)", answer, flags=re.IGNORECASE)
            if source_line:
                return f"{answer[:source_line.start()]} {upload_note}{answer[source_line.start():]}"
            return f"{answer.rstrip()}\n\n{upload_note}"
        practical_steps = match.group(2).strip()
        practical_lower = practical_steps.lower()
        if "uploaded document" in practical_lower or "uploaded file" in practical_lower or "user upload" in practical_lower:
            return answer
        updated_steps = f"{practical_steps} {upload_note}".strip()
        return f"{answer[:match.start(2)]}{updated_steps}{answer[match.end(2):]}"

    def _ensure_scope_notes_reflected(self, answer: str, evidence_packet: dict[str, Any]) -> str:
        notes = [str(evidence_packet.get("jurisdiction_note") or "").strip(), str(evidence_packet.get("recency_note") or "").strip()]
        scope_notes = [note for note in notes if note]
        if not scope_notes:
            return answer
        pattern = re.compile(r"(Legal Position:\s*)(.*?)(\nPractical Next Steps:)", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(answer)
        if not match:
            return answer
        legal_position = match.group(2).strip()
        merged_notes = " ".join(scope_notes)
        if merged_notes.lower() in legal_position.lower():
            return answer
        updated = f"{legal_position} {merged_notes}".strip()
        return f"{answer[:match.start(2)]}{updated}{answer[match.end(2):]}"

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
        response_mode: str = "research",
    ) -> dict[str, Any]:
        if str(response_mode).strip().lower() == "authority":
            return self._build_grounded_authority_payload(
                query=query,
                documents=documents,
                citations=citations,
            ) or {
                "answer": "",
                "follow_up_question": None,
                "likely_forum": None,
                "caution": None,
                "documents_to_keep": [],
                "source": "deterministic_grounded_authority",
                "authority_match_level": "none",
                "llm_failed": False,
            }
        if self._is_statute_query(query):
            primary_statute = self._select_primary_statute_document(query=query, documents=documents)
        else:
            primary_statute = None
        answer = self._grounded_structured_answer(
            query=query,
            domain=domain,
            documents=documents,
            citations=citations,
            primary_statute=primary_statute,
        )
        likely_forum = str(primary_statute.get("docsource") or "").strip() if primary_statute else self._infer_likely_forum(documents)
        return {
            "answer": answer,
            "follow_up_question": None,
            "likely_forum": likely_forum,
            "caution": self._grounded_caution(domain),
            "documents_to_keep": self._grounded_documents_to_keep(domain),
            "source": "deterministic_grounded_fallback",
            "llm_failed": False,
        }

    @staticmethod
    def _is_deterministic_grounded_payload(payload: dict[str, Any]) -> bool:
        return str(payload.get("source") or "").strip().lower() in {
            "deterministic_grounded_fallback",
            "deterministic_grounded_authority",
        }

    def _has_trusted_curated_google_authority_context(
        self,
        *,
        documents: list[dict[str, Any]],
        query: str,
        answer_mode: str,
    ) -> bool:
        if not self._query_prefers_curated_google_primary_authority(query=query, answer_mode=answer_mode):
            return False
        authoritative_docs = [
            doc
            for doc in documents
            if str(doc.get("source_kind") or "").strip().lower() == "google_custom_search"
            and (
                str(doc.get("authority_type") or "").strip().lower() in {"government_portal", "regulator", "court_portal"}
                or bool(doc.get("trusted_domain_match"))
                or str(doc.get("source_domain") or "").strip().lower().endswith(("gov.in", "nic.in"))
                or "indiacode.nic.in" in str(doc.get("docsource") or "").strip().lower()
            )
        ]
        if not authoritative_docs:
            return False
        return any(
            self._curated_google_authority_variant_document_relevant(
                authority_query=query,
                document=doc,
            )
            or self._looks_like_relaxed_authority_match_for_response(query=query, document=doc)
            for doc in authoritative_docs
        ) or bool(authoritative_docs)

    def _normalize_llm_payload(
        self,
        *,
        payload: dict[str, Any],
        query: str,
        domain: str,
        documents: list[dict[str, Any]],
        citations: list[str],
        response_mode: str = "research",
    ) -> dict[str, Any]:
        if str(response_mode).strip().lower() == "authority":
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
                response_mode=response_mode,
            )
        if not isinstance(payload, dict):
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
                response_mode=response_mode,
            )

        answer = str(payload.get("answer") or "").strip()
        if not answer:
            return self._build_structured_grounded_payload(
                query=query,
                domain=domain,
                documents=documents,
                citations=citations,
                response_mode=response_mode,
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
        primary_statute: dict[str, Any] | None = None,
    ) -> str:
        if not documents:
            return "No relevant legal authority material was retrieved for this query."
        if primary_statute is not None:
            return self._grounded_statute_answer(
                query=query,
                domain=domain,
                statute_document=primary_statute,
                citations=citations,
            )

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
        legal_position = self._clean_search_snippet(legal_excerpt) if legal_excerpt else "The retrieved authority materials indicate the closest grounded legal position available for this query."

        next_steps = self._grounded_next_steps(query=query, domain=domain, documents=documents)
        sources_line = "; ".join(citations[:3]) if citations else "Retrieved authority materials."
        return self._format_final_answer(
            summary=summary,
            legal_position=legal_position,
            practical_next_steps=next_steps,
            sources=sources_line,
            disclaimer="This is general legal information based on retrieved authority material, not a substitute for professional legal advice.",
        )

    def _select_primary_statute_document(self, *, query: str, documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        best_match = self._select_best_authority_document_match(query=query, documents=documents)
        if best_match is not None:
            return best_match["document"]
        if not documents:
            return None
        ranked = sorted(documents, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return ranked[0]

    @staticmethod
    def _is_deterministic_authority_source(doc: dict[str, Any]) -> bool:
        source = str(doc.get("docsource") or "").strip().lower()
        source_kind = str(doc.get("source_kind") or "").strip().lower()
        authority_type = str(doc.get("authority_type") or "").strip().lower()
        if source in {"laws", "constitution"}:
            return True
        if source_kind == "local_legal_dataset":
            return True
        if source_kind == "google_custom_search" and authority_type in {"government_portal", "regulator", "court_portal"}:
            return True
        if source.startswith("google:") and any(domain in source for domain in {"indiacode.nic.in", "gov.in", "nic.in"}):
            return True
        return False

    def _grounded_statute_answer(
        self,
        *,
        query: str,
        domain: str,
        statute_document: dict[str, Any],
        citations: list[str],
    ) -> str:
        title = self._clean_search_snippet(str(statute_document.get("title") or "the retrieved statutory provision"))
        excerpt = str(
            statute_document.get("fragment_excerpt")
            or statute_document.get("doc_excerpt")
            or statute_document.get("fragment_headline")
            or statute_document.get("headline")
            or ""
        ).strip()
        legal_position = self._clean_search_snippet(excerpt)
        if not legal_position:
            legal_position = (
                f"The retrieved bare-law authority points to {title} as the closest statutory match for this query."
            )
        summary = f"The strongest retrieved statutory match for your query is {title}."
        next_steps = self._grounded_statute_next_steps(domain=domain, title=title)
        sources_line = "; ".join(citations[:3]) if citations else f"{title} | laws"
        return self._format_final_answer(
            summary=summary,
            legal_position=legal_position,
            practical_next_steps=next_steps,
            sources=sources_line,
            disclaimer=self._disclaimer_text("low_risk" if domain not in {"criminal", "procedure"} else "medium_risk"),
        )

    @staticmethod
    def _grounded_statute_next_steps(*, domain: str, title: str) -> str:
        if domain == "tax":
            return (
                f"Read {title} with the linked source, match it against the relevant invoices, returns, or eligibility conditions, "
                "and then check any latest amendment or interpreting judgment before taking a tax or compliance step."
            )
        if domain == "criminal":
            return (
                f"Read {title} with the linked source, use it as the starting statutory position, "
                "and then check the latest judicial interpretation before relying on it for any complaint, charge, or defence step."
            )
        if domain == "procedure":
            return (
                f"Read {title} with the linked source, confirm the exact procedural stage it applies to, "
                "and then check any later procedural ruling before using it for filing or escalation."
            )
        if domain == "corporate":
            return (
                f"Read {title} with the linked source, compare it with the company records or filing requirement involved, "
                "and then check any latest amendment or interpretation before taking a compliance step."
            )
        return (
            f"Read {title} with the linked source, use that provision as the starting legal basis, "
            "and then check any latest amendment or interpreting judgment before relying on it for any legal, compliance, or procedural step."
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
        if domain == "property":
            return (
                f"Read the strongest retrieved authority from {top_authority}, compare the holding with the tenancy facts, notice stage, and dates involved, "
                "and then check whether there is any newer High Court or Supreme Court authority before relying on it."
            )
        if domain == "tax":
            return (
                f"Read the strongest retrieved authority from {top_authority}, match it against the transaction, return, or compliance record involved, "
                "and then check whether any later authority changes the position before acting on it."
            )
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

    @staticmethod
    def _authority_statute_aliases() -> list[tuple[tuple[str, ...], str]]:
        return [
            (("bns", "bharatiya nyaya sanhita"), "bharatiya nyaya sanhita"),
            (("bnss", "bharatiya nagarik suraksha sanhita"), "bharatiya nagarik suraksha sanhita"),
            (("ipc", "indian penal code"), "indian penal code"),
            (("crpc", "code of criminal procedure"), "code of criminal procedure"),
            (("cpc", "code of civil procedure"), "code of civil procedure"),
            (("ni act", "negotiable instruments act"), "negotiable instruments act"),
        ]

    @staticmethod
    def _document_authority_text(doc: dict[str, Any]) -> str:
        return " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("headline") or ""),
                str(doc.get("fragment_title") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
                str(doc.get("citations") or ""),
            ]
        ).lower()

    def _document_supports_requested_authority(self, *, doc: dict[str, Any], query: str) -> bool:
        query_lower = query.lower()
        text = self._document_authority_text(doc)
        if not text.strip():
            return False
        reference = self._parse_authority_reference(query_lower)
        kind = str(reference["kind"])
        identifier = str(reference["identifier"] or "").lower()
        statute_aliases = tuple(str(alias).lower() for alias in reference["statute_aliases"])
        if kind == "article" and identifier:
            if re.search(rf"\b(?:without|not|no)\b[^.:\n]{{0,40}}\barticle\s+{re.escape(identifier)}\b", text):
                return False
            article_present = f"article {identifier}" in text or (
                "constitution" in text and re.search(rf"\b{re.escape(identifier)}\b", text) is not None
            )
            if not article_present:
                return False
            if statute_aliases:
                return any(alias in text for alias in statute_aliases)
            return True
        if kind not in {"section", "rule"} or not identifier:
            return False
        if re.search(rf"\b(?:without|not|no)\b[^.:\n]{{0,40}}\b{kind}\s+{re.escape(identifier)}\b", text):
            return False
        label_present = f"{kind} {identifier}" in text or re.search(rf"\b{re.escape(identifier)}\b", text) is not None
        if not label_present:
            return False
        if statute_aliases:
            return any(alias in text for alias in statute_aliases)
        return f"{kind} {identifier}" in text

    def _document_matches_query_strategy(self, *, doc: dict[str, Any], query: str, answer_mode: str) -> bool:
        query_lower = query.lower()
        title = str(doc.get("title") or "").lower()
        source = str(doc.get("docsource") or "").strip().lower()
        source_kind = str(doc.get("source_kind") or "").strip().lower()
        authority_type = str(doc.get("authority_type") or "").strip().lower()
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
        article_match = re.search(r"\barticle\s+([0-9]+[a-z]?)\b", query_lower)
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query_lower)
        rule_match = re.search(r"\brule\s+([0-9]+[a-z]?)\b", query_lower)
        authoritative_curated_google = (
            source_kind == "google_custom_search"
            and authority_type in {"government_portal", "regulator", "court_portal"}
        )
        trusted_google_fallback = (
            source_kind == "google_custom_search"
            and str(doc.get("retrieval_source") or "").strip().lower() == "google"
            and str(doc.get("retrieval_confidence_level") or "").strip().lower() in {"medium", "strong"}
            and (
                authoritative_curated_google
                or bool(doc.get("trusted_domain_match"))
                or str(doc.get("source_domain") or "").strip().lower().endswith(("gov.in", "nic.in"))
                or "indiacode.nic.in" in source
            )
        )
        if trusted_google_fallback:
            return True
        if self._document_supports_requested_authority(doc=doc, query=query_lower):
            return True
        if authoritative_curated_google:
            if article_match and f"article {article_match.group(1)}" in text:
                return True
            if section_match and f"section {section_match.group(1)}" in text and meaningful_overlap >= 1:
                return True
            if rule_match and f"rule {rule_match.group(1)}" in text and meaningful_overlap >= 1:
                return True
            if self._query_prefers_curated_google_primary_authority(query=query, answer_mode=answer_mode):
                constitutional_markers = {
                    "constitution",
                    "fundamental rights",
                    "fundamental duties",
                    "directive principles",
                    "dpsp",
                }
                if meaningful_overlap >= 1 or any(marker in text for marker in constitutional_markers):
                    return True

        if source_kind == "internal":
            if answer_mode == "statute_first":
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
        if self._document_supports_requested_authority(doc=doc, query=query):
            return True
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
        source_kind = str(top.get("source_kind") or "").strip().lower()
        authority_type = str(top.get("authority_type") or "").strip().lower()
        normalized_query = query.lower()
        confidence = 0.35
        if answer_mode == "statute_first" and source == "laws":
            confidence += 0.35
        elif source_kind == "google_custom_search" and authority_type in {"government_portal", "regulator", "court_portal"}:
            confidence += 0.3
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
        upstream_confidence = top.get("retrieval_confidence")
        if upstream_confidence is not None:
            try:
                confidence = max(confidence, float(upstream_confidence))
            except (TypeError, ValueError):
                pass
        return min(confidence, 0.95)

    @staticmethod
    def _classify_retrieval_confidence_level(
        *,
        confidence: float,
        documents: list[dict[str, Any]],
        answer_mode: str,
    ) -> str:
        if not documents:
            return "low"
        top_source = str(documents[0].get("docsource") or "").strip().lower()
        high_threshold = 0.78
        medium_threshold = 0.6
        if answer_mode == "statute_first" and top_source == "laws":
            high_threshold = 0.72
        if confidence >= high_threshold:
            return "high"
        if confidence >= medium_threshold:
            return "medium"
        return "low"

    def _validate_final_output(
        self,
        *,
        answer: str,
        query: str,
        strategy: dict[str, str | bool],
        citations: list[str] | None = None,
        authorities: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        cleaned = answer or ""
        flags: list[str] = []
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
        if cleaned != (answer or ""):
            flags.append("filler_stripped")
        cleaned = self._strip_unsupported_criminal_references(
            answer=cleaned,
            query=query,
            allow_criminal_sections=bool(strategy.get("allow_criminal_sections")),
        )
        certainty_cleaned = self._strip_unsupported_certainty_claims(cleaned)
        if certainty_cleaned != cleaned:
            flags.append("certainty_claims_stripped")
        cleaned = certainty_cleaned
        deadline_cleaned = self._strip_unsupported_deadline_claims(cleaned, answer_mode=str(strategy.get("answer_mode") or ""))
        if deadline_cleaned != cleaned:
            flags.append("deadline_claims_stripped")
        cleaned = deadline_cleaned
        section_cleaned = self._strip_unsupported_reference_lines(
            cleaned,
            query=query,
            citations=citations or [],
            authorities=authorities or [],
        )
        if section_cleaned != cleaned:
            flags.append("unsupported_references_stripped")
        cleaned = section_cleaned
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if "Disclaimer:" not in cleaned:
            cleaned = self._ensure_disclaimer_present(cleaned)
            flags.append("disclaimer_added")
        if self._output_requires_safe_fallback(cleaned):
            flags.append("unsupported_output_fallback")
            fallback = self._build_safe_fallback_result(
                kind="unsupported_output",
                domain="civil",
                warnings=[],
                query=query,
            )
            return fallback.answer, flags
        return cleaned, flags

    @staticmethod
    def _should_use_strict_grounded_validation(
        *,
        strategy: dict[str, str | bool],
        raw_json: dict[str, Any] | None,
    ) -> bool:
        answer_mode = str(strategy.get("answer_mode") or "").strip().lower()
        payload = raw_json or {}
        source = str(payload.get("source") or "").strip().lower()
        if source in {
            "authority_fast_path",
            "authority_mixed_direct",
            "constitutional_explainer",
            "constitutional_mixed_direct",
            "curated_google_authority_lookup",
            "curated_google_authority_clarification",
            "deterministic_grounded_authority",
            "general_legal_explainer",
            "direct_answer_clarification",
            "local_legal_dataset",
            "safe_fallback",
        }:
            return False
        if answer_mode in {"statute_first", "case_first"}:
            return True
        query_profile = payload.get("query_profile") or {}
        query_type = str(query_profile.get("query_type") or "").strip().lower()
        route_target = str(query_profile.get("route_target") or "").strip().lower()
        if source in {"legal_help", "legal_help_interview"}:
            return False
        if answer_mode == "fact_guidance":
            return False
        if query_type in {"incident", "procedure", "document_review", "legal_help"}:
            return False
        if route_target == "playbook":
            return False
        return False

    def _should_prefer_practical_guidance(
        self,
        *,
        query_profile: dict[str, Any],
        strategy: dict[str, str | bool],
        message: str,
        domain: str,
    ) -> bool:
        answer_mode = str(strategy.get("answer_mode") or "").strip().lower()
        query_type = str(query_profile.get("query_type") or "").strip().lower()
        route_target = str(query_profile.get("route_target") or "").strip().lower()
        if answer_mode == "fact_guidance":
            return True
        if query_type in {"incident", "procedure", "document_review", "legal_help"}:
            return True
        if route_target == "playbook":
            return True
        return self._should_fallback_to_legal_intake(message=message, domain=domain)

    @staticmethod
    def _strip_unsupported_certainty_claims(answer: str) -> str:
        patterns = [
            r"\byou will definitely\b",
            r"\byou will certainly\b",
            r"\bit is guaranteed\b",
            r"\bguaranteed\b",
            r"\bassured\b",
            r"\bcertainly win\b",
            r"\bwill win\b",
            r"\bmust be granted\b",
        ]
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, "may", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    @staticmethod
    def _strip_unsupported_deadline_claims(answer: str, *, answer_mode: str) -> str:
        if answer_mode == "fact_guidance":
            return answer
        deadline_pattern = re.compile(
            r"\b(?:within|in)\s+\d+\s+(?:hour|hours|day|days|week|weeks|month|months)\b[^.\n]*",
            flags=re.IGNORECASE,
        )
        return deadline_pattern.sub("", answer)

    def _strip_unsupported_reference_lines(
        self,
        answer: str,
        *,
        query: str,
        citations: list[str],
        authorities: list[str],
    ) -> str:
        allowed_text = " ".join([query, *citations, *authorities]).lower()
        cleaned_lines: list[str] = []
        for line in answer.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            if normalized.startswith(
                (
                    "Legal Position:",
                    "Practical Next Steps:",
                    "Summary:",
                    "1. What it is:",
                    "2. Meaning:",
                    "3. Key points / elements:",
                    "4. Punishment:",
                    "5. Practical use:",
                )
            ):
                if self._line_has_unsupported_reference(normalized, allowed_text=allowed_text):
                    continue
            cleaned_lines.append(normalized)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _line_has_unsupported_reference(line: str, *, allowed_text: str) -> bool:
        section_refs = re.findall(r"\bsection\s+([0-9]+[a-z]?)\b", line, flags=re.IGNORECASE)
        for section in section_refs:
            if f"section {section.lower()}" not in allowed_text and section.lower() not in allowed_text:
                return True
        citation_markers = re.findall(r"\b(?:AIR|SCC|CriLJ|SCR)\b", line, flags=re.IGNORECASE)
        if citation_markers and not any(marker.lower() in allowed_text for marker in citation_markers):
            return True
        return False

    @staticmethod
    def _ensure_disclaimer_present(answer: str) -> str:
        cleaned = answer.strip()
        disclaimer = "Note: This is general legal information, not a substitute for professional legal advice."
        if re.search(r"^(?:Disclaimer|Note):.*$", cleaned, flags=re.IGNORECASE | re.MULTILINE):
            return re.sub(r"^Disclaimer:", "Note:", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        if not cleaned:
            return disclaimer
        return f"{cleaned}\n{disclaimer}"

    def _apply_disclaimer_mode(self, *, answer: str, mode: str) -> str:
        cleaned = self._ensure_disclaimer_present(answer)
        replacement = f"Note: {self._disclaimer_text(mode)}"
        if re.search(r"^(?:Disclaimer|Note):.*$", cleaned, flags=re.IGNORECASE | re.MULTILINE):
            return re.sub(r"^(?:Disclaimer|Note):.*$", replacement, cleaned, flags=re.IGNORECASE | re.MULTILINE)
        return f"{cleaned}\n{replacement}"

    @staticmethod
    def _output_requires_safe_fallback(answer: str) -> bool:
        has_source = bool(re.search(r"^(?:6\.\s*)?Source:\s*", answer, flags=re.IGNORECASE | re.MULTILINE))
        has_note = bool(re.search(r"^(?:Disclaimer|Note):\s*", answer, flags=re.IGNORECASE | re.MULTILINE))
        has_new_shape = all(
            re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
            for pattern in [
                r"^1\.\s*What it is:\s*",
                r"^2\.\s*Meaning:\s*",
                r"^4\.\s*Punishment:\s*",
                r"^5\.\s*Practical use:\s*",
            ]
        )
        has_old_shape = all(section in answer for section in ["Summary:", "Legal Position:", "Practical Next Steps:", "Sources:"])
        return not (has_source and has_note and (has_new_shape or has_old_shape))

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
        normalized_query = str(query or "").lower()
        article_match = re.search(r"\barticle\s+([0-9]+[a-z]?)\b", query, re.IGNORECASE)
        if article_match and f"article {article_match.group(1).lower()}" in normalized_title:
            return True
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query, re.IGNORECASE)
        if section_match and f"section {section_match.group(1)}" in normalized_title:
            if "bnss" in normalized_query and "bharatiya nagarik suraksha sanhita" not in normalized_title:
                return False
            if "bns" in normalized_query and "bharatiya nyaya sanhita" not in normalized_title:
                return False
            return True
        if "ipc" in query and "indian penal code" in normalized_title:
            return True
        if "ni act" in query and "negotiable instruments act" in normalized_title:
            return True
        if "constitution" in query and "constitution" in normalized_title:
            return True
        if (
            ("bns" in normalized_query or "bharatiya nyaya sanhita" in normalized_query)
            and "bharatiya nyaya sanhita" in normalized_title
            and section_match is None
        ):
            return True
        if (
            ("bnss" in normalized_query or "bharatiya nagarik suraksha sanhita" in normalized_query)
            and "bharatiya nagarik suraksha sanhita" in normalized_title
            and section_match is None
        ):
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
            "Key Points": "",
            "Punishment": "",
            "Practical Next Steps": "",
            "Sources": "",
            "Disclaimer": "",
        }
        current: str | None = None
        alias_map = {
            "summary": "Summary",
            "what it is": "Summary",
            "legal position": "Legal Position",
            "legal position ": "Legal Position",
            "legal_position": "Legal Position",
            "legal": "Legal Position",
            "meaning": "Legal Position",
            "key points / elements": "Key Points",
            "key points": "Key Points",
            "elements": "Key Points",
            "punishment": "Punishment",
            "practical next steps": "Practical Next Steps",
            "practical next step": "Practical Next Steps",
            "next steps": "Practical Next Steps",
            "practical": "Practical Next Steps",
            "practical use": "Practical Next Steps",
            "sources": "Sources",
            "source": "Sources",
            "disclaimer": "Disclaimer",
            "note": "Disclaimer",
        }

        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_for_match = re.sub(r"^\d+\.\s*", "", line).strip()
            matched = False
            for alias, canonical in alias_map.items():
                prefix = f"{alias}:"
                if line_for_match.lower().startswith(prefix):
                    value = line_for_match[len(prefix):].strip()
                    if value:
                        sections[canonical] = self._merge_section_value(sections[canonical], value)
                    current = canonical
                    matched = True
                    break
            if matched:
                continue
            if current:
                if current == "Key Points":
                    sections[current] = f"{sections[current]}\n{line}".strip() if sections[current] else line
                else:
                    sections[current] = self._merge_section_value(sections[current], line)

        if not sections["Summary"]:
            sections["Summary"] = self._first_meaningful_sentence(cleaned) or "I was able to retrieve relevant legal material for your question."
        if not sections["Legal Position"]:
            sections["Legal Position"] = sections["Summary"]
        if not sections["Practical Next Steps"]:
            sections["Practical Next Steps"] = "Review the strongest cited authority and then match it carefully against your facts before taking the next legal step."
        if citations:
            sections["Sources"] = "; ".join(citations[:3])
        elif not sections["Sources"]:
            sections["Sources"] = "Indian Kanoon retrieved authorities."
        if not sections["Disclaimer"]:
            sections["Disclaimer"] = self._default_brief_disclaimer()

        return self._format_final_answer(
            summary=sections["Summary"],
            legal_position=sections["Legal Position"],
            key_points=sections["Key Points"],
            punishment=sections["Punishment"],
            practical_next_steps=sections["Practical Next Steps"],
            sources=sections["Sources"],
            disclaimer=sections["Disclaimer"],
        )

    def _format_final_answer(
        self,
        *,
        summary: str,
        legal_position: str,
        key_points: str | list[str] | None = None,
        punishment: str | None = None,
        practical_next_steps: str,
        sources: str,
        disclaimer: str,
    ) -> str:
        cleaned_summary = self._sanitize_section_value(summary, fallback="I was able to retrieve relevant legal material for your question.")
        cleaned_legal_position = self._sanitize_section_value(
            legal_position,
            fallback="The retrieved authorities provide the closest grounded legal position available on the material currently in hand.",
        )
        cleaned_steps = self._sanitize_section_value(
            practical_next_steps,
            fallback="Review the strongest cited authority and then match it carefully against your facts before taking the next legal step.",
        )
        cleaned_disclaimer = self._sanitize_section_value(
            disclaimer,
            fallback=self._default_brief_disclaimer(),
        )
        derived_key_points = key_points if key_points else self._derive_key_points(cleaned_summary, cleaned_legal_position)
        derived_punishment = punishment or self._derive_punishment_text(cleaned_summary, cleaned_legal_position)
        return self._format_numbered_legal_answer(
            what_it_is=cleaned_summary,
            meaning=cleaned_legal_position,
            key_points=derived_key_points,
            punishment=derived_punishment,
            practical_use=cleaned_steps,
            source=sources,
            disclaimer=cleaned_disclaimer,
        )

    @staticmethod
    def _default_brief_disclaimer() -> str:
        return "This is general legal information, not a substitute for advice on your specific facts."

    @staticmethod
    def _sanitize_sources(text: str) -> str:
        cleaned = ChatService._sanitize_section_value(text)
        raw_parts = [part.strip(" ;,") for part in re.split(r"[;\n]+", cleaned) if part.strip(" ;,")]
        parts = [ChatService._clean_visible_authority_source(part) for part in raw_parts]
        deduped: list[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return "; ".join(deduped[:3]) if deduped else "Indian Kanoon retrieved authorities."

    @staticmethod
    def _sanitize_section_value(text: str, fallback: str = "Not available from the retrieved material.") -> str:
        cleaned = ChatService._sanitize_text_block(text)
        cleaned = re.sub(r"\b(Document \d+|Search snippet|Relevant fragment|Excerpt|Title|Authority|Date|URL|Citations)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(errmsg|error|debug|traceback|stack trace|request_id|permission_denied|insufficient_quota|provider)\b\s*:?\s*[^.;]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
        return cleaned or fallback

    @staticmethod
    def _sanitize_text_block(text: str) -> str:
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
        uploaded_texts: list[str] | None = None,
    ) -> dict[str, str | bool]:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        flow = self._classify_query_flow(
            message=message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts or [],
        )
        routing = self._classify_routing_precedence(normalized, domain=domain)
        flow_type = str(flow["flow_type"])
        if flow_type == "provision_lookup":
            answer_mode = "statute_first"
            intent = "authority_lookup"
        elif flow_type == "general_legal_research" and self._looks_like_grounded_authority_query(normalized):
            answer_mode = "case_first"
            intent = "authority_lookup"
        elif flow_type == "general_legal_research":
            answer_mode = "grounded_general"
            intent = "authority_lookup"
        elif flow_type == "uploaded_document_query":
            answer_mode = "grounded_general"
            intent = "uploaded_document_analysis"
        elif conversation_state.active_intent == "indiankanoon_rag":
            answer_mode = "grounded_general"
            intent = "authority_lookup"
        elif flow_type == "scenario_practical_legal_issue" or routing["prefer_playbook"]:
            answer_mode = "fact_guidance"
            intent = "practical_guidance"
        else:
            answer_mode = "grounded_general"
            intent = "authority_lookup"
        response_mode = self._response_mode_for_flow(flow_type)

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
            "flow_type": flow_type,
            "response_mode": response_mode,
        }

    def _build_query_profile(
        self,
        *,
        message: str,
        domain: str,
        conversation_state: ConversationState,
        strategy: dict[str, str | bool],
        uploaded_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        issue_type = conversation_state.issue_type or self._detect_issue_type(normalized, domain)
        routing = self._classify_routing_precedence(normalized, domain=domain)
        flow = self._classify_query_flow(
            message=message,
            domain=domain,
            conversation_state=conversation_state,
            uploaded_texts=uploaded_texts or [],
        )
        flow_type = str(flow["flow_type"])
        has_authority_signal = routing["has_authority_signal"]
        has_practical_signal = routing["prefer_playbook"] or issue_type != "general"
        if flow_type == "scenario_practical_legal_issue":
            query_type = "incident" if issue_type != "general" else "procedure"
        elif flow_type == "uploaded_document_query":
            query_type = "document_review"
        elif flow_type == "provision_lookup":
            query_type = "statute_lookup"
        elif flow_type == "general_legal_research" and self._looks_like_grounded_authority_query(normalized):
            query_type = "case_law"
        elif has_authority_signal and has_practical_signal:
            query_type = "mixed"
        else:
            query_type = "grounded_general"

        if flow_type == "scenario_practical_legal_issue" or routing["prefer_playbook"]:
            route_target = "playbook"
        else:
            route_target = "grounded"

        urgent_markers = {
            "urgent",
            "immediately",
            "today",
            "deadline",
            "arrest",
            "threat",
            "fake link",
            "fraud",
            "debit",
            "snatched",
            "stolen",
            "lockout",
            "notice",
        }
        if any(token in normalized for token in urgent_markers) or issue_type in {"cyber_fraud", "snatching_theft", "fir_refusal"}:
            urgency = "high"
        elif query_type in {"procedure", "incident", "document_review", "mixed"}:
            urgency = "medium"
        else:
            urgency = "low"

        official_source_needed = bool(
            any(token in normalized for token in {"latest", "recent", "current", "official", "portal", "procedure", "where to file", "which authority"})
            or query_type in {"procedure", "document_review"}
        )
        route_confidence = "high" if query_type != "mixed" else "medium"
        route_assist = {"label": "none", "confidence": 0.0, "scores": {}, "backend": "skipped"}
        if query_type in {"grounded_general", "mixed"}:
            route_assist = self._local_route_assist(normalized)
            if route_assist["label"] != "none" and route_assist["confidence"] >= 0.62:
                route_target = route_assist["label"]
                query_type = "procedure" if route_target == "playbook" else query_type
                route_confidence = "high" if route_assist["confidence"] >= 0.75 else "medium"
        return {
            "query_type": query_type,
            "flow_type": flow_type,
            "response_mode": self._response_mode_for_flow(flow_type),
            "route_target": route_target,
            "urgency": urgency,
            "official_source_needed": official_source_needed,
            "issue_type": issue_type,
            "mixed_query": query_type == "mixed",
            "route_confidence": route_confidence,
            "answer_mode": strategy.get("answer_mode"),
            "route_assist": route_assist,
        }

    def _classify_query_flow(
        self,
        *,
        message: str,
        domain: str,
        conversation_state: ConversationState,
        uploaded_texts: list[str],
    ) -> dict[str, str | bool]:
        normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
        issue_type = conversation_state.issue_type or self._detect_issue_type(normalized, domain)
        has_uploaded_context = bool(uploaded_texts or conversation_state.uploaded_document_summaries)
        routing = self._classify_routing_precedence(normalized, domain=domain)
        if has_uploaded_context and self._is_uploaded_document_query(
            normalized=normalized,
            conversation_state=conversation_state,
            domain=domain,
        ):
            return {"flow_type": "uploaded_document_query", "has_uploaded_context": True}
        if self._is_provision_lookup_query(normalized):
            return {"flow_type": "provision_lookup", "has_uploaded_context": has_uploaded_context}
        if routing["prefer_grounded_authority"]:
            return {"flow_type": "general_legal_research", "has_uploaded_context": has_uploaded_context}
        if routing["prefer_playbook"] or issue_type != "general":
            return {"flow_type": "scenario_practical_legal_issue", "has_uploaded_context": has_uploaded_context}
        return {"flow_type": "general_legal_research", "has_uploaded_context": has_uploaded_context}

    def _is_uploaded_document_query(
        self,
        *,
        normalized: str,
        conversation_state: ConversationState,
        domain: str,
    ) -> bool:
        if not normalized:
            return False
        if domain == "document_review" or conversation_state.legal_domain == "document_review":
            return True
        uploaded_markers = {
            "uploaded",
            "upload",
            "file",
            "document",
            "agreement",
            "notice",
            "clause",
            "draft",
            "pdf",
            "attached",
        }
        return any(marker in normalized for marker in uploaded_markers)

    def _is_provision_lookup_query(self, normalized: str) -> bool:
        if not normalized:
            return False
        if self.legal_dataset.parse_query(normalized) is not None:
            return True
        if self._is_statute_query(normalized):
            return True
        return bool(
            re.search(r"\b(article|section|rule)\s+[0-9]+[a-z]?\b", normalized)
            or re.search(r"\b(constitution|ipc|bns|bnss|crpc|cpc|ni act)\b", normalized)
        )

    def _local_route_assist(self, normalized_message: str) -> dict[str, Any]:
        labels = {
            "playbook": "Practical legal help, incident response, filing steps, complaint flow, documents to keep, and immediate next steps.",
            "grounded": "Authority lookup, statute query, judgment analysis, case law position, citation search, and legal authority explanation.",
        }
        decision = self.local_similarity.score_labels(normalized_message, labels)
        return {
            "label": decision.label,
            "confidence": decision.confidence,
            "scores": decision.scores,
            "backend": decision.backend,
        }

    def _annotate_grounded_documents(
        self,
        *,
        documents: list[dict[str, Any]],
        query: str,
        domain: str,
        state: str | None,
    ) -> list[dict[str, Any]]:
        return [
            self._annotate_grounded_document(document=doc, query=query, domain=domain, state=state)
            for doc in documents
        ]

    def _annotate_grounded_document(
        self,
        *,
        document: dict[str, Any],
        query: str,
        domain: str,
        state: str | None,
    ) -> dict[str, Any]:
        enriched = dict(document)
        source = str(enriched.get("docsource") or "").strip().lower()
        jurisdiction = str(enriched.get("jurisdiction") or "").strip()
        if not jurisdiction:
            if source in {"supremecourt", "laws", "constitution"}:
                jurisdiction = "India"
            elif source and source not in {"internal", "internal:statute", "internal:judgment"}:
                jurisdiction = source
            elif state:
                jurisdiction = state
        enriched["jurisdiction"] = jurisdiction
        authority_type = str(enriched.get("authority_type") or "").strip().lower()
        if not authority_type:
            if source == "laws" or str(enriched.get("document_kind") or "").lower() in {"statute", "rule"}:
                authority_type = "statute"
            elif source in SOURCE_AUTHORITY_SCORES:
                authority_type = "case_law"
            elif str(enriched.get("source_kind") or "").lower() == "internal":
                authority_type = "internal_guidance"
            elif source == "user_upload":
                authority_type = "user_upload"
            else:
                authority_type = "general_authority"
        enriched["authority_type"] = authority_type
        publishdate = str(enriched.get("publishdate") or "").strip()
        enriched["recency_bucket"] = self._recency_bucket(publishdate)
        if "metadata_confidence" not in enriched:
            confidence = 0.55
            if enriched["jurisdiction"]:
                confidence += 0.1
            if authority_type in {"statute", "case_law"}:
                confidence += 0.15
            if publishdate:
                confidence += 0.05
            if domain and str(enriched.get("legal_domain") or "").lower() == domain.lower():
                confidence += 0.1
            enriched["metadata_confidence"] = min(confidence, 0.95)
        return enriched

    def _rerank_grounded_documents(
        self,
        *,
        documents: list[dict[str, Any]],
        query: str,
        domain: str,
        state: str | None,
        answer_mode: str,
        query_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        normalized = query.lower()
        domain_pack = self.domain_packs.pack_for(
            domain=domain,
            issue_type=str(query_profile.get("issue_type") or ""),
        )
        preferred_authority_types = {str(item).lower() for item in domain_pack.get("preferred_authority_types", [])}
        preferred_sources = {str(item).lower() for item in domain_pack.get("preferred_sources", [])}
        boost_terms = [str(item).lower() for item in domain_pack.get("boost_terms", [])]
        ranked: list[dict[str, Any]] = []
        for index, doc in enumerate(documents):
            boosted = dict(doc)
            score = float(doc.get("score") or 0.0)
            authority_type = str(doc.get("authority_type") or "").lower()
            jurisdiction = str(doc.get("jurisdiction") or "").lower()
            recency_bucket = str(doc.get("recency_bucket") or "").lower()
            legal_domain = str(doc.get("legal_domain") or "").lower()
            if answer_mode == "statute_first" and authority_type == "statute":
                score += 6.0
            if answer_mode == "case_first" and authority_type == "case_law":
                score += 5.0
            if "supreme court" in normalized and "supremecourt" in str(doc.get("docsource") or "").lower():
                score += 4.0
            elif "supreme court" in normalized and authority_type != "case_law":
                score -= 2.5
            if "high court" in normalized and any(token in str(doc.get("docsource") or "").lower() for token in {"delhi", "bombay", "allahabad", "gujarat", "karnataka", "kerala"}):
                score += 3.0
            elif "high court" in normalized and authority_type != "case_law":
                score -= 2.0
            if state and state.lower() in jurisdiction:
                score += 2.5
            if jurisdiction == "india":
                score += 1.0
            if domain and domain.lower() == legal_domain:
                score += 2.0
            if query_profile.get("official_source_needed") and authority_type in {"statute", "case_law"}:
                score += 1.5
            if query_profile.get("official_source_needed") and str(doc.get("source_kind") or "").lower() == "internal":
                score -= 0.75
            if any(token in normalized for token in {"latest", "recent", "current", "updated"}) and recency_bucket in {"current", "recent"}:
                score += 2.0
            if preferred_authority_types and authority_type in preferred_authority_types:
                score += 1.75
            if preferred_sources and any(source_key in str(doc.get("docsource") or "").lower() for source_key in preferred_sources):
                score += 1.5
            if boost_terms and any(term in normalized for term in boost_terms):
                text = " ".join(
                    [
                        str(doc.get("title") or ""),
                        str(doc.get("headline") or ""),
                        str(doc.get("fragment_headline") or ""),
                        str(doc.get("doc_excerpt") or ""),
                    ]
                ).lower()
                overlap = sum(1 for term in boost_terms if term in normalized and term in text)
                score += min(overlap * 0.6, 2.4)
            score -= index * 0.01
            boosted["rerank_score"] = round(score, 3)
            boosted["domain_pack_id"] = domain_pack.get("pack_id")
            ranked.append(boosted)
        ranked.sort(key=lambda item: float(item.get("rerank_score") or item.get("score") or 0.0), reverse=True)
        return ranked

    def _assess_source_sufficiency(
        self,
        *,
        documents: list[dict[str, Any]],
        retrieval_confidence: float,
        retrieval_confidence_level: str,
        query_profile: dict[str, Any],
        state: str | None,
    ) -> dict[str, Any]:
        if not documents:
            return {"label": "weak", "score": 0.0, "reasons": ["no_documents"], "jurisdiction_note": "", "recency_note": ""}
        score = retrieval_confidence
        reasons: list[str] = [f"confidence_{retrieval_confidence_level}"]
        authority_types = {str(doc.get("authority_type") or "").lower() for doc in documents}
        jurisdictions = {str(doc.get("jurisdiction") or "").strip() for doc in documents if str(doc.get("jurisdiction") or "").strip()}
        recency_buckets = {str(doc.get("recency_bucket") or "").lower() for doc in documents}
        if {"statute", "case_law", "government_portal", "regulator", "court_portal"} & authority_types:
            score += 0.08
            reasons.append("authoritative_source_present")
        if state and any(state.lower() in item.lower() for item in jurisdictions):
            score += 0.05
            reasons.append("jurisdiction_match")
        if query_profile.get("official_source_needed") and "dated" in recency_buckets:
            score -= 0.08
            reasons.append("dated_sources_present")
        if len(documents) >= 2:
            score += 0.04
            reasons.append("multiple_sources")
        jurisdiction_note = ""
        if state and jurisdictions and not any(state.lower() in item.lower() or item.lower() == "india" for item in jurisdictions):
            jurisdiction_note = f"The strongest retrieved sources do not appear to be specific to {state}."
            score -= 0.08
            reasons.append("jurisdiction_unclear")
        recency_note = ""
        if query_profile.get("official_source_needed") and "dated" in recency_buckets and "current" not in recency_buckets:
            recency_note = "Some retrieved material may be dated, so the current procedure or authority position should be checked carefully."
        if score >= 0.8:
            label = "strong"
        elif score >= 0.62:
            label = "partial"
        else:
            label = "weak"
        retrieval_levels = {
            str(doc.get("retrieval_confidence_level") or "").strip().lower()
            for doc in documents[:3]
        }
        if label == "weak" and "medium" in retrieval_levels:
            label = "partial"
            reasons.append("medium_confidence_source_accepted")
        if label != "strong" and "strong" in retrieval_levels:
            label = "strong"
            reasons.append("strong_confidence_source_accepted")
        return {
            "label": label,
            "score": round(max(min(score, 0.99), 0.0), 3),
            "reasons": reasons,
            "jurisdiction_note": jurisdiction_note,
            "recency_note": recency_note,
        }

    @staticmethod
    def _recency_bucket(value: str) -> str:
        text = str(value or "").strip()
        year_match = re.search(r"(19|20)\d{2}", text)
        if not year_match:
            return "unknown"
        year = int(year_match.group(0))
        if year >= 2024:
            return "current"
        if year >= 2021:
            return "recent"
        return "dated"

    def _build_evidence_packet(
        self,
        *,
        query: str,
        retrieval_query: str,
        domain: str,
        state: str | None,
        documents: list[dict[str, Any]],
        citations: list[str],
        authorities: list[str],
        query_profile: dict[str, Any],
        source_sufficiency: dict[str, Any],
    ) -> dict[str, Any]:
        top_docs = documents[:3]
        domain_pack = self.domain_packs.pack_for(
            domain=domain,
            issue_type=str(query_profile.get("issue_type") or ""),
        )
        source_cards = [
            {
                "title": str(doc.get("title") or "Untitled"),
                "authority": str(doc.get("docsource") or ""),
                "authority_type": str(doc.get("authority_type") or ""),
                "jurisdiction": str(doc.get("jurisdiction") or ""),
                "recency_bucket": str(doc.get("recency_bucket") or ""),
                "score": float(doc.get("rerank_score") or doc.get("score") or 0.0),
            }
            for doc in top_docs
        ]
        disclaimer_mode = self._select_disclaimer_mode(
            route=query_profile.get("route_target", "grounded"),
            urgency=str(query_profile.get("urgency") or "low"),
            source_sufficiency=source_sufficiency.get("label", "partial"),
        )
        return {
            "query": query,
            "retrieval_query": retrieval_query,
            "domain": domain,
            "state": state or "Unknown",
            "query_profile": query_profile,
            "domain_pack_id": domain_pack.get("pack_id"),
            "source_sufficiency": source_sufficiency,
            "disclaimer_mode": disclaimer_mode,
            "jurisdiction_note": source_sufficiency.get("jurisdiction_note") or "",
            "recency_note": source_sufficiency.get("recency_note") or "",
            "top_sources": source_cards,
            "citations": citations[:3],
            "authorities": authorities[:5],
        }

    @staticmethod
    def _select_disclaimer_mode(*, route: str, urgency: str, source_sufficiency: str) -> str:
        if urgency == "high" or source_sufficiency == "weak":
            return "high_risk"
        if route == "playbook" or source_sufficiency == "partial":
            return "medium_risk"
        return "low_risk"

    @staticmethod
    def _disclaimer_text(mode: str) -> str:
        if mode == "high_risk":
            return "This is general legal information based on the available material. Because the matter may be urgent or fact-sensitive, please verify the exact facts and take qualified local legal advice before acting."
        if mode == "medium_risk":
            return "This is general legal information and procedural guidance, not a substitute for professional legal advice tailored to your facts."
        return "This is general legal information based on the retrieved material, not a substitute for professional legal advice."

    @staticmethod
    def _fast_authority_lookup_key(message: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", str(message or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None

        alias_key = ChatService._resolve_direct_constitutional_authority_alias(normalized)
        if alias_key:
            return alias_key

        article_match = re.fullmatch(
            r"(?:(?:what is|explain|meaning of|tell me about)\s+)?(?:article|art)\s+(14|19|21|22|32|226|300a)(?:\s+(?:of|under)\s+the\s+constitution)?(?:\s+of\s+india|\s+constitution)?(?:\s+(?:in\s+short|briefly|step\s+by\s+step|step-by-step|in\s+points|point\s+wise|pointwise|as\s+points|bullet\s+points))?",
            normalized,
        )
        if article_match:
            return f"constitution_article_{article_match.group(1)}"

        section_match = re.fullmatch(
            r"(?:(?:what is|explain|meaning of|tell me about)\s+)?section\s+(138|406|420|498a)(?:\s+(?:of|under)\s+the)?\s+(ni act|negotiable instruments act|ipc|indian penal code)(?:\s+(?:in\s+short|briefly|step\s+by\s+step|step-by-step|in\s+points|point\s+wise|pointwise|as\s+points|bullet\s+points))?",
            normalized,
        )
        if not section_match:
            reverse_section_match = re.fullmatch(
                r"(?:(?:what is|explain|meaning of|tell me about)\s+)?(?:ipc|indian penal code)\s+(138|406|420|498a)(?:\s+(?:in\s+short|briefly|step\s+by\s+step|step-by-step|in\s+points|point\s+wise|pointwise|as\s+points|bullet\s+points))?",
                normalized,
            )
            if not reverse_section_match:
                return None
            section_value = reverse_section_match.group(1)
            statute = "ipc"
        else:
            section_value = section_match.group(1)
            statute = section_match.group(2)
        if section_value == "138" and statute in {"ni act", "negotiable instruments act"}:
            return "ni_act_section_138"
        if statute in {"ipc", "indian penal code"}:
            if section_value == "420":
                return "ipc_section_420"
            if section_value == "406":
                return "ipc_section_406"
            if section_value == "498a":
                return "ipc_section_498a"
        return None

    @staticmethod
    def _resolve_direct_constitutional_authority_alias(message: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", str(message or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None
        if any(
            phrase in normalized
            for phrase in {
                "constitutional remedies",
                "constitution remedies",
                "constitutional remedy",
                "constitution remedy",
                "right to constitutional remedies",
                "right to constitution remedies",
                "article for constitutional remedies",
                "article for constitution remedies",
                "article gives constitutional remedies",
                "article gives constitution remedies",
                "which article gives constitutional remedies",
                "which article gives constitution remedies",
                "which article is constitutional remedies",
                "which article is constitution remedies",
            }
        ):
            return "constitution_article_32"
        return None

    @staticmethod
    def _constitutional_explainer_key(message: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", str(message or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None
        if any(phrase in normalized for phrase in {"fundamental duties", "what are duties in constitution", "duties under constitution"}):
            return "fundamental_duties"
        if any(phrase in normalized for phrase in {"directive principles", "directive principles of state policy", "dpsp"}):
            return "directive_principles"
        if any(phrase in normalized for phrase in {"fundamental rights", "basic rights in constitution", "rights under constitution"}):
            return "fundamental_rights"
        return None

    @staticmethod
    def _general_legal_explainer_key(message: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", str(message or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None
        explainer_phrases = {
            "arbitration": {
                "what is arbitration",
                "what arbitration is",
                "explain arbitration",
                "meaning of arbitration",
                "define arbitration",
                "arbitration meaning",
            },
            "fir": {
                "what is fir",
                "what is an fir",
                "explain fir",
                "fir meaning",
                "define fir",
            },
            "bail": {
                "what is bail",
                "explain bail",
                "bail meaning",
                "define bail",
            },
            "anticipatory_bail": {
                "what is anticipatory bail",
                "explain anticipatory bail",
                "anticipatory bail meaning",
                "define anticipatory bail",
            },
            "legal_notice": {
                "what is legal notice",
                "explain legal notice",
                "legal notice meaning",
                "define legal notice",
            },
        }
        for key, phrases in explainer_phrases.items():
            if any(phrase in normalized for phrase in phrases):
                return key
        return None

    @staticmethod
    def _mixed_constitutional_components(message: str) -> list[dict[str, str]]:
        normalized = re.sub(r"\s+", " ", str(message or "").strip())
        lowered = normalized.lower()
        if not normalized:
            return []
        if any(
            token in lowered
            for token in {
                "judgment",
                "judgement",
                "case law",
                "citation",
                "latest",
                "recent",
                "punishment",
                "penalty",
                "sentence",
                "offence",
            }
        ):
            return []

        components: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for key in ChatService._extract_grouped_authority_keys(lowered):
            marker = ("authority", key)
            if marker not in seen:
                seen.add(marker)
                components.append({"kind": "authority", "key": key})
        explainer_phrases = {
            "fundamental_duties": {"fundamental duties", "what are duties in constitution", "duties under constitution"},
            "directive_principles": {"directive principles", "directive principles of state policy", "dpsp"},
            "fundamental_rights": {"fundamental rights", "basic rights in constitution", "rights under constitution"},
        }
        for explainer_key, phrases in explainer_phrases.items():
            if any(phrase in lowered for phrase in phrases):
                marker = ("explainer", explainer_key)
                if marker not in seen:
                    seen.add(marker)
                    components.append({"kind": "explainer", "key": explainer_key})
        return components if len(components) >= 2 else []

    @staticmethod
    def _extract_grouped_authority_keys(message: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(message or "").strip().lower())
        if not normalized:
            return []

        keys: list[str] = []
        seen: set[str] = set()

        def add_key(key: str) -> None:
            if key and key not in seen and key in FAST_AUTHORITY_LOOKUPS:
                seen.add(key)
                keys.append(key)

        article_numbers: list[str] = []

        def add_article_number(value: str) -> None:
            if value not in article_numbers:
                article_numbers.append(value)

        for article_number in re.findall(r"\b(?:article|art)\s+([0-9]+[a-z]?)\b", normalized):
            add_article_number(article_number)
        grouped_article_pattern = re.compile(
            r"\b(?:article|art)\s+([0-9]+[a-z]?(?:(?:\s*(?:,|and)\s*|\s+)[0-9]+[a-z]?)+)(?:\s+(?:of|under)\s+the\s+constitution)?(?:\s+of\s+india|\s+constitution)?\b"
        )
        for match in grouped_article_pattern.finditer(normalized):
            for article_number in re.findall(r"[0-9]+[a-z]?", match.group(1)):
                add_article_number(article_number)
        for article_number in article_numbers:
            add_key(f"constitution_article_{article_number}")

        section_matches = re.finditer(
            r"\bsection(?:s)?\s+([0-9]+[a-z]?(?:(?:\s*(?:,|and)\s*|\s+)[0-9]+[a-z]?)*)(?:\s+(?:of|under)\s+the)?\s+(ni act|negotiable instruments act|ipc|indian penal code)\b",
            normalized,
        )
        for match in section_matches:
            statute = match.group(2)
            for section_value in re.findall(r"[0-9]+[a-z]?", match.group(1)):
                if section_value == "138" and statute in {"ni act", "negotiable instruments act"}:
                    add_key("ni_act_section_138")
                elif statute in {"ipc", "indian penal code"}:
                    if section_value == "420":
                        add_key("ipc_section_420")
                    elif section_value == "406":
                        add_key("ipc_section_406")
                    elif section_value == "498a":
                        add_key("ipc_section_498a")

        return keys

    def _run_semantic_support_check(
        self,
        *,
        answer: str,
        query: str,
        documents: list[dict[str, Any]],
        source_sufficiency: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.settings.local_support_check_enabled or source_sufficiency.get("label") == "strong":
            return {"status": "skipped", "score": 1.0, "claim_count": 0, "supported_claims": 0, "backend": "disabled", "details": []}
        evidence_texts = [
            " ".join(
                [
                    str(doc.get("title") or ""),
                    str(doc.get("headline") or ""),
                    str(doc.get("fragment_headline") or ""),
                    str(doc.get("fragment_excerpt") or ""),
                    str(doc.get("doc_excerpt") or ""),
                ]
            ).strip()
            for doc in documents[:4]
        ]
        decision = self.semantic_support_checker.check(answer=answer, query=query, evidence_texts=evidence_texts)
        return {
            "status": decision.status,
            "score": decision.score,
            "claim_count": decision.claim_count,
            "supported_claims": decision.supported_claims,
            "backend": decision.backend,
            "details": decision.details,
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
        disclaimer_mode = "low_risk"
        formatted_answer = self._format_final_answer(
            summary=answer or "General assistance was requested.",
            legal_position="This response is general guidance and does not rely on India Kanoon authority retrieval.",
            practical_next_steps="Ask a specific legal question or share the practical issue if you need more targeted guidance.",
            sources="General guidance response.",
            disclaimer=self._disclaimer_text(disclaimer_mode),
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
            raw_json={"results": [], "source": "intent", "disclaimer_mode": disclaimer_mode},
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

        turn_classification = (
            self._classify_legal_help_follow_up_turn(
                message=message,
                state=base_state,
                collected_facts=collected_facts,
                is_continuation=is_continuation,
            )
            if is_continuation
            else {"mode": "none", "decision_type": None, "raw_question": "", "confidence": 0.0}
        )

        if self._should_treat_message_as_fact_answer(
            turn_classification=turn_classification,
            current_intake_key=conversation_state.current_intake_key,
            is_continuation=is_continuation,
        ):
            collected_facts[conversation_state.current_intake_key] = self._extract_interview_answer(
                key=conversation_state.current_intake_key,
                message=message,
                state=base_state,
            )

        plan = self._interview_plan_for_issue(base_state.issue_type or "general")
        if self._has_enough_information_for_guidance(base_state.issue_type or "general", collected_facts):
            plan = []
        next_item = self._next_missing_interview_item(plan=plan, collected_facts=collected_facts)

        if turn_classification.get("mode") == "decision_question":
            internal = self._build_follow_up_decision_result(
                turn_classification=turn_classification,
                state=base_state,
                collected_facts=collected_facts,
                next_item=next_item,
                domain=domain,
                warnings=warnings,
            )
            next_state = base_state.model_copy(
                update={
                    "conversation_started": True,
                    "active_intent": "legal_help",
                    "awaiting_details": bool(internal.follow_up_question),
                    "interview_mode": bool(internal.follow_up_question),
                    "intake_stage": int(next_item["index"]) if next_item else len(plan),
                    "last_user_issue": merged_issue,
                    "legal_domain": domain,
                    "last_follow_up_question": internal.follow_up_question,
                    "current_intake_key": str(next_item["key"]) if next_item else None,
                    "last_guidance_key": str(internal.raw_json.get("style_key") or base_state.last_guidance_key or ""),
                    "collected_facts": collected_facts,
                }
            )
            return internal, next_state

        if next_item is None:
            internal = self._build_virtual_advocate_result(
                query=merged_issue,
                latest_message=message,
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
                    "last_guidance_key": str(internal.raw_json.get("style_key") or ""),
                    "collected_facts": collected_facts,
                }
            )
            return internal, next_state

        question_opening, style_key = self._guidance_opening_for_turn(
            issue_type=base_state.issue_type or "general",
            latest_message=message,
            state=base_state,
            facts=collected_facts,
        )
        question_text = self._build_virtual_advocate_question(
            state=base_state,
            issue_type=base_state.issue_type or "general",
            latest_message=message,
            question=next_item["question"],
            opening=question_opening,
            is_first_question=not is_continuation,
            collected_facts=collected_facts,
            allow_section_mentions=section_requested,
        )
        legal_references = self._legal_references_for_issue(state=base_state, facts=collected_facts) if section_requested else []
        likely_forum = self._forum_for_issue(base_state.issue_type or "general")
        authority_items = legal_references or ([likely_forum] if likely_forum else [])
        playbook = self._legal_help_playbook(base_state.issue_type or "general")
        disclaimer_mode = self._select_disclaimer_mode(
            route="playbook",
            urgency=str(playbook.get("risk_level") or "medium"),
            source_sufficiency="partial",
        )
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
            raw_json={
                "source": "legal_help_interview",
                "collected_facts": collected_facts,
                "legal_references": legal_references,
                "style_key": style_key,
                "playbook_id": playbook.get("playbook_id"),
                "required_facts": playbook.get("required_facts"),
                "risk_level": playbook.get("risk_level"),
                "disclaimer_mode": disclaimer_mode,
            },
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
                "last_guidance_key": str(style_key or ""),
                "collected_facts": collected_facts,
            }
        )
        return internal, next_state

    @staticmethod
    def _should_treat_message_as_fact_answer(
        *,
        turn_classification: dict[str, Any],
        current_intake_key: str | None,
        is_continuation: bool,
    ) -> bool:
        if not is_continuation or not current_intake_key:
            return False
        return str(turn_classification.get("mode") or "") == "fact_update"

    def _classify_legal_help_follow_up_turn(
        self,
        *,
        message: str,
        state: ConversationState,
        collected_facts: dict[str, str],
        is_continuation: bool,
    ) -> dict[str, Any]:
        del collected_facts
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        if not is_continuation or not cleaned:
            return {
                "mode": "none",
                "decision_type": None,
                "raw_question": cleaned,
                "confidence": 0.0,
            }

        decision = self._normalize_follow_up_decision_type(
            message=cleaned,
            state=state,
        )
        if decision is not None:
            return {
                "mode": "decision_question",
                "decision_type": decision.get("decision_type"),
                "raw_question": cleaned,
                "confidence": float(decision.get("confidence") or 0.9),
            }

        if "?" in cleaned or any(token in lowered for token in {"do i need", "should i", "which one", "or first"}):
            return {
                "mode": "question_other",
                "decision_type": None,
                "raw_question": cleaned,
                "confidence": 0.35,
            }

        return {
            "mode": "fact_update",
            "decision_type": None,
            "raw_question": cleaned,
            "confidence": 0.8,
        }

    def _normalize_follow_up_decision_type(
        self,
        *,
        message: str,
        state: ConversationState,
    ) -> dict[str, Any] | None:
        cleaned = re.sub(r"\s+", " ", message.strip())
        lowered = cleaned.lower()
        issue_type = state.issue_type or "general"
        if not cleaned:
            return None

        if any(
            phrase in lowered
            for phrase in {
                "fir or a normal complaint",
                "fir or normal complaint",
                "fir or complaint",
                "file fir or complaint",
            }
        ):
            if issue_type in {"cyber_fraud", "snatching_theft", "fir_refusal", "police_complaint"}:
                return {"decision_type": "fir_vs_complaint", "confidence": 0.96}
        if "need to file fir" in lowered and "complaint" in lowered and issue_type in {"cyber_fraud", "snatching_theft", "fir_refusal", "police_complaint"}:
            return {"decision_type": "fir_vs_complaint", "confidence": 0.94}

        if issue_type in {"cyber_fraud", "snatching_theft"}:
            if any(
                phrase in lowered
                for phrase in {
                    "police or bank",
                    "bank or police",
                    "police vs bank",
                    "bank vs police",
                    "go to police or bank first",
                    "go to bank or police first",
                    "bank first or police",
                    "police first or bank",
                }
            ):
                return {"decision_type": "police_vs_bank", "confidence": 0.95}
        if issue_type == "cyber_fraud":
            if any(
                phrase in lowered
                for phrase in {
                    "bank or app",
                    "app or bank",
                    "bank or platform",
                    "platform or bank",
                    "payment app or bank",
                    "bank first or app",
                    "app first or bank",
                }
            ):
                return {"decision_type": "bank_vs_platform", "confidence": 0.93}
            if any(
                phrase in lowered
                for phrase in {
                    "cyber cell or police",
                    "cyber cell or local police",
                    "police or cyber cell",
                    "local police or cyber cell",
                    "cyber portal or police",
                    "police or cyber portal",
                    "report to the police too",
                    "report it to the police too",
                    "report to police too",
                    "go to the police too",
                    "go to police too",
                    "also report to the police",
                    "also report to police",
                    "need to report to the police too",
                    "need to go to the police too",
                }
            ):
                return {"decision_type": "police_vs_cyber_channel", "confidence": 0.92}
        if issue_type in {"consumer", "food_safety"}:
            if any(
                phrase in lowered
                for phrase in {
                    "seller or consumer forum",
                    "consumer forum or seller",
                    "brand or consumer forum",
                    "consumer court or seller",
                    "consumer commission or seller",
                }
            ):
                return {"decision_type": "seller_vs_consumer_forum", "confidence": 0.94}
        if issue_type == "notice":
            if any(
                phrase in lowered
                for phrase in {
                    "reply first or collect documents",
                    "collect documents first or reply",
                    "reply now or gather documents",
                    "gather documents or reply first",
                    "send reply first or prepare documents",
                }
            ):
                return {"decision_type": "reply_vs_documents", "confidence": 0.91}

        return None

    def _build_follow_up_decision_result(
        self,
        *,
        turn_classification: dict[str, Any],
        state: ConversationState,
        collected_facts: dict[str, str],
        next_item: dict[str, str | int] | None,
        domain: str,
        warnings: list[str],
    ) -> InternalChatResult:
        issue_type = state.issue_type or "general"
        answer = self._answer_follow_up_decision_question(
            decision_type=str(turn_classification.get("decision_type") or ""),
            state=state,
            collected_facts=collected_facts,
        )
        resumed_follow_up = self._resume_follow_up_after_decision_answer(
            state=state,
            next_item=next_item,
            collected_facts=collected_facts,
        )
        return InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=resumed_follow_up,
            citations=[],
            authorities=self._legal_references_for_issue(state=state, facts=collected_facts)[:2] if self._user_requested_sections(state.last_user_issue or "") else [],
            documents_to_keep=self._documents_for_issue(issue_type, collected_facts)[:3],
            likely_forum=self._forum_for_issue(issue_type),
            caution=None,
            warnings=warnings,
            raw_json={
                "source": "legal_help_decision_answer",
                "decision_type": turn_classification.get("decision_type"),
                "style_key": state.last_guidance_key,
                "resumed_follow_up_question": resumed_follow_up,
                "playbook_id": self._legal_help_playbook(issue_type).get("playbook_id"),
                "required_facts": self._legal_help_playbook(issue_type).get("required_facts"),
                "risk_level": self._legal_help_playbook(issue_type).get("risk_level"),
                "disclaimer_mode": self._select_disclaimer_mode(
                    route="playbook",
                    urgency=str(self._legal_help_playbook(issue_type).get("risk_level") or "medium"),
                    source_sufficiency="partial",
                ),
            },
        )

    def _resume_follow_up_after_decision_answer(
        self,
        *,
        state: ConversationState,
        next_item: dict[str, str | int] | None,
        collected_facts: dict[str, str],
    ) -> str | None:
        if not next_item:
            return None
        question = str(next_item.get("question") or "").strip()
        if not question:
            return None
        return self._contextualize_follow_up_question(
            issue_type=state.issue_type or "general",
            question=question,
            state=state,
            collected_facts=collected_facts,
        )

    def _answer_follow_up_decision_question(
        self,
        *,
        decision_type: str,
        state: ConversationState,
        collected_facts: dict[str, str],
    ) -> str:
        return self._clarification_answer_text(
            kind=decision_type,
            state=state,
            collected_facts=collected_facts,
        )

    def _format_two_line_decision_answer(self, action: str, support: str) -> str:
        action_line = action.strip()
        support_line = support.strip()
        if not support_line:
            return action_line
        return f"{action_line}\n{support_line}"

    def _clarification_answer_text(
        self,
        *,
        kind: str,
        state: ConversationState,
        collected_facts: dict[str, str],
    ) -> str:
        issue_type = state.issue_type or "general"
        if kind == "fir_vs_complaint":
            if issue_type in {"cyber_fraud", "snatching_theft"}:
                return self._format_two_line_decision_answer(
                    "Ask for an FIR if the police treat this as a cognizable offence.",
                    "If they do not register it immediately, a written complaint with acknowledgement still protects your record.",
                )
            if issue_type == "fir_refusal":
                return self._format_two_line_decision_answer(
                    "Press for an FIR first on facts serious enough for registration.",
                    "If the station still refuses, keep the written complaint and the refusal trail for escalation.",
                )
            if issue_type == "police_complaint":
                incident = str(collected_facts.get("incident_details") or state.last_user_issue or "").strip()
                if self._facts_clearly_indicate_criminal_issue(incident):
                    return self._format_two_line_decision_answer(
                        "Ask for an FIR if the incident amounts to a cognizable offence.",
                        "If the station is not taking that step yet, keep proof of your written complaint.",
                    )
                return self._format_two_line_decision_answer(
                    "Start with a written complaint unless the facts clearly call for an FIR.",
                    "In either case, keep the acknowledgement because it anchors the record.",
                )
        if kind == "police_vs_bank":
            if issue_type == "cyber_fraud":
                return self._format_two_line_decision_answer(
                    "Contact the bank or payment app first to try to stop further debit.",
                    "The police or cyber complaint should follow quickly so the transaction is formally recorded.",
                )
            if issue_type == "snatching_theft":
                return self._format_two_line_decision_answer(
                    "Go to the police first for the incident record.",
                    "Bank follow-up also matters if the phone, SIM, or OTP access can affect your accounts.",
                )
        if kind == "bank_vs_platform":
            return self._format_two_line_decision_answer(
                "Contact the bank first if money has moved or the account is at risk.",
                "Inform the app or platform as well so the complaint trail stays aligned.",
            )
        if kind == "police_vs_cyber_channel":
            return self._format_two_line_decision_answer(
                "Use the cyber reporting channel first in a cyber-fraud matter.",
                "That early fraud record helps recovery efforts, and local police can also be approached in parallel if you need a local complaint record.",
            )
        if kind == "seller_vs_consumer_forum":
            return self._format_two_line_decision_answer(
                "Complain to the seller or brand first.",
                "If they do not resolve it, the consumer forum becomes the stronger next step.",
            )
        if kind == "reply_vs_documents":
            return self._format_two_line_decision_answer(
                "Organize the documents and timeline first.",
                "A reply is safer once the record is clear, unless the deadline is too close to wait.",
            )
        return self._format_two_line_decision_answer(
            "Share the next missing fact first.",
            "That will let the workflow continue without guessing.",
        )

    def _build_virtual_advocate_question(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        latest_message: str,
        question: str,
        opening: str,
        is_first_question: bool,
        collected_facts: dict[str, str],
        allow_section_mentions: bool,
    ) -> str:
        del is_first_question
        immediate = self._immediate_guidance_for_issue(
            state=state,
            issue_type=issue_type,
            collected_facts=collected_facts,
            latest_message=latest_message,
        )
        context_line = self._context_line_for_issue(
            state=state,
            issue_type=issue_type,
            collected_facts=collected_facts,
            latest_message=latest_message,
        )
        legal_basis = (
            self._early_legal_basis_text(state=state, issue_type=issue_type, collected_facts=collected_facts)
            if allow_section_mentions
            else ""
        )
        parts = [immediate]
        merged_context = " ".join(part.strip() for part in [opening, context_line] if part.strip()).strip()
        if merged_context:
            parts.append(merged_context)
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
            "cyber_fraud": "I know this is stressful.",
            "snatching_theft": "I am sorry this happened.",
            "landlord_harassment": "I understand this is difficult.",
            "food_safety": "I understand your concern.",
            "consumer": "I understand the problem.",
            "fir_refusal": "I understand the frustration.",
            "notice": "I understand the concern.",
            "police_complaint": "I understand the concern.",
            "documents": "I understand the issue.",
            "general": "I understand the concern.",
        }
        return mapping.get(issue_type, "I understand the concern.")

    @staticmethod
    def _parse_guidance_style_key(raw: str | None) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for part in str(raw or "").split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                parsed[key] = value
        return parsed

    @staticmethod
    def _serialize_guidance_style_key(style_state: dict[str, str]) -> str | None:
        parts = [f"{key}={value}" for key, value in style_state.items() if value]
        return "|".join(parts) if parts else None

    def _choose_guidance_variant(
        self,
        *,
        slot: str,
        options: list[tuple[str, str]],
        issue_type: str,
        latest_message: str,
        style_state: dict[str, str],
    ) -> str:
        if not options:
            return ""
        previous = style_state.get(slot)
        seed_text = f"{issue_type}|{latest_message}|{slot}"
        index = sum(ord(char) for char in seed_text) % len(options)
        if len(options) > 1 and options[index][0] == previous:
            index = (index + 1) % len(options)
        key, value = options[index]
        style_state[slot] = key
        return value

    def _latest_update_category(self, *, issue_type: str, latest_message: str) -> str:
        lowered = latest_message.lower()
        if any(token in lowered for token in {"1930", "reported", "report", "complaint number", "bank complaint", "emailed", "written complaint", "filed", "fir"}):
            return "progress"
        if any(token in lowered for token in {"transaction id", "invoice", "bill", "receipt", "imei", "photo", "photos", "screenshot", "proof", "packet"}):
            return "evidence"
        if any(token in lowered for token in {"today", "yesterday", "morning", "evening", "night", "date", "time", "when"}):
            return "timing"
        if issue_type == "notice" and any(token in lowered for token in {"deadline", "reply", "received", "notice"}):
            return "deadline"
        if any(token in lowered for token in {"station", "city", "police", "bank", "phonepe", "gpay", "paytm", "seller", "brand", "landlord", "authority"}):
            return "actor"
        return "general"

    def _latest_update_phrase(
        self,
        *,
        issue_type: str,
        latest_message: str,
        state: ConversationState,
        facts: dict[str, str],
    ) -> str:
        lowered = latest_message.lower()
        category = self._latest_update_category(issue_type=issue_type, latest_message=latest_message)
        if issue_type == "cyber_fraud":
            if category == "progress":
                bank_name = state.bank_name or self._display_platform_name(state.platform) or "the bank or app"
                return f"reporting has already started with {bank_name}"
            if category == "evidence":
                return "the transaction and proof record is clearer now"
            if category == "timing":
                return "the debit timing is clearer now"
        if issue_type == "snatching_theft":
            if category == "progress":
                return "the complaint stage is clearer now"
            if category == "evidence":
                return "the device proof and ownership record are clearer now"
            if category == "actor":
                return "the police location is clearer now"
        if issue_type == "food_safety":
            if category == "progress":
                return "the seller-side complaint trail has already started"
            if category == "evidence":
                return "the packet and supporting proof are available"
        if issue_type == "notice":
            if category == "deadline":
                return "the notice stage and timing are clearer now"
            if category == "evidence":
                return "the notice record is better supported"
        if issue_type == "consumer":
            if category == "progress":
                return "the complaint trail is already underway"
            if category == "evidence":
                return "the payment and defect record is clearer now"
        if issue_type == "landlord_harassment":
            if category == "evidence":
                return "the tenancy record is clearer now"
            if category == "actor":
                return "the pressure source is clearer now"
        if issue_type in {"fir_refusal", "police_complaint"}:
            if category == "progress":
                return "the complaint submission stage is clearer now"
            if category == "actor":
                return "the station detail is clearer now"
        if issue_type == "documents":
            if category in {"actor", "deadline"}:
                return "the filing requirement is narrower now"
        if category == "progress":
            return "the progress already made is clearer now"
        if category == "evidence":
            return "the supporting record is stronger now"
        if category == "timing":
            return "the timeline is clearer now"
        if category == "actor":
            return "the relevant person or authority is clearer now"
        return "the latest detail narrows the next step"

    def _guidance_opening_for_turn(
        self,
        *,
        issue_type: str,
        latest_message: str,
        state: ConversationState,
        facts: dict[str, str],
    ) -> tuple[str, str | None]:
        if not state.conversation_started:
            return self._acknowledgement_for_issue(issue_type), state.last_guidance_key

        style_state = self._parse_guidance_style_key(state.last_guidance_key)
        phrase = self._latest_update_phrase(
            issue_type=issue_type,
            latest_message=latest_message,
            state=state,
            facts=facts,
        )
        opening = self._choose_guidance_variant(
            slot="opening",
            options=[
                ("confirm", f"That helps. It is now clear that {phrase}."),
                ("latest", f"Your update makes one thing clearer: {phrase}."),
                ("update", f"That is useful, because it shows that {phrase}."),
                ("narrow", f"This narrows the position a bit, because {phrase}."),
            ],
            issue_type=issue_type,
            latest_message=latest_message,
            style_state=style_state,
        )
        return opening, self._serialize_guidance_style_key(style_state)

    def _action_prefix_for_turn(
        self,
        *,
        issue_type: str,
        latest_message: str,
        state: ConversationState,
    ) -> str:
        category = self._latest_update_category(issue_type=issue_type, latest_message=latest_message)
        options = {
            "progress": [("next", "From here"), ("follow", "The next practical step"), ("push", "What to do next")],
            "evidence": [("record", "On the evidence side"), ("proof", "The next useful step"), ("preserve", "For now")],
            "timing": [("time", "Given the timing"), ("now", "At this stage"), ("window", "While this is still fresh")],
            "deadline": [("deadline", "Because of the deadline"), ("reply", "At this stage"), ("time", "Do this promptly")],
            "actor": [("target", "Now focus on this"), ("authority", "The next step"), ("narrow", "From here")],
            "general": [("immediate", "The first step"), ("priority", "The practical next step"), ("first", "The first step I suggest")],
        }
        pool = options.get(category, options["general"])
        style_state = self._parse_guidance_style_key(state.last_guidance_key)
        return self._choose_guidance_variant(
            slot="action",
            options=pool,
            issue_type=issue_type,
            latest_message=latest_message,
            style_state=style_state,
        )

    def _urgency_prefix_for_turn(
        self,
        *,
        issue_type: str,
        latest_message: str,
        state: ConversationState,
    ) -> str:
        category = self._latest_update_category(issue_type=issue_type, latest_message=latest_message)
        options = {
            "progress": [("keep", "Keep the momentum going"), ("trail", "Do not let the complaint trail go cold"), ("same", "Stay with the same record")],
            "evidence": [("preserve", "Please preserve this carefully"), ("loss", "Before any proof is lost"), ("secure", "Secure this while it is still available")],
            "timing": [("window", "Timing matters here"), ("clock", "This is time-sensitive"), ("delay", "It is better not to wait")],
            "deadline": [("deadline", "The deadline matters"), ("delay", "Do not let the deadline slip"), ("reply", "The reply window is important")],
            "general": [("urgent", "The important point"), ("delay", "Do this promptly"), ("priority", "One caution")],
        }
        pool = options.get(category, options["general"])
        style_state = self._parse_guidance_style_key(state.last_guidance_key)
        return self._choose_guidance_variant(
            slot="urgency",
            options=pool,
            issue_type=issue_type,
            latest_message=latest_message,
            style_state=style_state,
        )

    def _immediate_guidance_for_issue(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        collected_facts: dict[str, str],
        latest_message: str,
    ) -> str:
        prefix = self._action_prefix_for_turn(issue_type=issue_type, latest_message=latest_message, state=state)
        category = self._latest_update_category(issue_type=issue_type, latest_message=latest_message)
        if issue_type == "cyber_fraud":
            bank_platform = (
                collected_facts.get("bank_platform")
                or state.bank_name
                or self._display_platform_name(state.platform)
                or "your bank or payment app"
            )
            if category == "timing":
                return (
                    f"{prefix}: note the exact debit time, whether it was one transaction or multiple debits, and keep that version consistent for the bank and complaint record."
                )
            if category == "evidence":
                return (
                    f"{prefix}: keep the transaction ID, screenshots, debit alerts, and bank statement together in one file for the complaint trail."
                )
            if category == "actor":
                return (
                    f"{prefix}: use the same transaction facts with {bank_platform}, 1930, and any police follow-up so the complaint trail stays consistent."
                )
            if category == "progress":
                if self._fact_indicates_completed_action(collected_facts.get("reporting_status"), {"reported", "1930", "cyber crime portal", "complaint number", "bank complaint"}):
                    return (
                        f"{prefix}: keep the complaint reference numbers, transaction details, and screenshots aligned for bank or police follow-up."
                    )
                return (
                    f"{prefix}: report the matter to {bank_platform} and through 1930 or the National Cyber Crime Portal without delay, and keep every complaint reference number."
                )
            return (
                f"{prefix}: contact {bank_platform} immediately, ask them to block further misuse, keep the transaction ID, screenshots, messages, and bank alerts together, and report the matter through 1930 or the National Cyber Crime Portal. Reporting within 24 hours often gives you a better chance of recovery."
            )
        if issue_type == "snatching_theft":
            location = collected_facts.get("station_details") or state.police_station or state.city or "the police station"
            if category == "timing":
                return (
                    f"{prefix}: write down the exact time, place, and route now so the complaint record stays consistent."
                )
            if category == "actor":
                return (
                    f"{prefix}: use the {location} complaint route with the incident details, item description, and any witness or CCTV clue."
                )
            if category == "evidence":
                return (
                    f"{prefix}: keep the IMEI, invoice, and ownership proof ready to add to the complaint record."
                )
            if category == "progress":
                if self._fact_indicates_completed_action(collected_facts.get("police_status"), {"filed", "fir", "complaint"}):
                    return (
                        f"{prefix}: add the item identifiers, acknowledgement, and any misuse-risk details to the existing complaint record."
                    )
                return (
                    f"{prefix}: file or supplement the complaint with the incident details, item identifiers, and any immediate misuse risk."
                )
            return (
                f"{prefix}: block the SIM, secure your email and banking apps, keep the IMEI or purchase proof ready, and take a short written complaint to the nearest police station."
            )
        if issue_type == "landlord_harassment":
            if category == "evidence":
                return (
                    f"{prefix}: keep the rent agreement, payment proof, messages, and threat record together in date order."
                )
            if category == "actor":
                return (
                    f"{prefix}: match the next step to the exact pressure point, such as eviction pressure, lockout, threat, or deposit dispute."
                )
            return (
                f"{prefix}: gather the rent agreement, rent proof, messages, and any threat record in one place. Do not vacate or hand over original papers under pressure, and if there is a threat, lockout, or force, preserve proof straight away."
            )
        if issue_type == "food_safety":
            if category == "evidence":
                return (
                    f"{prefix}: keep the packet, batch number, expiry date, invoice, photos or video, and any illness record together in one preserved file."
                )
            if category == "progress":
                if self._fact_indicates_completed_action(collected_facts.get("seller_contact"), {"complained", "seller", "brand", "platform", "emailed", "written complaint"}):
                    return (
                        f"{prefix}: keep the seller-side complaint trail ready, then escalate through the FSSAI Food Safety Connect channel or the National Consumer Helpline if there is no proper refund, replacement, or response."
                    )
                return (
                    f"{prefix}: send a written complaint to the seller or brand asking for refund or replacement, then keep the ticket number for FSSAI or Consumer Helpline escalation."
                )
            return (
                f"{prefix}: do not throw away the packet or product. Preserve the invoice, batch details, and photos, complain in writing to the seller or brand, and be ready to escalate on FSSAI Food Safety Connect or the National Consumer Helpline."
            )
        if issue_type == "consumer":
            if category == "evidence":
                return (
                    f"{prefix}: keep the invoice, payment proof, warranty or listing page, defect photos, chats, and complaint numbers together in one file."
                )
            if category == "progress":
                if self._fact_indicates_completed_action(collected_facts.get("complaint_status"), {"complaint", "emailed", "written", "ticket", "support"}):
                    return (
                        f"{prefix}: tighten the complaint record and state the exact refund, replacement, repair, compensation, and timeline you want before escalation."
                    )
                return (
                    f"{prefix}: put the complaint in writing to the seller or service provider, ask for refund, replacement, repair, or compensation, and preserve the reply trail."
                )
            return (
                f"{prefix}: keep the invoice, payment proof, and defect record together, complain in writing to the seller or platform, and escalate through the National Consumer Helpline if they do not resolve it."
            )
        if issue_type == "fir_refusal":
            if category == "actor":
                return (
                    f"{prefix}: tie the escalation to the exact police station, officer, and submission trail before moving it upward."
                )
            if category == "progress":
                return (
                    f"{prefix}: carry forward the same written complaint, refusal detail, and submission proof for escalation instead of restarting orally."
                )
            return (
                f"{prefix}: keep a copy of the complaint you already gave, note the police station, officer, date, and any refusal or diary detail, and preserve proof that the complaint was submitted but not registered."
            )
        if issue_type == "notice":
            if category in {"deadline", "timing"}:
                return (
                    f"{prefix}: mark the deadline and organise the notice file before any reply goes out."
                )
            if category == "evidence":
                return (
                    f"{prefix}: keep the notice copy, agreement, messages, and payment records together in one reply file."
                )
            if category == "progress":
                return (
                    f"{prefix}: map the demand, deadline, and your factual reply points in one file before drafting anything further."
                )
            return (
                f"{prefix}: check the notice deadline first, keep the notice, agreement, messages, and payment records together, and do not send a rushed reply before the record is organised."
            )
        if issue_type == "police_complaint":
            if category == "timing":
                return (
                    f"{prefix}: write the incident time and sequence down clearly so the complaint record starts with one consistent version."
                )
            if category == "evidence":
                return (
                    f"{prefix}: keep the main supporting proof ready to attach to the written complaint."
                )
            if category == "actor":
                return (
                    f"{prefix}: prepare the complaint for the exact station or city already identified so the submission is specific."
                )
            return (
                f"{prefix}: write the incident briefly in date order, keep the main proof ready, and submit a written complaint at the police station."
            )
        if issue_type == "documents":
            if category in {"actor", "deadline"}:
                return (
                    f"{prefix}: match the submission set to the exact office requirement and the current deadline position."
                )
            return (
                f"{prefix}: confirm exactly what document or format the authority wants, keep one set for submission and one for your own record, and preserve the receipt or acknowledgement."
            )
        return f"{prefix}: write down the facts clearly, keep the main proof together, and do not delay the first complaint or response if there is urgency or a deadline."

    def _context_line_for_issue(
        self,
        *,
        state: ConversationState,
        issue_type: str,
        collected_facts: dict[str, str],
        latest_message: str,
    ) -> str:
        prefix = self._urgency_prefix_for_turn(issue_type=issue_type, latest_message=latest_message, state=state)
        saved_context = self._case_context_sentence(state)
        saved_context_suffix = f" {saved_context}" if saved_context else ""
        if issue_type == "cyber_fraud":
            return f"{prefix}: the first 24 hours usually matter most because tracing and recovery chances are often better while the transaction trail is still fresh and consistent.{saved_context_suffix}"
        if issue_type == "snatching_theft":
            return f"{prefix}: misuse risk usually rises quickly after snatching or theft, and early account protection plus a clear complaint record often make follow-up easier.{saved_context_suffix}"
        if issue_type == "landlord_harassment":
            return f"{prefix}: early written proof usually matters more if the pressure escalates into threats, lockout, or a formal dispute.{saved_context_suffix}"
        if issue_type == "food_safety":
            return f"{prefix}: this kind of matter is easier to prove while the packet condition and defect images still reflect the original problem.{saved_context_suffix}"
        if issue_type == "consumer":
            return f"{prefix}: a specific written record usually makes the defect, timeline, and relief claimed much easier to prove later.{saved_context_suffix}"
        if issue_type == "fir_refusal":
            return f"{prefix}: the refusal trail becomes harder to establish if the complaint history changes or stretches out.{saved_context_suffix}"
        if issue_type == "notice":
            return f"{prefix}: the reply window matters because a late or rushed response can weaken your position.{saved_context_suffix}"
        if issue_type == "police_complaint":
            return f"{prefix}: a clear chronology usually makes the station-side process smoother from the beginning.{saved_context_suffix}"
        if issue_type == "documents":
            return f"{prefix}: format mismatch often causes avoidable rejection or delay in filing processes.{saved_context_suffix}"
        return f"{prefix}: the next step usually works better when the facts, documents, and urgency are already clear.{saved_context_suffix}"

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
            return "If you are asking about sections, cyber-fraud of this kind is usually explained first through the underlying offence, such as cheating under BNS section 318 and, where someone impersonated a trusted person or service, cheating by personation under BNS section 319. The FIR or police-complaint side then usually starts through BNSS section 173."
        if issue_type == "snatching_theft":
            return "If you want the legal framing, the underlying offence is usually theft under BNS section 303 or, where the property was forcibly taken from the person, snatching under BNS section 304. The police complaint usually begins through BNSS section 173."
        if issue_type in {"fir_refusal", "police_complaint"}:
            incident_text = (collected_facts.get("incident_details") or "").strip()
            substantive_refs = self._substantive_references_for_incident(incident_text)
            if substantive_refs:
                offence_text = ", ".join(substantive_refs[:2])
                return f"If you are asking about sections, start with the underlying offence itself, for example {offence_text}. After that, the FIR or police-complaint route usually begins through BNSS section 173."
            return "If you are asking about sections, the answer usually has two parts: the underlying offence section based on what happened, and the police-procedure part under BNSS section 173 for recording the complaint."
        if issue_type == "landlord_harassment":
            return "If the landlord's conduct includes threats or intimidation, there may also be a criminal angle, for example criminal intimidation under BNS section 351, apart from the tenancy dispute itself."
        return ""

    def _build_virtual_advocate_result(
        self,
        *,
        query: str,
        latest_message: str,
        domain: str,
        warnings: list[str],
        state: ConversationState,
        collected_facts: dict[str, str],
    ) -> InternalChatResult:
        legal_references = self._legal_references_for_issue(state=state, facts=collected_facts)
        paragraph = self._compose_virtual_advocate_paragraph(query=query, state=state, facts=collected_facts)
        steps = self._compose_virtual_advocate_steps(state=state, facts=collected_facts)
        intro, style_key = self._completed_guidance_intro(state=state, facts=collected_facts, latest_message=latest_message)
        saved_context = self._case_context_sentence(state)
        if saved_context:
            intro = f"{intro} {saved_context}".strip()
        action_block, context_block = self._split_completed_guidance_blocks(paragraph)
        answer = "\n\n".join(part for part in [intro, action_block, context_block] if part).strip()
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
            raw_json={
                "source": "virtual_advocate",
                "collected_facts": collected_facts,
                "legal_references": legal_references,
                "style_key": style_key,
                "playbook_id": self._legal_help_playbook(state.issue_type or "general").get("playbook_id"),
                "required_facts": self._legal_help_playbook(state.issue_type or "general").get("required_facts"),
                "risk_level": self._legal_help_playbook(state.issue_type or "general").get("risk_level"),
                "disclaimer_mode": self._select_disclaimer_mode(
                    route="playbook",
                    urgency=str(self._legal_help_playbook(state.issue_type or "general").get("risk_level") or "medium"),
                    source_sufficiency="partial",
                ),
            },
        )

    @staticmethod
    def _split_completed_guidance_blocks(paragraph: str) -> tuple[str, str | None]:
        cleaned = re.sub(r"\s+", " ", str(paragraph or "").strip())
        if not cleaned:
            return "", None
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [item.strip() for item in sentences if item.strip()]
        if len(sentences) <= 2:
            return cleaned, None
        action_block = " ".join(sentences[:2]).strip()
        context_block = " ".join(sentences[2:]).strip() or None
        return action_block, context_block

    def _completed_guidance_intro(
        self,
        *,
        state: ConversationState,
        facts: dict[str, str],
        latest_message: str,
    ) -> tuple[str, str | None]:
        issue_type = state.issue_type or "general"
        opening, style_key = self._guidance_opening_for_turn(
            issue_type=issue_type,
            latest_message=latest_message,
            state=state,
            facts=facts,
        )
        if issue_type == "cyber_fraud":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("reporting_status"), {"reported", "1930", "cyber crime portal", "complaint number", "bank complaint"}):
                progress = " You have already started the reporting process, so the focus now is on strengthening the record and following it up properly."
            return (
                f"{opening} On these facts, time matters, and reporting within 24 hours usually improves the chances of recovery."
                + progress,
                style_key,
            )
        if issue_type == "snatching_theft":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("police_status"), {"filed", "fir", "complaint"}):
                progress = " You have already started the complaint side, so the focus now is on strengthening the record and reducing misuse risk."
            return (
                f"{opening} The earlier you secure the linked accounts and tighten the complaint record, the better."
                + progress,
                style_key,
            )
        if issue_type == "landlord_harassment":
            return (
                f"{opening} If the pressure is escalating, preserving proof early will matter.",
                style_key,
            )
        if issue_type == "food_safety":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("seller_contact"), {"complained", "written complaint", "seller", "brand", "platform", "emailed"}):
                progress = " Since you have already complained to the seller side, move from waiting to escalation: keep the proof ready for FSSAI Food Safety Connect and the National Consumer Helpline."
            return (
                f"{opening} This becomes much stronger when the packet, batch details, invoice, photos, and complaint numbers are preserved before FSSAI or consumer escalation."
                + progress,
                style_key,
            )
        if issue_type == "consumer":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("complaint_status"), {"complaint", "emailed", "written", "ticket", "support"}):
                progress = " Since a complaint trail already exists, the next step is to set a deadline and state the exact refund, replacement, repair, or compensation expected."
            return (
                f"{opening} The strongest path is a short written demand backed by proof, followed by National Consumer Helpline or Consumer Commission escalation if the seller does not resolve it."
                + progress,
                style_key,
            )
        if issue_type == "fir_refusal":
            return (
                f"{opening} Delay can make the refusal trail harder to prove.",
                style_key,
            )
        if issue_type == "notice":
            progress = ""
            if self._fact_indicates_completed_action(facts.get("notice_stage_detail"), {"received", "reply"}):
                progress = " You are already at the notice stage, so the next step is to organise the reply record rather than repeat that part."
            return (
                f"{opening} The reply window is important because a late or rushed reply can weaken your position."
                + progress,
                style_key,
            )
        if issue_type == "police_complaint":
            return (
                f"{opening} Early reporting usually preserves the chronology better.",
                style_key,
            )
        if issue_type == "documents":
            return (
                f"{opening} Checking the requirement now is safer than correcting a rejection later.",
                style_key,
            )
        return (f"{opening} We should move carefully but without delay.", style_key)

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
        disclaimer_mode = self._select_disclaimer_mode(
            route="playbook",
            urgency=str(guidance.get("risk_level") or "medium"),
            source_sufficiency="partial",
        )
        answer = self._format_final_answer(
            summary=summary,
            legal_position=legal_position,
            practical_next_steps=practical_next_steps,
            sources=sources,
            disclaimer=self._disclaimer_text(disclaimer_mode),
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
            raw_json={
                "results": [],
                "source": "legal_help",
                "playbook_id": guidance.get("playbook_id"),
                "required_facts": guidance.get("required_facts"),
                "risk_level": guidance.get("risk_level"),
                "disclaimer_mode": disclaimer_mode,
            },
        )

    def _build_fallback_result(self, domain: str, warnings: list[str]) -> InternalChatResult:
        return self._build_safe_fallback_result(
            kind="no_relevant_authority",
            domain=domain,
            warnings=warnings,
            query=None,
        )

    def _build_safe_fallback_result(
        self,
        *,
        kind: str,
        domain: str,
        warnings: list[str],
        query: str | None,
    ) -> InternalChatResult:
        normalized_query = re.sub(r"\s+", " ", str(query or "").strip().lower())
        complete_authority_reference = self._looks_like_complete_authority_reference_prompt(normalized_query)
        bns_unavailable = self._is_bns_authority_query_with_unavailable_dataset(normalized_query)
        follow_up_question: str | None = None
        likely_forum: str | None = None
        caution: str | None = None
        disclaimer_mode = "medium_risk"
        if bns_unavailable and kind in {"low_confidence", "unsupported_output", "no_relevant_authority", "technical_failure"}:
            answer = self._complete_authority_no_result_answer(query=query or "")
            reason_code = "bns_dataset_source_unavailable"
            disclaimer_mode = "high_risk"
        elif complete_authority_reference and kind in {"low_confidence", "unsupported_output", "no_relevant_authority"}:
            answer = self._complete_authority_no_result_answer(query=query or "")
            reason_code = (
                "bns_dataset_source_unavailable"
                if "BNS dataset/source unavailable" in answer
                else "retrieval_low_confidence"
                if kind == "low_confidence"
                else ("unsupported_output" if kind == "unsupported_output" else "no_relevant_authority")
            )
            caution = None
            disclaimer_mode = "high_risk" if kind in {"low_confidence", "unsupported_output"} else "medium_risk"
        elif kind == "low_confidence":
            follow_up_question = self._grounded_clarifying_question(query or "")
            answer = self._human_fallback_clarification_answer(
                kind=kind,
                question=follow_up_question,
            )
            caution = "A more precise legal reference or factual context is needed before giving a grounded answer."
            reason_code = "retrieval_low_confidence"
            disclaimer_mode = "high_risk"
        elif kind == "technical_failure":
            follow_up_question = "Please retry once, or share the exact section, statute, judgment, citation, or short factual context you want checked."
            answer = self._human_fallback_clarification_answer(
                kind=kind,
                question=follow_up_question,
            )
            reason_code = "technical_failure"
            disclaimer_mode = "high_risk"
        elif kind == "unsupported_output":
            follow_up_question = self._grounded_clarifying_question(query or "")
            answer = self._human_fallback_clarification_answer(
                kind=kind,
                question=follow_up_question,
            )
            caution = "The previous draft was withheld because it contained unsupported or weakly supported legal output."
            reason_code = "unsupported_output"
            disclaimer_mode = "high_risk"
        else:
            follow_up_question = self._grounded_clarifying_question(query or "")
            answer = self._human_fallback_clarification_answer(
                kind=kind,
                question=follow_up_question,
            )
            reason_code = "no_relevant_authority"
            disclaimer_mode = "medium_risk"

        return InternalChatResult(
            answer=answer,
            domain=domain,
            follow_up_question=follow_up_question,
            citations=[],
            authorities=[],
            documents_to_keep=[],
            likely_forum=likely_forum,
            caution=caution,
            warnings=warnings,
            raw_json={
                "pipeline": "indiankanoon_rag",
                "source": "safe_fallback",
                "fallback_type": kind,
                "fallback_reason_code": reason_code,
                "query": query,
                "documents": [],
                "disclaimer_mode": disclaimer_mode,
            },
        )

    @staticmethod
    def _complete_authority_no_result_answer(*, query: str) -> str:
        normalized_query = re.sub(r"\s+", " ", str(query or "").strip())
        reference = normalized_query or "that legal reference"
        lowered = reference.lower()
        if " bns" in f" {lowered}" or "bharatiya nyaya sanhita" in lowered:
            return (
                f"BNS dataset/source unavailable for {reference} right now.\n"
                "I could not confirm it from the local BNS dataset, India Kanoon, or the available external authority sources.\n"
                f"Note: {ChatService._default_brief_disclaimer()}"
            )
        if re.search(r"\bsection\s+\d+[a-z]?\b", lowered) and (
            " ipc" in f" {lowered}" or "indian penal code" in lowered
        ):
            reference = "the requested IPC provision"
        return (
            f"I could not find a reliable official result for {reference} from the currently available sources.\n"
            "I cannot verify that legal reference or treat it as law without a reliable source. "
            "If you want, I can next try the exact statutory text, a broader official-source lookup, or the practical meaning of that provision.\n"
            f"Note: {ChatService._default_brief_disclaimer()}"
        )

    def _is_bns_authority_query_with_unavailable_dataset(self, normalized_query: str) -> bool:
        compact = re.sub(r"\s+", " ", str(normalized_query or "").strip().lower())
        if not compact:
            return False
        if not (" bns" in f" {compact}" or "bharatiya nyaya sanhita" in compact):
            return False
        return not self.legal_dataset.has_dataset_file("bns")

    @staticmethod
    def _human_fallback_clarification_answer(*, kind: str, question: str) -> str:
        normalized_kind = str(kind or "").strip().lower()
        normalized_question = str(question or "").strip()
        if normalized_kind == "technical_failure":
            lead = "I ran into a problem while checking that, so I do not want to guess."
        elif normalized_kind == "unsupported_output":
            lead = "I am not confident enough to give a safe answer in that form yet."
        elif normalized_kind == "low_confidence":
            lead = "I found something related, but it is not clear enough for a reliable answer yet."
        else:
            lead = "I am not seeing a clear enough legal match yet."
        if not normalized_question:
            return lead
        return f"{lead}\n{normalized_question}"

    @staticmethod
    def _grounded_clarifying_question(query: str) -> str:
        normalized = query.lower()
        if ChatService._looks_like_complete_authority_reference_prompt(normalized):
            return ""
        if "article " in normalized and not re.search(r"\barticle\s+\d+[a-z]?\b", normalized):
            return "Which Constitution article do you want checked?"
        if "section " in normalized or "rule " in normalized:
            return "Which exact statute or Act is this section or rule from?"
        if any(token in normalized for token in {"judgment", "judgement", "case law", "citation", "precedent"}):
            return "Can you share the exact case name, court, citation, or year you want checked?"
        return "Can you share one precise legal reference or a short factual context so I can narrow the authorities safely?"

    @staticmethod
    def _clean_search_snippet(text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\{[^{}]*\"errmsg\"[^{}]*\}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(errmsg|debug|traceback|stack trace)\b\s*:?\s*[^.;]*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
        return cleaned[:400]

    def _should_use_indiankanoon(self, query: str) -> bool:
        normalized = query.lower()
        routing = self._classify_routing_precedence(normalized)
        return routing["prefer_grounded_authority"]

    def _looks_like_grounded_authority_query(self, normalized: str) -> bool:
        routing = self._classify_routing_precedence(normalized)
        return routing["has_authority_signal"] and not routing["prefer_playbook"]

    def _is_procedural_or_practical_query(self, normalized: str) -> bool:
        routing = self._classify_routing_precedence(normalized)
        return routing["prefer_playbook"]

    def _should_fallback_to_legal_intake(self, *, message: str, domain: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip()).lower()
        routing = self._classify_routing_precedence(normalized, domain=domain)
        if routing["prefer_grounded_authority"]:
            return False
        return routing["prefer_playbook"] or self._detect_issue_type(normalized, domain) != "general"

    def _classify_routing_precedence(self, normalized: str, domain: str | None = None) -> dict[str, bool]:
        compact = re.sub(r"\s+", " ", normalized.strip().lower())
        issue_type = self._detect_issue_type(compact, domain or "general")
        has_personal_context = bool(re.search(r"\b(i|my|me|we|our)\b", compact))
        action_markers = {
            "how do i",
            "how to",
            "what should i do",
            "what can i do",
            "where to file",
            "where should i file",
            "can i file",
            "file an fir",
            "file a complaint",
            "consumer complaint",
            "police complaint",
            "notice reply",
            "reply to notice",
            "document filing",
            "document verification",
            "report it",
            "report this",
            "next step",
            "next steps",
            "what now",
            "what is the process",
            "what is the procedure",
            "help me file",
        }
        narrative_markers = {
            "scam",
            "scammed",
            "money got debited",
            "debited",
            "money deducted",
            "fake link",
            "payment link",
            "upi fraud",
            "cyber fraud",
            "bank fraud",
            "olx",
            "cheated",
            "unauthorized debit",
            "police refused",
            "refused to file fir",
            "fir not registered",
            "lockout",
            "threatening eviction",
            "snatched",
            "stolen",
            "theft",
            "robbed",
        }
        authority_markers = {
            "case law",
            "judgment",
            "judgement",
            "citation",
            "precedent",
            "supreme court",
            "high court",
            "authority",
            "authorities",
            "legal position",
            "interpretation",
            "ruling",
            "ratio",
            "latest authority",
            "latest judgment",
            "latest case law",
        }
        has_action_signal = any(marker in compact for marker in action_markers)
        has_incident_signal = any(marker in compact for marker in narrative_markers) or (
            has_personal_context and issue_type != "general"
        )
        has_authority_signal = self._is_statute_query(compact) or any(marker in compact for marker in authority_markers)
        prefer_playbook = has_action_signal or has_incident_signal
        prefer_grounded_authority = has_authority_signal and not prefer_playbook
        return {
            "has_action_signal": has_action_signal,
            "has_incident_signal": has_incident_signal,
            "has_authority_signal": has_authority_signal,
            "has_personal_context": has_personal_context,
            "prefer_playbook": prefer_playbook,
            "prefer_grounded_authority": prefer_grounded_authority,
        }

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
        context_sentence = self._case_context_sentence(state)
        if guidance_summary:
            if context_sentence:
                return f"{guidance_summary} {context_sentence}".strip()
            return guidance_summary

        issue_sentence = self._issue_summary_sentence(query=query, state=state)
        if intro_text:
            summary = f"{intro_text} {issue_sentence}".strip()
        else:
            summary = issue_sentence
        if context_sentence:
            return f"{summary} {context_sentence}".strip()
        return summary

    @staticmethod
    def _case_context_sentence(state: ConversationState) -> str:
        parts: list[str] = []
        location_parts = [str(item).strip() for item in [state.district, state.case_state] if str(item or "").strip()]
        if location_parts:
            parts.append(", ".join(location_parts))
        if state.case_stage:
            parts.append(f"{state.case_stage} stage")
        if state.is_own_matter is True:
            parts.append("your own matter")
        elif state.is_own_matter is False:
            parts.append("a matter you are helping with")

        uploaded_summary = ""
        if state.uploaded_document_summaries:
            uploaded_summary = str(state.uploaded_document_summaries[0] or "").strip()
            uploaded_summary = re.sub(r"\s+", " ", uploaded_summary)[:140].strip()

        context_bits: list[str] = []
        if parts:
            context_bits.append("; ".join(parts))
        if uploaded_summary:
            context_bits.append(f"uploaded document: {uploaded_summary}")
        if not context_bits:
            return ""
        return f"I am using the saved chat context ({'; '.join(context_bits)}) so you do not need to repeat it."

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
                "Unsafe or contaminated food can be taken through both the food-safety route and the consumer route. FSSAI/Food Safety Connect is useful for safety action, while the National Consumer Helpline or Consumer Commission is useful for refund, replacement, compensation, and costs."
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
                "Consumer matters usually turn on proof of purchase, proof of defect or service deficiency, a written demand, and a clear relief request such as refund, replacement, repair, compensation, or complaint costs."
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
            step_parts.append(f"{index}. {cleaned}")

        documents = [str(item).strip() for item in (guidance.get("documents_to_keep") or []) if str(item).strip()]
        if documents:
            step_parts.append("Keep with you: " + ", ".join(documents[:6]) + ".")

        caution = str(guidance.get("caution") or "").strip()
        if caution:
            step_parts.append("Also keep in mind: " + caution)

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
                " so the answer should focus on preserving evidence, asking for refund or replacement, and escalating to FSSAI/Food Safety Connect or the National Consumer Helpline if needed."
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
                "This appears to be a consumer dispute, so the useful path is a written seller or platform demand, proof-backed escalation, and a clear request for refund, replacement, repair, compensation, or costs."
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
        del query
        return " for the issue already described"

    @staticmethod
    def _sanitize_fact_text(text: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        cleaned = re.sub(r"^[\"'`]+|[\"'`]+$", "", cleaned)
        cleaned = re.sub(r"^(i|we)\s+(already\s+)?(have|had|received|sent|reported|filed|complained|emailed|got|am|was|were)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(there is|there are|it is|it's)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" ,.;:-")
        return cleaned

    def _summarize_fact(self, *, key: str, text: str | None, fallback: str) -> str:
        cleaned = self._sanitize_fact_text(text)
        lowered = cleaned.lower()
        if not cleaned:
            return fallback

        if key == "incident_timing":
            if any(token in lowered for token in {"today", "yesterday", "morning", "afternoon", "evening", "night"}):
                return cleaned
            return f"the reported timing details ({cleaned})"
        if key == "reporting_status":
            if "1930" in lowered and ("sbi" in lowered or "bank" in lowered):
                return "reported to the bank and 1930 with a complaint reference"
            if "1930" in lowered:
                return "reported through 1930 with a complaint reference"
            if any(token in lowered for token in {"complaint number", "complaint reference", "reference"}) and any(token in lowered for token in {"bank", "reported", "complained"}):
                return "reported with a complaint reference"
            return "the current reporting stage"
        if key == "amount_details":
            amount_match = re.search(r"(rs\.?\s*\d[\d,]*|\d[\d,]*\s*rupees?)", cleaned, re.IGNORECASE)
            if amount_match and any(token in lowered for token in {"transaction id", "utr"}):
                return f"the affected transaction worth {amount_match.group(1).strip()} and the transaction ID"
            if amount_match:
                return f"the affected transaction worth {amount_match.group(1).strip()}"
            if any(token in lowered for token in {"transaction id", "utr"}):
                return "the transaction ID and related debit details"
            return fallback
        if key == "proof_details":
            details: list[str] = []
            if "imei" in lowered:
                details.append("IMEI")
            if any(token in lowered for token in {"invoice", "bill", "receipt"}):
                details.append("invoice")
            if "box" in lowered:
                details.append("box details")
            if details:
                return "proof such as " + " and ".join(details)
            return fallback
        if key == "harassment_details":
            if "eviction" in lowered:
                return "eviction pressure"
            if "lockout" in lowered:
                return "lockout pressure"
            if "threat" in lowered:
                return "threats or intimidation"
            if "deposit" in lowered:
                return "a deposit dispute"
            return fallback
        if key == "record_status":
            if any(token in lowered for token in {"agreement", "rent"}):
                return "the tenancy records already available"
            if any(token in lowered for token in {"message", "call", "record"}):
                return "the available communication record"
            return fallback
        if key == "product_details":
            product = "the product details"
            if "chocolate" in lowered:
                product = "the sealed chocolate packet"
            elif "packet" in lowered:
                product = "the packet details"
            extras: list[str] = []
            if any(token in lowered for token in {"invoice", "bill"}):
                extras.append("invoice")
            if any(token in lowered for token in {"photo", "photos", "photograph"}):
                extras.append("photographs")
            if extras:
                return f"{product} with the " + " and ".join(extras)
            return product
        if key == "health_effect":
            if any(token in lowered for token in {"photo", "photos", "photograph", "video"}):
                return "the photographs or video of the defect"
            if any(token in lowered for token in {"ill", "illness", "sick", "vomit", "medical"}):
                return "the health impact and any medical record"
            return fallback
        if key == "seller_contact":
            if any(token in lowered for token in {"email", "emailed"}):
                return "the earlier seller complaint sent by email"
            if "seller" in lowered or "brand" in lowered or "platform" in lowered:
                return "the earlier seller or brand complaint"
            return fallback
        if key == "purchase_details":
            if any(token in lowered for token in {"invoice", "bill", "payment", "paid", "order"}):
                return "the purchase and payment record"
            return fallback
        if key == "complaint_status":
            if any(token in lowered for token in {"email", "emailed"}):
                return "the written complaint already sent by email"
            if any(token in lowered for token in {"ticket", "support"}):
                return "the complaint record already opened with the seller or platform"
            return fallback
        if key == "relief_sought":
            if "refund" in lowered:
                return "refund"
            if "replacement" in lowered:
                return "replacement"
            if "repair" in lowered:
                return "repair"
            if "compensation" in lowered:
                return "compensation"
            return fallback
        if key == "incident_details":
            if "fraud" in lowered:
                return "the underlying fraud complaint"
            if "theft" in lowered or "stolen" in lowered or "snatch" in lowered:
                return "the underlying theft complaint"
            if "threat" in lowered:
                return "the underlying threat complaint"
            return fallback
        if key == "submission_status":
            if any(token in lowered for token in {"receipt", "diary", "submission", "written complaint"}):
                return "the available complaint submission record"
            return fallback
        if key == "notice_stage_detail":
            if "received" in lowered:
                return "the notice already received"
            if "reply" in lowered:
                return "the reply stage"
            if "send" in lowered:
                return "the notice-drafting stage"
            return fallback
        if key == "notice_content":
            if "payment" in lowered:
                return "the payment demand and deadline"
            if "deadline" in lowered:
                return "the stated demand and deadline"
            return fallback
        if key == "document_purpose":
            return cleaned[:80]
        if key == "blocker_details":
            return cleaned[:80]
        if key == "deadline_details":
            return "the applicable filing deadline"
        return fallback

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
        if key == "incident_timing" and state.issue_type == "cyber_fraud":
            if self._message_has_timing_signal(lowered):
                return cleaned
            return str((state.collected_facts or {}).get("incident_timing") or "").strip()
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
        return bool(
            re.search(
                r"\b(today|yesterday|morning|afternoon|evening|night|am|pm|ago|when|date)\b",
                normalized,
            )
            or re.search(r"\b(at|around)\b", normalized)
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
            timing = self._summarize_fact(key="incident_timing", text=facts.get("incident_timing"), fallback="the recent incident")
            reporting = self._summarize_fact(key="reporting_status", text=facts.get("reporting_status"), fallback="the current reporting stage")
            amount = self._summarize_fact(key="amount_details", text=facts.get("amount_details"), fallback="the affected transaction details")
            return (
                f"Treat this as a cyber-fraud complaint involving {bank_platform}. Act quickly. Preserve {amount}. Keep the complaint trail consistent. Reporting soon after {timing} usually improves the chances of tracing the loss. If you want the legal framing, cheating is commonly covered under BNS section 318, personation under BNS section 319, and the police complaint route usually begins through BNSS section 173. At the moment, {reporting} is the part that will affect the next step most."
            )
        if issue_type == "snatching_theft":
            station = facts.get("station_details") or state.city or "the local police station"
            timing = self._summarize_fact(key="incident_timing", text=facts.get("incident_timing"), fallback="the incident")
            police_status = self._summarize_fact(key="complaint_status", text=facts.get("police_status"), fallback="the complaint stage")
            proof = (
                self._summarize_fact(key="proof_details", text=facts.get("proof_details"), fallback="")
                or self._extract_query_evidence_detail(query, {"imei", "invoice", "bill", "receipt", "serial", "box", "proof"})
                or "the device and evidence details"
            )
            return (
                f"This should be handled as a theft or snatching matter. First secure the phone, SIM, and linked accounts. Then make sure {station} receives a clear complaint covering the incident, the present police stage ({police_status}), and proof such as {proof}. Since this happened around {timing}, it is better to move quickly. If you want the legal framing, theft is generally covered by BNS section 303, snatching by BNS section 304, and the complaint route usually starts through BNSS section 173."
            )
        if issue_type == "landlord_harassment":
            details = self._summarize_fact(key="harassment_details", text=facts.get("harassment_details"), fallback="the landlord pressure")
            records = self._summarize_fact(key="record_status", text=facts.get("record_status"), fallback="the tenancy records")
            return (
                f"Keep this calm and document everything. Preserve {records}. Avoid reacting only through calls or arguments. If the issue is about {details}, the next step depends on whether there is immediate risk of lockout, force, or threats. If there are threats or intimidation, there may also be a criminal angle, for example BNS section 351 on criminal intimidation. Otherwise the matter is usually handled through the tenancy or civil side."
            )
        if issue_type == "food_safety":
            product = self._summarize_fact(key="product_details", text=facts.get("product_details"), fallback="the product and packet details")
            health = self._summarize_fact(key="health_effect", text=facts.get("health_effect"), fallback="the health impact and photographs")
            seller_contact = self._summarize_fact(key="seller_contact", text=facts.get("seller_contact"), fallback="the seller or brand complaint stage")
            return (
                f"Handle this first as a food-safety or consumer complaint. Keep {product}, {health}, and {seller_contact} aligned in one record. That will make the matter easier to prove if you need refund, replacement, or escalation. Criminal-law sections are usually not the starting point here unless the facts show something more serious such as deliberate deception or a separate harmful act."
            )
        if issue_type == "consumer":
            purchase = self._summarize_fact(key="purchase_details", text=facts.get("purchase_details"), fallback="the purchase details")
            complaint = self._summarize_fact(key="complaint_status", text=facts.get("complaint_status"), fallback="the complaint status")
            relief = self._summarize_fact(key="relief_sought", text=facts.get("relief_sought"), fallback="the relief you want")
            return (
                f"Treat this as a consumer matter first. Keep {purchase} and {complaint} in one written record. Make sure your demand for {relief} is clear. That is usually more useful than debating legal theory at this stage. It only becomes a criminal-law issue if the facts also show something separate such as cheating, intimidation, or another clear offence."
            )
        if issue_type == "fir_refusal":
            incident = self._summarize_fact(key="incident_details", text=facts.get("incident_details"), fallback="the underlying incident")
            station = facts.get("station_details") or "the concerned police station"
            submission = self._summarize_fact(key="submission_status", text=facts.get("submission_status"), fallback="the present submission record")
            if not self._facts_clearly_indicate_criminal_issue(incident):
                return (
                    f"This is mainly a complaint-registration problem. If {station} has not acted on {incident}, preserve {submission} and escalate the same written complaint properly. Do not rely on repeated oral requests."
                )
            return (
                f"This is an FIR-registration issue. If {station} has not acted on {incident}, preserve {submission} and escalate the same written complaint properly. Do not rely on repeated oral requests. If you want the legal context, the complaint route usually begins through BNSS section 173."
            )
        if issue_type == "notice":
            stage = self._summarize_fact(key="notice_stage_detail", text=facts.get("notice_stage_detail"), fallback="the notice stage")
            content = self._summarize_fact(key="notice_content", text=facts.get("notice_content"), fallback="the demand and deadline")
            records = self._summarize_fact(key="record_status", text=facts.get("record_status"), fallback="the agreement and supporting record")
            return (
                f"This is a notice-stage matter. Keep it practical. Organise {stage}, {content}, and {records} in one file before replying. That will usually tell you whether the next step should be a reply, rebuttal, settlement discussion, or further drafting. Criminal-law sections are usually not the main issue here unless the notice facts also show a separate offence."
            )
        if issue_type == "police_complaint":
            incident = self._summarize_fact(key="incident_details", text=facts.get("incident_details"), fallback="the incident facts")
            station = facts.get("station_details") or "the relevant police station"
            proof = self._summarize_fact(key="proof_details", text=facts.get("proof_details"), fallback="the available supporting proof")
            if not self._facts_clearly_indicate_criminal_issue(incident):
                return (
                    f"This is a police-complaint matter. Reduce {incident} into a short written chronology for {station}. Support it with {proof}. That will make the complaint record clear from the first submission."
                )
            return (
                f"This is a police-complaint matter. Reduce {incident} into a short written chronology for {station} and support it with {proof}. That will make the complaint record clear from the first submission. If you want the legal context, the complaint route usually starts through BNSS section 173."
            )
        if issue_type == "documents":
            purpose = self._summarize_fact(key="document_purpose", text=facts.get("document_purpose"), fallback="the filing purpose")
            blocker = self._summarize_fact(key="blocker_details", text=facts.get("blocker_details"), fallback="the present blocker")
            deadline = self._summarize_fact(key="deadline_details", text=facts.get("deadline_details"), fallback="the deadline position")
            return (
                f"This is mainly a filing or document-process problem. First clarify {purpose}. Then resolve {blocker}. Keep proof of submission in view because {deadline} may affect the next step. Criminal-law theory is usually not the useful starting point here."
            )
        return (
            "The next step depends on getting the facts, timeline, and supporting record into one clear version first. Once that is done, it becomes much easier to decide which authority to approach and what to do next."
        )

    def _compose_virtual_advocate_steps(self, *, state: ConversationState, facts: dict[str, str]) -> list[str]:
        issue_type = state.issue_type or "general"
        if issue_type == "cyber_fraud":
            steps = []
            if not self._fact_indicates_completed_action(facts.get("reporting_status"), {"bank complaint", "reported", "1930", "cyber crime portal"}):
                steps.append("Report the matter immediately to your bank or payment app and through 1930 or the National Cyber Crime Portal, and keep every complaint or service reference number.")
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
                steps.append("Send a written complaint to the seller, brand, or platform asking for refund or replacement, and keep the email, ticket number, or chat acknowledgement.")
            steps.append("If it appears unsafe or caused illness, file on FSSAI Food Safety Connect at https://foscos.fssai.gov.in/consumergrievance/ and keep the grievance number.")
            steps.append("For refund, replacement, compensation, or complaint costs, raise the matter on the National Consumer Helpline at https://consumerhelpline.gov.in/ or call 1915.")
            steps.append("If the seller still does not resolve it, use the same record for a Consumer Commission complaint through https://edaakhil.nic.in/.")
            return steps
        if issue_type == "consumer":
            steps = []
            if not self._fact_indicates_completed_action(facts.get("complaint_status"), {"complaint", "emailed", "written", "ticket", "support"}):
                steps.append("Put the complaint in writing to the seller, platform, or service provider and ask clearly for refund, replacement, repair, compensation, or costs.")
            steps.extend(
                [
                    "Collect the invoice, payment proof, listing or warranty terms, chats, emails, defect photos, and complaint numbers.",
                    "If there is no timely response, escalate through the National Consumer Helpline at https://consumerhelpline.gov.in/ or call 1915.",
                    "If the issue remains unresolved, prepare an e-Daakhil Consumer Commission filing at https://edaakhil.nic.in/ with the chronology, proof, and relief amount.",
                    "If you want, I can help draft the seller complaint, Consumer Helpline text, or Consumer Commission complaint summary.",
                ]
            )
            return steps
        if issue_type == "fir_refusal":
            return [
                "Keep a dated copy of the original complaint and any proof of refusal or non-registration.",
                "Escalate the same complaint in writing to the senior police officer with the supporting record.",
                "Keep the delivery proof, diary note, or acknowledgement ready for the escalation record and any later proceeding.",
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
                "If the complaint is not recorded properly, escalate the same written version instead of starting over orally.",
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
            "food_safety": "Seller or Brand / FSSAI Food Safety Connect / National Consumer Helpline / Consumer Commission",
            "consumer": "Seller or Platform / National Consumer Helpline / Consumer Commission",
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
            "food_safety": ["packet", "batch number", "expiry date", "invoice", "photos or video", "medical record if any", "seller/FSSAI complaint number"],
            "consumer": ["invoice", "payment proof", "listing or warranty terms", "complaint copy", "photos or chats", "ticket numbers"],
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
            refs = ChatService._substantive_references_for_incident(incident)
            refs.extend(
                [
                    "BNSS section 173 (information in cognizable cases)",
                    "BNSS section 175 (police officer's power to investigate cognizable case)",
                ]
            )
            return refs
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
            refs = ChatService._substantive_references_for_incident(incident)
            refs.extend(
                [
                    "BNSS section 173 (information in cognizable cases)",
                    "BNSS section 175 (police officer's power to investigate cognizable case)",
                ]
            )
            return refs
        return []

    @staticmethod
    def _substantive_references_for_incident(text: str | None) -> list[str]:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized:
            return []
        refs: list[str] = []
        if any(token in normalized for token in {"fraud", "cheat", "cheating"}):
            refs.append("BNS section 318 (cheating)")
        if any(token in normalized for token in {"personation", "impersonation", "fake account", "fake profile", "pretending"}):
            refs.append("BNS section 319 (cheating by personation)")
        if any(token in normalized for token in {"threat", "threatening", "intimidation"}):
            refs.append("BNS section 351 (criminal intimidation)")
        if any(token in normalized for token in {"snatch", "snatching"}):
            refs.append("BNS section 304 (snatching)")
        if any(token in normalized for token in {"theft", "stolen", "robbed", "steal"}):
            refs.append("BNS section 303 (theft)")
        if any(token in normalized for token in {"assault", "hit", "beaten", "beating", "attack"}):
            refs.append("BNS section 115 (voluntarily causing hurt)")
        deduped: list[str] = []
        for ref in refs:
            if ref not in deduped:
                deduped.append(ref)
        return deduped

    def _legal_help_guidance(
        self,
        *,
        query: str,
        domain: str,
        conversation_state: ConversationState,
    ) -> dict[str, str | list[str] | None]:
        normalized = query.lower()
        issue_type = conversation_state.issue_type or self._detect_issue_type(normalized, domain)
        playbook = self._legal_help_playbook(issue_type)
        if issue_type == "cyber_fraud":
            guidance = self._cyber_fraud_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "fir_refusal":
            guidance = self._fir_refusal_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "landlord_harassment":
            guidance = self._landlord_harassment_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "food_safety":
            guidance = self._food_safety_guidance(query, conversation_state)
            return {**guidance, **playbook}
        if issue_type == "snatching_theft":
            guidance = self._snatching_theft_guidance(query, conversation_state)
            return {**guidance, **playbook}
        if issue_type == "consumer":
            guidance = self._consumer_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "notice":
            guidance = self._notice_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "police_complaint":
            guidance = self._police_guidance(conversation_state)
            return {**guidance, **playbook}
        if issue_type == "documents":
            guidance = self._documents_guidance(conversation_state)
            return {**guidance, **playbook}
        if any(token in normalized for token in {"cyber fraud", "upi", "wallet", "bank fraud", "phishing", "fake link", "payment link", "otp", "debit", "debited", "scam", "scammed", "olx", "cheated"}):
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
                **playbook,
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
                **playbook,
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
                **playbook,
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
                **playbook,
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
                **playbook,
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
            **playbook,
        }

    @staticmethod
    def _legal_help_playbook(issue_type: str) -> dict[str, Any]:
        playbooks = {
            "cyber_fraud": {
                "playbook_id": "playbook_cyber_fraud_v1",
                "risk_level": "high",
                "required_facts": ["bank_name_or_platform", "transaction_status", "reporting_status"],
            },
            "snatching_theft": {
                "playbook_id": "playbook_snatching_theft_v1",
                "risk_level": "high",
                "required_facts": ["city_or_police_station", "device_or_property_details"],
            },
            "fir_refusal": {
                "playbook_id": "playbook_fir_refusal_v1",
                "risk_level": "high",
                "required_facts": ["police_station", "complaint_status"],
            },
            "landlord_harassment": {
                "playbook_id": "playbook_landlord_harassment_v1",
                "risk_level": "medium",
                "required_facts": ["location", "pressure_type"],
            },
            "food_safety": {
                "playbook_id": "playbook_food_safety_v1",
                "risk_level": "medium",
                "required_facts": ["packet_invoice_photos", "seller_status"],
            },
            "consumer": {
                "playbook_id": "playbook_consumer_v1",
                "risk_level": "medium",
                "required_facts": ["seller_status", "proof_of_purchase"],
            },
            "notice": {
                "playbook_id": "playbook_notice_v1",
                "risk_level": "medium",
                "required_facts": ["notice_stage", "deadline_status"],
            },
            "documents": {
                "playbook_id": "playbook_documents_v1",
                "risk_level": "medium",
                "required_facts": ["document_type", "filing_authority"],
            },
            "police_complaint": {
                "playbook_id": "playbook_police_complaint_v1",
                "risk_level": "medium",
                "required_facts": ["city_or_police_station", "incident_details"],
            },
        }
        return playbooks.get(
            issue_type,
            {
                "playbook_id": "playbook_general_legal_help_v1",
                "risk_level": "medium",
                "required_facts": ["issue_summary", "authority_or_person"],
            },
        )

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
        if any(token in normalized for token in {"cyber fraud", "upi", "wallet", "bank fraud", "phishing", "fake link", "payment link", "otp", "debit", "debited", "scam", "scammed", "olx", "cheated"}):
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
                "summary": "This looks like a cyber-fraud or unauthorized debit matter, so the priority is to stop further loss and create a clear complaint record immediately.",
                "legal_position": "In a fake-link, OTP, or unauthorized debit situation, the case usually becomes stronger when you report it quickly, preserve the transaction trail, and keep the same facts across the bank and cyber-crime complaint.",
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
            "summary": f"This looks like a cyber-fraud complaint linked to {bank_or_platform}, so the right approach is immediate reporting, complaint registration, and careful record preservation.",
            "legal_position": f"Since {bank_or_platform} is already identified, use the same transaction facts with the bank or platform, the cyber-crime channel, and, if money has already gone out, the police or cyber cell as well.",
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
            "summary": "This appears to be a consumer dispute, so move in writing: demand a clear remedy from the seller or platform, then escalate with the same proof if they do not resolve it.",
            "legal_position": "Consumer matters are easier to pursue when the defect or service deficiency, purchase proof, complaint trail, and requested relief are specific. The relief can include refund, replacement, repair, compensation for loss or inconvenience, and complaint costs.",
            "practical_next_steps": [
                "Send a written complaint to the seller, platform, brand, or service provider asking for the exact remedy you want: refund, replacement, repair, compensation, or costs. Give a short deadline and keep the ticket number or email trail.",
                "Organise the invoice, payment proof, product listing or warranty terms, chats, emails, photos or video of the defect, delivery details, and every complaint reference number.",
                "If there is no proper response, escalate on the National Consumer Helpline at https://consumerhelpline.gov.in/ or call 1915, then use the same record for e-Daakhil at https://edaakhil.nic.in/ if a Consumer Commission complaint is needed.",
                "If you want, I can draft a short seller complaint, Consumer Helpline complaint text, or Consumer Commission summary from your facts.",
            ],
            "lines": [
                "Start with a written demand to the seller, platform, brand, or service provider and ask clearly for refund, replacement, repair, compensation, or costs.",
                "Keep the invoice, payment proof, warranty or listing page, chats, emails, photos or video, and ticket numbers together.",
                "If they do not resolve it, use the National Consumer Helpline at https://consumerhelpline.gov.in/ or 1915; for a formal Consumer Commission filing, prepare the same record for https://edaakhil.nic.in/.",
            ],
            "likely_forum": "Seller or Platform / National Consumer Helpline / Consumer Commission",
            "authorities": ["National Consumer Helpline", "Consumer Commission", "e-Daakhil", "seller or platform"],
            "documents_to_keep": ["invoice", "payment proof", "listing or warranty terms", "complaint copy", "photos or video", "ticket numbers"],
            "caution": "Make the relief specific; a complaint that only says the product was bad is weaker than one that asks for refund, replacement, repair, compensation, or costs with proof.",
            "follow_up_question": "Was any written complaint already sent to the seller or service provider?",
        }

    def _fir_refusal_guidance(self, state: ConversationState) -> dict[str, str | list[str] | None]:
        location = state.police_station or (f"{state.city} police station" if state.city else "the local police station")
        return {
            "summary": f"This is an FIR-registration refusal issue at {location}, so the focus should now be on preserving the complaint record and escalating it properly.",
            "legal_position": "When the local police do not register the complaint, the practical way forward is usually to preserve the written complaint, keep proof of submission, and move the same facts to the senior officer instead of relying on oral follow-up alone.",
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
            "summary": f"This looks like a landlord-harassment or eviction-pressure situation in {location}, so the focus should be on tenancy records, written proof, and safe escalation.",
            "legal_position": "Landlord disputes are much easier to handle when rent proof, possession records, messages, and any notice or threat are preserved before you respond or escalate the matter.",
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
            "summary": f"This appears to be a defective or unsafe-food complaint involving {product_hint}. Treat it as both an evidence issue and a consumer-relief issue: preserve the product, demand refund or replacement, and escalate if needed.",
            "legal_position": "For unsafe or contaminated food, FSSAI/Food Safety Connect is the practical safety-regulator route, while the National Consumer Helpline or Consumer Commission route helps with refund, replacement, compensation, and complaint costs.",
            "practical_next_steps": [
                f"Do not discard {product_hint}. Keep the sealed packet or remaining product, batch number, expiry date, invoice, photos or video, and any medical record if someone felt unwell.",
                "Send a written complaint to the seller, brand, or platform asking for refund or replacement and a written acknowledgement. Keep the ticket number, email, chat, or call record.",
                "For food-safety action, file on FSSAI Food Safety Connect at https://foscos.fssai.gov.in/consumergrievance/ with packet photos, batch details, invoice, and illness details if any.",
                "For refund, replacement, compensation, or costs, raise the matter on the National Consumer Helpline at https://consumerhelpline.gov.in/ or call 1915; if unresolved, prepare an e-Daakhil filing at https://edaakhil.nic.in/.",
                "If you want, I can draft the seller complaint or the FSSAI/Consumer Helpline complaint text from your product details.",
            ],
            "lines": [
                f"Preserve {product_hint}, the batch number, expiry date, invoice, and clear photos or video before consuming or throwing anything away.",
                "Ask the seller, brand, or platform in writing for refund or replacement and keep the acknowledgement or ticket number.",
                "Escalate safety concerns through FSSAI Food Safety Connect at https://foscos.fssai.gov.in/consumergrievance/; use https://consumerhelpline.gov.in/ or 1915 for consumer relief such as refund, replacement, or compensation.",
            ],
            "likely_forum": "Seller or Brand / FSSAI Food Safety Connect / National Consumer Helpline / Consumer Commission",
            "authorities": ["FSSAI", "Food Safety Connect", "National Consumer Helpline", "Consumer Commission", "e-Daakhil"],
            "documents_to_keep": ["product packet", "batch number", "expiry date", "invoice", "photos or video", "written complaint", "medical records if any", "complaint numbers"],
            "caution": "Do not rely only on a verbal complaint or throw away the packet; the batch details and defect evidence are what make refund, replacement, compensation, and safety escalation practical.",
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
                "summary": f"This is a police-complaint issue linked to {location}, so the next useful step is a clear written complaint with supporting proof and acknowledgement.",
                "legal_position": "Police-complaint matters usually move better when the incident chronology, names, and supporting records are placed in one written complaint and the submission is acknowledged.",
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

        statute_expansions = {
            "ipc": "Indian Penal Code",
            "crpc": "Code of Criminal Procedure",
            "cpc": "Code of Civil Procedure",
            "bns": "Bharatiya Nyaya Sanhita",
            "bnss": "Bharatiya Nagarik Suraksha Sanhita",
            "ni act": "Negotiable Instruments Act",
        }
        expanded = cleaned
        matched_statutes: list[tuple[str, str]] = []
        for alias, full_name in statute_expansions.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered, flags=re.IGNORECASE):
                matched_statutes.append((alias, full_name))
                expanded = re.sub(rf"\b{re.escape(alias)}\b", full_name, expanded, flags=re.IGNORECASE)
        expanded = re.sub(r"\s+", " ", expanded).strip()
        if expanded and expanded.lower() != lowered:
            variants.append(expanded)

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
        if section_match:
            section_value = section_match.group(1).upper()
            for alias, full_name in matched_statutes:
                variants.extend(
                    [
                        f"Section {section_value} {full_name}",
                        f"{full_name} Section {section_value}",
                        f"Section {section_value} {full_name} explanation",
                        f"{full_name} Section {section_value} official text",
                        f"{alias.upper()} Section {section_value}",
                    ]
                )
        if article_match:
            article_value = article_match.group(1).upper()
            variants.extend(
                [
                    f"Article {article_value} Constitution of India",
                    f"Constitution of India Article {article_value}",
                    f"Article {article_value} explanation",
                    f"Article {article_value} official text",
                ]
            )

        return self._dedupe_values(variants)

    @staticmethod
    def _is_statute_query(query: str) -> bool:
        normalized = query.lower()
        if re.search(r"\bsection\s+[0-9]+[a-z]?\b", normalized):
            return True
        if re.search(r"\barticle\s+[0-9]+[a-z]?\b", normalized):
            return True
        if re.search(r"\brule\s+[0-9]+[a-z]?\b", normalized):
            return True
        shorthand_signals = [
            "ipc",
            "crpc",
            "cpc",
            "bns",
            "bnss",
            "constitution",
            "ni act",
            "negotiable instruments act",
        ]
        if any(signal in normalized for signal in shorthand_signals):
            return True
        return bool(re.search(r"\b[a-z][a-z .,&()-]{2,80}\s+act\b", normalized))

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
