from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from backend.app.core.config import DATA_DIR, Settings
from backend.app.services.google_custom_search_service import GoogleCustomSearchService
from backend.app.services.indiankanoon_service import IndianKanoonService, SOURCE_AUTHORITY_SCORES
from backend.app.services.legal_dataset_service import LocalLegalDatasetService
from backend.app.utils.request_context import get_logger
from training.embeddings import HashingEmbeddingProvider


logger = get_logger("lawyer_ai.legal_hybrid_retrieval")
DEFAULT_CORPUS_DIR = DATA_DIR / "training_ingestion" / "learning_data_corpus"


@dataclass(slots=True)
class HybridRetrievalResult:
    local_dataset_count: int
    documents: list[dict[str, Any]]
    internal_count: int
    live_count: int
    google_count: int
    curated_google_count: int
    general_google_count: int
    internal_confidence: float
    live_used: bool
    google_used: bool
    source_summary: list[str]


@dataclass(slots=True)
class CorpusRecord:
    doc_id: str
    source_id: str
    title: str
    text: str
    embedding_text: str
    embedding: list[float]
    metadata: dict[str, str]


class LegalCorpusIndex:
    def __init__(self, corpus_dir: Path | None = None, embedding_provider: HashingEmbeddingProvider | None = None) -> None:
        self.corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self._records = self._load_records()

    def search(self, query: str, *, domain: str | None, state: str | None, max_results: int = 6) -> list[dict[str, Any]]:
        if not self._records:
            return []

        query_vector = self.embedding_provider.embed_texts([query])[0]
        query_tokens = self._tokenize(query)
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        for record in self._records:
            semantic = self._cosine_similarity(query_vector, record.embedding)
            keyword = self._keyword_score(query_tokens, record.text, record.embedding_text)
            metadata_boost = self._metadata_boost(record.metadata, domain=domain, state=state, query=query_lower)
            score = round((semantic * 65.0) + (keyword * 8.0) + metadata_boost, 3)
            if score <= 0:
                continue
            results.append(self._to_document(record, score=score))

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:max_results]

    def _to_document(self, record: CorpusRecord, *, score: float) -> dict[str, Any]:
        metadata = record.metadata
        title = metadata.get("title") or record.title or "Untitled"
        source_file = metadata.get("source_filename") or metadata.get("file_name") or record.source_id
        docsource = metadata.get("legal_domain") or metadata.get("document_kind") or "internal_corpus"
        record_kind = metadata.get("document_kind") or ""
        excerpt = record.text.strip()
        headline = self._build_headline(excerpt, title)
        citations = [metadata.get("citation_hint") or source_file]
        payload = {
            "doc_id": f"internal:{record.doc_id}",
            "title": title,
            "headline": headline,
            "fragment_headline": headline,
            "fragment_excerpt": excerpt[:2000],
            "doc_excerpt": excerpt[:3200],
            "docsource": f"internal:{docsource}",
            "citations": [item for item in citations if item],
            "publishdate": metadata.get("year") or metadata.get("date") or "",
            "url": metadata.get("source_path") or metadata.get("source") or "",
            "score": score,
            "source_kind": "internal",
            "legal_domain": metadata.get("legal_domain") or "",
            "document_kind": metadata.get("document_kind") or "",
            "source_filename": source_file,
            "jurisdiction": metadata.get("jurisdiction") or metadata.get("state") or "",
            "authority_type": metadata.get("authority_type") or record_kind or "internal_guidance",
            "recency_bucket": self._recency_bucket(metadata.get("year") or metadata.get("date") or ""),
            "metadata_confidence": self._metadata_confidence(metadata),
        }
        return payload

    def _load_records(self) -> list[CorpusRecord]:
        vectors_path = self.corpus_dir / "vectors.jsonl"
        chunks_path = self.corpus_dir / "chunks.jsonl"
        path = vectors_path if vectors_path.exists() else chunks_path
        if not path.exists():
            return []

        records: list[CorpusRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = {str(key): str(value) for key, value in (payload.get("metadata") or {}).items()}
                embedding = payload.get("embedding") or []
                if not isinstance(embedding, list):
                    embedding = []
                text = str(payload.get("text") or payload.get("cleaned_text") or "")
                embedding_text = str(payload.get("embedding_text") or text)
                records.append(
                    CorpusRecord(
                        doc_id=str(payload.get("id") or payload.get("source_id") or len(records) + 1),
                        source_id=str(payload.get("source_id") or ""),
                        title=str(payload.get("title") or metadata.get("title") or ""),
                        text=text,
                        embedding_text=embedding_text,
                        embedding=[float(value) for value in embedding],
                        metadata=metadata,
                    )
                )
        logger.info("loaded internal corpus index path=%s records=%s", path, len(records))
        return records

    @staticmethod
    def _build_headline(text: str, title: str) -> str:
        snippet = re.sub(r"\s+", " ", text).strip()
        if not snippet:
            return title
        return snippet[:240]

    @staticmethod
    def _recency_bucket(value: str) -> str:
        text = str(value or "").strip()
        if len(text) >= 4 and text[:4].isdigit():
            year = int(text[:4])
            if year >= 2024:
                return "current"
            if year >= 2021:
                return "recent"
            return "dated"
        return "unknown"

    @staticmethod
    def _metadata_confidence(metadata: dict[str, str]) -> float:
        confidence = 0.45
        if metadata.get("legal_domain"):
            confidence += 0.15
        if metadata.get("document_kind"):
            confidence += 0.15
        if metadata.get("jurisdiction") or metadata.get("state"):
            confidence += 0.1
        if metadata.get("citation_hint"):
            confidence += 0.1
        return min(confidence, 0.95)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}

    def _keyword_score(self, query_tokens: set[str], text: str, embedding_text: str) -> float:
        target_tokens = self._tokenize(f"{text} {embedding_text}")
        overlap = len(query_tokens & target_tokens)
        if not query_tokens:
            return 0.0
        return overlap / max(len(query_tokens), 1)

    def _metadata_boost(self, metadata: dict[str, str], *, domain: str | None, state: str | None, query: str) -> float:
        boost = 0.0
        record_domain = (metadata.get("legal_domain") or "").lower()
        record_kind = (metadata.get("document_kind") or "").lower()
        record_text = " ".join([metadata.get("title", ""), metadata.get("citation_hint", ""), metadata.get("source_filename", "")]).lower()

        if domain and domain.lower() == record_domain:
            boost += 9.0
        if state and state.lower() in record_text:
            boost += 3.0
        if any(token in query for token in {"section", "article", "rule", "statute"}):
            if record_kind in {"statute", "rule"}:
                boost += 6.0
        if any(token in query for token in {"judgment", "case", "precedent", "court"}):
            if record_kind in {"judgment", "order"}:
                boost += 6.0
        if any(token in query for token in {"latest", "recent", "current", "updated"}):
            year = metadata.get("year") or ""
            if year and year[:4].isdigit():
                boost += min(max(int(year[:4]) - 2018, 0), 6)
        return boost

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        limit = min(len(left), len(right))
        dot = sum(left[index] * right[index] for index in range(limit))
        left_norm = math.sqrt(sum(value * value for value in left[:limit]))
        right_norm = math.sqrt(sum(value * value for value in right[:limit]))
        if not left_norm or not right_norm:
            return 0.0
        return max(min(dot / (left_norm * right_norm), 1.0), -1.0)


class LegalHybridRetrievalService:
    def __init__(
        self,
        settings: Settings,
        *,
        corpus_index: LegalCorpusIndex | None = None,
        indiankanoon_service: IndianKanoonService | None = None,
        google_search_service: GoogleCustomSearchService | None = None,
        legal_dataset_service: LocalLegalDatasetService | None = None,
    ) -> None:
        self.settings = settings
        self.corpus_index = corpus_index or LegalCorpusIndex()
        self.indiankanoon_service = indiankanoon_service or IndianKanoonService(settings)
        self.google_search_service = google_search_service or GoogleCustomSearchService(settings)
        self.legal_dataset_service = legal_dataset_service or LocalLegalDatasetService()

    def retrieve(
        self,
        *,
        query: str,
        query_variants: list[str],
        state: str | None,
        domain: str | None,
        answer_mode: str,
        query_type: str,
        doctypes_options: list[str | None],
        curated_google_query: str | None = None,
    ) -> HybridRetrievalResult:
        local_dataset_documents = (
            self._search_local_dataset(query=query, query_variants=query_variants)
            if query_type == "provision_lookup"
            else []
        )
        if local_dataset_documents:
            logger.info(
                "hybrid retrieval source=local_legal_dataset query=%r docs=%s",
                query[:120],
                len(local_dataset_documents),
            )
            return HybridRetrievalResult(
                local_dataset_count=len(local_dataset_documents),
                documents=local_dataset_documents,
                internal_count=0,
                live_count=0,
                google_count=0,
                curated_google_count=0,
                general_google_count=0,
                internal_confidence=1.0,
                live_used=False,
                google_used=False,
                source_summary=self._build_source_summary(local_dataset_documents),
            )

        google_query = str(curated_google_query or query or "").strip() or query
        source_order = self._source_order_for_query(
            query=query,
            domain=domain,
            answer_mode=answer_mode,
            query_type=query_type,
        )
        curated_google_documents: list[dict[str, Any]] = []
        internal_documents: list[dict[str, Any]] = []
        live_documents: list[dict[str, Any]] = []
        google_documents: list[dict[str, Any]] = []
        internal_confidence = 0.0
        selected_source = ""
        selected_confidence = 0.0
        selected_confidence_level = "weak"
        exhaustive_provision_lookup = query_type == "provision_lookup"

        for source in source_order:
            documents: list[dict[str, Any]] = []
            if source == "internal":
                internal_documents = self._search_internal(
                    query=query,
                    query_variants=query_variants,
                    state=state,
                    domain=domain,
                )
                internal_confidence = self._internal_confidence(
                    internal_documents=internal_documents,
                    answer_mode=answer_mode,
                    query=query,
                )
                documents = internal_documents
            elif source == "indiankanoon":
                if not self.indiankanoon_service.configured:
                    continue
                try:
                    live_documents = self.indiankanoon_service.retrieve_grounded_documents(
                        query_variants=query_variants,
                        doctypes_options=doctypes_options,
                        max_results=4,
                    )
                except Exception:
                    logger.exception("hybrid retrieval source=indiankanoon failed query=%r", query[:120])
                    live_documents = []
                documents = live_documents
            elif source == "google":
                if not self.google_search_service.configured:
                    continue
                site_restrict = self._google_site_restrict(google_query)
                curated_google_documents = self._search_google(
                    query=google_query,
                    site_restrict=site_restrict,
                    trusted_only=True,
                )
                curated_confidence = self._source_confidence(
                    source="google",
                    documents=curated_google_documents,
                    answer_mode=answer_mode,
                    query=query,
                )
                if self._confidence_level(curated_confidence) == "weak":
                    google_documents = self._search_google(
                        query=google_query,
                        site_restrict=site_restrict,
                        trusted_only=False,
                    )
                documents = [*curated_google_documents, *google_documents]
            else:
                continue

            confidence = self._source_confidence(
                source=source,
                documents=documents,
                answer_mode=answer_mode,
                query=query,
            )
            confidence_level = self._confidence_level(confidence)
            self._annotate_source_confidence(
                documents,
                source=source,
                confidence=confidence,
                confidence_level=confidence_level,
            )
            logger.info(
                "hybrid retrieval source=%s query=%r docs=%s confidence=%.2f level=%s order=%s",
                source,
                query[:120],
                len(documents),
                confidence,
                confidence_level,
                source_order,
            )
            if exhaustive_provision_lookup:
                if not selected_source and documents:
                    selected_source = source
                    selected_confidence = confidence
                    selected_confidence_level = confidence_level
                continue
            if documents and confidence_level != "weak":
                selected_source = source
                selected_confidence = confidence
                selected_confidence_level = confidence_level
                break

        merged_documents = (
            self._merge_provision_documents(
                live_documents=live_documents,
                internal_documents=internal_documents,
                curated_google_documents=curated_google_documents,
                google_documents=google_documents,
            )
            if exhaustive_provision_lookup
            else (
                self._documents_for_selected_source(
                    selected_source=selected_source,
                    curated_google_documents=curated_google_documents,
                    internal_documents=internal_documents,
                    live_documents=live_documents,
                    google_documents=google_documents,
                )
                if selected_source
                else self._merge_documents(curated_google_documents, internal_documents, live_documents, google_documents)
            )
        )
        logger.info(
            "hybrid retrieval query=%r selected_source=%s selected_confidence=%.2f selected_level=%s internal=%s live=%s curated_google=%s google=%s confidence=%.2f",
            query[:120],
            selected_source or "none",
            selected_confidence,
            selected_confidence_level,
            len(internal_documents),
            len(live_documents),
            len(curated_google_documents),
            len(google_documents),
            internal_confidence,
        )
        return HybridRetrievalResult(
            local_dataset_count=0,
            documents=merged_documents,
            internal_count=len(internal_documents),
            live_count=len(live_documents),
            google_count=len(curated_google_documents) + len(google_documents),
            curated_google_count=len(curated_google_documents),
            general_google_count=len(google_documents),
            internal_confidence=internal_confidence,
            live_used=bool(live_documents),
            google_used=bool(curated_google_documents or google_documents),
            source_summary=self._build_source_summary(merged_documents),
        )

    def _search_local_dataset(self, *, query: str, query_variants: list[str]) -> list[dict[str, Any]]:
        variants = query_variants or [query]
        seen_doc_ids: set[str] = set()
        documents: list[dict[str, Any]] = []
        for variant in variants:
            match = self.legal_dataset_service.lookup_query(variant)
            if match is None:
                continue
            document = match.to_document()
            doc_id = str(document.get("doc_id") or "")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            documents.append(document)
        return documents

    def _search_google(self, *, query: str, site_restrict: str | None, trusted_only: bool) -> list[dict[str, Any]]:
        try:
            result = self.google_search_service.search(
                query=query,
                max_results=min(self.settings.google_search_max_results, 3),
                site_restrict=site_restrict,
                trusted_only=trusted_only,
            )
            return result.documents
        except Exception:
            logger.exception(
                "hybrid retrieval source=google failed query=%r trusted_only=%s",
                query[:120],
                trusted_only,
            )
            return []

    def _source_order_for_query(self, *, query: str, domain: str | None, answer_mode: str, query_type: str) -> list[str]:
        if query_type == "provision_lookup":
            return ["indiankanoon", "google"]
        if query_type == "general_legal_research":
            return ["indiankanoon", "google"]
        if query_type == "uploaded_document_query":
            return ["internal", "indiankanoon", "google"]

        normalized = query.lower()
        case_or_statute = answer_mode in {"case_first", "statute_first"} or any(
            token in normalized
            for token in {
                "judgment",
                "judgement",
                "case law",
                "precedent",
                "citation",
                "legal position",
                "section ",
                "article ",
                "rule ",
                "act ",
                "ipc",
                "bns",
                "bnss",
                "constitution",
            }
        )
        latest_or_current = any(token in normalized for token in {"latest", "recent", "current", "updated"})
        official_portal = self._requires_official_or_latest_external_source(normalized)
        if case_or_statute or latest_or_current:
            return ["indiankanoon", "google"]
        if official_portal:
            return ["internal", "indiankanoon", "google"]
        if (domain or "").lower() in {"constitutional", "criminal", "consumer", "civil", "property", "tax", "corporate"}:
            return ["indiankanoon", "google"]
        return ["internal", "indiankanoon", "google"]

    def _source_confidence(self, *, source: str, documents: list[dict[str, Any]], answer_mode: str, query: str) -> float:
        if not documents:
            return 0.0
        if source == "internal":
            return self._internal_confidence(internal_documents=documents, answer_mode=answer_mode, query=query)
        if source == "indiankanoon":
            return self._live_confidence(live_documents=documents, answer_mode=answer_mode, query=query)
        if source == "google":
            return self._google_confidence(documents=documents, answer_mode=answer_mode, query=query)
        return 0.0

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        if confidence >= 0.72:
            return "strong"
        if confidence >= 0.52:
            return "medium"
        return "weak"

    @staticmethod
    def _annotate_source_confidence(
        documents: list[dict[str, Any]],
        *,
        source: str,
        confidence: float,
        confidence_level: str,
    ) -> None:
        for doc in documents:
            if source == "indiankanoon" and not doc.get("source_kind"):
                doc["source_kind"] = "indiankanoon"
            elif source == "internal" and not doc.get("source_kind"):
                doc["source_kind"] = "internal"
            elif source == "google" and not doc.get("source_kind"):
                doc["source_kind"] = "google_custom_search"
            doc["retrieval_source"] = source
            doc["retrieval_confidence"] = round(confidence, 3)
            doc["retrieval_confidence_level"] = confidence_level

    @staticmethod
    def _documents_for_selected_source(
        *,
        selected_source: str,
        curated_google_documents: list[dict[str, Any]],
        internal_documents: list[dict[str, Any]],
        live_documents: list[dict[str, Any]],
        google_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if selected_source == "indiankanoon":
            return LegalHybridRetrievalService._merge_documents([], [], live_documents, [])
        if selected_source == "internal":
            return LegalHybridRetrievalService._merge_documents([], internal_documents, [], [])
        if selected_source == "google":
            return LegalHybridRetrievalService._merge_documents(curated_google_documents, [], [], google_documents)
        return []

    @staticmethod
    def _merge_provision_documents(
        *,
        live_documents: list[dict[str, Any]],
        internal_documents: list[dict[str, Any]],
        curated_google_documents: list[dict[str, Any]],
        google_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in (
            sorted(live_documents, key=lambda item: float(item.get("score") or 0.0), reverse=True),
            sorted(internal_documents, key=lambda item: float(item.get("score") or 0.0), reverse=True),
            sorted(curated_google_documents, key=lambda item: float(item.get("score") or 0.0), reverse=True),
            sorted(google_documents, key=lambda item: float(item.get("score") or 0.0), reverse=True),
        ):
            for doc in group:
                source_key = str(doc.get("doc_id") or doc.get("url") or doc.get("title") or "")
                if not source_key or source_key in seen:
                    continue
                seen.add(source_key)
                merged.append(doc)
                if len(merged) >= 8:
                    return merged
        return merged[:8]

    @staticmethod
    def _should_query_curated_google_primary(*, query: str, domain: str | None, answer_mode: str) -> bool:
        normalized = query.lower()
        constitutional_signals = {
            "constitution",
            "fundamental rights",
            "fundamental duties",
            "directive principles",
            "dpsp",
        }
        has_article_marker = bool(re.search(r"\barticle\s+\d+[a-z]?\b", normalized))
        has_authority_marker = bool(re.search(r"\bsection\s+\d+[a-z]?\b", normalized)) or bool(
            re.search(r"\brule\s+\d+[a-z]?\b", normalized)
        ) or any(
            re.search(rf"\b{re.escape(marker)}\b", normalized)
            for marker in {"act", "ipc", "bns", "bnss", "ni act"}
        )
        if domain and domain.lower() == "constitutional":
            return True
        if answer_mode in {"statute_first", "case_first"} and (
            has_article_marker or has_authority_marker or any(signal in normalized for signal in constitutional_signals)
        ):
            return True
        return has_article_marker or any(signal in normalized for signal in constitutional_signals)

    def _search_internal(self, *, query: str, query_variants: list[str], state: str | None, domain: str | None) -> list[dict[str, Any]]:
        variants = query_variants or [query]
        combined: dict[str, dict[str, Any]] = {}
        for variant in variants:
            for document in self.corpus_index.search(variant, domain=domain, state=state, max_results=6):
                doc_id = str(document.get("doc_id") or "")
                if not doc_id:
                    continue
                current = combined.get(doc_id)
                if current is None or float(document.get("score") or 0.0) > float(current.get("score") or 0.0):
                    combined[doc_id] = document
        ranked = sorted(combined.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return ranked[:6]

    def _should_query_live(
        self,
        *,
        query: str,
        domain: str | None,
        answer_mode: str,
        curated_google_documents: list[dict[str, Any]],
        internal_documents: list[dict[str, Any]],
        internal_confidence: float,
    ) -> bool:
        normalized = query.lower()
        if self._trusted_curated_google_authority_documents_available(
            curated_google_documents=curated_google_documents,
            query=query,
            domain=domain,
            answer_mode=answer_mode,
        ):
            return False
        live_signals = [
            "latest",
            "recent",
            "current",
            "updated",
            "supreme court",
            "high court",
            "judgment",
            "judgement",
            "precedent",
            "legal position",
            "case law",
            "what is the law",
        ]
        has_live_signal = any(signal in normalized for signal in live_signals)
        internal_missing = not internal_documents
        weak_internal = internal_confidence < 0.55 or (internal_documents and float(internal_documents[0].get("score") or 0.0) < 18.0)
        outdated_internal = False
        if internal_documents and any(token in normalized for token in {"latest", "recent", "current", "updated"}):
            top = internal_documents[0]
            year = str(top.get("publishdate") or "").strip()
            if year and year[:4].isdigit():
                outdated_internal = int(year[:4]) < 2021

        authority_terms = {"section ", "article ", "rule ", "act ", "ipc", "bns", "bnss", "constitution"}
        authority_query = any(token in normalized for token in authority_terms)

        if answer_mode in {"case_first", "statute_first"} and (has_live_signal or authority_query):
            return True
        return internal_missing or weak_internal or outdated_internal or has_live_signal

    @classmethod
    def _trusted_curated_google_authority_documents_available(
        cls,
        *,
        curated_google_documents: list[dict[str, Any]],
        query: str,
        domain: str | None,
        answer_mode: str,
    ) -> bool:
        if not cls._curated_google_is_primary_lookup_query(
            query=query,
            domain=domain,
            answer_mode=answer_mode,
        ):
            return False
        if cls._curated_google_documents_strong(curated_google_documents):
            return True
        authoritative_kinds = {"government_portal", "regulator", "court_portal"}
        return any(
            str(doc.get("authority_type") or "").strip().lower() in authoritative_kinds
            and str(doc.get("source_kind") or "").strip().lower() == "google_custom_search"
            for doc in curated_google_documents
        )

    @staticmethod
    def _curated_google_is_primary_lookup_query(*, query: str, domain: str | None, answer_mode: str) -> bool:
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

        has_constitutional_marker = any(
            marker in normalized
            for marker in {
                "constitution",
                "fundamental rights",
                "fundamental duties",
                "directive principles",
                "dpsp",
            }
        ) or bool(re.search(r"\barticle\s+\d+[a-z]?\b", normalized))
        has_statute_marker = bool(re.search(r"\bsection\s+\d+[a-z]?\b", normalized)) or bool(
            re.search(r"\brule\s+\d+[a-z]?\b", normalized)
        ) or any(
            re.search(rf"\b{re.escape(marker)}\b", normalized)
            for marker in {"act", "ipc", "bns", "bnss", "ni act"}
        )
        if (domain or "").lower() == "constitutional":
            return True
        if has_statute_marker and answer_mode in {"statute_first", "case_first", "grounded"}:
            return True
        return has_constitutional_marker

    def _should_query_google(
        self,
        *,
        query: str,
        answer_mode: str,
        domain: str | None,
        curated_google_documents: list[dict[str, Any]],
        internal_documents: list[dict[str, Any]],
        internal_confidence: float,
        live_documents: list[dict[str, Any]],
    ) -> bool:
        if not self.google_search_service.configured:
            return False
        normalized = query.lower()
        if self._curated_google_documents_strong(curated_google_documents):
            return False
        official_need = self._requires_official_or_latest_external_source(normalized)
        constitutional_or_authority_query = (
            (domain or "").lower() == "constitutional"
            or any(token in normalized for token in {"constitution", "article ", "section ", "rule ", "act ", "ipc", "bns", "bnss"})
        )
        legal_authority_heavy = any(token in normalized for token in {"section ", "judgment", "judgement", "case law", "precedent", "citation"})
        if constitutional_or_authority_query:
            official_need = True
        if legal_authority_heavy and answer_mode in {"case_first", "statute_first"} and not official_need:
            return False
        if not official_need:
            return False
        if internal_confidence >= 0.72 and internal_documents:
            return False
        if live_documents and any(str(doc.get("authority_type") or "").lower() in {"government_portal", "regulator", "court_portal"} for doc in live_documents):
            return False
        internal_weak_or_missing = (not internal_documents) or internal_confidence < 0.58
        live_weak_or_missing = self._live_documents_weak_or_missing(live_documents)
        if not internal_weak_or_missing:
            return False
        if not live_weak_or_missing:
            return False
        return True

    @staticmethod
    def _curated_google_documents_strong(documents: list[dict[str, Any]]) -> bool:
        if not documents:
            return False
        authoritative_kinds = {"government_portal", "regulator", "court_portal"}
        return any(
            str(doc.get("authority_type") or "").lower() in authoritative_kinds and float(doc.get("score") or 0.0) >= 12.0
            for doc in documents
        )

    @staticmethod
    def _requires_official_or_latest_external_source(normalized_query: str) -> bool:
        official_source_signals = {
            "official",
            "portal",
            "website",
            "online",
            "complaint",
            "file complaint",
            "helpline",
            "rbi",
            "cybercrime",
            "government",
            "gov",
            "ministry",
            "police",
            "where to complain",
            "latest",
            "current",
            "updated",
            "notification",
            "circular",
        }
        return any(signal in normalized_query for signal in official_source_signals)

    @staticmethod
    def _live_documents_weak_or_missing(live_documents: list[dict[str, Any]]) -> bool:
        if not live_documents:
            return True
        authoritative_live_sources = {
            "supremecourt",
            "scorders",
            "laws",
            "constitution",
            "delhi",
            "bombay",
            "allahabad",
            "karnataka",
            "kerala",
            "chennai",
            "kolkata",
            "gujarat",
            "tribunals",
            "consumer",
        }
        strongest_score = max(float(doc.get("score") or 0.0) for doc in live_documents)
        if strongest_score >= 18.0:
            return False
        return not any(
            str(doc.get("docsource") or "").strip().lower() in authoritative_live_sources
            for doc in live_documents
        )

    @staticmethod
    def _google_site_restrict(query: str) -> str | None:
        normalized = query.lower()
        if "rbi" in normalized or "bank" in normalized:
            return "rbi.org.in"
        if "cyber" in normalized or "upi" in normalized or "fraud" in normalized:
            return "cybercrime.gov.in"
        if any(token in normalized for token in {"court", "supreme court", "high court"}):
            return "sci.gov.in"
        return None

    @staticmethod
    def _internal_confidence(*, internal_documents: list[dict[str, Any]], answer_mode: str, query: str) -> float:
        if not internal_documents:
            return 0.0
        top = internal_documents[0]
        score = float(top.get("score") or 0.0)
        document_kind = str(top.get("document_kind") or "").lower()
        text = " ".join(
            [
                str(top.get("title") or ""),
                str(top.get("headline") or ""),
                str(top.get("fragment_headline") or ""),
                str(top.get("fragment_excerpt") or ""),
                str(top.get("doc_excerpt") or ""),
            ]
        ).lower()
        query_lower = query.lower()
        confidence = min(score / 70.0, 1.0)
        if answer_mode == "statute_first" and document_kind == "statute":
            confidence += 0.15
        if answer_mode == "case_first" and str(top.get("source_kind") or "") == "internal" and "judgment" in query.lower():
            confidence += 0.1
        authority_match = re.search(r"\b(article|section|rule)\s+([0-9]+[a-z]?)\b", query_lower)
        if answer_mode == "statute_first" and authority_match:
            exact_reference = f"{authority_match.group(1)} {authority_match.group(2)}"
            if document_kind not in {"statute", "rule", "judgment", "order"} or exact_reference not in text:
                confidence = min(confidence, 0.48)
        return min(confidence, 0.99)

    @staticmethod
    def _live_confidence(*, live_documents: list[dict[str, Any]], answer_mode: str, query: str) -> float:
        if not live_documents:
            return 0.0
        top = live_documents[0]
        score = float(top.get("score") or 0.0)
        source = str(top.get("docsource") or "").strip().lower()
        text = " ".join(
            [
                str(top.get("title") or ""),
                str(top.get("headline") or ""),
                str(top.get("fragment_headline") or ""),
                str(top.get("fragment_excerpt") or ""),
                str(top.get("doc_excerpt") or ""),
            ]
        ).lower()
        query_tokens = {token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(token) > 2}
        overlap = len(query_tokens & {token for token in re.findall(r"[a-zA-Z0-9]+", text) if len(token) > 2})
        confidence = 0.35
        if source in SOURCE_AUTHORITY_SCORES:
            confidence += 0.2
        if answer_mode == "statute_first" and source in {"laws", "constitution"}:
            confidence += 0.2
        if answer_mode == "case_first" and source not in {"laws", "constitution"}:
            confidence += 0.16
        if score >= 30:
            confidence += 0.16
        elif score >= 22:
            confidence += 0.1
        elif score >= 16:
            confidence += 0.05
        if overlap >= 3:
            confidence += 0.1
        elif overlap >= 1:
            confidence += 0.05
        return min(confidence, 0.99)

    @staticmethod
    def _google_confidence(*, documents: list[dict[str, Any]], answer_mode: str, query: str) -> float:
        if not documents:
            return 0.0
        top = documents[0]
        score = float(top.get("score") or 0.0)
        authority_type = str(top.get("authority_type") or "").strip().lower()
        source_text = " ".join(
            [
                str(top.get("source_domain") or ""),
                str(top.get("docsource") or ""),
                str(top.get("url") or ""),
            ]
        ).lower()
        trusted = bool(top.get("trusted_domain_match")) or any(
            domain in source_text
            for domain in {
                "indiacode.nic.in",
                "gov.in",
                "nic.in",
                "rbi.org.in",
                "cybercrime.gov.in",
                "sci.gov.in",
            }
        )
        scope = str(top.get("google_scope") or "").strip().lower()
        confidence = 0.32
        if authority_type in {"government_portal", "regulator", "court_portal"}:
            confidence += 0.24
        if trusted:
            confidence += 0.12
        if trusted and any(domain in source_text for domain in {"indiacode.nic.in", "gov.in", "nic.in"}):
            confidence += 0.12
        if scope == "curated":
            confidence += 0.05
        if answer_mode in {"statute_first", "case_first"}:
            confidence += 0.04
        if score >= 16:
            confidence += 0.12
        elif score >= 12:
            confidence += 0.07
        return min(confidence, 0.95)

    @staticmethod
    def _merge_documents(
        curated_google_documents: list[dict[str, Any]],
        internal_documents: list[dict[str, Any]],
        live_documents: list[dict[str, Any]],
        google_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        def _dedupe(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for doc in docs:
                source_key = str(doc.get("doc_id") or doc.get("url") or doc.get("title") or "")
                if not source_key or source_key in seen:
                    continue
                seen.add(source_key)
                merged.append(doc)
            return merged

        def _score(doc: dict[str, Any]) -> float:
            return float(doc.get("score") or 0.0) + SOURCE_AUTHORITY_SCORES.get(str(doc.get("docsource") or "").lower(), 0.0)

        curated_ranked = sorted(_dedupe(curated_google_documents), key=_score, reverse=True)
        internal_ranked = sorted(_dedupe(internal_documents), key=_score, reverse=True)
        live_ranked = sorted(_dedupe(live_documents), key=_score, reverse=True)
        google_ranked = sorted(_dedupe(google_documents), key=_score, reverse=True)

        merged = curated_ranked[:2]
        merged.extend([doc for doc in internal_ranked[:4] if doc not in merged][: 6 - len(merged)])
        merged.extend(live_ranked[:2])
        if len(merged) < 6:
            merged.extend([doc for doc in google_ranked[:2] if doc not in merged][: 6 - len(merged)])
        if len(merged) < 6:
            remaining = [doc for doc in curated_ranked[2:] + internal_ranked[4:] + live_ranked[2:] + google_ranked[2:] if doc not in merged]
            merged.extend(remaining[: 6 - len(merged)])
        return merged[:6]

    @staticmethod
    def _build_source_summary(documents: list[dict[str, Any]]) -> list[str]:
        summary: list[str] = []
        for doc in documents:
            source = str(doc.get("docsource") or "").strip()
            if source and source not in summary:
                summary.append(source)
        return summary[:5]
