from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from backend.app.core.config import DATA_DIR, Settings
from backend.app.services.indiankanoon_service import IndianKanoonService, SOURCE_AUTHORITY_SCORES
from backend.app.utils.request_context import get_logger
from training.embeddings import HashingEmbeddingProvider


logger = get_logger("lawyer_ai.legal_hybrid_retrieval")
DEFAULT_CORPUS_DIR = DATA_DIR / "training_ingestion" / "learning_data_corpus"


@dataclass(slots=True)
class HybridRetrievalResult:
    documents: list[dict[str, Any]]
    internal_count: int
    live_count: int
    internal_confidence: float
    live_used: bool
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
    ) -> None:
        self.settings = settings
        self.corpus_index = corpus_index or LegalCorpusIndex()
        self.indiankanoon_service = indiankanoon_service or IndianKanoonService(settings)

    def retrieve(
        self,
        *,
        query: str,
        query_variants: list[str],
        state: str | None,
        domain: str | None,
        answer_mode: str,
        doctypes_options: list[str | None],
    ) -> HybridRetrievalResult:
        internal_documents = self._search_internal(query=query, query_variants=query_variants, state=state, domain=domain)
        internal_confidence = self._internal_confidence(internal_documents=internal_documents, answer_mode=answer_mode, query=query)
        live_needed = self._should_query_live(
            query=query,
            answer_mode=answer_mode,
            internal_documents=internal_documents,
            internal_confidence=internal_confidence,
        )
        live_documents: list[dict[str, Any]] = []
        if live_needed and self.indiankanoon_service.configured:
            try:
                live_documents = self.indiankanoon_service.retrieve_grounded_documents(
                    query_variants=query_variants,
                    doctypes_options=doctypes_options,
                    max_results=4,
                )
            except (requests.RequestException, ValueError):  # type: ignore[name-defined]
                live_documents = []

        merged_documents = self._merge_documents(internal_documents, live_documents)
        logger.info(
            "hybrid retrieval query=%r internal=%s live=%s live_needed=%s confidence=%.2f",
            query[:120],
            len(internal_documents),
            len(live_documents),
            live_needed,
            internal_confidence,
        )
        return HybridRetrievalResult(
            documents=merged_documents,
            internal_count=len(internal_documents),
            live_count=len(live_documents),
            internal_confidence=internal_confidence,
            live_used=bool(live_documents),
            source_summary=self._build_source_summary(merged_documents),
        )

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
        answer_mode: str,
        internal_documents: list[dict[str, Any]],
        internal_confidence: float,
    ) -> bool:
        normalized = query.lower()
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

    @staticmethod
    def _internal_confidence(*, internal_documents: list[dict[str, Any]], answer_mode: str, query: str) -> float:
        if not internal_documents:
            return 0.0
        top = internal_documents[0]
        score = float(top.get("score") or 0.0)
        confidence = min(score / 70.0, 1.0)
        if answer_mode == "statute_first" and str(top.get("document_kind") or "").lower() == "statute":
            confidence += 0.15
        if answer_mode == "case_first" and str(top.get("source_kind") or "") == "internal" and "judgment" in query.lower():
            confidence += 0.1
        return min(confidence, 0.99)

    @staticmethod
    def _merge_documents(internal_documents: list[dict[str, Any]], live_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

        internal_ranked = sorted(_dedupe(internal_documents), key=_score, reverse=True)
        live_ranked = sorted(_dedupe(live_documents), key=_score, reverse=True)

        merged = internal_ranked[:4]
        merged.extend(live_ranked[:2])
        if len(merged) < 6:
            remaining = [doc for doc in internal_ranked[4:] + live_ranked[2:] if doc not in merged]
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
