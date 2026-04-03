from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from backend.app.core.config import Settings
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.retrieval")


@dataclass
class RetrievalChunk:
    source: str
    text: str
    score: float
    metadata: dict[str, str] | None = None


class BaseRetrievalProvider:
    def retrieve(self, query: str, uploaded_texts: list[str], state: str | None, domain: str | None) -> list[RetrievalChunk]:
        raise NotImplementedError


class LocalKnowledgeRetrievalProvider(BaseRetrievalProvider):
    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir

    def _iter_files(self) -> Iterable[Path]:
        if not self.knowledge_dir.exists():
            return []
        return sorted(
            [
                path
                for path in self.knowledge_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".txt", ".md"}
            ]
        )

    def retrieve(self, query: str, uploaded_texts: list[str], state: str | None, domain: str | None) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []
        query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        domain_token = (domain or "").lower()
        state_token = (state or "").lower()

        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lowered = text.lower()
            overlap = len(query_tokens & set(re.findall(r"[a-zA-Z0-9]+", lowered)))
            if overlap <= 0:
                continue
            score = overlap
            if domain_token and domain_token in lowered:
                score += 3
            if state_token and state_token in lowered:
                score += 2
            chunks.append(
                RetrievalChunk(
                    source=path.name,
                    text=text[:3500],
                    score=float(score),
                    metadata={"type": "local_knowledge", "title": path.name},
                )
            )

        for index, text in enumerate(uploaded_texts, start=1):
            if not text.strip():
                continue
            overlap = len(query_tokens & set(re.findall(r"[a-zA-Z0-9]+", text.lower())))
            chunks.append(
                RetrievalChunk(
                    source=f"uploaded_file_{index}",
                    text=text[:3500],
                    score=float(50 + overlap),
                    metadata={"type": "uploaded_file", "title": f"uploaded_file_{index}"},
                )
            )

        chunks.sort(key=lambda item: item.score, reverse=True)
        logger.info("retrieval mode=local_context query=%r matches=%s", query[:80], len(chunks))
        return chunks[:6]


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_provider = LocalKnowledgeRetrievalProvider(settings.knowledge_dir)
        self.indiankanoon_provider = IndianKanoonService(settings)

    def get_context(self, query: str, uploaded_texts: list[str], state: str | None, domain: str | None) -> list[RetrievalChunk]:
        local_chunks = self.local_provider.retrieve(query=query, uploaded_texts=uploaded_texts, state=state, domain=domain)
        if not self._should_query_indiankanoon(query=query, domain=domain, local_chunks=local_chunks):
            return local_chunks

        try:
            live_chunks = self._retrieve_indiankanoon_chunks(query=query, state=state, domain=domain)
        except requests.RequestException as exc:
            logger.warning("indiankanoon lookup failed query=%r error=%s", query[:80], exc)
            return local_chunks
        except ValueError as exc:
            logger.warning("indiankanoon skipped query=%r error=%s", query[:80], exc)
            return local_chunks

        merged = self._merge_chunks(local_chunks, live_chunks)
        logger.info(
            "retrieval merged query=%r local_matches=%s live_matches=%s",
            query[:80],
            len(local_chunks),
            len(live_chunks),
        )
        return merged

    def _should_query_indiankanoon(
        self,
        query: str,
        domain: str | None,
        local_chunks: list[RetrievalChunk],
    ) -> bool:
        if not self.indiankanoon_provider.configured:
            return False

        normalized = query.lower()
        case_law_signals = [
            "case law",
            "judgment",
            "judgement",
            "precedent",
            "supreme court",
            "high court",
            "citation",
            "article ",
            "section ",
            "ipc",
            "crpc",
            "bnss",
            "bns",
            "consumer protection act",
            "ni act",
            "constitution",
        ]
        if any(signal in normalized for signal in case_law_signals):
            return True
        if domain in {"constitutional", "criminal", "property", "consumer"} and len(local_chunks) < 2:
            if not local_chunks:
                return True
            top_score = max(chunk.score for chunk in local_chunks)
            return top_score < 6
        if not local_chunks:
            return True
        top_score = max(chunk.score for chunk in local_chunks)
        return top_score < 4

    def _retrieve_indiankanoon_chunks(self, query: str, state: str | None, domain: str | None) -> list[RetrievalChunk]:
        doctypes = self._resolve_doctypes(state=state, domain=domain)
        results = self.indiankanoon_provider.retrieve_contextual_results(query=query, doctypes=doctypes, max_results=3)
        chunks: list[RetrievalChunk] = []

        for item in results:
            parts = [
                f"Title: {item.get('title') or 'Untitled'}",
                f"Source: {item.get('docsource') or 'Indian Kanoon'}",
            ]
            if item.get("publishdate"):
                parts.append(f"Date: {item['publishdate']}")
            if item.get("headline"):
                parts.append(f"Search Headline: {item['headline']}")
            if item.get("fragment_headline"):
                parts.append(f"Relevant Fragment: {item['fragment_headline']}")
            citations = item.get("citations") or []
            if citations:
                parts.append("Citations: " + ", ".join(citations[:3]))

            source = f"indiankanoon:{item['doc_id']}"
            chunks.append(
                RetrievalChunk(
                    source=source,
                    text="\n".join(parts)[:3500],
                    score=25.0,
                    metadata={
                        "type": "indiankanoon",
                        "doc_id": str(item["doc_id"]),
                        "title": str(item.get("title") or "Untitled"),
                        "docsource": str(item.get("docsource") or "Indian Kanoon"),
                        "url": f"https://indiankanoon.org/doc/{item['doc_id']}/",
                    },
                )
            )

        return chunks

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

    @staticmethod
    def _merge_chunks(local_chunks: list[RetrievalChunk], live_chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
        merged: list[RetrievalChunk] = []
        seen_sources: set[str] = set()
        for chunk in [*live_chunks, *local_chunks]:
            if chunk.source in seen_sources:
                continue
            seen_sources.add(chunk.source)
            merged.append(chunk)
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:6]
