from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.retrieval")


@dataclass
class RetrievalChunk:
    source: str
    text: str
    score: float


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
            chunks.append(RetrievalChunk(source=path.name, text=text[:3500], score=float(score)))

        for index, text in enumerate(uploaded_texts, start=1):
            if not text.strip():
                continue
            overlap = len(query_tokens & set(re.findall(r"[a-zA-Z0-9]+", text.lower())))
            chunks.append(RetrievalChunk(source=f"uploaded_file_{index}", text=text[:3500], score=float(50 + overlap)))

        chunks.sort(key=lambda item: item.score, reverse=True)
        logger.info("retrieval mode=local_context query=%r matches=%s", query[:80], len(chunks))
        return chunks[:6]


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_provider = LocalKnowledgeRetrievalProvider(settings.knowledge_dir)

    def get_context(self, query: str, uploaded_texts: list[str], state: str | None, domain: str | None) -> list[RetrievalChunk]:
        return self.local_provider.retrieve(query=query, uploaded_texts=uploaded_texts, state=state, domain=domain)
