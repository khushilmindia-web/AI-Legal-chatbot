from __future__ import annotations

import re
from dataclasses import dataclass

from training.models import DocumentChunk, ParsedDocument


@dataclass(slots=True)
class ChunkingConfig:
    max_chars: int = 900
    overlap_chars: int = 150
    min_chunk_chars: int = 120


class DocumentChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: ParsedDocument, metadata: dict[str, str]) -> list[DocumentChunk]:
        units = self._split_units(document)
        if not units:
            return [
                DocumentChunk(
                    chunk_id=f"{document.source_id}:chunk:1",
                    source_id=document.source_id,
                    title=document.title,
                    text=document.text.strip(),
                    metadata=dict(metadata),
                )
            ]

        chunks: list[DocumentChunk] = []
        current = ""

        for unit in units:
            candidate = unit if not current else f"{current}\n\n{unit}"
            if current and len(candidate) > self.config.max_chars:
                chunks.append(self._build_chunk(document=document, text=current, metadata=metadata, index=len(chunks) + 1))
                current = self._with_overlap(current=current, next_paragraph=unit)
            else:
                current = candidate

        if current.strip():
            chunks.append(self._build_chunk(document=document, text=current, metadata=metadata, index=len(chunks) + 1))

        return self._merge_small_tail(chunks)

    def _split_units(self, document: ParsedDocument) -> list[str]:
        text = (document.text or "").strip()
        if not text:
            return []

        raw_paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not raw_paragraphs:
            return []

        units: list[str] = []
        for paragraph in raw_paragraphs:
            legal_splits = self._split_on_legal_markers(paragraph)
            if legal_splits:
                units.extend(legal_splits)
            else:
                units.append(paragraph)
        return units

    @staticmethod
    def _split_on_legal_markers(paragraph: str) -> list[str]:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []

        marker_pattern = re.compile(
            r"^(section|article|rule|chapter|part)\s+[0-9ivx]+",
            flags=re.IGNORECASE,
        )
        heading_pattern = re.compile(r"^[A-Z0-9][A-Z0-9\s,()&./-]{10,}$")

        parts: list[str] = []
        current: list[str] = []
        for line in lines:
            should_split = bool(marker_pattern.match(line) or heading_pattern.match(line))
            if should_split and current:
                parts.append(" ".join(current).strip())
                current = [line]
                continue
            current.append(line)
        if current:
            parts.append(" ".join(current).strip())
        return [part for part in parts if len(part) >= 40]

    def _with_overlap(self, current: str, next_paragraph: str) -> str:
        overlap = current[-self.config.overlap_chars :].strip()
        if overlap:
            return f"{overlap}\n\n{next_paragraph}"
        return next_paragraph

    def _build_chunk(self, document: ParsedDocument, text: str, metadata: dict[str, str], index: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{document.source_id}:chunk:{index}",
            source_id=document.source_id,
            title=document.title,
            text=text.strip(),
            metadata={**metadata, "chunk_index": str(index)},
        )

    def _merge_small_tail(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if len(chunks) < 2:
            return chunks
        last_chunk = chunks[-1]
        if len(last_chunk.text) >= self.config.min_chunk_chars:
            return chunks
        merged_text = f"{chunks[-2].text}\n\n{last_chunk.text}".strip()
        chunks[-2] = DocumentChunk(
            chunk_id=chunks[-2].chunk_id,
            source_id=chunks[-2].source_id,
            title=chunks[-2].title,
            text=merged_text,
            metadata=dict(chunks[-2].metadata),
        )
        return chunks[:-1]
