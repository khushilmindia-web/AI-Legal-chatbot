from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RawDocument:
    text: str
    source_id: str
    title: str | None = None
    content_type: str = "text/plain"
    source_name: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    source_id: str
    title: str
    text: str
    content_type: str
    source_name: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkedDocument:
    source_id: str
    title: str
    cleaned_text: str
    metadata: dict[str, str]
    chunks: list[DocumentChunk]
