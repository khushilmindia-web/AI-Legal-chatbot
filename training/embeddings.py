from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from training.models import ChunkedDocument, DocumentChunk


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbeddingProvider:
    """Deterministic local fallback for embedding preparation and tests."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return values
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimensions):
                bucket = digest[index % len(digest)]
                values[index] += (bucket / 255.0) - 0.5
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]


@dataclass(slots=True)
class EmbeddingExportResult:
    output_dir: Path
    embeddings_path: Path
    vectors_path: Path
    record_count: int
    dimensions: int


class EmbeddingVectorExporter:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider or HashingEmbeddingProvider()

    def export_chunks(self, documents: list[ChunkedDocument], output_dir: Path) -> EmbeddingExportResult:
        vectors_path = output_dir / "vectors.jsonl"
        embeddings_path = output_dir / "embeddings.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = [chunk for document in documents for chunk in document.chunks]
        texts = [self._embedding_text(chunk) for chunk in chunks]
        vectors = self.provider.embed_texts(texts)
        dimensions = len(vectors[0]) if vectors else getattr(self.provider, "dimensions", 0)

        with vectors_path.open("w", encoding="utf-8") as vectors_handle, embeddings_path.open("w", encoding="utf-8") as embeddings_handle:
            for chunk, text, vector in zip(chunks, texts, vectors):
                payload = {
                    "id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "embedding_text": text,
                    "embedding": vector,
                    "metadata": chunk.metadata,
                }
                vectors_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                embeddings_handle.write(
                    json.dumps(
                        {
                            "id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "text": text,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        return EmbeddingExportResult(
            output_dir=output_dir,
            embeddings_path=embeddings_path,
            vectors_path=vectors_path,
            record_count=len(chunks),
            dimensions=dimensions,
        )

    @staticmethod
    def _embedding_text(chunk: DocumentChunk) -> str:
        metadata_bits = []
        for key in ("title", "legal_domain", "document_kind", "jurisdiction", "court", "section_refs"):
            value = chunk.metadata.get(key)
            if value:
                metadata_bits.append(f"{key}: {value}")
        metadata_prefix = " | ".join(metadata_bits)
        if metadata_prefix:
            return f"{metadata_prefix}\n\n{chunk.text}".strip()
        return chunk.text.strip()
