from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

from training.embeddings import EmbeddingExportResult, EmbeddingVectorExporter
from training.models import ChunkedDocument, RawDocument
from training.pipeline import LegalDocumentIngestionPipeline


SUPPORTED_SUFFIXES = {".txt", ".json", ".html", ".htm", ".md", ".pdf"}


@dataclass(slots=True)
class BatchIngestionResult:
    input_dir: Path
    output_dir: Path
    processed_files: int
    chunk_count: int
    skipped_files: list[str]
    manifest_path: Path
    documents_path: Path
    chunks_path: Path
    embeddings_path: Path | None = None
    vectors_path: Path | None = None
    embedding_records: int = 0


class FileSystemDocumentLoader:
    def iter_documents(self, input_dir: Path) -> Iterable[RawDocument]:
        for path in sorted(input_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            text = self._read_file(path)
            if not text.strip():
                continue

            source_id = path.relative_to(input_dir).as_posix()
            yield RawDocument(
                source_id=source_id,
                title=path.stem,
                content_type=self._content_type_for_path(path),
                text=text,
                source_name=path.name,
                metadata={
                    "source_path": str(path),
                    "source_filename": path.name,
                    "source_extension": path.suffix.lower(),
                },
            )

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        if PdfReader is None:
            raise RuntimeError("PyPDF2 is not available for PDF extraction")
        with path.open("rb") as handle:
            reader = PdfReader(handle)
            pages: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
        return "\n\n".join(pages)

    @staticmethod
    def _content_type_for_path(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "application/json"
        if suffix in {".html", ".htm"}:
            return "text/html"
        if suffix == ".md":
            return "text/markdown"
        if suffix == ".pdf":
            return "application/pdf"
        return "text/plain"


class EmbeddingCorpusWriter:
    def write(
        self,
        documents: list[ChunkedDocument],
        output_dir: Path,
        input_dir: Path,
        export_embeddings: bool = True,
        embedding_exporter: EmbeddingVectorExporter | None = None,
    ) -> BatchIngestionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        documents_path = output_dir / "documents.jsonl"
        chunks_path = output_dir / "chunks.jsonl"
        manifest_path = output_dir / "manifest.json"
        embeddings_path: Path | None = None
        vectors_path: Path | None = None
        embedding_records = 0

        with documents_path.open("w", encoding="utf-8") as documents_handle:
            for document in documents:
                payload = {
                    "source_id": document.source_id,
                    "title": document.title,
                    "cleaned_text": document.cleaned_text,
                    "metadata": document.metadata,
                    "chunk_count": len(document.chunks),
                }
                documents_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        with chunks_path.open("w", encoding="utf-8") as chunks_handle:
            for document in documents:
                for chunk in document.chunks:
                    payload = {
                        "id": chunk.chunk_id,
                        "source_id": chunk.source_id,
                        "title": chunk.title,
                        "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                chunks_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        if export_embeddings:
            exporter = embedding_exporter or EmbeddingVectorExporter()
            embedding_result: EmbeddingExportResult = exporter.export_chunks(documents=documents, output_dir=output_dir)
            embeddings_path = embedding_result.embeddings_path
            vectors_path = embedding_result.vectors_path
            embedding_records = embedding_result.record_count

        manifest_payload = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "processed_files": len(documents),
            "chunk_count": sum(len(document.chunks) for document in documents),
            "documents_file": str(documents_path),
            "chunks_file": str(chunks_path),
            "source_ids": [document.source_id for document in documents],
            "embeddings_file": str(embeddings_path) if embeddings_path else "",
            "vectors_file": str(vectors_path) if vectors_path else "",
            "embedding_records": embedding_records,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        return BatchIngestionResult(
            input_dir=input_dir,
            output_dir=output_dir,
            processed_files=len(documents),
            chunk_count=manifest_payload["chunk_count"],
            skipped_files=[],
            manifest_path=manifest_path,
            documents_path=documents_path,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            vectors_path=vectors_path,
            embedding_records=embedding_records,
        )


class BatchLegalCorpusIngestor:
    def __init__(
        self,
        pipeline: LegalDocumentIngestionPipeline | None = None,
        loader: FileSystemDocumentLoader | None = None,
        writer: EmbeddingCorpusWriter | None = None,
        embedding_exporter: EmbeddingVectorExporter | None = None,
    ) -> None:
        self.pipeline = pipeline or LegalDocumentIngestionPipeline()
        self.loader = loader or FileSystemDocumentLoader()
        self.writer = writer or EmbeddingCorpusWriter()
        self.embedding_exporter = embedding_exporter

    def ingest_directory(self, input_dir: Path, output_dir: Path, export_embeddings: bool = True) -> BatchIngestionResult:
        documents = [self.pipeline.ingest(raw_document) for raw_document in self.loader.iter_documents(input_dir)]
        return self.writer.write(
            documents=documents,
            output_dir=output_dir,
            input_dir=input_dir,
            export_embeddings=export_embeddings,
            embedding_exporter=self.embedding_exporter,
        )
