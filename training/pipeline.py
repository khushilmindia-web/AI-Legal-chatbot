from __future__ import annotations

from training.chunking import DocumentChunker
from training.cleaners import LegalTextCleaner
from training.metadata import LegalMetadataTagger
from training.models import ChunkedDocument, RawDocument
from training.parsers import LegalDocumentParser


class LegalDocumentIngestionPipeline:
    """Temporary training-time ingestion pipeline kept outside the active runtime stack."""

    def __init__(
        self,
        parser: LegalDocumentParser | None = None,
        cleaner: LegalTextCleaner | None = None,
        metadata_tagger: LegalMetadataTagger | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.parser = parser or LegalDocumentParser()
        self.cleaner = cleaner or LegalTextCleaner()
        self.metadata_tagger = metadata_tagger or LegalMetadataTagger()
        self.chunker = chunker or DocumentChunker()

    def ingest(self, raw_document: RawDocument) -> ChunkedDocument:
        parsed = self.parser.parse(raw_document)
        cleaned = self.cleaner.clean(parsed)
        metadata = self.metadata_tagger.tag(cleaned)
        chunks = self.chunker.chunk(cleaned, metadata=metadata)
        return ChunkedDocument(
            source_id=cleaned.source_id,
            title=cleaned.title,
            cleaned_text=cleaned.text,
            metadata=metadata,
            chunks=chunks,
        )

    def ingest_many(self, raw_documents: list[RawDocument]) -> list[ChunkedDocument]:
        return [self.ingest(raw_document) for raw_document in raw_documents]
