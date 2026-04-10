import json
from pathlib import Path

from training.batch_ingestion import BatchLegalCorpusIngestor
from training.embeddings import EmbeddingVectorExporter, HashingEmbeddingProvider
from training.chunking import ChunkingConfig, DocumentChunker
from training.models import ParsedDocument, RawDocument
from training.pipeline import LegalDocumentIngestionPipeline


def test_pipeline_parses_html_cleans_text_and_extracts_metadata():
    pipeline = LegalDocumentIngestionPipeline()
    raw_document = RawDocument(
        source_id="doc-html-1",
        content_type="text/html",
        text="""
        <html>
          <head><title>Delhi Rent Control Act</title></head>
          <body>
            <h1>Delhi Rent Control Act</h1>
            <p>Page 1 of 3</p>
            <p>Section 14 covers eviction limits for the landlord.</p>
            <p>Delhi Rent Control Act</p>
            <p>Tenant protection remains available in Delhi.</p>
            <p>Delhi Rent Control Act</p>
            <p>Page 2 of 3</p>
          </body>
        </html>
        """,
    )

    result = pipeline.ingest(raw_document)

    assert result.title == "Delhi Rent Control Act"
    assert "Page 1 of 3" not in result.cleaned_text
    assert result.metadata["document_kind"] == "statute"
    assert result.metadata["legal_domain"] == "property"
    assert "Section 14" in result.metadata["section_refs"]
    assert result.metadata["jurisdiction"] == "Delhi"
    assert result.chunks
    assert result.chunks[0].metadata["legal_domain"] == "property"


def test_pipeline_parses_json_and_preserves_declared_metadata():
    pipeline = LegalDocumentIngestionPipeline()
    raw_document = RawDocument(
        source_id="doc-json-1",
        content_type="application/json",
        text="""
        {
          "title": "GST Input Tax Credit Note",
          "text": "Under Section 16 of the CGST Act, input tax credit may be denied when invoice conditions fail.",
          "jurisdiction": "India",
          "metadata": {
            "source_url": "https://example.test/gst-note",
            "author": "training-seed"
          }
        }
        """,
    )

    result = pipeline.ingest(raw_document)

    assert result.title == "GST Input Tax Credit Note"
    assert result.metadata["document_kind"] == "statute"
    assert result.metadata["legal_domain"] == "tax"
    assert result.metadata["jurisdiction"] == "India"
    assert result.metadata["source_url"] == "https://example.test/gst-note"
    assert result.metadata["author"] == "training-seed"
    assert "Section 16" in result.metadata["section_refs"]


def test_chunker_creates_multiple_chunks_with_overlap_and_chunk_metadata():
    chunker = DocumentChunker(ChunkingConfig(max_chars=180, overlap_chars=40, min_chunk_chars=50))
    parsed_document = ParsedDocument(
        source_id="doc-text-1",
        title="Consumer Complaint Draft",
        content_type="text/plain",
        text=(
            "Consumer Complaint Draft\n\n"
            "The complainant purchased a defective refrigerator and preserved the invoice, warranty card, and chat record with the seller.\n\n"
            "The seller refused repair and refund despite repeated notices, causing loss and inconvenience to the consumer.\n\n"
            "Relief sought includes refund, compensation, and litigation costs before the consumer commission."
        ),
        metadata={},
    )

    chunks = chunker.chunk(parsed_document, metadata={"legal_domain": "consumer", "document_kind": "pleading"})

    assert len(chunks) >= 2
    assert chunks[0].metadata["chunk_index"] == "1"
    assert chunks[0].metadata["legal_domain"] == "consumer"
    assert "seller" in chunks[1].text.lower() or "consumer" in chunks[1].text.lower()


def test_chunker_splits_on_legal_markers():
    chunker = DocumentChunker(ChunkingConfig(max_chars=80, overlap_chars=20, min_chunk_chars=20))
    parsed_document = ParsedDocument(
        source_id="doc-statute-1",
        title="Bare Act Extract",
        content_type="text/plain",
        text=(
            "Section 1 Short title.\n"
            "This Act may be called the Example Act.\n\n"
            "Section 2 Applicability.\n"
            "It applies to the territory of India.\n\n"
            "Section 3 Penalty.\n"
            "Whoever contravenes the Act shall be liable."
        ),
        metadata={"document_kind": "statute", "legal_domain": "procedure"},
    )

    chunks = chunker.chunk(parsed_document, metadata={"legal_domain": "procedure", "document_kind": "statute"})

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("Section 1")
    assert any("Section 2 Applicability" in chunk.text for chunk in chunks)


def test_batch_ingestor_writes_embedding_ready_jsonl_outputs(tmp_path: Path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    source_dir.mkdir()

    (source_dir / "note.txt").write_text(
        "Consumer complaint note\n\nThe consumer preserved the invoice and warranty but the seller refused refund.",
        encoding="utf-8",
    )
    (source_dir / "gst.json").write_text(
        json.dumps(
            {
                "title": "GST advisory",
                "text": "Section 16 of the CGST Act governs input tax credit eligibility.",
                "jurisdiction": "India",
            }
        ),
        encoding="utf-8",
    )

    ingestor = BatchLegalCorpusIngestor()
    result = ingestor.ingest_directory(input_dir=source_dir, output_dir=output_dir)

    assert result.processed_files == 2
    assert result.chunk_count >= 2
    assert result.documents_path.exists()
    assert result.chunks_path.exists()
    assert result.embeddings_path is not None and result.embeddings_path.exists()
    assert result.vectors_path is not None and result.vectors_path.exists()
    assert result.embedding_records >= 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["processed_files"] == 2
    assert "note.txt" in "\n".join(manifest["source_ids"])
    assert manifest["embedding_records"] == result.embedding_records


def test_embedding_exporter_builds_deterministic_vectors(tmp_path: Path):
    pipeline = LegalDocumentIngestionPipeline()
    document = pipeline.ingest(
        RawDocument(
            source_id="doc-embed-1",
            content_type="text/plain",
            title="Property dispute note",
            text="Landlord eviction dispute under property law with notice and possession issues.",
        )
    )
    exporter = EmbeddingVectorExporter(provider=HashingEmbeddingProvider(dimensions=32))

    result = exporter.export_chunks([document], output_dir=tmp_path / "vectors")

    assert result.record_count >= 1
    assert result.dimensions == 32
    assert result.embeddings_path.exists()
    assert result.vectors_path.exists()
    vector_line = json.loads(result.vectors_path.read_text(encoding="utf-8").splitlines()[0])
    assert len(vector_line["embedding"]) == 32
    assert vector_line["source_id"] == "doc-embed-1"
