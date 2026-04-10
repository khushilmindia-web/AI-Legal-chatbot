from __future__ import annotations

import argparse
from pathlib import Path

from training.batch_ingestion import BatchLegalCorpusIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a legal corpus into embedding-ready JSONL artifacts.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source legal documents")
    parser.add_argument("--output-dir", required=True, help="Directory where processed artifacts will be written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = BatchLegalCorpusIngestor()
    result = ingestor.ingest_directory(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    print(f"Processed files: {result.processed_files}")
    print(f"Generated chunks: {result.chunk_count}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Documents: {result.documents_path}")
    print(f"Chunks: {result.chunks_path}")


if __name__ == "__main__":
    main()
