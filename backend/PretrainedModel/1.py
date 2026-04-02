from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import move
from typing import Any

import fitz
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema
from whoosh import writing
from backend.paths import DATA_DIR


LOGGER = logging.getLogger("pdf_pipeline")
ACT_PATTERN = re.compile(
    r"(?P<name>[A-Z][A-Za-z0-9,&()'./\-\s]{3,120}?\bAct)"
    r"(?:\s*(?:,|of)\s*(?P<year>(18|19|20)\d{2}))?",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(18|19|20)\d{2}\b")
GENERIC_TITLES = {
    "",
    "untitled",
    "document",
    "microsoft word",
    "scan",
    "scanned document",
}


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path
    corrupted_dir: Path
    extracted_text_dir: Path
    sqlite_path: Path
    whoosh_dir: Path
    log_path: Path


@dataclass(slots=True)
class ProcessedDocument:
    filename: str
    filepath: str
    text: str
    metadata: dict[str, Any]
    pages: int
    md5: str


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_directories(config: PipelineConfig) -> None:
    config.input_dir.mkdir(parents=True, exist_ok=True)
    config.corrupted_dir.mkdir(parents=True, exist_ok=True)
    config.extracted_text_dir.mkdir(parents=True, exist_ok=True)
    config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    config.whoosh_dir.mkdir(parents=True, exist_ok=True)


def init_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                pages INTEGER NOT NULL,
                md5 TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pdf_documents_filename ON pdf_documents(filename)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pdf_documents_pages ON pdf_documents(pages)"
        )
        connection.commit()


def iter_pdf_files(input_dir: Path, excluded_dirs: set[Path]) -> list[Path]:
    pdf_files: list[Path] = []
    for path in sorted(input_dir.rglob("*.pdf")):
        if any(excluded in path.parents for excluded in excluded_dirs):
            continue
        pdf_files.append(path)
    return pdf_files


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    value = clean_whitespace(value).lower()
    value = re.sub(r"[^\w\s-]", " ", value)
    value = re.sub(r"[-\s]+", "_", value)
    value = value.strip("._")
    return value[:180] or "document"


def safe_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    return clean_whitespace(str(value))


def parse_pdf(file_path: Path) -> tuple[str, dict[str, Any], int]:
    try:
        reader = PdfReader(str(file_path))
        raw_metadata = reader.metadata or {}
        pages = len(reader.pages)
        extracted_pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)
        text = "\n\n".join(page.strip() for page in extracted_pages if page.strip())
        metadata = {
            "source_name": file_path.name,
            "pdf_metadata": {
                key.lstrip("/"): safe_metadata_value(value)
                for key, value in dict(raw_metadata).items()
            },
        }
        return text, metadata, pages
    except (PdfReadError, Exception) as exc:
        raise RuntimeError(f"Unable to parse PDF: {exc}") from exc


def validate_pdf(file_path: Path) -> None:
    try:
        with fitz.open(file_path) as document:
            if document.page_count <= 0:
                raise RuntimeError("PDF has no pages")
    except Exception as exc:
        raise RuntimeError(f"Corrupted or unreadable PDF: {exc}") from exc


def infer_title_from_metadata(metadata: dict[str, Any]) -> str:
    pdf_meta = metadata.get("pdf_metadata", {})
    for key in ("Title", "title", "Subject", "subject"):
        candidate = clean_whitespace(str(pdf_meta.get(key, "")))
        lowered = candidate.lower()
        if candidate and lowered not in GENERIC_TITLES:
            candidate = re.sub(r"\.(pdf|cdr|docx?)$", "", candidate, flags=re.IGNORECASE)
            return clean_whitespace(candidate)
    return ""


def infer_title_and_year(text: str, metadata: dict[str, Any], original_stem: str) -> tuple[str, str]:
    metadata_title = infer_title_from_metadata(metadata)
    search_space = clean_whitespace(" ".join(filter(None, [metadata_title, text[:4000]])))

    match = ACT_PATTERN.search(search_space)
    if match:
        act_name = clean_whitespace(match.group("name"))
        year = match.group("year") or ""
        if not year:
            nearby_text = search_space[match.start() : match.end() + 80]
            year_match = YEAR_PATTERN.search(nearby_text)
            year = year_match.group(0) if year_match else ""
        return act_name, year

    if metadata_title:
        year_match = YEAR_PATTERN.search(metadata_title)
        return metadata_title, year_match.group(0) if year_match else ""

    cleaned_original = clean_whitespace(original_stem.replace("_", " ").replace("-", " "))
    year_match = YEAR_PATTERN.search(cleaned_original)
    return cleaned_original, year_match.group(0) if year_match else ""


def build_target_filename(file_path: Path, text: str, metadata: dict[str, Any]) -> str:
    act_name, year = infer_title_and_year(text, metadata, file_path.stem)
    cleaned_name = slugify(act_name or file_path.stem)
    if year and not cleaned_name.endswith(f"_{year}"):
        cleaned_name = f"{cleaned_name}_{year}"
    return f"{cleaned_name}.pdf"


def uniquify_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    counter = 2
    while True:
        candidate = target_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_corrupted(file_path: Path, corrupted_dir: Path) -> Path:
    target_path = uniquify_path(corrupted_dir / file_path.name)
    move(str(file_path), str(target_path))
    return target_path


def rename_pdf(file_path: Path, target_name: str) -> Path:
    target_path = file_path.with_name(target_name)
    if target_path.resolve() == file_path.resolve():
        return file_path
    target_path = uniquify_path(target_path)
    file_path.rename(target_path)
    return target_path


def save_extracted_text(text_dir: Path, pdf_path: Path, text: str) -> Path:
    text_path = text_dir / f"{pdf_path.stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    return text_path


def upsert_document(db_path: Path, document: ProcessedDocument) -> None:
    payload = (
        document.filename,
        document.filepath,
        document.text,
        json.dumps(document.metadata, ensure_ascii=False),
        document.pages,
        document.md5,
        datetime.now(timezone.utc).isoformat(),
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO pdf_documents (filename, filepath, text, metadata, pages, md5, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(md5) DO UPDATE SET
                filename = excluded.filename,
                filepath = excluded.filepath,
                text = excluded.text,
                metadata = excluded.metadata,
                pages = excluded.pages
            """,
            payload,
        )
        connection.commit()


def remove_stale_rows(db_path: Path, valid_md5s: set[str]) -> None:
    with sqlite3.connect(db_path) as connection:
        if not valid_md5s:
            connection.execute("DELETE FROM pdf_documents")
        else:
            placeholders = ",".join("?" for _ in valid_md5s)
            connection.execute(
                f"DELETE FROM pdf_documents WHERE md5 NOT IN ({placeholders})",
                tuple(valid_md5s),
            )
        connection.commit()


def build_whoosh_index(db_path: Path, index_dir: Path) -> int:
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        filename=TEXT(stored=True),
        filepath=STORED,
        pages=NUMERIC(stored=True),
        metadata=STORED,
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

    if index.exists_in(index_dir):
        ix = index.open_dir(index_dir)
    else:
        ix = index.create_in(index_dir, schema)

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, filename, filepath, text, metadata, pages FROM pdf_documents ORDER BY id"
        ).fetchall()

    writer = ix.writer()
    writer.mergetype = writing.CLEAR  # type: ignore[name-defined]
    for row in rows:
        writer.add_document(
            doc_id=str(row[0]),
            filename=row[1],
            filepath=row[2],
            pages=row[5],
            metadata=row[4],
            content=row[3],
        )
    writer.commit()
    return len(rows)


def verify_index(index_dir: Path) -> int:
    ix = index.open_dir(index_dir)
    with ix.searcher() as searcher:
        return searcher.doc_count_all()


def process_pdfs(config: PipelineConfig) -> dict[str, int]:
    seen_md5s: dict[str, Path] = {}
    valid_md5s: set[str] = set()
    stats = {
        "discovered": 0,
        "processed": 0,
        "renamed": 0,
        "duplicates_removed": 0,
        "corrupted_moved": 0,
        "text_files_written": 0,
    }

    excluded_dirs = {config.corrupted_dir}
    pdf_files = iter_pdf_files(config.input_dir, excluded_dirs)
    stats["discovered"] = len(pdf_files)
    LOGGER.info("Discovered %s PDF files in %s", len(pdf_files), config.input_dir)

    for pdf_path in pdf_files:
        LOGGER.info("Processing %s", pdf_path)
        try:
            md5_hash = compute_md5(pdf_path)
        except Exception as exc:
            LOGGER.exception("Failed to hash %s: %s", pdf_path, exc)
            continue

        existing = seen_md5s.get(md5_hash)
        if existing is not None:
            pdf_path.unlink()
            stats["duplicates_removed"] += 1
            LOGGER.info("Removed duplicate %s (same as %s)", pdf_path, existing)
            continue

        try:
            validate_pdf(pdf_path)
            text, metadata, pages = parse_pdf(pdf_path)
        except Exception as exc:
            corrupted_path = move_to_corrupted(pdf_path, config.corrupted_dir)
            stats["corrupted_moved"] += 1
            LOGGER.warning("Moved corrupted PDF to %s because %s", corrupted_path, exc)
            continue

        renamed_path = rename_pdf(pdf_path, build_target_filename(pdf_path, text, metadata))
        if renamed_path != pdf_path:
            stats["renamed"] += 1
            LOGGER.info("Renamed %s -> %s", pdf_path.name, renamed_path.name)

        text_path = save_extracted_text(config.extracted_text_dir, renamed_path, text)
        stats["text_files_written"] += 1

        metadata.update(
            {
                "extracted_text_path": str(text_path.resolve()),
                "original_filename": pdf_path.name,
                "final_filename": renamed_path.name,
            }
        )

        document = ProcessedDocument(
            filename=renamed_path.name,
            filepath=str(renamed_path.resolve()),
            text=text,
            metadata=metadata,
            pages=pages,
            md5=md5_hash,
        )
        upsert_document(config.sqlite_path, document)

        seen_md5s[md5_hash] = renamed_path
        valid_md5s.add(md5_hash)
        stats["processed"] += 1

    remove_stale_rows(config.sqlite_path, valid_md5s)
    stats["indexed"] = build_whoosh_index(config.sqlite_path, config.whoosh_dir)
    stats["index_verified"] = verify_index(config.whoosh_dir)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process PDFs, extract text, persist metadata, and build Whoosh index.")
    parser.add_argument("--input-dir", default=str(DATA_DIR / "pdfs"), help="Directory containing source PDF files.")
    parser.add_argument("--corrupted-dir", default=str(DATA_DIR / "pdfs" / "corrupted"), help="Directory where corrupted PDFs are moved.")
    parser.add_argument("--text-dir", default=str(DATA_DIR / "extracted_text"), help="Directory where extracted text files are saved.")
    parser.add_argument("--sqlite-path", default=str(DATA_DIR / "pdf_documents.sqlite"), help="SQLite database path.")
    parser.add_argument("--whoosh-dir", default=str(DATA_DIR / "whoosh_index"), help="Whoosh index directory.")
    parser.add_argument("--log-path", default=str(DATA_DIR / "logs" / "pdf_pipeline.log"), help="Pipeline log file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_dir=Path(args.input_dir).resolve(),
        corrupted_dir=Path(args.corrupted_dir).resolve(),
        extracted_text_dir=Path(args.text_dir).resolve(),
        sqlite_path=Path(args.sqlite_path).resolve(),
        whoosh_dir=Path(args.whoosh_dir).resolve(),
        log_path=Path(args.log_path).resolve(),
    )

    ensure_directories(config)
    configure_logging(config.log_path)
    init_database(config.sqlite_path)

    stats = process_pdfs(config)

    LOGGER.info("Pipeline complete.")
    LOGGER.info("Discovered: %s", stats["discovered"])
    LOGGER.info("Processed: %s", stats["processed"])
    LOGGER.info("Renamed: %s", stats["renamed"])
    LOGGER.info("Duplicates removed: %s", stats["duplicates_removed"])
    LOGGER.info("Corrupted moved: %s", stats["corrupted_moved"])
    LOGGER.info("Text files written: %s", stats["text_files_written"])
    LOGGER.info("Whoosh documents indexed: %s", stats["indexed"])
    LOGGER.info("Whoosh documents verified: %s", stats["index_verified"])


if __name__ == "__main__":
    main()
