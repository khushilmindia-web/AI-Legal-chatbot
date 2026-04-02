from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.paths import CASELAW_DIR, DOCS_DIR, MANIFESTS_DIR, RAW_SOURCES_DIR, STATELAWS_DIR


TARGET_DIRS = {
    "statute": DOCS_DIR,
    "state_rule": STATELAWS_DIR,
    "case_law": CASELAW_DIR,
}


@dataclass
class ImportResult:
    title: str
    target_type: str
    output_path: Path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120] or "document"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_header(entry: dict[str, Any]) -> str:
    lines = [
        f"Title: {entry.get('title', 'Untitled')}",
        f"Source Type: {entry.get('target_type', 'unknown')}",
    ]
    if entry.get("source_url"):
        lines.append(f"Source URL: {entry['source_url']}")
    if entry.get("jurisdiction"):
        lines.append(f"Jurisdiction: {entry['jurisdiction']}")
    if entry.get("state"):
        lines.append(f"Applicable State: {entry['state']}")
    if entry.get("citation"):
        lines.append(f"Citation: {entry['citation']}")
    if entry.get("court"):
        lines.append(f"Court: {entry['court']}")
    if entry.get("year"):
        lines.append(f"Year: {entry['year']}")
    if entry.get("tags"):
        lines.append("Tags: " + ", ".join(str(item) for item in entry["tags"]))
    return "\n".join(lines)


def normalize_case_law_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"(?im)^facts\s*[:\-]?", "Facts:", text)
    text = re.sub(r"(?im)^issues?\s*[:\-]?", "Issues:", text)
    text = re.sub(r"(?im)^held\s*[:\-]?", "Held:", text)
    text = re.sub(r"(?im)^ratio\s*[:\-]?", "Ratio:", text)
    return text


def normalize_statute_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"(?im)^\s*art\.?\s+([0-9A-Z]+)", r"Article \1", text)
    text = re.sub(r"(?im)^\s*sec\.?\s+([0-9A-Z]+)", r"Section \1", text)
    text = re.sub(r"(?im)^\s*r\.?\s+([0-9A-Z]+)", r"Rule \1", text)
    return text


def normalize_entry_text(entry: dict[str, Any], raw_text: str) -> str:
    target_type = entry.get("target_type")
    if target_type == "case_law":
        body = normalize_case_law_text(raw_text)
    else:
        body = normalize_statute_text(raw_text)
    return f"{build_header(entry)}\n\n{body}\n"


def output_path_for_entry(entry: dict[str, Any]) -> Path:
    target_type = entry["target_type"]
    target_dir = TARGET_DIRS[target_type]
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = slugify(entry.get("output_name") or entry.get("title") or Path(entry["source_file"]).stem)
    return target_dir / f"{base_name}.txt"


def import_entry(entry: dict[str, Any]) -> ImportResult:
    target_type = entry.get("target_type")
    if target_type not in TARGET_DIRS:
        raise ValueError(f"Unsupported target_type: {target_type}")

    source_file = entry.get("source_file")
    if not source_file:
        raise ValueError("Manifest entry missing source_file")

    raw_path = RAW_SOURCES_DIR / source_file
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw source not found: {raw_path}")

    raw_text = read_text_file(raw_path)
    normalized_text = normalize_entry_text(entry, raw_text)
    output_path = output_path_for_entry(entry)
    output_path.write_text(normalized_text, encoding="utf-8")
    return ImportResult(
        title=entry.get("title", output_path.stem),
        target_type=target_type,
        output_path=output_path,
    )


def import_manifest(manifest_path: Path) -> list[ImportResult]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an 'entries' list")

    results: list[ImportResult] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        results.append(import_entry(entry))
    return results


def ensure_structure() -> None:
    RAW_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STATELAWS_DIR.mkdir(parents=True, exist_ok=True)
    CASELAW_DIR.mkdir(parents=True, exist_ok=True)


def main(manifest_name: str = "corpus_manifest.json") -> None:
    ensure_structure()
    manifest_path = MANIFESTS_DIR / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    results = import_manifest(manifest_path)
    print(f"Imported {len(results)} corpus files from {manifest_path.name}")
    for item in results:
        print(f"- [{item.target_type}] {item.title} -> {item.output_path}")


if __name__ == "__main__":
    main()
