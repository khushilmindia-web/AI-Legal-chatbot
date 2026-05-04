from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DEFAULT_INDEX_URL = "https://devgan.in/all_sections_bns.php"
DEFAULT_OUTPUT_PATH = Path("data/legal_datasets/bns.json")
SOURCE_NAME = "Devgan.in"
USER_AGENT = "AI-Chatbot BNS dataset builder/1.0 (+polite one-time scrape)"


@dataclass(frozen=True, slots=True)
class SectionLink:
    section: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/legal_datasets/bns.json from Devgan.in BNS section pages."
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Fetch only the first 3 sections and do not write JSON.")
    parser.add_argument("--refresh", action="store_true", help="Refetch matching sections even if they already exist in output.")
    parser.add_argument("--start-section", default="", help="Start at this BNS section number when resuming a batch.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of section pages to fetch after filtering.")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--delay", type=float, default=0.4, help="Delay in seconds between section requests.")
    return parser.parse_args()


def fetch(session: requests.Session, url: str, *, timeout: float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def collect_section_links(index_html: str, index_url: str) -> list[SectionLink]:
    soup = BeautifulSoup(index_html, "html.parser")
    links: dict[str, SectionLink] = {}
    for anchor in soup.find_all("a", href=True):
        label = clean_inline_text(anchor.get_text(" ", strip=True))
        match = re.fullmatch(r"Section\s+([0-9]+[A-Za-z]?)", label, flags=re.IGNORECASE)
        if not match:
            continue
        href = str(anchor["href"])
        if "/bns/section/" not in href:
            continue
        section = normalize_section(match.group(1))
        links.setdefault(section, SectionLink(section=section, url=urljoin(index_url, href)))
    return sorted(links.values(), key=lambda item: section_sort_key(item.section))


def parse_section_page(html: str, source_url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    section = extract_section_number(soup, page_text, source_url)
    title = extract_title(soup, section, page_text)
    full_text = extract_full_text(page_text)
    if not full_text:
        raise ValueError(f"could not extract section text from {source_url}")

    return {
        "section": section,
        "title": title or f"Section {section}",
        "text": full_text,
        "source_url": source_url,
        "source_name": SOURCE_NAME,
    }


def extract_section_number(soup: BeautifulSoup, page_text: str, source_url: str) -> str:
    h1 = soup.find("h1")
    candidates = [h1.get_text(" ", strip=True) if h1 else "", page_text, source_url]
    for candidate in candidates:
        match = re.search(r"(?:BNS\s+)?Section\s+([0-9]+[A-Za-z]?)", candidate, flags=re.IGNORECASE)
        if match:
            return normalize_section(match.group(1))
    raise ValueError(f"could not extract section number from {source_url}")


def extract_title(soup: BeautifulSoup, section: str, page_text: str) -> str:
    title_pattern = re.compile(rf"^S\.\s*{re.escape(section)}\s*$", re.IGNORECASE)
    for anchor in soup.find_all("a"):
        if title_pattern.match(clean_inline_text(anchor.get_text(" ", strip=True))):
            next_anchor = anchor.find_next("a")
            if next_anchor is not None:
                title = clean_inline_text(next_anchor.get_text(" ", strip=True))
                if title and not re.fullmatch(r"S\.\s*[0-9]+[A-Za-z]?", title, flags=re.IGNORECASE):
                    return title

    compact = re.sub(r"\s+", " ", page_text)
    match = re.search(rf"\bS\.\s*{re.escape(section)}\s+(.+?)\s+Description\b", compact, flags=re.IGNORECASE)
    if match:
        return clean_inline_text(match.group(1))
    return ""


def extract_full_text(page_text: str) -> str:
    lines = [clean_inline_text(line) for line in page_text.splitlines()]
    lines = [line for line in lines if line]

    try:
        start = next(index for index, line in enumerate(lines) if line.lower() == "description") + 1
    except StopIteration:
        return ""

    stop_markers = ("By Raman Devgan", "Updated:", "Top", "Prev", "Index", "Next")
    selected: list[str] = []
    for line in lines[start:]:
        if any(line.startswith(marker) for marker in stop_markers):
            break
        selected.append(line)

    while selected and is_ipc_cross_reference(selected):
        selected.pop(0)
    if len(selected) >= 3 and selected[0].lower() == "ipc" and selected[1].lower() == "section":
        selected = selected[3:]

    return "\n".join(selected).strip()


def build_dataset(
    *,
    index_url: str,
    timeout: float,
    delay: float,
    dry_run: bool,
    refresh: bool = False,
    output_path: Path | None = None,
    start_section: str = "",
    limit: int = 0,
) -> tuple[list[dict[str, str]], list[str]]:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT})

    index_html = fetch(session, index_url, timeout=timeout)
    links = collect_section_links(index_html, index_url)
    if dry_run:
        links = links[:3]
    if start_section:
        start_key = section_sort_key(normalize_section(start_section))
        links = [link for link in links if section_sort_key(link.section) >= start_key]
    if limit > 0:
        links = links[:limit]

    rows_by_section = load_existing_rows(output_path) if output_path is not None and not dry_run else {}
    failures: list[str] = []
    for index, link in enumerate(links, start=1):
        if link.section in rows_by_section and not refresh:
            print(f"Skipping existing BNS section {link.section} ({index}/{len(links)})", flush=True)
            continue
        if index > 1:
            time.sleep(max(delay, 0.0))
        try:
            html = fetch(session, link.url, timeout=timeout)
            row = parse_section_page(html, link.url)
            rows_by_section[row["section"]] = row
            if output_path is not None and not dry_run:
                write_dataset(sort_rows(rows_by_section.values()), output_path)
            print(f"Fetched BNS section {link.section} ({index}/{len(links)})", flush=True)
        except Exception as exc:  # noqa: BLE001 - continue a one-time scrape after individual failures.
            failures.append(f"Section {link.section} {link.url}: {exc}")
            print(f"Skipped BNS section {link.section}: {exc}", file=sys.stderr, flush=True)

    return sort_rows(rows_by_section.values()), failures


def write_dataset(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_existing_rows(output_path: Path | None) -> dict[str, dict[str, str]]:
    if output_path is None or not output_path.exists():
        return {}
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}
    rows: dict[str, dict[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        section = normalize_section(str(item.get("section", "")))
        if section:
            rows[section] = dict(item)
    return rows


def sort_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: section_sort_key(normalize_section(row.get("section", ""))))


def is_ipc_cross_reference(lines: list[str]) -> bool:
    return bool(lines) and bool(re.fullmatch(r"IPC\s+Section\s+[0-9A-Za-z,\s]+", lines[0], flags=re.IGNORECASE))


def clean_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def normalize_section(value: str) -> str:
    match = re.search(r"([0-9]+)([A-Za-z]?)", str(value or ""))
    if not match:
        return ""
    return f"{int(match.group(1))}{match.group(2).upper()}"


def section_sort_key(section: str) -> tuple[int, str]:
    match = re.fullmatch(r"([0-9]+)([A-Z]?)", section)
    if not match:
        return (10**9, section)
    return (int(match.group(1)), match.group(2))


def main() -> int:
    args = parse_args()
    rows, failures = build_dataset(
        index_url=args.index_url,
        timeout=args.timeout,
        delay=args.delay,
        dry_run=args.dry_run,
        refresh=args.refresh,
        output_path=args.output,
        start_section=args.start_section,
        limit=args.limit,
    )

    if args.dry_run:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"Dry run fetched {len(rows)} section(s); no file written.")
    else:
        write_dataset(rows, args.output)
        print(f"Wrote {len(rows)} section(s) to {args.output}")

    if failures:
        print(f"Completed with {len(failures)} failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
