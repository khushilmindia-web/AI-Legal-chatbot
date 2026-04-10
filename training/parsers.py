from __future__ import annotations

import json
import re
from html import unescape

from training.models import ParsedDocument, RawDocument


class LegalDocumentParser:
    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        content_type = (raw_document.content_type or "text/plain").lower()

        if "json" in content_type:
            return self._parse_json(raw_document)
        if "html" in content_type:
            return self._parse_html(raw_document)
        return self._parse_plain_text(raw_document)

    def _parse_json(self, raw_document: RawDocument) -> ParsedDocument:
        metadata = dict(raw_document.metadata)
        title = raw_document.title
        text = raw_document.text

        try:
            payload = json.loads(raw_document.text)
        except json.JSONDecodeError:
            return self._parse_plain_text(raw_document)

        if isinstance(payload, dict):
            title = str(payload.get("title") or payload.get("name") or title or "").strip() or None
            body = payload.get("text") or payload.get("body") or payload.get("content")
            if isinstance(body, str) and body.strip():
                text = body
            extra_metadata = payload.get("metadata")
            if isinstance(extra_metadata, dict):
                metadata.update({str(key): str(value) for key, value in extra_metadata.items()})
            for key in ("jurisdiction", "court", "citation", "date", "source_url"):
                value = payload.get(key)
                if value is not None and str(value).strip():
                    metadata[key] = str(value).strip()

        return ParsedDocument(
            source_id=raw_document.source_id,
            title=title or self._infer_title_from_text(text),
            text=self._normalize_line_endings(text),
            content_type=raw_document.content_type,
            source_name=raw_document.source_name,
            metadata=metadata,
        )

    def _parse_html(self, raw_document: RawDocument) -> ParsedDocument:
        html = raw_document.text or ""
        title = raw_document.title or self._extract_html_title(html)
        text = self._strip_html(html)
        return ParsedDocument(
            source_id=raw_document.source_id,
            title=title or self._infer_title_from_text(text),
            text=self._normalize_line_endings(text),
            content_type=raw_document.content_type,
            source_name=raw_document.source_name,
            metadata=dict(raw_document.metadata),
        )

    def _parse_plain_text(self, raw_document: RawDocument) -> ParsedDocument:
        text = self._normalize_line_endings(raw_document.text or "")
        return ParsedDocument(
            source_id=raw_document.source_id,
            title=(raw_document.title or self._infer_title_from_text(text)),
            text=text,
            content_type=raw_document.content_type,
            source_name=raw_document.source_name,
            metadata=dict(raw_document.metadata),
        )

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        return text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _strip_html(html: str) -> str:
        without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        with_breaks = re.sub(r"(?i)</?(p|div|section|article|br|li|h[1-6]|tr|td|th)>", "\n", without_scripts)
        text = re.sub(r"(?s)<[^>]+>", " ", with_breaks)
        text = unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_html_title(html: str) -> str | None:
        match = re.search(r"(?is)<title>(.*?)</title>", html)
        if not match:
            return None
        title = unescape(match.group(1))
        title = re.sub(r"\s+", " ", title).strip()
        return title or None

    @staticmethod
    def _infer_title_from_text(text: str) -> str:
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip(" -:\t")
            if len(cleaned) >= 4:
                return cleaned[:160]
        return "Untitled legal document"
