from __future__ import annotations

import re

from training.models import ParsedDocument


class LegalTextCleaner:
    def clean(self, document: ParsedDocument) -> ParsedDocument:
        text = document.text or ""
        text = text.replace("\x00", " ")
        text = self._remove_page_artifacts(text)
        text = self._remove_repeated_headers_and_footers(text)
        text = self._normalize_spacing(text)
        return ParsedDocument(
            source_id=document.source_id,
            title=document.title,
            text=text,
            content_type=document.content_type,
            source_name=document.source_name,
            metadata=dict(document.metadata),
        )

    @staticmethod
    def _remove_page_artifacts(text: str) -> str:
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            if re.fullmatch(r"page \d+ of \d+", normalized.lower()):
                continue
            if re.fullmatch(r"\d+", normalized):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _remove_repeated_headers_and_footers(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        frequency: dict[str, int] = {}
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            if len(normalized) < 8:
                continue
            frequency[normalized] = frequency.get(normalized, 0) + 1

        filtered: list[str] = []
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            is_repeated_banner = frequency.get(normalized, 0) >= 3 and len(normalized) <= 120
            if is_repeated_banner:
                continue
            filtered.append(line)
        return "\n".join(filtered)

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n[ ]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
