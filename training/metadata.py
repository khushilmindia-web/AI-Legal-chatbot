from __future__ import annotations

import re

from training.models import ParsedDocument


class LegalMetadataTagger:
    DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
        "criminal": ("fir", "arrest", "bail", "theft", "snatching", "cheating", "fraud", "criminal", "police"),
        "civil": ("injunction", "damages", "recovery", "specific performance", "contract", "civil suit"),
        "tax": ("gst", "income tax", "vat", "assessment", "input tax credit", "tds"),
        "property": ("property", "sale deed", "lease", "landlord", "tenant", "mutation", "registry"),
        "corporate": ("company", "board", "shareholder", "merger", "director", "corporate", "oppression"),
        "consumer": ("consumer", "defect", "refund", "seller", "service deficiency", "warranty"),
        "document_review": ("agreement", "contract draft", "clause", "notice draft", "review this"),
        "procedure": ("procedure", "appeal", "limitation", "jurisdiction", "filing", "petition", "application"),
    }

    def tag(self, document: ParsedDocument) -> dict[str, str]:
        combined = "\n".join(filter(None, [document.title, document.text]))
        metadata = dict(document.metadata)

        document_kind = self._infer_document_kind(combined)
        if document_kind:
            metadata.setdefault("document_kind", document_kind)

        legal_domain = self._infer_domain(combined)
        if legal_domain:
            metadata.setdefault("legal_domain", legal_domain)

        section_refs = self._extract_section_references(document.text)
        if section_refs:
            metadata.setdefault("section_refs", ", ".join(section_refs))

        jurisdiction = self._infer_jurisdiction(combined)
        if jurisdiction:
            metadata.setdefault("jurisdiction", jurisdiction)

        court = self._infer_court(combined)
        if court:
            metadata.setdefault("court", court)

        year = self._infer_year(combined)
        if year:
            metadata.setdefault("year", year)

        return metadata

    def _infer_document_kind(self, text: str) -> str | None:
        lowered = text.lower()
        if "section " in lowered or "act" in lowered or "rule " in lowered:
            return "statute"
        if "petitioner" in lowered or "respondent" in lowered or "judgment" in lowered or "vs." in lowered or " v. " in lowered:
            return "judgment"
        if "agreement" in lowered or "clause" in lowered:
            return "agreement"
        if "legal notice" in lowered or lowered.startswith("notice"):
            return "notice"
        if "affidavit" in lowered or "petition" in lowered or "plaint" in lowered:
            return "pleading"
        return None

    def _infer_domain(self, text: str) -> str | None:
        lowered = text.lower()
        best_match: tuple[str, int] | None = None
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in lowered)
            if score <= 0:
                continue
            if best_match is None or score > best_match[1]:
                best_match = (domain, score)
        return best_match[0] if best_match else None

    @staticmethod
    def _extract_section_references(text: str) -> list[str]:
        matches = re.findall(r"\b(?:Section|Sec\.|Article|Art\.|Rule|Order)\s+[0-9A-Za-z\-()/.]+\b", text, flags=re.IGNORECASE)
        seen: set[str] = set()
        ordered: list[str] = []
        for match in matches:
            normalized = re.sub(r"\s+", " ", match).strip()
            canonical = normalized.lower()
            if canonical in seen:
                continue
            seen.add(canonical)
            ordered.append(normalized)
        return ordered[:12]

    @staticmethod
    def _infer_jurisdiction(text: str) -> str | None:
        lowered = text.lower()
        if "india" in lowered:
            return "India"
        states = [
            "Delhi",
            "Maharashtra",
            "Gujarat",
            "Karnataka",
            "Tamil Nadu",
            "Kerala",
            "Rajasthan",
            "Uttar Pradesh",
            "West Bengal",
        ]
        for state in states:
            if state.lower() in lowered:
                return state
        return None

    @staticmethod
    def _infer_court(text: str) -> str | None:
        lowered = text.lower()
        if "supreme court" in lowered:
            return "Supreme Court of India"
        high_court_match = re.search(r"([A-Za-z ]+?) high court", text, flags=re.IGNORECASE)
        if high_court_match:
            name = re.sub(r"\s+", " ", high_court_match.group(1)).strip()
            return f"{name.title()} High Court"
        return None

    @staticmethod
    def _infer_year(text: str) -> str | None:
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return match.group(0) if match else None
