from __future__ import annotations

import re
from typing import Any


class LegalEntityExtractionService:
    """Reusable lightweight legal entity extraction and normalization."""

    DATE_PATTERN = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4})\b",
        re.IGNORECASE,
    )
    DEADLINE_PATTERN = re.compile(
        r"\b(?:within\s+\d{1,3}\s+(?:day|days|week|weeks|month|months)|"
        r"(?:reply|respond|appear|comply|pay|vacate|file)\s+(?:by|within)\s+[^.;,\n]{2,80})\b",
        re.IGNORECASE,
    )
    MONEY_PATTERN = re.compile(r"\b(?:rs\.?|inr|rupees)\s*[\d,]+(?:\.\d+)?\b", re.IGNORECASE)
    SECTION_PATTERN = re.compile(
        r"\b(?:section|sec\.?|s\.)\s+(\d+[a-zA-Z]?(?:\(\d+[a-zA-Z]?\))*)"
        r"(?:\s+(?:of\s+)?(?:the\s+)?)?"
        r"((?:ipc|bns|bnss|crpc|cpc|ni act|negotiable instruments act|it act|information technology act|"
        r"consumer protection act|companies act|evidence act|indian evidence act|bharatiya sakshya adhiniyam)"
        r"(?:[, ]+\d{4})?)?",
        re.IGNORECASE,
    )
    ARTICLE_PATTERN = re.compile(r"\b(?:article|art\.?)\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
    FIR_PATTERN = re.compile(r"\bfir\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Z0-9./\-]+)\b", re.IGNORECASE)
    CASE_NUMBER_PATTERN = re.compile(
        r"\b(?:(?:case|complaint|cr\.?\s*case|criminal\s+case|civil\s+suit|appeal|writ|petition)\s*"
        r"(?:no\.?|number)?\s*[:\-]?\s*[A-Z0-9./\-]+)\b",
        re.IGNORECASE,
    )
    POLICE_STATION_PATTERN = re.compile(
        r"\b([A-Z][A-Za-z .'-]{2,80}?\s+Police Station|PS\s+[A-Z][A-Za-z .'-]{2,80})\b",
        re.IGNORECASE,
    )
    COURT_PATTERN = re.compile(
        r"\b((?:Supreme Court of India|High Court of [A-Z][A-Za-z ]+|[A-Z][A-Za-z ]+ High Court|"
        r"District Court(?: of)? [A-Z][A-Za-z ]+|Sessions Court(?: of)? [A-Z][A-Za-z ]+|"
        r"Consumer Commission(?:,)? [A-Z][A-Za-z ]+|Family Court(?:,)? [A-Z][A-Za-z ]+))\b"
    )
    ADDRESS_PATTERN = re.compile(
        r"\b(?:at|address(?:ed)? to|resident of|situated at)\s+([A-Za-z0-9][A-Za-z0-9 ,./#-]{10,140})",
        re.IGNORECASE,
    )

    STATUTE_ALIASES: dict[str, str] = {
        "ipc": "Indian Penal Code, 1860",
        "bns": "Bharatiya Nyaya Sanhita, 2023",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "crpc": "Code of Criminal Procedure, 1973",
        "cpc": "Code of Civil Procedure, 1908",
        "ni act": "Negotiable Instruments Act, 1881",
        "negotiable instruments act": "Negotiable Instruments Act, 1881",
        "it act": "Information Technology Act, 2000",
        "information technology act": "Information Technology Act, 2000",
        "consumer protection act": "Consumer Protection Act",
        "companies act": "Companies Act, 2013",
        "evidence act": "Indian Evidence Act, 1872",
        "indian evidence act": "Indian Evidence Act, 1872",
        "bharatiya sakshya adhiniyam": "Bharatiya Sakshya Adhiniyam, 2023",
    }

    PROCEDURAL_MARKERS: tuple[str, ...] = (
        "filed",
        "registered",
        "served",
        "issued",
        "replied",
        "hearing",
        "listed",
        "disposed",
        "directed",
        "summoned",
        "appeared",
        "arrested",
        "investigation",
    )

    def extract(self, text: str | None) -> dict[str, Any]:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if not cleaned:
            return self.empty()

        sections = self._extract_sections(cleaned)
        statutes = self._extract_statutes(cleaned, sections)
        people, organizations = self._extract_people_and_organizations(cleaned)
        return {
            "people": people,
            "organizations": organizations,
            "courts": self._unique_matches(self.COURT_PATTERN, cleaned, limit=8),
            "fir_numbers": self._extract_fir_numbers(cleaned),
            "case_numbers": self._unique_matches(self.CASE_NUMBER_PATTERN, cleaned, limit=8),
            "sections": sections,
            "statutes": statutes,
            "police_stations": self._extract_police_stations(cleaned),
            "money_amounts": self._extract_money_amounts(cleaned),
            "addresses": self._extract_addresses(cleaned),
            "dates": self._unique_matches(self.DATE_PATTERN, cleaned, limit=10),
            "deadlines": self._extract_deadlines(cleaned),
            "procedural_events": self._extract_procedural_events(cleaned),
        }

    def extract_many(self, texts: list[str] | None) -> dict[str, Any]:
        merged = self.empty()
        for text in texts or []:
            merged = self.merge(merged, self.extract(text))
        return merged

    @classmethod
    def empty(cls) -> dict[str, Any]:
        return {
            "people": [],
            "organizations": [],
            "courts": [],
            "fir_numbers": [],
            "case_numbers": [],
            "sections": [],
            "statutes": [],
            "police_stations": [],
            "money_amounts": [],
            "addresses": [],
            "dates": [],
            "deadlines": [],
            "procedural_events": [],
        }

    @classmethod
    def merge(cls, base: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
        merged = cls.empty()
        for source in (base or {}, incoming or {}):
            for key in merged:
                for item in source.get(key) or []:
                    cls._append_unique(merged[key], item)
        return merged

    @classmethod
    def _append_unique(cls, items: list[Any], value: Any) -> None:
        if value in (None, "", [], {}):
            return
        marker = cls._identity(value)
        if marker not in {cls._identity(item) for item in items}:
            items.append(value)

    @staticmethod
    def _identity(value: Any) -> str:
        if isinstance(value, dict):
            normalized = value.get("normalized") or value.get("value") or value.get("raw") or str(value)
        else:
            normalized = value
        return re.sub(r"\s+", " ", str(normalized).strip().lower())

    @staticmethod
    def _unique_matches(pattern: re.Pattern[str], text: str, *, limit: int) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1 if match.lastindex else 0).strip(" .,:;-"))
            key = value.lower()
            if value and key not in seen:
                items.append(value)
                seen.add(key)
            if len(items) >= limit:
                break
        return items

    @classmethod
    def _normalize_statute(cls, value: str | None) -> str | None:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip(" .,;:").lower())
        if not cleaned:
            return None
        for alias, canonical in cls.STATUTE_ALIASES.items():
            if alias in cleaned:
                return canonical
        return cleaned.title()

    def _extract_sections(self, text: str) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        for match in self.SECTION_PATTERN.finditer(text):
            section_number = match.group(1)
            statute = self._normalize_statute(match.group(2))
            raw = re.sub(r"\s+", " ", match.group(0).strip(" .,:;-"))
            payload = {
                "raw": raw,
                "section": section_number.upper(),
                "statute": statute or "",
                "normalized": f"section {section_number.lower()} {statute or ''}".strip().lower(),
            }
            self._append_unique(sections, payload)
            if len(sections) >= 12:
                break
        for match in self.ARTICLE_PATTERN.finditer(text):
            article = match.group(1)
            payload = {
                "raw": re.sub(r"\s+", " ", match.group(0).strip()),
                "section": article.upper(),
                "statute": "Constitution of India",
                "normalized": f"article {article.lower()} constitution of india",
            }
            self._append_unique(sections, payload)
        return sections[:12]

    def _extract_statutes(self, text: str, sections: list[dict[str, str]]) -> list[str]:
        statutes: list[str] = []
        for section in sections:
            self._append_unique(statutes, section.get("statute"))
        lowered = text.lower()
        for alias, canonical in self.STATUTE_ALIASES.items():
            if alias in lowered:
                self._append_unique(statutes, canonical)
        if "constitution" in lowered:
            self._append_unique(statutes, "Constitution of India")
        return statutes[:10]

    def _extract_fir_numbers(self, text: str) -> list[str]:
        firs = []
        for match in self.FIR_PATTERN.finditer(text):
            value = f"FIR {match.group(1).strip()}"
            self._append_unique(firs, value)
        return firs[:8]

    def _extract_police_stations(self, text: str) -> list[str]:
        stations: list[str] = []
        for raw in self._unique_matches(self.POLICE_STATION_PATTERN, text, limit=10):
            value = re.sub(
                r"^(?:was\s+registered\s+at|registered\s+at|filed\s+at|at|in|to|before)\s+",
                "",
                raw.strip(),
                flags=re.IGNORECASE,
            )
            self._append_unique(stations, value)
        return stations[:8]

    def _extract_money_amounts(self, text: str) -> list[dict[str, Any]]:
        amounts: list[dict[str, Any]] = []
        for raw in self._unique_matches(self.MONEY_PATTERN, text, limit=10):
            amount_text = re.sub(r"^(?:rs\.?|inr|rupees)\s*", "", raw.strip(), flags=re.IGNORECASE)
            digits = re.sub(r"[^\d.]", "", amount_text).strip(".")
            payload = {
                "raw": raw,
                "normalized": f"INR {digits}" if digits else raw.upper(),
                "amount": float(digits) if digits and "." in digits else int(digits) if digits else None,
                "currency": "INR",
            }
            self._append_unique(amounts, payload)
        return amounts

    def _extract_deadlines(self, text: str) -> list[str]:
        deadlines = self._unique_matches(self.DEADLINE_PATTERN, text, limit=8)
        for sentence in self._sentence_windows(text):
            lowered = sentence.lower()
            if any(token in lowered for token in ("deadline", "due date", "last date", "limitation", "next hearing")):
                self._append_unique(deadlines, sentence[:180])
        return deadlines[:10]

    def _extract_addresses(self, text: str) -> list[str]:
        addresses: list[str] = []
        for match in self.ADDRESS_PATTERN.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1).strip(" .,:;-"))
            value = re.split(r"\b(?:on|under|within|dated|whereas|and)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,:;-")
            self._append_unique(addresses, value[:160])
        return addresses[:8]

    def _extract_people_and_organizations(self, text: str) -> tuple[list[str], list[str]]:
        people: list[str] = []
        organizations: list[str] = []
        role_patterns = (
            r"\b(?:petitioner|complainant|informant|claimant|applicant|accused|respondent|defendant)\s*[:\-]\s*([A-Z][A-Za-z .,&'-]{2,80})",
            r"\b(?:from|by|between)\s+([A-Z][A-Za-z .,&'-]{2,80})\s+(?:and|to|against)\s+([A-Z][A-Za-z .,&'-]{2,80})",
        )
        organization_terms = {"ltd", "limited", "pvt", "llp", "bank", "company", "traders", "stores", "agency", "police", "department"}
        for pattern in role_patterns:
            for match in re.finditer(pattern, text):
                for group in match.groups():
                    value = re.sub(r"\s+", " ", group.strip(" .,:;-"))
                    if not value:
                        continue
                    if any(term in value.lower() for term in organization_terms):
                        self._append_unique(organizations, value[:100])
                    else:
                        self._append_unique(people, value[:100])
        return people[:8], organizations[:8]

    def _extract_procedural_events(self, text: str) -> list[str]:
        events: list[str] = []
        for sentence in self._sentence_windows(text):
            lowered = sentence.lower()
            if any(marker in lowered for marker in self.PROCEDURAL_MARKERS):
                self._append_unique(events, sentence[:220])
            if len(events) >= 10:
                break
        return events

    @staticmethod
    def _sentence_windows(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [re.sub(r"\s+", " ", item.strip()) for item in sentences if item.strip()]
