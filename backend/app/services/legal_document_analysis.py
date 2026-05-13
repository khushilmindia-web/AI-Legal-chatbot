from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.app.services.legal_entity_extraction import LegalEntityExtractionService


@dataclass(frozen=True)
class DocumentTypeRule:
    document_type: str
    markers: tuple[str, ...]
    procedural_stage: str
    next_steps: tuple[str, ...]


class LegalDocumentAnalysisService:
    """Extract lightweight structured facts from user-uploaded legal documents."""

    DOCUMENT_RULES: tuple[DocumentTypeRule, ...] = (
        DocumentTypeRule(
            document_type="fir",
            markers=("fir", "first information report", "police station", "informant", "accused"),
            procedural_stage="police investigation",
            next_steps=(
                "Verify the FIR number, police station, date, sections, and accused details.",
                "Keep the acknowledgement, incident timeline, witness details, and supporting evidence ready.",
            ),
        ),
        DocumentTypeRule(
            document_type="legal_notice",
            markers=("legal notice", "notice", "reply within", "called upon", "demand notice"),
            procedural_stage="notice stage",
            next_steps=(
                "Calendar the reply or compliance deadline immediately.",
                "Prepare a fact-wise reply with the agreement, payment records, and communication trail.",
            ),
        ),
        DocumentTypeRule(
            document_type="agreement",
            markers=("agreement", "contract", "party of the first part", "party of the second part", "clause"),
            procedural_stage="contract review",
            next_steps=(
                "Review payment, termination, liability, dispute-resolution, and notice clauses before acting.",
                "Preserve the signed copy, annexures, amendments, invoices, and correspondence.",
            ),
        ),
        DocumentTypeRule(
            document_type="complaint",
            markers=("complaint", "consumer complaint", "grievance", "prayer", "relief sought"),
            procedural_stage="complaint or grievance stage",
            next_steps=(
                "Check whether the complaint names the correct forum, parties, relief, and supporting documents.",
                "Keep proof of filing, service, invoice, payment proof, and reply records ready.",
            ),
        ),
        DocumentTypeRule(
            document_type="court_order",
            markers=("court", "order", "judgment", "petitioner", "respondent", "directed", "disposed"),
            procedural_stage="court-order compliance",
            next_steps=(
                "Identify the operative directions, compliance deadline, and next hearing or appeal window.",
                "Keep a certified copy, case number, order date, and proof of compliance or service.",
            ),
        ),
    )

    RISK_PATTERNS: tuple[tuple[str, str], ...] = (
        ("deadline", "deadline or limitation-sensitive step"),
        ("within", "time-bound compliance language"),
        ("penalty", "penalty exposure"),
        ("interest", "interest or financial exposure"),
        ("termination", "termination risk"),
        ("eviction", "possession or eviction risk"),
        ("arrest", "criminal-process risk"),
        ("non-bailable", "serious criminal-process risk"),
        ("warrant", "court process or coercive step"),
        ("default", "default allegation"),
        ("breach", "breach allegation"),
        ("dishonour", "cheque dishonour issue"),
        ("fraud", "fraud allegation"),
    )

    AUTHORITY_PATTERN = re.compile(
        r"\b(?:section|sec\.?|s\.|article|art\.?|rule|order)\s+"
        r"\d+[a-zA-Z]?(?:\(\d+[a-zA-Z]?\))*"
        r"(?:\s+of\s+the\s+[A-Z][A-Za-z\s,.\-()]{2,80})?",
        re.IGNORECASE,
    )
    CASE_NUMBER_PATTERN = re.compile(
        r"\b(?:fir|case|complaint|cr\.?\s*case|criminal\s+case|civil\s+suit|appeal|writ)\s*"
        r"(?:no\.?|number)?\s*[:\-]?\s*[A-Z0-9./\-]+",
        re.IGNORECASE,
    )
    DATE_PATTERN = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4})\b",
        re.IGNORECASE,
    )
    RELATIVE_DEADLINE_PATTERN = re.compile(
        r"\bwithin\s+\d{1,3}\s+(?:day|days|week|weeks|month|months)\b",
        re.IGNORECASE,
    )
    MONEY_PATTERN = re.compile(r"(?:rs\.?|inr|rupees)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)

    def __init__(self, entity_extractor: LegalEntityExtractionService | None = None) -> None:
        self.entity_extractor = entity_extractor or LegalEntityExtractionService()

    def analyze_uploads(self, uploaded_texts: list[str] | None) -> list[dict[str, Any]]:
        analyses: list[dict[str, Any]] = []
        for index, text in enumerate(uploaded_texts or [], start=1):
            analysis = self.analyze_text(text, document_index=index)
            if analysis:
                analyses.append(analysis)
        return analyses

    def analyze_text(self, text: str, *, document_index: int = 1) -> dict[str, Any]:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        if not cleaned:
            return {}

        rule = self._classify_document(cleaned)
        entities = self.entity_extractor.extract(cleaned)
        dates = list(dict.fromkeys([*entities["dates"], *self._unique_matches(self.DATE_PATTERN, cleaned, limit=8)]))[:10]
        relative_deadlines = self._unique_matches(self.RELATIVE_DEADLINE_PATTERN, cleaned, limit=6)
        deadlines = list(dict.fromkeys([*entities["deadlines"], *relative_deadlines, *self._deadline_contexts(cleaned)]))[:10]
        section_authorities = [str(item.get("raw") or item.get("normalized") or "").strip() for item in entities["sections"] if isinstance(item, dict)]
        authorities = list(dict.fromkeys([*section_authorities, *entities["statutes"], *self._unique_matches(self.AUTHORITY_PATTERN, cleaned, limit=12)]))[:12]
        obligations = self._extract_obligations(cleaned)
        risk_indicators = self._extract_risks(cleaned, deadlines=deadlines)
        parties = self._extract_parties(cleaned)
        for value in [*entities["people"], *entities["organizations"]]:
            if value not in parties:
                parties.append(value)
        procedural_events = list(dict.fromkeys([*entities["procedural_events"], *self._extract_procedural_events(cleaned)]))[:10]
        money_amounts = [str(item.get("raw") or item.get("normalized") or "").strip() for item in entities["money_amounts"] if isinstance(item, dict)]
        if not money_amounts:
            money_amounts = self._unique_matches(self.MONEY_PATTERN, cleaned, limit=8)
        case_numbers = list(dict.fromkeys([*entities["fir_numbers"], *entities["case_numbers"], *self._unique_matches(self.CASE_NUMBER_PATTERN, cleaned, limit=8)]))[:10]

        return {
            "document_id": f"upload-{document_index}",
            "document_type": rule.document_type,
            "confidence": self._classification_confidence(cleaned, rule),
            "summary": self._summary(cleaned, rule),
            "parties": parties,
            "dates": dates,
            "deadlines": deadlines,
            "authorities": authorities,
            "obligations": obligations,
            "risk_indicators": risk_indicators,
            "procedural_stage": rule.procedural_stage,
            "procedural_events": procedural_events,
            "money_amounts": money_amounts,
            "case_numbers": case_numbers,
            "entities": entities,
            "actionable_next_steps": list(rule.next_steps),
        }

    def _classify_document(self, text: str) -> DocumentTypeRule:
        lowered = text.lower()
        best_rule = self.DOCUMENT_RULES[-1]
        best_score = -1
        for rule in self.DOCUMENT_RULES:
            score = sum(1 for marker in rule.markers if marker in lowered)
            if score > best_score:
                best_rule = rule
                best_score = score
        if best_score <= 0:
            return DocumentTypeRule(
                document_type="legal_document",
                markers=(),
                procedural_stage="document review",
                next_steps=(
                    "Identify the document purpose, parties, dates, deadlines, and legal consequences before responding.",
                    "Keep the full document, annexures, proof of service, and related correspondence together.",
                ),
            )
        return best_rule

    @staticmethod
    def _classification_confidence(text: str, rule: DocumentTypeRule) -> float:
        if not rule.markers:
            return 0.35
        lowered = text.lower()
        hits = sum(1 for marker in rule.markers if marker in lowered)
        return min(0.95, 0.45 + (hits * 0.12))

    @staticmethod
    def _summary(text: str, rule: DocumentTypeRule) -> str:
        snippet = text[:220].strip()
        if len(text) > 220:
            snippet = f"{snippet}..."
        return f"{rule.document_type.replace('_', ' ').title()} detected: {snippet}"

    @staticmethod
    def _unique_matches(pattern: re.Pattern[str], text: str, *, limit: int) -> list[str]:
        items: list[str] = []
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0).strip())
            if value and value.lower() not in {item.lower() for item in items}:
                items.append(value)
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _sentence_windows(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [re.sub(r"\s+", " ", item.strip()) for item in sentences if item.strip()]

    def _deadline_contexts(self, text: str) -> list[str]:
        contexts: list[str] = []
        for sentence in self._sentence_windows(text):
            lowered = sentence.lower()
            if any(token in lowered for token in ("deadline", "due date", "last date", "reply by", "appear on", "hearing on")):
                contexts.append(sentence[:180])
            if len(contexts) >= 6:
                break
        return contexts

    def _extract_obligations(self, text: str) -> list[str]:
        obligations: list[str] = []
        obligation_markers = (
            "shall",
            "must",
            "required to",
            "called upon",
            "directed to",
            "undertakes",
            "agrees to",
            "pay",
            "reply",
            "appear",
            "vacate",
        )
        for sentence in self._sentence_windows(text):
            lowered = sentence.lower()
            if any(marker in lowered for marker in obligation_markers):
                obligations.append(sentence[:220])
            if len(obligations) >= 8:
                break
        return obligations

    def _extract_risks(self, text: str, *, deadlines: list[str]) -> list[str]:
        lowered = text.lower()
        risks = [label for marker, label in self.RISK_PATTERNS if marker in lowered]
        if deadlines:
            risks.append("missed or upcoming deadline risk")
        return list(dict.fromkeys(risks))[:10]

    @staticmethod
    def _extract_parties(text: str) -> list[str]:
        parties: list[str] = []
        patterns = (
            r"\b(?:between|from|by)\s+([A-Z][A-Za-z .,&'-]{2,80})\s+(?:and|to|against)\s+([A-Z][A-Za-z .,&'-]{2,80})",
            r"\b(?:petitioner|complainant|informant|claimant|applicant)\s*[:\-]\s*([A-Z][A-Za-z .,&'-]{2,80})",
            r"\b(?:respondent|accused|opposite party|defendant)\s*[:\-]\s*([A-Z][A-Za-z .,&'-]{2,80})",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                for group in match.groups():
                    value = re.sub(r"\s+", " ", group.strip(" .,:;-"))
                    if value and value.lower() not in {item.lower() for item in parties}:
                        parties.append(value[:100])
                if len(parties) >= 8:
                    return parties
        return parties

    def _extract_procedural_events(self, text: str) -> list[str]:
        events: list[str] = []
        markers = ("filed", "registered", "served", "issued", "replied", "hearing", "listed", "disposed", "directed")
        for sentence in self._sentence_windows(text):
            lowered = sentence.lower()
            if any(marker in lowered for marker in markers):
                events.append(sentence[:220])
            if len(events) >= 8:
                break
        return events
