from __future__ import annotations

import re
from typing import Any


SOURCE_TYPE_PRIORITY = {
    "constitution": 6,
    "statute": 5,
    "state_rule": 5,
    "rule": 4,
    "case_law": 4,
    "government_order": 3,
    "faq": 2,
    "plain_text": 1,
    "general": 1,
}


DOMAIN_REFERENCE_LIBRARY: dict[str, dict[str, list[str]]] = {
    "cyber": {
        "authorities": [
            "Information Technology Act, 2000",
            "Bharatiya Nyaya Sanhita, 2023 provisions on cheating, impersonation, and fraud where applicable",
            "Reserve Bank of India circulars, bank grievance procedures, and platform complaint mechanisms where applicable",
        ],
        "documents": [
            "bank statement or transaction ID",
            "screenshots, app logs, emails, OTP or SMS trail",
            "bank complaint acknowledgement and cyber portal complaint reference",
        ],
    },
    "criminal": {
        "authorities": [
            "Bharatiya Nyaya Sanhita, 2023",
            "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "Indian Evidence Act principles or successor framework, as applicable",
        ],
        "documents": [
            "FIR or complaint copy",
            "notice, summons, arrest memo, bail papers, or case status",
            "witness details, chronology, and supporting records",
        ],
    },
    "family": {
        "authorities": [
            "Hindu Marriage Act, 1955 or other personal law as applicable",
            "Protection of Women from Domestic Violence Act, 2005 where relevant",
            "Guardianship, custody, maintenance, and succession framework as applicable",
        ],
        "documents": [
            "marriage documents, residence proof, and identity records",
            "income records, child-related records, or medical records where relevant",
            "notices, complaints, mediation papers, or pending case papers",
        ],
    },
    "property": {
        "authorities": [
            "Transfer of Property Act, 1882",
            "Registration Act, 1908",
            "State land, revenue, tenancy, and municipal records framework where applicable",
        ],
        "documents": [
            "sale deed, title chain, mutation and tax records",
            "rent agreement, possession documents, boundary papers, or encumbrance records",
            "legal notice, revenue order, municipal record, or pending case papers",
        ],
    },
    "labour": {
        "authorities": [
            "labour codes or pre-code labour statutes as applicable",
            "Industrial disputes framework, gratuity, PF, wage, and standing order rules where applicable",
            "service rules, employment contract, and internal policy material where relevant",
        ],
        "documents": [
            "appointment letter, salary slips, and attendance records",
            "termination, warning, resignation, or HR communication",
            "PF, gratuity, internal complaint, or disciplinary records",
        ],
    },
    "consumer": {
        "authorities": [
            "Consumer Protection Act, 2019",
            "sector regulator or ombudsman framework where applicable",
            "contract, invoice, warranty, and deficiency in service materials",
        ],
        "documents": [
            "invoice, payment proof, warranty, service requests, and complaint emails",
            "product photos, screenshots, and call records",
            "legal notice or regulator complaint acknowledgement where applicable",
        ],
    },
}


def normalize_legal_citation(text: str | None) -> str:
    cleaned = " ".join((text or "").split()).strip()
    cleaned = re.sub(r"\s+\|\s+", " | ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned[:220]


def citation_from_metadata(metadata: dict[str, Any] | None, fallback_source: str = "Source") -> str | None:
    metadata = metadata or {}
    act_name = metadata.get("act_name") or metadata.get("title") or metadata.get("file_name") or fallback_source
    case_name = metadata.get("case_name")
    citation_text = metadata.get("citation")
    court = metadata.get("court")
    year = metadata.get("year")
    article_number = metadata.get("article_number")
    section_number = metadata.get("section_number")
    rule_number = metadata.get("rule_number")
    chapter_number = metadata.get("chapter_number")
    title = metadata.get("title")

    if case_name:
        bits = [str(case_name)]
        if citation_text:
            bits.append(str(citation_text))
        if court:
            bits.append(str(court))
        if year:
            bits.append(str(year))
        return normalize_legal_citation(" | ".join(bits))
    if article_number:
        return normalize_legal_citation(f"{act_name} - Article {article_number}" + (f" ({title})" if title else ""))
    if section_number:
        return normalize_legal_citation(f"{act_name} - Section {section_number}" + (f" ({title})" if title else ""))
    if rule_number:
        return normalize_legal_citation(f"{act_name} - Rule {rule_number}" + (f" ({title})" if title else ""))
    if chapter_number:
        return normalize_legal_citation(f"{act_name} - Chapter {chapter_number}" + (f" ({title})" if title else ""))
    return normalize_legal_citation(str(act_name))


def source_rank_bonus(metadata: dict[str, Any] | None, *, state: str | None = None, matter_type: str | None = None, query: str | None = None) -> float:
    metadata = metadata or {}
    bonus = 0.0
    source_type = str(metadata.get("source_type") or metadata.get("doc_type") or "general").lower()
    bonus -= 0.03 * SOURCE_TYPE_PRIORITY.get(source_type, 1)

    if state:
        applicable_state = str(metadata.get("applicable_state") or "").lower()
        jurisdiction_scope = str(metadata.get("jurisdiction_scope") or "").lower()
        if applicable_state and applicable_state == state.lower():
            bonus -= 0.12
        elif jurisdiction_scope == "central":
            bonus -= 0.02

    if matter_type:
        domain = str(metadata.get("legal_domain") or "").lower()
        if domain == matter_type.lower():
            bonus -= 0.08

    normalized_query = (query or "").lower()
    if metadata.get("article_number") and f"article {str(metadata['article_number']).lower()}" in normalized_query:
        bonus -= 0.18
    if metadata.get("section_number") and f"section {str(metadata['section_number']).lower()}" in normalized_query:
        bonus -= 0.18
    if metadata.get("rule_number") and f"rule {str(metadata['rule_number']).lower()}" in normalized_query:
        bonus -= 0.12

    if metadata.get("citation"):
        bonus -= 0.04
    return bonus


def build_research_brief(
    *,
    matter_type: str | None,
    state: str | None,
    district: str | None,
    urgency_flags: list[str] | None,
    citations: list[str] | None,
    uploaded_documents: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    domain_key = (matter_type or "").lower()
    profile = DOMAIN_REFERENCE_LIBRARY.get(domain_key, {})
    place_text = f"{district}, {state}" if district and state else state or "the relevant jurisdiction"
    urgency_flags = urgency_flags or []
    uploaded_documents = uploaded_documents or []

    forum_strategy = [f"Check territorial jurisdiction and immediate forum options for {place_text}."]
    if domain_key == "cyber":
        forum_strategy.append(f"Consider cyber cell, bank nodal escalation, and police complaint routing for {place_text}.")
    elif domain_key == "family":
        forum_strategy.append(f"Consider Family Court, Magistrate, or Protection Officer route in {place_text}, depending on relief.")
    elif domain_key == "property":
        forum_strategy.append(f"Consider civil, revenue, tenancy, or registrar-linked remedies in {place_text}.")
    elif domain_key == "labour":
        forum_strategy.append(f"Consider labour authority, Labour Court, Industrial Tribunal, EPFO, or gratuity forum in {place_text}.")
    elif domain_key == "consumer":
        forum_strategy.append(f"Consider District Consumer Commission or regulator-linked complaint route in {place_text}.")
    elif domain_key == "criminal":
        forum_strategy.append(f"Consider police station, Magistrate, Sessions Court, or High Court route in {place_text} based on stage.")

    if urgency_flags:
        forum_strategy.append("Urgency flags require immediate preservation of evidence and forum escalation.")

    return {
        "primary_authorities": profile.get("authorities", []),
        "recommended_evidence": profile.get("documents", []),
        "grounded_citations": [normalize_legal_citation(item) for item in (citations or []) if normalize_legal_citation(item)],
        "uploaded_material": [doc.get("original_name", "document") for doc in uploaded_documents[:5]],
        "forum_strategy": forum_strategy,
    }
