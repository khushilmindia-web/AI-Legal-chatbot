from __future__ import annotations

import re


class LegalDomainClassifier:
    _RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "document_review",
            (
                "review this document",
                "review this agreement",
                "review this contract",
                "review this notice",
                "document review",
                "contract review",
                "agreement review",
                "notice review",
                "vet this",
                "check this clause",
                "read this document",
                "uploaded document",
                "upload document",
                "document draft",
                "draft agreement",
                "draft notice",
            ),
        ),
        (
            "tax",
            (
                "gst",
                "income tax",
                "tds",
                "tax notice",
                "tax demand",
                "assessment",
                "itr",
                "refund order",
                "input tax credit",
                "cgst",
                "sgst",
                "customs duty",
            ),
        ),
        (
            "corporate",
            (
                "company law",
                "shareholder",
                "shareholders",
                "board resolution",
                "director",
                "oppression and mismanagement",
                "nclt",
                "mca",
                "memorandum of association",
                "articles of association",
                "incorporation",
                "private limited",
                "llp",
                "merger",
                "acquisition",
                "compliance filing",
            ),
        ),
        (
            "property",
            (
                "landlord",
                "tenant",
                "rent",
                "property",
                "eviction",
                "sale deed",
                "mutation",
                "partition",
                "possession",
                "encroachment",
                "lease",
                "registry",
            ),
        ),
        (
            "consumer",
            (
                "consumer",
                "refund",
                "replacement",
                "defective",
                "warranty",
                "seller",
                "service deficiency",
                "e-commerce",
                "amazon",
                "flipkart",
                "swiggy",
                "zomato",
                "product issue",
            ),
        ),
        (
            "procedure",
            (
                "how do i",
                "how to",
                "process",
                "procedure",
                "steps",
                "where to file",
                "where should i file",
                "what should i do",
                "what can i do",
                "filing process",
                "appeal process",
                "complaint process",
                "document submission",
            ),
        ),
        (
            "criminal",
            (
                "fraud",
                "cheating",
                "cheat",
                "threat",
                "intimidation",
                "theft",
                "snatched",
                "snatching",
                "stolen",
                "robbed",
                "assault",
                "fir",
                "police complaint",
                "police station",
                "cyber fraud",
                "upi fraud",
                "otp scam",
                "phishing",
                "extortion",
                "forgery",
                "criminal intimidation",
            ),
        ),
        (
            "civil",
            (
                "legal notice",
                "breach of contract",
                "recovery suit",
                "specific performance",
                "injunction",
                "damages",
                "money recovery",
                "civil suit",
                "contract dispute",
                "employment dispute",
                "salary dues",
                "defamation",
                "family dispute",
            ),
        ),
    )

    def classify(self, message: str) -> str:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        if not normalized:
            return "civil"
        if self._looks_like_statutory_tax_query(normalized):
            return "tax"
        if self._looks_like_corporate_query(normalized):
            return "corporate"
        if self._looks_like_document_review_query(normalized):
            return "document_review"
        if self._looks_like_procedure_query(normalized):
            return "procedure"
        for label, markers in self._RULES:
            if any(marker in normalized for marker in markers):
                return label
        return "civil"

    @staticmethod
    def _looks_like_document_review_query(normalized: str) -> bool:
        doc_terms = {"document", "agreement", "contract", "notice", "clause", "draft", "lease", "deed"}
        review_terms = {"review", "check", "read", "vet", "analyse", "analyze", "summarize"}
        return any(term in normalized for term in doc_terms) and any(term in normalized for term in review_terms)

    @staticmethod
    def _looks_like_procedure_query(normalized: str) -> bool:
        procedure_terms = {"how do i", "how to", "procedure", "process", "steps", "where to file", "filing", "apply"}
        return any(term in normalized for term in procedure_terms)

    @staticmethod
    def _looks_like_statutory_tax_query(normalized: str) -> bool:
        return bool(re.search(r"\b(?:section|rule)\s+\d+[a-z]?\b", normalized)) and any(
            token in normalized for token in {"gst", "income tax", "cgst", "sgst", "tds"}
        )

    @staticmethod
    def _looks_like_corporate_query(normalized: str) -> bool:
        return any(
            token in normalized
            for token in {"company", "shareholder", "director", "board", "nclt", "llp", "private limited", "mca"}
        )
