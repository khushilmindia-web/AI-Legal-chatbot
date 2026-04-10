from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import Settings
from backend.app.services.legal_hybrid_retrieval import LegalHybridRetrievalService


@dataclass
class FakeCorpusIndex:
    documents: list[dict]

    def search(self, query: str, *, domain: str | None, state: str | None, max_results: int = 6):
        return list(self.documents[:max_results])


class FakeKanoon:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return True

    def retrieve_grounded_documents(self, *, query_variants, doctypes_options, max_results=4):
        self.calls.append(
            {
                "query_variants": query_variants,
                "doctypes_options": doctypes_options,
                "max_results": max_results,
            }
        )
        return list(self.documents[:max_results])


def test_hybrid_retrieval_prefers_internal_corpus_when_strong():
    service = LegalHybridRetrievalService(
        Settings(INDIANKANOON_API_TOKEN="token"),
        corpus_index=FakeCorpusIndex(
            documents=[
                {
                    "doc_id": "internal:1",
                    "title": "Section 138 note from internal corpus",
                    "headline": "Internal statute note",
                    "fragment_headline": "Internal guidance on section 138",
                    "doc_excerpt": "Internal corpus already explains the statute well.",
                    "docsource": "internal:statute",
                    "citations": ["Section 138 note"],
                    "publishdate": "2024",
                    "url": "https://example.test/internal",
                    "score": 62.0,
                    "source_kind": "internal",
                }
            ]
        ),
        indiankanoon_service=FakeKanoon(
            documents=[
                {
                    "doc_id": "live:1",
                    "title": "Sample Supreme Court Decision",
                    "headline": "Live case law",
                    "fragment_headline": "Relevant fragment",
                    "doc_excerpt": "Live case law excerpt.",
                    "docsource": "supremecourt",
                    "citations": ["(2024) 1 SCC 100"],
                    "publishdate": "2024",
                    "url": "https://indiankanoon.org/doc/1/",
                    "score": 25.0,
                }
            ]
        ),
    )

    result = service.retrieve(
        query="section 138 cheque bounce",
        query_variants=["section 138 cheque bounce"],
        state="Gujarat",
        domain="consumer",
        answer_mode="statute_first",
        doctypes_options=["laws", "judgments"],
    )

    assert result.internal_count == 1
    assert result.live_count == 1
    assert result.documents[0]["source_kind"] == "internal"
    assert result.documents[0]["title"].startswith("Section 138 note")
    assert result.source_summary


def test_hybrid_retrieval_calls_live_when_internal_is_weak():
    fake_kanoon = FakeKanoon(
        documents=[
            {
                "doc_id": "live:99",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses procedural obligations.",
                "fragment_headline": "Relevant fragment from live case law.",
                "doc_excerpt": "The live document explains the applicable legal position.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "2024",
                "url": "https://indiankanoon.org/doc/99/",
                "score": 33.0,
            }
        ]
    )
    service = LegalHybridRetrievalService(
        Settings(INDIANKANOON_API_TOKEN="token"),
        corpus_index=FakeCorpusIndex(
            documents=[
                {
                    "doc_id": "internal:weak",
                    "title": "Loose internal note",
                    "headline": "Some tangential note",
                    "fragment_headline": "Low signal text",
                    "doc_excerpt": "Low signal text only.",
                    "docsource": "internal:general",
                    "citations": ["Internal note"],
                    "publishdate": "2018",
                    "url": "https://example.test/weak",
                    "score": 8.0,
                    "source_kind": "internal",
                }
            ]
        ),
        indiankanoon_service=fake_kanoon,
    )

    result = service.retrieve(
        query="latest Supreme Court judgment on cheque bounce",
        query_variants=["latest Supreme Court judgment on cheque bounce"],
        state="Gujarat",
        domain="criminal",
        answer_mode="case_first",
        doctypes_options=["laws", "judgments"],
    )

    assert fake_kanoon.calls
    assert result.live_used is True
    assert any(doc["source_kind"] == "internal" for doc in result.documents)
    assert any(doc["docsource"] == "supremecourt" for doc in result.documents)
