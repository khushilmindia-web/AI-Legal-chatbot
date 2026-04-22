from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import Settings
from backend.app.services.google_custom_search_service import GoogleSearchResult
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


class FakeGoogleSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return True

    def search(self, *, query: str, max_results: int | None = None, site_restrict: str | None = None, trusted_only: bool = True):
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "site_restrict": site_restrict,
                "trusted_only": trusted_only,
            }
        )
        return GoogleSearchResult(
            documents=list(self.documents[: max_results or len(self.documents)]),
            from_cache=False,
            trusted_result_count=min(len(self.documents), max_results or len(self.documents)),
        )


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
    assert result.google_count == 0


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


def test_hybrid_retrieval_calls_google_only_for_weak_official_source_queries():
    fake_google = FakeGoogleSearch(
        documents=[
            {
                "doc_id": "google:1",
                "title": "National Cyber Crime Portal",
                "headline": "Official complaint portal.",
                "fragment_headline": "Portal",
                "doc_excerpt": "Use the official cybercrime complaint portal.",
                "docsource": "google:cybercrime.gov.in",
                "citations": ["National Cyber Crime Portal"],
                "publishdate": "",
                "url": "https://cybercrime.gov.in/",
                "score": 16.0,
                "source_kind": "google_custom_search",
                "authority_type": "government_portal",
            }
        ]
    )
    service = LegalHybridRetrievalService(
        Settings(
            INDIANKANOON_API_TOKEN="token",
            GOOGLE_SEARCH_ENABLED="true",
            GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
            GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        ),
        corpus_index=FakeCorpusIndex(
            documents=[
                {
                    "doc_id": "internal:weak",
                    "title": "Weak procedural note",
                    "headline": "Short note",
                    "fragment_headline": "Short note",
                    "doc_excerpt": "A weak local note.",
                    "docsource": "internal:general",
                    "citations": ["Weak note"],
                    "publishdate": "2019",
                    "url": "https://example.test/weak",
                    "score": 6.0,
                    "source_kind": "internal",
                }
            ]
        ),
        indiankanoon_service=FakeKanoon(documents=[]),
        google_search_service=fake_google,
    )

    result = service.retrieve(
        query="official cyber fraud complaint portal for UPI scam",
        query_variants=["official cyber fraud complaint portal for UPI scam"],
        state="Gujarat",
        domain="criminal",
        answer_mode="grounded_general",
        doctypes_options=["judgments"],
    )

    assert fake_google.calls
    assert fake_google.calls[0]["max_results"] == 3
    assert fake_google.calls[0]["site_restrict"] == "cybercrime.gov.in"
    assert fake_google.calls[0]["trusted_only"] is False
    assert result.google_used is True
    assert result.google_count == 1
    assert any(doc["source_kind"] == "google_custom_search" for doc in result.documents)


def test_hybrid_retrieval_skips_google_without_official_or_latest_source_need():
    fake_google = FakeGoogleSearch(
        documents=[
            {
                "doc_id": "google:1",
                "title": "Unused official result",
                "headline": "Should not be called.",
                "fragment_headline": "Unused",
                "doc_excerpt": "Unused.",
                "docsource": "google:gov.in",
                "citations": ["Unused"],
                "publishdate": "",
                "url": "https://example.gov.in/",
                "score": 16.0,
                "source_kind": "google_custom_search",
                "authority_type": "government_portal",
            }
        ]
    )
    service = LegalHybridRetrievalService(
        Settings(
            INDIANKANOON_API_TOKEN="token",
            GOOGLE_SEARCH_ENABLED="true",
            GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
            GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        ),
        corpus_index=FakeCorpusIndex(documents=[]),
        indiankanoon_service=FakeKanoon(documents=[]),
        google_search_service=fake_google,
    )

    result = service.retrieve(
        query="consumer refund remedies for defective product",
        query_variants=["consumer refund remedies for defective product"],
        state="Gujarat",
        domain="consumer",
        answer_mode="grounded_general",
        doctypes_options=["judgments"],
    )

    assert fake_google.calls == []
    assert result.google_used is False
    assert result.google_count == 0


def test_hybrid_retrieval_skips_google_when_live_authority_is_already_strong():
    fake_google = FakeGoogleSearch(
        documents=[
            {
                "doc_id": "google:1",
                "title": "National Cyber Crime Portal",
                "headline": "Official complaint portal.",
                "fragment_headline": "Portal",
                "doc_excerpt": "Use the official cybercrime complaint portal.",
                "docsource": "google:cybercrime.gov.in",
                "citations": ["National Cyber Crime Portal"],
                "publishdate": "",
                "url": "https://cybercrime.gov.in/",
                "score": 16.0,
                "source_kind": "google_custom_search",
                "authority_type": "government_portal",
            }
        ]
    )
    fake_kanoon = FakeKanoon(
        documents=[
            {
                "doc_id": "live:77",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses fraud-reporting obligations.",
                "fragment_headline": "Relevant fragment from live case law.",
                "doc_excerpt": "The live document explains the applicable legal position.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "2024",
                "url": "https://indiankanoon.org/doc/77/",
                "score": 33.0,
            }
        ]
    )
    service = LegalHybridRetrievalService(
        Settings(
            INDIANKANOON_API_TOKEN="token",
            GOOGLE_SEARCH_ENABLED="true",
            GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
            GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        ),
        corpus_index=FakeCorpusIndex(documents=[]),
        indiankanoon_service=fake_kanoon,
        google_search_service=fake_google,
    )

    result = service.retrieve(
        query="official cyber fraud complaint portal for UPI scam",
        query_variants=["official cyber fraud complaint portal for UPI scam"],
        state="Gujarat",
        domain="criminal",
        answer_mode="grounded_general",
        doctypes_options=["judgments"],
    )

    assert fake_kanoon.calls
    assert fake_google.calls == []
    assert result.live_used is True
    assert result.google_used is False


def test_hybrid_retrieval_prioritizes_curated_google_for_constitutional_queries():
    fake_google = FakeGoogleSearch(
        documents=[
            {
                "doc_id": "google:1",
                "title": "India Code - Constitution of India",
                "headline": "Official Constitution text.",
                "fragment_headline": "Official",
                "doc_excerpt": "Official Constitution text.",
                "docsource": "google:indiacode.nic.in",
                "citations": ["India Code - Constitution of India"],
                "publishdate": "",
                "url": "https://www.indiacode.nic.in/",
                "score": 18.0,
                "source_kind": "google_custom_search",
                "authority_type": "government_portal",
            }
        ]
    )
    service = LegalHybridRetrievalService(
        Settings(
            INDIANKANOON_API_TOKEN="token",
            GOOGLE_SEARCH_ENABLED="true",
            GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
            GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        ),
        corpus_index=FakeCorpusIndex(documents=[]),
        indiankanoon_service=FakeKanoon(documents=[]),
        google_search_service=fake_google,
    )

    result = service.retrieve(
        query="Article 21 of the Constitution of India",
        query_variants=["Article 21 of the Constitution of India"],
        state="Gujarat",
        domain="constitutional",
        answer_mode="statute_first",
        doctypes_options=["laws"],
    )

    assert fake_google.calls
    assert fake_google.calls[0]["trusted_only"] is True
    assert result.curated_google_count == 1
    assert result.documents[0]["title"] == "India Code - Constitution of India"


def test_hybrid_retrieval_uses_general_google_only_after_curated_google_is_weak():
    class LayeredFakeGoogleSearch(FakeGoogleSearch):
        def search(self, *, query: str, max_results: int | None = None, site_restrict: str | None = None, trusted_only: bool = True):
            self.calls.append(
                {
                    "query": query,
                    "max_results": max_results,
                    "site_restrict": site_restrict,
                    "trusted_only": trusted_only,
                }
            )
            documents = (
                [
                    {
                        "doc_id": "google:curated-weak",
                        "title": "Unofficial note",
                        "headline": "Weak note.",
                        "fragment_headline": "Weak",
                        "doc_excerpt": "Weak note.",
                        "docsource": "google:example.org.in",
                        "citations": ["Unofficial note"],
                        "publishdate": "",
                        "url": "https://example.org.in/note",
                        "score": 9.0,
                        "source_kind": "google_custom_search",
                        "authority_type": "web_reference",
                    }
                ]
                if trusted_only
                else [
                    {
                        "doc_id": "google:general-1",
                        "title": "India Code portal",
                        "headline": "Official portal.",
                        "fragment_headline": "Portal",
                        "doc_excerpt": "Official portal.",
                        "docsource": "google:indiacode.nic.in",
                        "citations": ["India Code portal"],
                        "publishdate": "",
                        "url": "https://indiacode.nic.in/",
                        "score": 17.0,
                        "source_kind": "google_custom_search",
                        "authority_type": "government_portal",
                    }
                ]
            )
            return GoogleSearchResult(
                documents=documents,
                from_cache=False,
                trusted_result_count=len(documents),
            )

    fake_google = LayeredFakeGoogleSearch(documents=[])
    service = LegalHybridRetrievalService(
        Settings(
            INDIANKANOON_API_TOKEN="token",
            GOOGLE_SEARCH_ENABLED="true",
            GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
            GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        ),
        corpus_index=FakeCorpusIndex(documents=[]),
        indiankanoon_service=FakeKanoon(documents=[]),
        google_search_service=fake_google,
    )

    result = service.retrieve(
        query="Section 420 IPC official text",
        query_variants=["Section 420 IPC official text"],
        state="Gujarat",
        domain="criminal",
        answer_mode="statute_first",
        doctypes_options=["laws"],
    )

    assert len(fake_google.calls) == 2
    assert fake_google.calls[0]["trusted_only"] is True
    assert fake_google.calls[1]["trusted_only"] is False
    assert result.general_google_count >= 1
