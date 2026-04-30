from __future__ import annotations

import requests

from backend.app.core.config import Settings
from backend.app.core.config import get_settings
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.retrieval import RetrievalChunk, RetrievalService


def test_indiankanoon_token_alias_is_loaded(monkeypatch):
    monkeypatch.setenv("INDIA_KANOON_API_TOKEN", "token-from-legacy-name")
    monkeypatch.delenv("INDIANKANOON_API_TOKEN", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.indiankanoon_api_token == "token-from-legacy-name"
    assert settings.indiankanoon_configured is True

    get_settings.cache_clear()


def test_retrieval_uses_indiankanoon_when_local_context_is_weak(monkeypatch, tmp_path):
    service = RetrievalService(get_settings())
    service.local_provider.knowledge_dir = tmp_path / "knowledge"
    service.local_provider.knowledge_dir.mkdir(parents=True, exist_ok=True)
    service.settings.indiankanoon_api_token = "live-token"

    monkeypatch.setattr(
        service.indiankanoon_provider,
        "retrieve_contextual_results",
        lambda query, doctypes=None, max_results=3: [
            {
                "doc_id": "12345",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses cheque bounce liability.",
                "fragment_headline": "Section 138 of the Negotiable Instruments Act was examined.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
            }
        ],
    )

    chunks = service.get_context(
        query="Need a Supreme Court judgment on cheque bounce under section 138",
        uploaded_texts=[],
        state="Gujarat",
        domain="consumer",
    )

    assert chunks
    assert chunks[0].source == "indiankanoon:12345"
    assert "Sample Supreme Court Decision" in chunks[0].text


def test_retrieval_skips_indiankanoon_when_local_context_is_strong(monkeypatch, tmp_path):
    service = RetrievalService(get_settings())
    service.local_provider.knowledge_dir = tmp_path / "knowledge"
    service.local_provider.knowledge_dir.mkdir(parents=True, exist_ok=True)
    (service.local_provider.knowledge_dir / "consumer.txt").write_text(
        "Consumer Protection Act refund seller warranty service e-commerce refund seller warranty service.",
        encoding="utf-8",
    )
    service.settings.indiankanoon_api_token = "live-token"

    called = {"value": False}

    def fake_live_lookup(query, doctypes=None, max_results=3):
        called["value"] = True
        return []

    monkeypatch.setattr(service.indiankanoon_provider, "retrieve_contextual_results", fake_live_lookup)

    chunks = service.get_context(
        query="refund for defective seller service warranty issue",
        uploaded_texts=[],
        state="Gujarat",
        domain="consumer",
    )

    assert chunks
    assert chunks[0].source == "consumer.txt"
    assert called["value"] is False


def test_indiankanoon_ignores_broken_env_proxy_by_default():
    settings = Settings(
        INDIANKANOON_API_TOKEN="token",
        INDIANKANOON_TRUST_ENV_PROXY="false",
    )

    service = IndianKanoonService(settings)

    assert service.session.trust_env is False


def test_indiankanoon_prefers_exact_authority_and_filters_noisy_docs():
    settings = Settings(
        INDIANKANOON_API_TOKEN="token",
        INDIANKANOON_TRUST_ENV_PROXY="false",
    )
    service = IndianKanoonService(settings)

    docs = [
        {
            "doc_id": "1",
            "title": "Section 420 in The Indian Penal Code, 1860",
            "headline": "Punishment may extend to seven years and fine.",
            "fragment_excerpt": "Punishment may extend to seven years and fine.",
            "doc_excerpt": "Cheating may be punished with imprisonment which may extend to seven years and fine.",
            "docsource": "laws",
            "score": 42.0,
        },
        {
            "doc_id": "2",
            "title": "Noisy Result",
            "headline": "debug: internal server error",
            "fragment_excerpt": "{\"errmsg\":\"Error in evaluting the fragments\"}",
            "doc_excerpt": "traceback dump",
            "docsource": "gujarat",
            "score": 41.0,
        },
        {
            "doc_id": "3",
            "title": "Another Weak Match",
            "headline": "Some distant observation",
            "fragment_excerpt": "Unrelated facts",
            "doc_excerpt": "Unrelated facts",
            "docsource": "gujarat",
            "score": 30.0,
        },
    ]

    selected = service._limit_to_most_relevant(docs, max_results=4)

    assert len(selected) == 1
    assert selected[0]["doc_id"] == "1"


def test_indiankanoon_grounded_retrieval_continues_after_failed_query_variant(monkeypatch):
    settings = Settings(
        INDIANKANOON_API_TOKEN="token",
        INDIANKANOON_TRUST_ENV_PROXY="false",
    )
    service = IndianKanoonService(settings)
    calls: list[str] = []

    def fake_search(query, page_num=0, doctypes=None, max_cites=8):
        calls.append(query)
        if query == "Section 21 BNS":
            raise requests.RequestException("temporary exact-search failure")
        return {
            "docs": [
                {
                    "tid": "21",
                    "title": "Section 21 in The Bharatiya Nyaya Sanhita, 2023",
                    "headline": "Text of Section 21.",
                    "docsource": "laws",
                    "citations": [],
                }
            ]
        }

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr(
        service,
        "get_document_fragments",
        lambda doc_id, query: {
            "title": "Section 21 in The Bharatiya Nyaya Sanhita, 2023",
            "headline": "Section 21 Bharatiya Nyaya Sanhita fragment.",
        },
    )
    monkeypatch.setattr(
        service,
        "get_document",
        lambda doc_id, max_cites=6, max_cited_by=4: {
            "doc": "Section 21 of the Bharatiya Nyaya Sanhita applies here."
        },
    )
    monkeypatch.setattr(service, "get_document_meta", lambda doc_id: {})

    docs = service.retrieve_grounded_documents(
        query_variants=["Section 21 BNS", "Section 21 Bharatiya Nyaya Sanhita"],
        doctypes_options=["laws"],
        max_results=4,
    )

    assert calls == ["Section 21 BNS", "Section 21 Bharatiya Nyaya Sanhita"]
    assert docs
    assert docs[0]["doc_id"] == "21"
