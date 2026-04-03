from __future__ import annotations

from backend.app.core.config import get_settings
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
