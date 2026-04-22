from __future__ import annotations

import asyncio
import io

from backend.app.services.chat_service import ChatService
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.openai_service import OpenAIResponsesService


def test_chat_upload_accepts_text_file(client, monkeypatch):
    def no_live_docs(self, query_variants, doctypes_options, max_results=4):
        return []

    def fake_generate_json(self, user_prompt, conversation):
        assert "Uploaded Document 1" in user_prompt
        assert "Legal notice regarding payment default." in user_prompt
        return {
            "answer": "Summary: The uploaded notice has been reviewed.\nLegal position: The uploaded document mentions payment default.\nPractical next steps: Reply with the agreement and payment record.\nSources: Uploaded Document 1 | user_upload.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "Pre-litigation / Notice Reply",
            "caution": "Check the notice deadline carefully.",
            "documents_to_keep": ["notice copy", "agreement", "payment proof"],
        }

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", no_live_docs)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)
    monkeypatch.setattr(
        ChatService,
        "_run_semantic_support_check",
        lambda self, answer, query, documents, source_sufficiency: {
            "status": "supported",
            "confidence": 0.99,
            "reason": "test override",
        },
    )

    response = client.post(
        "/chat/upload",
        data={"message": "section 138 cheque bounce legal notice"},
        files={"files": ("notice.txt", io.BytesIO(b"Legal notice regarding payment default."), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] > 0
    assert "uploaded notice has been reviewed" in payload["answer"].lower()
    assert "payment default" in payload["answer"].lower()
    assert payload["warnings"] == []
    assert "notice copy" in " ".join(payload["documents_to_keep"]).lower()
    assert any("user_upload" in citation for citation in payload["citations"])


def test_chat_upload_reflects_uploaded_facts_even_when_llm_omits_upload_reference(client, monkeypatch):
    def fake_docs(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "200",
                "title": "Section 138 NI Act",
                "headline": "Dishonour of cheque can trigger statutory notice requirements.",
                "fragment_headline": "Notice requirements under the statute.",
                "doc_excerpt": "The provision discusses cheque dishonour and notice timelines.",
                "docsource": "laws",
                "citations": ["NI Act 138"],
                "publishdate": "01-01-1881",
                "url": "https://indiankanoon.org/doc/200/",
            }
        ]

    def fake_generate_json(self, user_prompt, conversation):
        return {
            "answer": "Summary: Grounded answer.\nLegal position: Statutory notice timeline applies.\nPractical next steps: Send a compliant notice.\nSources: Section 138 NI Act.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "laws",
            "caution": None,
            "documents_to_keep": ["bank memo"],
        }

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_docs)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)

    response = client.post(
        "/chat/upload",
        data={"message": "section 138 cheque bounce legal notice"},
        files={"files": ("notice.txt", io.BytesIO(b"Uploaded notice says cheque number 7788 was dishonoured."), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "uploaded document facts were considered" in payload["answer"].lower()
    assert any("user_upload" in citation for citation in payload["citations"])


def test_file_extractor_reads_text_file(tmp_path):
    from backend.app.core import config as config_module
    from backend.app.core.config import get_settings

    config_module.TEMP_DIR = tmp_path / "temp"
    config_module.KNOWLEDGE_DIR = tmp_path / "knowledge"
    config_module.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    config_module.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    get_settings.cache_clear()
    settings = get_settings()
    extractor = FileExtractionService(settings)

    class DummyUpload:
        filename = "facts.txt"

        async def read(self):
            return b"Consumer complaint invoice and seller response."

    result = asyncio.run(extractor.extract_uploads([DummyUpload()]))
    assert result.texts
    assert "Consumer complaint" in result.texts[0]
