from __future__ import annotations

import asyncio
import io

from backend.app.services.chat_service import ChatService
from backend.app.services.file_extractor import FileExtractionService
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.legal_document_analysis import LegalDocumentAnalysisService
from backend.app.services.legal_entity_extraction import LegalEntityExtractionService
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

    config_module.TEMP_DIR = tmp_path / "tmp"
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


def test_legal_document_analysis_extracts_notice_structure():
    service = LegalDocumentAnalysisService()

    analysis = service.analyze_text(
        "LEGAL NOTICE from Alpha Traders to Beta Stores. "
        "You are called upon to pay Rs. 1,25,000 within 15 days. "
        "The notice refers to Section 138 of the Negotiable Instruments Act and cheque dishonour dated 12/04/2026.",
        document_index=1,
    )

    assert analysis["document_type"] == "legal_notice"
    assert analysis["procedural_stage"] == "notice stage"
    assert any("within 15 days" in item.lower() for item in analysis["deadlines"])
    assert any("section 138" in item.lower() for item in analysis["authorities"])
    assert any("Alpha Traders" in item for item in analysis["parties"])
    assert any("cheque dishonour" in item.lower() for item in analysis["risk_indicators"])
    assert analysis["entities"]["sections"][0]["section"] == "138"
    assert "Negotiable Instruments Act, 1881" in analysis["entities"]["statutes"]
    assert analysis["actionable_next_steps"]


def test_legal_entity_extraction_normalizes_chat_and_document_facts():
    service = LegalEntityExtractionService()

    entities = service.extract(
        "FIR No. 45/2026 was registered at Navrangpura Police Station under Section 420 IPC. "
        "The complaint mentions Rs. 50,000, Case No. CC/17/2026, and hearing on 14 May 2026."
    )

    assert "FIR 45/2026" in entities["fir_numbers"]
    assert "Navrangpura Police Station" in entities["police_stations"]
    assert entities["sections"][0]["section"] == "420"
    assert entities["sections"][0]["statute"] == "Indian Penal Code, 1860"
    assert "Indian Penal Code, 1860" in entities["statutes"]
    assert entities["money_amounts"][0]["normalized"] == "INR 50000"
    assert any("CC/17/2026" in item for item in entities["case_numbers"])
    assert any("14 May 2026" in item for item in entities["dates"])


def test_chat_upload_persists_structured_document_analysis(client, monkeypatch):
    def no_live_docs(self, query_variants, doctypes_options, max_results=4):
        return []

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", no_live_docs)

    response = client.post(
        "/chat/upload",
        data={"message": "Please review this uploaded legal notice"},
        files={
            "files": (
                "notice.txt",
                io.BytesIO(
                    b"LEGAL NOTICE from Alpha Traders to Beta Stores. You are called upon to pay Rs. 125000 within 15 days under Section 138 NI Act."
                ),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    analyses = latest_assistant["metadata"]["conversation_state"]["uploaded_document_analyses"]

    assert analyses
    assert analyses[0]["document_type"] == "legal_notice"
    assert analyses[0]["procedural_stage"] == "notice stage"
    assert any("within 15 days" in item.lower() for item in analyses[0]["deadlines"])
    assert any("section 138" in item.lower() for item in analyses[0]["authorities"])
    assert analyses[0]["entities"]["sections"][0]["section"] == "138"
    assert latest_assistant["metadata"]["uploaded_document_analyses"][0]["document_type"] == "legal_notice"
    legal_entities = latest_assistant["metadata"]["conversation_state"]["legal_entities"]
    assert legal_entities["sections"][0]["section"] == "138"
    assert any("within 15 days" in item.lower() for item in legal_entities["deadlines"])
