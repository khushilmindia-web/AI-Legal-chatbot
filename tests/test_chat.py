from __future__ import annotations

import io
import requests

from backend.app.main import create_app
from backend.app.models.schemas import ConversationState, InternalChatResult
from backend.app.services.chat_service import ChatService
from backend.app.services.google_custom_search_service import GoogleSearchResult, GoogleCustomSearchService
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.legal_hybrid_retrieval import LegalHybridRetrievalService
from backend.app.services.local_ml import LocalTextSimilarityService
from backend.app.services.openai_service import OpenAIResponsesService


def test_first_message_creates_chat_session_and_returns_grounded_answer(client):
    response = client.post(
        "/chat",
        json={
            "message": "What is the law on cheque bounce under section 138?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] > 0
    assert payload["title"].startswith("What is the law on cheque")
    assert payload["citations"]
    assert "Section 138" in payload["answer"] or "Grounded legal answer based on Indian Kanoon" in payload["answer"]
    assert payload["likely_forum"] in {"laws", "supremecourt"}

    history = client.get("/chat/history").json()
    assert len(history["items"]) == 1
    assert history["items"][0]["id"] == payload["chat_id"]


def test_new_chat_does_not_reuse_previous_chat_context(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct legal-help guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    first = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account on PhonePe",
            "state": "Gujarat",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )

    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload["chat_id"] != first_payload["chat_id"]
    assert "sbi" not in second_payload["answer"].lower()
    assert "phonepe" not in second_payload["answer"].lower()
    assert "1930" not in second_payload["answer"]
    assert second_payload["follow_up_question"]
    assert "snatching or theft" in second_payload["follow_up_question"].lower() or "police" in second_payload["follow_up_question"].lower()


def test_chat_uses_llm_output_from_grounded_indiankanoon_context(client, monkeypatch):
    def fake_generate_json(self, user_prompt, conversation):
        assert "Grounded retrieved context:" in user_prompt
        assert "Sample Supreme Court Decision" in user_prompt
        return {
            "answer": "Summary: <b>Specific grounded answer.</b>\nLegal position: Relevant fragment. Relevant fragment.\nPractical next steps: Review the authority.\nSources: Sample Supreme Court Decision; Sample Supreme Court Decision\nDisclaimer: debug: ignore this text.",
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": "Check the full judgment.",
            "documents_to_keep": ["bank memo"],
        }

    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)

    response = client.post(
        "/chat",
        json={
            "message": "section 138 cheque bounce legal position",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Specific grounded answer" in payload["answer"]
    assert "Legal Position:" in payload["answer"]
    assert "Practical Next Steps:" in payload["answer"]
    assert "Sources:" in payload["answer"]
    assert "Disclaimer:" in payload["answer"]
    assert "<b>" not in payload["answer"]
    assert "debug:" not in payload["answer"].lower()
    assert payload["documents_to_keep"] == ["bank memo"]
    assert payload["caution"] == "Check the full judgment."


def test_chat_returns_no_data_fallback_only_after_retrieval_attempts_fail(client, monkeypatch):
    monkeypatch.setattr(
        IndianKanoonService,
        "retrieve_grounded_documents",
        lambda self, query_variants, doctypes_options, max_results=4: [],
    )

    response = client.post(
        "/chat",
        json={
            "message": "An extremely obscure legal query with no matching authority",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "I am not seeing a clear enough legal match yet." in answer
    assert "Summary:" not in answer
    assert payload["follow_up_question"]
    assert "Disclaimer:" in answer


def test_chat_returns_structured_grounded_answer_when_llm_call_fails(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "ipc-420",
                "title": "Section 420 in The Indian Penal Code, 1860",
                "headline": "Cheating and dishonestly inducing delivery of property.",
                "fragment_headline": "Punishment may extend to seven years and fine.",
                "doc_excerpt": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished with imprisonment which may extend to seven years, and shall also be liable to fine.",
                "docsource": "laws",
                "citations": ["IPC 420"],
                "publishdate": "01-01-1860",
                "url": "https://indiankanoon.org/doc/111/",
                "score": 42.0,
            },
            {
                "doc_id": "case-420",
                "title": "Sample Fraud Judgment",
                "headline": "A case discussing section 420.",
                "fragment_headline": "Judicial discussion of ingredients.",
                "doc_excerpt": "This judgment discusses the ingredients of section 420.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/222/",
                "score": 18.0,
            },
        ]

    def raise_llm_error(self, user_prompt, conversation):
        raise RuntimeError("OpenAI quota is exhausted for the configured API key.")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", raise_llm_error)

    response = client.post(
        "/chat",
        json={
            "message": "section 420 ipc punishment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Summary:" in payload["answer"]
    assert "Legal Position:" in payload["answer"]
    assert "Practical Next Steps:" in payload["answer"]
    assert "Sources:" in payload["answer"]
    assert "Disclaimer:" in payload["answer"]
    assert "could not complete the response" not in payload["answer"]
    assert "Section 420 in The Indian Penal Code, 1860" in payload["answer"]
    assert "may extend to seven years" in payload["answer"]
    assert "not reliable enough to return safely" not in payload["answer"]
    assert payload["likely_forum"] == "laws"
    assert payload["documents_to_keep"]


def test_statute_llm_failure_returns_useful_answer_for_section_138(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "ni-138",
                "title": "Section 138 in The Negotiable Instruments Act, 1881",
                "headline": "Dishonour of cheque for insufficiency of funds.",
                "fragment_headline": "Drawer may be punished with imprisonment or fine subject to statutory conditions.",
                "doc_excerpt": "Where any cheque drawn by a person is returned unpaid because funds are insufficient or it exceeds the arrangement, the statutory consequences under section 138 follow subject to the provisos.",
                "docsource": "laws",
                "citations": ["NI Act 138"],
                "publishdate": "01-01-1881",
                "url": "https://indiankanoon.org/doc/333/",
                "score": 40.0,
            },
            {
                "doc_id": "case-138",
                "title": "Cheque Bounce v. Sample Party",
                "headline": "A judgment discussing section 138.",
                "fragment_headline": "Case discussion of cheque dishonour.",
                "doc_excerpt": "The case discusses cheque dishonour procedure.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/444/",
                "score": 18.0,
            },
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: (_ for _ in ()).throw(
            RuntimeError("OpenAI quota is exhausted for the configured API key.")
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce punishment", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Section 138 in The Negotiable Instruments Act, 1881" in payload["answer"]
    assert "returned unpaid" in payload["answer"]
    assert "not reliable enough to return safely" not in payload["answer"]
    assert payload["likely_forum"] == "laws"
    assert payload["citations"][0].startswith("Section 138 in The Negotiable Instruments Act, 1881 | laws")


def test_tax_statute_llm_failure_uses_domain_specific_next_steps(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "gst-16",
                "title": "Section 16 of the CGST Act, 2017",
                "headline": "Eligibility and conditions for taking input tax credit.",
                "fragment_headline": "Registered person is entitled to input tax credit subject to conditions.",
                "doc_excerpt": "Every registered person shall, subject to conditions and restrictions, be entitled to take credit of input tax charged on any supply of goods or services used in the course or furtherance of business.",
                "docsource": "laws",
                "citations": ["CGST Act 16"],
                "publishdate": "01-07-2017",
                "url": "https://indiankanoon.org/doc/555/",
                "score": 39.0,
            },
            {
                "doc_id": "gst-case",
                "title": "GST Input Credit Judgment",
                "headline": "Interpretation of section 16.",
                "fragment_headline": "Case law around eligibility.",
                "doc_excerpt": "The judgment discusses conditions for ITC.",
                "docsource": "highcourts",
                "citations": ["2024 GSTL 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/666/",
                "score": 16.0,
            },
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: (_ for _ in ()).throw(
            RuntimeError("OpenAI quota is exhausted for the configured API key.")
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "GST section 16 input tax credit eligibility", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Section 16 of the CGST Act, 2017" in payload["answer"]
    assert "invoices, returns, or eligibility conditions" in payload["answer"]
    assert "tax or compliance step" in payload["answer"]
    assert payload["likely_forum"] == "laws"


def test_chat_returns_structured_grounded_answer_when_llm_payload_is_empty(client, monkeypatch):
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": "",
            "follow_up_question": None,
            "likely_forum": None,
            "caution": None,
            "documents_to_keep": [],
        },
    )

    response = client.post(
        "/chat",
        json={
            "message": "section 420 ipc punishment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Summary:" in payload["answer"]
    assert "Legal Position:" in payload["answer"]
    assert "Practical Next Steps:" in payload["answer"]
    assert "Sources:" in payload["answer"]
    assert "Disclaimer:" in payload["answer"]
    assert payload["documents_to_keep"]
    assert payload["caution"]


def test_chat_blocks_low_confidence_grounded_results_before_llm_generation(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "weak-1",
                "title": "Loose legal note",
                "headline": "General legal discussion.",
                "fragment_headline": "General note only.",
                "doc_excerpt": "This is broad legal discussion without a clear authority match.",
                "docsource": "bloglike",
                "citations": ["Loose legal note"],
                "publishdate": "01-01-2018",
                "url": "https://example.test/weak-1",
                "score": 5.0,
            }
        ]

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not be called when retrieval confidence is low")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "What is the law on an unclear disputed point with weak authority?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "not clear enough for a reliable answer yet" in payload["answer"]
    assert "Sources:" not in payload["answer"]
    assert payload["follow_up_question"]
    assert payload["citations"] == []
    assert payload["likely_forum"] is None


def test_weak_case_law_query_returns_low_confidence_grounded_fallback(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "weak-case-1",
                "title": "Loose Discussion on Eviction Procedure",
                "headline": "General comments with weak authority match.",
                "fragment_headline": "Unclear case-law discussion.",
                "doc_excerpt": "This note broadly discusses landlord procedure without a precise latest High Court holding.",
                "docsource": "bloglike",
                "citations": ["Loose note"],
                "publishdate": "01-01-2019",
                "url": "https://example.test/weak-case",
                "score": 5.0,
            }
        ]

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not be called for weak case-law retrieval")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={"message": "Latest High Court judgment on landlord eviction procedure", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "not clear enough for a reliable answer yet" in payload["answer"]
    assert "exact case name, court, citation, or year" in (payload["follow_up_question"] or "")
    assert payload["citations"] == []
    assert payload["likely_forum"] is None


def test_grounded_follow_up_preserves_raw_user_message_and_separate_retrieval_query(client, monkeypatch):
    captured_prompts: list[str] = []

    def fake_generate_json(self, user_prompt, conversation):
        captured_prompts.append(user_prompt)
        return {
            "answer": "Summary: Grounded answer.\nLegal Position: The retrieved authority is limited.\nPractical Next Steps: Review the authority.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
            "follow_up_question": "Which court level or authority focus do you want?",
            "likely_forum": "supremecourt",
            "caution": None,
            "documents_to_keep": [],
        }

    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)

    first = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce legal position", "state": "Gujarat"},
    )
    assert first.status_code == 200
    chat_id = first.json()["chat_id"]

    second = client.post(
        "/chat",
        json={"chat_id": chat_id, "message": "what about Supreme Court?", "state": "Gujarat"},
    )
    assert second.status_code == 200

    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_user = [item for item in messages if item["role"] == "user"][-1]
    assert latest_user["content"] == "what about Supreme Court?"
    assert latest_user["metadata"]["raw_user_message"] == "what about Supreme Court?"
    assert latest_user["metadata"]["retrieval_query"] != latest_user["metadata"]["raw_user_message"]
    assert "what about Supreme Court?" in latest_user["metadata"]["retrieval_query"]


def test_low_confidence_fallback_stores_reason_code_in_assistant_metadata(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "weak-1",
                "title": "Loose legal note",
                "headline": "General legal discussion.",
                "fragment_headline": "General note only.",
                "doc_excerpt": "This is broad legal discussion without a clear authority match.",
                "docsource": "bloglike",
                "citations": ["Loose legal note"],
                "publishdate": "01-01-2018",
                "url": "https://example.test/weak-1",
                "score": 5.0,
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)

    response = client.post(
        "/chat",
        json={"message": "What is the law on an unclear disputed point with weak authority?", "state": "Gujarat"},
    )
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]

    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    retrieval = latest_assistant["metadata"]["retrieval"]
    assert retrieval["fallback_type"] == "low_confidence"
    assert retrieval["fallback_reason_code"] == "retrieval_low_confidence"


def test_chat_normalizes_messy_llm_output_into_strict_sections(client, monkeypatch):
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": (
                "Summary: <div>Cheque bounce overview</div>\n"
                "Legal position: <b>Dishonour may trigger statutory consequences.</b>\n"
                "Legal position: Dishonour may trigger statutory consequences.\n"
                "Practical next steps: Send notice promptly.\n"
                "Sources: Sample Supreme Court Decision; Sample Supreme Court Decision\n"
                "Disclaimer: debug: stack trace hidden"
            ),
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": None,
            "documents_to_keep": [],
        },
    )

    response = client.post(
        "/chat",
        json={
            "message": "section 138 cheque bounce legal position",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer.startswith("Summary:")
    assert "\nLegal Position:" in answer
    assert "\nPractical Next Steps:" in answer
    assert "\nSources:" in answer
    assert "\nDisclaimer:" in answer
    assert "<div>" not in answer
    assert "<b>" not in answer
    assert "debug:" not in answer.lower()
    assert answer.count("Sample Supreme Court Decision") == 1


def test_final_output_validator_strips_unsupported_certainty_deadline_and_reference_claims(client, monkeypatch):
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": (
                "Summary: You will definitely win this matter.\n"
                "Legal Position: Section 999 clearly applies here.\n"
                "Practical Next Steps: You must file within 30 days.\n"
                "Sources: Sample Supreme Court Decision\n"
                "Disclaimer: This is general legal information."
            ),
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": None,
            "documents_to_keep": [],
        },
    )

    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce legal position", "state": "Gujarat"},
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "definitely win" not in answer.lower()
    assert "section 999" not in answer.lower()
    assert "within 30 days" not in answer.lower()
    assert "Disclaimer:" in answer


def test_grounded_metadata_includes_query_profile_source_sufficiency_and_disclaimer_mode(client):
    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce legal position", "state": "Gujarat"},
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    assert latest_assistant["metadata"]["query_profile"]["query_type"] in {"statute_lookup", "mixed"}
    assert latest_assistant["metadata"]["source_sufficiency"]["label"] in {"strong", "partial", "weak"}
    assert latest_assistant["metadata"]["disclaimer_mode"] in {"low_risk", "medium_risk", "high_risk"}


def test_legal_help_interview_records_playbook_metadata(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for this direct cyber-fraud playbook question")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={"message": "I clicked a fake UPI link and money got debited", "state": "Gujarat"},
    )
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    assert latest_assistant["metadata"]["playbook_id"] == "playbook_cyber_fraud_v1"
    assert latest_assistant["metadata"]["disclaimer_mode"] in {"medium_risk", "high_risk"}


def test_practical_legal_help_bypasses_strict_grounded_validation(client, monkeypatch):
    def fake_legal_help(self, *, message, domain, warnings, conversation_state, is_continuation):
        return (
            InternalChatResult(
                answer=(
                    "Summary: This looks like a cyber-fraud complaint that needs immediate reporting.\n"
                    "Legal Position: You should report the UPI fraud within 24 hours and can mention Section 154 CrPC when asking the police to record the complaint.\n"
                    "Practical Next Steps: 1. Call 1930 immediately. 2. Report the transaction to your bank and app. 3. Preserve screenshots, transaction ID, and complaint reference.\n"
                    "Sources: Practical legal-help workflow.\n"
                    "Disclaimer: This is general legal information."
                ),
                domain=domain,
                follow_up_question="Which bank and app were involved?",
                citations=[],
                authorities=[],
                documents_to_keep=["screenshots", "transaction ID"],
                likely_forum="cybercrime_portal",
                caution="Act quickly.",
                warnings=warnings,
                raw_json={
                    "source": "legal_help",
                    "playbook_id": "playbook_cyber_fraud_v1",
                    "disclaimer_mode": "high_risk",
                },
            ),
            ConversationState(
                conversation_started=True,
                active_intent="legal_help",
                awaiting_details=True,
                last_user_issue=message,
                legal_domain=domain,
                last_follow_up_question="Which bank and app were involved?",
                issue_type="cyber_fraud",
            ),
        )

    monkeypatch.setattr(ChatService, "_handle_legal_help_interview", fake_legal_help)

    response = client.post(
        "/chat",
        json={"message": "I clicked a fake UPI link and money got debited", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "within 24 hours" in payload["answer"].lower()
    assert "section 154 crpc" in payload["answer"].lower()
    assert "call 1930 immediately" in payload["answer"].lower()

    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    assert latest_assistant["metadata"]["pipeline"] == "legal_help"
    assert latest_assistant["metadata"]["validation_flags"] == []


def test_weak_practical_query_reroutes_to_legal_help_instead_of_fallback(client, monkeypatch):
    def force_grounded_path(self, *, message, domain, warnings, conversation_state, query_profile):
        return None

    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return []

    def fake_legal_help(self, *, message, domain, warnings, conversation_state, is_continuation):
        return (
            InternalChatResult(
                answer=(
                    "Summary: This looks like a landlord-harassment issue that needs immediate practical steps.\n"
                    "Legal Position: You can start with a written record of the threats and use the police or local authority route if the harassment escalates.\n"
                    "Practical Next Steps: 1. Preserve messages and call records. 2. Send one clear written objection. 3. If there is threat, lockout, or force, approach the local police immediately.\n"
                    "Sources: General procedural legal guidance.\n"
                    "Disclaimer: This is general legal information."
                ),
                domain=domain,
                follow_up_question="Has the landlord already changed the lock, cut utilities, or given a written threat?",
                citations=[],
                authorities=[],
                documents_to_keep=["messages", "rent receipts"],
                likely_forum="local_police_or_civil_forum",
                caution="Act quickly if there is any threat of lockout.",
                warnings=warnings,
                raw_json={
                    "source": "legal_help",
                    "playbook_id": "playbook_landlord_harassment_v1",
                    "disclaimer_mode": "medium_risk",
                },
            ),
            ConversationState(
                conversation_started=True,
                active_intent="legal_help",
                awaiting_details=True,
                last_user_issue=message,
                legal_domain=domain,
                last_follow_up_question="Has the landlord already changed the lock, cut utilities, or given a written threat?",
                issue_type="landlord_harassment",
            ),
        )

    monkeypatch.setattr(ChatService, "_route_direct_response", force_grounded_path)
    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(ChatService, "_handle_legal_help_interview", fake_legal_help)

    response = client.post(
        "/chat",
        json={"message": "My landlord is threatening me and may lock me out", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "landlord-harassment issue" in payload["answer"].lower()
    assert "preserve messages" in payload["answer"].lower()
    assert "No highly relevant India Kanoon authority was found" not in payload["answer"]
    assert "not strong enough yet for a reliable grounded answer" not in payload["answer"]


def test_grounded_answer_injects_jurisdiction_and_recency_note_when_sources_are_dated_or_mismatched(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "dated-1",
                "title": "Old procedural authority",
                "headline": "General procedural note.",
                "fragment_headline": "General procedural note.",
                "doc_excerpt": "General procedural note only.",
                "docsource": "delhi",
                "citations": ["Old authority"],
                "publishdate": "01-01-2010",
                "url": "https://indiankanoon.org/doc/dated-1/",
                "score": 26.0,
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(
        ChatService,
        "_run_semantic_support_check",
        lambda self, *, answer, query, documents, source_sufficiency: {
            "status": "supported",
            "score": 1.0,
            "claim_count": 1,
            "supported_claims": 1,
            "backend": "test",
            "details": [],
        },
    )
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": "Summary: Procedure answer.\nLegal Position: Old procedural authority gives a broad procedural answer.\nPractical Next Steps: Review the dated authority carefully.\nSources: Old procedural authority.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "delhi",
            "caution": None,
            "documents_to_keep": [],
        },
    )

    response = client.post(
        "/chat",
        json={"message": "latest procedural authority on cheque bounce", "state": "Gujarat"},
    )
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "do not appear to be specific to Gujarat" in answer or "may be dated" in answer


def test_chat_prefers_primary_authority_and_excludes_noisy_secondary_docs(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "111",
                "title": "Section 420 in The Indian Penal Code, 1860",
                "headline": "Whoever cheats and dishonestly induces delivery of property may be punished.",
                "fragment_headline": "Punishment may extend to seven years and fine.",
                "doc_excerpt": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished with imprisonment which may extend to seven years, and shall also be liable to fine.",
                "docsource": "laws",
                "citations": ["IPC 420"],
                "publishdate": "01-01-1860",
                "url": "https://indiankanoon.org/doc/111/",
                "score": 50.0,
            },
            {
                "doc_id": "222",
                "title": "Unrelated Case Dump",
                "headline": "debug: internal server error",
                "fragment_headline": "{\"errmsg\":\"Error in evaluting the fragments\"}",
                "doc_excerpt": "traceback debug dump",
                "docsource": "gujarat",
                "citations": ["AIR 1999 Guj 1"],
                "publishdate": "01-01-1999",
                "url": "https://indiankanoon.org/doc/222/",
                "score": 30.0,
            },
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": "Summary: Section 420 answer.\nLegal position: Punishment may extend to seven years and fine.\nPractical next steps: Read the section.\nSources: Section 420 in The Indian Penal Code, 1860; Unrelated Case Dump\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "laws",
            "caution": None,
            "documents_to_keep": [],
        },
    )

    response = client.post(
        "/chat",
        json={"message": "section 420 ipc punishment", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Unrelated Case Dump" not in payload["answer"]
    assert "traceback" not in payload["answer"].lower()
    assert "debug" not in payload["answer"].lower()
    assert payload["citations"] == ["Section 420 in The Indian Penal Code, 1860 | laws | 01-01-1860 | https://indiankanoon.org/doc/111/"]


def test_statute_query_answers_statute_first_before_cases(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "222",
                "title": "Cheque Bounce v. Sample Party",
                "headline": "A judgment discussing section 138.",
                "fragment_headline": "Case discussion of cheque dishonour.",
                "doc_excerpt": "The case discusses cheque dishonour procedure.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/222/",
                "score": 18.0,
            },
            {
                "doc_id": "111",
                "title": "Section 138 in The Negotiable Instruments Act, 1881",
                "headline": "Dishonour of cheque for insufficiency of funds.",
                "fragment_headline": "Statutory text of section 138.",
                "doc_excerpt": "Where any cheque drawn by a person is returned unpaid, the statutory consequence follows.",
                "docsource": "laws",
                "citations": ["NI Act 138"],
                "publishdate": "01-01-1881",
                "url": "https://indiankanoon.org/doc/111/",
                "score": 35.0,
            },
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)

    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce", "state": "Gujarat"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert payload["citations"][0].startswith("Section 138 in The Negotiable Instruments Act, 1881 | laws")


def test_irrelevant_case_is_filtered_before_grounded_answer(client, monkeypatch):
    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "555",
                "title": "Landlord Eviction Dispute",
                "headline": "A property dispute with no section 138 discussion.",
                "fragment_headline": "Tenancy and eviction issues.",
                "doc_excerpt": "This case is about landlord eviction and tenancy conflict.",
                "docsource": "supremecourt",
                "citations": ["(2020) 3 SCC 10"],
                "publishdate": "01-01-2020",
                "url": "https://indiankanoon.org/doc/555/",
                "score": 9.0,
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)

    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce", "state": "Gujarat"},
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "No highly relevant India Kanoon authority was found" in answer
    assert "Landlord Eviction Dispute" not in answer


def test_final_output_validation_removes_filler_phrases(client, monkeypatch):
    monkeypatch.setattr(
        OpenAIResponsesService,
        "generate_json",
        lambda self, user_prompt, conversation: {
            "answer": (
                "Summary: It is important to note that section 138 applies.\n"
                "Legal position: Kindly note that the statutory notice matters.\n"
                "Practical next steps: Please note that send the notice in time.\n"
                "Sources: Sample Supreme Court Decision\n"
                "Disclaimer: This is general legal information."
            ),
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": None,
            "documents_to_keep": ["bank memo"],
        },
    )

    response = client.post(
        "/chat",
        json={"message": "section 138 cheque bounce legal position", "state": "Gujarat"},
    )

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert "it is important to note that" not in answer
    assert "kindly note that" not in answer
    assert "please note that" not in answer


def test_chat_handles_indiankanoon_request_failure_without_crashing(client, monkeypatch):
    def raise_timeout(self, query_variants, doctypes_options, max_results=4):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", raise_timeout)

    response = client.post(
        "/chat",
        json={
            "message": "Show me a judgment on anticipatory bail.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer.startswith("Summary:")
    assert "Legal Position:" in answer
    assert "Practical Next Steps:" in answer
    assert "Sources:" in answer
    assert "Disclaimer:" in answer


def test_indiankanoon_follow_up_turn_merges_previous_query_and_changes_answer(client, monkeypatch):
    monkeypatch.setattr(
        ChatService,
        "_run_semantic_support_check",
        lambda self, *, answer, query, documents, source_sufficiency: {
            "status": "supported",
            "score": 1.0,
            "claim_count": 1,
            "supported_claims": 1,
            "backend": "test",
            "details": [],
        },
    )
    monkeypatch.setattr(
        IndianKanoonService,
        "retrieve_grounded_documents",
        lambda self, query_variants, doctypes_options, max_results=4: [
            {
                "doc_id": "111",
                "title": "Section 138 in The Negotiable Instruments Act, 1881",
                "headline": "Dishonour of cheque for insufficiency of funds.",
                "fragment_headline": "Statutory text of section 138.",
                "doc_excerpt": "Where any cheque drawn by a person is returned unpaid because funds are insufficient, the statutory consequences under section 138 follow subject to the provisos.",
                "docsource": "laws",
                "citations": ["NI Act 138"],
                "publishdate": "01-01-1881",
                "url": "https://indiankanoon.org/doc/111/",
                "score": 35.0,
            },
            {
                "doc_id": "12345",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses cheque bounce liability.",
                "fragment_headline": "Section 138 of the Negotiable Instruments Act was examined.",
                "doc_excerpt": "The document explains the applicable legal position and the immediate procedural steps available.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/12345/",
                "score": 24.0,
            }
        ],
    )

    call_count = {"value": 0}

    def fake_generate_json(self, user_prompt, conversation):
        call_count["value"] += 1
        if call_count["value"] == 1:
            assert "section 138 cheque bounce" in user_prompt.lower()
            return {
                "answer": "Summary: Initial grounded answer.\nLegal position: Section 138 is engaged.\nPractical next steps: Narrow the forum.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
                "follow_up_question": "Which court level or authority focus do you want?",
                "likely_forum": "supremecourt",
                "caution": None,
                "documents_to_keep": ["cheque return memo"],
            }
        assert "section 138 cheque bounce" in user_prompt.lower()
        assert "supreme court only" in user_prompt.lower()
        return {
            "answer": "Summary: Refined grounded answer for Supreme Court only.\nLegal position: Supreme Court interpretation of section 138 is the focus.\nPractical next steps: Review the Supreme Court line of authority.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": "Match the authority to your facts.",
            "documents_to_keep": ["cheque return memo"],
        }

    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)

    first = client.post(
        "/chat",
        json={
            "message": "section 138 cheque bounce",
            "state": "Gujarat",
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["follow_up_question"]

    second = client.post(
        "/chat",
        json={
            "chat_id": first_payload["chat_id"],
            "message": "Supreme Court only",
            "state": "Gujarat",
        },
    )

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["answer"] != first_payload["answer"]
    assert "Supreme Court only" in second_payload["answer"] or "Supreme Court interpretation" in second_payload["answer"]
    assert second_payload["follow_up_question"] is None
    assert second_payload["caution"] == "Match the authority to your facts."


def test_indiankanoon_follow_up_persists_merged_state_and_pipeline(client, monkeypatch):
    monkeypatch.setattr(
        ChatService,
        "_run_semantic_support_check",
        lambda self, *, answer, query, documents, source_sufficiency: {
            "status": "supported",
            "score": 1.0,
            "claim_count": 1,
            "supported_claims": 1,
            "backend": "test",
            "details": [],
        },
    )
    monkeypatch.setattr(
        IndianKanoonService,
        "retrieve_grounded_documents",
        lambda self, query_variants, doctypes_options, max_results=4: [
            {
                "doc_id": "111",
                "title": "Section 138 in The Negotiable Instruments Act, 1881",
                "headline": "Dishonour of cheque for insufficiency of funds.",
                "fragment_headline": "Statutory text of section 138.",
                "doc_excerpt": "Where any cheque drawn by a person is returned unpaid because funds are insufficient, the statutory consequences under section 138 follow subject to the provisos.",
                "docsource": "laws",
                "citations": ["NI Act 138"],
                "publishdate": "01-01-1881",
                "url": "https://indiankanoon.org/doc/111/",
                "score": 35.0,
            },
            {
                "doc_id": "12345",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses cheque bounce liability.",
                "fragment_headline": "Section 138 of the Negotiable Instruments Act was examined.",
                "doc_excerpt": "The document explains the applicable legal position and the immediate procedural steps available.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/12345/",
                "score": 24.0,
            }
        ],
    )

    call_count = {"value": 0}

    def fake_generate_json(self, user_prompt, conversation):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "answer": "Summary: Initial grounded answer.\nLegal position: Section 138 is engaged.\nPractical next steps: Narrow the forum.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
                "follow_up_question": "Which authority focus should I use?",
                "likely_forum": "supremecourt",
                "caution": None,
                "documents_to_keep": ["cheque return memo"],
            }
        return {
            "answer": "Summary: Refined grounded answer for Supreme Court only.\nLegal position: Supreme Court interpretation of section 138 is the focus.\nPractical next steps: Review the Supreme Court line of authority.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": None,
            "documents_to_keep": ["cheque return memo"],
        }

    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)

    first = client.post(
        "/chat",
        json={
            "message": "section 138 cheque bounce",
            "state": "Gujarat",
        },
    )
    assert first.status_code == 200
    chat_id = first.json()["chat_id"]

    second = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "Supreme Court only",
            "state": "Gujarat",
        },
    )
    assert second.status_code == 200

    messages_response = client.get(f"/chat/{chat_id}/messages")
    assert messages_response.status_code == 200
    items = messages_response.json()["items"]
    assert items[-1]["role"] == "assistant"
    metadata = items[-1]["metadata"]
    assert metadata["pipeline"] == "indiankanoon_rag"
    state = metadata["conversation_state"]
    assert state["active_intent"] == "indiankanoon_rag"
    assert state["awaiting_details"] is False
    assert state["last_user_issue"].lower() == "supreme court only"
    assert "section 138 cheque bounce" in state["last_grounded_query"].lower()
    assert "supreme court only" in state["last_grounded_query"].lower()


def test_chat_upload_accepts_text_file_and_uses_grounded_pipeline(client):
    response = client.post(
        "/chat/upload",
        data={"message": "section 138 cheque bounce legal notice"},
        files={"files": ("notice.txt", io.BytesIO(b"Legal notice regarding payment default."), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] > 0
    assert payload["answer"].startswith("Summary:")
    assert payload["warnings"] == []


def test_open_old_chat_returns_messages(client):
    created = client.post(
        "/chat",
        json={
            "message": "How is section 420 IPC usually interpreted?",
            "state": "Gujarat",
        },
    ).json()

    response = client.get(f"/chat/{created['chat_id']}/messages")

    assert response.status_code == 200
    messages = response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["pipeline"] == "indiankanoon_rag"


def test_clear_history_removes_all_chats(client):
    first = client.post("/chat", json={"message": "Need a grounded answer on a landlord notice."}).json()
    client.post("/chat", json={"message": "What is the law on cheque bounce under section 138?"})

    clear_response = client.delete("/chat/history")
    history_response = client.get("/chat/history")
    deleted_messages = client.get(f"/chat/{first['chat_id']}/messages")

    assert clear_response.status_code == 200
    assert history_response.json()["items"] == []
    assert deleted_messages.status_code == 404


def test_delete_single_chat_removes_only_selected_session(client):
    first = client.post("/chat", json={"message": "Need a grounded answer on a landlord notice."}).json()
    second = client.post("/chat", json={"message": "What is the law on cheque bounce under section 138?"}).json()

    delete_response = client.delete(f"/chat/{first['chat_id']}")
    history_response = client.get("/chat/history")
    deleted_messages = client.get(f"/chat/{first['chat_id']}/messages")
    kept_messages = client.get(f"/chat/{second['chat_id']}/messages")

    assert delete_response.status_code == 200
    assert [item["id"] for item in history_response.json()["items"]] == [second["chat_id"]]
    assert deleted_messages.status_code == 404
    assert kept_messages.status_code == 200


def test_delete_single_chat_returns_404_for_missing_or_other_users_chat(client, anonymous_client):
    created = client.post("/chat", json={"message": "Need a grounded answer on a landlord notice."}).json()

    other_client = anonymous_client
    signup = other_client.post(
        "/auth/signup",
        json={
            "full_name": "Other User",
            "email": "other@example.com",
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert signup.status_code == 200
    response = other_client.delete(f"/chat/{created['chat_id']}")

    assert response.status_code == 404


def test_procedural_query_routes_to_direct_legal_help_without_indiankanoon(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for procedural guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "How do I file an FIR after fraud?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "I understand" in payload["answer"]
    assert "police station" in payload["answer"].lower() or "1930" in payload["answer"] or "portal" in payload["answer"].lower()
    assert "first step" in payload["answer"].lower() or "next practical step" in payload["answer"].lower() or "what i suggest first" in payload["answer"].lower()
    assert "police station" in payload["answer"].lower() or "1930" in payload["answer"] or "portal" in payload["answer"].lower()
    assert "One more thing I need to advise you properly" not in payload["answer"]
    assert "Which city or police station" not in payload["answer"]
    assert payload["follow_up_question"]
    assert payload["citations"] == []


def test_legal_help_intake_hides_sections_until_user_asks(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct procedural guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "How do I file an FIR after fraud?",
            "state": "Gujarat",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    answer_lower = payload["answer"].lower()
    assert "bns section" not in answer_lower
    assert "bnss section" not in answer_lower
    assert payload["citations"] == []


def test_legal_help_intake_can_show_sections_when_user_explicitly_asks(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct procedural guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "How do I file an FIR after fraud and under which section?",
            "state": "Gujarat",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert "bns section 318" in answer or "bns section 319" in answer
    assert "bnss section 173" in answer
    assert "underlying offence" in answer or "cheating" in answer or "personation" in answer
    assert answer.count("1930") <= 1
    assert payload["citations"] == [] or any("BNS" in item or "BNSS" in item for item in payload["citations"])


def test_practical_landlord_query_with_authority_language_still_routes_to_legal_help(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for practical landlord guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "What legal position do I have if my landlord is threatening eviction and lockout?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert "landlord" in answer or "rent" in answer or "tenancy" in answer
    assert "lockout" in answer or "threat" in answer or "police" in answer
    assert "no highly relevant india kanoon authority was found" not in answer


def test_latest_authority_query_on_landlord_topic_stays_grounded(client, monkeypatch):
    captured: dict[str, object] = {}

    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        captured["query_variants"] = query_variants
        captured["doctypes_options"] = doctypes_options
        return [
            {
                "doc_id": "landlord-1",
                "title": "Latest High Court judgment on landlord eviction procedure",
                "headline": "High Court discusses eviction procedure and notice requirements.",
                "fragment_headline": "Eviction procedure and notice requirements.",
                "doc_excerpt": "The judgment discusses the procedure and notice requirements in landlord eviction matters.",
                "docsource": "highcourt",
                "citations": ["2024 HC 123"],
                "publishdate": "01-03-2024",
                "url": "https://indiankanoon.org/doc/landlord-1/",
                "score": 34.0,
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)

    response = client.post(
        "/chat",
        json={
            "message": "Latest High Court judgment on landlord eviction procedure",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["query_variants"]
    assert payload["citations"]
    assert "i understand this is stressful" not in payload["answer"].lower()
    assert "landlord-tenant dispute" not in payload["answer"].lower()


def test_statute_query_still_uses_indiankanoon_pipeline(client):
    response = client.post(
        "/chat",
        json={
            "message": "section 420 ipc punishment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert "Grounded legal answer based on Indian Kanoon" in payload["answer"] or "Section 420" in payload["answer"]


def test_fast_authority_lookup_bypasses_grounded_pipeline_for_article_query(client, monkeypatch):
    def fail_if_retrieval_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a direct Article lookup")

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not run for a direct Article lookup")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_retrieval_called)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21" in payload["answer"]
    assert "life and personal liberty" in payload["answer"].lower()
    assert "In simple terms" not in payload["answer"]
    assert "In broad terms" not in payload["answer"]
    assert "Practical Next Steps:" not in payload["answer"]
    assert payload["citations"] == ["Constitution of India | Article 21"]
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_fast_authority_lookup_uses_normalized_query_for_typo_article_request(client, monkeypatch):
    def fail_if_retrieval_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a typo-tolerant direct Article lookup")

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not run for a typo-tolerant direct Article lookup")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_retrieval_called)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "artcle 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21" in payload["answer"]
    assert "life and personal liberty" in payload["answer"].lower()
    assert payload["citations"] == ["Constitution of India | Article 21"]
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_fast_authority_lookup_uses_controlled_fuzzy_matching_for_article_alias(client, monkeypatch):
    def fail_if_retrieval_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fuzzy-matched direct Article lookup")

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not run for a fuzzy-matched direct Article lookup")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_retrieval_called)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "artikal 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21" in payload["answer"]
    assert "life and personal liberty" in payload["answer"].lower()
    assert "Practical Next Steps:" not in payload["answer"]
    assert payload["citations"] == ["Constitution of India | Article 21"]
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_fast_authority_lookup_bypasses_grounded_pipeline_for_common_section_query(client, monkeypatch):
    def fail_if_retrieval_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a direct section lookup")

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not run for a direct section lookup")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_retrieval_called)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "Section 420 IPC",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Section 420" in payload["answer"]
    assert "cheating" in payload["answer"].lower()
    assert "Practical Next Steps:" not in payload["answer"]
    assert payload["citations"] == ["Indian Penal Code, 1860 | Section 420"]
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_domain_classifier_labels_query_before_indiankanoon_call(client, monkeypatch):
    captured: dict[str, object] = {}

    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        captured["query_variants"] = query_variants
        captured["doctypes_options"] = doctypes_options
        return [
            {
                "doc_id": "tax-1",
                "title": "Section 16 of the CGST Act",
                "headline": "Input tax credit conditions under GST law.",
                "fragment_headline": "Section 16 eligibility conditions.",
                "doc_excerpt": "The statute explains conditions for availing input tax credit.",
                "docsource": "laws",
                "citations": ["CGST Act"],
                "publishdate": "01-07-2017",
                "url": "https://indiankanoon.org/doc/tax-1/",
                "score": 48.0,
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)

    response = client.post(
        "/chat",
        json={
            "message": "GST section 16 input tax credit eligibility",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "tax"
    assert captured["query_variants"]
    assert "laws" in captured["doctypes_options"]


def test_pipeline_classifier_marks_fast_path_in_message_metadata(client):
    response = client.post(
        "/chat",
        json={
            "message": "Article 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    user_message = [item for item in messages if item["role"] == "user"][-1]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert user_message["metadata"]["route_classification"]["path"] == "fast"
    assert assistant_message["metadata"]["route_classification"]["path"] == "fast"


def test_pipeline_classifier_marks_medium_path_for_legal_help_query(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct legal-help guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_pipeline_classifier_marks_heavy_path_for_grounded_authority_query(client):
    response = client.post(
        "/chat",
        json={
            "message": "section 420 ipc punishment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "heavy"


def test_classifier_activation_controls_fast_path_entry(client, monkeypatch):
    monkeypatch.setattr(
        ChatService,
        "_classify_pipeline_path",
        lambda self, message, domain, conversation_state: {"path": "heavy", "reason": "forced_heavy_for_test"},
    )
    monkeypatch.setattr(
        ChatService,
        "_route_fast_authority_lookup",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("Fast path should not run when classifier is not fast")),
    )

    response = client.post(
        "/chat",
        json={
            "message": "Article 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] != ["Constitution of India | Article 21"]
    assert "authority_fast_path" not in payload["answer"]


def test_fast_path_does_not_initialize_heavy_services(client, monkeypatch):
    monkeypatch.setattr(
        LegalHybridRetrievalService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("Hybrid retrieval should not initialize for fast path")),
    )
    monkeypatch.setattr(
        OpenAIResponsesService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI service should not initialize for fast path")),
    )
    monkeypatch.setattr(
        LocalTextSimilarityService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("Local similarity should not initialize for fast path")),
    )

    response = client.post(
        "/chat",
        json={
            "message": "Article 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    assert "Article 21" in response.json()["answer"]


def test_medium_path_does_not_initialize_heavy_services(client, monkeypatch):
    monkeypatch.setattr(
        LegalHybridRetrievalService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("Hybrid retrieval should not initialize for medium path")),
    )
    monkeypatch.setattr(
        OpenAIResponsesService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI service should not initialize for medium path")),
    )
    monkeypatch.setattr(
        LocalTextSimilarityService,
        "__init__",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("Local similarity should not initialize for medium path")),
    )

    response = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    assert response.json()["follow_up_question"]


def test_app_reuses_single_chat_service_instance(monkeypatch):
    app = create_app()
    first = app.state.chat_service
    second = app.state.chat_service
    assert first is second


def test_understand_legal_query_normalizes_typos_and_extracts_detail_instructions():
    service = ChatService.__new__(ChatService)

    profile = service._understand_legal_query("Explain fundamantal rights with artcle numbers in short")

    assert profile["normalized_query"] == "explain fundamental rights with article numbers in short"
    assert profile["corrected_terms"]
    assert profile["detail_instructions"]["include_articles"] is True
    assert profile["detail_instructions"]["concise"] is True
    assert profile["detail_instructions"]["step_by_step"] is False
    assert profile["answer_intent"] == "explainer"
    assert profile["mixed_components"] == []


def test_understand_legal_query_uses_controlled_fuzzy_matching_for_common_legal_terms():
    service = ChatService.__new__(ChatService)

    profile = service._understand_legal_query("Explain fundmental roghts with artickle numbers in short")

    assert profile["normalized_query"] == "explain fundamental rights with article numbers in short"
    assert "fundmental->fundamental" in profile["corrected_terms"]
    assert "roghts->rights" in profile["corrected_terms"]
    assert "artickle->article" in profile["corrected_terms"]
    assert profile["answer_intent"] == "explainer"


def test_understand_legal_query_controlled_fuzzy_matching_preserves_legal_identifiers():
    service = ChatService.__new__(ChatService)

    profile = service._understand_legal_query("artickle 226 and section 498a ipcc")

    assert profile["normalized_query"] == "article 226 and section 498a ipc"
    assert "226" in profile["normalized_query"]
    assert "498a" in profile["normalized_query"]
    assert "ipcc->ipc" in profile["corrected_terms"]


def test_understand_legal_query_classifies_direct_mixed_components():
    service = ChatService.__new__(ChatService)

    profile = service._understand_legal_query("Article 21 and fundamental rights")

    assert profile["answer_intent"] == "mixed_direct"
    assert profile["all_components_direct"] is True
    assert {"kind": "authority", "key": "constitution_article_21"} in profile["mixed_components"]
    assert {"kind": "explainer", "key": "fundamental_rights"} in profile["mixed_components"]


def test_understand_legal_query_classifies_grouped_authority_components():
    service = ChatService.__new__(ChatService)

    article_profile = service._understand_legal_query("Article 19 and 21")
    section_profile = service._understand_legal_query("Section 420 and 406 IPC")

    assert article_profile["answer_intent"] == "mixed_direct"
    assert {"kind": "authority", "key": "constitution_article_19"} in article_profile["mixed_components"]
    assert {"kind": "authority", "key": "constitution_article_21"} in article_profile["mixed_components"]
    assert section_profile["answer_intent"] == "mixed_direct"
    assert {"kind": "authority", "key": "ipc_section_420"} in section_profile["mixed_components"]
    assert {"kind": "authority", "key": "ipc_section_406"} in section_profile["mixed_components"]


def test_understand_legal_query_classifies_authority_explainer_intent():
    service = ChatService.__new__(ChatService)

    profile = service._understand_legal_query("artikal 21")

    assert profile["answer_intent"] == "authority_explainer"
    assert profile["all_components_direct"] is False


def test_understand_legal_query_builds_direct_answer_format_profile_for_direct_modes():
    service = ChatService.__new__(ChatService)

    authority = service._understand_legal_query("artikal 21")
    explainer = service._understand_legal_query("Explain fundamental rights with article numbers in short")
    mixed = service._understand_legal_query("Article 21 and fundamental rights with article numbers")

    assert authority["direct_answer_format"]["family"] == "authority"
    assert authority["direct_answer_format"]["layout"] == "definition"
    assert explainer["direct_answer_format"]["family"] == "explainer"
    assert explainer["direct_answer_format"]["layout"] == "article_breakdown"
    assert explainer["direct_answer_format"]["detail_instructions"]["concise"] is True
    assert mixed["direct_answer_format"]["family"] == "mixed"
    assert mixed["direct_answer_format"]["layout"] == "mixed_direct"
    assert mixed["direct_answer_format"]["authority_layout"] == "definition"
    assert mixed["direct_answer_format"]["explainer_layout"] == "article_breakdown"


def test_understand_legal_query_extracts_pointwise_direct_answer_instructions():
    service = ChatService.__new__(ChatService)

    explainer = service._understand_legal_query("Explain fundamental rights in points")
    authority = service._understand_legal_query("Article 21 in points")

    assert explainer["detail_instructions"]["in_points"] is True
    assert explainer["direct_answer_format"]["layout"] == "points"
    assert authority["detail_instructions"]["in_points"] is True
    assert authority["direct_answer_format"]["layout"] == "points"


def test_understand_legal_query_adds_clarification_hint_for_low_confidence_direct_queries():
    service = ChatService.__new__(ChatService)

    constitutional = service._understand_legal_query("constitutional rights")
    authority = service._understand_legal_query("article in constitution")

    assert constitutional["clarification_hint"]["kind"] == "constitutional_topic"
    assert "Fundamental Rights" in constitutional["clarification_hint"]["question"]
    assert authority["clarification_hint"]["kind"] == "authority_article"
    assert "Which Constitution article" in authority["clarification_hint"]["question"]


def test_safe_fallback_uses_human_clarification_for_unsupported_output():
    service = ChatService.__new__(ChatService)

    result = service._build_safe_fallback_result(
        kind="unsupported_output",
        domain="civil",
        warnings=[],
        query="some unclear legal query",
    )

    assert "I am not confident enough to give a safe answer in that form yet." in result.answer
    assert result.follow_up_question
    assert result.raw_json["fallback_reason_code"] == "unsupported_output"


def test_pipeline_classifier_uses_understanding_profile_for_direct_modes():
    service = ChatService.__new__(ChatService)
    state = ConversationState()

    authority = service._classify_pipeline_path(
        message="artikal 21",
        domain="civil",
        conversation_state=state,
        understanding_profile={"answer_intent": "authority_explainer", "all_components_direct": False},
    )
    explainer = service._classify_pipeline_path(
        message="fundamental rights",
        domain="constitutional",
        conversation_state=state,
        understanding_profile={"answer_intent": "explainer", "all_components_direct": False},
    )
    mixed = service._classify_pipeline_path(
        message="article 21 and fundamental rights",
        domain="constitutional",
        conversation_state=state,
        understanding_profile={"answer_intent": "mixed_direct", "all_components_direct": True},
    )

    assert authority == {"path": "fast", "reason": "authority_fast_path"}
    assert explainer == {"path": "medium", "reason": "constitutional_explainer"}
    assert mixed == {"path": "medium", "reason": "constitutional_mixed_direct"}


def test_fundamental_duties_query_uses_direct_constitutional_explainer_path(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "What are fundamental duties?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Fundamental Duties" in payload["answer"]
    assert "Fundamental Duties under the Constitution of India are constitutional expectations placed on citizens" in payload["answer"]
    assert "Summary:" not in payload["answer"]
    assert "Legal Position:" not in payload["answer"]
    assert "Practical Next Steps:" not in payload["answer"]
    assert "Documents:" not in payload["answer"]
    assert "Sources:" not in payload["answer"]
    assert "constitutional expectations placed on citizens" in payload["answer"].lower()
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_constitutional_explainer_uses_query_understanding_for_typo_and_detail_handling(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamantal rights with artcle numbers in short",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Fundamental Rights under the Constitution of India are" in answer
    assert "Articles 14-18" in answer
    assert "In simple terms" not in answer
    assert "In broad terms" not in answer
    assert "Summary:" not in answer
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_dpsp_query_uses_direct_constitutional_explainer_path(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a DPSP explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain DPSP",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Directive Principles of State Policy" in payload["answer"]
    assert "Summary:" not in payload["answer"]
    assert "Legal Position:" not in payload["answer"]
    assert "Sources:" not in payload["answer"]
    assert payload["citations"] == ["Constitution of India | Directive Principles of State Policy overview"]


def test_constitutional_explainer_uses_controlled_fuzzy_matching_for_common_legal_terms(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fuzzy-matched constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundmental roghts in short",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Fundamental Rights under the Constitution of India" in answer
    assert "Summary:" not in answer


def test_fundamental_rights_with_articles_gets_article_wise_breakdown(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamental rights along with articles",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Here is the article-wise breakdown" in answer
    assert "Articles 14-18" in answer
    assert "Articles 19-22" in answer
    assert "Article 32" in answer
    assert "In short:" not in answer
    assert "If you want the practical use later" not in answer
    assert "Summary:" not in answer
    assert "Sources:" not in answer


def test_fundamental_duties_with_articles_gets_complete_article_breakdown(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Tell me fundamental duties with article numbers",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Article 51A(a)" in answer
    assert "Article 51A(k)" in answer
    assert "Summary:" not in answer
    assert "Sources:" not in answer


def test_dpsp_with_articles_gets_article_wise_breakdown(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain DPSP along with articles",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Articles 36-37" in answer
    assert "Article 39A" in answer
    assert "Article 51" in answer
    assert "Summary:" not in answer
    assert "Sources:" not in answer


def test_constitutional_explainer_in_short_stays_concise_and_omits_extra_guidance(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamental rights in short",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Fundamental Rights" in answer
    assert "article-wise breakdown" not in answer.lower()
    assert "If you want the practical use later" not in answer
    assert "The exact remedy depends" not in answer
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_constitutional_explainer_step_by_step_uses_numbered_breakdown(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamental rights step by step",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Here it is step by step:" in answer
    assert "1. Articles 12-13" in answer
    assert "2. Articles 14-18" in answer
    assert "likely forum" not in answer.lower()
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_constitutional_explainer_practical_query_keeps_forum_and_documents_when_needed(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "What are fundamental rights and what remedy or court is used if there is a violation?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["likely_forum"] == "Constitution of India"
    assert payload["documents_to_keep"]


def test_mixed_constitutional_query_combines_article_and_explainer_without_grounded_retrieval(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a mixed simple constitutional query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 + fundamental rights",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Article 21 of the Constitution of India" in answer
    assert "Fundamental Rights under the Constitution of India" in answer
    assert "life and personal liberty" in answer.lower()
    assert "Summary:" not in answer
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []
    assert "Constitution of India | Article 21" in payload["citations"]
    assert "Constitution of India | Fundamental Rights overview" in payload["citations"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_direct_route_uses_understanding_profile_for_mixed_constitutional_query(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a simple mixed direct constitutional query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 and fundamental rights",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21 of the Constitution of India" in payload["answer"]
    assert "Fundamental Rights under the Constitution of India" in payload["answer"]
    assert "Constitution of India | Article 21" in payload["citations"]
    assert "Constitution of India | Fundamental Rights overview" in payload["citations"]


def test_mixed_constitutional_authority_query_combines_simple_articles_directly(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a mixed simple constitutional authority query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 and Article 19",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Article 21 of the Constitution of India" in answer
    assert "Article 19 of the Constitution of India" in answer
    assert "Constitution of India | Article 21" in payload["citations"]
    assert "Constitution of India | Article 19" in payload["citations"]
    assert payload["likely_forum"] is None
    assert payload["documents_to_keep"] == []


def test_grouped_article_authority_query_combines_multiple_articles_directly(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a grouped mixed article query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 19 and 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Article 19 of the Constitution of India" in answer
    assert "Article 21 of the Constitution of India" in answer
    assert "Constitution of India | Article 19" in payload["citations"]
    assert "Constitution of India | Article 21" in payload["citations"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_grouped_section_authority_query_combines_multiple_sections_directly(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a grouped mixed section query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Section 420 and 406 IPC",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert "Section 420 of the Indian Penal Code, 1860" in answer
    assert "Section 406 of the Indian Penal Code, 1860" in answer
    assert "Indian Penal Code, 1860 | Section 420" in payload["citations"]
    assert "Indian Penal Code, 1860 | Section 406" in payload["citations"]
    assert "Constitution of India" not in payload["authorities"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"
    assert assistant_message["metadata"]["retrieval"]["source"] == "authority_mixed_direct"


def test_fast_authority_response_uses_understanding_driven_format_metadata(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fast authority query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "artikal 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21 of the Constitution of India" in payload["answer"]
    assert "Summary:" not in payload["answer"]
    assert "Practical Next Steps:" not in payload["answer"]
    assert "Documents:" not in payload["answer"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["direct_answer_format"]["family"] == "authority"
    assert assistant_message["metadata"]["retrieval"]["direct_answer_format"]["layout"] == "definition"


def test_mixed_direct_response_uses_understanding_driven_format_metadata(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a mixed direct constitutional query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 and fundamental rights with article numbers",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Article 21 of the Constitution of India" in payload["answer"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    direct_format = assistant_message["metadata"]["retrieval"]["direct_answer_format"]
    assert direct_format["family"] == "mixed"
    assert direct_format["authority_layout"] == "definition"
    assert direct_format["explainer_layout"] == "article_breakdown"


def test_direct_answer_formatting_stays_conversational_for_plain_explainer_queries():
    answer = ChatService._format_constitutional_explainer_answer(
        title="Fundamental Rights under the Constitution of India",
        summary="the basic freedoms and protections guaranteed against State action.",
        legal_position="They are enforceable, and courts can protect them through constitutional remedies.",
        next_steps="",
        caution="",
        article_breakdown=[],
        format_profile={"layout": "conversational", "detail_instructions": {}},
    )

    assert answer.startswith("Fundamental Rights under the Constitution of India are the basic freedoms")
    assert "Practical Next Steps:" not in answer
    assert "Documents:" not in answer


def test_direct_answer_formatting_stays_definition_style_for_fast_authority_queries():
    answer = ChatService._format_fast_authority_answer(
        title="Article 21 of the Constitution of India",
        summary="protects life and personal liberty and requires a fair, just, and reasonable legal process.",
        legal_position="In broad terms, courts interpret it broadly to cover dignity, privacy, livelihood, and related protections.",
        format_profile={"layout": "definition"},
    )

    assert answer.startswith("Article 21 of the Constitution of India protects life and personal liberty")
    assert "In practice, courts interpret it broadly" in answer
    assert "In broad terms" not in answer
    assert "Practical Next Steps:" not in answer
    assert "Documents:" not in answer


def test_direct_answer_formatting_uses_pointwise_layout_for_explainer_queries():
    answer = ChatService._format_constitutional_explainer_answer(
        title="Fundamental Rights under the Constitution of India",
        summary="the basic freedoms and protections guaranteed against State action.",
        legal_position="In simple terms, they are enforceable, and courts can protect them through constitutional remedies.",
        next_steps="Use constitutional remedies if there is a violation.",
        caution="",
        article_breakdown=[
            "Articles 14-18: Right to Equality.",
            "Articles 19-22: Right to Freedom.",
        ],
        format_profile={"layout": "points", "detail_instructions": {"in_points": True, "practical_context": False}},
    )

    assert "Here are the key points:" in answer
    assert "- Fundamental Rights under the Constitution of India: the basic freedoms and protections guaranteed against State action." in answer
    assert "- Articles 14-18: Right to Equality." in answer
    assert "- Legal position: they are enforceable, and courts can protect them through constitutional remedies." in answer
    assert "In simple terms" not in answer


def test_direct_answer_formatting_uses_pointwise_layout_for_authority_queries():
    answer = ChatService._format_fast_authority_answer(
        title="Article 21 of the Constitution of India",
        summary="protects life and personal liberty and requires a fair, just, and reasonable legal process.",
        legal_position="In broad terms, courts interpret it broadly to cover dignity, privacy, livelihood, and related protections.",
        format_profile={"authority_layout": "points", "detail_instructions": {"in_points": True}},
    )

    assert answer.startswith("Article 21 of the Constitution of India")
    assert "- Provision: protects life and personal liberty and requires a fair, just, and reasonable legal process." in answer
    assert "- Legal position: courts interpret it broadly to cover dignity, privacy, livelihood, and related protections." in answer
    assert "In broad terms" not in answer


def test_fast_authority_formatting_uses_concise_layout_when_requested():
    answer = ChatService._format_fast_authority_answer(
        title="Article 21 of the Constitution of India",
        summary="protects life and personal liberty and requires a fair, just, and reasonable legal process.",
        legal_position="Courts interpret it broadly to cover dignity, privacy, livelihood, and related protections.",
        format_profile={"authority_layout": "concise", "detail_instructions": {"concise": True}},
    )

    assert answer.startswith("Article 21 of the Constitution of India protects life and personal liberty")
    assert "\n\n" not in answer
    assert "In practice" not in answer


def test_constitutional_explainer_in_points_uses_bulleted_direct_format(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a constitutional explainer query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamental rights in points",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "Here are the key points:" in answer
    assert "- Articles 14-18: Right to Equality" in answer
    assert "enforceable" in answer.lower()
    assert "Summary:" not in answer


def test_fast_authority_query_prefers_curated_google_citations_when_available(client, monkeypatch):
    client.app.state.chat_service._direct_answer_cache.clear()

    def fake_search(self, *, query: str, max_results: int | None = None, site_restrict: str | None = None, trusted_only: bool = True):
        assert trusted_only is True
        return GoogleSearchResult(
            documents=[
                {
                    "doc_id": "google:indiacode",
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
            ],
            from_cache=False,
            trusted_result_count=1,
        )

    monkeypatch.setattr(GoogleCustomSearchService, "search", fake_search)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == ["India Code - Constitution of India"]
    messages = client.get(f"/chat/{payload['chat_id']}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["curated_google_used"] is True
    assert assistant_message["metadata"]["route_classification"]["path"] == "fast"


def test_constitutional_explainer_falls_back_to_local_direct_citations_when_curated_google_is_weak(client, monkeypatch):
    client.app.state.chat_service._direct_answer_cache.clear()

    def fake_search(self, *, query: str, max_results: int | None = None, site_restrict: str | None = None, trusted_only: bool = True):
        return GoogleSearchResult(
            documents=[
                {
                    "doc_id": "google:weak",
                    "title": "Unofficial explainer",
                    "headline": "Weak explainer.",
                    "fragment_headline": "Weak",
                    "doc_excerpt": "Weak explainer.",
                    "docsource": "google:example.org.in",
                    "citations": ["Unofficial explainer"],
                    "publishdate": "",
                    "url": "https://example.org.in/",
                    "score": 9.0,
                    "source_kind": "google_custom_search",
                    "authority_type": "web_reference",
                }
            ],
            from_cache=False,
            trusted_result_count=0,
        )

    monkeypatch.setattr(GoogleCustomSearchService, "search", fake_search)

    response = client.post(
        "/chat",
        json={
            "message": "Explain fundamental rights",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == ["Constitution of India | Fundamental Rights overview"]
    messages = client.get(f"/chat/{payload['chat_id']}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["curated_google_used"] is False
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_fast_authority_query_uses_lightweight_direct_cache_on_repeat_requests(client, monkeypatch):
    service = client.app.state.chat_service
    service._direct_answer_cache.clear()
    original = ChatService._format_fast_authority_answer
    calls = {"count": 0}

    def wrapped(*, title, summary, legal_position, format_profile=None):
        calls["count"] += 1
        return original(
            title=title,
            summary=summary,
            legal_position=legal_position,
            format_profile=format_profile,
        )

    monkeypatch.setattr(ChatService, "_format_fast_authority_answer", staticmethod(wrapped))

    first = client.post("/chat", json={"message": "Article 21", "state": "Gujarat"})
    second = client.post("/chat", json={"message": "Article 21", "state": "Gujarat"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1

    first_messages = client.get(f"/chat/{first.json()['chat_id']}/messages").json()["items"]
    second_messages = client.get(f"/chat/{second.json()['chat_id']}/messages").json()["items"]
    first_assistant = [item for item in first_messages if item["role"] == "assistant"][-1]
    second_assistant = [item for item in second_messages if item["role"] == "assistant"][-1]
    assert first_assistant["metadata"]["retrieval"]["cache_hit"] is False
    assert second_assistant["metadata"]["retrieval"]["cache_hit"] is True


def test_constitutional_explainer_query_uses_lightweight_direct_cache_on_repeat_requests(client, monkeypatch):
    service = client.app.state.chat_service
    service._direct_answer_cache.clear()
    original = ChatService._format_constitutional_explainer_answer
    calls = {"count": 0}

    def wrapped(*, title, summary, legal_position, next_steps, caution, article_breakdown, format_profile=None):
        calls["count"] += 1
        return original(
            title=title,
            summary=summary,
            legal_position=legal_position,
            next_steps=next_steps,
            caution=caution,
            article_breakdown=article_breakdown,
            format_profile=format_profile,
        )

    monkeypatch.setattr(ChatService, "_format_constitutional_explainer_answer", staticmethod(wrapped))

    first = client.post("/chat", json={"message": "Explain fundamental rights", "state": "Gujarat"})
    second = client.post("/chat", json={"message": "Explain fundamental rights", "state": "Gujarat"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1

    first_messages = client.get(f"/chat/{first.json()['chat_id']}/messages").json()["items"]
    second_messages = client.get(f"/chat/{second.json()['chat_id']}/messages").json()["items"]
    first_assistant = [item for item in first_messages if item["role"] == "assistant"][-1]
    second_assistant = [item for item in second_messages if item["role"] == "assistant"][-1]
    assert first_assistant["metadata"]["retrieval"]["cache_hit"] is False
    assert second_assistant["metadata"]["retrieval"]["cache_hit"] is True


def test_grounded_queries_do_not_populate_direct_answer_cache(client):
    service = client.app.state.chat_service
    service._direct_answer_cache.clear()

    response = client.post(
        "/chat",
        json={
            "message": "section 420 ipc punishment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    assert service._direct_answer_cache == {}


def test_low_confidence_constitutional_direct_query_gets_human_clarification_instead_of_grounded_fallback(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a low-confidence direct constitutional clarification")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "constitutional rights",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "I can help with that. I just need one quick clarification first." in payload["answer"]
    assert payload["follow_up_question"] == "Do you want Fundamental Rights, Fundamental Duties, or DPSP?"
    assert "No highly relevant India Kanoon authority was found" not in payload["answer"]
    chat_id = payload["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["source"] == "direct_answer_clarification"
    assert assistant_message["metadata"]["route_classification"]["path"] == "medium"


def test_fast_authority_in_points_uses_bulleted_direct_format(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fast authority query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 in points",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer.startswith("Article 21 of the Constitution of India")
    assert "\n- Provision:" in answer
    assert "\n- Legal position:" in answer
    assert "life and personal liberty" in answer.lower()
    assert "lawful, fair, and non-arbitrary procedure" in answer.lower()


def test_fast_authority_in_short_uses_concise_direct_format(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fast authority query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 in short",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert answer.startswith("Article 21 of the Constitution of India protects life and personal liberty")
    assert "\n\n" not in answer
    messages = client.get(f"/chat/{payload['chat_id']}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["direct_answer_format"]["authority_layout"] == "concise"


def test_fast_authority_step_by_step_uses_numbered_direct_format(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fast authority query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Article 21 step by step",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"]
    assert answer.startswith("Article 21 of the Constitution of India")
    assert "\n1. " in answer
    messages = client.get(f"/chat/{payload['chat_id']}/messages").json()["items"]
    assistant_message = [item for item in messages if item["role"] == "assistant"][-1]
    assert assistant_message["metadata"]["retrieval"]["direct_answer_format"]["authority_layout"] == "step_by_step"


def test_fast_authority_lookup_uses_controlled_fuzzy_matching_for_statute_alias_without_touching_identifier(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a fuzzy-matched direct Section lookup")

    def fail_if_llm_called(self, user_prompt, conversation):
        raise AssertionError("LLM should not run for a fuzzy-matched direct Section lookup")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)
    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fail_if_llm_called)

    response = client.post(
        "/chat",
        json={
            "message": "section 498a ipcc",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Section 498A of the Indian Penal Code, 1860" in payload["answer"]
    assert "Summary:" not in payload["answer"]


def test_low_confidence_authority_direct_query_gets_human_clarification_instead_of_grounded_fallback(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for a low-confidence direct authority clarification")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "article in constitution",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "I can help with that, but I need one small clarification first." in payload["answer"]
    assert payload["follow_up_question"] == "Which Constitution article do you want explained?"
    assert "No highly relevant India Kanoon authority was found" not in payload["answer"]


def test_food_safety_issue_gets_consumer_style_guidance_not_theft_template(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "My chocolate has insect issue",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "I understand" in answer
    assert (
        "the first step i suggest" in answer.lower()
        or "what i suggest first" in answer.lower()
        or "what to do next" in answer.lower()
        or "practical next step" in answer.lower()
    )
    assert "product" in answer.lower() or "packet" in answer.lower() or "invoice" in answer.lower()
    assert "food safety authority" in answer.lower() or "consumer forum" in answer.lower() or "seller" in answer.lower()
    assert "food" in answer.lower() or "seller" in answer.lower()
    assert "imei" not in answer.lower()
    assert "sim" not in answer.lower()
    assert "No highly relevant India Kanoon authority was found" not in answer


def test_mobile_snatching_gets_theft_guidance_not_food_safety_template(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct snatching guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "I am sorry this happened" in answer or "I understand" in answer
    assert "practical next step" in answer.lower() or "next step" in answer.lower() or "what i suggest first" in answer.lower()
    assert "police station" in answer.lower()
    assert "snatching" in answer.lower() or "incident" in answer.lower() or "police" in answer.lower()
    assert "food safety" not in answer.lower()
    assert "packet" not in answer.lower()


def test_fir_refusal_gets_escalation_guidance_not_generic_police_template(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct FIR refusal guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "Police refused to file FIR for my complaint",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "I understand" in answer
    assert "what to do next" in answer.lower() or "next step" in answer.lower() or "keep the momentum going" in answer.lower()
    assert "escalation" in answer.lower() or "carry forward" in answer.lower() or "police station" in answer.lower()
    assert "police station" in answer.lower() or "what was the underlying incident" in answer.lower() or "refusal" in answer.lower()
    assert "food safety" not in answer.lower()
    assert "consumer commission" not in answer.lower()


def test_landlord_harassment_gets_tenancy_guidance_not_theft_template(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct landlord harassment guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "My landlord is harassing me and threatening eviction",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "recommended actions" in answer.lower() or "document everything" in answer.lower() or "tenancy records" in answer.lower()
    assert "landlord" in answer.lower() or "tenancy" in answer.lower() or "rent" in answer.lower()
    assert "imei" not in answer.lower()
    assert "packet" not in answer.lower()


def test_cyber_fraud_interview_uses_bank_and_platform_facts_in_final_answer(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account on PhonePe",
            "state": "Gujarat",
        },
    )
    chat_id = created.json()["chat_id"]

    response = client.post("/chat", json={"chat_id": chat_id, "message": "Today morning, one debit happened", "state": "Gujarat"})
    response = client.post("/chat", json={"chat_id": chat_id, "message": "I have not yet reported it to SBI or 1930", "state": "Gujarat"})
    response = client.post("/chat", json={"chat_id": chat_id, "message": "Rs 15000 and I have the transaction ID", "state": "Gujarat"})

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "sbi" in answer.lower() or "phonepe" in answer.lower()
    assert "contact upi" not in answer.lower()
    assert "your update makes one thing clearer" in answer.lower() or "that helps" in answer.lower() or "i understand" in answer.lower()
    assert "within 24 hours" in answer.lower()
    assert "bns section 318" in answer.lower() or "bns section 319" in answer.lower()
    assert "bnss section 173" in answer.lower()
    assert "1930" in answer or "cyber crime" in answer.lower()
    assert "immediate next steps:" in answer.lower() or "what i suggest first" in answer.lower()
    assert "consumer commission" not in answer.lower()
    assert "rent agreement" not in answer.lower()


def test_cyber_fraud_completed_guidance_skips_repeating_reporting_instruction(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account on PhonePe",
            "state": "Gujarat",
        },
    )
    chat_id = created.json()["chat_id"]

    client.post("/chat", json={"chat_id": chat_id, "message": "Today morning, one debit happened", "state": "Gujarat"})
    client.post(
        "/chat",
        json={"chat_id": chat_id, "message": "I already reported it to SBI and 1930 and I have the complaint number", "state": "Gujarat"},
    )
    response = client.post("/chat", json={"chat_id": chat_id, "message": "Rs 15000 and I have the transaction ID", "state": "Gujarat"})

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert "already started the reporting process" in answer or "focus now is on strengthening the record" in answer
    assert "notify the bank or payment app immediately" not in answer
    assert "report the same transaction trail through 1930" not in answer
    assert "transaction id" in answer


def test_cyber_fraud_first_turn_uses_brief_empathy_and_24_hour_urgency(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert "i know this is stressful" in answer or "i understand" in answer
    assert "immediate steps" in answer or "first step" in answer or "next practical step" in answer
    assert "within 24 hours" in answer
    assert "recovery" in answer
    assert payload["follow_up_question"]


def test_cyber_fraud_follow_up_acknowledges_latest_progress_and_varies_opening(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200
    chat_id = created.json()["chat_id"]
    first_answer = created.json()["answer"]

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "I already reported it to SBI and 1930",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"]
    lowered = answer.lower()
    assert answer != first_answer
    assert "reported" in lowered or "reporting has already started" in lowered
    assert "i know this is stressful" not in lowered
    assert "immediate action:" not in lowered
    assert "most urgent right now:" not in lowered
    assert "reported timing details" not in lowered
    assert "contact sbi immediately" in lowered or "reporting has already started" in lowered or "reported" in lowered
    assert "when did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_cyber_fraud_timing_follow_up_advances_workflow_without_repeating_initial_action(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Today morning, it was one debit",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "contact sbi immediately" not in answer
    assert "report the matter through 1930" not in answer
    assert "exact debit time" in answer or "one transaction or multiple debits" in answer or "complaint record stays consistent" in answer
    assert "have you already reported" in (follow_up.json()["follow_up_question"] or "").lower()


def test_legal_help_interview_keeps_second_line_context_only():
    service = ChatService.__new__(ChatService)
    blocked_words = {
        "contact",
        "report",
        "stop",
        "keep",
        "call",
        "file",
        "submit",
        "block",
        "secure",
        "send",
        "visit",
        "write",
        "check",
        "ask",
        "carry",
        "prepare",
        "collect",
        "preserve",
        "gather",
        "confirm",
    }
    scenarios = [
        (
            "cyber_fraud",
            ConversationState(conversation_started=False, issue_type="cyber_fraud", bank_name="SBI"),
            "I clicked a fake UPI link and money got debited from my SBI account",
            {"bank_platform": "SBI"},
        ),
        (
            "notice",
            ConversationState(conversation_started=False, issue_type="notice"),
            "I received a legal notice for payment default",
            {},
        ),
        (
            "snatching_theft",
            ConversationState(conversation_started=False, issue_type="snatching_theft"),
            "My mobile was snatched on the road",
            {},
        ),
    ]

    for issue_type, state, latest_message, facts in scenarios:
        answer = service._build_virtual_advocate_question(
            state=state,
            issue_type=issue_type,
            latest_message=latest_message,
            question="unused",
            opening=service._acknowledgement_for_issue(issue_type),
            is_first_question=True,
            collected_facts=facts,
            allow_section_mentions=False,
        )
        parts = answer.split("\n\n")
        assert len(parts) >= 2
        first_line = parts[0].lower()
        second_line = parts[1].lower()
        assert ":" in first_line
        assert not any(f" {word} " in f" {second_line} " for word in blocked_words), second_line


def test_theft_evidence_follow_up_advances_without_repeating_initial_phone_lock_step(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct theft-guidance follow-ups")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200
    chat_id = created.json()["chat_id"]

    response = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "I have the IMEI and invoice",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert "block the sim" not in answer
    assert "secure your email and banking apps" not in answer
    assert "imei" in answer
    assert "invoice" in answer
    assert "complaint record" in answer or "proof ready" in answer


def test_follow_up_fir_vs_complaint_question_gets_direct_short_answer_in_theft_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct theft-guidance follow-ups")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Do I need to file FIR or a normal complaint?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "fir" in answer
    assert "written complaint" in answer or "complaint" in answer
    assert "block the sim" not in answer
    assert "secure your email and banking apps" not in answer
    assert "nearest police station" not in answer
    assert "when and where did the snatching or theft happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_fir_vs_complaint_question_gets_direct_short_answer_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Do I need to file FIR or normal complaint?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "fir" in answer
    assert "written complaint" in answer or "complaint" in answer
    assert "contact your bank or payment app immediately" not in answer
    assert "report the matter through 1930" not in answer
    assert "when did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_police_vs_bank_question_gets_direct_answer_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I go to police or bank first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "bank" in answer
    assert "police" in answer or "cyber" in answer
    assert "contact your bank or payment app immediately" not in answer
    assert "report the matter through 1930" not in answer
    assert "when did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_police_vs_bank_question_gets_direct_answer_in_theft_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct theft-guidance follow-ups")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I go to police or bank first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "police comes first" in answer or "police" in answer
    assert "bank" in answer or "wallet" in answer
    assert "block the sim" not in answer
    assert "secure your email and banking apps" not in answer
    assert "when and where did the snatching or theft happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_turn_classifier_marks_decision_questions():
    service = ChatService.__new__(ChatService)

    fraud_state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud")
    theft_state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="snatching_theft")

    fraud_result = service._classify_legal_help_follow_up_turn(
        message="Should I go to police or bank first?",
        state=fraud_state,
        collected_facts={},
        is_continuation=True,
    )
    theft_result = service._classify_legal_help_follow_up_turn(
        message="Do I need to file FIR or a normal complaint?",
        state=theft_state,
        collected_facts={},
        is_continuation=True,
    )
    fact_result = service._classify_legal_help_follow_up_turn(
        message="It happened today morning.",
        state=fraud_state,
        collected_facts={},
        is_continuation=True,
    )

    assert fraud_result["mode"] == "decision_question"
    assert fraud_result["decision_type"] == "police_vs_bank"
    assert theft_result["mode"] == "decision_question"
    assert theft_result["decision_type"] == "fir_vs_complaint"
    assert fact_result["mode"] == "fact_update"
    assert fact_result["decision_type"] is None


def test_merge_case_details_into_state_preserves_prior_values_and_adds_upload_summaries():
    service = ChatService.__new__(ChatService)
    service.settings = type("SettingsStub", (), {"default_state": "Gujarat"})()
    merged = service._merge_case_details_into_state(
        current_state=ConversationState(
            conversation_started=True,
            case_state="Delhi",
            district="South Delhi",
            case_stage="notice",
            is_own_matter=True,
            uploaded_document_summaries=["Existing notice copy"],
        ),
        request=type(
            "RequestStub",
            (),
            {
                "state": "",
                "district": "Ahmedabad",
                "case_stage": "",
                "is_own_matter": None,
            },
        )(),
        fallback_state="Maharashtra",
        uploaded_texts=["Bank statement showing UPI debit and transaction reference 12345 on 14 April 2026."],
    )

    assert merged.case_state == "Delhi"
    assert merged.district == "Ahmedabad"
    assert merged.case_stage == "notice"
    assert merged.is_own_matter is True
    assert merged.uploaded_document_summaries[0] == "Existing notice copy"
    assert "Bank statement showing UPI debit" in merged.uploaded_document_summaries[1]


def test_chat_persists_case_details_in_conversation_state_metadata(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("Grounded retrieval should not run for this direct notice guidance query")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat/upload",
        data={
            "message": "I received a legal notice for payment default",
            "state": "Maharashtra",
            "district": "Pune",
            "case_stage": "notice",
            "is_own_matter": "true",
        },
        files={
            "files": ("notice.txt", io.BytesIO(b"Legal notice demanding payment within 7 days"), "text/plain"),
        },
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    messages = client.get(f"/chat/{chat_id}/messages").json()["items"]
    latest_assistant = [item for item in messages if item["role"] == "assistant"][-1]
    conversation_state = latest_assistant["metadata"]["conversation_state"]
    assert conversation_state["case_state"] == "Maharashtra"
    assert conversation_state["district"] == "Pune"
    assert conversation_state["case_stage"] == "notice"
    assert conversation_state["is_own_matter"] is True
    assert conversation_state["uploaded_document_summaries"]
    assert "Legal notice demanding payment" in conversation_state["uploaded_document_summaries"][0]


def test_follow_up_decision_normalizer_stays_issue_sensitive():
    service = ChatService.__new__(ChatService)

    notice_state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="notice")
    fraud_state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud")

    assert service._normalize_follow_up_decision_type(
        message="Should I go to police or bank first?",
        state=notice_state,
    ) is None
    normalized = service._normalize_follow_up_decision_type(
        message="Should I go to police or bank first?",
        state=fraud_state,
    )
    assert normalized is not None
    assert normalized["decision_type"] == "police_vs_bank"


def test_follow_up_fact_answer_gate_blocks_decision_questions():
    service = ChatService.__new__(ChatService)

    assert service._should_treat_message_as_fact_answer(
        turn_classification={"mode": "decision_question", "decision_type": "police_vs_bank"},
        current_intake_key="incident_timing",
        is_continuation=True,
    ) is False
    assert service._should_treat_message_as_fact_answer(
        turn_classification={"mode": "fact_update", "decision_type": None},
        current_intake_key="incident_timing",
        is_continuation=True,
    ) is True


def test_follow_up_decision_builder_returns_two_line_answer_and_preserves_next_question():
    service = ChatService.__new__(ChatService)
    state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud", bank_name="SBI")

    result = service._build_follow_up_decision_result(
        turn_classification={
            "mode": "decision_question",
            "decision_type": "police_vs_bank",
            "raw_question": "Should I go to police or bank first?",
            "confidence": 0.95,
        },
        state=state,
        collected_facts={},
        next_item={"index": 1, "key": "incident_timing", "question": "When did this happen, and was it one transaction or multiple debits?"},
        domain="civil",
        warnings=[],
    )

    answer_lines = result.answer.splitlines()
    assert len(answer_lines) == 2
    assert answer_lines[0] == "Contact the bank or payment app first to try to stop further debit."
    assert answer_lines[1] == "The police or cyber complaint should follow quickly so the transaction is formally recorded."
    assert result.follow_up_question == "For SBI, when exactly did this happen, and was it one transaction or multiple debits?"
    assert result.raw_json["source"] == "legal_help_decision_answer"
    assert result.raw_json["decision_type"] == "police_vs_bank"
    assert result.raw_json["resumed_follow_up_question"] == "For SBI, when exactly did this happen, and was it one transaction or multiple debits?"


def test_follow_up_decision_answer_polish_is_direct_and_confident():
    service = ChatService.__new__(ChatService)
    answer = service._answer_follow_up_decision_question(
        decision_type="reply_vs_documents",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="notice"),
        collected_facts={},
    )

    answer_lines = answer.splitlines()
    assert len(answer_lines) == 2
    assert answer_lines[0] == "Organize the documents and timeline first."
    assert answer_lines[1] == "A reply is safer once the record is clear, unless the deadline is too close to wait."
    assert "short answer" not in answer.lower()


def test_resume_follow_up_after_decision_answer_contextualizes_pending_question():
    service = ChatService.__new__(ChatService)
    state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud", bank_name="SBI")

    resumed = service._resume_follow_up_after_decision_answer(
        state=state,
        next_item={"index": 1, "key": "incident_timing", "question": "When did this happen, and was it one transaction or multiple debits?"},
        collected_facts={"bank_platform": "SBI"},
    )

    assert resumed == "For SBI, when exactly did this happen, and was it one transaction or multiple debits?"


def test_resume_follow_up_after_decision_answer_returns_none_without_pending_question():
    service = ChatService.__new__(ChatService)
    state = ConversationState(conversation_started=True, active_intent="legal_help", issue_type="notice")

    assert service._resume_follow_up_after_decision_answer(
        state=state,
        next_item=None,
        collected_facts={},
    ) is None


def test_follow_up_decision_normalizer_expands_to_more_question_forms():
    service = ChatService.__new__(ChatService)

    assert service._normalize_follow_up_decision_type(
        message="Should I contact bank or app first?",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud"),
    )["decision_type"] == "bank_vs_platform"
    assert service._normalize_follow_up_decision_type(
        message="Should I go to cyber cell or police first?",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud"),
    )["decision_type"] == "police_vs_cyber_channel"
    assert service._normalize_follow_up_decision_type(
        message="Do I need to report to the police too?",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="cyber_fraud"),
    )["decision_type"] == "police_vs_cyber_channel"
    assert service._normalize_follow_up_decision_type(
        message="Should I approach seller or consumer forum first?",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="consumer"),
    )["decision_type"] == "seller_vs_consumer_forum"
    assert service._normalize_follow_up_decision_type(
        message="Should I reply first or collect documents?",
        state=ConversationState(conversation_started=True, active_intent="legal_help", issue_type="notice"),
    )["decision_type"] == "reply_vs_documents"


def test_follow_up_bank_vs_platform_question_gets_direct_answer_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I contact bank or app first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"]
    answer_lines = answer.splitlines()
    assert len(answer_lines) >= 2
    assert answer_lines[0] == "Contact the bank first if money has moved or the account is at risk."
    assert answer_lines[1] == "Inform the app or platform as well so the complaint trail stays aligned."
    answer = answer.lower()
    assert "bank" in answer
    assert "app" in answer or "platform" in answer
    assert "contact your bank or payment app immediately" not in answer
    assert "when did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_police_vs_cyber_channel_question_gets_direct_answer_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I go to cyber cell or local police first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "cyber" in answer
    assert "police" in answer
    assert "immediately" in answer or "early fraud record" in answer
    assert "contact your bank or payment app immediately" not in answer
    assert "when did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_police_too_question_stays_direct_after_progress_update_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my SBI account",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    progress = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "I already reported it to SBI and 1930",
            "state": "Gujarat",
        },
    )
    assert progress.status_code == 200
    assert "when did this happen" in (progress.json()["follow_up_question"] or "").lower()

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Do I need to report to the police too?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "cyber reporting channel first" in answer
    assert "local police" in answer
    assert "contact sbi immediately" not in answer
    assert "immediate action:" not in answer
    assert "most urgent right now:" not in answer
    assert "next practical step" not in answer
    assert "when exactly did this happen" in (follow_up.json()["follow_up_question"] or "").lower()


def test_follow_up_seller_vs_consumer_forum_question_gets_direct_answer(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct consumer guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "My product is defective and seller is not responding",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I approach seller or consumer forum first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "seller" in answer
    assert "consumer forum" in answer or "consumer commission" in answer
    assert "written complaint to the seller" not in answer


def test_follow_up_seller_vs_consumer_forum_question_gets_direct_answer_in_food_safety_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I approach seller or consumer forum first?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "seller" in answer
    assert "consumer forum" in answer or "consumer commission" in answer
    assert "do not throw away the packet" not in answer


def test_follow_up_reply_vs_documents_question_gets_direct_answer_in_notice_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct notice guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I received a legal notice for payment default",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": created.json()["chat_id"],
            "message": "Should I reply first or collect documents?",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "reply" in answer
    assert "documents" in answer or "record" in answer
    assert "check the notice deadline first" not in answer
    assert follow_up.json()["follow_up_question"] is None or "what is the main demand" in (follow_up.json()["follow_up_question"] or "").lower() or "did you receive a notice" in (follow_up.json()["follow_up_question"] or "").lower()


def test_decision_follow_up_does_not_fill_pending_intake_slot_in_fraud_flow(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct cyber-fraud guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200
    chat_id = created.json()["chat_id"]

    decision = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "Should I go to police or bank first?",
            "state": "Gujarat",
        },
    )
    assert decision.status_code == 200
    assert "when did this happen" in (decision.json()["follow_up_question"] or "").lower()

    timing = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "Today morning, it was one debit",
            "state": "Gujarat",
        },
    )
    assert timing.status_code == 200
    assert "have you already reported" in (timing.json()["follow_up_question"] or "").lower()


def test_notice_follow_up_uses_new_fact_specific_opening_instead_of_reusing_same_intro(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct notice guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "I got a legal notice for payment default",
            "state": "Gujarat",
        },
    )
    assert created.status_code == 200
    chat_id = created.json()["chat_id"]

    follow_up = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "I received it yesterday and the deadline is three days away",
            "state": "Gujarat",
        },
    )

    assert follow_up.status_code == 200
    answer = follow_up.json()["answer"].lower()
    assert "timeline is clearer now" in answer or "notice stage and timing are clearer now" in answer
    assert "late or rushed reply can weaken your position" in answer or "response plan:" in answer or "deadline" in answer
    assert "i understand the concern" not in answer
    assert follow_up.json()["follow_up_question"] is None


def test_food_safety_completed_guidance_acknowledges_existing_seller_complaint_and_moves_forward(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet and I already complained to the seller by email. I still have the packet, invoice, and photos.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert payload["follow_up_question"] is None
    assert "already complained to the seller side" in answer or "earlier seller complaint sent by email" in answer
    assert "send a written complaint to the seller" not in answer
    assert "prepare the record for food-safety or consumer escalation" in answer


def test_notice_completed_guidance_does_not_inject_raw_user_sentence_into_paragraph(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct notice guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "I received a legal notice for payment default.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert "i received a legal notice for payment default." not in answer
    assert "notice stage" in answer
    assert "payment demand and deadline" in answer or "stated demand and deadline" in answer


def test_notice_completed_guidance_breaks_into_shorter_blocks(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct notice guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "I received a legal notice for payment default.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    blocks = [part.strip() for part in answer.split("\n\n") if part.strip()]
    assert len(blocks) >= 3
    assert "reply window is important" in blocks[0].lower()
    assert "this is a notice-stage matter" in blocks[1].lower()
    assert "criminal-law sections are usually not the main issue here" in blocks[2].lower()


def test_food_completed_guidance_uses_fact_summary_not_raw_repeated_input(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    message = "There is an insect in my sealed chocolate packet and I already complained to the seller by email. I still have the packet, invoice, and photos."
    response = client.post(
        "/chat",
        json={
            "message": message,
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert message.lower() not in answer
    assert "the sealed chocolate packet with the invoice and photographs" in answer
    assert "the earlier seller complaint sent by email" in answer


def test_completed_guidance_uses_issue_specific_heading_not_same_structure_every_time(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct legal-help guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    notice = client.post(
        "/chat",
        json={
            "message": "I received a legal notice for payment default.",
            "state": "Gujarat",
        },
    )
    food = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet and I still have the packet, invoice, and photos.",
            "state": "Gujarat",
        },
    )

    assert notice.status_code == 200
    assert food.status_code == 200
    assert "response plan:" in notice.json()["answer"].lower()
    assert "what to do next:" in food.json()["answer"].lower()
    assert "next steps:" not in notice.json()["answer"].lower()
    assert "next steps:" not in food.json()["answer"].lower()


def test_notice_intake_uses_clear_empathy_and_deadline_urgency(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct notice guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "I received a legal notice for payment default.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert "i understand the concern" in answer
    assert "safest course is to focus on the deadline" in answer or "reply window is important" in answer
    assert "deadline" in answer
    assert payload["follow_up_question"] is None or "notice" in (payload["follow_up_question"] or "").lower()


def test_food_safety_intake_uses_clear_empathy_and_evidence_urgency(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    answer = payload["answer"].lower()
    assert "i understand your concern" in answer
    assert "preserve the product and proof properly" in answer or "packet, batch details, and photographs" in answer
    assert "packet" in answer
    assert "photographs" in answer or "photos" in answer
    assert payload["follow_up_question"] is None


def test_food_issue_with_enough_initial_details_goes_directly_to_guidance(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct food-safety guidance queries")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet and I still have the packet, bill, and photos.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["follow_up_question"] is None
    assert "seller" in payload["answer"].lower() or "food safety authority" in payload["answer"].lower()
    assert "what to do next:" in payload["answer"].lower() or "response plan:" in payload["answer"].lower()


def test_theft_follow_up_uses_city_detail_to_make_guidance_more_specific(client, monkeypatch):
    def fail_if_called(self, query_variants, doctypes_options, max_results=4):
        raise AssertionError("India Kanoon should not be called for direct theft-guidance follow-ups")

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fail_if_called)

    created = client.post(
        "/chat",
        json={
            "message": "My mobile was snatched on the road",
            "state": "Gujarat",
        },
    )
    chat_id = created.json()["chat_id"]

    response = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "Ahmedabad city",
            "state": "Gujarat",
        },
    )
    assert response.status_code == 200
    assert response.json()["follow_up_question"]
    assert "police location is clearer now" in response.json()["answer"].lower() or "next step" in response.json()["answer"].lower()
    assert "I already have Ahmedabad" not in response.json()["answer"]

    response = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "I have not filed the complaint yet",
            "state": "Gujarat",
        },
    )
    earlier_answer = response.json()["answer"]
    response = client.post(
        "/chat",
        json={
            "chat_id": chat_id,
            "message": "I have the IMEI and invoice",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer != earlier_answer
    assert "ahmedabad" in answer.lower()
    assert "police" in answer.lower()
    assert "bns section 303" in answer.lower() or "bns section 304" in answer.lower()
    assert "bnss section 173" in answer.lower()
    assert "imei and invoice" in answer.lower() or "proof such as i have the imei and invoice" in answer.lower()
    assert "food safety" not in answer.lower()


def test_food_issue_reroutes_to_legal_intake_when_retrieval_is_empty(client):
    response = client.post(
        "/chat",
        json={
            "message": "There is an insect in my sealed chocolate packet",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "i understand" in answer.lower() or "this appears to be a food-safety" in answer.lower() or "main thing now" in answer.lower()
    assert "food safety authority" in answer.lower() or "seller" in answer.lower()
    assert "No highly relevant India Kanoon authority was found" not in answer
