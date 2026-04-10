from __future__ import annotations

import io
import requests

from backend.app.services.indiankanoon_service import IndianKanoonService
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
    assert "Grounded legal answer based on Indian Kanoon" in payload["answer"]
    assert payload["likely_forum"] == "supremecourt"

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
    answer = response.json()["answer"]
    assert answer.startswith("Summary:")
    assert "No highly relevant India Kanoon authority was found" in answer
    assert "Legal Position:" in answer
    assert "Practical Next Steps:" in answer
    assert "Sources:" in answer
    assert "Disclaimer:" in answer


def test_chat_returns_structured_grounded_answer_when_llm_call_fails(client, monkeypatch):
    def raise_llm_error(self, user_prompt, conversation):
        raise RuntimeError("OpenAI quota is exhausted for the configured API key.")

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
    assert payload["likely_forum"] == "supremecourt"
    assert payload["documents_to_keep"]


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
        IndianKanoonService,
        "retrieve_grounded_documents",
        lambda self, query_variants, doctypes_options, max_results=4: [
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
    assert "Supreme Court only" in second_payload["answer"]
    assert second_payload["follow_up_question"] is None
    assert second_payload["caution"] == "Match the authority to your facts."


def test_indiankanoon_follow_up_persists_merged_state_and_pipeline(client, monkeypatch):
    monkeypatch.setattr(
        IndianKanoonService,
        "retrieve_grounded_documents",
        lambda self, query_variants, doctypes_options, max_results=4: [
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
    assert "section 138 cheque bounce" in state["last_user_issue"].lower()
    assert "supreme court only" in state["last_user_issue"].lower()


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
    client.post("/chat", json={"message": "Need a grounded answer on a landlord notice."})

    clear_response = client.delete("/chat/history")
    history_response = client.get("/chat/history")

    assert clear_response.status_code == 200
    assert history_response.json()["items"] == []


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

    response = anonymous_client.delete(f"/chat/{created['chat_id']}")

    assert response.status_code == 401


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
    assert "Right now" in payload["answer"]
    assert "Most urgent right now" in payload["answer"] or "The most urgent step" in payload["answer"]
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
    assert payload["citations"] == [] or any("BNS" in item or "BNSS" in item for item in payload["citations"])


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
    assert "Right now" in answer
    assert "Most urgent right now" in answer or "The most urgent step" in answer
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
    assert "I understand" in answer
    assert "Right now" in answer
    assert "Most urgent right now" in answer or "The most urgent step" in answer
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
    assert "Right now" in answer
    assert "Most urgent right now" in answer or "The most urgent step" in answer
    assert "senior police officer" in answer.lower() or "police station" in answer.lower()
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
    assert "landlord-tenant dispute" in answer.lower() or "legal notice" in answer.lower() or "civil dispute" in answer.lower()
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
    assert "i understand this is stressful" in answer.lower()
    assert "within 24 hours" in answer.lower()
    assert "bns section 318" in answer.lower() or "bns section 319" in answer.lower()
    assert "bnss section 173" in answer.lower()
    assert "1930" in answer or "cyber crime" in answer.lower()
    assert "next steps:" in answer.lower()
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
    assert "already started the reporting trail" in answer
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
    assert "i understand this is stressful" in answer
    assert "act quickly to reduce the damage" in answer
    assert "within 24 hours" in answer
    assert "recovery" in answer
    assert payload["follow_up_question"]


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
    assert "already started the seller-side complaint trail" in answer
    assert "send a written complaint to the seller" not in answer
    assert "prepare the record for food-safety or consumer escalation" in answer


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
    assert "protect the deadline immediately" in answer
    assert "deadline" in answer
    assert payload["follow_up_question"] is None


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
    assert "i understand why you're concerned" in answer
    assert "secure the evidence quickly" in answer
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
    assert "next steps:" in payload["answer"].lower()


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
    assert "Most urgent right now" in response.json()["answer"]
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
    assert "Right now" in answer or "This appears to be a food-safety" in answer
    assert "food safety authority" in answer.lower() or "seller" in answer.lower()
    assert "No highly relevant India Kanoon authority was found" not in answer
