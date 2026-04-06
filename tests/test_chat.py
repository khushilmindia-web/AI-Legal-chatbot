from __future__ import annotations

import requests

from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.intent_service import IntentDecision, IntentRoutingService


def test_first_message_creates_chat_session(client):
    response = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited from my account.",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] > 0
    assert payload["title"].startswith("I clicked a fake UPI")
    assert payload["domain"] == "cyber"
    assert payload["follow_up_question"] is None
    assert "Because UPI was involved" in payload["answer"]
    assert "India Kanoon search results:" not in payload["answer"]

    history = client.get("/chat/history").json()
    assert len(history["items"]) == 1
    assert history["items"][0]["id"] == payload["chat_id"]


def test_chat_prioritizes_upi_fraud_issue_over_greeting_response(client):
    response = client.post(
        "/chat",
        json={
            "message": "I clicked a fake UPI link and money got debited",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "cyber"
    assert "Because UPI was involved" in payload["answer"]
    assert "Legal India" not in payload["answer"]
    assert "How can I help you" not in payload["answer"]


def test_chat_returns_exact_no_results_fallback(client, monkeypatch):
    monkeypatch.setattr(
        IntentRoutingService,
        "route",
        lambda self, message, **kwargs: IntentDecision(handled=False),
    )
    monkeypatch.setattr(
        IndianKanoonService,
        "search_references_multi",
        lambda self, query_variants, doctypes_options, max_results=3: [],
    )

    response = client.post(
        "/chat",
        json={
            "message": "Explain an unknown niche legal theory with no filing path",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    assert "Start by writing down the facts" in response.json()["answer"]


def test_chat_returns_fallback_when_indiankanoon_request_fails(client, monkeypatch):
    monkeypatch.setattr(
        IntentRoutingService,
        "route",
        lambda self, message, **kwargs: IntentDecision(handled=False),
    )

    def raise_timeout(self, query_variants, doctypes_options, max_results=3):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(IndianKanoonService, "search_references_multi", raise_timeout)

    response = client.post(
        "/chat",
        json={
            "message": "Show me a judgment for cheque bounce",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Start by writing down the facts" in payload["answer"] or "Once the facts and documents are clear" in payload["answer"]
    assert payload["chat_id"] > 0


def test_chat_broadens_statute_query_search_when_state_filter_is_too_narrow(client, monkeypatch):
    attempts: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        IntentRoutingService,
        "route",
        lambda self, message, **kwargs: IntentDecision(handled=False),
    )

    def fake_search_references_multi(self, query_variants, doctypes_options, max_results=3):
        for query in query_variants:
            for doctypes in doctypes_options:
                attempts.append((query, doctypes))
                if query == "Indian Penal Code section 420" and doctypes == "judgments,laws":
                    return [
                        {
                            "doc_id": "420",
                            "title": "Section 420 in The Indian Penal Code",
                            "headline": "Cheating and dishonestly inducing delivery of property.",
                            "docsource": "laws",
                            "citations": [],
                            "url": "https://indiankanoon.org/doc/420/",
                        }
                    ]
        return []

    monkeypatch.setattr(IndianKanoonService, "search_references_multi", fake_search_references_multi)

    response = client.post(
        "/chat",
        json={
            "message": "Section 420 IPC",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Section 420 in The Indian Penal Code" in payload["answer"]
    assert payload["domain"] == "criminal"
    assert ("Section 420 IPC", "judgments,laws") in attempts
    assert ("Indian Penal Code section 420", "judgments,laws") in attempts


def test_chat_returns_intent_response_before_indiankanoon(client, monkeypatch):
    monkeypatch.setattr(
        IntentRoutingService,
        "route",
        lambda self, message, **kwargs: IntentDecision(
            handled=True,
            answer="Hello. How can I help you with your legal issue today?",
            intent="greeting",
            confidence=1.0,
            source="heuristic",
        ),
    )

    def fail_if_called(self, query_variants, doctypes_options, max_results=3):
        raise AssertionError("India Kanoon should not be called for handled greeting intents")

    monkeypatch.setattr(IndianKanoonService, "search_references_multi", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "hello",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Hello. How can I help you with your legal issue today?"
    assert payload["citations"] == []
    assert payload["authorities"] == []


def test_chat_routes_procedural_query_to_general_legal_help_without_indiankanoon(client, monkeypatch):
    def procedural_intent(self, message, **kwargs):
        return IntentDecision(
            handled=True,
            answer="I can guide you through the legal process step by step.",
            intent="legal_help",
            confidence=1.0,
            source="pattern",
        )

    def fail_if_called(self, query_variants, doctypes_options, max_results=3):
        raise AssertionError("India Kanoon should not be called for procedural legal-help queries")

    monkeypatch.setattr(IntentRoutingService, "route", procedural_intent)
    monkeypatch.setattr(IndianKanoonService, "search_references_multi", fail_if_called)

    response = client.post(
        "/chat",
        json={
            "message": "How do I file a cyber fraud complaint?",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "legal process step by step" in payload["answer"]
    assert "National Cyber Crime Portal" in payload["answer"]


def test_chat_uses_legal_help_when_indiankanoon_returns_no_result_for_caselaw_query(client, monkeypatch):
    monkeypatch.setattr(
        IntentRoutingService,
        "route",
        lambda self, message, **kwargs: IntentDecision(handled=False),
    )
    monkeypatch.setattr(
        IndianKanoonService,
        "search_references_multi",
        lambda self, query_variants, doctypes_options, max_results=3: [],
    )

    response = client.post(
        "/chat",
        json={
            "message": "Section 420 IPC judgment",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] != "No relevant legal data found on India Kanoon"
    assert "facts in date order" in payload["answer"] or "next step is to prepare the complaint" in payload["answer"]


def test_chat_continues_legal_help_across_turns_in_same_chat(client):
    first = client.post(
        "/chat",
        json={
            "message": "My landlord is harassing me.",
            "state": "Gujarat",
        },
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["chat_id"] > 0

    second = client.post(
        "/chat",
        json={
            "chat_id": first_payload["chat_id"],
            "message": "He keeps threatening to evict me without notice.",
            "state": "Gujarat",
        },
    )

    assert second.status_code == 200
    second_payload = second.json()
    assert "Based on what you've shared so far" in second_payload["answer"]
    assert "India Kanoon search results:" not in second_payload["answer"]

    messages = client.get(f"/chat/{first_payload['chat_id']}/messages").json()["items"]
    assistant_state = messages[-1]["metadata"]["conversation_state"]
    assert assistant_state["active_intent"] == "legal_help"
    assert assistant_state["conversation_started"] is True


def test_chat_uses_city_follow_up_to_advance_police_guidance(client):
    first = client.post(
        "/chat",
        json={
            "message": "How do I file a police complaint for threats?",
            "state": "Gujarat",
        },
    )

    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["follow_up_question"] == "Which city or police station is connected with the complaint?"

    second = client.post(
        "/chat",
        json={
            "chat_id": first_payload["chat_id"],
            "message": "Ahmedabad city",
            "state": "Gujarat",
        },
    )

    assert second.status_code == 200
    payload = second.json()
    assert "Ahmedabad" in payload["answer"]
    assert "diary number or FIR acknowledgement" in payload["answer"]
    assert payload["follow_up_question"] is None

    messages = client.get(f"/chat/{first_payload['chat_id']}/messages").json()["items"]
    assistant_state = messages[-1]["metadata"]["conversation_state"]
    assert assistant_state["city"] == "Ahmedabad"
    assert assistant_state["awaiting_details"] is False


def test_chat_disables_greeting_after_legal_help_conversation_starts(client):
    first = client.post(
        "/chat",
        json={
            "message": "What should I do if my bank account was used in fraud?",
            "state": "Gujarat",
        },
    ).json()

    second = client.post(
        "/chat",
        json={
            "chat_id": first["chat_id"],
            "message": "hello",
            "state": "Gujarat",
        },
    )

    assert second.status_code == 200
    payload = second.json()
    assert "Based on what you've shared so far" in payload["answer"]
    assert "Legal India" not in payload["answer"]
    assert "India Kanoon search results:" not in payload["answer"]


def test_open_old_chat_returns_messages(client):
    created = client.post(
        "/chat",
        json={
            "message": "How do I file an FIR for phone theft?",
            "state": "Gujarat",
        },
    ).json()

    response = client.get(f"/chat/{created['chat_id']}/messages")

    assert response.status_code == 200
    messages = response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_clear_history_removes_all_chats(client):
    client.post("/chat", json={"message": "What can I do if my landlord is harassing me?"})

    clear_response = client.delete("/chat/history")
    history_response = client.get("/chat/history")

    assert clear_response.status_code == 200
    assert history_response.json()["items"] == []
