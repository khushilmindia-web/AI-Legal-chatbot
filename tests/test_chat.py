from __future__ import annotations

from backend.app.services.indiankanoon_service import IndianKanoonService


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
    assert "India Kanoon search results:" in payload["answer"]

    history = client.get("/chat/history").json()
    assert len(history["items"]) == 1
    assert history["items"][0]["id"] == payload["chat_id"]


def test_chat_returns_exact_no_results_fallback(client, monkeypatch):
    monkeypatch.setattr(IndianKanoonService, "search_references", lambda self, query, doctypes=None, max_results=3: [])

    response = client.post(
        "/chat",
        json={
            "message": "Explain an unknown niche statute with no indexed result",
            "state": "Gujarat",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "No relevant legal data found on India Kanoon"


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
