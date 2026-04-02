from __future__ import annotations


def test_blank_history_before_first_real_chat(client):
    response = client.get("/chat/history")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_history_appears_only_after_message_is_sent(client):
    client.post(
        "/chat",
        json={"message": "How do I file an FIR?", "state": "Gujarat"},
    )

    response = client.get("/chat/history")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
