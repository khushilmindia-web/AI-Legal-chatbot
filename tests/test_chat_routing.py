from __future__ import annotations


def test_new_chat_is_not_created_until_first_message(client):
    history = client.get("/chat/history")
    assert history.status_code == 200
    assert history.json()["items"] == []


def test_chat_response_contains_session_id_and_guidance_fields(client):
    response = client.post(
        "/chat",
        json={
            "message": "I broke a traffic signal. What happens now?",
            "state": "Gujarat",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["chat_id"] > 0
    assert payload["authorities"]
    assert payload["documents_to_keep"]
    assert payload["likely_forum"]
