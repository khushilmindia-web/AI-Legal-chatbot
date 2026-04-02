from __future__ import annotations


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_debug_status_endpoint(client):
    response = client.get("/debug/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_status"] == "up"
    assert payload["configured_model"] == "gpt-4.1-mini"
    assert payload["openai_key_configured"] is True
