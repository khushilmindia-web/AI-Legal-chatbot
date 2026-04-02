from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core import config as config_module
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services.openai_service import OpenAIResponsesService


@pytest.fixture()
def anonymous_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "cyber_fraud_basics.txt").write_text(
        "Cyber guidance: Preserve screenshots, transaction IDs, bank complaint acknowledgements, and complaint reference numbers.",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "DB_PATH", tmp_path / "lawyer_ai_test.db")
    monkeypatch.setattr(config_module, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(config_module, "TEMP_DIR", tmp_path / "temp_uploads")
    config_module.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("DEBUG", "true")
    get_settings.cache_clear()

    def fake_generate_json(self, user_prompt, conversation):
        return {
            "answer": "Summary: This appears to be a cyber-fraud issue.\nImmediate next step: Contact the bank and preserve all proof.\nDocuments: Keep screenshots and transaction details.\nAuthority/forum: Cyber cell or local police station.\nCaution: State-specific procedure may differ.",
            "follow_up_question": "Which bank or payment app was involved?",
            "authorities": ["Information Technology Act, 2000"],
            "documents_to_keep": ["Screenshots", "Transaction ID"],
            "likely_forum": "Cyber cell or local police station",
            "caution": "Act quickly to preserve complaint options.",
            "citations": ["knowledge/cyber_fraud_basics.txt"],
        }

    monkeypatch.setattr(OpenAIResponsesService, "generate_json", fake_generate_json)
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def client(anonymous_client: TestClient) -> TestClient:
    response = anonymous_client.post(
        "/auth/signup",
        json={
            "full_name": "Test User",
            "email": "test@example.com",
            "password": "password123",
            "state": "Gujarat",
        },
    )
    assert response.status_code == 200
    return anonymous_client
