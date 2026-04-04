from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core import config as config_module
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services.indiankanoon_service import IndianKanoonService


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
    monkeypatch.setenv("INDIANKANOON_API_TOKEN", "test-indiankanoon-token")
    monkeypatch.setenv("DEBUG", "true")
    get_settings.cache_clear()

    def fake_search_references(self, query, doctypes=None, max_results=3):
        return [
            {
                "doc_id": "12345",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses cheque bounce liability.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "url": "https://indiankanoon.org/doc/12345/",
            }
        ]

    monkeypatch.setattr(IndianKanoonService, "search_references", fake_search_references)
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
