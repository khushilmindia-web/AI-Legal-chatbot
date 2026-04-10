from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core import config as config_module
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.openai_service import OpenAIResponsesService


def _apply_windows_pytest_tempdir_workaround() -> None:
    if os.name != "nt":
        return
    try:
        import _pytest.pathlib as pytest_pathlib
    except Exception:
        return

    original_make_numbered_dir = pytest_pathlib.make_numbered_dir

    def _safe_make_numbered_dir(root: Path, prefix: str, mode: int = 0o700) -> Path:
        # Use a less restrictive mode on Windows sandboxed workspaces.
        return original_make_numbered_dir(root=root, prefix=prefix, mode=0o777)

    pytest_pathlib.make_numbered_dir = _safe_make_numbered_dir
    pytest_pathlib.cleanup_dead_symlinks = lambda root: None


_apply_windows_pytest_tempdir_workaround()


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

    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        return [
            {
                "doc_id": "12345",
                "title": "Sample Supreme Court Decision",
                "headline": "This judgment discusses legal remedies and procedural obligations.",
                "fragment_headline": "Relevant fragment from the retrieved Indian Kanoon document.",
                "doc_excerpt": "The document explains the applicable legal position and the immediate procedural steps available.",
                "docsource": "supremecourt",
                "citations": ["(2024) 1 SCC 100"],
                "publishdate": "01-01-2024",
                "url": "https://indiankanoon.org/doc/12345/",
            }
        ]

    def fake_generate_json(self, user_prompt, conversation):
        return {
            "answer": "Summary: Grounded legal answer based on Indian Kanoon.\nLegal position: The retrieved authority explains the relevant position.\nPractical next steps: Follow the immediate remedy described in the grounded material.\nSources: Sample Supreme Court Decision.\nDisclaimer: This is general legal information.",
            "follow_up_question": None,
            "likely_forum": "supremecourt",
            "caution": "Verify facts with the original record before taking action.",
            "documents_to_keep": ["complaint copy", "ID proof"],
        }

    monkeypatch.setattr(IndianKanoonService, "retrieve_grounded_documents", fake_retrieve_grounded_documents)
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
