from __future__ import annotations

from uuid import uuid4
from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient

from backend.app.core import config as config_module
from backend.app.core.config import get_settings
from backend.app.main import create_app
from backend.app.services.indiankanoon_service import IndianKanoonService
from backend.app.services.openai_service import OpenAIResponsesService


def _configure_local_pytest_temp_root() -> Path:
    base_dir = (Path(__file__).resolve().parent.parent / "manual_pytest_tmp").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


PYTEST_TEMP_ROOT = _configure_local_pytest_temp_root()


def _apply_windows_pytest_tempdir_workaround() -> None:
    try:
        import _pytest.pathlib as pytest_pathlib
    except Exception:
        return

    original_make_numbered_dir = pytest_pathlib.make_numbered_dir

    def _safe_make_numbered_dir(root: Path, prefix: str, mode: int = 0o700) -> Path:
        target_root = PYTEST_TEMP_ROOT if os.name == "nt" else root
        target_root.mkdir(parents=True, exist_ok=True)
        target_mode = 0o777 if os.name == "nt" else mode
        return original_make_numbered_dir(root=target_root, prefix=prefix, mode=target_mode)

    pytest_pathlib.make_numbered_dir = _safe_make_numbered_dir
    pytest_pathlib.cleanup_dead_symlinks = lambda root: None


_apply_windows_pytest_tempdir_workaround()


@pytest.fixture()
def tmp_path() -> Path:
    path = PYTEST_TEMP_ROOT / f"case-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def anonymous_client(monkeypatch: pytest.MonkeyPatch):
    tmp_path = PYTEST_TEMP_ROOT / f"chat-tests-{uuid4().hex[:8]}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    mongo_database = f"lawyer_ai_test_{uuid4().hex[:12]}"
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "cyber_fraud_basics.txt").write_text(
        "Cyber guidance: Preserve screenshots, transaction IDs, bank complaint acknowledgements, and complaint reference numbers.",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "DB_PATH", tmp_path / "lawyer_ai_test.db")
    monkeypatch.setattr(config_module, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(config_module, "TEMP_DIR", tmp_path / "tmp")
    config_module.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("INDIANKANOON_API_TOKEN", "test-indiankanoon-token")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DATABASE_BACKEND", "mongodb")
    monkeypatch.setenv("MONGODB_DATABASE", mongo_database)
    get_settings.cache_clear()

    def fake_retrieve_grounded_documents(self, query_variants, doctypes_options, max_results=4):
        query_text = " ".join(str(item) for item in query_variants).lower()
        if "section 138" in query_text or "cheque bounce" in query_text:
            return [
                {
                    "doc_id": "138-law",
                    "title": "Section 138 in The Negotiable Instruments Act, 1881",
                    "headline": "Dishonour of cheque for insufficiency of funds.",
                    "fragment_headline": "Statutory text of section 138.",
                    "doc_excerpt": "Where any cheque drawn by a person is returned unpaid because funds are insufficient, the statutory consequences under section 138 follow subject to the provisos.",
                    "docsource": "laws",
                    "citations": ["NI Act 138"],
                    "publishdate": "01-01-1881",
                    "url": "https://indiankanoon.org/doc/138-law/",
                    "score": 41.0,
                },
                {
                    "doc_id": "138-case",
                    "title": "Sample Supreme Court Decision",
                    "headline": "This judgment discusses cheque bounce liability.",
                    "fragment_headline": "Relevant fragment from the retrieved Indian Kanoon document.",
                    "doc_excerpt": "The judgment explains the applicable legal position and notice-related procedural steps for section 138 matters.",
                    "docsource": "supremecourt",
                    "citations": ["(2024) 1 SCC 100"],
                    "publishdate": "01-01-2024",
                    "url": "https://indiankanoon.org/doc/12345/",
                    "score": 24.0,
                },
            ]
        if "section 420" in query_text or "ipc 420" in query_text:
            return [
                {
                    "doc_id": "420-law",
                    "title": "Section 420 in The Indian Penal Code, 1860",
                    "headline": "Cheating and dishonestly inducing delivery of property.",
                    "fragment_headline": "Punishment may extend to seven years and fine.",
                    "doc_excerpt": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished with imprisonment which may extend to seven years, and shall also be liable to fine.",
                    "docsource": "laws",
                    "citations": ["IPC 420"],
                    "publishdate": "01-01-1860",
                    "url": "https://indiankanoon.org/doc/420-law/",
                    "score": 42.0,
                },
                {
                    "doc_id": "420-case",
                    "title": "Sample Supreme Court Decision",
                    "headline": "This judgment discusses section 420 ingredients and interpretation.",
                    "fragment_headline": "Relevant fragment from the retrieved Indian Kanoon document.",
                    "doc_excerpt": "The judgment explains the applicable legal position for section 420.",
                    "docsource": "supremecourt",
                    "citations": ["(2024) 1 SCC 100"],
                    "publishdate": "01-01-2024",
                    "url": "https://indiankanoon.org/doc/12345/",
                    "score": 21.0,
                },
            ]
        if "landlord" in query_text and "notice" in query_text:
            return [
                {
                    "doc_id": "landlord-1",
                    "title": "Sample High Court Landlord Notice Decision",
                    "headline": "This judgment discusses landlord notice requirements and procedural obligations.",
                    "fragment_headline": "Relevant fragment from the retrieved Indian Kanoon document.",
                    "doc_excerpt": "The document explains the applicable legal position and notice-related steps in landlord matters.",
                    "docsource": "highcourt",
                    "citations": ["2024 HC 321"],
                    "publishdate": "01-03-2024",
                    "url": "https://indiankanoon.org/doc/landlord-1/",
                    "score": 34.0,
                }
            ]
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
                "score": 32.0,
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
    try:
        with TestClient(app) as client:
            yield client
    finally:
        store = getattr(app.state, "session_store", None)
        client_obj = getattr(store, "client", None)
        if client_obj is not None and mongo_database.startswith("lawyer_ai_test_"):
            client_obj.drop_database(mongo_database)
            client_obj.close()
        get_settings.cache_clear()


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
