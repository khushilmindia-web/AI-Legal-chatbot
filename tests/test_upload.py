from __future__ import annotations

import asyncio
import io

from backend.app.services.file_extractor import FileExtractionService


def test_chat_upload_accepts_text_file(client):
    response = client.post(
        "/chat/upload",
        data={"message": "Please review this notice and explain the next step."},
        files={"files": ("notice.txt", io.BytesIO(b"Legal notice regarding payment default."), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_id"] > 0
    assert "general legal information" in payload["answer"].lower()


def test_file_extractor_reads_text_file(tmp_path):
    from backend.app.core import config as config_module
    from backend.app.core.config import get_settings

    config_module.TEMP_DIR = tmp_path / "temp"
    config_module.KNOWLEDGE_DIR = tmp_path / "knowledge"
    config_module.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    config_module.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    get_settings.cache_clear()
    settings = get_settings()
    extractor = FileExtractionService(settings)

    class DummyUpload:
        filename = "facts.txt"

        async def read(self):
            return b"Consumer complaint invoice and seller response."

    result = asyncio.run(extractor.extract_uploads([DummyUpload()]))
    assert result.texts
    assert "Consumer complaint" in result.texts[0]
