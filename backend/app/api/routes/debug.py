from pathlib import Path

from fastapi import APIRouter, Request

from backend.app.models.schemas import DebugStatusResponse
from backend.app.services.file_extractor import PdfReader, pdfplumber, pytesseract


router = APIRouter(tags=["debug"])


@router.get("/debug/status", response_model=DebugStatusResponse)
def debug_status(request: Request) -> DebugStatusResponse:
    settings = request.app.state.settings
    knowledge_files = len(
        [
            path
            for path in settings.knowledge_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ]
    )
    return DebugStatusResponse(
        api_status="up",
        configured_model=settings.openai_model,
        openai_key_configured=bool(settings.openai_api_key),
        retrieval_mode=settings.retrieval_mode,
        upload_support={
            "pdfplumber": pdfplumber is not None,
            "pypdf2": PdfReader is not None,
            "pillow": True,
            "pytesseract_fallback": pytesseract is not None,
        },
        knowledge_files=knowledge_files,
    )
