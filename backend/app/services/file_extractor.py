from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from fastapi import UploadFile
from PIL import Image

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.file_extractor")
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}


@dataclass
class ExtractionResult:
    texts: list[str]
    warnings: list[str]


class FileExtractionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def extract_uploads(self, files: Iterable[UploadFile] | None, image_urls: list[str] | None = None) -> ExtractionResult:
        texts: list[str] = []
        warnings: list[str] = []
        for file in files or []:
            try:
                file_text = await self._extract_single_upload(file)
                if file_text:
                    texts.append(file_text)
            except Exception as exc:
                warnings.append(f"{file.filename or 'file'} could not be processed: {exc}")
                logger.warning("upload extraction failed filename=%s error=%s", file.filename, exc)

        for url in image_urls or []:
            try:
                url_text = self._extract_from_url(url)
                if url_text:
                    texts.append(url_text)
            except Exception as exc:
                warnings.append(f"{url} could not be processed: {exc}")
                logger.warning("url extraction failed url=%s error=%s", url, exc)
        return ExtractionResult(texts=texts, warnings=warnings)

    async def _extract_single_upload(self, upload: UploadFile) -> str:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

        content = await upload.read()
        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(f"File exceeds {self.settings.max_file_size_mb} MB limit")

        if suffix == ".txt":
            return content.decode("utf-8", errors="ignore")[:12000]
        if suffix == ".pdf":
            return self._extract_pdf(content)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._extract_image_text(content)
        raise ValueError("Unsupported file content")

    def _extract_pdf(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=self.settings.temp_dir) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        try:
            text = []
            if pdfplumber is not None:
                with pdfplumber.open(temp_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text.append(page_text)
            if not text and PdfReader is not None:
                with temp_path.open("rb") as handle:
                    reader = PdfReader(handle)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text.append(page_text)
            if not text and pdfplumber is None and PdfReader is None:
                raise RuntimeError("PDF extraction dependencies are not installed")
            return "\n".join(text)[:15000]
        finally:
            temp_path.unlink(missing_ok=True)

    def _extract_image_text(self, content: bytes) -> str:
        if pytesseract is None:
            return ""
        image = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(image)[:10000]

    def _extract_from_url(self, url: str) -> str:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "text/html" in content_type or "text/plain" in content_type:
            return response.text[:8000]
        return ""
