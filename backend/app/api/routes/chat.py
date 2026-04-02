from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.app.models.schemas import (
    ChatHistoryResponse,
    ChatMessageRecord,
    ChatMessagesResponse,
    ChatRequest,
    ChatSessionSummary,
    ChatUploadResponse,
)
from backend.app.services.chat_service import ChatService


router = APIRouter(tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    return ChatService(settings=request.app.state.settings, store=request.app.state.session_store)


@router.post("/chat", response_model=ChatUploadResponse)
async def chat(request_body: ChatRequest, request: Request) -> ChatUploadResponse:
    service = get_chat_service(request)
    try:
        return await service.handle_chat(request_body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat/upload", response_model=ChatUploadResponse)
async def chat_upload(
    request: Request,
    message: str = Form(...),
    state: str | None = Form(default=None),
    district: str | None = Form(default=None),
    case_stage: str | None = Form(default=None),
    is_own_matter: bool | None = Form(default=None),
    chat_id: int | None = Form(default=None),
    image_urls: str | None = Form(default=None),
    files: list[UploadFile] = File(default_factory=list),
) -> ChatUploadResponse:
    service = get_chat_service(request)
    image_url_list = [item.strip() for item in (image_urls or "").split(",") if item.strip()]
    extraction = await service.extractor.extract_uploads(files=files, image_urls=image_url_list)
    payload = ChatRequest(
        message=message,
        state=state,
        district=district,
        case_stage=case_stage,
        is_own_matter=is_own_matter,
        chat_id=chat_id,
    )
    try:
        return await service.handle_chat(payload, uploaded_texts=extraction.texts, extraction_warnings=extraction.warnings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/chat/history", response_model=ChatHistoryResponse)
def chat_history(request: Request) -> ChatHistoryResponse:
    store = request.app.state.session_store
    items = [
        ChatSessionSummary(
            id=item["id"],
            title=item["title"],
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            last_message_preview=item.get("last_message_preview"),
        )
        for item in store.list_sessions()
    ]
    return ChatHistoryResponse(items=items)


@router.get("/chat/{chat_id}/messages", response_model=ChatMessagesResponse)
def chat_messages(chat_id: int, request: Request) -> ChatMessagesResponse:
    store = request.app.state.session_store
    items = [
        ChatMessageRecord(
            id=item["id"],
            chat_id=item["chat_id"],
            role=item["role"],
            content=item["content"],
            created_at=datetime.fromisoformat(item["created_at"]),
            metadata=item["metadata"],
        )
        for item in store.get_messages(chat_id)
    ]
    return ChatMessagesResponse(items=items)


@router.delete("/chat/history")
def clear_history(request: Request) -> dict[str, str]:
    request.app.state.session_store.clear_history()
    return {"status": "ok"}
