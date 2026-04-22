from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.app.api.auth_utils import require_current_user
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
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = ChatService(settings=request.app.state.settings, store=request.app.state.session_store)
        request.app.state.chat_service = service
    return service


@router.post("/chat", response_model=ChatUploadResponse)
async def chat(request_body: ChatRequest, request: Request) -> ChatUploadResponse:
    user = require_current_user(request)
    service = get_chat_service(request)
    try:
        return await service.handle_chat(request_body, user_id=user["id"], fallback_state=user.get("state"))
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
    user = require_current_user(request)
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
        return await service.handle_chat(
            payload,
            user_id=user["id"],
            fallback_state=user.get("state"),
            uploaded_texts=extraction.texts,
            extraction_warnings=extraction.warnings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/chat/history", response_model=ChatHistoryResponse)
def chat_history(request: Request) -> ChatHistoryResponse:
    user = require_current_user(request)
    store = request.app.state.session_store
    items = [
        ChatSessionSummary(
            id=item["id"],
            title=item["title"],
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            last_message_preview=item.get("last_message_preview"),
        )
        for item in store.list_sessions(user["id"])
    ]
    return ChatHistoryResponse(items=items)


@router.get("/chat/{chat_id}/messages", response_model=ChatMessagesResponse)
def chat_messages(chat_id: int, request: Request) -> ChatMessagesResponse:
    user = require_current_user(request)
    store = request.app.state.session_store
    session = store.get_session(chat_id, user["id"])
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    items = [
        ChatMessageRecord(
            id=item["id"],
            chat_id=item["chat_id"],
            role=item["role"],
            content=item["content"],
            created_at=datetime.fromisoformat(item["created_at"]),
            metadata=item["metadata"],
        )
        for item in store.get_messages(chat_id, user["id"])
    ]
    return ChatMessagesResponse(items=items)


@router.delete("/chat/history")
def clear_history(request: Request) -> dict[str, str]:
    user = require_current_user(request)
    request.app.state.session_store.clear_history(user["id"])
    return {"status": "ok"}


@router.delete("/chat/{chat_id}")
def delete_chat(chat_id: int, request: Request) -> dict[str, str]:
    user = require_current_user(request)
    deleted = request.app.state.session_store.delete_session(chat_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "ok"}
