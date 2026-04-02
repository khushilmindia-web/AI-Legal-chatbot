from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    chat_id: int | None = None
    state: str | None = None
    district: str | None = None
    case_stage: str | None = None
    is_own_matter: bool | None = None


class ChatUploadResponse(BaseModel):
    chat_id: int
    title: str
    answer: str
    created_at: datetime
    domain: str | None = None
    follow_up_question: str | None = None
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    documents_to_keep: list[str] = Field(default_factory=list)
    likely_forum: str | None = None
    caution: str | None = None


class ChatSessionSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_preview: str | None = None


class ChatHistoryResponse(BaseModel):
    items: list[ChatSessionSummary]


class ChatMessageRecord(BaseModel):
    id: int
    chat_id: int
    role: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessagesResponse(BaseModel):
    items: list[ChatMessageRecord]


class HealthResponse(BaseModel):
    status: str


class DebugStatusResponse(BaseModel):
    api_status: str
    configured_model: str
    openai_key_configured: bool
    retrieval_mode: str
    upload_support: dict[str, bool]
    knowledge_files: int


class InternalChatResult(BaseModel):
    answer: str
    domain: str | None = None
    follow_up_question: str | None = None
    citations: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    documents_to_keep: list[str] = Field(default_factory=list)
    likely_forum: str | None = None
    caution: str | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_json: dict[str, Any] = Field(default_factory=dict)
