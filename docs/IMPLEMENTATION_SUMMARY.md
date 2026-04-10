# AI Legal Chatbot Implementation Summary

## Active Runtime

- Primary backend entrypoint: `backend/app/main.py`
- API routes: `backend/app/api/routes/auth.py`, `chat.py`, `health.py`, `debug.py`
- Chat orchestration: `backend/app/services/chat_service.py`
- Grounded legal retrieval: `backend/app/services/indiankanoon_service.py`
- Session/auth/chat persistence: `backend/app/services/session_store.py`
- Frontend auth + chat UI: `Frontend/auth.html`, `Frontend/auth.js`, `Frontend/index.html`, `Frontend/app.js`

## Active Chat Flow

1. Authenticated user sends `POST /chat` or `POST /chat/upload`.
2. Chat service restores conversation state from the session store.
3. Direct practical legal-help flows are handled in the legal-help interview path when appropriate.
4. Authority/statute/case-law queries run through India Kanoon grounded retrieval.
5. Uploaded document text is added into grounded context and reflected in final citations/answer guidance.
6. Final answers are normalized into strict sections: Summary, Legal Position, Practical Next Steps, Sources, Disclaimer.
7. Updated conversation state is persisted for follow-up continuity across turns.

## Notes

- `app.py` is legacy and not the active runtime path for the current architecture.
- Cookie-based session auth is the active frontend/backend contract; frontend token persistence is not required for normal operation.
- The test suite runs through `pytest tests -q` with workspace-safe temp handling in `tests/conftest.py` and `pytest.ini`.
