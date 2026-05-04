# AI Legal Chatbot Implementation Summary

## Active Runtime

- Primary backend entrypoint: `backend/app/main.py`
- API routes: `backend/app/api/routes/auth.py`, `chat.py`, `health.py`, `debug.py`
- Chat orchestration: `backend/app/services/chat_service.py`
- Grounded legal retrieval: `backend/app/services/indiankanoon_service.py`
- Active session/auth/chat persistence: `backend/app/services/mongo_session_store.py`
- Storage dependency boundary: `backend/app/services/storage.py`
- SQLite backup implementation retained but inactive: `backend/app/services/session_store.py`
- Frontend auth + chat UI: `Frontend/auth.html`, `Frontend/auth.js`, `Frontend/index.html`, `Frontend/app.js`

## Active Chat Flow

1. Authenticated user sends `POST /chat` or `POST /chat/upload`.
2. Chat service restores conversation state from the session store.
3. Direct practical legal-help flows are handled in the legal-help interview path when appropriate.
4. Authority/statute/case-law queries run through India Kanoon grounded retrieval.
5. Uploaded document text is added into grounded context and reflected in final citations/answer guidance.
6. Final answers are normalized into strict sections: Summary, Legal Position, Practical Next Steps, Sources, Disclaimer.
7. Updated conversation state is persisted for follow-up continuity across turns.

## Database Runtime

- MongoDB is the active runtime database.
- Required settings: `DATABASE_BACKEND=mongodb`, `MONGODB_URI`, and `MONGODB_DATABASE`.
- The active app startup path calls `create_session_store(settings)` and creates `MongoSessionStore`.
- SQLite is temporarily disabled during MongoDB migration. The old SQLite store remains in `backend/app/services/session_store.py` as backup code only and is not instantiated by the active runtime.
- The SQLite backup file remains at `data/lawyer_ai.db`.
- One-time migration utility: `scripts/migrate_sqlite_to_mongo.py`.
- Dry-run: `python scripts/migrate_sqlite_to_mongo.py --dry-run`.
- Real migration: `python scripts/migrate_sqlite_to_mongo.py`.
- Rollback requires an intentional code/config change to re-enable the SQLite branch in `backend/app/services/storage.py`; do not delete `data/lawyer_ai.db`.

## Notes

- `app.py` is legacy and not the active runtime path for the current architecture.
- Cookie-based session auth is the active frontend/backend contract; frontend token persistence is not required for normal operation.
- The test suite runs through `pytest tests -q` with workspace-safe temp handling in `tests/conftest.py` and `pytest.ini`.
