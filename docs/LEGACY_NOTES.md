# Legacy Notes

## Deprecated / Historical Paths

- `app.py` is retained as a historical legacy file and is not the active runtime entrypoint.
- The current runtime entrypoint is `backend/app/main.py`.
- `backend/app/services/session_store.py` is the retained SQLite backup implementation. It is not active at runtime while MongoDB migration is enabled.
- `data/lawyer_ai.db` is retained as the SQLite backup data file and should not be deleted during the MongoDB cutover window.

## Guidance

- Use `backend/app/main.py` and modules under `backend/app/` for all new development.
- Use `backend/app/services/mongo_session_store.py` through `backend/app/services/storage.py` for active persistence work.
- Treat legacy root-level runtime code as archival context only unless an explicit migration task requires touching it.
- Rollback to SQLite must be an intentional change: preserve/export current MongoDB data first, then re-enable the SQLite branch in `backend/app/services/storage.py` and set `DATABASE_BACKEND=sqlite`.
