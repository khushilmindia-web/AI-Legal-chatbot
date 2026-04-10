# Legacy Notes

## Deprecated / Historical Paths

- `app.py` is retained as a historical legacy file and is not the active runtime entrypoint.
- The current runtime entrypoint is `backend/app/main.py`.

## Guidance

- Use `backend/app/main.py` and modules under `backend/app/` for all new development.
- Treat legacy root-level runtime code as archival context only unless an explicit migration task requires touching it.
