# Changelog

## 2026-04-03

### Changed
- Completed the local runtime migration to `http://127.0.0.1:5000` across the active FastAPI app, env defaults, startup helpers, README instructions, and remaining legacy fallback entrypoints.
- Updated `backend/legacy_app_v1.py` so its fallback frontend URL now uses lowercase `/frontend/index.html` and its direct `uvicorn` launcher no longer points back to port `8000`.

### Verified
- Verified the active app responds successfully on `http://127.0.0.1:5000/health`.
- Verified `http://127.0.0.1:5000/frontend/auth.html` loads successfully from the FastAPI-served frontend.
- Verified unauthenticated access to `http://127.0.0.1:5000/frontend/index.html` is still protected and redirects before loading the app workspace.
- Verified the active runtime/config files no longer contain remaining `5500`, `8000`, or `8001` project-port dependencies outside historical changelog notes and a non-port numeric text limit in `backend/app/services/file_extractor.py`.
- Verified the live `127.0.0.1:5000` auth flow end-to-end for signup, logout, login, and `/auth/me`.
- Verified the live `127.0.0.1:5000` chat flow for `/chat`, `/chat/history`, and `/chat/{chat_id}/messages`.
- Verified the live `127.0.0.1:5000` forgot-password flow for `/auth/password-reset`, persisted reset-token hashing in SQLite, `/auth/password-reset/confirm`, old-password rejection, and new-password login.

### Pending
- Resolve the external auto-restarting listener on `127.0.0.1:8000`, which is still outside this repo and remains the reason earlier manual browser checks on `8000` were unreliable.

## 2026-04-02

### Added
- Added explicit OpenAI quota-failure detection in `backend/app/services/openai_service.py` so insufficient quota is reported clearly instead of being retried like a transient network error.
- Added a live localhost runtime on `127.0.0.1:5500` with FastAPI serving both frontend and backend on the same origin after clearing the conflicting local web-server port usage.
- Added a one-origin `5500` runtime path across docs and startup helpers so FastAPI can serve both API routes and frontend assets from the same local origin.
- Added a single-origin mounted chat workspace in `frontend/index.html` that now serves the active chat UI directly from FastAPI instead of the broken redirect stub.
- Added Git automation helpers in `scripts/git_sync.ps1` and `scripts/update_changelog.ps1` to support repeatable local commit/push and changelog maintenance workflows.
- Added runtime auth routes in `backend/app/api/routes/auth.py` for signup, login, logout, current-user lookup, password reset messaging, Google OAuth status, login, and callback.
- Added shared auth helpers in `backend/app/api/auth_utils.py` for bearer-token parsing, cookie lookup, current-user enforcement, and auth cookie management.
- Added auth request/response schemas in `backend/app/models/schemas.py`.
- Added a lowercase `Frontend/index.html` alias that forwards to the existing app entrypoint.
- Added auth regression coverage in `tests/test_auth_flow.py`.

### Changed
- Changed local runtime configuration to prefer `http://127.0.0.1:5500` for FastAPI, Google OAuth redirect configuration, and same-origin CORS defaults in `.env`, `.env.example`, `backend/app/core/config.py`, `scripts/run_local.ps1`, `README.md`, and helper setup scripts.
- Changed the active frontend to use relative same-origin auth and API paths in `frontend/config.js`, `frontend/auth.js`, and `frontend/app.js`, removing the old `:5500` and hardcoded `:8000` frontend assumptions.
- Changed runtime CORS defaults and local env examples to prefer the FastAPI origin only, keeping frontend and backend on `127.0.0.1:8000`.
- Normalized the active FastAPI-served frontend flow around `http://127.0.0.1:8000` by updating env defaults, startup helpers, Google OAuth setup guidance, and README run instructions.
- Updated `backend/app/main.py` to redirect `/`, `/frontend`, and legacy `/frontend/Index.html` into the mounted same-origin frontend/auth flow instead of relying on mixed entrypoints.
- Normalized the active frontend app entrypoint to lowercase `/frontend/index.html` so routing no longer depends on Windows-only case-insensitive file handling.
- Updated repository automation so the local `main` branch is connected to `origin/main` and GitHub pushes use the working system SSH/Git configuration.
- Integrated authentication into the active runtime app in `backend/app/main.py` instead of relying on the separate root `app.py` flow.
- Updated `backend/app/services/session_store.py` to support users, auth sessions, Google account linking, and per-user chat ownership.
- Updated `backend/app/api/routes/chat.py` and `backend/app/services/chat_service.py` so chat history, message loading, uploads, and new chats are scoped to the authenticated user.
- Made authentication mandatory before accessing `frontend/index.html` and `frontend/Index.html`, with unauthenticated users redirected to `frontend/auth.html`.
- Refreshed `Frontend/auth.html`, `Frontend/auth.js`, and `Frontend/style.css` so the auth experience matches the main app with a more modern, responsive UI.
- Updated `Frontend/Index.html` and `Frontend/app.js` to verify session state on boot, render the signed-in user, support logout, and use the integrated backend auth flow.
- Updated `backend/app/core/config.py` so frontend path detection, auth cookie settings, Google OAuth config, and CORS-related runtime settings come from one place.
- Updated `tests/conftest.py` to create authenticated test clients for the runtime app.
- Updated `pytest.ini` to stop using the previously locked fixed temp directory.

### Verified
- Verified direct OpenAI access is no longer blocked by the broken local proxy environment, and the current remaining failure is `429 insufficient_quota` from the configured API key.
- Verified the full live one-port flow on `http://127.0.0.1:5500` for `/health`, `/frontend/auth.html`, protected `/frontend/index.html`, `/auth/google/status`, signup, `/auth/me`, `/chat`, `/chat/history`, `/chat/{chat_id}/messages`, and `/chat/history` delete.
- Verified the active runtime files no longer contain hardcoded `8000` frontend/API assumptions, and the FastAPI app remains same-origin ready through mounted `/frontend` routes.
- Verified the one-port FastAPI flow on `http://127.0.0.1:8000` for `/frontend/auth.html`, protected `/frontend/index.html`, signup, `/auth/me`, `/chat`, `/chat/history`, `/chat/{chat_id}/messages`, `/chat/history` delete, and `/auth/google/status`.
- Verified with an in-process FastAPI smoke harness that auth, redirects, Google OAuth status, sessions, `/chat`, `/chat/history`, `/chat/{chat_id}/messages`, and delete-history all pass under the repo's current app configuration.
- Verified the live machine still has a conflicting listener on `127.0.0.1:8000`; the active process there does not match this repo's route contract and returns `404` for `/health` and `405` for `DELETE /chat/history`.
- Verified GitHub sync is working: the repository now pushes successfully to `git@github.com:khushilmindia-web/AI-Legal-chatbot.git` and `main` tracks `origin/main`.
- Verified signup, login, `/auth/me`, protected root redirect, protected frontend access, per-user chat flow, history loading, logout, and mocked Google OAuth callback/session persistence using a direct FastAPI `TestClient` integration harness.

### Pending
- Run the full `pytest` suite cleanly once the local Windows temp-directory permission issue no longer interferes with pytest cleanup.
- Perform a live browser roundtrip against real Google OAuth credentials after deployment/runtime environment confirmation.
- Clear the foreign `127.0.0.1:8000` listener on this workstation and rerun the live same-origin smoke test against the repo's own FastAPI process.

## 2026-04-01

### Added
- Introduced a stronger legal research core in `backend/legal_research.py`.
- Added citation normalization, authority ranking, and domain reference libraries.
- Added `research_brief` support to backend chat responses and stored metadata.
- Added document kind inference for FIR, notice, order, agreement, salary, property, complaint, and bank records.
- Added richer legal-source metadata in TXT ingestion, including `source_type`, `authority_level`, `citation_hint`, and support for `Rule` and `Chapter`.
- Upgraded the mounted `frontend/` chat UI to the newer session-aware, document-aware workflow.
- Installed Git for Windows locally.
- Added a root `.gitignore` to keep secrets, logs, runtime databases, and cache files out of Git.

### Changed
- Improved RAG reranking for TXT, state-law, case-law, and PDF sources using state/domain/source-aware bonuses.
- Strengthened final legal responses with better grounded citations, forum strategy, and research brief sections.
- Restarted FastAPI with `backend.app:app` so the backend v2 stack is the active runtime.
- Reinitialized the project as a clean Git repository after removing a broken partial `.git` folder with a stale lock file and deny ACL.
- Updated `scripts/git_sync.ps1` to fall back to the installed Git path when PATH is not refreshed yet.
- Updated `scripts/git_sync.ps1` to use Windows OpenSSH for pushes because Git-for-Windows SSH was failing locally with permission errors.

### Pending
- Ingest real statute corpus into `data/Docs`.
- Ingest real state-rule corpus into `data/StateLaws`.
- Ingest real case-law corpus into `data/CaseLaw`.
- Run `/debug/ingest` after legal source files are added.
- Set up Git on this machine and connect authenticated pushes to `git@github.com:khushilmindia-web/AI-Legal-chatbot.git`.

## Repo Update Note

Git is now installed locally.
The remaining step for full automation is GitHub authentication and remote setup, after which this file should be updated on every meaningful backend, frontend, corpus, or ingestion change.
