# Changelog

## 2026-04-02

### Added
- Added Git automation helpers in `scripts/git_sync.ps1` and `scripts/update_changelog.ps1` to support repeatable local commit/push and changelog maintenance workflows.
- Added runtime auth routes in `backend/app/api/routes/auth.py` for signup, login, logout, current-user lookup, password reset messaging, Google OAuth status, login, and callback.
- Added shared auth helpers in `backend/app/api/auth_utils.py` for bearer-token parsing, cookie lookup, current-user enforcement, and auth cookie management.
- Added auth request/response schemas in `backend/app/models/schemas.py`.
- Added a lowercase `Frontend/index.html` alias that forwards to the existing app entrypoint.
- Added auth regression coverage in `tests/test_auth_flow.py`.

### Changed
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
- Verified GitHub sync is working: the repository now pushes successfully to `git@github.com:khushilmindia-web/AI-Legal-chatbot.git` and `main` tracks `origin/main`.
- Verified signup, login, `/auth/me`, protected root redirect, protected frontend access, per-user chat flow, history loading, logout, and mocked Google OAuth callback/session persistence using a direct FastAPI `TestClient` integration harness.

### Pending
- Run the full `pytest` suite cleanly once the local Windows temp-directory permission issue no longer interferes with pytest cleanup.
- Perform a live browser roundtrip against real Google OAuth credentials after deployment/runtime environment confirmation.

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
