# Lawyer AI

Lawyer AI is a full-stack legal-tech demo that helps users ask Indian law questions in plain language, upload supporting documents, and continue multi-turn guidance in a secure authenticated workspace.

The project is built to show production-oriented engineering rather than a one-off chatbot: it has FastAPI APIs, MongoDB-backed auth and chat persistence, password reset email delivery, grounded retrieval, upload-aware answers, request timeouts, health checks, and deployment notes for common hosting platforms.

## Feature Highlights

- Secure signup, login, logout, protected chat workspace, and password reset emails.
- Per-user chat sessions with history, reopen, single-chat delete, and full-history clear.
- Grounded legal answers using local datasets, India Kanoon retrieval, and optional trusted-source search.
- Upload-aware chat flow for text, PDF, and image files, with extraction warnings handled gracefully.
- Stateful follow-up behavior that remembers the current legal issue and asks one useful next question at a time.
- Responsive frontend with loading indicators, friendly API timeout errors, and mobile-friendly layout.
- Deployment-ready FastAPI entrypoint, `/health` endpoint, `.env.example`, and production checklist.

## Tech Stack

- Backend: FastAPI, Pydantic, Uvicorn
- Frontend: HTML, CSS, vanilla JavaScript
- Database: MongoDB active runtime, SQLite retained only as migration/backup reference
- AI and retrieval: OpenAI API wrapper, India Kanoon retrieval, local legal datasets, optional Google Custom Search
- Auth and email: HttpOnly cookies, bearer token compatibility, SMTP password reset email
- Testing: Pytest, FastAPI TestClient
- Deployment targets: Render, Railway, VPS, Docker-compatible ASGI hosting

## Architecture Overview

```text
Browser UI
  |-- auth.html / auth.js: signup, login, Google OAuth redirect, password reset
  |-- index.html / app.js: protected chat, uploads, history, delete flows
        |
        v
FastAPI app: backend.main:app
  |-- auth routes: sessions, cookies, OAuth, password reset
  |-- chat routes: chat, upload, history, messages, deletion
  |-- health/debug routes
        |
        v
Service layer
  |-- ChatService: routing, follow-up state, grounded response assembly
  |-- FileExtractionService: PDF/text/image/url extraction
  |-- Mailer: SMTP HTML + plain-text reset emails
  |-- Storage boundary: MongoSessionStore
        |
        v
External and data sources
  |-- MongoDB for users, sessions, reset tokens, chat sessions, messages
  |-- Local legal datasets and knowledge corpus
  |-- India Kanoon API
  |-- OpenAI API
  |-- Optional Google Custom Search
```

## How It Works

1. A user signs up or logs in, then enters the protected chat workspace.
2. The first message creates a chat session only when there is real user input.
3. The backend classifies the turn as practical guidance, provision lookup, research, or uploaded-document analysis.
4. For grounded answers, the app gathers context from local legal datasets, uploaded files, India Kanoon, and optional trusted search.
5. The answer is generated from the grounded context with citations, warnings, likely forum, caution notes, and documents to keep.
6. Follow-up state is stored with the chat so later turns can continue the same legal issue instead of restarting.
7. Users can reopen, delete, or clear their chat history.

## Demo Queries

Try these in a video demo or local walkthrough:

- `I clicked a fake UPI link and money got debited. What should I do first?`
- `Section 420 IPC punishment`
- `Explain Article 21 in simple words`
- `My landlord is threatening to lock me out. What can I do?`
- `I received a cheque bounce notice. What should I check before replying?`
- Upload a notice or payment record, then ask: `Please review this document and tell me the immediate next steps.`

## Screenshots And Demo

Placeholders for release screenshots:

- Auth screen: `docs/screenshots/auth.png`
- Chat workspace: `docs/screenshots/chat.png`
- Upload-grounded answer: `docs/screenshots/upload-answer.png`
- Mobile layout: `docs/screenshots/mobile.png`
- Demo video: add a GitHub release asset or portfolio link here when available.

## Project Structure

```text
backend/
  app/
    api/
      routes/
        auth.py
        chat.py
        debug.py
        health.py
    core/
      config.py
      logging.py
      prompts/
        lawyer_ai_system.md
    models/
      schemas.py
    services/
      chat_service.py
      file_extractor.py
      indiankanoon_service.py
      intent_service.py
      mongo_session_store.py
      session_store.py
      storage.py
      openai_service.py
    utils/
      request_context.py
    main.py
Frontend/
  auth.html
  Index.html
  style.css
  app.js
  auth.js
knowledge/
tests/
.env.example
requirements.txt
README.md
```

## Setup

Prerequisites:

- Python 3.11+
- MongoDB running locally or reachable through `MONGODB_URI`
- OpenAI API key
- Optional: India Kanoon API token, SMTP account, Google OAuth credentials

Steps:

1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env`.
4. Add required secrets.
5. Start MongoDB.
6. Start the FastAPI server.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and set at least `OPENAI_API_KEY`.

MongoDB is the active runtime database. The default local configuration is:

```env
DATABASE_BACKEND=mongodb
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DATABASE=lawyer_ai
```

## Run

Start the backend locally:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload
```

Open the frontend in the browser:

- `http://127.0.0.1:5000/frontend/auth.html`

After login, the app redirects to:

- `http://127.0.0.1:5000/frontend/index.html`

## Deployment

Production entrypoint:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-5000}
```

Use `backend.main:app` for Render, Railway, VPS systemd services, Docker, or any ASGI host. Do not use `--reload` outside local development.

### Render

- Build command: `pip install -r requirements.txt`
- Start command: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Set `APP_BASE_URL` to the Render public URL, for example `https://your-service.onrender.com`.
- Add the same origin to `CORS_ALLOW_ORIGINS`.
- Use a managed MongoDB URI for `MONGODB_URI`; do not rely on localhost MongoDB.

### Railway

- Start command: `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Set `APP_BASE_URL` to the Railway public domain.
- Add the Railway public domain to `CORS_ALLOW_ORIGINS`.
- Provision MongoDB or point `MONGODB_URI` to an external managed MongoDB instance.

### VPS

1. Install Python, MongoDB or configure a managed MongoDB URI, and a process manager such as `systemd`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Put production environment variables in a service env file.
4. Run `python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000` behind Nginx or Caddy.
5. Terminate HTTPS at the reverse proxy and forward `X-Forwarded-Proto: https`.
6. Set `APP_BASE_URL` to the HTTPS public origin and include that origin in `CORS_ALLOW_ORIGINS`.

### Production Environment Checklist

- `DEBUG=false`.
- `APP_BASE_URL` is the exact public HTTPS origin users open in the browser.
- `CORS_ALLOW_ORIGINS` includes only trusted frontend origins, including `APP_BASE_URL`.
- `AUTH_COOKIE_NAME=session_token` or another stable cookie name.
- `AUTH_SESSION_DURATION_DAYS` is set intentionally.
- `MONGODB_URI` points to production MongoDB.
- `OPENAI_API_KEY` is set and `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` are reasonable.
- `SMTP_HOST`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `SMTP_FROM_EMAIL` are set if password reset emails should work.
- `INDIANKANOON_API_TOKEN` is set if live India Kanoon retrieval is required.
- `GOOGLE_SEARCH_ENABLED=false` unless Google Custom Search credentials are configured.
- If Google Search is enabled, set `GOOGLE_CUSTOM_SEARCH_API_KEY`, `GOOGLE_CUSTOM_SEARCH_CX`, and `GOOGLE_SEARCH_TRUSTED_DOMAINS`.
- If Google OAuth is enabled, set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and add `${APP_BASE_URL}/auth/google/callback` to Google Cloud Console authorized redirect URIs.

### Health Check

Verify deployment health after startup:

```bash
curl -fsS "$APP_BASE_URL/health"
```

Expected JSON:

```json
{"status":"ok"}
```

Also verify protected frontend path handling:

```bash
curl -I "$APP_BASE_URL/"
curl -I "$APP_BASE_URL/frontend/auth.html"
curl -I "$APP_BASE_URL/frontend/index.html"
```

Unauthenticated users should be redirected to `/frontend/auth.html` for the protected app page.

## Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `DATABASE_BACKEND` (`mongodb` is active; SQLite is temporarily disabled during migration)
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `INDIANKANOON_API_TOKEN` (or legacy alias `INDIA_KANOON_API_TOKEN`)
- `INDIANKANOON_API_BASE_URL`
- `INDIANKANOON_TIMEOUT_SECONDS`
- `DEBUG`
- `DEFAULT_STATE`
- `CORS_ALLOW_ORIGINS`
- `AUTH_COOKIE_NAME`
- `AUTH_SESSION_DURATION_DAYS`
- `FRONTEND_AUTH_PATH`
- `FRONTEND_APP_PATH`
- `APP_BASE_URL`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `SMTP_FROM_NAME`
- `SMTP_USE_TLS`
- `PASSWORD_RESET_TOKEN_TTL_MINUTES`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI` (kept for compatibility; `APP_BASE_URL` determines the effective callback URL)
- `GOOGLE_SEARCH_ENABLED`
- `GOOGLE_CUSTOM_SEARCH_API_KEY`
- `GOOGLE_CUSTOM_SEARCH_CX`
- `GOOGLE_SEARCH_TIMEOUT_SECONDS`
- `GOOGLE_SEARCH_MAX_RESULTS`
- `GOOGLE_SEARCH_CACHE_TTL_SECONDS`
- `GOOGLE_SEARCH_TRUSTED_DOMAINS`

## Google OAuth Setup

If email login works but Google login fails, the usual cause is a redirect URI mismatch.

Set these values to the exact origin you open in the browser:

- `APP_BASE_URL`
- `CORS_ALLOW_ORIGINS`

The backend derives the effective Google callback as `${APP_BASE_URL}/auth/google/callback`. Add that exact URL to Google Cloud Console.

Example for production:

```env
APP_BASE_URL=https://your-domain.example
CORS_ALLOW_ORIGINS=https://your-domain.example
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

In Google Cloud Console, the OAuth client must also include the same exact values:

- Authorized JavaScript origins: `https://your-domain.example`
- Authorized redirect URIs: `https://your-domain.example/auth/google/callback`

Do not mix `127.0.0.1`, `localhost`, staging domains, and production domains across these settings. The browser URL, backend config, and Google Console redirect URI must all match.

## Chat Flow

1. The page opens with a blank chat area and blank history.
2. No chat session is created on load.
3. The first user message triggers `POST /chat` or `POST /chat/upload`.
4. The backend creates a session only at that point.
5. That session then appears in the chat history sidebar.
6. Clicking an old session reopens the stored messages.
7. `New Chat` resets the UI to a fresh draft.
8. `Clear History` deletes all sessions and messages from MongoDB.

## Database Migration And Backup

MongoDB is the active storage layer for runtime auth, sessions, password resets, chat sessions, and chat messages. The legacy SQLite file remains at `data/lawyer_ai.db` as a backup and must not be deleted during the migration window.

Dry-run the one-time migration without writing to MongoDB:

```powershell
python scripts/migrate_sqlite_to_mongo.py --dry-run
```

Run the migration:

```powershell
python scripts/migrate_sqlite_to_mongo.py
```

The script upserts by preserved integer `id`, converts SQLite JSON columns into Mongo fields, syncs Mongo counters, and prints validation counts for `users`, `auth_sessions`, `password_reset_tokens`, `chat_sessions`, and `chat_messages`.

Rollback note: SQLite code is preserved in `backend/app/services/session_store.py`, but app startup intentionally rejects SQLite with `SQLite temporarily disabled during MongoDB migration`. To roll back, first preserve the current MongoDB data/export, then explicitly re-enable the SQLite branch in `backend/app/services/storage.py` and set `DATABASE_BACKEND=sqlite`. Keep `data/lawyer_ai.db` unchanged as the rollback source.

## Upload Flow

`POST /chat/upload` accepts:

- `message`
- `state`
- `district`
- `case_stage`
- `is_own_matter`
- `chat_id`
- `files`
- `image_urls`

Extraction strategy:

- PDF with `pdfplumber` first, fallback `PyPDF2`
- text files directly
- image OCR only as fallback through `Pillow` and `pytesseract`
- optional URL text fetch for simple text/html sources

If one upload fails, the others continue and warnings are returned safely.

## How the Assistant Behaves

- helps directly when the issue is already clear
- asks only one follow-up question at a time
- keeps replies concise and practical
- uses simple Indian English
- avoids claiming to be legal counsel of record
- includes a disclaimer that this is general legal information only

For example, a message like:

`I clicked a fake UPI link and money got debited.`

should receive immediate practical guidance first, not a repetitive category-selection loop.

## Limitations / Disclaimer

- Lawyer AI provides general legal information, not legal advice or representation.
- Outputs may be incomplete, outdated, or unsuitable for urgent facts unless verified against current law and local procedure.
- Users should consult a qualified advocate before acting on high-stakes legal, financial, criminal, or deadline-sensitive matters.
- Retrieval depends on available local data, configured API keys, provider uptime, and source quality.
- Uploaded documents are processed for chat context; do not upload sensitive documents unless the deployment and storage environment are trusted.

## Testing

Run:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Covered basics:

- health endpoint
- debug endpoint
- signup, login, logout, and protected route redirects
- forgot password request, reset token flow, and login with the new password
- first-message session creation and history
- follow-up state persistence
- reopening old chat
- text upload grounded response
- single-chat deletion and full-history deletion

For a focused stabilization check:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_auth_flow.py tests\test_upload.py tests\test_chat.py::test_clear_history_removes_all_chats tests\test_chat.py::test_delete_single_chat_removes_only_selected_session -q
```

## Future Upgrade Path

This MVP is designed to upgrade cleanly to:

- OpenAI-hosted file search or vector stores
- curated case-law ingestion
- better citation attribution per source chunk
- multilingual UX
- Indian State-specific expansion
- advocate referral or human escalation

To move beyond the MVP later, you can keep the same API surface and extend the active grounded path in `backend/app/services/chat_service.py` and `backend/app/services/indiankanoon_service.py`.
