# Lawyer AI MVP

Lawyer AI is a production-style MVP for Indian legal guidance. The active runtime is a FastAPI app with authenticated chat sessions, India Kanoon grounded retrieval, upload-aware grounding, and structured legal response formatting.

This version is intentionally simple:

- no custom model training
- no heavy vector database pipeline
- no giant corpus ingestion requirement before the app becomes useful
- retrieval is modular so you can later upgrade to OpenAI-hosted file search, hybrid retrieval, or a custom RAG stack

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
      session_store.py
      openai_service.py
    utils/
      request_context.py
    main.py
frontend/
  index.html
  style.css
  app.js
knowledge/
tests/
.env.example
requirements.txt
README.md
```

## Features

- FastAPI routes for auth, chat, health, and debug status
- Cookie-based authenticated sessions for chat history and account access
- shared chat orchestration in one service layer
- blank chat history until the first real user message
- old chats reopen properly
- clear history deletes all local sessions and messages
- India Kanoon grounded retrieval with structured answer generation
- upload support for text, PDF, and image files
- safe OpenAI wrapper with timeout, retry, and graceful fallback
- stateful multi-turn follow-up handling across legal-help and grounded-RAG flows
- structured logging with request IDs

## Setup

1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env`.
4. Add your OpenAI API key.
5. Start the FastAPI server.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`.

## Run

Start the backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload
```

Open the frontend in the browser:

- `http://127.0.0.1:5000/frontend/index.html`

## Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `INDIANKANOON_API_TOKEN` (or legacy alias `INDIA_KANOON_API_TOKEN`)
- `INDIANKANOON_API_BASE_URL`
- `INDIANKANOON_TIMEOUT_SECONDS`
- `DEBUG`
- `DEFAULT_STATE`
- `CORS_ALLOW_ORIGINS`
- `AUTH_COOKIE_NAME`
- `AUTH_SESSION_DURATION_DAYS`
- `APP_BASE_URL`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`
- `SMTP_FROM_NAME`
- `SMTP_USE_TLS`
- `PASSWORD_RESET_TOKEN_TTL_MINUTES`

## Chat Flow

1. The page opens with a blank chat area and blank history.
2. No chat session is created on load.
3. The first user message triggers `POST /chat` or `POST /chat/upload`.
4. The backend creates a session only at that point.
5. That session then appears in the chat history sidebar.
6. Clicking an old session reopens the stored messages.
7. `New Chat` resets the UI to a fresh draft.
8. `Clear History` deletes all sessions and messages from SQLite.

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

## Testing

Run:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Covered basics:

- health endpoint
- debug endpoint
- first-message session creation
- reopening old chat
- clear history
- upload endpoint
- extraction helper
- session store behavior

## Future Upgrade Path

This MVP is designed to upgrade cleanly to:

- OpenAI-hosted file search or vector stores
- curated case-law ingestion
- better citation attribution per source chunk
- multilingual UX
- Indian State-specific expansion
- advocate referral or human escalation

To move beyond the MVP later, you can keep the same API surface and extend the active grounded path in `backend/app/services/chat_service.py` and `backend/app/services/indiankanoon_service.py`.
