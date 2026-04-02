# Lawyer AI MVP

Lawyer AI is a production-style MVP for Indian legal guidance. It uses a simple FastAPI backend, OpenAI Responses API, lightweight local knowledge files, upload-based grounding, and a ChatGPT-like frontend.

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
      openai_service.py
      retrieval.py
      session_store.py
    utils/
      request_context.py
    main.py
  main.py
frontend/
  index.html
  Index.html
  style.css
  app.js
knowledge/
tests/
.env.example
requirements.txt
README.md
```

## Features

- FastAPI routes for `POST /chat`, `POST /chat/upload`, `GET /health`, `GET /debug/status`
- shared chat orchestration in one service layer
- blank chat history until the first real user message
- old chats reopen properly
- clear history deletes all local sessions and messages
- simple legal grounding using curated files in `knowledge/`
- upload support for text, PDF, and image files
- safe OpenAI wrapper with timeout, retry, and graceful fallback
- state-aware guidance without forcing unnecessary gating questions
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
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the frontend in the browser:

- `http://127.0.0.1:8000/frontend/index.html`

## Environment Variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `MAX_FILE_SIZE_MB`
- `DEBUG`
- `DEFAULT_STATE`
- `RETRIEVAL_MODE`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_RETRIES`
- `CORS_ALLOW_ORIGINS`
- `OPENAI_VECTOR_STORE_ID`

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

To move beyond the MVP later, you can keep the same API surface and swap out the retrieval provider behind `backend/app/services/retrieval.py`.
