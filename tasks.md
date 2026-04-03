# Task Tracker

Use this file as the single source of truth for manual task entry and sequential execution.

## Rules

1. Add tasks in the order they should be executed.
2. Every new task must start with status `Pending`.
3. Only one task may be in `In Progress` at a time.
4. A task can start only when all previous tasks are `Completed`.
5. Update task status as it progresses: `Pending` -> `In Progress` -> `Completed` or `Blocked`.
6. Do not proceed to the next task unless explicitly instructed.
7. Do not skip or reorder tasks.
8. If a task cannot proceed, mark it `Blocked` with a short reason and stop.
9. Always prioritize pending tasks in `tasks.md` over newly assigned tasks.
10. Always read and update `tasks.md` before starting any newly assigned task.
11. If a task is blocked, attempt resolution for up to 3 minutes before marking it `Blocked`.
12. Retry a `Blocked` task up to 3 times only.
13. If still blocked after 3 attempts, leave it with a clear reason and do not retry unless conditions change.

## Status Legend

- `Pending`: queued and not started yet
- `In Progress`: currently being worked on
- `Completed`: finished
- `Blocked`: cannot continue until an issue is resolved

## Task List

| Order | Task | Status | Notes |
| --- | --- | --- | --- |
| 1 | Scaffold Indian Kanoon integration config and backend service without enabling it in live chat flow | Completed | Added config placeholders and `backend/app/services/indiankanoon_service.py` |
| 2 | Add RSA public-private key generation and request signing utilities for API integration | Completed | Added generator script, signing utility, env placeholders, and usage docs |
| 3 | Wire `INDIAKANOON_API_TOKEN` from `.env` into the chatbot backend and route user legal queries through the Indian Kanoon API to generate concise, relevant, low-token responses | Completed | Token alias loads from `.env`, Indian Kanoon retrieval is active when needed, and concise grounded source data feeds answer generation |
| 4 | Verify database connection and ensure stable connectivity with successful read/write operations | Completed | Verified active SQLite path, user lookup, session write/read, and cleanup on `data/lawyer_ai.db` |
| 5 | Verify SMTP setup and confirm successful email delivery using test email | Completed | Current Gmail-backed SMTP config sent a reset email successfully through the backend mailer |
| 6 | Validate "Forgot Password" button functionality in `index.html` including event binding and API trigger | Completed | Forgot-password UI is served from `Frontend/auth.html`; link, modal forms, event bindings, and API targets are present and live |
| 7 | Test complete forgot-password workflow (frontend -> backend -> SMTP -> email delivery) | Completed | Verified signup, reset request, reset link/token capture, password reset confirm, old-password rejection, new-password login, and reused-token rejection |
| 8 | Resolve the rogue local listener on `127.0.0.1:8000` so the browser cannot hit the wrong backend | Blocked | PID `9744` can be terminated but immediately respawns and reclaims `8000`; this appears to be an external auto-restarting process outside the repo |
| 9 | Migrate the project frontend and backend to port `5000` and ensure all internal connections use only `127.0.0.1:5000` with no remaining dependency on any other project port | Completed | Active runtime, env defaults, startup scripts, README, and legacy fallback entrypoints now point to `127.0.0.1:5000`; live `/health`, `/frontend/auth.html`, and protected `/frontend/index.html` checks passed |
