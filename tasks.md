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
14. If any command runs for more than 3 minutes, stop it and move forward with the next safe step.

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
| 8 | Migrate the project frontend and backend to port `5000` and ensure all internal connections use only `127.0.0.1:5000` with no remaining dependency on any other project port | Completed | Active runtime, env defaults, startup scripts, README, and legacy fallback entrypoints now point to `127.0.0.1:5000`; live `/health`, `/frontend/auth.html`, and protected `/frontend/index.html` checks passed |
| 9 | Fix the port `5000` bind/startup issues, align all startup paths to `127.0.0.1:5000`, and verify health/frontend reachability | Completed | Added the 3-minute command timeout rule, cleaned stale repo background processes, updated root startup to `127.0.0.1:5000`, and verified `/health` plus `/frontend/auth.html` on the active `5000` instance |
| 10 | Modify chatbot to fetch responses ONLY from India Kanoon API using token auth and disable/comment any LLM-based or other answer generation | Completed | Active `ChatService` now builds chat answers only from India Kanoon `/search/` results, returns the exact no-result fallback, preserves auth/history/upload flows, and was verified with a direct FastAPI `TestClient` smoke harness because pytest remains blocked by the repo's existing Windows temp-permission issue |
