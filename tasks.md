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
| 11 | Debug India Kanoon-only chatbot flow to resolve infinite loading and ensure proper API response handling | Completed | Added bounded frontend request timeouts, backend India Kanoon response/status/body logging, safer exception handling, and guaranteed fallback JSON/message handling so `/chat` and `/chat/upload` now resolve with “No relevant legal data found on India Kanoon” instead of hanging when India Kanoon fails or returns empty data |
| 12 | Fix frontend busy-state crash caused by null DOM selector in `setBusy()` and ensure chat loader always clears | Completed | Restored the missing `statusText` node in both frontend entrypoints and made `Frontend/app.js` busy/error/event handling null-safe so the chat loader clears reliably on success, error, timeout, and fallback paths without crashing on missing selectors |
| 13 | Fix India Kanoon statute-query retrieval so basic lookups like `Section 420 IPC` do not get lost behind state-only filtering | Completed | Updated chat search to classify statute/IPC-style prompts as criminal, try normalized query variants plus broader doctypes before state-specific filters, added multi-attempt result deduping in `backend/app/services/indiankanoon_service.py`, and verified the new path with direct Python smoke checks because pytest cleanup is still blocked by the repo's Windows temp-permission issue |
| 14 | Restore intent-first routing for greetings and simple handled intents, and fix India Kanoon live-request failures caused by the broken proxy environment | Completed | Added a new intent-routing layer ahead of India Kanoon in the active v2 chat flow, restored the missing `data/Json/Intents.json` asset expected by the trained intent classes, and hardened India Kanoon requests to ignore the broken local `127.0.0.1:9` proxy env by default so valid legal queries can reach the API instead of falling straight to the fallback |
| 15 | Sync the active intent-routing layer with the updated `data/Json/Intents.json`, refresh intent model assets, and stop non-legal/API-topic prompts from falling into India Kanoon fallback | Completed | Updated the active intent service to read the current JSON patterns/responses instead of stale hardcoded shortcuts, retrained the intent model artifacts from the edited intents file, hardened training/runtime tokenization against missing local NLTK corpora, and added a non-legal technical-topic guard so API/helpdesk-style prompts no longer route into India Kanoon and trigger the legal-data fallback |
| 16 | Add `legal_help` / `procedural` intent routing for process-based legal questions and restrict India Kanoon to case-law / section / judgment queries with procedural fallback on API misses | Completed | Added new `legal_help` and `procedural` intents in `data/Json/Intents.json`, retrained the intent model, updated routing so process-based legal questions go to a general legal-help handler instead of India Kanoon, restricted India Kanoon usage to case-law / section / judgment style queries, and made empty or failed India Kanoon lookups fall back to procedural legal guidance before ever returning the old no-data response |
| 17 | Preserve chat conversation state across turns so legal-help follow-ups continue instead of resetting into greeting or India Kanoon fallback | Completed | Added session-backed `active_intent` / `awaiting_details` state in chat sessions, disabled greeting reuse after the conversation starts, treated substantive issue statements as legal-help instead of greeting, and verified with a direct FastAPI smoke harness because pytest cleanup is still blocked by the repo's Windows `.pytest_tmp` permission issue |
| 18 | Make `legal_help` follow-up replies use newly provided details to advance the conversation instead of repeating generic templates | Completed | Added structured procedural slots and staged legal-help guidance so fraud/UPI prompts now return immediate actionable advice and replies like `Ahmedabad city` advance the police-complaint flow instead of replaying the same generic template; verified with a direct FastAPI smoke harness because pytest cleanup is still blocked by the repo's Windows `.pytest_tmp` permission issue |
| 19 | Harden intent routing so legal issue prompts cannot be misclassified as greeting or welcome responses | Completed | Reordered intent precedence so fraud/UPI/scam/bank-debit and other substantive legal-help signals win before greeting, added stricter greeting acceptance rules, and verified with direct Python and FastAPI smoke checks that issue statements no longer receive welcome responses |
