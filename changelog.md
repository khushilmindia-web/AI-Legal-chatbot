# Changelog

## 2026-04-06

### Changed
- Hardened [`backend/app/services/intent_service.py`](d:/AI-Chatbot/backend/app/services/intent_service.py) so intent routing now applies strict priority ordering: technical-topic guard first, then case-law guard, then high-priority legal-help issue detection for fraud/UPI/scam/bank-debit and similar legal issue statements, with greeting matching only after substantive legal-help checks have been exhausted.
- Added stricter greeting acceptance rules in [`backend/app/services/intent_service.py`](d:/AI-Chatbot/backend/app/services/intent_service.py) so a `greeting` prediction or pattern match is rejected when the same message already contains substantive legal-help markers, and generic welcome replies are now limited to pure greeting-only messages such as `hello`.
- Updated [`tests/test_intent_service.py`](d:/AI-Chatbot/tests/test_intent_service.py) and [`tests/test_chat.py`](d:/AI-Chatbot/tests/test_chat.py) with regressions for `I clicked a fake UPI link and money got debited`, `bank fraud complaint`, and clean `hello` routing so the legal-help path wins before greeting whenever a real issue is already present.

### Verified
- Verified directly in Python that `I clicked a fake UPI link and money got debited`, `bank fraud complaint`, and `fake link scam` now route to `legal_help`, while `hello` still routes to `greeting`.
- Verified with an in-process FastAPI `TestClient` smoke harness that the `/chat` flow now returns cyber-fraud guidance for `I clicked a fake UPI link and money got debited` and does not return a welcome response, while a separate clean `hello` message still receives the greeting path.

### Pending
- Full pytest execution is still blocked by the repository's existing Windows `.pytest_tmp` permission cleanup issue, so this stricter intent-priority behavior was verified through direct Python and FastAPI smoke checks instead of a clean pytest pass.

## 2026-04-06

### Changed
- Expanded [`backend/app/models/schemas.py`](d:/AI-Chatbot/backend/app/models/schemas.py) conversation state to persist structured procedural slots such as `issue_type`, `city`, `police_station`, `bank_name`, `platform`, `notice_stage`, `document_type`, and the last guidance key so legal-help follow-ups can move forward instead of replaying the same generic answer.
- Updated [`backend/app/services/chat_service.py`](d:/AI-Chatbot/backend/app/services/chat_service.py) with a staged legal-help workflow that extracts user-provided follow-up details, stores them in conversation state, and generates the next practical step for fraud, police complaint, notice, consumer, and document flows.
- Changed the legal-help handling so fraud/UPI prompts like `I clicked a fake UPI link` now return direct cyber-fraud action guidance immediately, while follow-up replies such as `Ahmedabad city` advance the police-complaint advice with location-specific next steps instead of repeating the earlier template.
- Updated [`tests/test_chat.py`](d:/AI-Chatbot/tests/test_chat.py) and [`tests/test_session_store.py`](d:/AI-Chatbot/tests/test_session_store.py) to lock in direct fraud guidance, structured session-state persistence, and non-repetitive city-based police complaint follow-up behavior.

### Verified
- Verified directly with an in-process FastAPI `TestClient` smoke harness that `I clicked a fake UPI link and money got debited from my account.` now returns actionable fraud guidance without India Kanoon fallback or a repeated generic prompt.
- Verified directly with the same smoke harness that `How do I file a police complaint for threats?` asks for the city once, and the follow-up `Ahmedabad city` produces the next specific police-station guidance instead of repeating the previous response.

### Pending
- Normal `pytest` remains blocked by the repository's existing Windows `.pytest_tmp` permission cleanup failure, so this staged legal-help follow-up behavior was verified through direct Python/TestClient smoke checks instead of a clean pytest pass.

## 2026-04-06

### Changed
- Added session-backed conversation state in [`backend/app/services/session_store.py`](d:/AI-Chatbot/backend/app/services/session_store.py) and a typed internal state model in [`backend/app/models/schemas.py`](d:/AI-Chatbot/backend/app/models/schemas.py) so each chat can persist `conversation_started`, `active_intent`, `awaiting_details`, the last issue summary, and the last follow-up prompt across turns.
- Updated [`backend/app/services/chat_service.py`](d:/AI-Chatbot/backend/app/services/chat_service.py) so ongoing `legal_help` chats no longer re-run the conversation through greeting or India Kanoon routing on every turn; instead, follow-up messages are merged into the existing issue context and answered with continuing advice, rights, and next steps.
- Updated [`backend/app/services/intent_service.py`](d:/AI-Chatbot/backend/app/services/intent_service.py) so greeting / thanks / goodbye intents are disabled once a conversation has already started, and substantive issue statements such as `My landlord is harassing me` are routed into `legal_help` before the intent model can misclassify them as greeting.
- Updated [`tests/test_chat.py`](d:/AI-Chatbot/tests/test_chat.py), [`tests/test_intent_service.py`](d:/AI-Chatbot/tests/test_intent_service.py), and [`tests/test_session_store.py`](d:/AI-Chatbot/tests/test_session_store.py) with regressions for cross-turn legal-help continuity, greeting suppression after conversation start, issue-statement routing, and session-state persistence.

### Verified
- Verified directly with an in-process FastAPI `TestClient` smoke harness that `My landlord is harassing me` now enters the `legal_help` flow, the next message `He keeps threatening to evict me without notice.` stays in the same legal-help conversation, and a later `hello` in the same chat no longer resets the assistant back to greeting.
- Verified directly in Python that the intent router now classifies `My landlord is harassing me` as `legal_help`, keeps `Section 420 IPC` out of intent shortcuts, and still routes `api token issue` to the technical handler rather than India Kanoon.

### Pending
- `pytest tests/test_session_store.py tests/test_intent_service.py tests/test_chat.py -q` still hits the repository's existing Windows `.pytest_tmp` permission cleanup failure, so the new stateful-conversation behavior was verified with direct Python/TestClient smoke checks instead of a clean pytest pass.

## 2026-04-06

### Changed
- Added new `legal_help` and `procedural` intents to `data/Json/Intents.json`, then retrained `backend/Model/Intents/Legalchatbot.h5` so the updated intent model recognizes common process-based legal questions such as FIR, cyber fraud complaints, consumer complaints, police complaints, bank fraud, notices, document process, and `what should I do` prompts.
- Updated `backend/app/services/intent_service.py` so procedural legal questions are classified into the general legal-help path, while India Kanoon is now reserved for authority-lookup questions such as sections, articles, judgments, citations, and case-law requests.
- Updated `backend/app/services/chat_service.py` with a dedicated legal-help response builder for process-based legal questions and changed the chat flow so an empty or failed India Kanoon lookup now falls back to general legal-help guidance before ever showing the old no-data message.
- Updated `tests/test_chat.py` and `tests/test_intent_service.py` to cover procedural routing, India Kanoon miss-to-legal-help fallback behavior, and the new legal-help intent path.

### Verified
- Verified directly in Python that `how do i file a cyber fraud complaint`, `what should i do after bank fraud`, and `consumer complaint process` now route to the legal-help path instead of India Kanoon.
- Verified directly in Python that `Section 420 IPC judgment` still bypasses procedural routing and remains eligible for the India Kanoon authority-search path.
- Verified the intent retraining command completed successfully after adding the new procedural intents.

### Pending
- Full pytest execution remains blocked by the repository's existing Windows `.pytest_tmp` permission issue during setup/cleanup, so the new routing behavior was verified through direct Python checks instead of pytest.

## 2026-04-06

### Changed
- Updated `backend/app/services/intent_service.py` so the active intent-first routing now reads exact patterns and responses directly from the current `data/Json/Intents.json` instead of relying on stale hardcoded greeting/advice text.
- Updated `backend/Intent_Bot/Train_Model.py` and `backend/Intent_Bot/Intent_bot.py` to tolerate missing local NLTK corpora by falling back to simple tokenization/lemmatization, then retrained `backend/Model/Intents/Legalchatbot.h5` from the edited `data/Json/Intents.json`.
- Added a technical-topic guard in `backend/app/services/intent_service.py` so API/helpdesk-style prompts such as `api token issue` are handled before India Kanoon and no longer collapse into the legal fallback message.
- Added `tests/test_intent_service.py` coverage for updated-intent pattern routing, API-topic handling, and legal-query fallthrough behavior.

### Verified
- Verified directly in Python that `hi` and `Can you help me with legal advice?` now return responses from the updated `data/Json/Intents.json` pattern set.
- Verified directly in Python that `api token issue` is now handled by the intent-routing layer instead of falling through to India Kanoon fallback.
- Verified directly in Python that `Section 420 IPC` still bypasses intent shortcuts and remains eligible for the legal-search path.
- Verified that the intent retraining command completed successfully with the updated intents file and refreshed `backend/Model/Intents/Legalchatbot.h5`.

### Pending
- Full pytest execution remains blocked by the repository's existing Windows `.pytest_tmp` permission issue during setup/cleanup, so the added intent regressions were verified through direct Python execution instead of pytest.

## 2026-04-06

### Changed
- Added `backend/app/services/intent_service.py` and updated `backend/app/services/chat_service.py` so the active v2 chat flow now handles greetings, thanks, goodbye, and simple generic help intents before calling India Kanoon, while detailed legal queries fall through directly to the legal-search path.
- Added the missing `data/Json/Intents.json` metadata file so the existing trained intent classes (`advice`, `goodbye`, `greeting`, `thanks`) have matching tag/response definitions when the optional intent bot model is available.
- Updated `backend/app/services/indiankanoon_service.py` to use a dedicated `requests.Session` and ignore broken environment proxy variables by default, and added `INDIANKANOON_TRUST_ENV_PROXY` in `backend/app/core/config.py` plus `.env.example` so India Kanoon requests are no longer forced through the machine's bad `127.0.0.1:9` proxy unless explicitly enabled.
- Updated `tests/test_chat.py` and `tests/test_indiankanoon_integration.py` with regression coverage for intent-first greeting routing and the default India Kanoon proxy-bypass behavior.

### Verified
- Verified directly in Python that `IntentRoutingService` now returns an immediate response for `hello`, `thank you`, and `bye`, while `Section 420 IPC` bypasses intent handling and falls through to the legal-search path.
- Verified directly in Python that `IndianKanoonService` now initializes with `session.trust_env == False` by default, which bypasses the broken `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` values currently set to `http://127.0.0.1:9` in this environment.
- Verified directly in Python that the chat service can build the intent-first response payload without calling India Kanoon when a greeting intent is handled.

### Pending
- `pytest tests/test_chat.py -q` and `pytest tests/test_indiankanoon_integration.py -q` remain blocked by the repository's existing Windows `.pytest_tmp` permission issue during setup/cleanup, so full pytest verification is still deferred until that environment problem is cleared.

## 2026-04-06

### Changed
- Updated `backend/app/services/chat_service.py` so statute-style prompts such as `Section 420 IPC` are classified as criminal queries, generate normalized search variants, and try broader India Kanoon doctypes like `judgments,laws` before falling back to state-specific doctypes.
- Added `backend/app/services/indiankanoon_service.py` `search_references_multi()` support so the chat flow can retry India Kanoon searches across multiple query/doctypes combinations while deduplicating repeated document hits.
- Updated `tests/conftest.py` and `tests/test_chat.py` to match the new multi-attempt India Kanoon search path and added a regression test that locks in the broader-search behavior for `Section 420 IPC`.

### Verified
- Verified directly in Python that `ChatService` now classifies `Section 420 IPC` as `criminal`, builds normalized variants including `Indian Penal Code section 420`, and prefers broader doctypes before `gujarat`.
- Verified with a direct Python smoke harness that the multi-attempt India Kanoon search path stops on the first broader-match result and returns the expected statute hit for `Section 420 IPC`.

### Pending
- `pytest tests/test_chat.py -q` and `pytest tests/test_indiankanoon_integration.py -q` are still blocked by the repository's existing Windows `.pytest_tmp` permission issue during setup/cleanup, so pytest-based verification remains deferred until that environment problem is cleared.

## 2026-04-04

### Changed
- Restored the missing `statusText` element in `Frontend/index.html` and `Frontend/Index.html` so the active chat UI matches the selectors used by `Frontend/app.js`.
- Hardened `Frontend/app.js` with null-safe busy-state, error-banner, reset, render, and event-binding logic so a missing DOM node no longer crashes `setBusy()` or leaves the loader stuck on screen.
- Hardened `backend/app/services/indiankanoon_service.py` with explicit timeout/error logging, HTTP status/body preview logging, and invalid-JSON payload handling for India Kanoon API calls.
- Updated `backend/app/services/chat_service.py` so India Kanoon failures, empty results, and downstream persistence issues no longer bubble into a hanging chat request and instead return the final fallback answer `No relevant legal data found on India Kanoon`.
- Updated `Frontend/app.js` to use abortable request timeouts for chat requests and to always settle the loading state with either a normal assistant message or the India Kanoon fallback answer.
- Added regression coverage in `tests/test_chat.py` for the India Kanoon request-failure fallback path.
- Updated `backend/app/services/chat_service.py` so the active chat flow no longer uses OpenAI or local-knowledge answer generation and now returns chat responses strictly from India Kanoon search data.
- Added `backend/app/services/indiankanoon_service.py` `search_references()` support so chat can call India Kanoon `/search/` directly and format titles, snippets, court/source labels, and document links into the existing response schema.
- Updated the active chat/upload test fixtures in `tests/conftest.py`, `tests/test_chat.py`, `tests/test_chat_routing.py`, and `tests/test_upload.py` to reflect the India Kanoon-only behavior and lock in the exact fallback message `No relevant legal data found on India Kanoon`.

### Verified
- Verified with a lightweight frontend static check that both frontend entrypoints now include `#statusText` and that the guarded busy-state/event paths are present in `Frontend/app.js`.
- Verified with a direct FastAPI `TestClient` smoke harness that signup, `/chat`, `/chat/upload`, India Kanoon timeout failure, and fallback-response handling all complete successfully without hanging the UI request flow.
- Verified with a direct FastAPI `TestClient` smoke harness that signup, authenticated `/chat`, `/chat/upload`, no-result fallback handling, and `/chat/history` all work with India Kanoon-only answer generation on the active runtime stack.

### Pending
- The repository's existing Windows temp-directory permission issue still blocks normal `pytest` cleanup, so full pytest verification remains deferred until that environment problem is cleared.

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
