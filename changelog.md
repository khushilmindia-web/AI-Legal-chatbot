# Changelog

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
