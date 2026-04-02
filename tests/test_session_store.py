from __future__ import annotations

from backend.app.services.session_store import SessionStore


def test_session_store_hides_blank_chats(tmp_path):
    store = SessionStore(str(tmp_path / "store.db"))
    store.create_session("Draft session")
    second = store.create_session("Real session")
    store.add_message(second["id"], "user", "Hello")

    sessions = store.list_sessions()

    assert len(sessions) == 1
    assert sessions[0]["title"] == "Real session"
    assert sessions[0]["id"] == second["id"]
