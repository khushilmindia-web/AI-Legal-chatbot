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


def test_session_store_persists_conversation_state(tmp_path):
    store = SessionStore(str(tmp_path / "store.db"))
    session = store.create_session("Stateful session")

    state = {
        "conversation_started": True,
        "active_intent": "legal_help",
        "awaiting_details": True,
        "interview_mode": True,
        "intake_stage": 2,
        "last_user_issue": "My landlord is harassing me",
        "current_intake_key": "record_status",
        "collected_facts": {"issue_summary": "My landlord is harassing me"},
        "issue_type": "police_complaint",
        "city": "Ahmedabad",
    }
    store.update_conversation_state(session["id"], state)

    saved = store.get_conversation_state(session["id"])

    assert saved["active_intent"] == "legal_help"
    assert saved["awaiting_details"] is True
    assert saved["interview_mode"] is True
    assert saved["intake_stage"] == 2
    assert saved["current_intake_key"] == "record_status"
    assert saved["collected_facts"]["issue_summary"] == "My landlord is harassing me"
    assert saved["issue_type"] == "police_complaint"
    assert saved["city"] == "Ahmedabad"
