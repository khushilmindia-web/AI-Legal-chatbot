from __future__ import annotations

from backend.app.services.intent_service import IntentRoutingService


def test_intent_service_uses_updated_json_pattern_response():
    service = IntentRoutingService()

    result = service.route("hi")

    assert result.handled is True
    assert result.intent == "greeting"
    assert "Legal India" in result.answer


def test_intent_service_handles_api_topics_without_indiankanoon_fallback():
    service = IntentRoutingService()

    result = service.route("api token issue")

    assert result.handled is True
    assert result.intent == "technical_redirect"
    assert "technical question" in result.answer.lower()


def test_intent_service_routes_procedural_query_to_legal_help():
    service = IntentRoutingService()

    result = service.route("how do i file a cyber fraud complaint")

    assert result.handled is True
    assert result.intent in {"legal_help", "procedural"}


def test_intent_service_routes_issue_statement_to_legal_help():
    service = IntentRoutingService()

    result = service.route("My landlord is harassing me")

    assert result.handled is True
    assert result.intent == "legal_help"


def test_intent_service_prioritizes_upi_fraud_issue_over_greeting():
    service = IntentRoutingService()

    result = service.route("I clicked a fake UPI link and money got debited")

    assert result.handled is True
    assert result.intent == "legal_help"
    assert "Legal India" not in result.answer


def test_intent_service_prioritizes_bank_fraud_issue_over_welcome():
    service = IntentRoutingService()

    result = service.route("bank fraud complaint")

    assert result.handled is True
    assert result.intent == "legal_help"


def test_intent_service_keeps_clean_hello_as_greeting():
    service = IntentRoutingService()

    result = service.route("hello")

    assert result.handled is True
    assert result.intent == "greeting"


def test_intent_service_keeps_detailed_legal_queries_out_of_intent_shortcuts():
    service = IntentRoutingService()

    result = service.route("Section 420 IPC")

    assert result.handled is False


def test_intent_service_disables_greeting_once_conversation_has_started():
    service = IntentRoutingService()

    result = service.route("hi", conversation_started=True, active_intent="legal_help")

    assert result.handled is True
    assert result.intent == "legal_help"
