from __future__ import annotations

from backend.app.services.openai_service import OpenAIResponsesService


def test_parse_json_extracts_dict_from_fenced_response():
    payload = OpenAIResponsesService._parse_json(
        '```json\n{"answer":"Structured answer","likely_forum":"supremecourt"}\n```'
    )

    assert payload["answer"] == "Structured answer"
    assert payload["likely_forum"] == "supremecourt"


def test_parse_json_extracts_embedded_json_object():
    payload = OpenAIResponsesService._parse_json(
        'Here is the result:\n{"answer":"Grounded answer","documents_to_keep":["complaint copy"]}'
    )

    assert payload["answer"] == "Grounded answer"
    assert payload["documents_to_keep"] == ["complaint copy"]


def test_parse_json_returns_empty_dict_for_blank_output():
    assert OpenAIResponsesService._parse_json("   ") == {}
