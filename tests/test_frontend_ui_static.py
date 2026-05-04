from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_chat_resets_optional_case_detail_inputs():
    script = (ROOT / "Frontend" / "app.js").read_text(encoding="utf-8")

    assert "function clearCaseDetails()" in script

    set_blank_draft = script.split("function setBlankDraft()", 1)[1].split(
        "function getRenderedMessages()", 1
    )[0]
    assert "clearCaseDetails();" in set_blank_draft

    for field_name in ["stateInput", "districtInput", "caseStageInput", "ownMatterInput"]:
        assert f"elements.{field_name}" in script
        assert f"elements.{field_name}.value = \"\";" in script
