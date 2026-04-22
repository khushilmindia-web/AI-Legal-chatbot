from __future__ import annotations

import json
from pathlib import Path


def test_compact_eval_set_exists_and_has_expected_cases():
    path = Path("data/evals/legal_bot_eval_set.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) >= 8
    assert any(row["case_id"] == "cyber_fraud_001" for row in rows)
    assert any(row["case_id"] == "statute_001" for row in rows)
    assert all("expected_route" in row for row in rows)
