from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the compact legal-bot evaluation set and expected scoring fields.")
    parser.add_argument("--eval-set", default="data/evals/legal_bot_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cases = load_eval_cases(Path(args.eval_set))[: args.limit]
    summary = {
        "case_count": len(cases),
        "domains": sorted({str(case.get("domain") or "") for case in cases}),
        "query_types": sorted({str(case.get("query_type") or "") for case in cases}),
        "score_fields": ["grounding", "usefulness", "fallback_correctness", "disclaimer_correctness", "hallucination_risk"],
    }
    print(json.dumps(summary, indent=2))
    print("\nSample cases:")
    for case in cases:
        print(f"- {case.get('case_id')}: {case.get('query')}")


if __name__ == "__main__":
    main()
