from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.services.local_ml import LocalSemanticSupportChecker, LocalTextSimilarityService


def test_local_route_assist_scores_labels_without_external_model():
    service = LocalTextSimilarityService(Settings())
    decision = service.score_labels(
        "I clicked a fake UPI link and money got debited",
        {
            "playbook": "Practical legal help, fraud reporting, complaint steps, and next actions.",
            "grounded": "Statute query, judgment lookup, citation search, and legal authority explanation.",
        },
    )

    assert decision.label in {"playbook", "grounded"}
    assert decision.backend in {"hashing", "sentence_transformer"}
    assert decision.scores


def test_semantic_support_checker_marks_unrelated_claims_as_unsupported():
    checker = LocalSemanticSupportChecker(Settings())
    decision = checker.check(
        answer="Summary: You will definitely win.\nLegal Position: Section 999 applies here.\nPractical Next Steps: File immediately.",
        query="section 138 cheque bounce legal position",
        evidence_texts=["Section 138 of the Negotiable Instruments Act discusses cheque dishonour consequences."],
    )

    assert decision.status in {"borderline", "unsupported"}
    assert decision.claim_count >= 1
