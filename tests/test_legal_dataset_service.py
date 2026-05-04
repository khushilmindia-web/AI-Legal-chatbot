from __future__ import annotations

import json

from backend.app.services.legal_dataset_service import LocalLegalDatasetService


def test_local_legal_dataset_supports_bns_reference_variants(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "bns.json").write_text(
        json.dumps(
            [
                {
                    "section": "34",
                    "title": "Things done in private defence.",
                    "text": "Nothing is an offence which is done in the exercise of the right of private defence.",
                    "source_url": "https://devgan.in/bns/section/34/",
                    "source_name": "Devgan.in",
                },
                {
                    "section": "318",
                    "title": "Cheating.",
                    "text": "Whoever, by deceiving any person, fraudulently or dishonestly induces delivery of property, cheats.",
                    "source_url": "https://devgan.in/bns/section/318/",
                    "source_name": "Devgan.in",
                },
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)

    section_34_queries = ["bns 34", "section 34 bns", "bns section 34"]
    for query in section_34_queries:
        match = service.lookup_query(query)
        assert match is not None
        assert match.dataset == "bns"
        assert match.provision_number == "Section 34"
        assert match.title == "Things done in private defence."
        assert "right of private defence" in match.text
        assert match.source == "Devgan.in: https://devgan.in/bns/section/34/ (local dataset: data/legal_datasets/bns.json)"

    section_318 = service.lookup_query("bns section 318")
    assert section_318 is not None
    assert section_318.dataset == "bns"
    assert section_318.provision_number == "Section 318"
    assert section_318.title == "Cheating."
    assert "dishonestly induces delivery of property" in section_318.text
