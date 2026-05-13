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


def test_local_legal_dataset_discovers_new_json_statute_files(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "NIA.json").write_text(
        json.dumps(
            [
                {
                    "chapter": 17,
                    "section": 138,
                    "section_title": "Dishonour of cheque for insufficiency, etc., of funds in the account.",
                    "section_desc": "Where any cheque drawn by a person is returned unpaid because funds are insufficient, this section may apply subject to statutory conditions.",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    match = service.lookup_query("section 138 ni act")

    assert match is not None
    assert match.dataset == "nia"
    assert match.provision_number == "Section 138"
    assert match.statute_name == "Negotiable Instruments Act, 1881"
    assert match.jurisdiction == "India"
    assert match.source_path.endswith("NIA.json")
    assert "Dishonour of cheque" in match.title
    assert "funds are insufficient" in match.text

    document = match.to_document()
    assert document["source_kind"] == "local_legal_dataset"
    assert document["authority_type"] == "statute"
    assert document["statute_name"] == "Negotiable Instruments Act, 1881"
    assert document["section_reference"] == "Section 138"


def test_local_legal_dataset_searches_discovered_files_for_grounded_retrieval(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "IEA.json").write_text(
        json.dumps(
            [
                {
                    "section": 65,
                    "section_title": "Cases in which secondary evidence relating to documents may be given.",
                    "section_desc": "Secondary evidence may be given of the existence, condition, or contents of a document in specified cases.",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    documents = service.search_documents("secondary evidence under evidence act", max_results=3)

    assert documents
    assert documents[0]["source_kind"] == "local_legal_dataset"
    assert documents[0]["statute_name"] == "Indian Evidence Act, 1872"
    assert documents[0]["title"] == "Section 65 of Indian Evidence Act, 1872"


def test_local_legal_dataset_uses_known_metadata_for_new_bsa_style_files(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "BSA.json").write_text(
        json.dumps(
            [
                {
                    "statute": "BSA",
                    "section": "3",
                    "title": "Evidence may be given of facts in issue and relevant facts.",
                    "content": "Evidence may be given in any suit or proceeding of every fact in issue and of such other facts as declared relevant.",
                    "jurisdiction": "India",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    match = service.lookup_query("section 3 bsa")

    assert match is not None
    assert match.dataset == "bsa"
    assert match.statute_name == "Bharatiya Sakshya Adhiniyam, 2023"
    assert match.domain == "evidence"
    assert match.authority_type == "statute"
    assert match.document_title == "Bharatiya Sakshya Adhiniyam, 2023 Section 3: Evidence may be given of facts in issue and relevant facts."


def test_local_legal_dataset_preserves_decimal_section_references(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "consumer laws.json").write_text(
        json.dumps(
            [
                {
                    "statute": "consumer laws",
                    "section": "1.1",
                    "title": "Who is a consumer?",
                    "content": "",
                    "jurisdiction": "India",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    match = service.lookup_query("section 1.1 consumer laws")

    assert match is not None
    assert match.dataset == "consumer laws"
    assert match.provision_number == "Section 1.1"
    assert match.text == "Who is a consumer?"
    assert match.statute_name == "Consumer Protection Law Dataset"


def test_local_legal_dataset_uses_known_metadata_for_arms_and_juvenile_justice(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "ARMS.json").write_text(
        json.dumps(
            [
                {
                    "statute": "ARMS",
                    "section": "3",
                    "title": "Licence for acquisition and possession of firearms and ammunition.",
                    "content": "",
                    "jurisdiction": "India",
                }
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "JUVENILE JUSTICE.json").write_text(
        json.dumps(
            [
                {
                    "statute": "JUVENILE JUSTICE",
                    "section": "3",
                    "title": "General principles to be followed in administration of Act.",
                    "content": "The Board and other agencies shall be guided by child-friendly principles.",
                    "jurisdiction": "India",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    arms = service.lookup_query("section 3 arms act")
    juvenile = service.lookup_query("section 3 juvenile justice act")

    assert arms is not None
    assert arms.dataset == "arms"
    assert arms.statute_name == "Arms Act, 1959"
    assert arms.domain == "criminal"
    assert juvenile is not None
    assert juvenile.dataset == "juvenile justice"
    assert juvenile.statute_name == "Juvenile Justice (Care and Protection of Children) Act, 2015"
    assert juvenile.document_title.startswith("Juvenile Justice (Care and Protection of Children) Act, 2015 Section 3")


def test_local_legal_dataset_ingests_topic_style_json_object_for_grounded_search(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "criminal_law.json").write_text(
        json.dumps(
            {
                "area": "Criminal Law",
                "description": "Body of law that relates to crime, prosecution, and punishment.",
                "key_principles": ["Presumption of innocence until proven guilty"],
                "statutes": [{"jurisdiction": "India", "name": "Bharatiya Nyaya Sanhita (BNS), 2023"}],
            }
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    documents = service.search_documents("criminal law presumption of innocence", max_results=3)

    assert documents
    assert documents[0]["source_kind"] == "local_legal_dataset"
    assert documents[0]["authority_type"] == "legal_overview"
    assert documents[0]["statute_name"] == "Criminal Law Overview"
    assert documents[0]["section_reference"] == "Topic overview"


def test_local_legal_dataset_ranks_exact_topic_alias_over_other_keyword_hits(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "criminal_law.json").write_text(
        json.dumps(
            {
                "area": "Criminal Law",
                "description": "Body of law that includes presumption of innocence and prosecution principles.",
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "CRPC.json").write_text(
        json.dumps(
            [
                {
                    "section": "321",
                    "section_title": "Withdrawal from prosecution",
                    "section_desc": "The Public Prosecutor may withdraw from prosecution in criminal proceedings.",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    documents = service.search_documents("criminal law presumption of innocence", max_results=2)

    assert documents
    assert documents[0]["source_filename"] == "criminal_law.json"
    assert documents[0]["authority_type"] == "legal_overview"


def test_local_legal_dataset_adds_simplified_alias_for_parenthesized_statute_name(tmp_path):
    dataset_dir = tmp_path / "legal_datasets"
    dataset_dir.mkdir()
    (dataset_dir / "Environment (Protection) Act.json").write_text(
        json.dumps(
            [
                {
                    "statute": "Environment (Protection) Act",
                    "section": "5",
                    "title": "Power to give directions.",
                    "content": "The Central Government may issue directions under this Act.",
                    "jurisdiction": "India",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = LocalLegalDatasetService(datasets_dir=dataset_dir)
    match = service.lookup_query("section 5 environment protection act")

    assert match is not None
    assert match.dataset == "environment (protection) act"
    assert match.provision_number == "Section 5"
    assert match.statute_name == "Environment (protection) Act"
