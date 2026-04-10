from backend.app.services.legal_domain_classifier import LegalDomainClassifier


def test_legal_domain_classifier_maps_queries_to_expected_labels():
    classifier = LegalDomainClassifier()

    assert classifier.classify("My phone was snatched and I need to file an FIR") == "criminal"
    assert classifier.classify("How do I file an appeal procedure for a government order?") == "procedure"
    assert classifier.classify("Review this lease agreement and check this clause") == "document_review"
    assert classifier.classify("GST section 16 input tax credit eligibility") == "tax"
    assert classifier.classify("Shareholder oppression and mismanagement under company law") == "corporate"
    assert classifier.classify("My landlord is refusing to return possession of the property") == "property"
    assert classifier.classify("Defective product refund from seller under consumer law") == "consumer"
    assert classifier.classify("Legal notice for breach of contract and money recovery") == "civil"
