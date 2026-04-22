from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.services.domain_packs import LegalDomainPackService


def test_domain_pack_manifest_loads_default_and_issue_pack():
    service = LegalDomainPackService(Settings())
    issue_pack = service.pack_for(domain="criminal", issue_type="cyber_fraud")
    default_pack = service.pack_for(domain="unknown", issue_type=None)

    assert issue_pack["pack_id"] == "cyber_fraud_pack_v1"
    assert "boost_terms" in issue_pack
    assert default_pack["pack_id"] == "general_local_pack_v1"
