from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.services.google_custom_search_service import GoogleCustomSearchService


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.content = b"ok"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_google_custom_search_settings_load_and_configure():
    settings = Settings(
        GOOGLE_SEARCH_ENABLED="true",
        GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
        GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        GOOGLE_SEARCH_TRUSTED_DOMAINS="gov.in,rbi.org.in",
    )

    assert settings.google_custom_search_configured is True
    assert settings.google_search_trusted_domains == ["gov.in", "rbi.org.in"]


def test_google_custom_search_filters_to_trusted_domains_and_caches(monkeypatch):
    settings = Settings(
        GOOGLE_SEARCH_ENABLED="true",
        GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
        GOOGLE_CUSTOM_SEARCH_CX="cx-id",
        GOOGLE_SEARCH_CACHE_TTL_SECONDS="3600",
        GOOGLE_SEARCH_TRUSTED_DOMAINS="gov.in,rbi.org.in",
    )
    service = GoogleCustomSearchService(settings)
    calls: list[dict] = []

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return FakeResponse(
            {
                "items": [
                    {
                        "title": "RBI official circular",
                        "link": "https://www.rbi.org.in/scripts/NotificationUser.aspx?id=1",
                        "snippet": "Official RBI circular snippet.",
                    },
                    {
                        "title": "Random blog",
                        "link": "https://blog.example.com/post",
                        "snippet": "Untrusted summary.",
                    },
                ]
            }
        )

    monkeypatch.setattr(GoogleCustomSearchService, "_http_get", staticmethod(fake_get))

    first = service.search(query="latest RBI circular on unauthorized transaction", max_results=3)
    second = service.search(query="latest RBI circular on unauthorized transaction", max_results=3)

    assert len(calls) == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert len(first.documents) == 1
    assert first.documents[0]["source_kind"] == "google_custom_search"
    assert first.documents[0]["authority_type"] == "regulator"


def test_google_custom_search_general_mode_applies_basic_domain_filtering(monkeypatch):
    settings = Settings(
        GOOGLE_SEARCH_ENABLED="true",
        GOOGLE_CUSTOM_SEARCH_API_KEY="api-key",
        GOOGLE_CUSTOM_SEARCH_CX="cx-id",
    )
    service = GoogleCustomSearchService(settings)

    def fake_get(url, params, timeout):
        return FakeResponse(
            {
                "items": [
                    {
                        "title": "India Code",
                        "link": "https://www.indiacode.nic.in/handle/123",
                        "snippet": "Official code portal.",
                    },
                    {
                        "title": "Random blog",
                        "link": "https://randomblog.blogspot.com/post",
                        "snippet": "Should be filtered.",
                    },
                ]
            }
        )

    monkeypatch.setattr(GoogleCustomSearchService, "_http_get", staticmethod(fake_get))

    result = service.search(query="section 420 ipc official text", max_results=3, trusted_only=False)

    assert len(result.documents) == 1
    assert result.documents[0]["source_domain"] == "indiacode.nic.in"
    assert result.documents[0]["google_scope"] == "general_fallback"


def test_google_custom_search_http_get_ignores_proxy_environment(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self):
            captured["entered_trust_env"] = self.trust_env
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def get(self, url, **kwargs):
            captured["trust_env"] = self.trust_env
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse({"items": []})

    monkeypatch.setattr("backend.app.services.google_custom_search_service.requests.Session", FakeSession)

    response = GoogleCustomSearchService._http_get(
        "https://customsearch.googleapis.com/customsearch/v1",
        params={"q": "Article 21"},
        timeout=8,
    )

    assert isinstance(response, FakeResponse)
    assert captured["trust_env"] is False
    assert captured["url"] == "https://customsearch.googleapis.com/customsearch/v1"
