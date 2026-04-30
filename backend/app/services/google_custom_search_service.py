from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.google_custom_search")


@dataclass(slots=True)
class GoogleSearchResult:
    documents: list[dict[str, Any]]
    from_cache: bool
    trusted_result_count: int


class GoogleCustomSearchService:
    SEARCH_URL = "https://customsearch.googleapis.com/customsearch/v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    @property
    def configured(self) -> bool:
        return self.settings.google_custom_search_configured

    def search(
        self,
        *,
        query: str,
        max_results: int | None = None,
        site_restrict: str | None = None,
        trusted_only: bool = True,
    ) -> GoogleSearchResult:
        normalized_query = self._normalize_query(query=query, site_restrict=site_restrict, trusted_only=trusted_only)
        cached = self._read_cache(normalized_query)
        if cached is not None:
            return GoogleSearchResult(
                documents=list(cached),
                from_cache=True,
                trusted_result_count=len(cached),
            )

        if not self.configured:
            return GoogleSearchResult(documents=[], from_cache=False, trusted_result_count=0)

        result_limit = max(1, min(int(max_results or self.settings.google_search_max_results), 3))
        params = {
            "key": self.settings.google_custom_search_api_key,
            "cx": self.settings.google_custom_search_cx,
            "q": query.strip(),
            "num": result_limit,
        }
        if site_restrict:
            params["siteSearch"] = site_restrict.strip()
            params["siteSearchFilter"] = "i"

        response = self._http_get(
            self.SEARCH_URL,
            params=params,
            timeout=self.settings.google_search_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        items = payload.get("items") or []
        raw_item_preview = [
            {
                "title": str(item.get("title") or "").strip(),
                "link": str(item.get("link") or "").strip(),
                "snippet": str(item.get("snippet") or "").strip()[:240],
            }
            for item in items[:3]
        ]
        logger.info(
            "google custom search raw_response query=%r item_count=%s site=%s trusted_only=%s raw_items=%s",
            query[:140],
            len(items),
            site_restrict or "",
            trusted_only,
            raw_item_preview,
        )
        documents = self._normalize_items(items, trusted_only=trusted_only)
        self._write_cache(normalized_query, documents)
        logger.info(
            "google custom search query=%r results=%s trusted=%s cache=false site=%s trusted_only=%s",
            query[:140],
            len(documents),
            len(documents),
            site_restrict or "",
            trusted_only,
        )
        return GoogleSearchResult(
            documents=documents,
            from_cache=False,
            trusted_result_count=len(documents),
        )

    @staticmethod
    def _http_get(url: str, **kwargs: Any) -> requests.Response:
        with requests.Session() as session:
            # Keep CX calls independent from machine-level proxy variables. Some
            # local test environments set HTTP_PROXY/HTTPS_PROXY to dead ports.
            session.trust_env = False
            return session.get(url, **kwargs)

    def _normalize_items(self, items: list[dict[str, Any]], *, trusted_only: bool) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        trusted_domains = self.settings.google_search_trusted_domains
        for index, item in enumerate(items, start=1):
            link = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip() or "Official web result"
            snippet = str(item.get("snippet") or "").strip()
            domain = self._domain_for_url(link)
            reject_reasons: list[str] = []
            if not link:
                reject_reasons.append("missing_link")
            if link and not domain:
                reject_reasons.append("invalid_or_missing_domain")
            is_trusted = any(domain == trusted or domain.endswith(f".{trusted}") for trusted in trusted_domains)
            logger.info(
                "google custom search candidate index=%s title=%r domain=%s trusted=%s trusted_only=%s link=%s snippet=%r",
                index,
                title[:140],
                domain,
                is_trusted,
                trusted_only,
                link,
                snippet[:240],
            )
            if trusted_only and trusted_domains and not is_trusted:
                reject_reasons.append("trusted_domain_filter_miss")
            if not trusted_only and link and domain and not self._is_allowed_general_domain(domain):
                reject_reasons.append("general_domain_filter_blocked")
            if reject_reasons:
                logger.info(
                    "google custom search reject index=%s title=%r domain=%s trusted=%s trusted_only=%s reasons=%s",
                    index,
                    title[:140],
                    domain or "",
                    is_trusted,
                    trusted_only,
                    reject_reasons,
                )
                continue
            authority_type = self._authority_type_for_domain(domain)
            score = max(8.0, 20.0 - float(index - 1) * 2.5)
            documents.append(
                {
                    "doc_id": f"google:{domain}:{index}:{abs(hash(link))}",
                    "title": title,
                    "headline": snippet or title,
                    "fragment_headline": title,
                    "fragment_excerpt": snippet[:1200],
                    "doc_excerpt": snippet[:1600],
                    "docsource": f"google:{domain}",
                    "citations": [title, link],
                    "publishdate": "",
                    "url": link,
                    "score": score,
                    "source_kind": "google_custom_search",
                    "authority_type": authority_type,
                    "jurisdiction": "India" if domain.endswith(".in") else "",
                    "document_kind": "official_web_result",
                    "recency_bucket": "unknown",
                    "metadata_confidence": 0.78 if is_trusted else (0.72 if authority_type in {"government_portal", "regulator"} else 0.58),
                    "source_domain": domain,
                    "google_scope": "curated" if trusted_only else "general_fallback",
                    "trusted_domain_match": is_trusted,
                }
            )
        return documents[: self.settings.google_search_max_results]

    @staticmethod
    def _authority_type_for_domain(domain: str) -> str:
        if domain.endswith("rbi.org.in"):
            return "regulator"
        if domain.endswith("gov.in") or domain.endswith("nic.in"):
            return "government_portal"
        if "court" in domain or domain.endswith("sci.gov.in"):
            return "court_portal"
        return "web_reference"

    @staticmethod
    def _domain_for_url(url: str) -> str:
        host = urlparse(url).netloc.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _normalize_query(self, *, query: str, site_restrict: str | None, trusted_only: bool) -> str:
        normalized_query = " ".join(str(query or "").lower().split())
        scope = "trusted" if trusted_only else "general"
        return f"{normalized_query}|site={str(site_restrict or '').lower().strip()}|scope={scope}"

    @staticmethod
    def _is_allowed_general_domain(domain: str) -> bool:
        blocked_markers = {"blogspot.", "wordpress.", "medium.com", "youtube.com", "facebook.com", "instagram.com", "x.com"}
        if any(marker in domain for marker in blocked_markers):
            return False
        if domain.endswith((".gov.in", ".nic.in", ".org.in", ".ac.in", ".edu.in")):
            return True
        if "court" in domain or domain.endswith("sci.gov.in") or domain.endswith("indiacode.nic.in"):
            return True
        return False

    def _read_cache(self, key: str) -> list[dict[str, Any]] | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        expires_at, documents = cached
        if expires_at <= time.time():
            self._cache.pop(key, None)
            return None
        return documents

    def _write_cache(self, key: str, documents: list[dict[str, Any]]) -> None:
        ttl = max(int(self.settings.google_search_cache_ttl_seconds), 60)
        self._cache[key] = (time.time() + ttl, list(documents))
