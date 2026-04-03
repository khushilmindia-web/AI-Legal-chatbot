from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests

from backend.app.core.config import Settings


class IndianKanoonService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.indiankanoon_api_base_url.rstrip("/") + "/"

    @property
    def configured(self) -> bool:
        return self.settings.indiankanoon_configured

    def search(
        self,
        query: str,
        *,
        page_num: int = 0,
        doctypes: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        title: str | None = None,
        cite: str | None = None,
        author: str | None = None,
        bench: str | None = None,
        max_cites: int | None = None,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "formInput": query,
            "pagenum": page_num,
        }
        optional_filters = {
            "doctypes": doctypes,
            "fromdate": from_date,
            "todate": to_date,
            "title": title,
            "cite": cite,
            "author": author,
            "bench": bench,
            "maxcites": max_cites,
            "maxpages": max_pages,
        }
        params.update({key: value for key, value in optional_filters.items() if value not in (None, "")})
        return self._get_json("search/", params=params)

    def get_document(self, doc_id: str | int, *, max_cites: int | None = None, max_cited_by: int | None = None) -> dict[str, Any]:
        params = {
            "maxcites": max_cites,
            "maxcitedby": max_cited_by,
        }
        return self._get_json(f"doc/{doc_id}/", params={key: value for key, value in params.items() if value is not None})

    def get_document_fragments(self, doc_id: str | int, query: str) -> dict[str, Any]:
        return self._get_json(f"docfragment/{doc_id}/", params={"formInput": query})

    def get_document_meta(self, doc_id: str | int) -> dict[str, Any]:
        return self._get_json(f"docmeta/{doc_id}/")

    def get_court_copy(self, doc_id: str | int) -> str:
        return self._get_text(f"origdoc/{doc_id}/")

    def retrieve_contextual_results(
        self,
        query: str,
        *,
        doctypes: str | None = None,
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        payload = self.search(query, page_num=0, doctypes=doctypes, max_cites=5)
        docs = payload.get("docs") or []
        results: list[dict[str, Any]] = []

        for doc in docs[:max_results]:
            doc_id = doc.get("tid")
            if not doc_id:
                continue
            result = {
                "doc_id": str(doc_id),
                "title": str(doc.get("title") or "").strip(),
                "headline": str(doc.get("headline") or "").strip(),
                "docsource": str(doc.get("docsource") or "").strip(),
                "citations": [str(item) for item in (doc.get("citations") or [])[:5]],
            }
            try:
                fragment_payload = self.get_document_fragments(doc_id, query)
                result["fragment_title"] = str(fragment_payload.get("title") or "").strip()
                result["fragment_headline"] = str(fragment_payload.get("headline") or "").strip()
            except requests.RequestException:
                result["fragment_title"] = ""
                result["fragment_headline"] = ""
            try:
                meta_payload = self.get_document_meta(doc_id)
                result["bench"] = meta_payload.get("bench")
                result["author"] = meta_payload.get("author")
                result["publishdate"] = meta_payload.get("publishdate")
            except requests.RequestException:
                result["bench"] = None
                result["author"] = None
                result["publishdate"] = None
            results.append(result)

        return results

    def _headers(self) -> dict[str, str]:
        token = self.settings.indiankanoon_api_token.strip()
        if not token:
            raise ValueError("Indian Kanoon API token is not configured")
        return {
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            urljoin(self.base_url, path),
            headers=self._headers(),
            params=params,
            timeout=self.settings.indiankanoon_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _get_text(self, path: str, *, params: dict[str, Any] | None = None) -> str:
        response = requests.get(
            urljoin(self.base_url, path),
            headers=self._headers(),
            params=params,
            timeout=self.settings.indiankanoon_timeout_seconds,
        )
        response.raise_for_status()
        return response.text
