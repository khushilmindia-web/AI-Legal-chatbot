from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

import requests

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.indiankanoon")

SOURCE_AUTHORITY_SCORES = {
    "supremecourt": 12.0,
    "scorders": 11.0,
    "laws": 10.0,
    "constitution": 10.0,
    "delhi": 8.0,
    "bombay": 8.0,
    "allahabad": 8.0,
    "karnataka": 8.0,
    "kerala": 8.0,
    "chennai": 8.0,
    "kolkata": 8.0,
    "gujarat": 8.0,
    "tribunals": 6.0,
    "consumer": 6.0,
}


class IndianKanoonService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.indiankanoon_api_base_url.rstrip("/") + "/"
        self.session = requests.Session()
        self.session.trust_env = settings.indiankanoon_trust_env_proxy

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

    def search_references(
        self,
        query: str,
        *,
        doctypes: str | None = None,
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        return self.search_references_multi(
            query_variants=[query],
            doctypes_options=[doctypes],
            max_results=max_results,
        )

    def search_references_multi(
        self,
        *,
        query_variants: list[str],
        doctypes_options: list[str | None],
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        seen_doc_ids: set[str] = set()
        results: list[dict[str, Any]] = []

        for query in query_variants:
            if len(results) >= max_results:
                break
            for doctypes in doctypes_options:
                payload = self.search(query, page_num=0, doctypes=doctypes, max_cites=5)
                docs = payload.get("docs") or []
                logger.info(
                    "indiankanoon search variant query=%r doctypes=%r docs=%s",
                    query[:120],
                    doctypes,
                    len(docs),
                )
                for doc in docs:
                    if len(results) >= max_results:
                        break
                    doc_id = doc.get("tid")
                    if not doc_id:
                        continue
                    doc_id_text = str(doc_id)
                    if doc_id_text in seen_doc_ids:
                        continue
                    seen_doc_ids.add(doc_id_text)
                    results.append(
                        {
                            "doc_id": doc_id_text,
                            "title": str(doc.get("title") or "").strip(),
                            "headline": str(doc.get("headline") or "").strip(),
                            "docsource": str(doc.get("docsource") or "").strip(),
                            "citations": [str(item) for item in (doc.get("citations") or [])[:5]],
                            "url": f"https://indiankanoon.org/doc/{doc_id_text}/",
                        }
                    )

        return results

    def retrieve_grounded_documents(
        self,
        *,
        query_variants: list[str],
        doctypes_options: list[str | None],
        max_results: int = 4,
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}

        for query in query_variants:
            for doctypes in doctypes_options:
                payload = self.search(query, page_num=0, doctypes=doctypes, max_cites=8)
                docs = payload.get("docs") or []
                logger.info(
                    "indiankanoon pipeline stage=search query=%r doctypes=%r docs=%s",
                    query[:160],
                    doctypes,
                    len(docs),
                )
                for rank, doc in enumerate(docs[:8], start=1):
                    doc_id = doc.get("tid")
                    if not doc_id:
                        continue
                    doc_id_text = str(doc_id)
                    score = self._score_search_doc(doc, query=query, rank=rank)
                    current = candidates.get(doc_id_text)
                    if current is not None and current["score"] >= score:
                        continue
                    candidates[doc_id_text] = {
                        "doc_id": doc_id_text,
                        "title": str(doc.get("title") or "").strip(),
                        "headline": str(doc.get("headline") or "").strip(),
                        "docsource": str(doc.get("docsource") or "").strip(),
                        "citations": [str(item) for item in (doc.get("citations") or [])[:5]],
                        "url": f"https://indiankanoon.org/doc/{doc_id_text}/",
                        "matched_query": query,
                        "doctypes": doctypes,
                        "score": score,
                    }

        if not candidates:
            logger.info("indiankanoon pipeline stage=search no_candidates")
            return []

        ranked_candidates = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
        enriched: list[dict[str, Any]] = []
        for candidate in ranked_candidates[: max(max_results * 2, max_results)]:
            enriched_item = dict(candidate)
            doc_id_text = candidate["doc_id"]
            query = candidate["matched_query"]
            try:
                fragment_payload = self.get_document_fragments(doc_id_text, query)
                enriched_item["fragment_title"] = str(fragment_payload.get("title") or "").strip()
                enriched_item["fragment_headline"] = self._clean_text(fragment_payload.get("headline") or "")
                enriched_item["fragment_excerpt"] = self._extract_document_excerpt(fragment_payload)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("indiankanoon fragment failed doc_id=%s error=%s", doc_id_text, exc)
                enriched_item["fragment_title"] = ""
                enriched_item["fragment_headline"] = ""
                enriched_item["fragment_excerpt"] = ""
            try:
                document_payload = self.get_document(doc_id_text, max_cites=6, max_cited_by=4)
                enriched_item["doc_excerpt"] = self._extract_document_excerpt(document_payload)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("indiankanoon doc fetch failed doc_id=%s error=%s", doc_id_text, exc)
                enriched_item["doc_excerpt"] = ""
            try:
                meta_payload = self.get_document_meta(doc_id_text)
                enriched_item["bench"] = meta_payload.get("bench")
                enriched_item["author"] = meta_payload.get("author")
                enriched_item["publishdate"] = meta_payload.get("publishdate")
            except (requests.RequestException, ValueError) as exc:
                logger.warning("indiankanoon metadata fetch failed doc_id=%s error=%s", doc_id_text, exc)
                enriched_item["bench"] = None
                enriched_item["author"] = None
                enriched_item["publishdate"] = None

            enriched_item["score"] = candidate["score"] + self._score_enriched_doc(enriched_item, query=query)
            logger.info(
                "indiankanoon pipeline stage=enrich doc_id=%s matched_query=%r score=%.2f",
                doc_id_text,
                query[:160],
                enriched_item["score"],
            )
            enriched.append(enriched_item)

        enriched.sort(key=lambda item: item["score"], reverse=True)
        return enriched[:max_results]

    def _headers(self) -> dict[str, str]:
        token = self.settings.indiankanoon_api_token.strip()
        if not token:
            raise ValueError("Indian Kanoon API token is not configured")
        return {
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = urljoin(self.base_url, path)
        try:
            response = self.session.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.settings.indiankanoon_timeout_seconds,
            )
        except requests.Timeout:
            logger.warning(
                "indiankanoon request timed out path=%s timeout=%ss params=%s",
                path,
                self.settings.indiankanoon_timeout_seconds,
                params,
            )
            raise
        except requests.RequestException:
            logger.exception("indiankanoon request failed path=%s params=%s", path, params)
            raise

        body_preview = self._body_preview(response.text)
        logger.info(
            "indiankanoon response path=%s status=%s params=%s body=%r",
            path,
            response.status_code,
            params,
            body_preview,
        )
        if not response.ok:
            logger.warning(
                "indiankanoon non-success path=%s status=%s body=%r",
                path,
                response.status_code,
                body_preview,
            )
            response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning(
                "indiankanoon invalid json path=%s status=%s body=%r",
                path,
                response.status_code,
                body_preview,
            )
            raise ValueError("Indian Kanoon returned an invalid JSON response") from exc
        if not isinstance(payload, dict):
            logger.warning(
                "indiankanoon unexpected payload type path=%s payload_type=%s body=%r",
                path,
                type(payload).__name__,
                body_preview,
            )
            raise ValueError("Indian Kanoon returned an unexpected response payload")
        return payload

    def _get_text(self, path: str, *, params: dict[str, Any] | None = None) -> str:
        response = self.session.get(
            urljoin(self.base_url, path),
            headers=self._headers(),
            params=params,
            timeout=self.settings.indiankanoon_timeout_seconds,
        )
        logger.info(
            "indiankanoon text response path=%s status=%s params=%s body=%r",
            path,
            response.status_code,
            params,
            self._body_preview(response.text),
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _body_preview(text: str, limit: int = 500) -> str:
        return " ".join((text or "").split())[:limit]

    @staticmethod
    def _clean_text(text: Any) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", str(text or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _score_search_doc(self, doc: dict[str, Any], *, query: str, rank: int) -> float:
        normalized_query = self._normalize_tokens(query)
        haystack = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("headline") or ""),
                str(doc.get("docsource") or ""),
            ]
        )
        overlap = len(normalized_query & self._normalize_tokens(haystack))
        source = str(doc.get("docsource") or "").strip().lower()
        authority_score = SOURCE_AUTHORITY_SCORES.get(source, 4.0)
        return authority_score + float(overlap * 2) + max(0.0, 8.0 - rank)

    def _score_enriched_doc(self, doc: dict[str, Any], *, query: str) -> float:
        normalized_query = self._normalize_tokens(query)
        text = " ".join(
            [
                doc.get("fragment_headline") or "",
                doc.get("fragment_excerpt") or "",
                doc.get("doc_excerpt") or "",
            ]
        )
        overlap = len(normalized_query & self._normalize_tokens(text))
        return float(overlap * 1.5)

    def _extract_document_excerpt(self, payload: dict[str, Any]) -> str:
        preferred_keys = [
            "doc",
            "docfragment",
            "fragment",
            "headline",
            "judgment",
            "text",
            "body",
            "content",
        ]
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return self._clean_text(value)[:1800]
            if isinstance(value, dict):
                nested_text = self._extract_document_excerpt(value)
                if nested_text:
                    return nested_text[:1800]
        longest = ""
        for value in payload.values():
            if isinstance(value, str):
                cleaned = self._clean_text(value)
                if len(cleaned) > len(longest):
                    longest = cleaned
            elif isinstance(value, dict):
                cleaned = self._extract_document_excerpt(value)
                if len(cleaned) > len(longest):
                    longest = cleaned
        return longest[:1800]

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
