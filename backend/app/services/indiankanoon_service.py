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
                try:
                    payload = self.search(query, page_num=0, doctypes=doctypes, max_cites=8)
                except (requests.RequestException, ValueError) as exc:
                    logger.warning(
                        "indiankanoon pipeline stage=search_failed query=%r doctypes=%r error=%s",
                        query[:160],
                        doctypes,
                        exc,
                    )
                    continue
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
                    if self._is_low_quality_search_doc(doc):
                        logger.info(
                            "indiankanoon pipeline stage=search_skip doc_id=%s reason=low_quality",
                            doc_id_text,
                        )
                        continue
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

            enriched_item["doc_excerpt"] = self._sanitize_excerpt(enriched_item.get("doc_excerpt") or "")
            enriched_item["fragment_excerpt"] = self._sanitize_excerpt(enriched_item.get("fragment_excerpt") or "")
            enriched_item["fragment_headline"] = self._sanitize_excerpt(enriched_item.get("fragment_headline") or "")
            if self._is_low_quality_enriched_doc(enriched_item):
                logger.info(
                    "indiankanoon pipeline stage=enrich_skip doc_id=%s reason=low_quality_enriched",
                    doc_id_text,
                )
                continue

            enriched_item["score"] = candidate["score"] + self._score_enriched_doc(enriched_item, query=query)
            logger.info(
                "indiankanoon pipeline stage=enrich doc_id=%s matched_query=%r score=%.2f",
                doc_id_text,
                query[:160],
                enriched_item["score"],
            )
            enriched.append(enriched_item)

        enriched.sort(key=lambda item: item["score"], reverse=True)
        return self._limit_to_most_relevant(enriched, max_results=max_results, query=query_variants[0])

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
            if path == "search/" or path.startswith("doc/") or path.startswith("docfragment/") or path.startswith("docmeta/"):
                response = self.session.post(
                    url,
                    headers=self._headers(),
                    data=params or {},
                    timeout=self.settings.indiankanoon_timeout_seconds,
                )
            else:
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
        cleaned = re.sub(r"\{[^{}]*\"errmsg\"[^{}]*\}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(errmsg|debug|traceback|stack trace|error in evaluting the fragments)\b\s*:?\s*[^.;]*", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _sanitize_excerpt(self, text: str) -> str:
        cleaned = self._clean_text(text)
        cleaned = re.sub(r"\b(Document \d+|Search snippet|Relevant fragment|Excerpt|Title|Authority|Date|URL|Citations)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,-")
        return cleaned[:1800]

    def _is_low_quality_search_doc(self, doc: dict[str, Any]) -> bool:
        title = self._clean_text(doc.get("title") or "")
        headline = self._clean_text(doc.get("headline") or "")
        haystack = f"{title} {headline}".lower()
        if not title:
            return True
        noisy_markers = [
            "error in evaluting the fragments",
            "no matching",
            "internal server error",
            "traceback",
            "debug",
        ]
        return any(marker in haystack for marker in noisy_markers)

    def _is_low_quality_enriched_doc(self, doc: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
            ]
        ).lower()
        if not text.strip():
            return True
        noisy_markers = [
            "error in evaluting the fragments",
            "traceback",
            "debug",
            "internal server error",
        ]
        return any(marker in text for marker in noisy_markers)

    def _score_search_doc(self, doc: dict[str, Any], *, query: str, rank: int) -> float:
        normalized_query = self._normalize_tokens(query)
        title = str(doc.get("title") or "")
        source = str(doc.get("docsource") or "").strip().lower()
        focus = self._query_focus(query)
        haystack = " ".join(
            [
                title,
                str(doc.get("headline") or ""),
                source,
            ]
        )
        overlap = len(normalized_query & self._normalize_tokens(haystack))
        authority_score = SOURCE_AUTHORITY_SCORES.get(source, 4.0)
        score = authority_score + float(overlap * 2) + max(0.0, 8.0 - rank)
        normalized_title = title.lower()
        if "section" in query.lower() and "section" in normalized_title:
            score += 8.0
        if "ipc" in query.lower() and "indian penal code" in normalized_title:
            score += 8.0
        if "punishment" in query.lower() and "punished" in str(doc.get("headline") or "").lower():
            score += 3.0
        if "union of india - section" in source:
            score += 6.0
        if self._looks_like_exact_authority_match(title=title, query=query):
            score += 10.0
        if focus == "statute":
            if source == "laws":
                score += 8.0
            elif source in {"supremecourt", "scorders"}:
                score -= 2.0
            if re.search(r"\b(vs\.?|v\.)\b", title.lower()):
                score -= 3.0
        elif focus == "case":
            if source in {"supremecourt", "scorders"}:
                score += 5.0
            elif source == "laws":
                score -= 2.0
        return score

    def _score_enriched_doc(self, doc: dict[str, Any], *, query: str) -> float:
        normalized_query = self._normalize_tokens(query)
        focus = self._query_focus(query)
        text = " ".join(
            [
                doc.get("fragment_headline") or "",
                doc.get("fragment_excerpt") or "",
                doc.get("doc_excerpt") or "",
            ]
        )
        overlap = len(normalized_query & self._normalize_tokens(text))
        score = float(overlap * 1.5)
        title = str(doc.get("title") or "")
        if self._looks_like_exact_authority_match(title=title, query=query):
            score += 8.0
        if doc.get("fragment_excerpt"):
            score += 2.0
        if doc.get("doc_excerpt"):
            score += 2.0
        if focus == "statute" and str(doc.get("docsource") or "").strip().lower() == "laws":
            score += 4.0
        if focus == "case" and str(doc.get("docsource") or "").strip().lower() in {"supremecourt", "scorders"}:
            score += 3.0
        return score

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
            if isinstance(value, list):
                flattened = " ".join(self._clean_text(item) for item in value if str(item).strip())
                if flattened.strip():
                    return flattened[:1800]
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
            elif isinstance(value, list):
                cleaned = " ".join(self._clean_text(item) for item in value if str(item).strip())
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

    def _looks_like_exact_authority_match(self, *, title: str, query: str) -> bool:
        normalized_title = title.lower()
        normalized_query = query.lower()
        section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", normalized_query)
        article_match = re.search(r"\barticle\s+([0-9]+[a-z]?)\b", normalized_query)
        if section_match and f"section {section_match.group(1)}" in normalized_title:
            return True
        if article_match and f"article {article_match.group(1)}" in normalized_title:
            return True
        if "ipc" in normalized_query and "indian penal code" in normalized_title:
            return True
        if "constitution" in normalized_query and "constitution" in normalized_title:
            return True
        return False

    def _limit_to_most_relevant(
        self,
        documents: list[dict[str, Any]],
        *,
        max_results: int,
        query: str = "",
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        filtered = [doc for doc in documents if not self._is_low_quality_enriched_doc(doc)]
        if query.strip():
            filtered = [doc for doc in filtered if not self._looks_irrelevant_for_query(doc=doc, query=query)]
        if not filtered:
            return []
        selected: list[dict[str, Any]] = [filtered[0]]
        top_source = str(filtered[0].get("docsource") or "").strip().lower()
        top_score = float(filtered[0].get("score") or 0.0)
        for doc in filtered[1:]:
            if len(selected) >= max_results:
                break
            source = str(doc.get("docsource") or "").strip().lower()
            score = float(doc.get("score") or 0.0)
            if score < top_score - 8.0:
                continue
            if source == top_source and len(selected) >= 1:
                continue
            selected.append(doc)
            if len(selected) >= 2:
                break
        return selected

    def _looks_irrelevant_for_query(self, *, doc: dict[str, Any], query: str) -> bool:
        focus = self._query_focus(query)
        title = str(doc.get("title") or "").lower()
        source = str(doc.get("docsource") or "").strip().lower()
        text = " ".join(
            [
                str(doc.get("title") or ""),
                str(doc.get("fragment_headline") or ""),
                str(doc.get("fragment_excerpt") or ""),
                str(doc.get("doc_excerpt") or ""),
            ]
        ).lower()
        overlap = len(self._normalize_tokens(query) & self._normalize_tokens(text))
        if focus == "statute":
            section_match = re.search(r"\bsection\s+([0-9]+[a-z]?)\b", query.lower())
            if source == "laws":
                if section_match and f"section {section_match.group(1)}" not in title and overlap < 2:
                    return True
                return False
            if section_match and f"section {section_match.group(1)}" not in text:
                return True
        if focus == "case" and overlap < 2:
            return True
        return overlap == 0

    @staticmethod
    def _query_focus(query: str) -> str:
        normalized = query.lower()
        if any(marker in normalized for marker in {"section ", "article ", " act", "ipc", "crpc", "cpc", "bns", "bnss", "constitution"}):
            return "statute"
        if any(marker in normalized for marker in {"judgment", "judgement", "precedent", "citation", "supreme court", "high court", "case law"}):
            return "case"
        return "general"
