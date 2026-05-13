from __future__ import annotations

import json
import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import DATA_DIR
from backend.app.utils.request_context import get_logger


logger = get_logger("lawyer_ai.legal_dataset")
LEGAL_DATASETS_DIR = DATA_DIR / "legal_datasets"


@dataclass(frozen=True, slots=True)
class LegalDatasetSpec:
    dataset: str
    file_candidates: tuple[str, ...]
    item_key: str
    item_label: str
    statute_title: str
    source_label: str
    domain: str
    docsource: str
    authority_type: str
    title_fallback_prefix: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegalProvisionMatch:
    dataset: str
    provision_number: str
    title: str
    text: str
    explanation: str
    source: str
    domain: str
    docsource: str
    authority_type: str
    statute_name: str = ""
    jurisdiction: str = "India"
    source_path: str = ""
    document_title: str = ""

    @property
    def answer_title(self) -> str:
        return f"{self.provision_number} of {self._statute_display_name()}"

    def to_document(self) -> dict[str, Any]:
        doc_id = re.sub(r"[^a-z0-9]+", "-", f"{self.dataset}-{self.provision_number.lower()}").strip("-")
        excerpt = self.text.strip()
        return {
            "doc_id": f"local_legal_dataset:{doc_id}",
            "title": self.answer_title,
            "headline": self.title,
            "fragment_headline": self.title,
            "fragment_excerpt": excerpt[:2000],
            "doc_excerpt": excerpt[:3200],
            "docsource": self.docsource,
            "citations": [self.source],
            "publishdate": "",
            "url": self.source,
            "score": 100.0,
            "source_kind": "local_legal_dataset",
            "retrieval_source": "local_legal_dataset",
            "retrieval_confidence": 1.0,
            "retrieval_confidence_level": "strong",
            "document_kind": "statute",
            "authority_type": self.authority_type,
            "legal_domain": self.domain,
            "source_filename": Path(self.source_path).name if self.source_path else f"{self.dataset}.json",
            "source_path": self.source_path,
            "jurisdiction": self.jurisdiction,
            "statute_name": self.statute_name,
            "document_title": self.document_title,
            "section_reference": self.provision_number,
            "citation_hint": f"{self.statute_name} | {self.provision_number}",
            "metadata_confidence": 1.0,
            "recency_bucket": "dated",
        }

    def _statute_display_name(self) -> str:
        if self.dataset == "constitution":
            return "the Constitution of India"
        if self.dataset == "ipc":
            return "the Indian Penal Code, 1860"
        if self.dataset == "bns":
            return "the Bharatiya Nyaya Sanhita, 2023"
        if self.dataset == "bnss":
            return "the Bharatiya Nagarik Suraksha Sanhita, 2023"
        if self.statute_name:
            return self.statute_name
        return self.dataset.replace("_", " ")


class LocalLegalDatasetService:
    KNOWN_DATASET_METADATA: dict[str, dict[str, Any]] = {
        "constitution": {
            "item_key": "article",
            "item_label": "Article",
            "statute_title": "Constitution of India",
            "source_label": "Constitution of India",
            "domain": "constitutional",
            "docsource": "constitution",
            "authority_type": "constitution",
            "title_fallback_prefix": "Article",
            "aliases": ("constitution", "article", "articles"),
        },
        "ipc": {
            "item_key": "chapter",
            "item_label": "Section",
            "statute_title": "Indian Penal Code, 1860",
            "source_label": "Indian Penal Code, 1860",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("ipc", "indian penal code"),
        },
        "bns": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Bharatiya Nyaya Sanhita, 2023",
            "source_label": "Bharatiya Nyaya Sanhita, 2023",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("bns", "bharatiya nyaya sanhita"),
        },
        "bnss": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "source_label": "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("bnss", "bharatiya nagarik suraksha sanhita"),
        },
        "bsa": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Bharatiya Sakshya Adhiniyam, 2023",
            "source_label": "Bharatiya Sakshya Adhiniyam, 2023",
            "domain": "evidence",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("bsa", "bharatiya sakshya adhiniyam", "bharatiya sakshya act"),
        },
        "nia": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Negotiable Instruments Act, 1881",
            "source_label": "Negotiable Instruments Act, 1881",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("nia", "ni act", "n i act", "negotiable instruments act", "negotiable instruments"),
        },
        "iea": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Indian Evidence Act, 1872",
            "source_label": "Indian Evidence Act, 1872",
            "domain": "evidence",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("iea", "evidence act", "indian evidence act"),
        },
        "ida": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Indian Divorce Act, 1869",
            "source_label": "Indian Divorce Act, 1869",
            "domain": "family",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("ida", "divorce act", "indian divorce act"),
        },
        "hma": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Hindu Marriage Act, 1955",
            "source_label": "Hindu Marriage Act, 1955",
            "domain": "family",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("hma", "hindu marriage act"),
        },
        "mva": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Motor Vehicles Act, 1988",
            "source_label": "Motor Vehicles Act, 1988",
            "domain": "motor_vehicles",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("mva", "motor vehicles act", "motor vehicle act"),
        },
        "crpc": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Code of Criminal Procedure, 1973",
            "source_label": "Code of Criminal Procedure, 1973",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("crpc", "cr.p.c", "code of criminal procedure", "criminal procedure code"),
        },
        "cpc": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Code of Civil Procedure, 1908",
            "source_label": "Code of Civil Procedure, 1908",
            "domain": "civil",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("cpc", "c.p.c", "code of civil procedure", "civil procedure code"),
        },
        "companies act": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Companies Act, 2013",
            "source_label": "Companies Act, 2013",
            "domain": "corporate",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("companies act", "company act", "companies law", "company law"),
        },
        "consumer laws": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Consumer Protection Law Dataset",
            "source_label": "Consumer Protection Law Dataset",
            "domain": "consumer",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("consumer laws", "consumer law", "consumer protection", "consumer protection act"),
        },
        "it": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Information Technology Act, 2000",
            "source_label": "Information Technology Act, 2000",
            "domain": "cyber",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("it act", "information technology act", "information technology"),
        },
        "scst": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
            "source_label": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("scst", "sc st act", "sc/st act", "scheduled castes and scheduled tribes act", "prevention of atrocities act"),
        },
        "arms": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Arms Act, 1959",
            "source_label": "Arms Act, 1959",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("arms", "arms act", "arms law"),
        },
        "juvenile justice": {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": "Juvenile Justice (Care and Protection of Children) Act, 2015",
            "source_label": "Juvenile Justice (Care and Protection of Children) Act, 2015",
            "domain": "criminal",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": ("juvenile justice", "jj act", "juvenile justice act", "care and protection of children act"),
        },
        "criminal_law": {
            "item_key": "section",
            "item_label": "Topic",
            "statute_title": "Criminal Law Overview",
            "source_label": "Criminal Law Overview",
            "domain": "criminal",
            "docsource": "internal_corpus",
            "authority_type": "legal_overview",
            "title_fallback_prefix": "Topic",
            "aliases": ("criminal law", "crime law", "criminal procedure", "criminal offences"),
        },
    }
    DATASET_SPECS: dict[str, LegalDatasetSpec] = {
        "constitution": LegalDatasetSpec(
            dataset="constitution",
            file_candidates=("constitution.json",),
            item_key="article",
            item_label="Article",
            statute_title="Constitution of India",
            source_label="Constitution of India",
            domain="constitutional",
            docsource="constitution",
            authority_type="constitution",
            title_fallback_prefix="Article",
            aliases=("constitution", "article", "articles"),
        ),
        "ipc": LegalDatasetSpec(
            dataset="ipc",
            file_candidates=("ipc.json", "IPC.json"),
            item_key="chapter",
            item_label="Section",
            statute_title="Indian Penal Code, 1860",
            source_label="Indian Penal Code, 1860",
            domain="criminal",
            docsource="laws",
            authority_type="statute",
            title_fallback_prefix="Section",
            aliases=("ipc", "indian penal code"),
        ),
        "bns": LegalDatasetSpec(
            dataset="bns",
            file_candidates=("bns.json",),
            item_key="section",
            item_label="Section",
            statute_title="Bharatiya Nyaya Sanhita, 2023",
            source_label="Bharatiya Nyaya Sanhita, 2023",
            domain="criminal",
            docsource="laws",
            authority_type="statute",
            title_fallback_prefix="Section",
            aliases=("bns", "bharatiya nyaya sanhita"),
        ),
        "bnss": LegalDatasetSpec(
            dataset="bnss",
            file_candidates=("BNSS.json", "bnss.json"),
            item_key="section",
            item_label="Section",
            statute_title="Bharatiya Nagarik Suraksha Sanhita, 2023",
            source_label="Bharatiya Nagarik Suraksha Sanhita, 2023",
            domain="criminal",
            docsource="laws",
            authority_type="statute",
            title_fallback_prefix="Section",
            aliases=("bnss", "bharatiya nagarik suraksha sanhita"),
        ),
    }

    def __init__(self, datasets_dir: Path | None = None) -> None:
        self.datasets_dir = datasets_dir or LEGAL_DATASETS_DIR
        self.dataset_specs = self._build_dataset_specs()
        self._indices: dict[str, dict[str, LegalProvisionMatch]] = {}
        self._loaded = False

    def lookup_query(self, query: str) -> LegalProvisionMatch | None:
        reference = self.parse_query(query)
        if reference is None:
            return None
        return self.lookup(dataset=reference["dataset"], identifier=reference["identifier"])

    def lookup(self, *, dataset: str, identifier: str) -> LegalProvisionMatch | None:
        self._ensure_loaded()
        normalized_identifier = self._normalize_identifier(identifier)
        if not normalized_identifier:
            return None
        return self._indices.get(dataset, {}).get(normalized_identifier)

    def has_dataset_file(self, dataset: str) -> bool:
        spec = self.dataset_specs.get(str(dataset or "").strip().lower())
        if spec is None:
            return False
        return any((self.datasets_dir / file_name).exists() for file_name in spec.file_candidates)

    def aliases_for_dataset(self, dataset: str) -> tuple[str, ...]:
        spec = self.dataset_specs.get(str(dataset or "").strip().lower())
        return spec.aliases if spec is not None else ()

    def dataset_inventory(self) -> dict[str, dict[str, Any]]:
        self._ensure_loaded()
        inventory: dict[str, dict[str, Any]] = {}
        for dataset, spec in self.dataset_specs.items():
            payload, source_path = self._read_dataset_payload(spec.file_candidates)
            available = source_path is not None
            count = len(self._indices.get(dataset, {}))
            source_names: set[str] = set()
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    source_name = self._clean_text(item.get("source_name"))
                    if source_name:
                        source_names.add(source_name)
            if not source_names:
                source_names.add(spec.source_label)
            inventory[dataset] = {
                "name": spec.statute_title,
                "available": available,
                "status": "available" if count > 0 else ("empty" if available else "missing"),
                "count": count,
                "file_name": source_path.name if source_path is not None else "",
                "source_names": sorted(source_names),
            }
        return inventory

    def search_documents(self, query: str, *, max_results: int = 6) -> list[dict[str, Any]]:
        self._ensure_loaded()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        query_lower = str(query or "").lower()
        scored: list[tuple[float, LegalProvisionMatch]] = []
        for spec in self.dataset_specs.values():
            alias_hit = any(re.search(rf"\b{re.escape(alias)}\b", query_lower) for alias in spec.aliases)
            for match in self._indices.get(spec.dataset, {}).values():
                haystack = " ".join(
                    [
                        match.statute_name,
                        match.provision_number,
                        match.title,
                        match.text[:3000],
                    ]
                ).lower()
                target_tokens = self._tokenize(haystack)
                overlap = len(query_tokens & target_tokens)
                if not overlap and not alias_hit:
                    continue
                score = overlap * 8.0
                if alias_hit:
                    score += 60.0
                if match.provision_number.lower() in query_lower:
                    score += 25.0
                if match.authority_type == "legal_overview" and alias_hit:
                    score += 20.0
                scored.append((score, match))
        scored.sort(key=lambda item: item[0], reverse=True)
        documents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, match in scored:
            document = match.to_document()
            doc_id = str(document.get("doc_id") or "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            document["score"] = round(score, 3)
            documents.append(document)
            if len(documents) >= max_results:
                break
        return documents

    def parse_query(self, query: str) -> dict[str, str] | None:
        normalized = re.sub(r"\s+", " ", str(query or "").strip().lower())
        if not normalized:
            return None

        article_match = re.search(
            r"\b(?:explain\s+|tell me about\s+|what is\s+)?article\s+([0-9]+(?:\.[0-9]+)*[a-z]?)\b",
            normalized,
        )
        if article_match:
            return {"dataset": "constitution", "identifier": article_match.group(1)}

        for dataset, spec in self.dataset_specs.items():
            if dataset == "constitution":
                continue
            aliases_pattern = "|".join(re.escape(alias) for alias in sorted(spec.aliases, key=len, reverse=True))
            if not aliases_pattern:
                continue
            patterns = (
                rf"\bsection\s+([0-9]+(?:\.[0-9]+)*[a-z]?)\s+(?:of\s+the\s+)?(?:{aliases_pattern})\b",
                rf"\b(?:{aliases_pattern})\s+(?:section\s+)?([0-9]+(?:\.[0-9]+)*[a-z]?)\b",
                rf"\b([0-9]+(?:\.[0-9]+)*[a-z]?)\s+(?:{aliases_pattern})\s+section\b",
            )
            for pattern in patterns:
                match = re.search(pattern, normalized)
                if match:
                    return {"dataset": dataset, "identifier": match.group(1)}

        bare_section_match = re.fullmatch(
            r"section\s+([0-9]+(?:\.[0-9]+)*[a-z]?)(?:\s+(?:in short|briefly|short|step by step|in points|pointwise))?",
            normalized,
        )
        if bare_section_match:
            return {"dataset": "ipc", "identifier": bare_section_match.group(1)}

        return None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._indices = {
            dataset: self._load_dataset_index(spec)
            for dataset, spec in self.dataset_specs.items()
        }
        self._loaded = True

    def _build_dataset_specs(self) -> dict[str, LegalDatasetSpec]:
        discovered_paths = sorted(self.datasets_dir.glob("*.json")) if self.datasets_dir.exists() else []
        known_file_names = {candidate.lower() for spec in self.DATASET_SPECS.values() for candidate in spec.file_candidates}
        newly_detected = [path.name for path in discovered_paths if path.name.lower() not in known_file_names]
        logger.info(
            "legal dataset scan dir=%s json_files=%s newly_detected=%s",
            self.datasets_dir,
            [path.name for path in discovered_paths],
            newly_detected,
        )

        specs: dict[str, LegalDatasetSpec] = dict(self.DATASET_SPECS)
        for path in discovered_paths:
            dataset = path.stem.strip().lower()
            if not dataset or dataset in specs:
                continue
            meta = self.KNOWN_DATASET_METADATA.get(dataset) or self._infer_dataset_metadata_from_file(path, dataset)
            specs[dataset] = LegalDatasetSpec(
                dataset=dataset,
                file_candidates=(path.name,),
                item_key=str(meta["item_key"]),
                item_label=str(meta["item_label"]),
                statute_title=str(meta["statute_title"]),
                source_label=str(meta["source_label"]),
                domain=str(meta["domain"]),
                docsource=str(meta["docsource"]),
                authority_type=str(meta["authority_type"]),
                title_fallback_prefix=str(meta["title_fallback_prefix"]),
                aliases=tuple(str(alias) for alias in meta["aliases"]),
            )
            logger.info(
                "legal dataset authority_metadata dataset=%s statute=%s authority_type=%s jurisdiction=%s aliases=%s path=%s",
                dataset,
                meta["statute_title"],
                meta["authority_type"],
                "India",
                tuple(meta["aliases"]),
                path,
            )
        return specs

    @staticmethod
    def _infer_dataset_metadata(dataset: str) -> dict[str, Any]:
        title = dataset.replace("_", " ").replace("-", " ").strip()
        statute_title = LocalLegalDatasetService._titleize_statute_name(title)
        return {
            "item_key": "section",
            "item_label": "Section",
            "statute_title": statute_title,
            "source_label": statute_title,
            "domain": "general",
            "docsource": "laws",
            "authority_type": "statute",
            "title_fallback_prefix": "Section",
            "aliases": tuple(
                cls_alias
                for cls_alias in LocalLegalDatasetService._expand_aliases(
                    {
                        dataset,
                        dataset.replace("_", " "),
                        dataset.replace("-", " "),
                        statute_title.lower(),
                    }
                )
                if cls_alias
            ),
        }

    @classmethod
    def _infer_dataset_metadata_from_file(cls, path: Path, dataset: str) -> dict[str, Any]:
        meta = cls._infer_dataset_metadata(dataset)
        payload = cls._read_json_file(path)
        statute_name = ""
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    statute_name = cls._clean_text(item.get("statute") or item.get("act") or item.get("name"))
                    if statute_name:
                        break
        elif isinstance(payload, dict):
            statute_name = cls._clean_text(payload.get("statute") or payload.get("act") or payload.get("name") or payload.get("area"))
            if payload.get("area") and not payload.get("statute"):
                meta["item_label"] = "Topic"
                meta["authority_type"] = "legal_overview"
                meta["docsource"] = "internal_corpus"

        if statute_name:
            statute_title = cls._titleize_statute_name(statute_name)
            meta["statute_title"] = statute_title
            meta["source_label"] = statute_title
            aliases = set(meta["aliases"])
            aliases.add(statute_name.lower())
            aliases.add(statute_title.lower())
            meta["aliases"] = tuple(alias for alias in cls._expand_aliases(aliases) if alias)
        return meta

    @staticmethod
    def _expand_aliases(values: set[str] | tuple[str, ...]) -> set[str]:
        expanded: set[str] = set()
        for value in values:
            alias = re.sub(r"\s+", " ", str(value or "").strip().lower())
            if not alias:
                continue
            expanded.add(alias)
            expanded.add(alias.replace("_", " ").replace("-", " "))
            no_parentheses = re.sub(r"\s*\([^)]*\)\s*", " ", alias)
            no_parentheses = re.sub(r"\s+", " ", no_parentheses).strip()
            if no_parentheses:
                expanded.add(no_parentheses)
            alpha_numeric = re.sub(r"[^a-z0-9]+", " ", alias)
            alpha_numeric = re.sub(r"\s+", " ", alpha_numeric).strip()
            if alpha_numeric:
                expanded.add(alpha_numeric)
        return expanded

    @staticmethod
    def _titleize_statute_name(value: str) -> str:
        cleaned = re.sub(r"[_-]+", " ", str(value or "")).strip()
        if not cleaned:
            return "Legal Dataset"
        known_upper = {"BNS", "BNSS", "BSA", "IPC", "CRPC", "CPC", "MVA", "NIA", "SCST", "CGST", "RBI", "FEM", "SARFAESI", "UGC", "IT", "AIC"}
        words = []
        for word in cleaned.split():
            stripped = word.strip()
            if stripped.upper() in known_upper:
                words.append(stripped.upper())
            elif stripped.isupper() and len(stripped) <= 4:
                words.append(stripped)
            else:
                words.append(stripped.capitalize())
        title = " ".join(words)
        title = title.replace("Boi", "BOI")
        return title

    def _load_dataset_index(self, spec: LegalDatasetSpec) -> dict[str, LegalProvisionMatch]:
        payload, source_path = self._read_dataset_payload(spec.file_candidates)
        items = self._payload_items(payload)
        if not items or source_path is None:
            if source_path is not None:
                logger.warning(
                    "legal dataset skipped path=%s reason=no_supported_records dataset=%s payload_type=%s",
                    source_path,
                    spec.dataset,
                    type(payload).__name__,
                )
            return {}

        index: dict[str, LegalProvisionMatch] = {}
        skipped_items = 0
        for item in items:
            if not isinstance(item, dict):
                skipped_items += 1
                continue
            item = self._normalize_dataset_item(item)
            if not item:
                skipped_items += 1
                continue
            identifier = self._resolve_item_identifier(spec, item)
            if not identifier:
                skipped_items += 1
                continue
            title = self._clean_text(item.get("title") or item.get("section_title") or item.get("article_title"))
            text = self._clean_text(
                item.get("description")
                or item.get("section_desc")
                or item.get("article_desc")
                or item.get("text")
                or item.get("content")
            )
            provision_number = f"{spec.item_label} {identifier}"
            resolved_title = title or f"{spec.title_fallback_prefix} {identifier}"
            if not text:
                text = resolved_title
            if not text:
                skipped_items += 1
                continue
            index[identifier] = LegalProvisionMatch(
                dataset=spec.dataset,
                provision_number=provision_number,
                title=resolved_title,
                text=text,
                explanation=self._build_explanation(
                    dataset=spec.dataset,
                    provision_number=provision_number,
                    title=resolved_title,
                ),
                source=self._resolve_source_label(spec, item, source_path),
                domain=spec.domain,
                docsource=spec.docsource,
                authority_type=spec.authority_type,
                statute_name=spec.statute_title,
                jurisdiction=self._clean_text(item.get("jurisdiction")) or "India",
                source_path=str(source_path),
                document_title=f"{spec.statute_title} {provision_number}: {resolved_title}",
            )

        logger.info(
            "legal dataset ingestion dataset=%s statute=%s entries=%s skipped_items=%s authority_type=%s jurisdiction=%s path=%s",
            spec.dataset,
            spec.statute_title,
            len(index),
            skipped_items,
            spec.authority_type,
            "India",
            source_path,
        )
        return index

    def _payload_items(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        title = self._clean_text(payload.get("title") or payload.get("area") or payload.get("name") or "Overview")
        text_parts: list[str] = []
        for key in ("description", "notes", "summary", "content"):
            text = self._clean_text(payload.get(key))
            if text:
                text_parts.append(text)
        for key in ("key_principles", "statutes", "regulatory_bodies", "offences", "procedures", "defenses", "rights"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        text_parts.append(
                            "; ".join(
                                f"{self._clean_text(k)}: {self._clean_text(v)}"
                                for k, v in item.items()
                                if self._clean_text(v)
                            )
                        )
                    else:
                        text = self._clean_text(item)
                        if text:
                            text_parts.append(text)
        return [
            {
                "section": "overview",
                "title": title,
                "content": "\n".join(part for part in text_parts if part) or title,
                "jurisdiction": self._clean_text(payload.get("jurisdiction")) or "India",
            }
        ]

    def _resolve_item_identifier(self, spec: LegalDatasetSpec, item: dict[str, Any]) -> str:
        if spec.dataset == "ipc":
            for value in (item.get("description"), item.get("text"), item.get("title"), item.get(spec.item_key)):
                identifier = self._extract_embedded_identifier(value)
                if identifier:
                    return identifier
        return self._normalize_identifier(item.get(spec.item_key) or item.get("section") or item.get("article"))

    @staticmethod
    def _normalize_dataset_item(item: dict[str, Any]) -> dict[str, Any]:
        if "chapter,section,section_title,section_desc" not in item:
            return item
        raw = str(item.get("chapter,section,section_title,section_desc") or "").strip()
        if not raw:
            return {}
        try:
            row = next(csv.reader([raw]))
        except csv.Error:
            row = []
        if len(row) >= 4:
            return {
                "chapter": row[0],
                "section": row[1],
                "section_title": row[2],
                "section_desc": ",".join(row[3:]).strip().strip('"'),
            }
        return {"section_desc": raw}

    def _read_dataset_payload(self, file_candidates: tuple[str, ...]) -> tuple[Any, Path | None]:
        for file_name in file_candidates:
            path = self.datasets_dir / file_name
            payload = self._read_json_file(path)
            if payload is not None:
                return payload, path
        return [], None

    def _resolve_source_label(self, spec: LegalDatasetSpec, item: dict[str, Any], source_path: Path) -> str:
        source_name = self._clean_text(item.get("source_name"))
        source_url = self._clean_text(item.get("source_url"))
        dataset_label = f"local dataset: data/legal_datasets/{source_path.name}"
        if source_name and source_url:
            return f"{source_name}: {source_url} ({dataset_label})"
        if source_name:
            return f"{source_name} ({dataset_label})"
        if source_url:
            return f"{spec.source_label}: {source_url} ({dataset_label})"
        return f"{spec.source_label} ({dataset_label})"

    @staticmethod
    def _read_json_file(path: Path) -> Any | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            logger.info("legal dataset missing path=%s", path)
        except json.JSONDecodeError as exc:
            logger.warning("legal dataset invalid_json path=%s error=%s", path, exc)
        except OSError as exc:
            logger.warning("legal dataset unreadable path=%s error=%s", path, exc)
        return None

    @staticmethod
    def _build_explanation(*, dataset: str, provision_number: str, title: str) -> str:
        normalized_title = title.strip().rstrip(".")
        if dataset == "constitution":
            return f"{provision_number} is the constitutional provision dealing with {normalized_title.lower()}."
        if dataset == "ipc":
            return f"{provision_number} is the IPC provision dealing with {normalized_title.lower()}."
        if dataset == "bns":
            return f"{provision_number} is the BNS provision dealing with {normalized_title.lower()}."
        if dataset == "bnss":
            return f"{provision_number} is the BNSS provision dealing with {normalized_title.lower()}."
        return f"{provision_number} addresses {normalized_title.lower()}."

    @classmethod
    def _extract_embedded_identifier(cls, value: Any) -> str:
        text = cls._clean_text(value)
        if not text:
            return ""
        match = re.search(r"(?:^|content:\s*)([0-9]+(?:\.[0-9]+)*[a-zA-Z]?)\.", text, re.IGNORECASE)
        if not match:
            return ""
        return cls._normalize_identifier(match.group(1))

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"([0-9]+(?:\.[0-9]+)*)([a-zA-Z]?)", text)
        if not match:
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            return slug[:60]
        number = ".".join(str(int(part)) for part in match.group(1).split("."))
        suffix = match.group(2).upper()
        return f"{number}{suffix}"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower()) if len(token) > 2}

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        replacements = {
            "\u00a0": " ",
            "Ã‚ ": " ",
            "Ã‚": "",
            "Ã¢â‚¬â€": "-",
            "Ã¢â‚¬â€œ": "-",
            "Ã¢â‚¬Â": '"',
            "Ã¢â‚¬Å“": '"',
            "Ã¢â‚¬Ëœ": "'",
            "Ã¢â‚¬â„¢": "'",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        text = re.sub(r"-{20,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
