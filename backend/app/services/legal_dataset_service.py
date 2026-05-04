from __future__ import annotations

import json
import re
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
            "source_filename": f"{self.dataset}.json",
            "jurisdiction": "India",
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
        return self.dataset.replace("_", " ")


class LocalLegalDatasetService:
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
            file_candidates=("bnss.json",),
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
        spec = self.DATASET_SPECS.get(str(dataset or "").strip().lower())
        if spec is None:
            return False
        return any((self.datasets_dir / file_name).exists() for file_name in spec.file_candidates)

    @classmethod
    def parse_query(cls, query: str) -> dict[str, str] | None:
        normalized = re.sub(r"\s+", " ", str(query or "").strip().lower())
        if not normalized:
            return None

        article_match = re.search(
            r"\b(?:explain\s+|tell me about\s+|what is\s+)?article\s+([0-9]+[a-z]?)\b",
            normalized,
        )
        if article_match:
            return {"dataset": "constitution", "identifier": article_match.group(1)}

        statute_patterns = {
            "ipc": (
                r"\bsection\s+([0-9]+[a-z]?)\s+(?:of\s+the\s+)?(?:ipc|indian penal code)\b",
                r"\b(?:ipc|indian penal code)\s+(?:section\s+)?([0-9]+[a-z]?)\b",
            ),
            "bns": (
                r"\bsection\s+([0-9]+[a-z]?)\s+(?:of\s+the\s+)?(?:bns|bharatiya nyaya sanhita)\b",
                r"\b(?:bns|bharatiya nyaya sanhita)\s+(?:section\s+)?([0-9]+[a-z]?)\b",
            ),
            "bnss": (
                r"\bsection\s+([0-9]+[a-z]?)\s+(?:of\s+the\s+)?(?:bnss|bharatiya nagarik suraksha sanhita)\b",
                r"\b(?:bnss|bharatiya nagarik suraksha sanhita)\s+(?:section\s+)?([0-9]+[a-z]?)\b",
            ),
        }
        for dataset, patterns in statute_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, normalized)
                if match:
                    return {"dataset": dataset, "identifier": match.group(1)}

        bare_section_match = re.fullmatch(
            r"section\s+([0-9]+[a-z]?)(?:\s+(?:in short|briefly|short|step by step|in points|pointwise))?",
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
            for dataset, spec in self.DATASET_SPECS.items()
        }
        self._loaded = True

    def _load_dataset_index(self, spec: LegalDatasetSpec) -> dict[str, LegalProvisionMatch]:
        payload, source_path = self._read_dataset_payload(spec.file_candidates)
        if not isinstance(payload, list) or source_path is None:
            return {}

        index: dict[str, LegalProvisionMatch] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            identifier = self._resolve_item_identifier(spec, item)
            if not identifier:
                continue
            title = self._clean_text(item.get("title"))
            text = self._clean_text(item.get("description") or item.get("text"))
            if not text:
                continue
            provision_number = f"{spec.item_label} {identifier}"
            resolved_title = title or f"{spec.title_fallback_prefix} {identifier}"
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
            )

        logger.info("legal dataset load dataset=%s entries=%s path=%s", spec.dataset, len(index), source_path)
        return index

    def _resolve_item_identifier(self, spec: LegalDatasetSpec, item: dict[str, Any]) -> str:
        if spec.dataset == "ipc":
            for value in (item.get("description"), item.get("text"), item.get("title"), item.get(spec.item_key)):
                identifier = self._extract_embedded_identifier(value)
                if identifier:
                    return identifier
        return self._normalize_identifier(item.get(spec.item_key))

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
        match = re.search(r"(?:^|content:\s*)([0-9]+[a-zA-Z]?)\.", text, re.IGNORECASE)
        if not match:
            return ""
        return cls._normalize_identifier(match.group(1))

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        match = re.search(r"([0-9]+)([a-zA-Z]?)", text)
        if not match:
            return ""
        number = str(int(match.group(1)))
        suffix = match.group(2).upper()
        return f"{number}{suffix}"

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
