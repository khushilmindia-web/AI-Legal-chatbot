from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger
from training.embeddings import HashingEmbeddingProvider

logger = get_logger("lawyer_ai.local_ml")

try:  # pragma: no cover - optional dependency/runtime path
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass(slots=True)
class LocalRouteDecision:
    label: str
    confidence: float
    scores: dict[str, float]
    backend: str


@dataclass(slots=True)
class SemanticSupportDecision:
    status: str
    score: float
    claim_count: int
    supported_claims: int
    backend: str
    details: list[dict[str, Any]]


class LocalTextSimilarityService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.hashing = HashingEmbeddingProvider(dimensions=128)
        self._model = None
        self._backend = "hashing"

    @property
    def backend(self) -> str:
        return self._backend

    def score_labels(self, text: str, label_descriptions: dict[str, str]) -> LocalRouteDecision:
        normalized = str(text or "").strip()
        if not normalized or not label_descriptions:
            return LocalRouteDecision(label="none", confidence=0.0, scores={}, backend=self.backend)
        scores = {label: round(self.similarity(normalized, description), 4) for label, description in label_descriptions.items()}
        best_label = max(scores, key=scores.get)
        ordered_scores = sorted(scores.values(), reverse=True)
        best_score = ordered_scores[0]
        margin = best_score - (ordered_scores[1] if len(ordered_scores) > 1 else 0.0)
        confidence = max(min(best_score + (margin * 0.35), 0.99), 0.0)
        return LocalRouteDecision(label=best_label, confidence=round(confidence, 4), scores=scores, backend=self.backend)

    def similarity(self, left: str, right: str) -> float:
        left_text = str(left or "").strip()
        right_text = str(right or "").strip()
        if not left_text or not right_text:
            return 0.0
        embeddings = self.embed_texts([left_text, right_text])
        return round(self._cosine_similarity(embeddings[0], embeddings[1]), 4)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        if model is not None:  # pragma: no cover - depends on local cached model availability
            vectors = model.encode(texts, normalize_embeddings=True)
            self._backend = "sentence_transformer"
            return [list(map(float, row)) for row in vectors]
        self._backend = "hashing"
        return self.hashing.embed_texts(texts)

    def _load_model(self):  # pragma: no cover - optional local cached path
        if self._model is not None:
            return self._model
        if not self.settings.local_model_assist_enabled:
            return None
        if SentenceTransformer is None:
            return None
        model_name = self.settings.local_text_similarity_model.strip()
        if not model_name:
            return None
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
            logger.info("loaded local text similarity model=%s", model_name)
            return self._model
        except Exception as exc:
            logger.warning("local text similarity model unavailable model=%s error=%s", model_name, exc)
            self._model = None
            return None

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        limit = min(len(left), len(right))
        if limit == 0:
            return 0.0
        dot = sum(left[index] * right[index] for index in range(limit))
        left_norm = math.sqrt(sum(value * value for value in left[:limit])) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right[:limit])) or 1.0
        return max(min(dot / (left_norm * right_norm), 1.0), -1.0)


class LocalSemanticSupportChecker:
    def __init__(self, settings: Settings, similarity_service: LocalTextSimilarityService | None = None) -> None:
        self.settings = settings
        self.similarity_service = similarity_service or LocalTextSimilarityService(settings)

    def check(self, *, answer: str, query: str, evidence_texts: list[str]) -> SemanticSupportDecision:
        claims = self._extract_claims(answer)
        if not claims or not evidence_texts:
            return SemanticSupportDecision(
                status="unsupported",
                score=0.0,
                claim_count=len(claims),
                supported_claims=0,
                backend=self.similarity_service.backend,
                details=[],
            )

        details: list[dict[str, Any]] = []
        supported_claims = 0
        aggregate = 0.0
        for claim in claims:
            similarities = [self.similarity_service.similarity(claim, evidence) for evidence in evidence_texts if evidence.strip()]
            best_score = max(similarities) if similarities else 0.0
            supported = best_score >= self.settings.local_support_check_threshold
            if supported:
                supported_claims += 1
            aggregate += best_score
            details.append({"claim": claim, "best_score": round(best_score, 4), "supported": supported})

        avg_score = aggregate / max(len(claims), 1)
        support_ratio = supported_claims / max(len(claims), 1)
        if avg_score >= 0.72 and support_ratio >= 0.75:
            status = "supported"
        elif avg_score >= 0.56 and support_ratio >= 0.4:
            status = "borderline"
        else:
            status = "unsupported"
        return SemanticSupportDecision(
            status=status,
            score=round(avg_score, 4),
            claim_count=len(claims),
            supported_claims=supported_claims,
            backend=self.similarity_service.backend,
            details=details,
        )

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        lines = [line.strip() for line in str(answer or "").splitlines() if line.strip()]
        claims: list[str] = []
        for line in lines:
            if line.startswith(("Sources:", "Disclaimer:")):
                continue
            line = re.sub(r"^(Summary|Legal Position|Practical Next Steps):\s*", "", line, flags=re.IGNORECASE)
            if not line:
                continue
            segments = re.split(r"(?<=[.!?])\s+", line)
            for segment in segments:
                cleaned = segment.strip()
                if cleaned:
                    claims.append(cleaned)
        return claims[:8]
