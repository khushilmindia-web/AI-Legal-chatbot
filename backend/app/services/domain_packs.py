from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.utils.request_context import get_logger

logger = get_logger("lawyer_ai.domain_packs")


class LegalDomainPackService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.manifest_path = settings.domain_packs_manifest_path
        self._packs = self._load_manifest(self.manifest_path)

    def pack_for(self, *, domain: str, issue_type: str | None = None) -> dict[str, Any]:
        issue_key = str(issue_type or "").strip().lower()
        if issue_key and issue_key in self._packs.get("issue_packs", {}):
            return dict(self._packs["issue_packs"][issue_key])
        domain_key = str(domain or "").strip().lower()
        return dict(self._packs.get("domain_packs", {}).get(domain_key, self._packs.get("default_pack", {})))

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        if not path.exists():
            logger.warning("domain pack manifest missing path=%s", path)
            return {"default_pack": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            logger.warning("domain pack manifest invalid path=%s error=%s", path, exc)
        return {"default_pack": {}}
