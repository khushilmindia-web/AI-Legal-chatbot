from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from backend.app.core.config import DATA_DIR, ROOT_DIR, get_settings


LEGACY_TEMP_PATTERNS = (
    "manual_*",
    "temp_uploads_*",
    "tmp*",
)

LEGACY_DATA_PATTERNS = (
    "manual_*",
    "temp_uploads_*",
    "pytest*",
)


def _is_older_than(path: Path, max_age_hours: int) -> bool:
    if max_age_hours <= 0:
        return True
    age_seconds = max_age_hours * 3600
    try:
        newest_mtime = max(
            [path.stat().st_mtime, *[child.stat().st_mtime for child in path.rglob("*")]],
            default=path.stat().st_mtime,
        )
    except OSError:
        return False
    return (time.time() - newest_mtime) >= age_seconds


def _remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    print(f"{'Would remove' if dry_run else 'Removing'} {path}")
    if dry_run:
        return
    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Lawyer AI temporary folders.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything.")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Only remove matching folders older than this many hours. Use 0 to remove regardless of age.",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    for pattern in LEGACY_TEMP_PATTERNS:
        candidates.extend(path for path in ROOT_DIR.glob(pattern) if path.is_dir())
    for pattern in LEGACY_DATA_PATTERNS:
        candidates.extend(path for path in DATA_DIR.glob(pattern) if path.is_dir())
    candidates.extend(path for path in settings.temp_dir.iterdir() if path.exists())

    seen: set[Path] = set()
    for path in sorted(candidates):
        if path in seen:
            continue
        seen.add(path)
        if path == settings.uploads_dir or path == settings.temp_dir:
            for child in sorted(path.iterdir()):
                if _is_older_than(child, args.max_age_hours):
                    _remove_path(child, args.dry_run)
            continue
        if _is_older_than(path, args.max_age_hours):
            _remove_path(path, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
