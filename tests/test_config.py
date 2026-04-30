from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def test_database_settings_default_to_sqlite_until_migration_cutover():
    settings = Settings()

    assert settings.database_backend == "sqlite"
    assert settings.mongodb_uri == "mongodb://127.0.0.1:27017"
    assert settings.mongodb_database == "lawyer_ai"


def test_database_backend_accepts_mongodb_choice():
    settings = Settings(DATABASE_BACKEND="mongodb")

    assert settings.database_backend == "mongodb"


def test_database_backend_rejects_unknown_choice():
    with pytest.raises(ValidationError):
        Settings(DATABASE_BACKEND="postgres")
