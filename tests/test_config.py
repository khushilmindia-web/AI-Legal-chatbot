from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.services import storage


def test_database_settings_default_to_mongodb_after_cutover():
    settings = Settings()

    assert settings.database_backend == "mongodb"
    assert settings.mongodb_uri == "mongodb://127.0.0.1:27017"
    assert settings.mongodb_database == "lawyer_ai"


def test_database_backend_accepts_mongodb_choice():
    settings = Settings(DATABASE_BACKEND="mongodb")

    assert settings.database_backend == "mongodb"


def test_storage_factory_creates_mongodb_store(monkeypatch):
    created = []

    class FakeMongoSessionStore:
        def __init__(self, settings):
            created.append(settings)

    monkeypatch.setattr(storage, "MongoSessionStore", FakeMongoSessionStore)
    settings = Settings(DATABASE_BACKEND="mongodb")

    store = storage.create_session_store(settings)

    assert isinstance(store, FakeMongoSessionStore)
    assert created == [settings]


def test_storage_factory_rejects_sqlite_runtime():
    settings = Settings(DATABASE_BACKEND="sqlite")

    with pytest.raises(RuntimeError, match="SQLite temporarily disabled during MongoDB migration"):
        storage.create_session_store(settings)


def test_database_backend_rejects_unknown_choice():
    with pytest.raises(ValidationError):
        Settings(DATABASE_BACKEND="postgres")
