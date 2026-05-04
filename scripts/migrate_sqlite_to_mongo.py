from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import DB_PATH, get_settings  # noqa: E402


TABLES = (
    "users",
    "auth_sessions",
    "password_reset_tokens",
    "chat_sessions",
    "chat_messages",
)

logger = logging.getLogger("sqlite_to_mongo_migration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate Lawyer AI SQLite data into MongoDB.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print source/target counts and planned actions; do not insert or update MongoDB documents.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(DB_PATH),
        help=f"SQLite database path. Defaults to {DB_PATH}.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def sqlite_rows(connection: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = connection.execute(f"SELECT * FROM {table_name} ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


def parse_json_object(raw_value: str | None) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def transform_row(table_name: str, row: dict[str, Any]) -> dict[str, Any]:
    document = dict(row)
    if table_name == "users" and document.get("google_sub") is None:
        document.pop("google_sub", None)
    elif table_name == "chat_sessions":
        document["state"] = parse_json_object(document.pop("state_json", None))
    elif table_name == "chat_messages":
        document["metadata"] = parse_json_object(document.pop("metadata_json", None))
    return document


def create_indexes(database) -> None:
    database.users.create_index([("id", ASCENDING)], unique=True)
    database.users.create_index([("email", ASCENDING)], unique=True)
    database.users.create_index([("google_sub", ASCENDING)], unique=True, sparse=True)

    database.auth_sessions.create_index([("id", ASCENDING)], unique=True)
    database.auth_sessions.create_index([("token", ASCENDING)], unique=True)
    database.auth_sessions.create_index([("user_id", ASCENDING)])
    database.auth_sessions.create_index([("expires_at", ASCENDING)])

    database.password_reset_tokens.create_index([("id", ASCENDING)], unique=True)
    database.password_reset_tokens.create_index([("token_hash", ASCENDING)], unique=True)
    database.password_reset_tokens.create_index([("user_id", ASCENDING)])

    database.chat_sessions.create_index([("id", ASCENDING)], unique=True)
    database.chat_sessions.create_index([("user_id", ASCENDING)])

    database.chat_messages.create_index([("id", ASCENDING)], unique=True)
    database.chat_messages.create_index([("chat_id", ASCENDING)])


def sync_counter(database, collection_name: str, max_id: int) -> None:
    database.counters.update_one(
        {"_id": collection_name},
        {"$max": {"seq": max_id}},
        upsert=True,
    )


def migrate_table(database, table_name: str, rows: list[dict[str, Any]]) -> None:
    collection = database[table_name]
    logger.info("Migrating %s (%s rows)", table_name, len(rows))
    for row in rows:
        document = transform_row(table_name, row)
        collection.replace_one({"id": document["id"]}, document, upsert=True)
    max_id = max((int(row["id"]) for row in rows), default=0)
    sync_counter(database, table_name, max_id)


def count_matching_ids(database, table_name: str, rows: list[dict[str, Any]]) -> int:
    ids = [row["id"] for row in rows]
    if not ids:
        return 0
    return database[table_name].count_documents({"id": {"$in": ids}})


def validate_counts(database, rows_by_table: dict[str, list[dict[str, Any]]]) -> bool:
    logger.info("")
    logger.info("Validation")
    ok = True
    for table_name in TABLES:
        sqlite_count = len(rows_by_table[table_name])
        migrated_count = count_matching_ids(database, table_name, rows_by_table[table_name])
        total_count = database[table_name].count_documents({})
        logger.info("%s: SQLite=%s -> MongoMigrated=%s (MongoTotal=%s)", table_name, sqlite_count, migrated_count, total_count)
        if migrated_count != sqlite_count:
            ok = False
    return ok


def main() -> int:
    configure_logging()
    args = parse_args()
    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        logger.error("SQLite database not found: %s", sqlite_path)
        return 1

    settings = get_settings()
    logger.info("SQLite source: %s", sqlite_path)
    logger.info("Mongo target: %s / %s", settings.mongodb_uri, settings.mongodb_database)
    logger.info("Mode: %s", "DRY RUN" if args.dry_run else "MIGRATE")
    logger.info("")

    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        rows_by_table = {table_name: sqlite_rows(connection, table_name) for table_name in TABLES}

    sqlite_counts = {table_name: len(rows) for table_name, rows in rows_by_table.items()}
    for table_name in TABLES:
        logger.info("%s: SQLite=%s", table_name, sqlite_counts[table_name])

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    database = client[settings.mongodb_database]
    try:
        client.admin.command("ping")
        logger.info("")
        logger.info("MongoDB counts before migration")
        for table_name in TABLES:
            logger.info("%s: MongoBefore=%s", table_name, database[table_name].count_documents({}))

        if args.dry_run:
            logger.info("")
            logger.info("Dry run complete. No MongoDB documents were inserted or updated.")
            logger.info("Planned collections: %s", ", ".join(TABLES))
            return 0

        create_indexes(database)
        for table_name in TABLES:
            migrate_table(database, table_name, rows_by_table[table_name])
        return 0 if validate_counts(database, rows_by_table) else 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
