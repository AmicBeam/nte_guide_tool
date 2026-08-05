from contextlib import contextmanager
from collections.abc import Iterator

from peewee import Model, SqliteDatabase

from app.config import DATABASE_PATH

db = SqliteDatabase(DATABASE_PATH, pragmas={
    'foreign_keys': 1,
    'journal_mode': 'wal',
    'synchronous': 'normal',
    'busy_timeout': 5000,
})


def init_db(models: list[type[Model]]) -> None:
    if db.is_closed():
        db.connect(reuse_if_open=True)
    _reset_incompatible_snapshot_tables(models)
    # Existing tables need compatible columns before Peewee creates indexes for
    # newly-declared fields. SQLite otherwise treats a missing quoted column as
    # a constant expression while building the index.
    _add_compatible_columns(models)
    db.create_tables(models)
    _add_compatible_columns(models)


def _add_compatible_columns(models: list[type[Model]]) -> None:
    for model in models:
        table_name = model._meta.table_name
        if not db.table_exists(table_name):
            continue
        existing_columns = {column.name for column in db.get_columns(table_name)}
        if table_name == 'player' and 'shaft_test_whitelisted' not in existing_columns:
            db.execute_sql(
                'ALTER TABLE "player" '
                'ADD COLUMN "shaft_test_whitelisted" INTEGER NOT NULL DEFAULT 0'
            )
        if table_name == 'shaftaxis' and 'dislike_count' not in existing_columns:
            db.execute_sql(
                'ALTER TABLE "shaftaxis" '
                'ADD COLUMN "dislike_count" INTEGER NOT NULL DEFAULT 0'
            )
        if table_name == 'shaftaxis' and 'share_token' not in existing_columns:
            db.execute_sql(
                'ALTER TABLE "shaftaxis" '
                'ADD COLUMN "share_token" VARCHAR(64)'
            )
        if table_name == 'shaftaxis' and 'forked_from_id' not in existing_columns:
            db.execute_sql(
                'ALTER TABLE "shaftaxis" '
                'ADD COLUMN "forked_from_id" INTEGER REFERENCES "shaftaxis" ("id") ON DELETE SET NULL'
            )
        if table_name == 'shaftaxis':
            db.execute_sql(
                'CREATE UNIQUE INDEX IF NOT EXISTS "shaftaxis_share_token" '
                'ON "shaftaxis" ("share_token")'
            )
            db.execute_sql(
                'CREATE INDEX IF NOT EXISTS "shaftaxis_forked_from_visibility" '
                'ON "shaftaxis" ("forked_from_id", "visibility")'
            )


def _reset_incompatible_snapshot_tables(models: list[type[Model]]) -> None:
    for model in models:
        if model._meta.table_name != 'gamerun':
            continue
        if not db.table_exists(model._meta.table_name):
            continue
        existing_columns = {column.name for column in db.get_columns(model._meta.table_name)}
        expected_columns = {field.column_name for field in model._meta.sorted_fields}
        if existing_columns != expected_columns:
            db.drop_tables([model], safe=True)


@contextmanager
def atomic_transaction() -> Iterator[object]:
    if db.is_closed():
        db.connect(reuse_if_open=True)
    with db.atomic() as txn:
        yield txn
