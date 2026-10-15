import sqlite3

import pytest

from src.domains.rbac.runtime.exec_eval import exec_match_safe, get_cursor_from_path


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "benchmark.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
    return path


def test_sibling_databases_do_not_change_sql_score(database):
    (database.parent / "database.sqlite").touch()
    with sqlite3.connect(database.parent / "unrelated.sqlite") as connection:
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.execute("INSERT INTO items VALUES (99)")
    assert exec_match_safe(str(database), "SELECT 3", "SELECT SUM(value) FROM items")
    assert not exec_match_safe(str(database), "SELECT 4", "SELECT SUM(value) FROM items")
    assert not exec_match_safe(str(database), "SELECT missing FROM items", "SELECT value FROM items")


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    assert not exec_match_safe(str(path), "SELECT 1", "SELECT 1")
    assert not path.exists()


def test_database_connection_is_read_only(database):
    cursor = get_cursor_from_path(str(database))
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            cursor.execute("DELETE FROM items")
        assert cursor.execute("SELECT COUNT(*) FROM items").fetchone() == (2,)
    finally:
        cursor.connection.close()


def test_gold_self_match_and_ordering(database):
    gold = "SELECT value FROM items ORDER BY value"
    assert exec_match_safe(str(database), gold, gold)
    assert not exec_match_safe(str(database), gold + " DESC", gold)
