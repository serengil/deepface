# built-in dependencies
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# 3rd party dependencies
import pytest

# project dependencies
from deepface.modules.database.neo4j import Neo4jClient


@pytest.mark.parametrize(
    "connection_details, expected_uri, expected_auth",
    [
        ("bolt://localhost:7687", "bolt://localhost:7687", None),
        ("neo4j+s://alice:secret@db.example", "neo4j+s://db.example", ("alice", "secret")),
        ("bolt://alice:secret@[::1]:7687", "bolt://[::1]:7687", ("alice", "secret")),
        (
            "neo4j://alice%40team:p%3Aa%40ss@db.example:7687?policy=readers",
            "neo4j://db.example:7687?policy=readers",
            ("alice@team", "p:a@ss"),
        ),
        ("bolt://alice:secret@localhost:7687", "bolt://localhost:7687", ("alice", "secret")),
    ],
)
def test_connection_uri_keeps_driver_address(monkeypatch, connection_details,
                                            expected_uri, expected_auth):
    driver = MagicMock()
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=driver))

    client = Neo4jClient(connection_details=connection_details)

    driver.driver.assert_called_once_with(expected_uri, auth=expected_auth)
    assert client.conn is driver.driver.return_value


def test_caller_connection_is_reused(monkeypatch):
    driver = MagicMock()
    connection = MagicMock()
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=driver))

    assert Neo4jClient(connection=connection).conn is connection
    driver.driver.assert_not_called()
