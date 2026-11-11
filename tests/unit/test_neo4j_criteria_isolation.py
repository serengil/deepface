"""Native vector queries should work across independent Neo4j databases and ID types."""

from unittest.mock import MagicMock, Mock

import pytest

from deepface.modules.database import neo4j

pytest.importorskip("neo4j")


def driver(allow_gds=True):
    conn = MagicMock()
    session = conn.session.return_value.__enter__.return_value
    if not allow_gds:
        session.run.side_effect = RuntimeError("Unknown function gds.version")
    return conn, session


def test_native_vector_client_does_not_require_gds():
    conn, session = driver(allow_gds=False)
    client = neo4j.Neo4jClient(connection=conn)
    assert client.conn is conn
    session.run.assert_not_called()


def test_ann_identity_and_verified_edges_use_the_public_key(monkeypatch):
    conn, session = driver()
    client = neo4j.Neo4jClient(connection=conn)
    monkeypatch.setattr(client, "initialize_database", lambda **unused: None)
    session.run.return_value = []
    client.search_by_vector([1.0, 0.0])
    statement = session.run.call_args.args[0]
    assert "coalesce(node.id, elementId(node)) AS id" in statement
    write = Mock(return_value=1)
    monkeypatch.setattr(client, "_Neo4jClient__run_in_batches", write)
    client.link_verified_identities([[41, "4:element:9"]])
    args = write.call_args.kwargs
    assert args["payload"] == [{"src": 41, "dst": "4:element:9"}]
    assert "coalesce(a.id, elementId(a)) = p.src" in args["query"]
    assert "coalesce(b.id, elementId(b)) = p.dst" in args["query"]
