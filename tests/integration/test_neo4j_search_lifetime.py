"""Owned Neo4j connections close after asynchronous graph writes, including errors."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from deepface.modules import datastore
from deepface.modules.database.neo4j import Neo4jClient

LABEL = "FacenetSkipAlignedRaw"


@pytest.fixture
def graph(monkeypatch):
    neo4j = pytest.importorskip("neo4j")
    uri = os.environ.get("DEEPFACE_NEO4J_URI")
    if not uri:
        pytest.skip("set DEEPFACE_NEO4J_URI to run real Neo4j lifetime tests")
    administrator = neo4j.GraphDatabase.driver(uri)
    administrator.verify_connectivity()
    with administrator.session() as session:
        session.run(f"MATCH (n:{LABEL}) DETACH DELETE n").consume()
    monkeypatch.setattr(datastore, "represent", lambda **_kwargs: [{"embedding": [1.0, 0.0]}])
    clients = []
    original_connect = datastore.__connect_database

    def connect(**kwargs):
        client = original_connect(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(datastore, "__connect_database", connect)
    try:
        yield administrator, uri, clients
    finally:
        for client in clients:
            if client.conn is not administrator:
                client.close()
        with administrator.session() as session:
            session.run(f"MATCH (n:{LABEL}) DETACH DELETE n").consume()
        administrator.close()


@pytest.mark.parametrize("borrowed", [False, True], ids=["owned", "borrowed"])
@pytest.mark.parametrize("outcome", ["linked", "single", "empty", "link-error"])
def test_graph_search_releases_owned_driver_after_its_last_write(
    graph, monkeypatch, borrowed, outcome
):
    administrator, uri, clients = graph
    count = 0 if outcome == "empty" else 1 if outcome == "single" else 2
    with administrator.session() as session:
        session.run(
            f"UNWIND range(1, $count) AS i CREATE (n:{LABEL}) "
            "SET n.embedding = [1.0, 0.0], n.img_name = toString(i)",
            count=count,
        ).consume()

    started, release = Event(), Event()
    original_link = Neo4jClient.link_verified_identities

    def delayed_link(self, **kwargs):
        started.set()
        assert release.wait(timeout=10), "the test did not release graph writes"
        assert self.conn._closed is False
        if outcome == "link-error":
            raise RuntimeError("test background write failed")
        return original_link(self, **kwargs)

    monkeypatch.setattr(Neo4jClient, "link_verified_identities", delayed_link)
    connection = administrator if borrowed else None
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(datastore, "_LINK_EXECUTOR", executor)
        try:
            options = dict(
                img="fixed-query",
                model_name="Facenet",
                detector_backend="skip",
                database_type="neo4j",
                connection_details=uri,
                connection=connection,
                search_method="exact",
                return_type="dict",
            )
            if outcome == "empty":
                with pytest.raises(ValueError, match="No embeddings found"):
                    datastore.search(**options)
            else:
                rows = datastore.search(**options)[0]
                assert len(rows) == count
                assert all(row["distance"] == 0 for row in rows)
            assert len(clients) == 1
            if outcome in ("linked", "link-error"):
                assert started.wait(timeout=10)
                # Search has returned, but the worker still needs the actual driver.
                assert clients[0].conn._closed is False
                with administrator.session() as session:
                    assert (
                        session.run(
                            f"MATCH (:{LABEL})-[r:VERIFIED]->(:{LABEL}) RETURN count(r) AS total"
                        ).single()["total"]
                        == 0
                    )
            else:
                assert not started.is_set()
        finally:
            release.set()

    # Joining the single worker includes any queued close after its link operation.
    assert clients[0].conn._closed is (not borrowed)
    if not borrowed:
        assert not clients[0].conn._pool.connections
    else:
        administrator.verify_connectivity()

    with administrator.session() as session:
        total = session.run(
            f"MATCH (:{LABEL})-[r:VERIFIED]->(:{LABEL}) RETURN count(r) AS total"
        ).single()["total"]
    assert total == (1 if outcome == "linked" else 0)
