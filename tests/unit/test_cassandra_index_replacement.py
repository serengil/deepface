"""Readers can retry when an index publication removes their old version's chunks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepface.modules.database.cassandra import CassandraClient


def client_for(versions, chunks):
    client = CassandraClient.__new__(CassandraClient)
    client.keyspace = "gallery"
    client.SimpleStatement = lambda query, **kwargs: query
    client.session = Mock()
    state = {"pointers": 0, "chunks": 0}

    def execute(query, args):
        if "SELECT version, chunks" in query:
            index = min(state["pointers"], len(versions) - 1)
            state["pointers"] += 1
            return SimpleNamespace(one=lambda: versions[index])
        state["chunks"] += 1
        return [SimpleNamespace(data=data) for data in chunks[args[0]]]

    client.session.execute.side_effect = execute
    return client, state


def read(client):
    return client.get_embeddings_index("VGG-Face", "opencv", True, False)


def test_reader_recovers_from_old_chunks_removed_during_publication():
    client, state = client_for(
        [
            SimpleNamespace(version="old", chunks=2),
            SimpleNamespace(version="new", chunks=2),
        ],
        {"old": [], "new": [b"new-", b"index"]},
    )
    assert read(client) == b"new-index"
    assert state == {"pointers": 2, "chunks": 2}


def test_stable_complete_version_uses_one_read():
    client, state = client_for(
        [SimpleNamespace(version="one", chunks=2)], {"one": [b"a", b"b"]}
    )
    assert read(client) == b"ab"
    assert state == {"pointers": 1, "chunks": 1}


def test_incomplete_current_version_is_still_reported():
    client, _ = client_for(
        [SimpleNamespace(version="broken", chunks=2)], {"broken": [b"a"]}
    )
    with pytest.raises(ValueError, match="incomplete: found 1 of 2"):
        read(client)


def test_continuously_replaced_indexes_do_not_retry_forever():
    client, state = client_for(
        [
            SimpleNamespace(version="old", chunks=1),
            SimpleNamespace(version="new", chunks=1),
        ],
        {"old": [], "new": []},
    )
    with pytest.raises(ValueError, match="incomplete"):
        read(client)
    assert state["pointers"] <= 2


def test_missing_index_keeps_existing_error():
    client, _ = client_for([None], {})
    with pytest.raises(ValueError, match="build_index"):
        read(client)
