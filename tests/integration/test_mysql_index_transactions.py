"""Check index publication against a real MySQL server, including autocommit clients."""

import os
from hashlib import sha256
from threading import Event, Thread
from time import monotonic

import pytest

from deepface.modules.database.mysql import MySqlClient, _parse_uri

CRITERIA = ("Facenet", "opencv", True, False)
CHUNK_SIZE = 1024 * 1024


@pytest.fixture(params=[False, True], ids=["transactional", "autocommit"])
def mysql_client(request):
    pymysql = pytest.importorskip("pymysql")
    uri = os.environ.get("DEEPFACE_MYSQL_URI")
    if not uri:
        pytest.skip("set DEEPFACE_MYSQL_URI to run real MySQL index tests")
    details = _parse_uri(uri)
    connection = pymysql.connect(**details, autocommit=request.param)
    client = MySqlClient(connection=connection)
    with connection.cursor() as cursor:
        cursor.execute("DROP TRIGGER IF EXISTS reject_index_chunk")
        cursor.execute("DROP TRIGGER IF EXISTS slow_index_chunks")
        cursor.execute("DELETE FROM embeddings_index_chunks")
        cursor.execute("DELETE FROM embeddings_index")
    connection.commit()
    try:
        yield client, details, request.param
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("DROP TRIGGER IF EXISTS reject_index_chunk")
            cursor.execute("DROP TRIGGER IF EXISTS slow_index_chunks")
        connection.commit()
        client.close()


@pytest.mark.parametrize("has_previous_index", [False, True], ids=["first-write", "replacement"])
def test_failed_chunk_write_keeps_the_previous_published_index(mysql_client, has_previous_index):
    client, _details, autocommit = mysql_client
    old_index = b"old-complete-index"
    if has_previous_index:
        client.upsert_embeddings_index(*CRITERIA, old_index)

    with client.conn.cursor() as cursor:
        cursor.execute(
            "CREATE TRIGGER reject_index_chunk BEFORE INSERT ON embeddings_index_chunks "
            "FOR EACH ROW BEGIN "
            "IF NEW.chunk = 1 THEN "
            "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'test chunk write rejected'; "
            "END IF; END"
        )

    with pytest.raises(client.pymysql.err.OperationalError, match="test chunk write rejected"):
        client.upsert_embeddings_index(*CRITERIA, b"new-data" * (CHUNK_SIZE // 4))

    assert client.conn.get_autocommit() is autocommit
    if has_previous_index:
        assert client.get_embeddings_index(*CRITERIA) == old_index
    else:
        with client.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM embeddings_index")
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT COUNT(*) FROM embeddings_index_chunks")
            assert cursor.fetchone()[0] == 0

    # A failed attempt must not prevent a later successful complete replacement.
    client.conn.rollback()
    with client.conn.cursor() as cursor:
        cursor.execute("DROP TRIGGER reject_index_chunk")
    replacement = b"valid-index" * (CHUNK_SIZE // 3)
    client.upsert_embeddings_index(*CRITERIA, replacement)
    assert client.get_embeddings_index(*CRITERIA) == replacement
    assert client.conn.get_autocommit() is autocommit


def test_concurrent_reader_only_observes_complete_index_versions(mysql_client):
    client, details, autocommit = mysql_client
    old_index = b"previous-version"
    new_index = b"new-version" * (CHUNK_SIZE // 4)
    client.upsert_embeddings_index(*CRITERIA, old_index)
    with client.conn.cursor() as cursor:
        cursor.execute(
            "CREATE TRIGGER slow_index_chunks BEFORE INSERT ON embeddings_index_chunks "
            "FOR EACH ROW DO SLEEP(0.15)"
        )

    reader_connection = client.pymysql.connect(**details, autocommit=True)
    reader = MySqlClient(connection=reader_connection)
    finished = Event()
    errors = []

    def publish():
        try:
            client.upsert_embeddings_index(*CRITERIA, new_index)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=publish, daemon=True)
    thread.start()
    observed = []
    complete_versions = {
        (len(value), sha256(value).hexdigest()) for value in (old_index, new_index)
    }
    deadline = monotonic() + 15
    try:
        while not finished.is_set() and monotonic() < deadline:
            try:
                value = reader.get_embeddings_index(*CRITERIA)
                observed.append((len(value), sha256(value).hexdigest()))
            except ValueError:
                observed.append(None)
            finished.wait(0.01)
        assert finished.wait(5), "index writer did not finish"
        assert not errors
        assert observed, "the test did not observe an in-progress replacement"
        assert all(value in complete_versions for value in observed), observed
        assert reader.get_embeddings_index(*CRITERIA) == new_index
        assert client.conn.get_autocommit() is autocommit
    finally:
        thread.join(timeout=5)
        reader.close()
