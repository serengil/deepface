"""Exercise real gallery read plans and schema upgrades without timing assertions."""

import json
import os
import sqlite3
import statistics
import struct
import time
import uuid

import pytest

from deepface.modules.database.mongo import MongoDbClient
from deepface.modules.database.postgres import PostgresClient, _SCHEMA_CHECKED
from deepface.modules.database.sqlite import SqliteClient

ROW_COUNT = 20000
MATCH_IDS = list(range(1, ROW_COUNT + 1, 500))
CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)
QUERY = """
    SELECT id, img_name, embedding FROM embeddings
    WHERE model_name = %s AND detector_backend = %s AND aligned = %s AND l2_normalized = %s
    ORDER BY id ASC
"""
COLUMNS = (
    "img_name, face, face_shape, model_name, detector_backend, aligned, "
    "l2_normalized, embedding, face_hash, embedding_hash"
)


def records(binary):
    for index in range(ROW_COUNT):
        vector = [float(index), 0.5]
        yield (
            f"image-{index}.png",
            b"\0" * 12,
            "[1, 1, 3]" if binary else [1, 1, 3],
            "Facenet" if index % 500 == 0 else "VGG-Face",
            "skip",
            True,
            False,
            struct.pack("<2d", *vector) if binary else vector,
            str(index),
            str(index),
        )


@pytest.fixture(scope="module")
def sqlite_database():
    connection = sqlite3.connect(":memory:")
    client = SqliteClient(connection=connection)
    connection.executemany(
        f"INSERT INTO embeddings ({COLUMNS}) VALUES ({', '.join(['?'] * 10)})",
        records(binary=True),
    )
    connection.commit()
    try:
        yield "sqlite", client
    finally:
        connection.close()


@pytest.fixture(scope="module")
def postgres_database():
    psycopg = pytest.importorskip("psycopg")
    uri = os.environ.get("DEEPFACE_POSTGRES_URI")
    if not uri:
        pytest.skip("set DEEPFACE_POSTGRES_URI to an isolated test server")
    schema = "deepface_" + uuid.uuid4().hex
    administrator = psycopg.connect(uri, autocommit=True)
    administrator.execute(f"CREATE SCHEMA {schema}")
    connection = psycopg.connect(uri, options=f"-c search_path={schema}")
    try:
        client = PostgresClient(connection=connection)
        with connection.cursor() as cursor:
            with cursor.copy(f"COPY embeddings ({COLUMNS}) FROM STDIN") as copy:
                for record in records(binary=False):
                    copy.write_row(record)
            cursor.execute("ANALYZE embeddings")
        connection.commit()
        yield "postgres", client
    finally:
        connection.close()
        administrator.execute(f"DROP SCHEMA {schema} CASCADE")
        administrator.close()


@pytest.fixture(scope="module")
def mongo_database():
    pymongo = pytest.importorskip("pymongo")
    uri = os.environ.get("DEEPFACE_MONGO_URI")
    if not uri:
        pytest.skip("set DEEPFACE_MONGO_URI to an isolated test server")
    name = "deepface_" + uuid.uuid4().hex
    connection = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)
    client = MongoDbClient(connection=connection, db_name=name)
    columns = COLUMNS.replace(" ", "").split(",")
    documents = [
        dict(zip(columns, record), sequence=index + 1)
        for index, record in enumerate(records(binary=False))
    ]
    client.embeddings.insert_many(documents)
    try:
        yield "mongo", client
    finally:
        connection.drop_database(name)
        connection.close()


@pytest.fixture(params=["sqlite", "postgres", "mongo"])
def database(request):
    return request.getfixturevalue(request.param + "_database")


def gallery_rows(client):
    rows = client.fetch_all_embeddings(**CRITERIA, batch_size=7)
    assert [row["id"] for row in rows] == MATCH_IDS
    assert [row["img_name"] for row in rows] == [f"image-{i - 1}.png" for i in MATCH_IDS]
    assert [row["embedding"] for row in rows] == [[float(i - 1), 0.5] for i in MATCH_IDS]
    return rows


def walk_plan(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from walk_plan(child)


def gallery_work(database):
    backend, client = database
    if backend == "sqlite":
        steps = [0]

        def progress():
            steps[0] += 100
            return 0

        client.conn.set_progress_handler(progress, 100)
        try:
            gallery_rows(client)
        finally:
            client.conn.set_progress_handler(None, 0)
        plan = client.conn.execute(
            "EXPLAIN QUERY PLAN " + QUERY.replace("%s", "?"), tuple(CRITERIA.values())
        ).fetchall()
        return dict(work=steps[0], limit=ROW_COUNT // 10, plan=plan)
    if backend == "postgres":
        gallery_rows(client)
        plan = client.conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + QUERY, tuple(CRITERIA.values())
        ).fetchone()[0][0]["Plan"]
        filtered = sum(node.get("Rows Removed by Filter", 0) for node in walk_plan(plan))
        return dict(work=filtered, limit=ROW_COUNT // 10, plan=plan)
    gallery_rows(client)
    plan = (
        client.embeddings.find(CRITERIA, {"sequence": 1, "img_name": 1, "embedding": 1})
        .sort("sequence", 1)
        .explain()
    )
    return dict(
        work=plan["executionStats"]["totalDocsExamined"],
        limit=len(MATCH_IDS),
        plan=plan["queryPlanner"]["winningPlan"],
    )


def report_work(database, label):
    backend, client = database
    metrics = gallery_work(database)
    durations = []
    for _ in range(7):
        start = time.perf_counter()
        gallery_rows(client)
        durations.append(1000 * (time.perf_counter() - start))
    print(
        json.dumps(
            dict(
                backend=backend,
                case=label,
                phase=os.environ.get("DEEPFACE_BENCHMARK_PHASE", "test"),
                median_ms=statistics.median(durations),
                **metrics,
            )
        )
    )
    assert metrics["work"] <= metrics["limit"], metrics


def test_gallery_reads_avoid_scanning_unrelated_configurations(database):
    report_work(database, "gallery")


def test_identity_and_gallery_results_are_preserved(database):
    _, client = database
    gallery_rows(client)
    row = client.fetch_embedding(ROW_COUNT)
    assert row["id"] == ROW_COUNT
    assert row["embedding"] == [float(ROW_COUNT - 1), 0.5]
    assert row["model_name"] == "VGG-Face"
    assert {row["id"] for row in client.search_by_id([1, ROW_COUNT])} == {1, ROW_COUNT}


@pytest.mark.parametrize("ids", [[ROW_COUNT], [ROW_COUNT - 1, ROW_COUNT]])
def test_mongo_identity_reads_examine_only_matching_records(mongo_database, ids):
    _, client = mongo_database
    query = {"sequence": ids[0]} if len(ids) == 1 else {"sequence": {"$in": ids}}
    plan = client.embeddings.find(query).explain()
    statistics_ = plan["executionStats"]
    assert statistics_["nReturned"] == len(ids)
    print(json.dumps(dict(case="mongo-identities", ids=ids, statistics=statistics_)))
    assert statistics_["totalDocsExamined"] <= len(ids)


def test_existing_database_receives_indexes_during_initialization(database):
    backend, client = database
    if backend == "mongo":
        for name in ("ix_embeddings_criteria", "ix_embeddings_sequence"):
            if name in client.embeddings.index_information():
                client.embeddings.drop_index(name)
    else:
        client.conn.execute("DROP INDEX IF EXISTS ix_embeddings_criteria")
        client.conn.commit()
        if backend == "postgres":
            _SCHEMA_CHECKED.pop(client.conn.info.dsn, None)
    client.initialize_database()
    client.initialize_database()
    report_work(database, "upgrade")
