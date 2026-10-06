# built-in dependencies
import os

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import datastore
from deepface.modules.database.inventory import database_inventory
from deepface.modules.exceptions import DuplicateEntryError

# run docker/docker-compose.yml first, or point the env variables to running servers
connection_details = {
    "redis": os.environ.get("DEEPFACE_REDIS_URI", "redis://localhost:6379/15"),
    "cassandra": os.environ.get(
        "DEEPFACE_CASSANDRA_URI", "cassandra://localhost:9042/deepface_test"
    ),
    "mysql": os.environ.get(
        "DEEPFACE_MYSQL_URI", "mysql://deepface_user:deepface_pass@localhost:3306/deepface"
    ),
}


def _record(seed: int, model_name: str = "Facenet", l2_normalized: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "img_name": f"img{seed}.jpg",
        "face": rng.random((8, 8, 3)),
        "model_name": model_name,
        "detector_backend": "opencv",
        "aligned": True,
        "l2_normalized": l2_normalized,
        "embedding": rng.standard_normal(128).tolist(),
    }


def _flush(database_type: str, db_client) -> None:
    if database_type == "redis":
        db_client.conn.flushdb()
    elif database_type == "mysql":
        with db_client.conn.cursor() as cur:
            for table in ["embeddings_index_chunks", "embeddings_index", "embeddings"]:
                cur.execute(f"DELETE FROM {table}")
            # restart ids from 1, as tests assert them
            cur.execute("ALTER TABLE embeddings AUTO_INCREMENT = 1")
        db_client.conn.commit()
    else:
        for table in [
            "embeddings",
            "embeddings_by_criteria",
            "embeddings_buckets",
            "embeddings_by_hash",
            "sequences",
            "embeddings_index",
            "embeddings_index_chunks",
        ]:
            db_client.session.execute(f"TRUNCATE {db_client.keyspace}.{table}")


@pytest.fixture(params=["redis", "cassandra", "mysql"])
def database_type(request):
    return request.param


@pytest.fixture
def client(database_type):
    client_class = database_inventory[database_type]["client"]
    db_client = client_class(connection_details=connection_details[database_type])
    _flush(database_type, db_client)
    yield db_client
    db_client.close()


def test_embeddings_round_trip_losslessly(client):
    records = [_record(i) for i in range(3)]
    assert client.insert_embeddings(records) == 3

    fetched = client.fetch_all_embeddings(
        model_name="Facenet", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    assert [r["id"] for r in fetched] == [1, 2, 3]
    assert [r["img_name"] for r in fetched] == ["img0.jpg", "img1.jpg", "img2.jpg"]
    for source, target in zip(records, fetched):
        assert target["embedding"] == source["embedding"]

    single = client.fetch_embedding(identity_id=2)
    assert single["embedding"] == records[1]["embedding"]
    assert single["model_name"] == "Facenet"
    assert single["aligned"] is True
    assert single["l2_normalized"] is False

    assert client.fetch_embedding(identity_id=999) is None


def test_fetch_all_filters_by_criteria(client):
    client.insert_embeddings(
        [_record(0), _record(1, model_name="ArcFace"), _record(2, l2_normalized=True)]
    )
    fetched = client.fetch_all_embeddings(
        model_name="ArcFace", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    assert [r["img_name"] for r in fetched] == ["img1.jpg"]


def test_fetch_all_spans_batches_in_id_order(client):
    client.insert_embeddings([_record(i) for i in range(25)], batch_size=7)
    fetched = client.fetch_all_embeddings(
        model_name="Facenet",
        detector_backend="opencv",
        aligned=True,
        l2_normalized=False,
        batch_size=4,
    )
    assert [r["id"] for r in fetched] == list(range(1, 26))


def test_duplicates(client):
    assert client.insert_embeddings([_record(0)]) == 1
    # single duplicate is skipped with a warning
    assert client.insert_embeddings([_record(0)]) == 0
    with pytest.raises(DuplicateEntryError):
        client.insert_embeddings([_record(1), _record(0)])
    # claims of the rejected batch are released, so the new record can be inserted later
    assert client.insert_embeddings([_record(1)]) == 1

    fetched = client.fetch_all_embeddings(
        model_name="Facenet", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    assert [r["img_name"] for r in fetched] == ["img0.jpg", "img1.jpg"]


def test_search_by_id_skips_missing(client):
    client.insert_embeddings([_record(i) for i in range(3)])
    assert client.search_by_id([3, 1, 42]) == [
        {"id": 1, "img_name": "img0.jpg"},
        {"id": 3, "img_name": "img2.jpg"},
    ]
    assert client.search_by_id([]) == []


def test_embeddings_index_upsert(client):
    with pytest.raises(ValueError, match="build_index"):
        client.get_embeddings_index("Facenet", "opencv", True, False)

    # larger than a single chunk in cassandra and mysql
    large = os.urandom(3 * 1024 * 1024 + 17)
    client.upsert_embeddings_index("Facenet", "opencv", True, False, large)
    assert client.get_embeddings_index("Facenet", "opencv", True, False) == large

    client.upsert_embeddings_index("Facenet", "opencv", True, False, b"second")
    assert client.get_embeddings_index("Facenet", "opencv", True, False) == b"second"


def test_build_index_and_ann_search(monkeypatch, client, database_type):
    pytest.importorskip("faiss")

    records = [_record(i) for i in range(5)]
    client.insert_embeddings(records)

    datastore.build_index(
        model_name="Facenet",
        detector_backend="opencv",
        database_type=database_type,
        connection_details=connection_details[database_type],
    )

    target = records[3]["embedding"]
    monkeypatch.setattr(datastore, "represent", lambda **kw: [{"embedding": target}])

    frames = datastore.search(
        img="synthetic",
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="euclidean",
        search_method="ann",
        database_type=database_type,
        connection_details=connection_details[database_type],
    )
    assert frames[0]["img_name"].tolist()[0] == "img3.jpg"
    assert frames[0]["distance"].tolist()[0] == pytest.approx(0.0, abs=1e-3)
