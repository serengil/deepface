# built-in dependencies
import sqlite3

# 3rd party dependencies
import numpy as np
import pytest

# project dependencies
from deepface.modules import datastore
from deepface.modules.database.sqlite import SqliteClient
from deepface.modules.exceptions import DuplicateEntryError


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


@pytest.fixture
def client(tmp_path):
    db_client = SqliteClient(connection_details=str(tmp_path / "deepface.db"))
    yield db_client
    db_client.close()


def test_embeddings_round_trip_losslessly(client):
    records = [_record(i) for i in range(3)]
    assert client.insert_embeddings(records) == 3

    fetched = client.fetch_all_embeddings(
        model_name="Facenet", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    assert [r["img_name"] for r in fetched] == ["img0.jpg", "img1.jpg", "img2.jpg"]
    for source, target in zip(records, fetched):
        assert target["embedding"] == source["embedding"]

    single = client.fetch_embedding(identity_id=fetched[1]["id"])
    assert single["embedding"] == records[1]["embedding"]
    assert single["aligned"] is True
    assert single["l2_normalized"] is False

    assert client.fetch_embedding(identity_id=999) is None


def test_fetch_all_filters_by_criteria(client):
    client.insert_embeddings([_record(0), _record(1, model_name="ArcFace"), _record(2, l2_normalized=True)])
    fetched = client.fetch_all_embeddings(
        model_name="ArcFace", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    assert [r["img_name"] for r in fetched] == ["img1.jpg"]


def test_duplicates(client):
    assert client.insert_embeddings([_record(0)]) == 1
    # single duplicate is skipped with a warning
    assert client.insert_embeddings([_record(0)]) == 0
    with pytest.raises(DuplicateEntryError):
        client.insert_embeddings([_record(1), _record(0)])


def test_search_by_id_spans_chunks(client):
    client.insert_embeddings([_record(i) for i in range(3)])
    ids = [1, 3] + list(range(1000, 2000))
    assert client.search_by_id(ids) == [
        {"id": 1, "img_name": "img0.jpg"},
        {"id": 3, "img_name": "img2.jpg"},
    ]
    assert client.search_by_id([]) == []


def test_embeddings_index_upsert(client):
    with pytest.raises(ValueError, match="build_index"):
        client.get_embeddings_index("Facenet", "opencv", True, False)

    client.upsert_embeddings_index("Facenet", "opencv", True, False, b"first")
    client.upsert_embeddings_index("Facenet", "opencv", True, False, b"second")
    assert client.get_embeddings_index("Facenet", "opencv", True, False) == b"second"


def test_existing_connection_is_reused(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "deepface.db"))
    SqliteClient(connection=conn).insert_embeddings([_record(0)])
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    conn.close()


def test_build_index_and_ann_search(monkeypatch, tmp_path):
    pytest.importorskip("faiss")
    db_path = str(tmp_path / "deepface.db")

    records = [_record(i) for i in range(5)]
    with_client = SqliteClient(connection_details=db_path)
    with_client.insert_embeddings(records)
    with_client.close()

    datastore.build_index(
        model_name="Facenet",
        detector_backend="opencv",
        database_type="sqlite",
        connection_details=db_path,
    )

    target = records[3]["embedding"]
    monkeypatch.setattr(datastore, "represent", lambda **kw: [{"embedding": target}])

    frames = datastore.search(
        img="synthetic",
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="euclidean",
        search_method="ann",
        database_type="sqlite",
        connection_details=db_path,
    )
    assert frames[0]["img_name"].tolist()[0] == "img3.jpg"
    assert frames[0]["distance"].tolist()[0] == pytest.approx(0.0, abs=1e-3)
