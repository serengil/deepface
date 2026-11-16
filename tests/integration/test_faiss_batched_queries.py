"""Keep FAISS batch results identical to individual searches while issuing one query."""

import json
import sqlite3
import statistics
import time

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from deepface.modules import datastore
from deepface.modules.database.sqlite import SqliteClient


@pytest.fixture
def gallery():
    connection = sqlite3.connect(":memory:")
    client = SqliteClient(connection=connection)
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(2)
    try:
        yield connection, client
    finally:
        faiss.omp_set_num_threads(previous_threads)
        connection.close()


def configure_gallery(client, normalized):
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype="float32")
    if normalized:
        faiss.normalize_L2(vectors)
    else:
        vectors *= 10
    criteria = dict(
        model_name="Facenet512", detector_backend="skip", aligned=True, l2_normalized=normalized
    )
    records = [
        dict(
            img_name=f"image-{index}.png",
            face=np.full((2, 2, 3), index, dtype=np.uint8),
            embedding=vector.tolist(),
            **criteria,
        )
        for index, vector in enumerate(vectors)
    ]
    assert client.insert_embeddings(records) == 4
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(2, 16))
    index.add_with_ids(vectors, np.arange(1, 5, dtype="int64"))
    client.upsert_embeddings_index(**criteria, index_data=faiss.serialize_index(index).tobytes())
    return vectors


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("similarity", [False, True])
@pytest.mark.parametrize("k", [None, 1, 8])
@pytest.mark.parametrize("face_count", [1, 3])
def test_batched_ann_matches_individual_faces(
    gallery, monkeypatch, normalized, similarity, k, face_count
):
    connection, client = gallery
    vectors = configure_gallery(client, normalized)
    first_query = -vectors[0] if normalized else -100 * vectors[0]
    queries = [first_query, vectors[0], vectors[2]][:face_count]
    faces = [
        {
            "embedding": vector.tolist(),
            "facial_area": {"x": i * 10, "y": i + 1, "w": 4, "h": 7},
        }
        for i, vector in enumerate(queries)
    ]
    current_faces = []
    monkeypatch.setattr(datastore, "represent", lambda **kwargs: current_faces)
    calls = []
    original_search = faiss.IndexIDMap.search

    def measured_search(self, vectors, limit, *args, **kwargs):
        calls.append(len(vectors))
        return original_search(self, vectors, limit, *args, **kwargs)

    monkeypatch.setattr(faiss.IndexIDMap, "search", measured_search)

    def search():
        return datastore.search(
            img="precomputed-embeddings",
            model_name="Facenet512",
            detector_backend="skip",
            l2_normalize=normalized,
            similarity_search=similarity,
            k=k,
            search_method="ann",
            database_type="sqlite",
            connection=connection,
            return_type="dict",
        )

    expected = []
    for face in faces:
        current_faces[:] = [face]
        expected.extend(search())
    current_faces[:] = faces
    calls.clear()

    actual = search()

    assert actual == expected
    assert len(actual) == face_count
    assert calls == [face_count]
    assert all(row["id"] != -1 for matches in actual for row in matches)
    if not normalized and not similarity:
        assert actual[0] == []


@pytest.mark.parametrize("nested", [False, True])
def test_empty_face_batch_does_not_query_faiss(gallery, monkeypatch, nested):
    connection, client = gallery
    configure_gallery(client, normalized=False)
    monkeypatch.setattr(datastore, "represent", lambda **kwargs: [[]] if nested else [])

    def unexpected_search(*args, **kwargs):
        raise AssertionError("an empty face batch must not query FAISS")

    monkeypatch.setattr(faiss.IndexIDMap, "search", unexpected_search)

    def search():
        return datastore.search(
            img=[],
            model_name="Facenet512",
            detector_backend="skip",
            search_method="ann",
            database_type="sqlite",
            connection=connection,
            return_type="dict",
        )

    if nested:
        assert search() == []
    else:
        with pytest.raises(ValueError, match="No embeddings were detected"):
            search()


def test_gallery_names_are_fetched_once_for_all_faces(gallery, monkeypatch):
    connection, client = gallery
    vectors = configure_gallery(client, normalized=False)
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": vector.tolist()} for vector in vectors[:3]],
    )
    calls = []
    original_lookup = SqliteClient.search_by_id

    def measured_lookup(self, ids):
        calls.append(list(ids))
        return original_lookup(self, ids)

    monkeypatch.setattr(SqliteClient, "search_by_id", measured_lookup)
    matches = datastore.search(
        img="precomputed-embeddings",
        model_name="Facenet512",
        detector_backend="skip",
        similarity_search=True,
        k=8,
        search_method="ann",
        database_type="sqlite",
        connection=connection,
        return_type="dict",
    )
    assert len(matches) == 3
    assert all(len(face) == 4 for face in matches)
    assert all(row["img_name"] == f"image-{row['id'] - 1}.png" for face in matches for row in face)
    assert len(calls) == 1
    assert set(calls[0]) == {1, 2, 3, 4}
    assert len(calls[0]) == 4


def test_real_hnsw_batch_search_measurement(gallery, monkeypatch):
    rng = np.random.default_rng(714)
    vectors = rng.standard_normal((10000, 128)).astype("float32")
    queries = rng.standard_normal((128, 128)).astype("float32")
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(128, 32))
    faiss.downcast_index(index.index).hnsw.efSearch = 64
    index.add_with_ids(vectors, np.arange(10000, dtype="int64"))

    def individual():
        results = [index.search(query.reshape(1, -1), 20) for query in queries]
        return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])

    expected = individual()
    actual = index.search(queries, 20)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    measurements = {}
    for name, operation in (
        ("individual", individual),
        ("batch", lambda: index.search(queries, 20)),
    ):
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            operation()
            durations.append(1000 * (time.perf_counter() - start))
        measurements[name] = statistics.median(durations)
    connection, client = gallery
    criteria = dict(
        model_name="Facenet512", detector_backend="skip", aligned=True, l2_normalized=False
    )
    client.insert_embeddings(
        [
            dict(
                img_name=f"image-{i}.png",
                face=np.zeros((1, 1, 3), dtype=np.uint8),
                embedding=vector.tolist(),
                **criteria,
            )
            for i, vector in enumerate(vectors)
        ]
    )
    # SQLite identities start at one.
    stored_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(128, 32))
    faiss.downcast_index(stored_index.index).hnsw.efSearch = 64
    stored_index.add_with_ids(vectors, np.arange(1, 10001, dtype="int64"))
    client.upsert_embeddings_index(
        **criteria, index_data=faiss.serialize_index(stored_index).tobytes()
    )
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": vector.tolist()} for vector in queries],
    )

    def search():
        return datastore.search(
            img="precomputed-embeddings",
            model_name="Facenet512",
            detector_backend="skip",
            similarity_search=True,
            k=20,
            search_method="ann",
            database_type="sqlite",
            connection=connection,
            return_type="dict",
        )

    reference = search()
    assert len(reference) == 128
    assert all(len(matches) == 20 for matches in reference)
    durations = []
    for _ in range(7):
        start = time.perf_counter()
        actual = search()
        durations.append(1000 * (time.perf_counter() - start))
        assert actual == reference

    print(
        json.dumps(
            dict(
                faiss=faiss.__version__,
                threads=faiss.omp_get_max_threads(),
                gallery=10000,
                dimensions=128,
                queries=128,
                k=20,
                search_median_ms=measurements,
                datastore_median_ms=statistics.median(durations),
            )
        )
    )
