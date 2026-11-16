"""Exercise both pgvector Python decoder generations against PostgreSQL."""

import json
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from deepface.modules import datastore
from deepface.modules.database import pgvector as backend

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)
TABLE = "embeddings_facenet_skip_aligned_raw"


@pytest.fixture
def gallery(monkeypatch):
    uri = os.environ.get("DEEPFACE_POSTGRES_URI")
    if uri is None:
        pytest.skip("Set DEEPFACE_POSTGRES_URI for the pgvector integration tests")
    conn = psycopg.connect(uri)
    schema = "test_decoder_" + uuid.uuid4().hex
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
    conn.execute(
        psycopg.sql.SQL("SET search_path TO {}, public").format(psycopg.sql.Identifier(schema))
    )
    conn.commit()
    monkeypatch.setattr(backend, "_SCHEMA_CHECKED", {})
    monkeypatch.setattr(backend, "build_model", lambda **kwargs: SimpleNamespace(output_shape=128))
    adapter = backend.PGVectorClient(connection=conn)
    values = [np.linspace(-0.25 + index, 0.75 + index, 128, dtype=np.float32) for index in range(3)]
    try:
        assert (
            adapter.insert_embeddings(
                [
                    dict(
                        **CRITERIA,
                        face=np.full((2, 2, 3), index, dtype=np.float32),
                        embedding=value.tolist(),
                        img_name=f"face-{index}.png",
                    )
                    for index, value in enumerate(values)
                ]
            )
            == 3
        )
        yield conn, adapter, values
    finally:
        conn.rollback()
        conn.execute("SET search_path TO public")
        conn.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
        )
        conn.commit()
        conn.close()


@pytest.mark.parametrize("identity", [1, 2, 3])
def test_fetch_embedding_decodes_driver_vectors(gallery, identity):
    _, adapter, values = gallery
    record = adapter.fetch_embedding(identity, **CRITERIA)
    assert record["id"] == identity
    assert record["img_name"] == f"face-{identity - 1}.png"
    assert isinstance(record["embedding"], list)
    np.testing.assert_array_equal(record["embedding"], values[identity - 1])


@pytest.mark.parametrize("batch_size", [1, 2, 1000])
def test_streaming_gallery_reads_decode_every_vector(gallery, batch_size):
    _, adapter, values = gallery
    records = adapter.fetch_all_embeddings(**CRITERIA, batch_size=batch_size)
    assert [record["id"] for record in records] == [1, 2, 3]
    for record, expected in zip(records, values):
        assert isinstance(record["embedding"], list)
        np.testing.assert_array_equal(record["embedding"], expected)


def test_public_exact_search_uses_decoded_embeddings(gallery, monkeypatch):
    conn, _, values = gallery
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": values[0].tolist(), "facial_area": {"x": 3, "y": 4}}],
    )
    result = datastore.search(
        "precomputed",
        model_name="Facenet",
        detector_backend="skip",
        database_type="pgvector",
        search_method="exact",
        distance_metric="euclidean",
        similarity_search=True,
        connection=conn,
        return_type="dict",
    )
    assert [record["id"] for record in result[0]] == [1, 2, 3]
    assert [record["distance"] for record in result[0]] == pytest.approx(
        [0, 128**0.5, 2 * 128**0.5], abs=1e-5
    )
    assert not conn.closed


def test_public_identify_uses_decoded_embedding(gallery, monkeypatch):
    conn, _, values = gallery
    monkeypatch.setattr(
        datastore,
        "__get_embeddings",
        lambda **kwargs: [{"embedding": values[0].tolist(), "facial_area": {"x": 5, "y": 6}}],
    )
    result = datastore.identify(
        np.zeros((2, 2, 3), dtype=np.uint8),
        identity_id=1,
        model_name="Facenet",
        detector_backend="skip",
        database_type="pgvector",
        distance_metric="euclidean",
        connection=conn,
    )
    assert result["verified"] is True
    assert result["distance"] == 0
    assert result["img_name"] == "face-0.png"
    assert result["facial_areas"]["img1"] == {"x": 5, "y": 6}
    json.dumps(result)
    assert not conn.closed


def test_missing_identity_still_returns_none(gallery):
    _, adapter, _ = gallery
    assert adapter.fetch_embedding(999, **CRITERIA) is None


def test_empty_gallery_still_returns_empty_list(gallery):
    conn, adapter, _ = gallery
    conn.execute(psycopg.sql.SQL("DELETE FROM {}").format(psycopg.sql.Identifier(TABLE)))
    conn.commit()
    assert adapter.fetch_all_embeddings(**CRITERIA) == []


def test_native_ann_search_remains_usable(gallery):
    _, adapter, values = gallery
    results = adapter.search_by_vector(values[0].tolist(), limit=3, **CRITERIA)
    assert [record["id"] for record in results] == [1, 2, 3]
    assert results[0]["distance"] == 0
