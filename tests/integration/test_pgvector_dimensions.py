"""Exercise pgvector dimensional limits using a real PostgreSQL extension."""

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


@pytest.fixture
def database(monkeypatch):
    uri = os.environ.get("DEEPFACE_POSTGRES_URI")
    if uri is None:
        pytest.skip("Set DEEPFACE_POSTGRES_URI to run the pgvector integration tests")
    conn = psycopg.connect(uri)
    schema = "test_dimensions_" + uuid.uuid4().hex
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
    conn.execute(
        psycopg.sql.SQL("SET search_path TO {}, public").format(psycopg.sql.Identifier(schema))
    )
    conn.commit()
    # Isolate the existing process-level schema cache between independent test schemas.
    monkeypatch.setattr(backend, "_SCHEMA_CHECKED", {})
    try:
        yield conn, schema
    finally:
        conn.rollback()
        conn.execute("SET search_path TO public")
        conn.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
        )
        conn.commit()
        conn.close()


def metadata(monkeypatch, dimensions):
    # Model output metadata is the boundary under test; no recognition weights are needed.
    monkeypatch.setattr(
        backend, "build_model", lambda **kwargs: SimpleNamespace(output_shape=dimensions)
    )


def criteria(normalized):
    return dict(
        model_name="VGG-Face", detector_backend="skip", aligned=True, l2_normalized=normalized
    )


def vectors(dimensions):
    values = np.zeros((3, dimensions), dtype=np.float32)
    values[0, 0] = 1
    values[1, :2] = [0.8, 0.6]
    values[2, 1] = 1
    return values


@pytest.mark.parametrize("dimensions", [128, 2000, 2001, 4096])
@pytest.mark.parametrize("normalized", [False, True])
def test_register_and_search_preserve_full_embedding_dimensions(
    database, monkeypatch, dimensions, normalized
):
    conn, schema = database
    metadata(monkeypatch, dimensions)
    adapter = backend.PGVectorClient(connection=conn)
    embeddings = vectors(dimensions)
    names = ["closest.png", "near.png", "far.png"]
    records = [
        dict(
            **criteria(normalized),
            face=np.full((2, 2, 3), index, dtype=np.float32),
            embedding=vector.tolist(),
            img_name=names[index],
        )
        for index, vector in enumerate(embeddings)
    ]
    assert adapter.insert_embeddings(records) == 3
    results = adapter.search_by_vector(embeddings[0].tolist(), limit=3, **criteria(normalized))
    assert [row["img_name"] for row in results] == names
    expected = [0, 0.2, 1] if normalized else [0, np.sqrt(0.4), np.sqrt(2)]
    np.testing.assert_allclose([row["distance"] for row in results], expected, atol=1e-6)
    for row, expected_embedding in zip(results, embeddings):
        table = "embeddings_vggface_skip_aligned_" + ("norm" if normalized else "raw")
        stored_dimensions, stored_text = conn.execute(
            psycopg.sql.SQL(
                "SELECT vector_dims(embedding), embedding::text FROM {} WHERE id = %s"
            ).format(psycopg.sql.Identifier(schema, table)),
            (row["id"],),
        ).fetchone()
        assert stored_dimensions == dimensions
        np.testing.assert_allclose(json.loads(stored_text), expected_embedding, atol=1e-7)
    hnsw_indexes = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexdef LIKE %s",
        (schema, "%USING hnsw%"),
    ).fetchall()
    assert len(hnsw_indexes) == int(dimensions <= 2000)
    if hnsw_indexes:
        assert ("vector_cosine_ops" if normalized else "vector_l2_ops") in hnsw_indexes[0][0]
    assert not conn.closed


@pytest.mark.parametrize("normalized", [False, True])
def test_empty_default_model_gallery_is_searchable(database, monkeypatch, normalized):
    conn, _ = database
    metadata(monkeypatch, 4096)
    adapter = backend.PGVectorClient(connection=conn)
    assert adapter.search_by_vector(vectors(4096)[0].tolist(), **criteria(normalized)) == []


@pytest.mark.parametrize("normalized", [False, True])
def test_public_registration_accepts_default_vgg_face_dimensions(database, monkeypatch, normalized):
    conn, _ = database
    metadata(monkeypatch, 4096)
    embedding = vectors(4096)[0].tolist()
    monkeypatch.setattr(
        datastore,
        "__get_embeddings",
        lambda **kwargs: [
            {
                "img_index": 0,
                "face": np.zeros((2, 2, 3), dtype=np.float32),
                "embedding": embedding,
            }
        ],
    )
    assert datastore.register(
        np.zeros((2, 2, 3), dtype=np.uint8),
        img_name="default-model.png",
        detector_backend="skip",
        l2_normalize=normalized,
        database_type="pgvector",
        connection=conn,
    ) == {"inserted": 1}
    adapter = backend.PGVectorClient(connection=conn)
    results = adapter.search_by_vector(embedding, **criteria(normalized))
    assert results[0]["img_name"] == "default-model.png"
    assert not conn.closed


def test_invalid_vector_column_dimensions_still_raise(database, monkeypatch):
    conn, _ = database
    metadata(monkeypatch, 16001)
    adapter = backend.PGVectorClient(connection=conn)
    with pytest.raises(psycopg.Error, match="dimensions"):
        adapter.initialize_database(**criteria(False))
