"""Keep normalization, distance, threshold and confidence in the same space."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from deepface.modules import datastore, verification


class MemoryDatabase:
    """Deterministic storage boundary; real search/identify logic runs above it."""

    def __init__(self, records):
        self.records = records
        self.criteria = []
        self.closed = False
        self.neighbours = []

    def fetch_all_embeddings(self, **criteria):
        self.criteria.append(criteria)
        return deepcopy(self.records)

    def fetch_embedding(self, identity_id, **criteria):
        self.criteria.append(criteria)
        return next(
            (deepcopy(row) for row in self.records if row["id"] == identity_id), None
        )

    def search_by_vector(self, **criteria):
        self.criteria.append(criteria)
        return deepcopy(self.neighbours)

    def close(self):
        self.closed = True


@pytest.fixture
def storage(monkeypatch):
    def configure(model, normalized, queries=None, vectors=None):
        queries = [[1.0, 0.0]] if queries is None else queries
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]] if vectors is None else vectors
        records = [
            dict(
                id=index,
                img_name=f"vector-{index}",
                embedding=list(vector),
                model_name=model,
                detector_backend="skip",
                aligned=True,
                l2_normalized=normalized,
            )
            for index, vector in enumerate(vectors)
        ]
        db = MemoryDatabase(records)
        monkeypatch.setattr(datastore, "__connect_database", lambda **kwargs: db)
        monkeypatch.setitem(
            datastore.database_inventory,
            "postgres",
            {"is_vector_db": False, "is_graph_db": False},
        )
        monkeypatch.setitem(
            datastore.database_inventory,
            "pgvector",
            {"is_vector_db": True, "is_graph_db": False},
        )

        def synthetic_represent(**kwargs):
            assert kwargs["l2_normalize"] is normalized
            result = []
            for index, vector in enumerate(queries):
                embedding = np.asarray(vector, dtype=np.float64)
                if kwargs["l2_normalize"]:
                    embedding = verification.l2_normalize(embedding)
                result.append(
                    dict(
                        embedding=embedding.tolist(),
                        facial_area=dict(x=index, y=0, w=1, h=1),
                    )
                )
            return result

        monkeypatch.setattr(datastore, "represent", synthetic_represent)
        options = dict(
            img="synthetic-input",
            model_name=model,
            detector_backend="skip",
            l2_normalize=normalized,
            connection=object(),
        )
        return db, options

    return configure


@pytest.mark.parametrize(
    "model", ["Facenet", "Facenet512", "ArcFace", "GhostFaceNet", "DeepID"]
)
def test_exact_search_uses_normalized_metric_and_threshold(storage, model):
    db, options = storage(model, True)
    actual = datastore.search(
        **options, search_method="exact", distance_metric="euclidean"
    )
    reference = datastore.search(
        **options, search_method="exact", distance_metric="euclidean_l2"
    )
    assert actual[0]["id"].tolist() == [0]
    pd.testing.assert_frame_equal(actual[0], reference[0])
    assert set(actual[0]["distance_metric"]) == {"euclidean_l2"}
    assert set(actual[0]["threshold"]) == {
        verification.find_threshold(model, "euclidean_l2")
    }
    assert all(criteria["l2_normalized"] for criteria in db.criteria)


@pytest.mark.parametrize(
    "model", ["Facenet", "Facenet512", "ArcFace", "GhostFaceNet", "DeepID"]
)
def test_identify_uses_normalized_metric_threshold_and_confidence(storage, model):
    db, options = storage(model, True)
    result = datastore.identify(**options, identity_id=2, distance_metric="euclidean")
    reference = datastore.identify(
        **options, identity_id=2, distance_metric="euclidean_l2"
    )
    for key in [
        "verified",
        "distance",
        "threshold",
        "confidence",
        "similarity_metric",
        "id",
    ]:
        assert result[key] == reference[key]
    assert result["verified"] is False
    assert result["distance"] == pytest.approx(2.0)
    assert result["similarity_metric"] == "euclidean_l2"
    assert all(criteria["l2_normalized"] for criteria in db.criteria)


@pytest.mark.parametrize("delta", [-0.01, 0.01])
def test_decisions_on_both_sides_of_normalized_threshold(storage, delta):
    model = "Facenet512"
    threshold = verification.find_threshold(model, "euclidean_l2")
    distance = threshold + delta
    cosine = 1 - distance**2 / 2
    query = [cosine, np.sqrt(1 - cosine**2)]
    _, options = storage(model, True, queries=[query], vectors=[[1.0, 0.0]])
    frames = datastore.search(
        **options, search_method="exact", distance_metric="euclidean"
    )
    identified = datastore.identify(
        **options, identity_id=0, distance_metric="euclidean"
    )
    assert (len(frames[0]) > 0) == (delta < 0)
    assert identified["verified"] == (delta < 0)
    assert identified["threshold"] == threshold


@pytest.mark.parametrize("k", [None, 2])
def test_similarity_search_keeps_results_but_corrects_confidence(storage, k):
    _, options = storage("Facenet512", True)
    actual = datastore.search(
        **options,
        search_method="exact",
        distance_metric="euclidean",
        similarity_search=True,
        k=k,
    )[0]
    reference = datastore.search(
        **options,
        search_method="exact",
        distance_metric="euclidean_l2",
        similarity_search=True,
        k=k,
    )[0]
    pd.testing.assert_frame_equal(actual, reference)
    assert len(actual) == (3 if k is None else k)
    assert actual.loc[1, "confidence"] < 50


def test_multiple_query_faces_use_the_same_effective_metric(storage):
    _, options = storage("Facenet512", True, queries=[[1.0, 0.0], [-1.0, 0.0]])
    actual = datastore.search(
        **options, search_method="exact", distance_metric="euclidean"
    )
    assert [frame["id"].tolist() for frame in actual] == [[0], [2]]
    match = datastore.identify(**options, identity_id=2, distance_metric="euclidean")
    assert match["verified"] is True
    assert match["facial_areas"]["img1"]["x"] == 1
    assert match["similarity_metric"] == "euclidean_l2"


@pytest.mark.parametrize("metric", ["cosine", "angular", "euclidean_l2"])
def test_other_metrics_remain_unchanged(storage, metric):
    _, options = storage("Facenet512", True)
    frame = datastore.search(**options, search_method="exact", distance_metric=metric)[
        0
    ]
    result = datastore.identify(**options, identity_id=2, distance_metric=metric)
    assert set(frame["distance_metric"]) == {metric}
    assert result["similarity_metric"] == metric
    assert result["threshold"] == verification.find_threshold("Facenet512", metric)
    assert result["verified"] is False


def test_raw_euclidean_keeps_its_existing_threshold(storage):
    _, options = storage("Facenet512", False, vectors=[[1.0, 0.0], [100.0, 0.0]])
    frame = datastore.search(
        **options, search_method="exact", distance_metric="euclidean"
    )[0]
    result = datastore.identify(**options, identity_id=1, distance_metric="euclidean")
    assert frame["id"].tolist() == [0]
    assert set(frame["threshold"]) == {23.56}
    assert result["threshold"] == 23.56
    assert result["similarity_metric"] == "euclidean"
    assert result["verified"] is False


@pytest.mark.parametrize(
    "normalized,expected_metric", [(True, "cosine"), (False, "euclidean")]
)
def test_ann_metric_selection_is_preserved(storage, normalized, expected_metric):
    db, options = storage("Facenet512", normalized)
    db.neighbours = [dict(id=0, img_name="vector-0", distance=0.0)]
    frames = datastore.search(
        **options,
        database_type="pgvector",
        search_method="ann",
        distance_metric="angular",
    )
    assert frames[0]["distance_metric"].tolist() == [expected_metric]
    assert frames[0]["threshold"].tolist() == [
        verification.find_threshold("Facenet512", expected_metric)
    ]
    assert db.criteria[0]["l2_normalized"] == normalized


def test_identify_still_rejects_incompatible_stored_normalization(storage):
    db, options = storage("Facenet512", True)
    db.records[0]["l2_normalized"] = False
    with pytest.raises(ValueError, match="registered with l2_normalize"):
        datastore.identify(**options, identity_id=0, distance_metric="euclidean")


def test_identify_closes_owned_connection_and_preserves_payload_types(storage):
    db, options = storage("Facenet512", True)
    options["connection"] = None
    db.records[0]["id"] = np.int64(0)
    result = datastore.identify(**options, identity_id=0, distance_metric="euclidean")
    assert db.closed
    assert type(result["id"]) is int
    assert type(result["threshold"]) is float
    assert type(result["verified"]) is bool


def test_search_closes_owned_connection(storage):
    db, options = storage("Facenet512", True)
    options["connection"] = None
    datastore.search(**options, search_method="exact", distance_metric="euclidean")
    assert db.closed


def test_search_closes_owned_connection_on_error(storage):
    db, options = storage("Facenet512", True)
    options["connection"] = None
    with pytest.raises(ValueError, match="No embeddings found"):
        datastore.search(**options, database_type="pgvector", search_method="ann")
    assert db.closed


def test_search_keeps_caller_connection_open(storage):
    db, options = storage("Facenet512", True)
    datastore.search(**options, search_method="exact", distance_metric="euclidean")
    assert not db.closed
