# built-in dependencies
import os
import json

# 3rd party dependencies
import numpy as np
import pandas as pd
import pytest
from deepface import DeepFace

# project dependencies
from deepface.commons.logger import Logger
from deepface.modules.database.weaviate import WeaviateClient

weaviate = pytest.importorskip("weaviate")

logger = Logger()

# docker/docker-compose.yml exposes weaviate on 8080 (REST) and 50051 (gRPC)
connection_details = os.getenv("DEEPFACE_WEAVIATE_URI") or json.dumps(
    {"url": "http://localhost:8080", "grpc_port": 50051}
)

COLLECTION = "Embeddings_facenet_opencv_aligned_raw"
DATASET_IMAGES = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "couple.jpg"]


# pylint: disable=unused-argument, redefined-outer-name
@pytest.fixture
def flush_data():
    client = WeaviateClient(connection_details=connection_details)
    if client.client.collections.exists(COLLECTION):
        client.client.collections.delete(COLLECTION)
    client.close()
    logger.info("🗑️ Weaviate embeddings flushed.")


@pytest.fixture
def load_data(flush_data):
    inserted = 0
    for img_name in DATASET_IMAGES:
        result = DeepFace.register(
            img=f"../unit/dataset/{img_name}",
            model_name="Facenet",
            detector_backend="opencv",
            database_type="weaviate",
            connection_details=connection_details,
        )
        inserted += result["inserted"]
    return inserted


def test_weaviate_register_is_idempotent(load_data):
    assert load_data >= len(DATASET_IMAGES)

    result = DeepFace.register(
        img="../unit/dataset/img1.jpg",
        model_name="Facenet",
        detector_backend="opencv",
        database_type="weaviate",
        connection_details=connection_details,
    )
    assert result["inserted"] == 0
    logger.info("✅ Weaviate register idempotency test passed.")


@pytest.mark.parametrize("search_method", ["exact", "ann"])
def test_weaviate_search(load_data, search_method):
    dfs = DeepFace.search(
        img="../unit/dataset/img1.jpg",
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="euclidean",
        search_method=search_method,
        database_type="weaviate",
        connection_details=connection_details,
    )
    assert len(dfs) == 1
    df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["img_name"].endswith("img1.jpg")
    assert df.iloc[0]["distance"] == pytest.approx(0.0, abs=1e-4)
    logger.info(f"✅ Weaviate {search_method} search test passed.")


def test_weaviate_identify(load_data):
    dfs = DeepFace.search(
        img="../unit/dataset/img1.jpg",
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="euclidean",
        search_method="ann",
        database_type="weaviate",
        connection_details=connection_details,
    )
    result = DeepFace.identify(
        img="../unit/dataset/img2.jpg",
        identity_id=dfs[0].iloc[0]["id"],
        model_name="Facenet",
        detector_backend="opencv",
        distance_metric="euclidean",
        database_type="weaviate",
        connection_details=connection_details,
    )
    assert result["verified"] is True
    logger.info("✅ Weaviate identify test passed.")


def test_weaviate_exact_search_scans_whole_collection(flush_data):
    # graphql Get without a limit returned at most QUERY_DEFAULTS_LIMIT (100) objects
    rng = np.random.default_rng(0)
    records = [
        {
            "img_name": f"synthetic_{i}.jpg",
            "face": rng.random((4, 4, 3)),
            "model_name": "Facenet",
            "detector_backend": "opencv",
            "aligned": True,
            "l2_normalized": False,
            "embedding": rng.normal(size=128).tolist(),
        }
        for i in range(250)
    ]
    client = WeaviateClient(connection_details=connection_details)
    assert client.insert_embeddings(records) == 250
    embeddings = client.fetch_all_embeddings(
        model_name="Facenet", detector_backend="opencv", aligned=True, l2_normalized=False
    )
    client.close()
    assert len(embeddings) == 250
    logger.info("✅ Weaviate full scan test passed.")


def with_options(**options):
    details = json.loads(connection_details) if connection_details.startswith("{") else {
        "url": connection_details
    }
    return {**details, **options}


def test_weaviate_tenants_are_isolated(flush_data):
    for tenant, img_name in [("acme", "img1.jpg"), ("globex", "img3.jpg")]:
        result = DeepFace.register(
            img=f"../unit/dataset/{img_name}",
            model_name="Facenet",
            detector_backend="opencv",
            database_type="weaviate",
            connection_details=with_options(tenant=tenant),
        )
        assert result["inserted"] == 1

    for tenant, img_name in [("acme", "img1.jpg"), ("globex", "img3.jpg")]:
        dfs = DeepFace.search(
            img="../unit/dataset/img2.jpg",
            model_name="Facenet",
            detector_backend="opencv",
            distance_metric="euclidean",
            search_method="ann",
            similarity_search=True,
            database_type="weaviate",
            connection_details=with_options(tenant=tenant),
        )
        assert [name.split("/")[-1] for name in dfs[0]["img_name"]] == [img_name]

    with pytest.raises(ValueError, match="No embeddings found"):
        DeepFace.search(
            img="../unit/dataset/img2.jpg",
            model_name="Facenet",
            detector_backend="opencv",
            search_method="ann",
            database_type="weaviate",
            connection_details=with_options(tenant="initech"),
        )

    with pytest.raises(ValueError, match="a tenant is required"):
        DeepFace.search(
            img="../unit/dataset/img2.jpg",
            model_name="Facenet",
            detector_backend="opencv",
            search_method="ann",
            database_type="weaviate",
            connection_details=connection_details,
        )
    logger.info("✅ Weaviate tenant isolation test passed.")


def test_weaviate_quantized_search_matches_exact(flush_data):
    details = with_options(quantization="bq")
    for img_name in DATASET_IMAGES:
        DeepFace.register(
            img=f"../unit/dataset/{img_name}",
            model_name="Facenet",
            detector_backend="opencv",
            database_type="weaviate",
            connection_details=details,
        )
    kwargs = {
        "img": "../unit/dataset/img1.jpg",
        "model_name": "Facenet",
        "detector_backend": "opencv",
        "distance_metric": "euclidean",
        "database_type": "weaviate",
        "connection_details": details,
    }
    exact = DeepFace.search(search_method="exact", **kwargs)[0]
    ann = DeepFace.search(search_method="ann", **kwargs)[0]
    assert list(ann["img_name"]) == list(exact["img_name"])
    assert list(ann["distance"]) == pytest.approx(list(exact["distance"]), abs=1e-4)
    logger.info("✅ Weaviate quantized search test passed.")
