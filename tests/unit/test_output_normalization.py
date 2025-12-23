# 3rd-party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.modules.normalization import normalize_embedding_minmax
from deepface.config.minmax import minmax_values


def test_minmax_normalization():
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        minmax_normalize=False,
    )
    assert isinstance(results, list)
    assert len(results) == 1

    for result in results:
        embedding = result["embedding"]
        assert any(x < 0.0 or x > 1.0 for x in embedding)  # not normalized

    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        minmax_normalize=True,
    )
    for result in results:
        embedding = result["embedding"]
        assert all(x >= 0.0 and x <= 1.0 for x in embedding)  # normalized


def test_minmax_normalization_batch():
    results = DeepFace.represent(
        img_path=["dataset/img1.jpg", "dataset/couple.jpg"],
        model_name="Facenet",
        minmax_normalize=True,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    expected_embeddings = [1, 2]
    for idx, current_results in enumerate(results):
        assert isinstance(current_results, list)
        assert len(current_results) == expected_embeddings[idx]
        for result in current_results:
            embedding = result["embedding"]
            assert all(x >= 0.0 and x <= 1.0 for x in embedding)  # normalized


def test_minmax_normalization_edge_values():
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        minmax_normalize=False,
    )
    for result in results:
        embedding = result["embedding"]
        dim_min, dim_max = minmax_values["Facenet"]

        add_val = dim_max - max(embedding) + 0.2
        sub_val = abs(dim_min) - abs(min(embedding)) + 0.2

        shifted_embedding = [x + add_val if x > 0 else x - sub_val for x in embedding]
        normalized_embedding = normalize_embedding_minmax("Facenet", shifted_embedding)
        assert any(x < 0.0 or x > 1.0 for x in shifted_embedding)  # not normalized
        assert all(x >= 0.0 and x <= 1.0 for x in normalized_embedding)  # normalized


def test_l2_normalization():
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        l2_normalize=False,
    )
    raw_embedding = results[0]["embedding"]
    norm = np.linalg.norm(raw_embedding)
    assert not np.isclose(norm, 1.0), f"L2 norm of norm embedding shouldn't be 1, got {norm}"

    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        l2_normalize=True,
    )
    l2_embedding = results[0]["embedding"]

    # L2 embedding != raw embedding
    assert not np.allclose(
        l2_embedding, raw_embedding
    ), "L2 normalized embedding should differ from raw embedding"

    # Norm of L2 embedding should be 1
    norm = np.linalg.norm(l2_embedding)
    assert np.isclose(norm, 1.0), f"L2 norm of normalized embedding should be 1, got {norm}"
